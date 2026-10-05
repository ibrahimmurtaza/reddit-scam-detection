"""The Content Embeddings: what is embedded, what the model is, and what is published.

Ticket #21 is the substrate ticket. It builds no grouping, no corroboration and no
judgement: it computes one vector per Content Item, stores it in pgvector under an index,
and publishes the model and its dimensionality so a vector stored today can still be
compared with one stored tomorrow. Ticket #24 is the first thing to read them, and every
acceptance criterion here is about that reading being possible later rather than about it
being good now.

Seam under test: the pure half of the `content-embeddings` command â€” `embedded_text`,
`embed`, and the file the command publishes â€” observed through what it returns and what it
writes. The pgvector half needs a server and is in `test_embedding_store.py`; it is split
out so this file runs, and holds the committed file byte for byte, with no Postgres anywhere
near it.

Two claims are load-bearing and get tests of their own rather than a mention in a
docstring.

**No account metadata is embedded.** The Corpus boundary exists so the pipeline cannot see
Planted Campaign membership (ADR-0008), and a vector built out of who posted would put that
back through a side door. So the whole Corpus is re-read with every account renamed, every
subreddit swapped and every timestamp moved, and the two sets of records have to come back
identical â€” not merely close.

**The published file is recomputable.** Every generated file in this repository is committed
and held to its bytes by a test, so a reader can check the record rather than take the
code's word. That only means something if the file can be reproduced, which turns on the
model being stable across processes: Python's built-in `hash()` is salted per run, so a
model built on it would produce a different file every time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.embeddings import (
    DIMENSIONS,
    MODEL_NAME,
    ContentRecord,
    compute_embeddings,
    embed,
    digest_of,
    embedded_text,
    read_content_records,
    recipe_digest,
    write_content_records,
)

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_RECORDS = REPO_ROOT / "data" / "embeddings" / "content-embeddings.jsonl"

Row = Mapping[str, object]

# The vocabulary this project publishes on a row, restated rather than imported, so a change
# to the dataclass that forgot to change the file is caught here too.
PUBLISHED = {"post_id", "model", "dimensions", "recipe", "text_sha256"}


def post(**overrides: object) -> CorpusItem:
    """One Content Item. Every field is given so a test can change any one of them."""
    fields: dict[str, object] = {
        "post_id": "syn_p_0001",
        "account": "syn_somebody_0001",
        "subreddit": "CryptoCurrency",
        "title": "A paid desk that published its losing months",
        "body": "Three months on the desk and the drawdowns were mine.",
        "created_at": "2026-05-30T09:30:00Z",
        "links": ("https://vantage-ledger.example/entry",),
    }
    fields.update(overrides)
    return CorpusItem(**fields)  # type: ignore[arg-type]


def records_of(items: tuple[CorpusItem, ...]) -> tuple[ContentRecord, ...]:
    """The records this module publishes for these Content Items."""
    return tuple(
        ContentRecord(
            post_id=item.post_id,
            model=MODEL_NAME,
            dimensions=DIMENSIONS,
            recipe=recipe_digest(),
            text_sha256=digest_of(embedded_text(item)),
        )
        for item in items
    )


def rows(path: Path) -> list[Row]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def damaged(tmp_path: Path, *rows_written: Row) -> Path:
    """A file holding exactly the rows a test hands over, and nothing else."""
    path = tmp_path / "content-embeddings.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_written),
        encoding="utf-8",
    )
    return path


def as_row(record: ContentRecord) -> dict[str, object]:
    return {
        "post_id": record.post_id,
        "model": record.model,
        "dimensions": record.dimensions,
        "recipe": record.recipe,
        "text_sha256": record.text_sha256,
    }


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Cosine similarity, worked out here so a test is not trusting the model."""
    return sum(a * b for a, b in zip(left, right, strict=True))


# --- what is embedded ------------------------------------------------------------


def test_what_is_embedded_is_the_title_the_body_and_the_links() -> None:
    """The three fields, all of them, so the reader can see the whole of the input.

    The ticket says text and links and this holds both halves rather than one. A model fed
    the title alone would drop the body and a model fed the body alone would drop the links,
    and either would be a different model rather than a differently-weighted one.
    """
    text = embedded_text(post())

    assert "paid desk" in text
    assert "drawdowns were mine" in text
    assert "vantage-ledger" in text


def test_no_account_metadata_is_embedded() -> None:
    """Renaming every account, moving every post, and re-timing them changes nothing.

    This is the structural form of the acceptance criterion, and the whole Corpus is used
    rather than a hand-written item: the claim is about a field of the Corpus file, and a
    two-post example would not test it. The Corpus row carries an account, a subreddit and a
    timestamp, and all three are gone here â€” the records have to be identical, not close.
    """
    original = read_corpus(COMMITTED_CORPUS)
    renamed = tuple(
        replace(
            item,
            account=f"syn_renamed_{index:04d}",
            subreddit="somewhere-else",
            created_at="2026-06-01T00:00:00Z",
        )
        for index, item in enumerate(original)
    )

    assert records_of(original) == records_of(renamed)
    assert [embedded_text(item) for item in original] == [embedded_text(item) for item in renamed]


def test_the_embedded_text_carries_nothing_that_is_not_the_posts_own_words() -> None:
    """The text itself, read rather than inferred from two runs agreeing.

    The test above proves a rename moves nothing, and this proves why: an empty embedding
    would satisfy that one too.
    """
    item = post()
    text = embedded_text(item)

    assert item.account not in text
    assert item.subreddit not in text
    assert item.created_at not in text
    assert item.post_id not in text


def test_a_pair_of_words_does_not_span_two_fields() -> None:
    """A pair inside one field is evidence; one word from each of two fields is an artefact.

    Joining the title, the body and the links into one run of text would pair the last word
    of a title with the first word of its body, so two posts with nothing in common could
    share a feature neither of them contains. The three fields are therefore three separate
    sequences, and the embedded text keeps them on lines of their own so that is visible to a
    reader rather than a claim in a docstring.
    """
    text = embedded_text(post(title="alpha", body="beta", links=("gamma",)))

    assert text.split("\n") == ["alpha", "beta", "gamma"]


def test_a_content_item_with_nothing_to_read_is_refused_rather_than_stored_as_a_zero_vector() -> None:
    """A zero vector has no direction, and a cosine distance against one is not a number.

    It would store, it would come back, and every similarity query touching it would return
    NaN â€” so the refusal happens before a byte is written.
    """
    with pytest.raises(ValueError, match="no text to embed"):
        compute_embeddings((post(title="", body="", links=()),))


def test_a_content_item_whose_text_is_punctuation_is_refused_by_the_same_rule() -> None:
    """The floor of the case: it is the model that refuses, not a check bolted onto a row.

    Written so the refusal cannot be a special case for an empty string, which is the shape a
    hand-written example would have reached for.
    """
    with pytest.raises(ValueError, match="no text to embed"):
        compute_embeddings((post(title="!!! ???", body="---", links=()),))


def test_the_model_publishes_a_name_a_width_and_a_recipe_digest() -> None:
    """Three facts, because any one of them alone will not tell a reader whether two stores
    hold vectors that can be compared.

    Not a test of arithmetic: a test that the three facts this project publishes about its
    model are three facts, that they are the ones the code uses, and that the digest is a
    digest of something stated rather than of the file that states it. A width changed without
    the name changing would be the worst version of this bug, because every stored vector
    would still parse.
    """
    assert MODEL_NAME == "hashed-word-ngrams-v1"
    assert DIMENSIONS == 256
    assert len(recipe_digest()) == 64


def test_the_vector_has_the_published_width() -> None:
    computed = compute_embeddings((post(),))

    assert len(computed) == 1
    assert computed[0].record.dimensions == DIMENSIONS
    assert len(computed[0].embedding) == DIMENSIONS


def test_the_vector_is_of_unit_length_because_cosine_distance_is_what_the_column_computes() -> None:
    """Length 1, to within the width the column stores.

    `<=>` is cosine distance, so the length of a stored vector is not information: it would
    change every distance the vector appears in. A model returning raw term weights would
    make the distance between two posts depend on how long they are.
    """
    computed = compute_embeddings((post(body="desk"),))

    length = math.sqrt(sum(value * value for value in computed[0].embedding))
    assert math.isclose(length, 1.0, abs_tol=1e-9)


def test_computing_the_same_corpus_twice_gives_the_same_numbers() -> None:
    """Byte-stability, which is what makes the published file holdable at all.

    A model built on Python's built-in `hash()` would pass every other test here and produce
    a different file on every run, because that hash is salted per process.
    """
    items = read_corpus(COMMITTED_CORPUS)

    assert compute_embeddings(items) == compute_embeddings(items)


def test_two_posts_using_the_same_words_come_out_in_the_same_direction() -> None:
    """The one property the similarity query exists to exploit, on the model's own terms.

    Not a claim about meaning: it is the claim that shared words move two vectors towards
    each other, which is the whole of what a lexical model does and the whole of what ticket
    #24 may rely on.
    """
    words = "remote annotation role equipment provided no experience needed"
    same = compute_embeddings(
        (post(title=words, body=""), post(post_id="p2", title=words, body=""))
    )

    assert same[0].embedding == same[1].embedding


def test_two_posts_sharing_no_topic_come_out_near_orthogonal_rather_than_near() -> None:
    """The other end of the same claim, and the reason a threshold can exist at all.

    Neither post links anything, which matters: the next test is about what a shared link does
    on its own, and this one would be measuring that rather than the model if the two shared a
    host.

    The bound is 0.25 and not 0.1 because two paragraphs of ordinary English with nothing in
    common share their function words, and the model reads those as readily as any other. That
    is the baseline a threshold has to clear rather than zero, and it is the figure ticket #24
    measures when it states one.
    """
    left = compute_embeddings(
        (
            post(
                title="Eight months of DCA and I was wrong about the drawdowns",
                body="Ran a numbers-only model against BTC and ETH for most of last year and "
                "every recovery I waited for turned into a deeper drawdown.",
                links=(),
            ),
        )
    )[0]
    right = compute_embeddings(
        (
            post(
                post_id="p2",
                title="Complaint about a warehouse that never paid anybody",
                body="Three shifts a week for six weeks and the payout is still pending, and "
                "nobody at the desk will say when it lands.",
                links=(),
            ),
        )
    )[0]

    assert cosine(left.embedding, right.embedding) < 0.25


def test_two_posts_sharing_nothing_but_one_link_come_out_close() -> None:
    """The sharpest available demonstration of the limit the output states.

    Two posts about different things, sharing no sentence and no word, and one host between
    them. The words of that host are part of what is embedded, so the two come out close on
    the strength of an edge ADR-0005 already has â€” which is exactly why a similarity read
    this way is corroboration and never evidence, and why the output says so where a reader
    will meet a distance rather than in a docstring.
    """
    apart = compute_embeddings(
        (
            post(
                title="Eight months of DCA and I was wrong about the drawdowns",
                body="Ran a numbers-only model against BTC and ETH for most of last year.",
                links=(),
            ),
            post(
                post_id="p2",
                title="Complaint about a warehouse that never paid anybody",
                body="Three shifts a week for six weeks and the payout is still pending.",
                links=(),
            ),
        )
    )
    linked = compute_embeddings(
        (
            post(
                post_id="p3",
                title="Eight months of DCA and I was wrong about the drawdowns",
                body="Ran a numbers-only model against BTC and ETH for most of last year.",
                links=("https://hopcut.example/abc",),
            ),
            post(
                post_id="p4",
                title="Complaint about a warehouse that never paid anybody",
                body="Three shifts a week for six weeks and the payout is still pending.",
                links=("https://hopcut.example/def",),
            ),
        )
    )

    assert cosine(linked[0].embedding, linked[1].embedding) > cosine(
        apart[0].embedding, apart[1].embedding
    )


def test_a_post_and_its_own_paraphrase_are_closer_than_two_unrelated_posts() -> None:
    """What the model is for, and the claim ticket #24 would be making if it used this.

    The Corpus plants staggered paraphrases of one pitch across a Planted Campaign's accounts,
    and this is the property that makes them findable: the same offer reworded shares
    vocabulary with itself and much less with anything else.
    """
    original = compute_embeddings(
        (
            post(
                title="Remote annotation, 420 a day, kit provided",
                body="No experience needed, they train you, six hours five days a week.",
                links=(),
            ),
        )
    )[0]
    reworded = compute_embeddings(
        (
            post(
                post_id="p2",
                title="data annotation, 420 a day, equipment provided",
                body="no experience required, they train you from scratch, six hours a week "
                "five days.",
                links=(),
            ),
        )
    )[0]
    unrelated = compute_embeddings(
        (
            post(
                post_id="p3",
                title="Complaint about a warehouse that never paid anybody",
                body="Three shifts a week for six weeks and the payout is still pending.",
                links=(),
            ),
        )
    )[0]

    assert cosine(original.embedding, reworded.embedding) > cosine(
        original.embedding, unrelated.embedding
    )


def test_one_shared_word_moves_two_posts_towards_each_other() -> None:
    """Similarity is graded rather than a match or a miss, which is what pgvector needs.

    A model that put a post either in or out of a neighbourhood would make every threshold in
    ticket #24 arbitrary; this holds that adding shared vocabulary increases the cosine
    between two vectors.
    """
    bare = compute_embeddings((post(title="alpha beta gamma", body=""),))[0]
    shared = compute_embeddings((post(title="alpha beta gamma", body="delta"),))[0]
    apart = compute_embeddings((post(post_id="p2", title="epsilon zeta eta", body=""),))[0]

    assert cosine(bare.embedding, shared.embedding) > cosine(bare.embedding, apart.embedding)


def test_the_published_record_is_five_fields_and_no_vector(tmp_path: Path) -> None:
    """A vector stored beside the digest is not published, and the file says why it can be
    recomputed instead.

    Five fields, every one of which a later run needs before it can decide not to
    recompute. A sixth â€” the vector â€” would be 34 posts of 256 numbers a reader would have to
    take on trust rather than check, and the values are in the database.
    """
    path = tmp_path / "content-embeddings.jsonl"
    write_content_records(path, records_of((post(),)))
    written = rows(path)[0]

    assert set(written) == PUBLISHED
    assert written["model"] == MODEL_NAME
    assert written["dimensions"] == DIMENSIONS
    assert written["recipe"] == recipe_digest()


def test_a_row_read_back_is_the_row_written(tmp_path: Path) -> None:
    """The round trip, so the committed file can be compared rather than recomputed."""
    path = tmp_path / "content-embeddings.jsonl"
    items = (post(), post(post_id="syn_p_0002"))
    write_content_records(path, records_of(items))

    assert read_content_records(path) == records_of(items)


def test_a_row_carrying_a_field_this_project_does_not_publish_is_refused(
    tmp_path: Path,
) -> None:
    """A membership field here would be the one leak ADR-0008 rules out.

    The same reason `read_corpus` and `read_policy_scores` refuse a field they do not know: a
    reader that skipped what it did not recognise would skip that without saying so, and this
    is a file a later ticket reads records out of.
    """
    path = damaged(
        tmp_path, as_row(records_of((post(),))[0]) | {"campaign_id": "pc-alpha"}
    )

    with pytest.raises(ValueError, match="which is not the Content Embedding vocabulary"):
        read_content_records(path)


def test_a_file_holding_one_post_twice_is_refused(tmp_path: Path) -> None:
    """Two rows for one post would let a later run reuse whichever it read last."""
    row = as_row(records_of((post(),))[0])
    path = damaged(tmp_path, row, row)

    with pytest.raises(ValueError, match="on two rows each"):
        read_content_records(path)


def test_rows_disagreeing_about_the_model_are_refused(tmp_path: Path) -> None:
    """Distances are only comparable within one model, so a mixed file holds nothing.

    Refused as a whole rather than row by row, because the disagreement is between rows and
    neither row is wrong on its own.
    """
    path = damaged(
        tmp_path,
        as_row(records_of((post(),))[0]) | {"model": "one"},
        as_row(records_of((post(post_id="p2"),))[0]) | {"model": "two"},
    )

    with pytest.raises(ValueError, match="holds 2 models"):
        read_content_records(path)


def test_rows_disagreeing_about_the_width_are_refused(tmp_path: Path) -> None:
    """A row whose width differs from its neighbours' would describe a different vector."""
    path = damaged(
        tmp_path,
        as_row(records_of((post(),))[0]),
        as_row(records_of((post(post_id="p2"),))[0]) | {"dimensions": 3},
    )

    with pytest.raises(ValueError, match="holds 2 widths"):
        read_content_records(path)


def test_a_row_whose_digest_is_not_a_digest_is_refused(tmp_path: Path) -> None:
    """The digest is what a rerun compares against to decide not to recompute, so a row
    holding something that is not one would make every rerun either recompute everything or
    reuse a vector belonging to text that has since changed."""
    path = damaged(
        tmp_path, as_row(records_of((post(),))[0]) | {"text_sha256": "abc"}
    )

    with pytest.raises(ValueError, match="SHA-256"):
        read_content_records(path)


def test_rows_disagreeing_about_the_recipe_are_refused(tmp_path: Path) -> None:
    """Two rules cannot both have produced the rows, and the name cannot tell them apart.

    The name carries a version, and a version is a promise somebody has to keep. The recipe
    digest is the promise checked, which is why this refusal exists and why the recipe is a
    published field rather than only something the console prints.
    """
    path = damaged(
        tmp_path,
        as_row(records_of((post(),))[0]),
        as_row(records_of((post(post_id="p2"),))[0]) | {"recipe": "0" * 64},
    )

    with pytest.raises(ValueError, match="holds 2 recipes"):
        read_content_records(path)


def test_a_row_whose_recipe_is_not_a_digest_is_refused(tmp_path: Path) -> None:
    """The same check as for the text's digest, and for the same reason: a value here that is
    not one would make the reuse rule compare against nothing."""
    path = damaged(
        tmp_path, as_row(records_of((post(),))[0]) | {"recipe": "not-a-digest"}
    )

    with pytest.raises(ValueError, match="recipe="):
        read_content_records(path)


def test_a_row_with_no_width_is_refused(tmp_path: Path) -> None:
    """A width of zero or a width that is text would describe no vector at all."""
    for width in (0, "256"):
        path = damaged(
            tmp_path, as_row(records_of((post(),))[0]) | {"dimensions": width}
        )
        with pytest.raises(ValueError, match="dimensions="):
            read_content_records(path)


# --- the Corpus the project actually ships ---------------------------------------


def test_the_committed_file_is_the_records_this_corpus_computes(tmp_path: Path) -> None:
    """Held to the model, with no database in the way.

    The file is committed so a reader can check the record without running anything, and that
    only means something if a rerun reproduces it byte for byte. This asserts the computed
    half, so it holds on a machine with no Postgres at all; the other half â€” that the command
    writes these bytes to disk â€” is the store's test.
    """
    path = tmp_path / "content-embeddings.jsonl"
    write_content_records(path, records_of(read_corpus(COMMITTED_CORPUS)))

    assert path.read_bytes() == COMMITTED_RECORDS.read_bytes()


def test_the_closest_pair_the_readme_quotes_is_the_closest_pair_this_corpus_has() -> None:
    """The two distances the README and ADR-0024 quote, computed with no database at all.

    Those figures are printed by `rfi content-embeddings`, which needs pgvector, so the only
    test that could hold the README to them would be one that skips without a server â€” and
    `tests/test_readme_figures.py` exists because figures copied out of a generated file went
    stale with nothing in a position to see it. The vector column stores 32-bit floats and the
    run prints four decimal places, which the 64-bit cosine below reproduces, so the figure is
    checkable offline and this is where that is done.
    """
    vectors = {
        item.post_id: embed(embedded_text(item)) for item in read_corpus(COMMITTED_CORPUS)
    }
    pairs = sorted(
        (1.0 - cosine(vectors[left], vectors[right]), left, right)
        for left in vectors
        for right in vectors
        if left < right
    )
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    closest, left, right = pairs[0]
    assert {left, right} == {"syn_p_0012", "syn_p_0013"}, pairs[0]
    for figure in (
        f"{closest:.4f}",
        f"{1.0 - cosine(vectors['syn_p_0012'], vectors['syn_p_0014']):.4f}",
    ):
        assert figure in readme, f"README.md does not quote the distance {figure}"


def test_every_content_item_in_the_corpus_has_a_record() -> None:
    """One row per Content Item, in the Corpus's own order.

    A post whose text is short and whose links name one host is the item an implementation
    filters out for being thin, and this Corpus holds several. The floor of the ticket is
    that a Content Item is embedded or it is not in the Corpus.
    """
    items = read_corpus(COMMITTED_CORPUS)
    written = read_content_records(COMMITTED_RECORDS)

    assert [row.post_id for row in written] == [item.post_id for item in items]
    assert len({row.post_id for row in written}) == len(written)


def test_the_digest_beside_every_committed_record_is_the_digest_of_its_posts_own_text() -> None:
    """Every row, recomputed from the Corpus, so a stale vector cannot hide here.

    This is the reuse rule's audit. The digest is what a later run compares before deciding
    not to recompute, so a digest that did not match its post would make every rerun either
    recompute everything or reuse a vector belonging to text that has since changed â€” and
    both of those are invisible in the output.
    """
    embedded = {item.post_id: embedded_text(item) for item in read_corpus(COMMITTED_CORPUS)}

    for row in read_content_records(COMMITTED_RECORDS):
        assert row.text_sha256 == hashlib.sha256(
            embedded[row.post_id].encode("utf-8")
        ).hexdigest(), f"{row.post_id}'s digest is not the digest of its own text"


def test_the_committed_file_holds_no_account_and_no_membership_field() -> None:
    """The leak boundary, checked on the bytes a reader would actually open.

    The rename test proves the vector cannot carry an account. This proves the published row
    has nowhere to put one even if somebody wanted it to, which is what makes the first of
    the two a property of the file rather than of a habit.
    """
    for row in rows(COMMITTED_RECORDS):
        assert set(row) == PUBLISHED
        assert not [field for field in row if field in {"account", "campaign_id", "subreddit"}]


def test_every_content_item_in_the_corpus_produces_a_vector() -> None:
    """The other half of the same floor: a record is not an embedding.

    A pipeline that published a digest for a post it could not embed would look identical in
    the file and be useless to ticket #24, so every record is checked to have a vector behind
    it.
    """
    written = read_content_records(COMMITTED_RECORDS)
    embedded = compute_embeddings(read_corpus(COMMITTED_CORPUS))

    assert [found.record for found in embedded] == list(written)
    assert all(len(found.embedding) == DIMENSIONS for found in embedded)


def test_the_committed_records_describe_the_model_this_repository_runs() -> None:
    """Every row says the current model, its width and its recipe, so a file left behind by
    another model is visible before somebody regenerates it.

    The reader refuses a mixed file, so a file holding an old model's rows beside new ones
    would stop `rfi content-embeddings` rather than publish it.
    """
    written = read_content_records(COMMITTED_RECORDS)

    assert {row.model for row in written} == {MODEL_NAME}
    assert {row.dimensions for row in written} == {DIMENSIONS}
    assert {row.recipe for row in written} == {recipe_digest()}