"""The Corpus file is the credibility boundary, so these tests read the file.

Seam under test: the `generate-corpus` command, observed through the five files it
writes. Nothing here inspects the code that wrote them.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_SEED, main

COMMITTED_CORPUS = Path(__file__).parent.parent / "data" / "corpus" / "corpus.jsonl"
COMMITTED_LABELLED = (
    Path(__file__).parent.parent / "data" / "corpus" / "labelled-contacts.jsonl"
)

CORPUS_FIELDS = frozenset(
    {"post_id", "account", "subreddit", "title", "body", "created_at", "links"}
)

LABELLED_FIELDS = frozenset({"post_id", "published"})
PUBLISHED_FIELDS = frozenset({"kind", "value", "written_as", "written"})

# A membership field, by any name a generator might reach for. Checked against the
# field names on the file rather than its text, so that legitimate post copy can
# never trip it.
MEMBERSHIP_FIELDS = frozenset(
    {
        "campaign",
        "campaign_id",
        "campaigns",
        "ground_truth",
        "is_fraud",
        "label",
        "labels",
        "membership",
        "planted",
        "truth",
    }
)

# A quotation of the planted pitch, so a reviewer can tell from a failure message
# that the content is still readable rather than truncated. Update it when the
# planted copy is rewritten.
PLANTED_PITCH = (
    "What sold me is that they publish the month's result whether it was green or not, "
    "including the two months it lost."
)


def run_generator(
    directory: Path, seed: int = DEFAULT_SEED, background: int = 0
) -> tuple[Path, Path, Path]:
    corpus_path = directory / "corpus.jsonl"
    truth_path = directory / "truth.jsonl"
    labelled_path = directory / "labelled-contacts.jsonl"
    exit_code = main(
        [
            "generate-corpus",
            "--seed",
            str(seed),
            "--background",
            str(background),
            "--corpus",
            str(corpus_path),
            "--truth",
            str(truth_path),
            "--labelled",
            str(labelled_path),
            "--nuisance",
            str(directory / "nuisance.jsonl"),
            "--shared-infrastructure",
            str(directory / "shared-hosts.jsonl"),
        ]
    )
    assert exit_code == 0
    return corpus_path, truth_path, labelled_path


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def text(row: dict[str, object], field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} is not text: {value!r}"
    return value


def texts(row: dict[str, object], field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [str(item) for item in value]


def records(row: dict[str, object], field: str) -> list[dict[str, object]]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [item for item in value if isinstance(item, dict)]


def test_corpus_file_carries_no_membership_fields(tmp_path: Path) -> None:
    corpus_path, _, _ = run_generator(tmp_path)

    corpus_rows = rows(corpus_path)
    assert corpus_rows, "the generator wrote no Corpus items"

    for row in corpus_rows:
        assert set(row) == CORPUS_FIELDS
        assert not set(row) & MEMBERSHIP_FIELDS


def test_the_corpus_holds_a_few_accounts_and_shared_domains_to_look_at(
    tmp_path: Path,
) -> None:
    """The Corpus is deliberately small enough to read in full, and still shaped like
    something worth grouping: there are fewer accounts than posts, so some accounts
    post more than once, and there are far more hosts than Planted Campaigns. Widen
    these bounds deliberately when the Corpus changes shape, not by accident.
    """
    corpus_path, _, _ = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)

    hosts = {
        urlparse(link).hostname
        for row in corpus_rows
        for link in texts(row, "links")
    }
    accounts = {text(row, "account") for row in corpus_rows}

    assert 10 < len(accounts) < len(corpus_rows)
    assert 8 < len(hosts) < len(corpus_rows)


def test_same_seed_writes_byte_identical_files(tmp_path: Path) -> None:
    first_corpus, first_truth, _ = run_generator(tmp_path / "first")
    second_corpus, second_truth, _ = run_generator(tmp_path / "second")

    assert first_corpus.read_bytes() == second_corpus.read_bytes()
    assert first_truth.read_bytes() == second_truth.read_bytes()


def test_a_different_seed_writes_a_different_corpus(tmp_path: Path) -> None:
    first_corpus, _, _ = run_generator(tmp_path / "first")
    second_corpus, _, _ = run_generator(tmp_path / "second", seed=DEFAULT_SEED + 1)

    assert first_corpus.read_bytes() != second_corpus.read_bytes()


def test_the_committed_corpus_is_what_the_default_seed_produces(tmp_path: Path) -> None:
    corpus_path, _, labelled_path = run_generator(tmp_path)

    assert corpus_path.read_bytes() == COMMITTED_CORPUS.read_bytes()
    assert labelled_path.read_bytes() == COMMITTED_LABELLED.read_bytes()


def test_a_background_population_adds_posts_without_moving_the_ones_already_there(
    tmp_path: Path,
) -> None:
    """Story #7: scaling is a run, and a run at another size is the same Corpus plus more.

    The property this rests on is the one `_key` already provides for timestamps. If
    growing the background renumbered a post, every timestamp keyed on it would move, and
    the 34-post Corpus a reviewer checks against would no longer be a prefix of the 236-post
    one — which would make the two incomparable and the scaling meaningless.
    """
    plain_path, _, _ = run_generator(tmp_path / "plain")
    scaled_path, _, _ = run_generator(tmp_path / "scaled", background=200)

    plain = {text(row, "post_id"): row for row in rows(plain_path)}
    scaled = {text(row, "post_id"): row for row in rows(scaled_path)}

    assert len(scaled) == len(plain) + 200
    # Matched on the identifier rather than on position, because the Corpus is written in
    # chronological order and a background post lands wherever its timestamp falls.
    assert {key: scaled[key] for key in plain} == plain, (
        "the posts already in the Corpus have to come through a larger run unchanged, "
        "identifiers and timestamps alike, or the two runs are not the same Corpus"
    )


def test_a_background_population_is_same_seed_same_bytes(tmp_path: Path) -> None:
    """The determinism the committed Corpus is held to extends to the parameter."""
    first, _, first_labelled = run_generator(tmp_path / "first", background=200)
    second, _, second_labelled = run_generator(tmp_path / "second", background=200)

    assert first.read_bytes() == second.read_bytes()
    assert first_labelled.read_bytes() == second_labelled.read_bytes()


def test_a_background_population_carries_nothing_two_posts_could_share(tmp_path: Path) -> None:
    """Safe by construction, not by inspection — and this is what makes it safe.

    A background post can reach another account only through a shared registrable domain
    or a shared Contact Point (ADR-0005). So the parameter that makes a Corpus larger has
    to add posts that do neither, and the text has to hold no `@` either, or a scaled run
    would plant false positives of its own and move the invented figure `rfi
    contact-points` measures — a defect introduced by asking for a bigger Corpus rather
    than by anything the build did wrong.
    """
    plain_path, _, _ = run_generator(tmp_path / "plain")
    scaled_path, _, scaled_labelled = run_generator(tmp_path / "scaled", background=200)
    added = [
        row
        for row in rows(scaled_path)
        if text(row, "account").startswith("syn_background_")
    ]

    assert len(added) == 200
    assert all(not texts(row, "links") for row in added), (
        "a background post reaching a domain could share it with another account and form a "
        "Campaign Candidate the scale parameter planted"
    )
    assert all("@" not in text(row, "title") + text(row, "body") for row in added), (
        "a background post whose text carries an @ is a Contact Point the generator did not "
        "declare, and the Labelled Set would be wrong about it"
    )
    accounts = [text(row, "account") for row in added]
    assert len(set(accounts)) == len(accounts), (
        "an account appearing in two background posts would be a shared identifier, which is "
        "the other edge a post could be joined on"
    )
    background_ids = {text(row, "post_id") for row in added}
    assert all(
        not entry["published"]
        for entry in rows(scaled_labelled)
        if text(entry, "post_id") in background_ids
    ), "every background post has to be labelled as publishing nothing"


def test_a_background_population_does_not_scale_the_planted_evidence(tmp_path: Path) -> None:
    """What does not scale is stated here because it is the honest limit of story #7.

    The Planted Campaigns and the Nuisance Structure are planted individually, each to make
    a particular failure measurable, so they are a fixed set. A larger Corpus therefore
    means more population to recover the same campaigns from, not more campaigns — and a
    reader who took `--background 5000` as five thousand posts of evidence would be reading
    a claim the parameter does not make.
    """
    _, plain_truth, _ = run_generator(tmp_path / "plain")
    _, scaled_truth, _ = run_generator(tmp_path / "scaled", background=500)

    assert rows(scaled_truth) == rows(plain_truth), (
        "the Planted Campaigns are the same two at every scale: growing the background grows "
        "the population they have to be recovered from, not the number of them"
    )


def test_a_negative_background_population_is_refused(tmp_path: Path) -> None:
    """Refused by name rather than producing an empty Corpus that looks like a clean sweep."""
    with pytest.raises(SystemExit):
        main(
            [
                "generate-corpus",
                "--seed",
                str(DEFAULT_SEED),
                "--background",
                "-1",
                "--corpus",
                str(tmp_path / "corpus.jsonl"),
                "--truth",
                str(tmp_path / "truth.jsonl"),
                "--labelled",
                str(tmp_path / "labelled-contacts.jsonl"),
                "--nuisance",
                str(tmp_path / "nuisance.jsonl"),
                "--shared-infrastructure",
                str(tmp_path / "shared-hosts.jsonl"),
            ]
        )


def test_generation_refuses_to_write_both_files_to_one_path(tmp_path: Path) -> None:
    shared = tmp_path / "one.jsonl"

    with pytest.raises(SystemExit):
        main(
            [
                "generate-corpus",
                "--seed",
                str(DEFAULT_SEED),
                "--corpus",
                str(shared),
                "--truth",
                str(shared),
                "--labelled",
                str(tmp_path / "labelled-contacts.jsonl"),
                "--nuisance",
                str(tmp_path / "nuisance.jsonl"),
                "--shared-infrastructure",
                str(tmp_path / "shared-hosts.jsonl"),
            ]
        )

    assert not shared.exists()


def test_the_labelled_set_names_a_contact_point_only_where_a_post_publishes_one(
    tmp_path: Path,
) -> None:
    """The measurement needs a Labelled Set, and it has to be the Corpus's own.

    Recalling an extraction is only meaningful against content somebody has labelled,
    and this project generates its content, so the labels come from the same plan the
    posts do: every post the Corpus holds has a row, and every Contact Point the
    generator planted in it is in that row with the spelling it was planted in. A row
    naming a post the Corpus does not hold, or a Contact Point that is not written in
    the post, is a measurement of nothing.

    Posts publishing nothing are labelled as publishing nothing, which is the half that
    makes the false-positive rate a rate: without them the denominator of "how often did
    this invent a Contact Point" would be zero and the answer would be free.
    """
    corpus_path, _, labelled_path = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    labelled_rows = rows(labelled_path)

    assert [text(row, "post_id") for row in labelled_rows] == sorted(
        text(row, "post_id") for row in corpus_rows
    )
    for row in labelled_rows:
        assert set(row) == LABELLED_FIELDS

    posts = {text(row, "post_id"): row for row in corpus_rows}
    published = 0
    for row in labelled_rows:
        post = posts[text(row, "post_id")]
        # The spelling is in the post, in its text or in its links: a Contact Point
        # published only as a picture of one is written as the link carrying it, since
        # that is the only place the post holds it.
        somewhere = (
            text(post, "title"),
            text(post, "body"),
            *texts(post, "links"),
        )
        for entry in records(row, "published"):
            published += 1
            assert set(entry) == PUBLISHED_FIELDS
            spelling = text(entry, "written_as")
            assert any(spelling in field for field in somewhere), (
                f"{text(row, 'post_id')} is labelled with {spelling!r}, which it does "
                "not write"
            )

    assert published >= 6, "the Corpus publishes too few Contact Points to measure"
    assert any(not records(row, "published") for row in labelled_rows), (
        "every post publishes one, so there is nothing to measure a false positive on"
    )


def test_the_corpus_file_carries_no_contact_point_labels(tmp_path: Path) -> None:
    """The Labelled Set is a separate file for the reason ADR-0008 is about.

    The Corpus is what the pipeline sees, and a post that carried the Contact Points it
    publishes would be carrying its own answer: a reader could grep for the labels, and
    a later step could read them out of the Corpus rather than out of the text. So the
    labels live at a path of their own and the Corpus file has no field for them.
    """
    corpus_path, _, _ = run_generator(tmp_path)

    for row in rows(corpus_path):
        assert not {"published", "contact_points", "written_as"} & set(row)


def test_truth_file_holds_membership_that_matches_the_corpus(tmp_path: Path) -> None:
    corpus_path, truth_path, _ = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    memberships = rows(truth_path)

    accounts = {text(row, "account") for row in corpus_rows}
    post_ids = {text(row, "post_id") for row in corpus_rows}

    assert memberships
    for membership in memberships:
        assert set(texts(membership, "accounts")) <= accounts
        assert set(texts(membership, "posts")) <= post_ids
        # A grouping of one is not a grouping: ADR-0005 requires two or more.
        assert len(texts(membership, "accounts")) >= 2
        assert len(texts(membership, "posts")) >= 2


def test_no_planted_campaign_is_named_in_the_corpus_file(tmp_path: Path) -> None:
    corpus_path, truth_path, _ = run_generator(tmp_path)
    corpus_text = corpus_path.read_text(encoding="utf-8")

    for membership in rows(truth_path):
        assert text(membership, "campaign_id") not in corpus_text


def test_the_planted_post_can_be_read_to_judge_its_realism(tmp_path: Path) -> None:
    corpus_path, _, _ = run_generator(tmp_path)

    corpus_text = corpus_path.read_text(encoding="utf-8")

    assert PLANTED_PITCH in corpus_text


def test_every_synthetic_entity_is_marked_as_synthetic(tmp_path: Path) -> None:
    corpus_path, _, _ = run_generator(tmp_path)

    for row in rows(corpus_path):
        assert text(row, "post_id").startswith("syn_")
        assert text(row, "account").startswith("syn_")
        for link in texts(row, "links"):
            hostname = urlparse(link).hostname
            assert hostname is not None
            assert hostname.endswith(".example")
