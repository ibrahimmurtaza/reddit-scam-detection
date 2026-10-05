"""The pgvector half: the table, the index, the reuse rule, and the similarity query.

Everything here needs a PostgreSQL server with `pgvector` activated, which is the
environment ticket #4 built and `docs/pgvector.md` documents. The tests are marked and
skipped with the reason when no `PG*` variable is set, rather than passing quietly: the
offline suite in `test_content_embeddings.py` still holds the model and the published file
to their bytes, but a round trip through a column is not something a machine without a
server can check, and a test that skips without saying why is a test that can disappear.

Each test names its own table and drops it afterwards, so a run leaves the database as it
found it and two tests cannot see each other's rows. The table name is the argument, which is
also how a reader would keep two models' vectors apart (ADR-0024).

The claims worth stating plainly, because they are the ticket's:

* Every Content Item in the Corpus gets a vector, and the run says so.
* A similarity query returns the nearest items, checked against arithmetic performed here
  from the vectors the table holds â€” not against whatever order the rows arrive in.
* A rerun over an unchanged Corpus embeds nothing, and the run says how much it computed.
* The index is HNSW, and the run reports the plan the planner actually chose.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

import psycopg

from reddit_fraud_intelligence.cli import DEFAULT_CORPUS_PATH, DEFAULT_EMBEDDING_TABLE, main
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.embeddings import (
    DIMENSIONS,
    MODEL_NAME,
    Database,
    compute_embeddings,
    connection_string,
    digest_of,
    embed_corpus,
    embedded_text,
    ensure_index,
    ensure_table,
    index_name,
    nearest,
    open_store,
    read_content_records,
    recipe_digest,
    render_table,
)

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_RECORDS = REPO_ROOT / "data" / "embeddings" / "content-embeddings.jsonl"

Row = Mapping[str, object]

# The environment a run reads its connection from. Four names rather than one, because
# libpq reads all four and the project keeps credentials in the environment rather than in a
# command line where they would end up in a shell history.
CONFIGURED = ("PGHOST", "PGPORT", "PGUSER", "PGDATABASE", "RFI_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not any(name in os.environ for name in CONFIGURED),
    reason=(
        "no database is configured: set PGHOST, PGPORT, PGUSER and PGDATABASE, or "
        "RFI_DATABASE_URL, to run the pgvector half. docs/pgvector.md has the activation "
        "step this needs"
    ),
)


class _Opened:
    """Records every path a run opens for reading, so a claim about what it read can be checked.

    An audit hook rather than a monkeypatch, because the point is to catch a read from
    anywhere at all â€” including from inside the standard library, which a patched `open`
    would miss. Writes are left out because this command writes one file and that is not what
    the claim is about. It cannot be uninstalled, so it records only while `recording` is set
    and does nothing for the rest of the session.
    """

    def __init__(self) -> None:
        self.recording = False
        self.paths: list[str] = []

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        if not self.recording or event != "open":
            return
        mode = arguments[1] if len(arguments) > 1 else "r"
        if isinstance(mode, str) and set(mode) <= {"r", "b", "U"} and "w" not in mode:
            self.paths.append(str(arguments[0]))

    def record(self) -> list[str]:
        return list(self.paths)


@pytest.fixture
def table(tmp_path: Path) -> Iterator[str]:
    """A table of this test's own, dropped afterwards whatever the test did to it."""
    name = f"embedding_{tmp_path.name.replace('-', '_')}"[:60]
    connection = open_store(connection_string())
    try:
        yield name
    finally:
        connection.execute(f'DROP TABLE IF EXISTS "{name}"')
        connection.close()


def run(
    directory: Path,
    corpus: Path = DEFAULT_CORPUS_PATH,
    capsys: pytest.CaptureFixture[str] | None = None,
    table: str = "content_embedding",
) -> tuple[Path, str]:
    """Run the command over a table of the caller's choosing and report what it printed.

    Returned rather than captured so a test can read the printed text: the figures this
    command publishes â€” the model, the index, the plan the planner chose, how much it
    computed â€” are claims about the run, and a claim printed nowhere is a claim a reader
    cannot check.
    """
    records_path = directory / "content-embeddings.jsonl"
    exit_code = main(
        [
            "content-embeddings",
            "--corpus",
            str(corpus),
            "--embeddings",
            str(records_path),
            "--table",
            table,
        ]
    )
    assert exit_code == 0
    return records_path, capsys.readouterr().out if capsys is not None else ""


def stored(connection: Database, table: str) -> dict[str, tuple[float, ...]]:
    """Every vector the table holds, as the numbers the column stores.

    Read through `::text` rather than through the driver, so a distance is compared against
    what the column would compute it from rather than against a wider copy of it. The values
    are narrowed back through 32 bits because that is how pgvector prints them: the text is
    the shortest decimal that names the stored float, so parsing it and narrowing it returns
    the float itself, and a value that had been mangled rather than narrowed would not come
    back equal.

    The table name goes in as an identifier because the module refuses to interpolate one and
    a test that did otherwise would be reading the module's own claim down.
    """
    statement = sql.SQL(
        "SELECT post_id, embedding::text AS literal FROM {table} ORDER BY post_id"
    ).format(table=sql.Identifier(table))
    vectors: dict[str, tuple[float, ...]] = {}
    for row in connection.execute(statement).fetchall():
        values = str(row["literal"]).strip("[]").split(",")
        vectors[str(row["post_id"])] = tuple(
            as_float32(float(value)) for value in values if value
        )
    return vectors


def as_float32(value: float) -> float:
    """The value as a 32-bit float and back, which is the width the column keeps."""
    packed: tuple[float, ...] = struct.unpack("<f", struct.pack("<f", value))
    return packed[0]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, worked out here so the test is not trusting the column."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """The distance the column computes, from the same 32-bit values it would use."""
    return 1.0 - cosine(left, right)


def write_corpus(path: Path, items: Sequence[CorpusItem]) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "post_id": item.post_id,
                    "account": item.account,
                    "subreddit": item.subreddit,
                    "title": item.title,
                    "body": item.body,
                    "created_at": item.created_at,
                    "links": list(item.links),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    return path


def post(post_id: str, account: str, *links: str, body: str = "body") -> CorpusItem:
    return CorpusItem(
        post_id=post_id,
        account=account,
        subreddit="test",
        title=f"title of {post_id}",
        body=body,
        created_at="2026-01-05T09:00:00Z",
        links=links,
    )


# --- what goes in the column ------------------------------------------------------


def test_every_content_item_in_the_corpus_is_stored_as_a_vector(
    tmp_path: Path, table: str
) -> None:
    """The first acceptance criterion, read as a question with two answers.

    Every post in the Corpus and every row in the table, including the ones whose text is
    short and whose links name one host â€” the rows an implementation that filters for
    embeddability would drop, and the Corpus holds several. Nothing is stored that the Corpus
    does not hold either, checked in the test below.
    """
    run(tmp_path, table=table)

    items = read_corpus(COMMITTED_CORPUS)
    connection = open_store(connection_string())
    try:
        vectors = stored(connection, table)
    finally:
        connection.close()

    assert set(vectors) == {item.post_id for item in items}
    assert all(len(vector) == DIMENSIONS for vector in vectors.values())


def test_a_row_the_corpus_no_longer_holds_is_removed_rather_than_left_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The table is a projection of the Corpus, and a stale row would come back from a query.

    A vector nobody can reach is not harmless: it comes out of every similarity query as
    though it were a post somebody could go and read, and the nearest-neighbour table would
    name a post id the Corpus does not contain.
    """
    wide = write_corpus(
        tmp_path / "wide.jsonl",
        (
            post("syn_p_9001", "syn_a_0001"),
            post("syn_p_9002", "syn_b_0002"),
            post("syn_p_9003", "syn_c_0003"),
        ),
    )
    narrow = write_corpus(tmp_path / "narrow.jsonl", (post("syn_p_9001", "syn_a_0001"),))

    run(tmp_path / "wide", wide, capsys=capsys, table=table)
    _, printed = run(tmp_path / "narrow", narrow, capsys=capsys, table=table)

    assert "2 removed" in printed, printed
    connection = open_store(connection_string())
    try:
        assert set(stored(connection, table)) == {"syn_p_9001"}
    finally:
        connection.close()


def test_the_vector_the_table_holds_is_the_vector_the_model_produced(
    tmp_path: Path, table: str
) -> None:
    """The round trip, to the width the column stores.

    `float32` rather than equality, because pgvector's `vector` is a column of 32-bit floats
    and the model produces 64-bit ones. The comparison is against the model narrowed the same
    way rather than against a tolerance, so it holds exactly and a vector that was mangled
    rather than narrowed would fail it.
    """
    run(tmp_path, table=table)

    computed = {
        found.record.post_id: tuple(as_float32(value) for value in found.embedding)
        for found in compute_embeddings(read_corpus(COMMITTED_CORPUS))
    }
    connection = open_store(connection_string())
    try:
        assert stored(connection, table) == computed
    finally:
        connection.close()


def test_the_column_is_a_vector_of_this_projects_width(table: str) -> None:
    """The width is in the schema, not in a setting, so a table built for another one cannot
    be written to at all.

    Checked here rather than inferred from the refusal below, because a refusal only fires
    when there is something to refuse and this holds on a table this project created.
    """
    connection = open_store(connection_string())
    try:
        ensure_table(connection, table)
        width = connection.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS declared
              FROM pg_attribute a
              JOIN pg_class c ON c.oid = a.attrelid
             WHERE c.relname = %s AND a.attname = 'embedding'
            """,
            (table,),
        ).fetchone()
    finally:
        connection.close()

    assert width is not None
    assert width["declared"] == f"vector({DIMENSIONS})"


# --- the similarity query ---------------------------------------------------------


def test_a_similarity_query_returns_the_nearest_items(
    tmp_path: Path, table: str
) -> None:
    """The second acceptance criterion, checked against arithmetic rather than row order.

    The query is run for every post over the Corpus this project ships, and the distance it
    returns is compared with the cosine distance worked out here from the vectors the column
    holds â€” both 32-bit values, so the two agree to the precision pgvector computes at. A
    query that returned rows in the right order with the wrong numbers, or the right numbers
    for the wrong rows, fails this.

    The order is held to non-decreasing and the identities are held to their distances rather
    than compared position by position, because two posts at exactly the same distance may
    come back in either order and a test that insisted on one of them would be testing the
    planner rather than the query.
    """
    run(tmp_path, table=table)
    connection = open_store(connection_string())
    try:
        vectors = stored(connection, table)
        for post_id, vector in sorted(vectors.items()):
            found = nearest(connection, table, post_id, limit=3)
            expected = sorted(
                (cosine_distance(vector, vectors[other]), other)
                for other in vectors
                if other != post_id
            )

            assert len(found) == 3, f"{post_id} came back with {len(found)} neighbours"
            assert post_id not in {one.nearest for one in found}
            returned = [one.distance for one in found]
            assert returned == sorted(returned), f"{post_id}'s neighbours are out of order"
            for one, (want, _) in zip(found, expected[:3], strict=True):
                assert math.isclose(
                    one.distance, want, abs_tol=1e-5
                ), f"{post_id}'s distance to {one.nearest} is not the one the model gives"
            assert found[0].nearest == expected[0][1], f"{post_id}'s nearest is not the closest"
    finally:
        connection.close()


def test_the_nearest_post_to_a_post_is_the_one_a_reader_would_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The Corpus's own staggered paraphrases, which is the case the model exists for.

    `syn_p_0012`, `syn_p_0013` and `syn_p_0014` are one Planted Campaign's offer reworded
    across its accounts, and the run has to put them at the top of its own nearest-neighbour
    table with a distance well below anything else on the Corpus. If the closest pair on the
    Corpus were two unrelated posts, the substrate would be building nothing ticket #24 could
    use, and this figure would be the only thing standing between that and a reader's
    assumption.
    """
    _, printed = run(tmp_path, capsys=capsys, table=table)

    connection = open_store(connection_string())
    try:
        vectors = stored(connection, table)
    finally:
        connection.close()

    pairs = sorted(
        (cosine_distance(vectors[left], vectors[right]), left, right)
        for left in vectors
        for right in vectors
        if left < right
    )
    closest = pairs[0]

    assert {closest[1], closest[2]} == {"syn_p_0012", "syn_p_0013"}, closest
    assert cosine_distance(vectors["syn_p_0012"], vectors["syn_p_0014"]) < 0.3
    for post_id in ("syn_p_0012", "syn_p_0013", "syn_p_0014"):
        line = [row for row in printed.splitlines() if row.startswith(post_id)]
        assert line, f"{post_id} is not in the nearest table"
        assert float(line[0].split()[-1]) < 0.25, line[0]


def test_the_table_prints_no_post_as_its_own_nearest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """A nearest-neighbour list that led with the post itself would answer nothing.

    Every row of the printed table has to name a different post, and the run has to say how
    many rows there are rather than letting a reader count them.
    """
    _, printed = run(tmp_path, capsys=capsys, table=table)

    lines = [line for line in printed.splitlines() if line.startswith("syn_p_")]
    pairs = [(line.split()[0], line.split()[1]) for line in lines]

    assert len(pairs) == len(read_corpus(COMMITTED_CORPUS))
    assert [post_id for post_id, _ in pairs] == [
        item.post_id for item in read_corpus(COMMITTED_CORPUS)
    ]
    assert [nearest_name for _, nearest_name in pairs if _ == nearest_name] == []


def test_asking_about_a_post_with_no_vector_is_refused(tmp_path: Path, table: str) -> None:
    """The two failures a caller cannot tell apart: no vector, and no table.

    A query that returned nothing for an unknown post would read as a post nothing is near,
    which is a claim about the Corpus rather than a statement about the question.
    """
    run(tmp_path, table=table)
    connection = open_store(connection_string())
    try:
        with pytest.raises(ValueError, match="has no vector stored"):
            nearest(connection, table, "syn_p_9999")
    finally:
        connection.close()


# --- the reuse rule ---------------------------------------------------------------


def test_a_corpus_holding_one_post_twice_is_refused_before_anything_is_stored(
    tmp_path: Path, table: str
) -> None:
    """Two rows for one post would be embedded twice, written twice, and counted once.

    `read_corpus` does not refuse a repeated post id, and nothing else in this path needs it
    to: a duplicate only becomes a problem here, where the reuse figures and the delete's
    rowcount are both computed over the same list. So it is refused where the damage is, and
    the refusal says what the damage would have been rather than only naming the post.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_9001", "syn_a_0001"),
            post("syn_p_9001", "syn_b_0002"),
        ),
    )
    records_path = tmp_path / "content-embeddings.jsonl"

    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "content-embeddings",
                "--corpus",
                str(corpus),
                "--embeddings",
                str(records_path),
                "--table",
                table,
            ]
        )

    assert "syn_p_9001" in str(refusal.value)
    assert not records_path.exists(), "the run published a record for a Corpus it refused"
    connection = open_store(connection_string())
    try:
        assert stored(connection, table) == {}
    finally:
        connection.close()


def test_a_rerun_over_an_unchanged_corpus_computes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The fourth acceptance criterion, as the figure the run prints.

    Not "the file is the same" â€” a run that recomputed every vector and wrote the same file
    would pass that. What has to hold is that the second run embedded nothing, because the
    digest stored beside each vector said the text had not changed.
    """
    items = read_corpus(COMMITTED_CORPUS)
    first, first_printed = run(tmp_path / "first", capsys=capsys, table=table)
    second, second_printed = run(tmp_path / "second", capsys=capsys, table=table)

    assert f"{len(items)} computed" in first_printed, first_printed
    assert "0 computed" in second_printed, second_printed
    assert f"{len(items)} reused" in second_printed, second_printed
    assert first.read_bytes() == second.read_bytes()


def test_only_the_content_item_that_changed_is_computed_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """Reuse is per Content Item, not per Corpus, or one edited post costs the whole Corpus.

    Two posts against a table holding three: one edited, one not. The run has to compute one
    and reuse two, which is what makes the rule a rule about content rather than a rule about
    runs.
    """
    first = write_corpus(
        tmp_path / "first.jsonl",
        (
            post("syn_p_9001", "syn_a_0001", body="the ledger published its losing months"),
            post("syn_p_9002", "syn_b_0002", body="a warehouse that never paid anybody"),
            post("syn_p_9003", "syn_c_0003", body="complaint about a clinic"),
        ),
    )
    second = write_corpus(
        tmp_path / "second.jsonl",
        (
            post("syn_p_9001", "syn_a_0001", body="the ledger published its losing months"),
            post("syn_p_9002", "syn_b_0002", body="a warehouse that eventually paid"),
            post("syn_p_9003", "syn_c_0003", body="complaint about a clinic"),
        ),
    )

    run(tmp_path / "one", first, capsys=capsys, table=table)
    _, printed = run(tmp_path / "two", second, capsys=capsys, table=table)

    assert "1 computed, 2 reused" in printed, printed


def test_the_reuse_rule_is_the_digest_of_the_embedded_text_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """Renaming every account reuses everything, because no account is embedded.

    The structural half of the leak boundary, run through the database rather than through
    the model: a table full of reused rows after every account has been renamed is a table
    whose vectors cannot carry who posted. `test_content_embeddings.py` holds the same claim
    on the pure half.
    """
    original = read_corpus(COMMITTED_CORPUS)
    renamed = write_corpus(
        tmp_path / "renamed.jsonl",
        [
            CorpusItem(
                post_id=item.post_id,
                account=f"syn_renamed_{index:04d}",
                subreddit="somewhere-else",
                title=item.title,
                body=item.body,
                created_at=item.created_at,
                links=item.links,
            )
            for index, item in enumerate(original)
        ],
    )

    run(tmp_path / "first", capsys=capsys, table=table)
    _, printed = run(tmp_path / "second", renamed, capsys=capsys, table=table)

    assert "0 computed" in printed, printed
    assert f"{len(original)} reused" in printed, printed


def test_the_record_never_holds_a_vector_and_the_table_does(
    tmp_path: Path, table: str
) -> None:
    """The published file is five fields; the vectors are in the column.

    Checked from both sides in one test because the claim is a division of labour: a reader
    who opens the file is looking at a record of what is stored, not at the stored numbers.
    """
    records_path, _ = run(tmp_path, table=table)

    assert set(json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])) == {
        "post_id",
        "model",
        "dimensions",
        "recipe",
        "text_sha256",
    }
    connection = open_store(connection_string())
    try:
        vectors = stored(connection, table)
    finally:
        connection.close()
    assert len(vectors) == len(read_corpus(COMMITTED_CORPUS))


# --- the index and the plan --------------------------------------------------------


def test_the_index_is_hnsw_and_the_run_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The third acceptance criterion's first half: the index type is this one, on purpose.

    Read out of the catalogue rather than out of the run's own word for it, so a run that
    printed `hnsw` over a B-tree would fail here. The reason HNSW was chosen over IVFFlat is
    in ADR-0024 and the command points at it.
    """
    _, printed = run(tmp_path, capsys=capsys, table=table)

    connection = open_store(connection_string())
    try:
        method = connection.execute(
            """
            SELECT am.amname AS method
              FROM pg_class c
              JOIN pg_am am ON am.oid = c.relam
              JOIN pg_index i ON i.indexrelid = c.oid
             WHERE i.indrelid = to_regclass(%s) AND c.relname = %s
            """,
            (table, f"{table}_hnsw"),
        ).fetchone()
    finally:
        connection.close()

    assert method is not None
    assert method["method"] == "hnsw"
    assert f"{table}_hnsw" in printed, printed
    assert "vector_cosine_ops" in printed, printed


def test_an_index_that_is_not_hnsw_is_refused_rather_than_assumed(
    tmp_path: Path, table: str
) -> None:
    """`CREATE INDEX IF NOT EXISTS` is satisfied by a name, so the method has to be checked.

    Somebody else's B-tree under the name this project would use would otherwise be left in
    place, every similarity query would ignore it, and the figure printed beside the index
    would describe an index that cannot answer a cosine query at all.
    """
    connection = open_store(connection_string())
    try:
        connection.execute(
            f"""
            CREATE TABLE "{table}" (
                post_id text PRIMARY KEY,
                model text NOT NULL,
                dimensions integer NOT NULL,
                recipe char(64) NOT NULL,
                text_sha256 char(64) NOT NULL,
                embedding vector({DIMENSIONS}) NOT NULL
            )
            """
        )
        connection.execute(f'CREATE INDEX "{table}_hnsw" ON "{table}" (post_id)')

        with pytest.raises(ValueError, match="a btree index"):
            ensure_index(connection, table)
    finally:
        connection.close()


def test_the_run_reports_the_plan_the_planner_actually_chose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The third acceptance criterion's second half, and the reason it is printed at all.

    Over thirty-odd rows the planner prefers a sequential scan, so the run reports that and
    the footer says why it is the right answer rather than claiming the index is at work. A
    run that printed `Index Scan` here would be claiming something no `EXPLAIN` agrees with,
    and the first Corpus to grow past the planner's threshold would inherit the claim.
    """
    _, printed = run(tmp_path, capsys=capsys, table=table)

    assert "Seq Scan" in printed, printed
    assert "not the HNSW index" in printed, printed
    assert "and Postgres knows it" in printed, printed


def test_the_index_the_run_reports_is_the_size_of_the_index_the_run_built(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The memory half of the index argument, as a figure rather than an assertion.

    An argument about memory that prints no size is an argument a reader cannot check, and
    the whole reason HNSW over IVFFlat is cheap at this size is that both are kilobytes here.
    The printed number is compared to the catalogue's, not merely checked for containing
    `kB`: a formatter that printed the wrong size would otherwise pass.
    """
    _, printed = run(tmp_path, capsys=capsys, table=table)

    connection = open_store(connection_string())
    try:
        size = connection.execute(
            "SELECT pg_relation_size(c.oid) AS bytes FROM pg_class c WHERE c.relname = %s",
            (f"{table}_hnsw",),
        ).fetchone()
    finally:
        connection.close()

    assert size is not None
    assert isinstance(size["bytes"], int)
    assert size["bytes"] > 0

    line = [row for row in printed.splitlines() if row.strip().startswith("indexed")][0]
    quoted = re.search(r"([0-9.]+) (kB|MB|bytes)", line)
    assert quoted is not None, line
    scale = {"bytes": 1, "kB": 1024, "MB": 1024 * 1024}[quoted.group(2)]
    assert math.isclose(
        float(quoted.group(1)) * scale, float(size["bytes"]), rel_tol=0.05
    ), f"the run printed {line!r} and the index is {size['bytes']} bytes"


# --- what the table is allowed to hold ---------------------------------------------


def test_a_table_holding_another_models_vectors_is_refused_rather_than_rewritten(
    tmp_path: Path, table: str
) -> None:
    """The fifth acceptance criterion, enforced where it can be enforced.

    A vector from one model has no distance to a vector from another, so a table holding both
    holds nothing a query can answer. Rewriting would destroy the old vectors and make the two
    runs incomparable after the fact; refusing keeps both, and `--table` is how a reader
    chooses to keep them.
    """
    connection = open_store(connection_string())
    try:
        connection.execute(
            f"""
            CREATE TABLE "{table}" (
                post_id text PRIMARY KEY,
                model text NOT NULL,
                dimensions integer NOT NULL,
                recipe char(64) NOT NULL,
                text_sha256 char(64) NOT NULL,
                embedding vector({DIMENSIONS}) NOT NULL
            )
            """
        )
        connection.execute(
            f'INSERT INTO "{table}" VALUES (%s, %s, %s, %s, %s, %s::vector)',
            (
                "syn_p_9001",
                "some-other-model",
                DIMENSIONS,
                recipe_digest(),
                "0" * 64,
                "[" + ",".join("1" for _ in range(DIMENSIONS)) + "]",
            ),
        )

        with pytest.raises(ValueError, match="another model"):
            ensure_table(connection, table)
    finally:
        connection.close()


def test_a_table_whose_recipe_has_changed_is_refused_too(
    tmp_path: Path, table: str
) -> None:
    """The name carries a version and the recipe digest is the promise checked.

    Same name, same width, same table â€” but the rule that produced those vectors is not the
    rule this run writes under. Nothing else would notice: the model name would match, the
    width would match, and the digests beside the rows would match the text. Only the recipe
    says the vectors came from a different rule, and refusing is what keeps a rerun from
    reusing them.
    """
    connection = open_store(connection_string())
    try:
        connection.execute(
            f"""
            CREATE TABLE "{table}" (
                post_id text PRIMARY KEY,
                model text NOT NULL,
                dimensions integer NOT NULL,
                recipe char(64) NOT NULL,
                text_sha256 char(64) NOT NULL,
                embedding vector({DIMENSIONS}) NOT NULL
            )
            """
        )
        connection.execute(
            f'INSERT INTO "{table}" VALUES (%s, %s, %s, %s, %s, %s::vector)',
            (
                "syn_p_9001",
                MODEL_NAME,
                DIMENSIONS,
                "0" * 64,
                "0" * 64,
                "[" + ",".join("1" for _ in range(DIMENSIONS)) + "]",
            ),
        )

        with pytest.raises(ValueError, match="another model"):
            ensure_table(connection, table)
    finally:
        connection.close()


def test_the_model_and_its_width_are_published_on_every_row_and_in_the_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """Both, and the recipe digest too, because a reader comparing two stores needs all three.

    The file is the durable record and the output is the one a reader meets without opening
    anything, so a model this project changed without saying so would be caught in two places
    rather than none.
    """
    records_path, printed = run(tmp_path, capsys=capsys, table=table)

    assert {record.model for record in read_content_records(records_path)} == {MODEL_NAME}
    assert {record.dimensions for record in read_content_records(records_path)} == {DIMENSIONS}
    assert {record.recipe for record in read_content_records(records_path)} == {recipe_digest()}
    assert f"{MODEL_NAME}, {DIMENSIONS} dimensions" in printed, printed
    assert recipe_digest() in printed, printed


def test_a_database_without_the_extension_is_refused_with_the_statement_to_run() -> None:
    """`CREATE EXTENSION` is not this command's to run.

    It is per database, it needs a role permitted to create one, and it is a persistent change
    to the database rather than to a row in it â€” which is why `docs/pgvector.md` holds the
    activation step and this command refuses rather than performing it. Exercised on a
    scratch database because the extension is per database and the one this suite is pointed
    at has it.
    """
    scratch_name = "rfi_without_vector"
    connection = open_store(connection_string())
    try:
        connection.execute(f'DROP DATABASE IF EXISTS "{scratch_name}"')
        try:
            connection.execute(f'CREATE DATABASE "{scratch_name}"')
        except psycopg.Error as refusal:
            pytest.skip(f"this role may not create a database: {refusal}")
    finally:
        connection.close()

    scratch = open_store(make_conninfo(connection_string(), dbname=scratch_name))
    try:
        with pytest.raises(ValueError, match=r"CREATE EXTENSION vector;"):
            ensure_table(scratch, "content_embedding")
    finally:
        scratch.close()
        open_store(connection_string()).execute(f'DROP DATABASE IF EXISTS "{scratch_name}"')


# --- what the run is allowed to reach ----------------------------------------------


def test_the_run_reads_the_corpus_and_its_own_published_record_and_nothing_else(
    tmp_path: Path, table: str
) -> None:
    """Watched rather than asserted from the code, because what a reader can check is what
    the process touched.

    A command that reached into the Planted Campaign membership, the Nuisance Structure
    manifest, or the Public Suffix List would be building something other than what the
    ticket asked for, and none of those is a file it has any reason to open. The record is
    here because the command writes it and then reads it back, which is the check that keeps
    the file within the vocabulary this project publishes (ADR-0008); the Corpus is read twice
    because it is read once for the content and once for the digest printed beside it.
    """
    opened = _Opened()
    sys.addaudithook(opened)

    opened.recording = True
    try:
        run(tmp_path, table=table)
    finally:
        opened.recording = False

    assert {Path(path).name for path in opened.record()} == {
        "corpus.jsonl",
        "content-embeddings.jsonl",
    }
    assert not [path for path in opened.record() if Path(path).name.startswith("truth")]
    assert not [path for path in opened.record() if Path(path).name == "nuisance.jsonl"]


def test_the_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str) -> None:
    """Same argument as every other command here: the numbers come from the Corpus and from
    a database the caller already had."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the content-embeddings command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    run(tmp_path, table=table)


def test_the_connection_string_is_read_from_the_environment_and_never_from_a_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `--dsn`, because an argument is where a password ends up.

    Asserted on the refusal as well as the two sources: a run with nothing configured would
    otherwise be handed a database libpq picked by itself, and writing an embeddings table
    into whatever that is is not a decision this command should make.
    """
    for name in ("PGHOST", "PGPORT", "PGUSER", "PGDATABASE", "RFI_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="no database is configured"):
        connection_string()

    monkeypatch.setenv("RFI_DATABASE_URL", "postgresql://somebody:secret@localhost:5433/rfi")
    assert connection_string() == "postgresql://somebody:secret@localhost:5433/rfi"

    monkeypatch.delenv("RFI_DATABASE_URL")
    monkeypatch.setenv("PGHOST", "localhost")
    assert connection_string() == ""


def test_no_password_reaches_the_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The run prints its own connection, and a connection carries a credential.

    `Connection.info` holds the password and the figures read only the host, the port, the
    database and the user. A run over a real credential that printed one would be putting it in
    whatever the output was pasted into, which is the whole of why the credentials are read
    from the environment in the first place. The password is read from the environment here
    too, so no test file holds a credential either.
    """
    password = os.environ.get("PGPASSWORD")
    if not password:
        pytest.skip(
            "PGPASSWORD is not set, so there is no credential for the output to have leaked"
        )

    _, printed = run(tmp_path, capsys=capsys, table=table)

    assert password not in printed
    assert "password=" not in printed.lower()


# --- the committed file and the shape of the output --------------------------------


def test_the_committed_record_is_what_the_command_writes(
    tmp_path: Path, table: str
) -> None:
    """Held to the command the way every generated file here is held to the thing that wrote
    it. `test_content_embeddings.py` holds the other half, which needs no database."""
    records_path, _ = run(tmp_path, table=table)

    assert records_path.read_bytes() == COMMITTED_RECORDS.read_bytes()


def test_the_console_block_the_readme_quotes_is_the_block_this_run_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The README copies figures out of the output, and nothing held it to them.

    `tests/test_readme_figures.py` exists because three of the README's figures were once a
    commit behind. This one is the console block the Content Embeddings section quotes, and
    every figure in it is read out of a run rather than written into the test: the index's
    size is read out of the catalogue and its shape out of the output, so a Corpus or a
    pgvector that moved them fails here instead of quietly leaving the README behind.

    It runs twice because the reuse figures the README quotes are the ones a *second* run
    prints, and a test that read them off a first run would be reading something else. The
    table name is the one figure checked against the README on its own, because the README
    quotes a run over the project's own table and this is a table of the test's own.
    """
    run(tmp_path / "first", capsys=capsys, table=table)
    _, printed = run(tmp_path / "second", capsys=capsys, table=table)
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    items = read_corpus(COMMITTED_CORPUS)

    connection = open_store(connection_string())
    try:
        size = connection.execute(
            "SELECT pg_relation_size(c.oid) AS bytes FROM pg_class c WHERE c.relname = %s",
            (f"{table}_hnsw",),
        ).fetchone()
    finally:
        connection.close()
    assert size is not None and isinstance(size["bytes"], int)

    figures = (
        f"({len(items)} rows)",
        f"hnsw on embedding vector_cosine_ops, {size['bytes'] / 1024:.1f} kB "
        f"over {len(items)} rows",
        f"{len(items)} Content Items: 0 computed, {len(items)} reused, 0 removed",
        "Seq Scan, not the HNSW index",
    )
    missing = [line for line in figures if line not in readme]
    quiet = [line for line in figures if line not in printed]

    assert not missing, f"README.md does not quote the run's own figures: {missing}"
    assert not quiet, f"the run does not print the figure the README quotes: {quiet}"
    assert f"{DEFAULT_EMBEDDING_TABLE} ({len(items)} rows)" in readme
    assert index_name(DEFAULT_EMBEDDING_TABLE) in readme
    assert f"{table} ({len(items)} rows)" in printed
    assert index_name(table) in printed


def test_running_twice_over_the_same_table_writes_byte_identical_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], table: str
) -> None:
    """The bytes do not move because a run reused its vectors, which is the whole point of
    the rule. Asserted over two runs against one table rather than two tables, so a run that
    wrote the current time into the file would fail."""
    first, _ = run(tmp_path / "first", capsys=capsys, table=table)
    second, _ = run(tmp_path / "second", capsys=capsys, table=table)

    assert first.read_bytes() == second.read_bytes()


def test_the_console_output_is_plain_ascii_within_its_own_width(
    tmp_path: Path, table: str
) -> None:
    """So it prints the same way on a console that cannot encode anything else and the same
    way when it is redirected, which is what lets it be pasted into an issue.

    Asserted on `render_table` rather than on the command's whole stdout, because the command
    adds two lines naming the paths it wrote and how long a temporary path is has nothing to
    do with what this module claims about its own output. The nearest-neighbour table is
    exempt too: a row is as wide as its widest cell, and truncating one would make a post id
    unreadable.
    """
    printed = render_table(embed_corpus(COMMITTED_CORPUS, table))

    assert printed.isascii()
    assert "\r" not in printed
    too_wide = [
        line
        for line in printed.splitlines()
        if len(line) > 100
        and not line.startswith("syn_p_")
        and not line.startswith("post ")
    ]
    assert not too_wide, f"lines wider than 100 columns: {too_wide}"


def test_the_database_figure_prints_only_the_numeric_version(
    tmp_path: Path, table: str
) -> None:
    """The width of that line is a property of the machine, not of this module.

    A Debian-packaged server answers `server_version` with `18.6 (Debian 18.6-1.pgdg12+2)`,
    forty-one characters of distribution bookkeeping; a source build answers `18.1`. The
    version that matters to a `vector` column is the numeric one, so that is what is printed.
    A line whose width changes with the packaging of the host is a line the width test above
    cannot pass on every machine it claims to, and that test passed here only because this
    machine's string is short.
    """
    printed = render_table(embed_corpus(COMMITTED_CORPUS, table))
    line = next(
        line for line in printed.splitlines() if line.strip().startswith("database")
    )
    matched = re.search(r"PostgreSQL ([^,]+), pgvector", line)
    assert matched is not None, f"no PostgreSQL version on the database line: {line!r}"
    printed_version = matched.group(1)
    with open_store(connection_string()) as connection:
        row = connection.execute(
            "SELECT current_setting('server_version') AS version"
        ).fetchone()
    assert row is not None
    answered = str(row["version"])

    assert re.fullmatch(r"\d+(\.\d+)*", printed_version), (
        f"printed {printed_version!r}, which is not a bare version: the host answered "
        f"{answered!r}"
    )
    assert printed_version == answered.split(" ")[0]


def test_the_corpus_figure_is_printed_relative_to_where_the_command_runs(
    tmp_path: Path, table: str
) -> None:
    """So the line is the same length everywhere the repository happens to be checked out.

    `Path(__file__).parent.parent` is absolute, so a caller that hands this command an absolute
    path gets that whole path back in its output. A CI checkout sits at
    `/home/runner/work/<owner>/<repository>`, and the line carrying it runs past the width the
    output claims to hold. A path under the working directory is printed relative to it, which
    is both shorter and what the README already quotes.
    """
    inside = Path.cwd() / "data" / "corpus" / "corpus.jsonl"

    printed = render_table(embed_corpus(inside, table))

    assert "corpus    data/corpus/corpus.jsonl (" in printed
    assert str(Path.cwd()) not in printed


def test_a_corpus_outside_the_working_directory_is_still_printed_in_full(
    tmp_path: Path, table: str
) -> None:
    """Relative-to-here is a shortening, not a truncation.

    A Corpus in a temporary directory is not under the working directory, so there is nothing
    to be relative to, and printing only its last component would leave a reader unable to say
    which of two such files was read. The SHA-256 on the line below identifies it either way,
    but the path is what a reader opens.
    """
    corpus = write_corpus(tmp_path / "elsewhere.jsonl", (post("syn_p_9101", "syn_x_0101"),))

    printed = render_table(embed_corpus(corpus, table))

    assert f"corpus    {corpus.as_posix()} (1 post across 1 account)" in printed


def test_the_command_refuses_to_write_the_corpus_away(tmp_path: Path, table: str) -> None:
    """The Corpus is the input. A path collision that overwrote it would destroy the thing
    the whole measurement is against, and every other command here has the same guard."""
    corpus = write_corpus(tmp_path / "corpus.jsonl", (post("syn_p_9005", "syn_x_0005"),))

    with pytest.raises(SystemExit):
        main(
            [
                "content-embeddings",
                "--corpus",
                str(corpus),
                "--embeddings",
                str(corpus),
                "--table",
                table,
            ]
        )

    assert read_corpus(corpus) == (post("syn_p_9005", "syn_x_0005"),)


def test_the_run_publishes_the_digest_of_every_content_items_own_text(
    tmp_path: Path, table: str
) -> None:
    """What the file is for, end to end: a reader can recompute every digest from the Corpus
    and see which text each stored vector belongs to."""
    records_path, _ = run(tmp_path, table=table)

    published = {record.post_id: record.text_sha256 for record in read_content_records(records_path)}
    for item in read_corpus(COMMITTED_CORPUS):
        assert published[item.post_id] == digest_of(embedded_text(item))