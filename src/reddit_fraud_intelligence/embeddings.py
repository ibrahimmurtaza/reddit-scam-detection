"""A sentence embedding for every Content Item, stored in pgvector (ADR-0024).

This is the substrate ticket: one vector per Content Item, in a column a similarity query
can be asked of, under an index chosen on purpose. It builds no grouping, corroborates
nothing and decides nothing. Ticket #24 is the first thing to read these vectors, and every
acceptance criterion here is about that reading being possible later rather than about it
being good now.

**What is embedded is the post's own title, its own body, and its links, and nothing
else.** No account, no subreddit, no timestamp, and no field the Corpus file does not
already carry. The Corpus boundary exists so the pipeline cannot see Planted Campaign
membership (ADR-0008), and a vector built out of who posted would put that back through a
side door no reader of the file could see. `tests/test_content_embeddings.py` holds that
structurally: the whole Corpus is re-read with every account renamed and every post moved,
and the vectors and the digests have to come back identical.

**The model is this project's own and it is published.** `MODEL_NAME`, `DIMENSIONS` and a
digest of the recipe below travel on every row, because a vector's comparability is a property
of that vector: two models' numbers have no common distance, and a table holding both would
hold nothing. The recipe digest is the half of that a version number cannot be on its own —
editing the recipe without bumping the name stops the run rather than quietly reusing vectors
the new recipe would not have produced. The recipe is stated in prose and digested rather than
derived from this file, because a digest of the code would move whenever a comment did, and
what a reader needs is a statement of what the model does.

The model is lexical. It reads shared words and the pairs of adjacent words inside one
field, and it does not read meaning: two posts making the same pitch in different words
can be far apart, and two posts linking the same shortener share the tokens of that host.
Both are limits rather than surprises and both are printed, because ticket #24 rests on
this and a substrate that quietly oversells itself is worse than one that admits what it is.

**A vector is computed once and reused.** Every row carries the SHA-256 of the text it was
computed from, so a rerun over an unchanged Corpus embeds nothing and prints that it
computed nothing. The decision is taken against the stored digest *before* anything is
embedded: an implementation that embedded first and skipped the write would satisfy the
rule at the writer rather than at the work.

**What is published is the record, not the vectors.** `data/embeddings/content-embeddings.jsonl`
says which post, which model, how wide, under which recipe, and the digest of the text behind
each stored vector. The vectors live in Postgres, which is where the ticket asked for them,
and 34 posts of 256 numbers is a file of noise a reader would have to trust rather than check.
What the record buys is the reuse rule in the open — every digest can be recomputed from the
Corpus, so a reader can see which text each stored vector belongs to — and it is held to its
bytes by a test like every other generated file here.

**The index is HNSW, chosen rather than defaulted.** IVFFlat needs training data before it
answers anything and its `lists` parameter is a guess that has to be revisited as a corpus
grows; HNSW answers from the moment it is built and holds its recall without a parameter.
At this corpus size the index is a few kilobytes and the planner will not use it, so the
output reports what the planner actually chose rather than claiming the index is at work.
Both halves of that argument, and the memory figures on both sides of it, are in ADR-0024.

Nothing on this path reads the truth file or the Nuisance Structure manifest. The Corpus
and one database are the whole input, and no account's age, karma, posting rate or activity
change is read anywhere on it (ADR-0007).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.jsonl import (
    JsonObject,
    read_object,
    read_rows,
    read_text,
    refuse_repeated,
    write_lines,
)
from reddit_fraud_intelligence.text import wrap

_HEADING = "Content Embeddings"

# Wrapped rather than continued, so the paragraph is as many lines a reader sees as the
# footer below it is. A backslash-continued literal is one very long line at run time, which
# is the shape this project's console output is trying not to have.
_SUBHEADING = """\
One sentence embedding per Content Item, stored in Postgres through pgvector and indexed for
similarity search. What is embedded is the post's own title, its own body, and its links, and
nothing else: no account, no subreddit, no timestamp, because the Corpus file is the boundary
this project measures itself against, and a vector built out of who posted would put Planted
Campaign membership back through a side door (ADR-0008). Nothing here groups anything,
corroborates anything, or decides anything, and no account's age, karma, posting rate, or
activity change is read on the way (ADR-0007)."""

# The model, as two facts a reader needs before a stored vector means anything. The name
# carries the recipe's version, because a change to any part of that recipe changes every
# number the model produces — and a width changed without the name changing would be the
# worst version of this bug, since every stored vector would still parse.
MODEL_NAME = "hashed-word-ngrams-v1"
DIMENSIONS = 256

# What the model does, in one paragraph, and the digest of it beside the name and the width.
# Published rather than derived from this file: a digest taken over the code would move
# whenever a comment or a type annotation did, and what a reader needs is a statement of the
# rule they can hold their own vectors against.
_RECIPE = (
    "tokens are runs of Unicode letters and digits, case folded, read from the title, the "
    "body and each link as three separate sequences so no pair of words spans two of them; "
    "a feature is each token and each pair of adjacent tokens within one sequence; a "
    "feature's count is damped as 1 + ln(count); each feature is placed into one of 256 "
    "buckets by the first eight bytes of blake2b over its UTF-8 bytes, and the top bit of "
    "that digest is its sign, so a collision cancels rather than compounds; the vector is "
    "scaled to unit length, and a Content Item with no tokens at all is refused"
)

# A run of letters and digits, case folded, with the underscore treated as a separator:
# `syn_vantageledger` is two tokens. The `syn_` marker every Synthetic Entity in this Corpus
# carries then becomes one feature shared by every post, which cosine distance reads as a
# direction every post has rather than as evidence about any of them.
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

# Words and pairs rather than characters, and two rather than three. Words are what two
# posts about the same thing share; characters make a short post and a long one on the same
# subject neighbours for the wrong reason. The pair is what separates "no experience needed"
# from "no experience wanted", and it is the cheapest way to stop every short post about
# employment embedding near every other one. Declared in one place so a reader can see the
# whole of what the model looks at, and change it in one place.
_NGRAM_ORDERS = (1, 2)

# The width a footer line is wrapped to, which is the console claim this command makes and
# the width its own test holds it to.
_WIDTH = 100

# Where the connection comes from, and what libpq reads when it is not given a string.
_DATABASE_URL = "RFI_DATABASE_URL"
_ENVIRONMENT = ("PGHOST", "PGPORT", "PGUSER", "PGDATABASE")

Database = psycopg.Connection[Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class ContentRecord:
    """One Content Item's embedding, as this project publishes it.

    No vector. A record says which post, which model, how wide, and the digest of the text
    the stored vector was computed from, which is what lets a rerun decide not to recompute
    and what lets a reader recompute every digest from the Corpus and see which text each
    stored vector belongs to.

    `recipe` is what closes the gap between the model's name and its behaviour. A name carrying
    a version is a promise somebody has to keep; the digest of the recipe is the promise
    checked, so editing the recipe without bumping the name stops the run rather than quietly
    reusing vectors the new recipe would not have produced.
    """

    post_id: str
    model: str
    dimensions: int
    recipe: str
    text_sha256: str


@dataclass(frozen=True, slots=True)
class ContentEmbedding:
    """One record and the vector it names."""

    record: ContentRecord
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Neighbour:
    """One Content Item, the stored Content Item nearest to it, and how far.

    `distance` is cosine distance as pgvector computes it: 0 is the same direction, 1 is
    orthogonal, and above 1 is opposite. It is a distance between directions rather than a
    claim about either post, which is the only reading a threshold in ticket #24 can use.
    """

    post_id: str
    nearest: str
    distance: float


@dataclass(frozen=True, slots=True)
class VectorStore:
    """What the database says about itself, as claims about bytes rather than promises.

    The extension's version is here because `docs/pgvector.md` records it for exactly this
    reason: a hosted environment has to be comparable with the local one before anything
    else it holds is worth reading. The index's size is here because the argument for HNSW
    over IVFFlat is partly about memory, and an argument about memory that prints no figure
    is an argument a reader cannot check.
    """

    host: str
    port: int
    database: str
    user: str
    server_version: str
    extension_version: str
    table: str
    index: str
    index_method: str
    index_bytes: int
    rows: int


@dataclass(frozen=True, slots=True)
class EmbedFacts:
    """What reading the Corpus and the database establishes, from the rows and the bytes."""

    accounts: int
    computed: int
    corpus_path: str
    corpus_sha256: str
    plan_node: str
    plan_index: str | None
    posts: int
    removed: int
    reused: int
    stored: int
    shortest_words: int
    median_words: int


@dataclass(frozen=True, slots=True)
class Embedded:
    """Everything one run establishes, so the table and the file cannot disagree.

    `neighbours` holds one entry per Content Item rather than a demonstration over a chosen
    few: the ticket asks that a similarity query return the nearest items, and a query shown
    on the one post that reads well is a claim rather than a measurement.
    """

    facts: EmbedFacts
    records: tuple[ContentRecord, ...]
    store: VectorStore
    neighbours: tuple[Neighbour, ...]


def recipe_digest() -> str:
    """The digest of what the model does, beside its name and its dimensionality.

    Three facts rather than one, because a reader comparing two stores needs all three and
    any one of them alone will not tell them whether the vectors can be compared.
    """
    return hashlib.sha256(_RECIPE.encode("utf-8")).hexdigest()


def embedded_text(item: CorpusItem) -> str:
    """The text of one Content Item that is embedded, and the whole of it.

    The three fields, one per line, so a reader can print this and see exactly what went in.
    A reader reproducing a vector hashes this string; the digest is beside every stored
    vector for that purpose.

    The account is not here and cannot get here: `CorpusItem` is the whole vocabulary this
    pipeline is given (ADR-0008), and three of its seven fields are the post's own words. A
    refactor that joined all seven is caught by the test that renames every account in the
    Corpus and requires the vectors to come back identical.
    """
    return "\n".join([item.title, item.body, *item.links])


def embed(text: str) -> tuple[float, ...]:
    """One vector, `DIMENSIONS` long, of unit length.

    The hashing trick: each feature lands in one bucket of a fixed width and is added to it
    with a sign taken from the same digest. The sign is not decoration — without it two
    unrelated features that collide make each vector noisier, and with it a collision tends
    to cancel. `blake2b` rather than Python's built-in `hash()` because that one is salted
    per process, and a model producing a different Corpus on every run could not have its
    output committed and held to its bytes.
    """
    counts: Counter[str] = Counter()
    for sequence in _sequences(text):
        counts.update(_features(sequence))

    if not counts:
        raise ValueError(
            "a Content Item with no text to embed has no direction, and a cosine distance "
            f"against a zero vector is not a number: {text!r}"
        )

    vector = [0.0] * DIMENSIONS
    for feature, count in counts.items():
        bucket, sign = _place(feature)
        vector[bucket] += sign * (1.0 + math.log(count))
    return _unit(vector)


def _sequences(text: str) -> tuple[list[str], ...]:
    """The tokens of each line of the embedded text, kept apart.

    One sequence per line rather than one run of tokens, because a pair of words is evidence
    and a pair drawn across a boundary is an artefact: joining the title to the body would
    let the last word of a title pair with the first word of its body, so two posts with
    nothing in common could share a feature neither of them contains.
    """
    return tuple(
        tokens
        for tokens in (_TOKEN.findall(line.lower()) for line in text.split("\n"))
        if tokens
    )


def _features(tokens: Sequence[str]) -> Iterable[str]:
    """Every feature of one sequence: its tokens, and the pairs inside it."""
    for order in _NGRAM_ORDERS:
        for start in range(len(tokens) - order + 1):
            yield "\x1f".join(tokens[start : start + order])


def _place(feature: str) -> tuple[int, float]:
    """Which bucket a feature lands in, and which way it pushes.

    `blake2b` is used with an explicit digest size rather than taken to its default, because
    the bucket is the low bits and the sign is the top bit and the two have to come from
    different parts of one fixed-width digest rather than from two unrelated hashes.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % DIMENSIONS, -1.0 if value >> 63 else 1.0


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    """Scaled to unit length, which is what makes cosine distance a distance between
    directions.

    pgvector normalises internally, so this is not what makes the arithmetic right. It is
    what makes the stored vector's length carry no information a reader could mistake for
    something, and it is what makes a similarity between two posts independent of how long
    either of them is.
    """
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0.0:
        # Every feature cancelled with its opposite: astronomically unlikely, and not worth
        # storing. A reader handed a zero vector would have no way of telling.
        raise ValueError(
            "every feature of this text cancelled itself out, and there is no direction "
            "left to store"
        )
    return tuple(value / length for value in vector)


def record_for(item: CorpusItem) -> ContentRecord:
    """One Content Item's record: the post, the model, the width, and the digest.

    Computed without embedding anything, which is what lets a rerun decide what to reuse
    before it does any work: hashing a post is free next to embedding one.
    """
    return ContentRecord(
        post_id=item.post_id,
        model=MODEL_NAME,
        dimensions=DIMENSIONS,
        recipe=recipe_digest(),
        text_sha256=digest_of(embedded_text(item)),
    )


def digest_of(text: str) -> str:
    """The SHA-256 of one embedded text, as every stored vector carries beside it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_embeddings(items: Iterable[CorpusItem]) -> tuple[ContentEmbedding, ...]:
    """Every Content Item's vector, computed fresh.

    What a run does for the Content Items it holds nothing for, and what a test reads to
    check the model rather than the store. It embeds all of them, so a caller deciding what
    to reuse must not use this: `store_embeddings` is what compares against what the
    database already holds and embeds only the difference.
    """
    return tuple(
        ContentEmbedding(record=record_for(item), embedding=embed(embedded_text(item)))
        for item in items
    )


def write_content_records(path: Path, records: Sequence[ContentRecord]) -> None:
    """The published record: which post, which model, how wide, under which recipe, and which
    text.

    Five fields and no vector, for the reason the module docstring gives. Every one of them is
    a fact a later run needs before it can decide not to recompute, and not one of them is a
    fact a reader has to take on trust.
    """

    def objects() -> Iterable[JsonObject]:
        for record in records:
            yield {
                "post_id": record.post_id,
                "model": record.model,
                "dimensions": record.dimensions,
                "recipe": record.recipe,
                "text_sha256": record.text_sha256,
            }

    write_lines(path, objects())


def read_content_records(path: Path) -> tuple[ContentRecord, ...]:
    """The published record, read back, refusing what it cannot account for.

    Exists for the reason `read_corpus` and `read_policy_scores` exist rather than because
    a command reads the file today: this is a file the project publishes, so the day a field
    appears in it that nothing checks, nothing will notice. The field that must never appear
    is a Planted Campaign's identifier, and a reader that skipped what it did not recognise
    would skip that without saying so (ADR-0008).

    Checked row by row and then as a whole. Two rows for one post would let a later run
    reuse whichever it read last and leave the other invisible, and rows disagreeing about
    the model, the width or the recipe would hold distances between two kinds of number with
    nothing in the file to notice it.
    """
    records = tuple(_record(path, number, text) for number, text in read_rows(path))
    refuse_repeated(
        path.as_posix(),
        (record.post_id for record in records),
        "a later run would reuse whichever row it read last and leave the other invisible",
    )
    models = {record.model for record in records}
    if len(models) > 1:
        raise ValueError(
            f"{path.as_posix()} holds {len(models)} models {sorted(models)}, and a vector "
            "from one of them has no distance to a vector from another"
        )
    widths = {record.dimensions for record in records}
    if len(widths) > 1:
        raise ValueError(
            f"{path.as_posix()} holds {len(widths)} widths {sorted(widths)}, and a vector "
            f"of {min(widths)} values is not a vector of {max(widths)}"
        )
    recipes = {record.recipe for record in records}
    if len(recipes) > 1:
        raise ValueError(
            f"{path.as_posix()} holds {len(recipes)} recipes {sorted(recipes)}, and one of "
            "them is not the rule the other rows were computed under"
        )
    return records


def _record(path: Path, number: int, text: str) -> ContentRecord:
    """One row, checked field by field against the vocabulary this project publishes."""
    where = f"{path.as_posix()}:{number}"
    record = read_object(where, text)
    vocabulary = tuple(field.name for field in fields(ContentRecord))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds {sorted(record)}, which is not the Content Embedding vocabulary "
            f"{sorted(vocabulary)}"
        )
    width = record["dimensions"]
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError(
            f"{where} has dimensions={width!r}, and a width is a whole number above zero"
        )
    digest = read_text(where, record, "text_sha256")
    if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ValueError(
            f"{where} has text_sha256={digest!r}, and the digest beside a stored vector is a "
            "SHA-256 of the text that vector was computed from"
        )
    recipe = read_text(where, record, "recipe")
    if len(recipe) != 64 or set(recipe) - set("0123456789abcdef"):
        raise ValueError(
            f"{where} has recipe={recipe!r}, and the recipe beside a stored vector is the "
            "digest of the rule it was computed under"
        )
    return ContentRecord(
        post_id=read_text(where, record, "post_id"),
        model=read_text(where, record, "model"),
        dimensions=width,
        recipe=recipe,
        text_sha256=digest,
    )


# --- the database ----------------------------------------------------------------


def connection_string() -> str:
    """Where the connection comes from, or a refusal naming every place it could.

    Two sources and no argument. `RFI_DATABASE_URL` if it is set, otherwise the standard
    `PG*` variables libpq reads on its own. There is deliberately no `--dsn`, because an
    argument is a place a password ends up in a shell history and in a CI log, and this
    repository keeps credentials in the environment.

    Refused rather than defaulted when neither is there: libpq would otherwise pick a
    database named after the operating-system user on the local socket, and writing an
    embeddings table into whatever that happens to be is not a decision this command should
    make on its own.
    """
    url = os.environ.get(_DATABASE_URL)
    if url:
        return url
    if any(name in os.environ for name in _ENVIRONMENT):
        return ""
    raise ValueError(
        f"no database is configured. Set {_DATABASE_URL} to a connection string, or the "
        + ", ".join(_ENVIRONMENT)
        + " variables libpq reads; this command takes no --dsn, because an argument is a "
        "place a password ends up in a shell history"
    )


def open_store(dsn: str) -> Database:
    """The connection, refused by name rather than failing somewhere inside the run.

    `autocommit` because every statement here is its own transaction: this command creates a
    table, writes rows, deletes the rows the Corpus no longer holds, and reads figures back,
    and there is no point in the run at which all of that should be rolled back because the
    last figure could not be printed.

    Row dictionaries rather than tuples, because every field of every row is named at the
    point it is read. A positional read of a five-column query is a read that breaks the
    moment the query gains a column, and this project publishes figures it asks a reader to
    check.
    """
    try:
        return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
    except psycopg.Error as refusal:
        raise ValueError(f"cannot connect to that database: {refusal}") from refusal


def ensure_table(connection: Database, table: str) -> None:
    """The column, the primary key, and the refusal that makes the width a fact.

    The vector column's width is part of the table's definition, so a table built for a
    different width cannot be written to at all. That is why the width is a fact rather than a
    setting: it is in the schema, and `CREATE TABLE IF NOT EXISTS` will not widen it.

    A table already holding vectors from another model, another width or another recipe is
    refused rather than rewritten. A vector from one model has no distance to a vector from
    another, so a table holding both holds nothing a query can answer; rewriting would destroy
    the old vectors and make the two runs incomparable after the fact, and refusing keeps
    both. `--table` is how a reader chooses where a second model goes.

    `CREATE EXTENSION` is not here either. It is per database, it needs a role permitted to
    create one, and it is a persistent change to the database rather than to a row in it —
    `docs/pgvector.md` says so and this command holds to it. A database without the extension
    is refused with the statement to run.
    """
    if _one(connection, _EXTENSION_QUERY, "extversion") is None:
        raise ValueError(
            "the vector extension is not activated in this database. Run "
            "`CREATE EXTENSION vector;` as a role permitted to create it, then run this "
            "again; docs/pgvector.md holds the activation step"
        )

    connection.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                post_id text PRIMARY KEY,
                model text NOT NULL,
                dimensions integer NOT NULL,
                recipe char(64) NOT NULL,
                text_sha256 char(64) NOT NULL,
                embedding vector({dimensions}) NOT NULL
            )
            """
        ).format(table=sql.Identifier(table), dimensions=sql.Literal(DIMENSIONS))
    )

    stored = connection.execute(
        sql.SQL("SELECT DISTINCT model, dimensions, recipe FROM {table}").format(
            table=sql.Identifier(table)
        )
    ).fetchall()
    disagreeing = sorted(
        {
            (_text(row, "model"), _whole(row, "dimensions"), _text(row, "recipe"))
            for row in stored
        }
        - {(MODEL_NAME, DIMENSIONS, recipe_digest())}
    )
    if disagreeing:
        raise ValueError(
            f"{table} holds vectors from another model: {disagreeing}, and this run writes "
            f"{MODEL_NAME} of {DIMENSIONS} dimensions under recipe "
            f"{recipe_digest()[:12]}. A vector from one model has no distance to a vector "
            f"from another, so the table is refused rather than rewritten: point --table at "
            f"another name to keep both, or drop this one"
        )


def ensure_index(connection: Database, table: str) -> None:
    """Build the HNSW index if it is not there, then check that it is one.

    `CREATE INDEX IF NOT EXISTS` is satisfied by an index of the same *name*, so a table
    somebody has put a B-tree on would silently keep it, every figure printed below would
    describe an index that cannot answer a cosine query at all, and no error would ever be
    raised. So the method is read back out of the catalogue and refused on, rather than created
    and assumed. Why HNSW rather than IVFFlat is ADR-0024.
    """
    connection.execute(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {index} ON {table} USING hnsw (embedding vector_cosine_ops)"
        ).format(index=sql.Identifier(index_name(table)), table=sql.Identifier(table))
    )
    method = _one(connection, _index_query(), "method", table, index_name(table))
    if method != "hnsw":
        found = "no index of that name" if method is None else f"a {method} index"
        raise ValueError(
            f"{index_name(table)} is {found}, and a cosine query cannot use it. HNSW is the "
            "index this project chose and the reasoning is in ADR-0024; drop the index "
            "under that name and run this again"
        )


_EXTENSION_QUERY = "SELECT extversion FROM pg_extension WHERE extname = 'vector'"


def index_name(table: str) -> str:
    """The index's name, derived from the table's rather than configured beside it.

    One place, so the name the command creates and the name it checks cannot drift — which
    is the failure `CREATE INDEX IF NOT EXISTS` invites, since it is satisfied by a name
    rather than by a definition.
    """
    return f"{table}_hnsw"


def _index_query() -> sql.SQL:
    """The catalogue read behind both index figures: its access method and its size.

    One query written once, because the method is what `prepare` refuses on and the size is
    what the output publishes, and two hand-written joins over `pg_class` would be two chances
    for them to disagree about which index they are describing. The table and the index name
    are parameters rather than interpolated identifiers because this statement formats
    nothing and so has nothing to quote.
    """
    return sql.SQL(
        """
        SELECT am.amname AS method, pg_relation_size(c.oid) AS bytes
          FROM pg_class c
          JOIN pg_am am ON am.oid = c.relam
          JOIN pg_index i ON i.indexrelid = c.oid
         WHERE i.indrelid = to_regclass(%s) AND c.relname = %s
        """
    )


@dataclass(frozen=True, slots=True)
class Stored:
    """What one run put in the table, and what it did not have to.

    A named type rather than four integers travelling together, because `computed` and
    `removed` are both counts and a caller that took them in the wrong order would type-check
    and publish one figure as another. Every field is printed on every run: "computed once and
    reused" is a claim about work done rather than about a result, and a claim about work done
    that prints nothing is a claim nobody can check.
    """

    records: tuple[ContentRecord, ...]
    computed: int
    reused: int
    removed: int


def store_embeddings(
    connection: Database, table: str, items: Sequence[CorpusItem]
) -> Stored:
    """Bring the table into line with the Corpus, embedding only what is not already there.

    Reuse is decided against the digest stored beside each vector, before anything is
    embedded, so an unchanged Corpus costs a hash per post rather than a vector.

    Rows for Content Items the Corpus no longer holds are removed rather than left behind.
    The table is a projection of the Corpus, and a row left behind would come back from every
    similarity query as though it were a post somebody could go and read.

    A Corpus naming one post twice is refused here rather than by `read_corpus`, because a
    duplicate is only a problem in this path: it would be embedded twice, written twice, and
    counted once. The check is over the items rather than over `known`, which is a dictionary
    and therefore cannot repeat anything — which is exactly why passing it would have made
    the refusal unreachable.
    """
    known = {item.post_id: item for item in items}
    refuse_repeated(
        f"{len(items)} Content Items",
        (item.post_id for item in items),
        "the store would embed one of them twice, the second write would overwrite the first, "
        "and the removed count below would not be the number of rows that went",
    )

    records = tuple(record_for(item) for item in items)
    held = {
        _text(row, "post_id"): _text(row, "text_sha256")
        for row in connection.execute(
            sql.SQL("SELECT post_id, text_sha256 FROM {table}").format(table=sql.Identifier(table))
        ).fetchall()
    }

    reused = {record.post_id for record in records if held.get(record.post_id) == record.text_sha256}
    computed = [
        ContentEmbedding(
            record=record, embedding=embed(embedded_text(known[record.post_id]))
        )
        for record in records
        if record.post_id not in reused
    ]

    with connection.cursor() as cursor:
        if computed:
            cursor.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table}
                        (post_id, model, dimensions, recipe, text_sha256, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (post_id) DO UPDATE SET
                        model = EXCLUDED.model,
                        dimensions = EXCLUDED.dimensions,
                        recipe = EXCLUDED.recipe,
                        text_sha256 = EXCLUDED.text_sha256,
                        embedding = EXCLUDED.embedding
                    """
                ).format(table=sql.Identifier(table)),
                [
                    (
                        found.record.post_id,
                        found.record.model,
                        found.record.dimensions,
                        found.record.recipe,
                        found.record.text_sha256,
                        vector_literal(found.embedding),
                    )
                    for found in computed
                ],
            )
        removed = cursor.execute(
            sql.SQL("DELETE FROM {table} WHERE NOT (post_id = ANY(%s))").format(
                table=sql.Identifier(table)
            ),
            (list(known),),
        ).rowcount

    return Stored(
        records=records,
        computed=len(computed),
        reused=len(reused),
        removed=max(removed, 0),
    )


def vector_literal(embedding: Sequence[float]) -> str:
    """A vector as the text a `vector` column is written from.

    `repr` on a float is the shortest text that reads back as that same float, so what
    reaches the column is the number the model produced rather than a rounded version of it.
    The column then stores it as a 32-bit float, which is the width pgvector keeps and the
    reason the distances printed are the ones the database computed.
    """
    return "[" + ",".join(repr(value) for value in embedding) + "]"


def _nearest_query(table: str) -> sql.Composed:
    """The similarity query, written once so the query and the explained query cannot
    differ.

    The table's own name is interpolated as an identifier because a table name is not a
    value; the vector, the post asked about and the limit are parameters, so nothing a
    Corpus contains is ever formatted into a statement.
    """
    return sql.SQL(
        """
        SELECT post_id, embedding <=> %s::vector AS distance
          FROM {table}
         WHERE post_id <> %s
         ORDER BY embedding <=> %s::vector
         LIMIT %s
        """
    ).format(table=sql.Identifier(table))


def nearest(connection: Database, table: str, post_id: str, limit: int = 1) -> tuple[Neighbour, ...]:
    """The stored Content Items nearest to one, by cosine distance.

    The query vector is read out of the table rather than handed in, so the question cannot
    be answered about a vector nobody has stored: what comes back is the distance from the
    post as the database holds it, which is the post this run wrote or reused.

    The post asked about is excluded from its own neighbours. A nearest-neighbour list that
    led with the post itself would be a list of one at distance 0 for every entry, and a
    reader checking whether the query works could not tell that from a query returning
    nothing.
    """
    literal = _stored_vector(connection, table, post_id)
    rows = connection.execute(
        _nearest_query(table), (literal, post_id, literal, limit)
    ).fetchall()
    return tuple(
        Neighbour(post_id=post_id, nearest=_text(row, "post_id"), distance=_measure(row, "distance"))
        for row in rows
    )


def _stored_vector(connection: Database, table: str, post_id: str) -> str:
    """The vector the table holds for one post, as the text pgvector reads a vector from.

    Refused rather than returned as nothing when the post is not stored, because the two
    failures a caller cannot tell apart are a query against a post with no vector and a query
    against a table that was never filled.
    """
    held = _one(
        connection,
        sql.SQL("SELECT embedding::text AS literal FROM {table} WHERE post_id = %s").format(
            table=sql.Identifier(table)
        ),
        "literal",
        post_id,
    )
    if not isinstance(held, str):
        raise ValueError(
            f"{post_id} has no vector stored in {table}, so there is nothing to ask what is "
            f"near it. Run `rfi content-embeddings` over a Corpus holding {post_id} first"
        )
    return held


def nearest_plan(connection: Database, table: str, post_id: str, limit: int = 1) -> tuple[str, str | None]:
    """What the planner actually chose for a similarity query: the node, and the index.

    Returned and printed rather than asserted, because at a corpus size this small it will
    not choose the index: over a few dozen rows reading the table once is cheaper than any
    index, and the honest thing to publish is which of the two it did. A run claiming its
    index was at work over thirty-odd rows would be claiming something no `EXPLAIN` agrees
    with, and the next Corpus to grow past the planner's threshold would inherit a claim
    nobody had checked.

    The plan is taken for the same statement `nearest` runs, so the figure is about the query
    this project asks rather than about a simplified one that might be planned differently.
    """
    literal = _stored_vector(connection, table, post_id)
    explained = _mapping(
        connection.execute(
            "EXPLAIN (FORMAT JSON) " + _nearest_query(table).as_string(connection),
            (literal, post_id, literal, limit),
        ).fetchone()
    )
    roots = _list(explained.get("QUERY PLAN"))
    if not roots:
        raise ValueError(f"Postgres explained nothing about the query for {post_id}")
    return _plan_of(_mapping(roots[0].get("Plan")))


def _plan_of(plan: Mapping[str, object]) -> tuple[str, str | None]:
    """The node that decided how the table was read, and the index it reached for if any.

    The search is recursive and it skips the nodes that do not decide anything: the root of a
    `LIMIT` query is a `Limit`, and reporting that would tell a reader the planner answered
    with a `Limit` rather than with the scan underneath. The node named here is the first scan
    in the plan, which is the one `Seq Scan` or `Index Scan` says.
    """
    return _scan_within(plan) or (str(plan.get("Node Type", "")), None)


def _scan_within(plan: Mapping[str, object]) -> tuple[str, str | None] | None:
    """The first scan in a plan tree, or nothing when it holds none."""
    node = str(plan.get("Node Type", ""))
    if node.endswith("Scan"):
        name = plan.get("Index Name")
        return node, None if name is None else str(name)
    for child in _list(plan.get("Plans")):
        found = _scan_within(child)
        if found is not None:
            return found
    return None


def store_facts(connection: Database, table: str) -> VectorStore:
    """What the database says about itself, read rather than assumed.

    The password is not among these fields and never is: `Connection.info` holds it and this
    function does not reach for it, which is the whole of how a run over a credentialed
    database can print its own connection without printing the credential.
    """
    info = connection.info
    server = _mapping(
        connection.execute(
            """
            SELECT current_setting('server_version') AS server_version,
                   (SELECT extversion FROM pg_extension WHERE extname = 'vector') AS extension
            """
        ).fetchone()
    )
    index = _mapping(
        connection.execute(_index_query(), (table, index_name(table))).fetchone()
    )
    rows = connection.execute(
        sql.SQL("SELECT count(*) AS held FROM {table}").format(table=sql.Identifier(table))
    ).fetchone()

    return VectorStore(
        host=info.host or "local socket",
        port=info.port or 0,
        database=info.dbname or "unknown",
        user=info.user or "unknown",
        server_version=str(server["server_version"]),
        extension_version=str(server["extension"]),
        table=table,
        index=index_name(table),
        index_method=str(index["method"]),
        index_bytes=_whole(index, "bytes"),
        rows=_whole(_mapping(rows), "held"),
    )


def embed_corpus(corpus_path: Path, table: str) -> Embedded:
    """Read the Corpus, store every Content Item's vector, and ask what is near what.

    One call, for the same reason the grouping and the scoring are one call: the records
    published, the table written, the figures printed and the neighbours looked up are four
    views of one pass over one Corpus, and a caller that read them separately could end up
    printing one run's figures over another's vectors.

    The connection is opened here and handed to every step, so there is one place the
    database is opened and one place it is refused, and a caller cannot pair a Corpus with a
    table another run filled.
    """
    items = read_corpus(corpus_path)
    connection = open_store(connection_string())
    try:
        ensure_table(connection, table)
        ensure_index(connection, table)
        stored = store_embeddings(connection, table, items)
        records = stored.records
        return Embedded(
            facts=_facts(
                corpus_path,
                items,
                stored,
                nearest_plan(connection, table, records[0].post_id)
                if records
                else ("nothing to explain", None),
            ),
            records=records,
            store=store_facts(connection, table),
            neighbours=tuple(
                neighbour
                for record in records
                for neighbour in nearest(connection, table, record.post_id)
            ),
        )
    finally:
        connection.close()


def _facts(
    corpus_path: Path,
    items: Sequence[CorpusItem],
    stored: Stored,
    plan: tuple[str, str | None],
) -> EmbedFacts:
    """Every figure about the Corpus this run read, from the items, the store, and the bytes."""
    counts = sorted(len(_TOKEN.findall(embedded_text(item).lower())) for item in items)
    return EmbedFacts(
        accounts=len({item.account for item in items}),
        computed=stored.computed,
        corpus_path=corpus_path.as_posix(),
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        plan_node=plan[0],
        plan_index=plan[1],
        posts=len(items),
        removed=stored.removed,
        reused=stored.reused,
        stored=len(stored.records),
        # How much text each vector is built out of, which is the thing a distance has to be
        # read against: the same cosine between two paragraphs and the same cosine between two
        # one-line posts are very different thicknesses of claim, and a reader picking a
        # threshold has to know which of the two they are looking at. Counted from the Corpus on
        # every run rather than written down, because the Corpus moves and a number written
        # here would not. Zero for an empty Corpus, which has no words to count.
        shortest_words=counts[0] if counts else 0,
        median_words=counts[len(counts) // 2] if counts else 0,
    )


def render_table(embedded: Embedded) -> str:
    """The console output: what was read, what was stored, and what is near what.

    ASCII only, so it prints the same way on a console that cannot encode anything else and
    the same way when it is redirected, which is what lets it be pasted into an issue or
    diffed between runs.
    """
    sections = (
        f"{_HEADING}\n\n{_SUBHEADING}",
        _figures(embedded),
        _recipe(),
        _neighbours(embedded),
        _footer(embedded),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _figures(embedded: Embedded) -> str:
    """What was read, and what came of it. One figure per line, labelled."""
    facts = embedded.facts
    store = embedded.store
    figures = (
        (
            "corpus",
            f"{facts.corpus_path} ({_count(facts.posts, 'post')} across "
            f"{_count(facts.accounts, 'account')})",
        ),
        ("sha256", facts.corpus_sha256),
        ("model", f"{MODEL_NAME}, {DIMENSIONS} dimensions"),
        ("recipe", recipe_digest()),
        (
            "database",
            f"{store.host}:{store.port}/{store.database} as {store.user} "
            f"(PostgreSQL {store.server_version}, pgvector {store.extension_version})",
        ),
        ("table", f"{store.table} ({_count(store.rows, 'row')})"),
        ("index", store.index),
        (
            "indexed",
            f"{store.index_method} on embedding vector_cosine_ops, "
            f"{_bytes(store.index_bytes)} over {_count(store.rows, 'row')}",
        ),
        (
            "stored",
            f"{_count(facts.stored, 'Content Item')}: {facts.computed} computed, "
            f"{facts.reused} reused, {facts.removed} removed",
        ),
        (
            "words",
            f"the shortest Content Item holds {_count(facts.shortest_words, 'word')}, the "
            f"median {facts.median_words}",
        ),
        ("plan", _plan_line(embedded)),
    )
    width = max(len(name) for name, _ in figures)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in figures)


def _plan_line(embedded: Embedded) -> str:
    """Which node the planner chose for a similarity query, and which index it reached for.

    Short, because a figure line is not wrapped: the reading of it is in the footer, where a
    reader looks for what a figure means rather than scrolling for it.
    """
    facts = embedded.facts
    if facts.plan_index is not None:
        return f"Index Scan using {facts.plan_index}, which is the index this run built"
    return (
        f"{facts.plan_node}, not the HNSW index; see the footer for why that is the right "
        "answer here"
    )


def _recipe() -> str:
    """The model, and what it reads — published, because a rule a reader cannot see is a rule
    they have to take on trust.

    The recipe digest is a figure above rather than part of this block, so a reader comparing
    two stores has all three facts in front of them in one place: which model, how wide, and
    whether the two runs computed it the same way.
    """
    label = "  what it is  "
    paragraph = [
        f"{label if line == 0 else ' ' * len(label)}{body}"
        for line, body in enumerate(wrap(_RECIPE, indent=0, width=_WIDTH - len(label)))
    ]
    return "\n".join(
        [
            "model  what the model above reads, and nothing else it could be reading",
            "  fields read    the post's own title, its own body, and its links",
            "  features       each word, and each pair of adjacent words within one of "
            "those three",
            "  weighting      each feature's count damped as 1 + ln(count), then the "
            "vector scaled to unit length",
            *paragraph,
        ]
    )


def _neighbours(embedded: Embedded) -> str:
    """One row per Content Item: the nearest other one, and how far.

    Every post rather than a demonstration over a chosen few, because the ticket asks that a
    similarity query return the nearest items and a query shown on the one post that reads
    well is a claim. The distance is printed to four decimal places because the column
    stores 32-bit floats: the number here is the one the database computed from what it
    holds, rather than from the widest numbers the model can produce.
    """
    if not embedded.neighbours:
        return "No Content Item to compare: the Corpus holds fewer than two posts."

    headings = ("post", "nearest", "distance")
    rows = [
        (neighbour.post_id, neighbour.nearest, f"{neighbour.distance:.4f}")
        for neighbour in embedded.neighbours
    ]
    widths = [max(len(cell) for cell in column) for column in zip(headings, *rows, strict=True)]
    return "\n".join(
        [
            f"nearest  {_count(len(rows), 'Content Item')}, each to its closest other one, "
            "by cosine distance",
            _row(headings, widths),
            *(_row(row, widths) for row in rows),
        ]
    )


def _footer(embedded: Embedded) -> str:
    """What the vectors can and cannot say, and what is deliberately not stored with them.

    The limits are the ones a reader would otherwise have to guess at: the model reads words
    rather than meaning, a distance is a statement about shared vocabulary and about nothing
    else, the links are part of what is embedded so a shared host moves two vectors towards
    each other on its own, the column is narrower than the model, the planner is not using the
    index, and nothing here groups or judges anything.

    Wrapped rather than hand-broken, because three of these paragraphs carry figures read out
    of the run and a hand-broken line is where a number ends up stranded at the end of a
    sentence it does not belong to.
    """
    facts = embedded.facts
    paragraphs = (
        """
        The model is lexical. It reads shared words and the pairs of adjacent words inside one
        field, and it does not read meaning: two posts making the same pitch in different words
        can come out far apart, and a post and a loose paraphrase of it can come out close
        without saying the same thing. Ticket #24 rests on this and inherits the limit, which
        is the whole of why it may corroborate a grouping and never establish one.
        """,
        f"""
        What a distance between two of these vectors is a statement about is shared vocabulary,
        and about nothing else. It does not say the two posts mean the same thing. The words
        figure above is how much text a reader is weighing that statement against: this
        Corpus's shortest Content Item is {_count(facts.shortest_words, 'word')} and its
        median is {facts.median_words}, so the same cosine over two paragraphs is a much
        thicker claim than the same cosine over two titles.
        """,
        """
        Links are part of what is embedded, so a similarity over these vectors is partly a
        similarity of the hosts two posts link. Two posts sharing nothing but a link shortener
        share the words of that host whichever pitch they are making, and ADR-0005's own edge
        is therefore reached again here through the text. That is a reason to read a small
        distance as weak evidence rather than as none, and it is not independent of the edge
        the grouping already has.
        """,
        """
        A cosine distance is a distance between directions and not a claim about either post: 0
        is the same direction, 1 is orthogonal, and above 1 is opposite. The column stores
        32-bit floats, so the distance printed beside a pair is the one the database computed
        from what it holds rather than from the widest numbers this model can produce.
        """,
        _plan_paragraph(embedded),
        """
        Nothing here groups accounts, corroborates a grouping, or decides what anything is,
        and nothing reads the Planted Campaign membership or the Nuisance Structure manifest.
        No account's age, karma, posting rate, or activity change is read on this path
        (ADR-0007). A similarity published as evidence of a shared operator would be a claim
        about meaning that a bag of hashed words cannot support.
        """,
        """
        What is published is the record rather than the vectors.
        data/embeddings/content-embeddings.jsonl says which post, which model, how wide, under
        which recipe, and the SHA-256 of the text each stored vector was computed from. That is
        what makes the reuse rule above checkable, because a reader can recompute every digest
        from the Corpus and see which text each vector belongs to, and the vectors themselves
        stay in the database, which is where the ticket asked for them.
        """,
    )
    return "\n\n".join(_paragraph(text) for text in paragraphs)


def _plan_paragraph(embedded: Embedded) -> str:
    """What the planner chose, and what that is a fact about.

    Said here rather than in the figures block because a figure line is not wrapped and this
    is a paragraph. The point it makes is that the index is not being exercised, that this is
    the right answer at this size, and that the figure above would change on its own when the
    Corpus grows — which is what makes publishing it rather than assuming it worth doing.
    """
    facts = embedded.facts
    store = embedded.store
    if facts.plan_index is not None:
        return f"""
            The planner chose an Index Scan over {store.index} for the similarity query above,
            so the index is doing what it was built for.
        """
    return f"""
        The planner chose a {facts.plan_node} for the similarity query above, and over
        {_count(store.rows, 'row')} that is the right answer: reading the table once is cheaper
        than any index could be, and Postgres knows it. The HNSW index is in place at
        {_bytes(store.index_bytes)} and it is not what answered this query, which is a fact
        about the size of this Corpus rather than about the index. A Corpus large enough for
        the planner to prefer the index would flip the line above without a line of code
        changing, and a run that had claimed its index was at work here would be claiming
        something no EXPLAIN agrees with (ADR-0024).
    """


def _paragraph(text: str) -> str:
    """One paragraph, wrapped to the console's width, whatever it was written as."""
    return "\n".join(wrap(" ".join(text.split()), indent=0, width=_WIDTH))


def _count(number: int, noun: str) -> str:
    """One count, agreeing with its noun. Every figure here is small, and a reader seeing
    "1 posts" stops to wonder whether the figure is right."""
    return f"{number} {noun if number == 1 else f"{noun}s"}"


def _bytes(size: int) -> str:
    """A size a reader can hold in their head.

    Deliberately coarse: the question the index's size answers is whether an index over a
    handful of rows is kilobytes or megabytes, and a byte count would answer the second
    question while hiding the first.
    """
    for limit, unit in ((1 << 20, "MB"), (1 << 10, "kB")):
        if size >= limit:
            return f"{size / limit:.1f} {unit}"
    return f"{size} bytes"


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """One line of a table. A cell of nothing but digits is a count, and counts are
    right-aligned so the digits line up down the column; the rest is text of no fixed width
    and reads better against the left edge."""
    return "  ".join(
        cell.rjust(width) if cell.isdigit() else cell.ljust(width)
        for cell, width in zip(cells, widths, strict=True)
    )


def _mapping(row: object) -> Mapping[str, object]:
    """One row, as something this module can read fields off."""
    if row is None:
        return {}
    if not isinstance(row, Mapping):
        raise ValueError(f"the database returned a row that is not a mapping: {row!r}")
    return row


def _list(value: object) -> tuple[Mapping[str, object], ...]:
    """One field of a row, as the rows it holds."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(entry, dict) for entry in value):
        raise ValueError(f"the database returned {value!r}, which is not a list of rows")
    return tuple(value)


def _one(
    connection: Database, statement: sql.SQL | sql.Composed | str, field: str, *args: object
) -> object:
    """One scalar out of one row, or `None` when the query found no row at all.

    `None` is the answer to "is this here" and never to "what is it": every caller either
    refuses on `None` or refuses when it is not the expected type, so a query that returned
    the wrong kind of value could not pass for one that returned nothing.
    """
    row = connection.execute(statement, args or None).fetchone()
    if row is None:
        return None
    return _mapping(row).get(field)


def _text(row: Mapping[str, object], field: str) -> str:
    """One text field of a row the database returned."""
    value = _mapping(row).get(field)
    if not isinstance(value, str):
        raise ValueError(f"the database returned {field}={value!r}, which is not text")
    return value


def _whole(row: Mapping[str, object], field: str) -> int:
    """One whole-number field of a row the database returned."""
    value = _mapping(row).get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"the database returned {field}={value!r}, which is not a whole number")
    return value


def _measure(row: Mapping[str, object], field: str) -> float:
    """One number out of a row the database returned."""
    value = _mapping(row).get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"the database returned {field}={value!r}, which is not a number")
    return float(value)