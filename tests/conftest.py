"""What several test files need about the vector store, in one place.

`rfi campaign-candidates` reads the stored vectors (ADR-0026), so every test that runs
the command needs a database to read them from. Two things follow from that, and both
are here rather than in one test file: the skip with the reason, because a test that
fails because no `PG*` variable is set is a test about environment rather than about
the grouping, and the seeding, because a scratch table holding a test Corpus's vectors
is the same work wherever it is written down.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from psycopg import sql

from reddit_fraud_intelligence.corpus import read_corpus
from reddit_fraud_intelligence.embeddings import (
    connection_string,
    ensure_table,
    open_store,
    store_embeddings,
)

# The environment a run reads its connection from. Five names rather than one, because
# libpq reads four of them and the project keeps credentials in the environment rather
# than in a command line where they would end up in a shell history.
CONFIGURED = ("PGHOST", "PGPORT", "PGUSER", "PGDATABASE", "RFI_DATABASE_URL")

needs_database = pytest.mark.skipif(
    not any(name in os.environ for name in CONFIGURED),
    reason=(
        "no database is configured: set PGHOST, PGPORT, PGUSER and PGDATABASE, or "
        "RFI_DATABASE_URL, to run the tests that read the stored vectors: the "
        "grouping reads them and refuses to guess. docs/pgvector.md has the "
        "activation step this needs"
    ),
)

# A scratch table the tests write their own vectors into, so the committed table stays
# the projection of the committed Corpus. The command reads vectors from whatever
# `--table` names, so no test touches the default one.
SCRATCH_TABLE = "campaign_candidates_test"


def seed_vectors(corpus: Path, table: str = SCRATCH_TABLE) -> None:
    """Hold one Corpus's vectors where the command can read them.

    The same `store_embeddings` path `rfi content-embeddings` uses, over the named
    table: the vectors the command under test reads are ones the store holds, which is
    the seam ticket #24 is about, and the model is deterministic so they are the same
    numbers the committed table holds for the committed Corpus.
    """
    items = read_corpus(corpus)
    connection = open_store(connection_string())
    try:
        connection.execute(
            sql.SQL("DROP TABLE IF EXISTS {table}").format(table=sql.Identifier(table))
        )
        ensure_table(connection, table)
        store_embeddings(connection, table, items)
    finally:
        connection.close()
