# Reddit Scam Detection

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `ibrahimmurtaza/reddit-scam-detection` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Commands

- `uv sync` — install the toolchain (Python 3.13, pinned; 3.14 has no prebuilt ML wheels yet).
- `uv run pytest` — the test suite.
- `uv run mypy` — typecheck (`strict`).
- `uv run rfi <command>` — the CLI, the only interface in v1.

### Environment

- PostgreSQL 18.1 on port `5433`, credentials in the environment, never in the repo.
- The password is not in an environment variable on this machine: libpq reads it from
  `%APPDATA%\postgresql\pgpass.conf` (`%APPDATA%` is `C:\Users\PC\AppData\Roaming`), which is
  the only place it is recorded. Setting `PGHOST`, `PGPORT`, `PGUSER` and `PGDATABASE` is
  therefore enough to run everything — the commands see a configured database and libpq
  supplies the password. Only `tests/test_embedding_store.py`'s credential-leak test needs
  `PGPASSWORD` in the environment, and it skips with the reason rather than passing quietly.
  Never write that file's contents into this repository, a command line, or a commit.
- `pgvector` `0.8.6` is built from source into that install. See `docs/pgvector.md` for
  prerequisites, the build script, and how to verify it.
- Build tooling that the project needs and that is not preinstalled: `scripts/install-pgvector.ps1` (elevated).
- Two commands need a database: `rfi content-embeddings` writes the vectors and
  `rfi campaign-candidates` reads them back out (`--table`, default `content_embedding`)
  to corroborate each candidate on content similarity. Both read
  `RFI_DATABASE_URL`, or `PGHOST`/`PGPORT`/`PGUSER`/`PGDATABASE`/`PGPASSWORD`, and take
  no `--dsn`. Both need `CREATE EXTENSION vector` to have been run in the database first.
- `tests/test_embedding_store.py` and `tests/test_campaign_candidates.py` skip, with the
  reason, when no `PG*` variable and no `RFI_DATABASE_URL` is set.
  `tests/test_content_embeddings.py` and `tests/test_content_similarity.py` are the offline
  halves: the first holds the model and `data/embeddings/content-embeddings.jsonl` to
  their bytes, the second the within-component similarity figures to the vectors the same
  model produces.
- `rfi confidence` needs no database: it reads the Corpus, the published Public Suffix List,
  the Planted Campaign membership (its training labels) and the published Policy Scores, and
  writes `data/model/confidences.jsonl` and `docs/confidence.md`. Nothing reads that file
  back, because the Confidence is displayed nowhere (ADR-0027, ADR-0003).
  `tests/test_confidence.py` and `tests/test_confidence_store.py` are both offline.
- `psycopg` is the one runtime dependency. The embedding model and the Confidence model are
  this project's own and need no wheel, which is deliberate (ADR-0024, ADR-0027).