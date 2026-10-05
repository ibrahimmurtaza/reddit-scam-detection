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
- `pgvector` `0.8.6` is built from source into that install. See `docs/pgvector.md` for
  prerequisites, the build script, and how to verify it.
- Build tooling that the project needs and that is not preinstalled: `scripts/install-pgvector.ps1` (elevated).
- `rfi content-embeddings` is the only command that needs a database. It reads
  `RFI_DATABASE_URL`, or `PGHOST`/`PGPORT`/`PGUSER`/`PGDATABASE`/`PGPASSWORD`, and takes
  no `--dsn`. It needs `CREATE EXTENSION vector` to have been run in the database first.
- `tests/test_embedding_store.py` skips, with the reason, when no `PG*` variable and no
  `RFI_DATABASE_URL` is set. `tests/test_content_embeddings.py` is the offline half and
  holds the model and `data/embeddings/content-embeddings.jsonl` to their bytes.
- `psycopg` is the one runtime dependency. The embedding model is this project's own and
  needs no wheel, which is deliberate (ADR-0024).