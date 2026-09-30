# Reddit Scam Detection

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `ibrahimmurtaza/reddit-scam-detection` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Environment

- PostgreSQL 18.1 on port `5433`, credentials in the environment, never in the repo.
- `pgvector` `0.8.6` is built from source into that install. See `docs/pgvector.md` for prerequisites, the build script, and how to verify it.
- Build tooling that the project needs and that is not preinstalled: `scripts/install-pgvector.ps1` (elevated).