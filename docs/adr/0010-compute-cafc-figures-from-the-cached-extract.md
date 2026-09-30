# Compute CAFC figures from the cached extract, never from a transcription

Every base rate this project publishes is computed by reading a committed, compressed copy of the CAFC extract. Nothing about CAFC's figures is written down by hand — not the report count, not the date range, not a category's share — because a transcribed number cannot be checked and cannot be re-derived. The command that reads the cache is the only producer of the provenance file, the base rates, and the report, and a test holds all three to what a fresh run produces from the same bytes.

## Considered Options

- **Transcribe CAFC's published tables into the repository.** Rejected: a copied number is indistinguishable from a typed one, and the next quarterly release would change the published figures with nothing in the repository to show why.
- **Fetch on demand and compute from the network.** Rejected: it makes every comparison depend on a third party's uptime and on whatever CAFC is serving that day, and a reviewer without the network can check nothing.
- **Commit the uncompressed CSV.** Rejected on disk grounds. It is 72 MiB and roughly doubles in `.git`; gzip takes it to 3.7 MiB, and a digest of the *uncompressed* bytes means the recorded figure is unaffected by the compression setting.
- **Derive the SHA-256 of the cached file instead of the CSV inside it.** Rejected: gzip output is not stable across compression levels, so that digest would report a change where nothing about the extract moved.

## Consequences

The comparison is reproducible offline, and a truncated or altered download fails loudly rather than quietly halving a base rate — the recorded report count and digest are checked against the cache on every test run. `fetch-cafc` refuses to replace a cache that is already there, so adopting a new CAFC release is a deliberate act rather than something a routine command does.

The cache is committed, which means CAFC's figures are frozen at the release named in `docs/cafc-base-rates.md`. Two counts are kept apart there: CAFC's annex documents 41 thematic categories and this window holds 39, so the projection in ticket #7 reconciles against the annex rather than against the extract. The extract carries no free-text field, which the report states as a measured fact — the longest value in any column is 62 characters — because that limit is what stops it being read as validation of a text classifier.
