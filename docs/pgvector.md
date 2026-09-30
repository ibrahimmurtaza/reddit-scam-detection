# pgvector

`pgvector` provides the `vector` type and the HNSW and IVFFlat access methods.
The project uses it to store sentence embeddings alongside Corpus content
(see issue #21).

## Installed version

| | |
|---|---|
| Extension version | `0.8.6` |
| Upstream tag | [`v0.8.6`](https://github.com/pgvector/pgvector/releases/tag/v0.8.6) |
| Server | PostgreSQL 18.1, x64, on port `5433` |
| Install root | `C:\Program Files\PostgreSQL\18` |
| Build date | 2026-09-30 |

Version numbers are recorded in the table above rather than hard-coded
anywhere, so a local and a hosted environment can be compared. If a hosted
database reports a different `extversion`, the comparison starts here.

## Why it was built from source

PostgreSQL does not ship `pgvector` for Windows, and EDB's installer does not
bundle it. There is no supported package manager path on this platform, so the
extension is compiled against the installed server's headers.

## Prerequisites

These were not present on the machine and are what made the build possible:

| Requirement | Detail |
|---|---|
| Visual Studio Build Tools 2026 | `18.6.11806.211`, at `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools` |
| MSVC x64 toolset | `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`, compiler `19.51.36243` |
| Windows SDK | `Microsoft.VisualStudio.Component.Windows11SDK.26100` |
| PostgreSQL server headers | `include\server\postgres.h` and `lib\postgres.lib` — present from the EDB installer |
| PostgreSQL client | `psql.exe` |
| Git | to fetch the pinned tag |

Only the two Visual Studio components above are needed. Adding the whole
"Desktop development with C++" workload would pull roughly 4.2 GB instead of
about 600 MB, and nothing else in the project needs it.

The compiler must be the **x64** toolset. The server is 64-bit, and a 32-bit
build produces a `vector.dll` that will not load.

## Building it

Run elevated, because `nmake install` writes into the PostgreSQL install
directory:

```powershell
cd "D:\Reddit Scam detetction"
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-pgvector.ps1
```

The script is idempotent. It skips the Visual Studio download when the
toolset and SDK are already present, so a re-run only rebuilds. Pass
`-PGVECTOR_TAG v0.8.5` to build a different version, or `-PGPort 5432` if the
server is not on the default port.

It writes its log to `%LOCALAPPDATA%\pgvector-src\install-pgvector.log` and a
machine-readable summary to `install-pgvector.result` beside it.

## Activating it

`CREATE EXTENSION` is per database and is deliberately not in the build
script, so the script never handles a database password. For a new database:

```sql
CREATE EXTENSION vector;
```

Check what a server offers without activating anything:

```sql
SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

## Verifying a fresh install

A table with a `vector` column, a round trip, and a real distance query:

```sql
CREATE TABLE items (
  id        bigserial PRIMARY KEY,
  label     text      NOT NULL,
  embedding vector(3) NOT NULL
);

INSERT INTO items (label, embedding) VALUES
  ('alpha', '[1,2,3]'),
  ('beta',  '[4,5,6]'),
  ('gamma', '[1,2,4]');

SELECT a.label, b.label, round((a.embedding <=> b.embedding)::numeric, 6) AS cosine_distance
FROM items a CROSS JOIN items b
WHERE a.id <> b.id
ORDER BY a.embedding <=> b.embedding
LIMIT 3;
```

Expected nearest neighbour for `[1,2,3]` is `gamma` at a cosine distance of
`0.008540`, and `beta` is the furthest at `0.055007`. Assert those values
rather than only checking that rows come back, so a silently broken distance
operator cannot pass as a working install.

Confirm the index path is used rather than a sequential scan:

```sql
CREATE INDEX items_hnsw ON items USING hnsw (embedding vector_cosine_ops);
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT label FROM items ORDER BY embedding <=> '[1,2,3]' LIMIT 2;
```

Expect `Index Scan using items_hnsw`. With `enable_seqscan` off this proves the
index is usable; on a table this small Postgres would prefer a scan anyway, so
the setting is load-bearing.
