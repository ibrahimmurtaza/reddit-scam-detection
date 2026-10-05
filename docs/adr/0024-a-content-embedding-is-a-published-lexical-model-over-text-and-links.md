# A Content Embedding is a published lexical model over text and links, in an HNSW-indexed vector column

Every Content Item gets one sentence embedding, computed from the post's own title, its own
body and its links, and stored in a `vector` column in Postgres under an HNSW index. Nothing
more: this is the substrate ticket, and it builds no grouping, corroborates nothing and
decides nothing. Ticket #24 is the first thing to read these vectors.

Four decisions travel with that, and each of them is one a later reader will want to argue
with.

**What is embedded is the three fields, and the account is not one of them.** The Corpus
boundary exists so the pipeline cannot see Planted Campaign membership (ADR-0008), and a
vector built out of who posted would put that back through a side door no reader of the file
could see. `tests/test_content_embeddings.py` holds this structurally rather than as a claim:
the whole Corpus is re-read with every account renamed, every subreddit swapped and every
timestamp moved, and the two sets of vectors and digests have to come back identical.

**The model is this project's own, and its name, its width and a digest of its recipe travel
on every row.** `hashed-word-ngrams-v1`, 256 dimensions: a feature is each word and each pair
of adjacent words inside one field, a feature's count is damped as `1 + ln(count)`, each
feature lands in one of 256 buckets with its sign taken from the same digest, and the vector
is scaled to unit length. `blake2b` rather than Python's built-in `hash()` because that one
is salted per process and a model producing a different Corpus on every run could not have
its output committed and held to its bytes.

**A vector is computed once and reused, decided before any embedding happens.** Every row
carries the SHA-256 of the text it was computed from, so a rerun over an unchanged Corpus
hashes each post and embeds nothing. An implementation that embedded first and skipped the
write would satisfy the rule at the writer rather than at the work.

**The index is HNSW.** IVFFlat needs training data before it answers anything and its
`lists` parameter is a guess that has to be revisited as a corpus grows; HNSW answers from
the moment it is built and holds its recall without a parameter. On this Corpus the index is
56 kB over 34 rows and the planner will not use it, so the command runs `EXPLAIN` on the query
it actually issues and prints the node the planner chose.

The memory characteristics are the half of this the ticket asks about by name, and they differ
between the free tier and a large corpus in opposite directions for the two access methods.

**HNSW's memory grows with the table; IVFFlat's does not.** HNSW builds a multilayer graph
resident in `maintenance_work_mem`, reads it on every query, and gets roughly `rows × m` edges
— `m` being the neighbours per node, 16 by default. Here that is 56 kB over 34 rows, which is
nothing. It does not need to fit in RAM to be correct, but it is the largest single object a
query touches and the one that most wants to be cached, so on a constrained instance it is what
decides how large a Corpus can be indexed at all. `m` is the dial: raising it buys recall at a
linear cost in memory.

**IVFFlat holds `lists` centroids and nothing else**, so its memory is `lists × dim` — a few
hundred kilobytes whatever the table holds, and pgvector's own guidance is to create the index
only *after* the table has some data. That is why it is the better answer on a constrained
instance with a large table, and it is also why it is worse here: pgvector suggests about
`rows / 1000` lists, so at 34 rows the index is one or two partitions holding a handful of
vectors each, and its recall collapses long before the planner stops preferring a scan.

So the choice is not "HNSW is smaller". It is: HNSW's memory grows with the Corpus and is the
dominant cost once the Corpus is large; IVFFlat's memory does not grow and its recall does. For
a Corpus this project can hold in one table — a few tens of thousands of posts — HNSW pays a
few megabytes and answers from the first query, and IVFFlat pays nothing and answers badly.
Revisit it at the size where the graph stops fitting, not before, which is the same reason the
run publishes the plan rather than assuming it.

## Considered Options

- **Download a sentence-transformer and commit its weights.** Rejected, and this is the
  decision most worth arguing with. A real semantic model is a better substrate than a
  lexical one and ticket #24 is better off with it. It also puts roughly 90 MB of a
  third-party artefact in a repository whose entire claim is that a reader can check its
  outputs against committed bytes, and it puts a download in the middle of a test suite whose
  own documentation says it needs no network. The cost of rejecting it is written down
  rather than discovered later: **this model reads words and not meaning.** Two posts making
  the same pitch in different words can come out far apart, and ticket #24 inherits that
  limit. If a hosted Corpus arrives with a model this project is permitted to pin, the
  dimensionality is in the schema and the model name is on every row, so the change is a
  refusal followed by a `--table` argument rather than a migration.
- **Publish the vectors, not just the record of them.** Rejected. 34 posts of 256 numbers is
  about 100 kB of `0.10000000149011612`, which is a file a reader has to trust rather than
  check, and the values are in the database the ticket asked for them to be in. What the
  published file buys instead is the reuse rule in the open: a reader recomputes every digest
  from the Corpus and sees which text each stored vector belongs to. `tests/test_content_embeddings.py`
  holds the file to its bytes, and a reader who wants a vector asks the database.
- **Narrow the vector to 32 bits in the model, so the published record and the stored vector
  are the same numbers.** Rejected, because it was not needed. The column stores 32-bit
  floats whatever the model produces, so the model stays at 64 bits and the distances the
  output prints are the ones pgvector computed from what it holds rather than from the widest
  numbers available. Rounding at the source would have bought a file of the same length, and
  the length was the reason for rejecting it.
- **Recompute every vector on every run and write them all.** Rejected: it satisfies
  "computed once" only in the sense that the answer is the same, and the acceptance criterion
  is about not doing the work. The digest is what makes the difference observable, which is
  why the run prints `computed` and `reused` separately.
- **Leave rows for Content Items the Corpus no longer holds.** Rejected: the table is a
  projection of the Corpus, and a row left behind comes out of every similarity query as
  though it were a post somebody could go and read.
- **Record which Corpus the table is a projection of, and refuse a different one.** Rejected,
  and it is the near miss worth recording, because it is the more obvious guard and it is
  wrong. The obvious identity is the digest of the Corpus file, and editing one post's body
  rewrites that file: every legitimate edit would be refused, which is the exact operation the
  reuse rule exists to make cheap. Any identity stable across an edit is a property of the
  posts rather than the file, so the guard reads post ids instead — two Corpora sharing no
  Content Item is the shape of a mistyped `--corpus`, and a Corpus that edits posts shares
  every post it did not touch and is never refused. `--replace` lifts it deliberately, because
  a guard with no way to lift it on purpose is a guard that gets disabled rather than obeyed.
- **`ivfflat` with a `lists` count chosen for this corpus.** Rejected, and it is the near miss
  worth recording. It is one index type away from the one chosen, it is what several
  tutorials reach for first, and on a table this small it is *worse* in the way that matters:
  it will not answer a single query until it has been given training data, and a reader
  running this over a three-post Corpus to see what it does would get nothing at all. HNSW
  answers from the moment it is built.
- **`halfvec`, which halves the memory an index costs.** Rejected for now, and the memory
  paragraph above is why rather than a preference. It stores 32-bit values against a full
  vector's 64, so it halves the same growing quantity HNSW's graph dominates — which is the
  wrong trade at a Corpus of 34 rows, where the index's 56 kB is not a constraint, and the
  right one at the size where it is. It is also a one-way door: the stored bytes change, so
  every stored vector has to be rewritten and every distance recomputed. The argument against
  it is that it should be revisited at the size where the index's own footprint is the largest
  thing in the database, and that decision belongs with a corpus that size rather than with
  this one.
- **No index at all, since the planner does not use it here anyway.** Rejected. The planner's
  answer at 34 rows is about this Corpus, not about the substrate: a Corpus large enough for
  HNSW to be chosen is a Corpus this project expects to meet, and a command with no index
  would need a code change at exactly the moment its output matters most.
- **Claim the index is at work, and check it with `enable_seqscan = off`.** Rejected, and
  this one is a trap. `docs/pgvector.md` uses that setting to prove the index is *usable*,
  which is a different claim from the one the output would be making. A run that forced the
  planner's hand and then printed `Index Scan` would be printing a figure it had arranged.
  The command explains the query it issues and publishes what came back, which on this Corpus
  is a sequential scan.
- **`CREATE EXTENSION vector` from the command, so the first run needs no manual step.**
  Rejected: it is per database, it needs a role permitted to create one, and it is a
  persistent change to the database rather than to a row in it. A database without the
  extension is refused with the statement to run, which is what `docs/pgvector.md` already
  told whoever built it.
- **A `--dsn` argument.** Rejected. An argument is a place a password ends up in a shell
  history and in a CI log, and this repository keeps credentials in the environment. The
  connection comes from `RFI_DATABASE_URL` or from the `PG*` variables libpq reads, and the
  command refuses when neither is set rather than accepting whatever database libpq would
  have picked by itself.

## Consequences

`data/embeddings/content-embeddings.jsonl` is a new generated file with five fields per row —
`post_id`, `model`, `dimensions`, `recipe`, `text_sha256` — and **no vector and no account**.
The absence of an account field is the point: there is nowhere in the published record for one
to go, which is what makes "renaming every account changes nothing" a property of the file
rather than of a habit. `read_content_records` refuses a row carrying a field this project
does not publish, for ADR-0008's reason, and refuses a file whose rows disagree about the
model, the width or the recipe, because a distance between two models' numbers is not a
distance.

`recipe` is the field that closes the gap between the model's name and its behaviour. A name
carrying a version is a promise somebody has to keep; the digest of the recipe is the promise
checked. Editing the recipe without bumping the name would otherwise reuse every stored vector
on the strength of a digest that still matched — the model name would match, the width would
match, and the text digests would match, because the text is not what changed.

`psycopg` is this repository's first runtime dependency. The embedding model needs no wheel,
which is deliberate: a lexical model that ships as fifteen lines of arithmetic keeps the
toolchain pinned to Python 3.13 for the reason the README gives — no prebuilt machine-learning
wheels on 3.14 — without needing one yet. Ticket #25 will need one.

The similarity query reads its query vector out of the table rather than taking one as an
argument, so the question cannot be asked about a vector nobody has stored: a reader gets the
distance from the post as the database holds it. It is two statements rather than a
self-join, because a self-join for a table's own vector stops the planner using the index at
all, and the index is the part of this substrate a larger Corpus would rely on.

**Links are part of what is embedded, and that is a limit ticket #24 has to be told about
rather than find out.** Two posts sharing nothing but a link shortener share the words of that
host whichever pitch they are making, so a similarity over these vectors is partly a
similarity of the hosts two posts link, and ADR-0005's own edge is reached again here through
the text. It is not independent of the edge the grouping already has. The output says so where
a reader meets a distance, and `tests/test_content_embeddings.py` demonstrates it with two
posts whose only shared vocabulary is one host.

What the model does well on this Corpus, measured and published on every run: the closest
pair of posts is `syn_p_0012` and `syn_p_0013` at a cosine distance of `0.1504`, and
`syn_p_0014` is `0.2091` from the first of them — one Planted Campaign's staggered paraphrase
of a single offer, found by vocabulary alone, and the run prints both figures in its
nearest-neighbour table for anyone to check. What it does not do is read meaning, and no
threshold in this repository may pretend otherwise. The distance above is only legible against
a baseline — the distance two posts with nothing in common sit at — and this command does not
publish one, because a baseline over every pair of posts is `O(n²)` in the database and the
figure belongs to the step that uses it. Ticket #24 measures that baseline and states its
threshold beside it.