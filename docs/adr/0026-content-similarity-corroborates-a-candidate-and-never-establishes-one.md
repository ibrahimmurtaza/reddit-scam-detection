# Content similarity corroborates a candidate and never establishes one

Every Campaign Candidate carries the content similarity of the posts inside its
component: for each post, the distance to its nearest post by a different account in
the same candidate, over the stored content embeddings, and the candidate reads
corroborated only when every post has a near-twin by another account within the
stated threshold. The verdict and the threshold it was counted against are
published, per candidate and in the file's summary of every candidate.

The threshold is a figure in the output and a parameter on the command: cosine
distance at most 0.5 by default (`--similarity-threshold`), ordered first ahead of
every figure it was counted against. Cosine distances on this Corpus put the
decoy cluster's pasted advert at 0.15-0.21, the shop's recycled updates at
0.45-0.48, and the three Planted Campaigns' own texts at 0.53-0.74 - so 0.5
separates "the same advert pasted" from "the same offer retold", and the only
candidate whose posts all have a near-twin on this Corpus is the shop.

Similarity may deprioritise and nothing else. A candidate the threshold
corroborates nothing under is printed in full, with its nearest pair beside its
posts, is named again below the table under its own heading, and sorts below
every candidate both signals corroborate; it is never removed. Timing and
similarity each carry their own below-table section so a reader can tell which
Signal demoted a candidate and can undo the demotion at a different window or a
different threshold. Two accounts that paste the same advert hour for hour and
share no registrable domain and no Contact Point are not grouped, whatever the
threshold is set to: ADR-0005 has already refused, and the vectors are not a
second way in. The Corpus's own copy of that case is the decoy cluster, whose
text is the nearest on the Corpus and whose accounts never group.

Timing remains the clock's Signal; similarity is the vectors'. The two are kept as two keys
rather than one score: the clock keeps the first key because it had this job first
(ADR-0025) and nothing here retires it, and the vectors get the second, so a candidate both
corroborate is printed ahead of every candidate the clock corroborates and the vectors do
not. Within both, the grouping's usual largest-first, most-posts, name rule. Turning the
two into one number is the cohesion system (#26), which will add the adjacency baseline
beside it rather than replace either Signal's figure.

The rationale for treating similarity as corroboration only is written here and
in the output's footer, because it is the question every reader will ask. A scam
template converges across unrelated operators: two unrelated desks pasting the
same advert are running the same template, not the same operation, and
near-identical text is what such a convergence looks like when it crosses. The
model reads words and not meaning (ADR-0024), so "same words" cannot stand in
for "same operator": semantic identity is not what a hashed bag of unigrams and
bigrams measures, and even a stronger model would still measure the template
rather than the desk. So similarity can only ever strengthen a grouping the
shared infrastructure already proposed.

## Considered Options

- **Let similarity remove a candidate the threshold corroborates nothing under.**
  Rejected for the same reason ADR-0025 rejects it for timing: it would make a
  threshold a decision, and a threshold is a number a reader can disagree with
  while a decision is one they have to undo. The ticket asks for diagnosable
  misses, and a candidate deleted for want of similarity is a miss with nothing
  left to diagnose.
- **Decide on the closest pair of posts rather than every post.** Rejected, for
  the near-miss ADR-0025 recorded. The closest pair is the most forgiving
  reading and it lets one coincidence corroborate a whole component: two of
  seven posts matching by accident would carry the other five. The deciding
  figure is the worst per-post nearest distance within the candidate - every
  post covered, which is the span-like reading. The closest pair is still
  published, because it is the figure a reader needs to disagree.
- **Weight similarity against timing in one score.** Rejected, because that is
  the cohesion system (#26): the two Signals' figures are kept apart and each is
  inspectable on its own, and a score would be a second account of the facts.
- **Recompute boolean thresholds per row and store them.** Rejected: the file
  holds the distances and the threshold, and corroborated is worked out from
  them on read, the same way ADR-0025 keeps the verdict a recomputation.
- **Publish the vectors.** Rejected in ADR-0024, and unchanged here: the file
  publishes one row per post - its nearest post by another account in the same
  candidate, and the distance - and nothing more.

The cost is stated rather than hidden. A Planted Campaign that paraphrases its own advert
across its accounts is deprioritised for it — which is alpha on this Corpus, whose four
staggered paraphrases read as 0.59 to 0.74 apart to a model that reads words rather than
meaning. That is a real loss to a reviewer reading the list in order and a false alarm to
one weighing the figures, and both are the price of a threshold anybody can see and change.
The alternative — a rule the reader cannot check — was not available at any price, and a
semantic model that would read the paraphrases alike was rejected in ADR-0024 with its own
cost written down there.

## Consequences

`CampaignCandidate` gains `similarity` and `content_similarity`, so the
candidates file grows two fields and every consumer of it grows a reader:
`rfi campaign-recovery` and `rfi review-queue` check the rows by the same
arithmetic check the grouping's own reader does and act on neither. Identifiers
move again: on this Corpus the shop (both signals agree) is `cc-01`, alpha
(timing only) is `cc-02`, beta (timing only) is `cc-03`, and the desk (neither)
is `cc-04`. `campaign-candidates` now reads the stored vectors from pgvector via
the same environment variables as `content-embeddings` - the one thing the
command takes from a table - and refuses with the table's name and the remedy
when a post's vector is missing. The recovery figure is unchanged: similarity
never creates a candidate, and on this Corpus it removes nothing either.
