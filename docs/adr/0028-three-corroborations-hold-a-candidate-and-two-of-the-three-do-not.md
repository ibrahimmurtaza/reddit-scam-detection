# Two of three corroborations hold a candidate, and the baseline ships beside the system

ADR-0009 decided that both tiers are reported and left "what corroboration
adds" open. It is three named verdicts rather than one weighted number: a
Campaign Candidate is scored on Temporal Proximity, Content Similarity and
Category Agreement — the Scam Category its own posts agree on, decided by the
same strict-majority rule ADR-0021 gives a Registrable Domain — and is retained
when two of the three hold and removed when they do not. The count is published
with the names of the corroborations that held and the ones that went against
it, and the direct-adjacency baseline stays in the candidates file with its
identifiers where they were.

The bar is two rather than one because one corroboration is one reading
agreeing with itself. A candidate the clock corroborates and nothing else is a
registration worked months apart; a candidate the categories corroborate and
nothing else is ten classes wide. The third is what catches the case where the
first two would both have said yes for the same reason.

Category Agreement is read over the candidate's own posts rather than over the
registration it rests on. The claim a candidate makes is about the accounts in
it, and the question the third corroboration answers is whether those accounts
are making the same pitch; the registration's own Associated Scam Category is
already what the conflict Signal fires on, and reading it again here would
measure a candidate against something outside it.

This does not overturn ADR-0025 or ADR-0026. Neither the window nor the
threshold may remove a candidate on its own, and neither does here: a candidate
the clock corroborates and the vectors do not is retained on the clock and the
categories together, and one the categories corroborate and neither of the other
two does is retained the same way. What removes a candidate is the third verdict
being counted with the first two, which is a step of its own and says so.

## Considered Options

- **Weight the three into one score.** Rejected: ADR-0026 refused exactly this
  for two Signals, and the refusal holds. A weighted total is a second account
  of the facts with the weights hidden inside it, and a reader who disagrees with
  a weight has nowhere to go. A count of named verdicts is checkable — the names
  are beside it, each keeps its own published threshold, and the file publishes
  the tally the category verdict is counted from.
- **Replace the baseline with the system's output.** Rejected, and it is what
  ADR-0009 already refused: publishing the lower number alone leaves a reader with
  no way to tell a filtering decision from a bug, and it discards the fallback
  the ticket asks for. The baseline is also the number every figure measured
  against — the recovery figure and the false-grouping rate are counts of what
  the union-find proposed — so replacing it would mean re-deriving all of them
  and losing the record of what they were measured against.
- **Reorder the candidates so the retained ones come first.** Rejected: the
  baseline and its identifiers are what `data/campaigns/campaign-candidates.jsonl`
  has published since before this system existed, and reordering would make the
  pre-corroboration number a reconstruction rather than a record. What the system
  decided is said beside each candidate and in its own section below the table
  instead.
- **Let the cohesion system also set the queue's order.** Rejected as out of
  scope: the Review Queue is the Policy Score and nothing else (ADR-0020), and
  making it depend on the tier would change a figure this ticket does not own.

## Consequences

`CampaignCandidate` gains `cohesion`, so the candidates file grows one field and
every consumer of it grows a reader: `rfi campaign-recovery` and
`rfi review-queue` check the two name lists against the three verdicts on the
same row and act on neither. The tally the file publishes has to name Scam
Categories the projection publishes and has to account for every post the row
holds, and the names have to partition the corroborations between them, so a
count nothing recomputes is not a thing the file can carry.

Identifiers do not move on this Corpus: the ordering is unchanged, so `cc-01` is
still the shop, `cc-02` still alpha, `cc-03` still beta and `cc-04` still the
desk. What changes is what the system says about them. Both planted campaigns
rest on the clock and the categories together and lose their vectors, so on
category agreement alone the system would retain the shop and filter both — which
is what makes the third corroboration load-bearing rather than decorative. The
desk behind the obfuscated handles is the one candidate removed: seven posts over
three weeks and one of the three corroborated. That is also a false grouping, so
the removal costs the recovery figure nothing, and the report says so in the same
breath as the removal rather than leaving a reader to work out whether the filter
was lucky.

`placement.agreed_category` and `placement.in_print_order` are now shared with
`signals.Association`, because the majority, the floor of two and the printing
order are the same rule read at two scopes, and a rule a published figure rests on
may not be written twice. `CohesionSystem` is the grouping's own count over its
baseline and `rfi campaign-recovery` builds it from the file it read rather than
reworking it, so the two commands' copies of the tier figures are one function
over one tuple.