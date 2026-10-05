# Recovery, the false-grouping rate, and queue precision are measured together

`rfi campaign-recovery` publishes its two headline numbers together, because either
one alone is half a claim: a recovery number produced by a grouping that also merges
unrelated accounts is a number that flatters the system, and a precision number
without the recovery beside it says nothing about what was missed. The report
therefore carries the recovery figure, the false-grouping rate against the same
Nuisance Structure manifest, and the Review Queue's precision at several depths in
one page, and it states plainly that of the two rates — recovery and precision — the
lower of the two numbers is the more trustworthy, deliberately.

A false grouping is defined in the output itself, not in a ticket: a Campaign
Candidate whose accounts belong to different Planted Campaigns, or to none. The
first case is a grouping that reached across a membership boundary; the second is
the decoy account cluster, the shop whose three accounts ADR-0005 puts together
correctly and whose candidate therefore counts as a false grouping of accounts that
belong to no Planted Campaign at all. The rate is reported per candidate, and each
one is named with the Nuisance Structure records its accounts appear in, because a
rate readable without its baseline is not checkable.

Precision at a depth is the share of the top entries of the Review Queue whose posts
hold an account of a Planted Campaign, and it is published at several depths rather
than one so the shape of the trade-off is visible. It is measured by this evaluator,
against the Planted Campaign membership, not by a person reviewing the queue, and a
precision number is never printed without its depth.

## Considered Options

- **Report the false-grouping rate in a new command over the recovery join.** Rejected:
  dividing the measurement across two commands would split the membership boundary
  ADR-0018 draws. One evaluator command, one place the truth file is read, and the
  scores it joins are a fifth file rather than a second membership. The queue itself
  keeps its own rule: it prints no figure over the top D (ADR-0020), and precision
  stays with the evaluator that may open the membership.
- **Count a candidate a true grouping only when it equals one Planted Campaign's whole
  membership.** Rejected: that is the recovery rule, and it conflates the two
  questions. A candidate holding two of a campaign's three accounts is not false — the
  grouping is consistent with the planted structure — so a grouping that held only
  part of one campaign and nothing else counts as true here, while a candidate holding
  one campaign's accounts plus one from no campaign, or accounts of two campaigns, is
  false. Partial matches keep their own view in the recovery figure.
- **Measure false groupings per pair of accounts, or against the Nuisance Structure's
  account set alone.** Rejected: the candidate is the unit the grouping publishes, and
  the rate has to be checkable against the file the grouping wrote. Account-pair
  figures would not decompose back to the candidates, and a rate measured against the
  manifest instead of the candidates would not say how often the grouping was wrong.
- **Publish precision at one depth.** Rejected: a single depth is one point on the
  trade-off rather than its shape. The default list is 5, 10, 20, and 50, so a reader
  sees precision shrink as the queue deepens; a caller can state their own list with
  `--depths`.
- **Name accuracy.** Rejected, and the string itself appears nowhere in the output:
  a share of posts called right or wrong measures agreement with the generator
  (ADR-0004), not fraud found in the world. Precision against the planted membership
  is the only right-or-wrong figure this project publishes.

## Consequences

The committed `docs/campaign-recovery.md` carries all three numbers, and the tests in
`tests/test_campaign_recovery.py` hold them to a hand-computed value on a small Corpus,
to the audit-hook test that watches which files each step opens, to the reproducibility
test that rebuilds the chain from the seed the report names, and to the absence of the
string the report is never allowed to say. `data/evaluation/recovery.jsonl` keeps its
schema — the join, one row per Planted Campaign — so a candidate that reached no
campaign stays a named leftover rather than a second kind of row in one file, and the
new figures live in the console and the report rather than in the file. Nothing in the
grouping path reads a score, a membership, or a manifest; nothing in the scoring path
reads a membership; and the evaluator still opens no Public Suffix List and no
shared-infrastructure list.
