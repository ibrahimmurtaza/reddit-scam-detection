# A Contact Point joins two accounts, and the recall bound is stated as a count

Two accounts reach the same Campaign Candidate when they share a Registrable Domain or a
shared Contact Point. Both edges are read from the Corpus in one pass and fed to the same
union-find, so the graph is one graph and a candidate is a component of it. Every
candidate is labelled with the edge that justified it — registrations, Contact Points, or
both — and a candidate resting on a Contact Point alone carries, on its own line, the
statement that this is the weaker of the two readings and where the measured recall
behind it is published.

A Contact Point published under a withheld registration is withheld with that
registration, and named in the output beside the registrations that were withheld.

The method's recall bound is published as a count. A Planted Campaign with nothing
inside its own membership that reaches a Campaign Candidate is beyond this method by
construction: no candidate can hold two of its accounts without also reaching outside
the membership, and a candidate that does is an over-grouping rather than a recovery.
`rfi campaign-recovery` counts those campaigns, names each one, lists what every other
membership shares that reaches a candidate, and states the ceiling the recovery figure
sits under. The bound is printed on every run, including a run where it is zero.

## Considered Options

- **Give the Contact Point edge half the weight, or require two accounts of a
  candidate to share something else as well.** Rejected: a rule that needs a second kind
  of agreement to let the first kind count is a rule that quietly decides the edge does
  not count, and it would leave the campaign that rotates registrations per post
  invisible for the same reason it was invisible before. The weighting is done by
  labelling the evidence and saying where the measurement behind it is, which a reader
  can act on and a hidden threshold cannot.
- **Do not publish the bound and let `1 of 2` stand on its own.** Rejected: the figure
  reads as a rate of fraud found in the world, and the ceiling under it is the difference
  between a measurement and a claim. ADR-0005 already requires the limitation to be
  stated rather than papered over; a count of the campaigns concerned is the checkable
  form of that.
- **Publish the candidates' edges again in a second file so the evaluator can ask what a
  membership shares without re-deriving it.** Rejected: any edge two accounts share puts
  them in one component, so every shared edge is already named under the candidate holding
  those accounts, and a second file would be the same facts in a place that could drift
  from the first. The evaluator reads the evidence already published beside each
  candidate, which is why it still opens no Public Suffix List and no
  shared-infrastructure list and does no grouping of its own (ADR-0018, ADR-0022).
- **Define the bound as "the accounts share nothing with each other" and measure that
  from the Corpus.** Rejected, and it is the near miss worth recording. The evaluator may
  not open the shared-infrastructure list, so it cannot tell a membership that shares
  nothing from one whose only shared registration was withheld — and the second is just
  as beyond the method as the first. The bound is therefore measured against the edges the
  grouping published: nothing inside the membership reaches a candidate. That is a smaller
  claim than "they share nothing" and it is the one the evidence supports, and the output
  says which of the two it is making.
- **Store the evidence label as a column of `campaign-candidates.jsonl`.** Rejected: a
  stored label is a second account of the same fact, and a file holding a label nothing
  checks is a file that can disagree with the table printed beside it. The label is
  derived from the two evidence lists on demand, in both views, so there is nothing to
  disagree.
- **Stop the grouping reading the Corpus's Contact Points and read
  `data/contacts/post-contacts.jsonl` instead.** Rejected: the grouping would then depend
  on somebody having run the earlier command, and would open a fourth file to admit it.
  The reading is called rather than re-read, so the grouping stays a function of the
  Corpus and of the two published lists it already names in its output.

## Consequences

The published figures move, and the movement is the point rather than a side effect. On
this Corpus the second edge finds a desk no registration could reach —
`syn-nuisance-obfuscated-copperlantern`, three accounts publishing one Telegram handle
three ways, two of which reach no registration at all — and it also joins
`syn_greyloch_6612` to the alpha Planted Campaign, because that account published the
desk's handle while writing the complaint down. The candidate therefore holds the
membership and one account from outside it, alpha's outcome becomes `partial`, and the
recovery figure is 1 of 2 rather than 2 of 2. Nothing in this project can tell an operator
from a customer, and a rule that tried would be the guess ADR-0005 forbids; the figure is
published as it falls, with the account that cost it named beside it, and the
false-grouping rate rises with it. The tests hold the outcome rather than the outcome
being discovered later.

The bound is zero on this Corpus, because both Planted Campaigns share something within
their own memberships, and it is printed at zero anyway: a figure that appears only when
it is bad is a figure a reader cannot tell from a missing one.

`CampaignCandidate` carries a `shared_contact_points` field, so the candidates file grows
a list and every consumer of it grows a reader: `rfi campaign-recovery` and
`rfi review-queue` both name the Contact Points a candidate is joined on under the same
`Evidence` label the grouping printed them with, because a Contact Point and a
Registrable Domain are the same shape and a different claim and a reader meeting one
candidate in two views must not be shown two vocabularies for it. The four account
figures the grouping prints are a partition over both edges now — an account grouped by
a shared Contact Point has been reached, and counting it as unreachable would put it in
two of the four lines at once.