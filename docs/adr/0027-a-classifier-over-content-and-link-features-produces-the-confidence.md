# A classifier over content and link features produces the Confidence

Every Content Item gets a Confidence: a probability between 0 and 1, produced by a
model this project fits itself, stored in its own file at its own path, and never
rendered as a severity, never summed with the Policy Score, and never printed by any
command a reviewer reads. The Policy Score stays what ADR-0003 made it — an additive
rules engine whose weights are published and whose arithmetic a reader can redo — and
the Confidence is a second quantity, held internally, that exists so this project's
machine-learning claim is a measurement rather than an adjective.

**The model is plain logistic regression, fitted here, over seven published features.**
Each feature is computable from a post's own title, its own body and its own links, and
each is a count a reader can check by hand. Nothing reads an account's age, karma,
posting rate or activity change, no account is a feature, and a feature is read from
one post rather than across the Corpus (ADR-0007, ADR-0008).

**The labels are the Planted Campaign membership, and the report says so in its own
words.** A post's label is whether its account is a member of a Planted Campaign, taken
from `data/corpus/truth.jsonl`. That is a file only the evaluator is otherwise permitted
to read, and this command reads it for the same reason the evaluator does: a supervised
model needs labels, and a Corpus from a provider has none beside it. So what is measured
here is **recovery of planted structure** — whether a classifier can tell the generator's
planted posts from the rest of a Corpus the generator also wrote — and it is not a rate
of fraud found in the world. The report states that in a section of its own rather than in
a footnote, because a reader who takes a number from this page for anything else has been
told something false about how it came about.

**Performance is measured held out, over the whole Corpus at once, and reported beside
two trivial baselines.** Every post's Confidence is produced by a model fitted without
that post and without every other post of that post's Planted Campaign — leave-one-campaign-out
— so the figure is about posts the model has not seen rather than about posts it
memorised. The baselines are the two things a classifier has to beat to be worth
reading: a constant equal to the base rate, and the Policy Score's own ordering of the
same posts. Both are printed with the same metrics, so the model's contribution is a
difference a reader can see rather than a claim.

## Considered Options

- **A decision tree, a random forest, or gradient boosting.** Rejected: with 7 positives
  and 27 negatives, a tree ensemble fits the training rows exactly and has nothing to say
  about a post it has not seen, and the number of splits it chooses is a hyperparameter
  with no published reason behind it. Logistic regression on 7 features fitted by
  gradient descent is one hyperparameter (a fixed step count and a fixed penalty), it is
  fitted in this repository with no wheel to install, and every coefficient is a number a
  reader can see in the output.
- **scikit-learn.** Rejected for the reason ADR-0024 gives for refusing a downloaded
  sentence-transformer: a 30 MB third-party artefact in a repository whose claim is that a
  reader can recompute every published number. Logistic regression on seven features is a
  page of arithmetic.
- **Reuse the Content Embedding as the feature vector.** Rejected as 256 features against
  7 positives — the model would memorise the Corpus and the held-out figure would be
  reported as though it measured generalisation. The embedding is a lexical reading of
  words and not meaning (ADR-0024), so its 256 buckets are not 256 things a reviewer can
  check either.
- **Read the Policy Score as a feature.** Rejected, and it is the decision this ADR
  exists for. A model given the Policy Score as an input returns the Policy Score with a
  decimal point on it, and ADR-0003's separation would become a matter of arithmetic
  happening not to fire. The features below are read from the post and its links; no
  Signal, no weight, and no Policy Score is an input to the Confidence, and
  `tests/test_confidence.py` asserts that no arithmetic in this repository combines the
  two numbers.
- **Train and report on the whole Corpus at once.** Rejected: with 7 positives the
  training fit is perfect by construction and its metrics say nothing. Every figure
  published here is out-of-fold.
- **Refuse a Corpus with one Planted Campaign.** Accepted, and the cost is stated on the
  page. One campaign means one fold, the fold holds out every planted post with it, and the
  fit is left with no positive — a logistic regression over one class has no finite answer.
  The alternative would be a training fit reading back its own answer, or a figure with no
  fold behind it, and both are what this ADR exists to avoid. The cost is that no Corpus
  with a single campaign can be measured this way at all, which is a limit on the method
  rather than a figure about the run.
- **Calibrate the Confidence here.** Rejected, and it is ticket #27. ADR-0003 already
  says calibration is a separate concern with its own evaluation, and a calibration
  figure published before the model exists would be a figure about a model nobody has
  chosen. What this ADR publishes is discrimination — how well the Confidence orders
  posts — not calibration, and the report says which of the two it is reporting.
- **Store the Confidence in the `content_embedding` table.** Rejected: that table's
  columns are one model's output and a digest of the text behind it, and a column
  holding a probability from a different model beside a `vector` column is a row whose
  two halves have no common meaning. The Confidence has its own file, with the same
  vocabulary discipline `read_content_records` applies: a field this project does not
  publish is refused rather than ignored, which is what keeps a Planted Campaign's
  identifier out of it.

## Consequences

`data/model/confidences.jsonl` gains a row per Content Item: the post, the account,
the probability, the model that produced it, and the digest of the recipe.
`docs/confidence.md` is the report, generated from those rows and from the evaluation.
`rfi confidence` needs no database, and reads four files: the Corpus, the published
Public Suffix List, the Planted Campaign membership for its labels, and the published
Policy Scores for the baseline and nothing else.

A fifth file is named here rather than left implicit: the ticket's hard constraint says
the Confidence is stored in its own **column**, and ADR-0003 says the two are stored in
separate columns. There is no column for either number here — the Policy Score lives in a
JSON Lines field, not a table — so the separation is at file level, and a field for one
number beside a field for the other in one record would defeat the point. The reader
refuses a row holding any field beyond the five published, so a Policy Score cannot later
be added to a Confidence row without the run stopping.

**One fold's weights are never published as the run's.** Leave-one-campaign-out fits once
per fold, so there is no single model behind the published probabilities, and a table of one
set of coefficients beside all of them would be a claim the numbers do not support. Each
fold's weights are printed under a heading of its own, with the number of posts that fit
scored.

The fold a post was held out in is published in the report rather than in the file. A
`fold` field would partition the posts by the Planted Campaign they came from, which is
membership by another name (ADR-0008) — two rows sharing a fold number would say two
posts may be one campaign, and that is the label this command is not allowed to hand
back.

The cost is stated rather than hidden, and it is large. **A Confidence fitted on 7
positives measured out-of-fold over a Corpus of 34 is a measurement of this project's
generator, not a model.** The report names the fold structure, the base rate, the size of
the positive class, and the two baselines, and states that a figure over 7 positives
cannot distinguish a model that generalises from one that has been lucky. A real Corpus
with reviewer labels is what would make the figure mean anything, and ADR-0002 already
says supervised learning on human review is the destination rather than the start.

**And on this Corpus the model does not win.** It orders the planted posts better than the
base rate and worse than the Policy Score this project already ships, and it is scored
worse than the constant by a proper scoring rule. The report states that in a sentence of
its own, computed from the figures rather than written into the template, because a model
that cannot beat the base rate is worse than saying nothing and a reader who has to
subtract it out of a table has been handed the work. Publishing the loss is the ticket's
"performance measured rather than assumed" being taken literally: a claim of a useful model
on these numbers would be the assumption, and the numbers are what they are.

Nothing reads the Confidence. `rfi review-queue` and `rfi campaign-candidates` do not
open the file, print no column for it, and name no probability in their output, and a
test asserts both — the absence in the printed output and the absence of any arithmetic
that would combine it with the Policy Score. The Confidence is stored because ADR-0003
says it is stored in its own column and never displayed, and a quantity that exists only
in a report would be a claim; storing it and reading it nowhere is what makes it a fact
that can later be measured rather than asserted.