# Calibration is measured over a stated binning, and nothing is corrected toward the model

Calibration is measured by `rfi confidence`, over the same held-out probabilities every
other figure on the page is measured over, and published as a section of its own: a
reliability table over **ten equal-width bins** on the open unit interval, the **expected
calibration error**, the **worst single bin**, and the **mean Confidence against the base
rate**, with a verdict sentence computed from those four rather than written into the page.
The binning, the weighting and the bar are named in the output, because the same
probabilities binned five ways give five different figures and a reader cannot redo one
they were not told how to build.

**Three figures rather than one, because the one everybody publishes is the one that
hides.** The expected calibration error is an average of per-bin gaps, weighted by the
posts in each bin, and a model whose bins are mostly right and whose top bin is badly
wrong has a small average and a large worst bin. The worst bin is therefore published
beside the average rather than folded into it, and the mean against the base rate is
published beside both because it needs no binning at all: it is the published column
added up and divided by the number of rows.

**Equal width rather than equal count.** Ten equal-width bins are whole numbers a reader
can lay a ruler against, and the bin a post falls in is decided by its own probability
rather than by its rank among the other 33 — so dropping one post cannot move every other
post into a different bin, which is the same reason the Features are per-post
(ADR-0007). The cost is real and is paid in the table rather than hidden: 34 posts over
ten bins leaves bins holding one post and bins holding none, so every bin's count is
printed beside its figures, and a bin with no posts prints dashes rather than zeros.

**Calibration and discrimination are different questions and are never printed as each
other's evidence.** AUC asks whether the Confidence *orders* the planted posts above the
rest; calibration asks whether the number attached to a post is the frequency it claims.
A model can answer the first and fail the second. The two live in a section and a table
rather than in two columns of one row for the same reason ADR-0003 keeps the Confidence
and the Policy Score apart: one reading of a figure as a reading of another is how a
miscalibrated model gets read as vindicated by a good AUC, or the other way round.

**Nothing is corrected toward anything.** No recalibration is fitted over the model's
output: the published probabilities are what the folds' fits produced, seven positives
cannot support a fitted mapping, and a mapped probability would no longer be a number
this project could recompute from the coefficients it prints. And the Policy Score is
never corrected toward the model either — it stays the sum of the published weights of
the Signals a post carries (ADR-0014), a sum `tests/test_policy_score.py` holds by
checking that no module the score is computed through reaches the Confidence or the
embeddings, and that a file of probabilities beside the Corpus moves no score.

**The Confidence stays displayed nowhere whatever the figure says** (ADR-0003,
ADR-0027). A miscalibrated model is an argument for keeping a number off every screen
and never an argument for putting one on it, so both views print that where the
calibration figure is printed, and the structural enforcement is unchanged by it.

**On this Corpus the model is not calibrated**, and the report says so in the first line
of the section rather than in a footnote: an expected calibration error of 0.1957 against
a bar of 0.0500, a mean Confidence of 0.1425 against a base rate of 0.2059, and a worst
bin of 0.8098 — a post the model is 81% sure of that is not planted.

## Considered Options

- **Fit a recalibration and publish the mapped probabilities.** Rejected: it would make
  the published number a second model's output, fitted on seven positives, with no
  published recipe behind it and no way for a reader to redo it from the coefficients on
  the page. ADR-0027 publishes the model's output as the model's own; a mapping fitted
  to make a miscalibrated model look calibrated would also make the published file stale
  against every corpus but this one. The bar, not the mapping, is what this project
  publishes.
- **Report the expected calibration error alone.** Rejected: it is an average, and the
  acceptance criterion is a miscalibrated Confidence reported plainly rather than in a
  form that hides it. 0.1957 is a number to interpret; "the bin holding one post the
  model is 0.81 sure of is wrong" is the finding.
- **Equal-mass bins, so every bin holds the same number of posts.** Rejected: the bin a
  post falls in would then depend on every other post in the Corpus, so dropping one post
  would move the rest between bins and the table would not be checkable against the
  published file row by row. The thinness is real either way over 34 posts; publishing the
  counts beside the figures is the honest answer to it, and hiding it behind a binning
  chosen to look balanced is not.
- **A decomposition of the Brier score into reliability, resolution and uncertainty.**
  Rejected: it is three more figures to read for the same question, its "reliability"
  term is an unweighted squared gap that is not the quantity a reader means by
  calibration, and nothing in this ticket needs a resolution figure. The three figures
  published are the three a reader can recompute from the file and the labels.
- **Measure how sensible the Policy Score is and publish it beside.** Rejected, and it is
  what ADR-0003 rules out in the other direction: a 0-100 editorial figure has no
  frequency to be measured against, so "calibrating" it would mean fitting a likelihood
  to a judgement — the fusion by the other door. Whether the Policy Score is sensible is
  a question about the published weights and is answered where the weights are.
- **A separate `rfi calibration` command.** Rejected: the calibration is measured over
  the same folds, the same features and the same held-out rows as the figures it sits
  beside, so a second command would re-fit the model to measure a property of the fit it
  had already made, and a disagreement between the two commands would be a disagreement
  about which of them had fitted. One command, two sections, one pass over four files.

## Consequences

`Evaluation` carries a `Calibration` beside its figures rather than a fourth row of its
table, and `rfi confidence` writes it into both of its views: the console output gains a
`calibration` block after the baselines, and `docs/confidence.md` gains a `## Calibration`
section between the baselines and what the figures are not. `data/model/confidences.jsonl`
is unchanged — the probabilities are the model's own and no mapping is applied to them —
so the ticket cost the file nothing and the report one section.

The report's section on what the figures are not now points at the calibration rather
than saying it is unmeasured, and `GLOSSARY.md` gains **Calibration** beside **Confidence**
so the word means one thing across the repository.

The cost is stated rather than hidden. Over 7 positives and 34 posts the expected
calibration error is itself a coarse figure: five of the ten bins hold a single post, so
an observed share there is 0 or 1 whatever the model said. The weighting keeps those bins
from carrying the figure — one post contributes at most 0.0294 of it — and the counts are
printed so a reader can see which rows of the table are evidence and which are a single
post. A Corpus with reviewer labels and enough positives to bin would make this figure
mean something it cannot yet mean.