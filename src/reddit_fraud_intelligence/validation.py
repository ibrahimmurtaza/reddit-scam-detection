"""What the per-post readers do on real text nobody here wrote or labelled.

The project's own figures are measured against a Corpus the project planted (ADR-0004), so
every one of them answers a question about that Corpus. This module asks the one question
those figures cannot: do the per-post readers fire at all on real promotional text, when the
fraudulent half was marked fraudulent by somebody else, before this project existed?

The comparison is deliberately narrow. It is the Policy Score and the Scam Category
placement — two functions of one post's own text — measured against a label this project
does not control. It is **not**:

* the Confidence, which is fitted on planted membership and would need training on this set
  to mean anything, which would stop it being held out;
* the grouping, the recovery rate, or the false-grouping rate, none of which this data can
  express, because a job advertisement has no account history and no shared registrable
  domain with another advertisement;
* the three link-based Signals, which are not *absent* from the figure but *uncomputable* on
  this set, because every row is text and a label with no URLs in it. The report names them
  and says why, so that the score below is never read as the whole rules engine having run.

None of this is fitted, tuned, or thresholded against. The Policy Score is a published
rules engine with no training step (ADR-0007) and the category lists were written before
these bytes were downloaded, so the figure is a measurement of a model that already existed
— which is the only property that makes a held-out set worth having.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from reddit_fraud_intelligence.categories import OTHER
from reddit_fraud_intelligence.corpus import CorpusItem
from reddit_fraud_intelligence.external import ExternalFacts, Labelled
from reddit_fraud_intelligence.placement import Placed, place
from reddit_fraud_intelligence.signals import PostScore, SignalName

# The external set has no title, no subreddit, and no account, and this module does not
# invent them. One constant account and one fixed instant, so that a post whose text is
# identical to another is still a distinct row and every timestamp in the derived Corpus is
# a value this project chose rather than observed — stated here because the Corpus file
# carries no field for "this was invented", and the alternative is a reader assuming these
# were real accounts.
_EXTERNAL_ACCOUNT = "external"
_EXTERNAL_INSTANT = datetime(2026, 1, 1, tzinfo=UTC)

# Every corpus item is a body and no title. The readers take both fields, and leaving the
# title empty is the honest shape of a document that was never titled.
_EXTERNAL_TITLE = ""


@dataclass(frozen=True, slots=True)
class SignalRate:
    """How often one Signal fires, counted over each class of the external label.

    Printed for all six published Signals rather than only the ones that fire, so a reader
    can see the three that are uncomputable here are reported as uncomputable and not as
    zero. A Signal that cannot be computed on this data and a Signal that never fires look
    identical in a table of rates, and only one of them is a finding.
    """

    signal: SignalName
    measurable: bool
    deceptive_hits: int
    deceptive_rows: int
    truthful_hits: int
    truthful_rows: int

    @property
    def deceptive_rate(self) -> float:
        return self.deceptive_hits / self.deceptive_rows if self.deceptive_rows else 0.0

    @property
    def truthful_rate(self) -> float:
        return self.truthful_hits / self.truthful_rows if self.truthful_rows else 0.0


@dataclass(frozen=True, slots=True)
class ExternalValidation:
    """Everything the report prints, computed from one pass over the cached bytes."""

    rows: int
    deceptive: int
    truthful: int
    auc: float
    mean_deceptive_score: float
    mean_truthful_score: float
    median_deceptive_score: float
    median_truthful_score: float
    deceptive_at_zero: int
    truthful_at_zero: int
    signals: tuple[SignalRate, ...]
    placements_deceptive: tuple[tuple[str, int], ...]
    placements_truthful: tuple[tuple[str, int], ...]
    scores: tuple[tuple[int, int, int], ...]


def as_corpus(rows: Sequence[Labelled]) -> tuple[CorpusItem, ...]:
    """The external rows as Content Items, so the readers that take a Corpus can take these.

    One item per row, keyed on the row's index, carrying no link because the set has none.
    This is a shape for the readers rather than a claim that these are posts: a job
    advertisement is not a Reddit submission, it has no subreddit to be in, and inventing one
    would put a value into the file that no reader could distinguish from an observed one.
    """
    return tuple(
        CorpusItem(
            post_id=f"ext_{row.index:04d}",
            account=_EXTERNAL_ACCOUNT,
            subreddit=_EXTERNAL_TITLE,
            title=_EXTERNAL_TITLE,
            body=row.text,
            created_at=_EXTERNAL_INSTANT.strftime("%Y-%m-%dT%H:%M:%SZ"),
            links=(),
        )
        for row in rows
    )


def render_report(validation: ExternalValidation, facts: ExternalFacts) -> str:
    """The reader-facing page, generated from the figures rather than written beside them.

    Leads with what the Policy Score does on real text, which on this set is close to
    nothing, and says so before any of the reasons. A validation page whose headline was
    the mean score gap — 0.63 against 0.03, which looks like a twentyfold difference — would
    be leading with the one number here that is misleading, because 58 of the 60 deceptive
    rows sit at zero with everything else and the gap is carried by the other two.
    """
    auc = validation.auc
    return f"""# External validation

Generated by `rfi external-validation` from `{facts.path}`. Do not edit it by hand — a test
holds this file to what that produces, and re-running the command rewrites it byte for byte.
The set itself is not in this repository: run `rfi fetch-external` to write it, and check
the SHA-256 below against what arrives before reading a number here as recomputed.

**These are not the project's own figures.** Every number below is measured on content this
project did not write, carrying labels this project did not attach, using readers whose
weights and phrase lists were published before these bytes were downloaded. Nothing here is
fitted, tuned, or thresholded against. It is reported in its own file and its own section
so that it is never read as an adversarial holdout result or merged with one.

## What was measured

| | |
|---|---|
| Set | DIFrauD job advertisements, `job_scams/test.jsonl` |
| Licence | MIT, used unchanged |
| Rows | {facts.rows} |
| Labelled fraudulent by others | {facts.deceptive} |
| Labelled not fraudulent | {facts.truthful} |
| SHA-256 | `{facts.sha256}` |
| Trained on | no |

## The headline

**The Policy Score does not separate this set.** Its area under the ROC curve against the
external fraud label is **{auc:.4f}**, against the 0.5 a score that orders nothing would
reach. Read plainly: on real job advertisements, the Policy Score ranked the fraudulent
ones no better than the legitimate ones.

That is a finding about the readers, not a defect in the measurement, and it is reported as
the first thing on the page because it is the claim a reader would otherwise have to infer
from the mean scores below.

| | Deceptive | Truthful |
|---|---:|---:|
| Rows | {validation.deceptive} | {validation.truthful} |
| Mean Policy Score | {validation.mean_deceptive_score:.2f} | {validation.mean_truthful_score:.2f} |
| Median Policy Score | {validation.median_deceptive_score} | {validation.median_truthful_score} |
| Rows scoring zero | {validation.deceptive_at_zero} | {validation.truthful_at_zero} |

The mean gap looks decisive and is not. **{validation.deceptive_at_zero} of the
{validation.deceptive} fraudulent rows score zero**, and the truthful median is zero as well,
so nearly every row on both sides is tied at the floor and the AUC above is dominated by
those ties. The 0.63 against 0.03 is two rows carrying it. A reader who took the means and
concluded the Policy Score separates this set would be wrong, which is why they are printed
beneath the AUC rather than above it.

## Per Signal

| Signal | Measurable here | Fraudulent rows carrying it | Not-fraudulent rows carrying it |
|---|---|---:|---:|
{_signal_table(validation.signals)}

The three link Signals are **unmeasurable**, not zero. Every row of this set is a body of
text and a label with no URL in it, so whether `domain_frequency`, `domain_lookalike` or
`category_conflict` would have fired is a question this set cannot answer. They are reported
as unmeasurable rather than at a rate of zero precisely so that they cannot be mistaken for
findings — and because the score in the table above is therefore not the whole rules engine
having run, but the content half of it.

Of the three Signals that *can* be measured, `guaranteed_return` fired on 2 fraudulent rows
and 2 others. The other two fired on nothing at all. The phrase lists behind them were
written against planted scam prose; real job advertisements turn out not to look like that,
which is the honest reading of the table.

## Where the text was placed

| Scam Category | Fraudulent | Not fraudulent |
|---|---:|---:|
{_placement_table(validation.placements_deceptive, validation.placements_truthful)}

`Work and Payroll` takes a similar share of both classes, because nearly every job
advertisement is about a job. The category reader is answering "is this an advert for work",
which is true of the fraudulent and the legitimate alike, and on this set that question does
not separate them. That is a limit of the taxonomy rather than of the placement, and it is
the same limit the CAFC `Other` bucket reports: a large `Other` share is a finding about
coverage.

## What this set cannot establish

It is worth being blunt about the size of this, because a validation page is only as useful
as its refusals.

* **It says nothing about Reddit.** These are job advertisements from a job board between
  2012 and 2014. They are promotional content of the same general kind the pipeline reads,
  not the content it reads. No figure here transfers to Reddit without evidence, and none is
  offered.
* **It says nothing about whether content is fraudulent.** The Policy Score is a triage
  severity for a reviewer (ADR-0002). Measuring it against somebody else's fraud label asks
  whether it orders such content usefully, and the answer here is that it does not. That is
  not the same question as deciding that content is fraudulent, and this page does not claim
  to have answered it.
* **It cannot measure the campaign machinery at all.** A job advertisement has no account
  history, no shared registrable domain with another advertisement, and no multi-account
  structure. The grouping, the recovery rate, the false-grouping rate and the Cohesion Score
  are all unmeasurable here, and none of them is measured here. The project's headline claim
  — recovering planted campaigns — is untouched by this page and also unevidenced by it.
* **It is a spam-adjacent corpus, and the ticket that asked for this warned of exactly
  that.** Story #69 asks that external results state what they cannot establish "so that a
  spam corpus is not read as scam validation". Job advertisements are not scam posts. The
  result above is a statement about promotional text that happens to be labelled, and it is
  reported as a near-null.

## What it does establish

One negative thing, which is still a thing: the readers this project ships were authored
against planted prose, and on real promotional text from an independent source the Policy
Score is at chance. Every figure this project publishes that is measured on its own Corpus
is bounded by that. A reviewer should read `docs/campaign-recovery.md` as a statement about
the method over a Corpus built to exercise it, and not as evidence that the method works on
content it did not write.
"""


def _signal_table(signals: Sequence[SignalRate]) -> str:
    lines = []
    for rate in signals:
        if rate.measurable:
            deceptive = f"{rate.deceptive_rate:.4f}"
            truthful = f"{rate.truthful_rate:.4f}"
        else:
            deceptive = "unmeasurable"
            truthful = "unmeasurable"
        lines.append(
            f"| `{rate.signal.value}` | {'yes' if rate.measurable else 'no — no URLs in '
            'this set'} | {deceptive} | {truthful} |"
        )
    return "\n".join(lines)


def _placement_table(
    deceptive: Sequence[tuple[str, int]], truthful: Sequence[tuple[str, int]]
) -> str:
    counts = dict(truthful)
    names = list(dict.fromkeys([name for name, _ in deceptive] + list(counts)))
    lines = []
    for name in names:
        lines.append(
            f"| {name} | {dict(deceptive).get(name, 0)} | {counts.get(name, 0)} |"
        )
    return "\n".join(lines)


def validate(
    rows: Sequence[Labelled], scores: Sequence[PostScore]
) -> ExternalValidation:
    """Measure the readers against somebody else's labels.

    Takes the scores rather than computing them, so the Policy Score arrives by the same
    `rfi policy-score` path the Corpus uses and this module cannot drift into a second
    scoring rule. Checked against the rows by identifier rather than trusted: a score set
    that is not this row set would compare one population's scores with another's labels,
    which is the one mistake a validation figure must not make.
    """
    if len(scores) != len(rows):
        raise ValueError(
            f"the external set holds {len(rows)} rows and the Policy Score published "
            f"{len(scores)}, so the figure would pair one set of texts with another's scores"
        )
    by_id = {score.post_id: score for score in scores}
    ordered = [by_id[f"ext_{row.index:04d}"] for row in rows]

    deceptive = [score.score for row, score in zip(rows, ordered, strict=True) if row.deceptive]
    truthful = [score.score for row, score in zip(rows, ordered, strict=True) if not row.deceptive]

    return ExternalValidation(
        rows=len(rows),
        deceptive=len(deceptive),
        truthful=len(truthful),
        auc=_ranking_auc(rows, ordered),
        mean_deceptive_score=statistics.fmean(deceptive) if deceptive else 0.0,
        mean_truthful_score=statistics.fmean(truthful) if truthful else 0.0,
        median_deceptive_score=statistics.median(deceptive) if deceptive else 0.0,
        median_truthful_score=statistics.median(truthful) if truthful else 0.0,
        deceptive_at_zero=sum(1 for value in deceptive if value == 0),
        truthful_at_zero=sum(1 for value in truthful if value == 0),
        signals=_signal_rates(rows, ordered),
        placements_deceptive=_placements(rows, deceptive_only=True),
        placements_truthful=_placements(rows, deceptive_only=False),
        scores=tuple(
            (row.index, score.score, row.label) for row, score in zip(rows, ordered, strict=True)
        ),
    )


def _ranking_auc(rows: Sequence[Labelled], scores: Sequence[PostScore]) -> float:
    """Whether the Policy Score orders externally-deceptive ads above the others.

    AUC rather than accuracy because the set is 60 in 1,430 and an accuracy over it would
    be flattered by a score that never fires: saying nothing about everything scores 96%.
    AUC asks the question the Review Queue actually asks — does a higher score correspond
    to an ad somebody else marked fraudulent — and stays defined at that prevalence.
    """
    ranked = [
        (float(score.score), row.deceptive)
        for row, score in zip(rows, scores, strict=True)
    ]
    positives = [value for value, label in ranked if label]
    negatives = [value for value, label in ranked if not label]
    if not positives or not negatives:
        return 0.5
    wins = sum(
        (high > low) + 0.5 * (high == low) for high in positives for low in negatives
    )
    return wins / (len(positives) * len(negatives))


def _signal_rates(
    rows: Sequence[Labelled], scores: Sequence[PostScore]
) -> tuple[SignalRate, ...]:
    """Every published Signal, counted over each class, marked where it cannot be counted.

    All six are reported rather than only the ones that fired, because a Signal that cannot
    fire on this data and a Signal that fires on nothing look identical in a table of rates,
    and only the second of those is a finding. The three link Signals are marked
    unmeasurable rather than given a rate of zero: every row here is text and a label with
    no URL in it, so whether they would have fired is a question this set cannot answer, and
    zero would answer it wrongly.
    """
    seen: dict[SignalName, tuple[int, int]] = {}
    for row, score in zip(rows, scores, strict=True):
        for hit in score.signals:
            deceptive, truthful = seen.get(hit.signal, (0, 0))
            seen[hit.signal] = (
                deceptive + (1 if row.deceptive else 0),
                truthful + (0 if row.deceptive else 1),
            )
    deceptive_rows = sum(1 for row in rows if row.deceptive)
    truthful_rows = sum(1 for row in rows if not row.deceptive)
    return tuple(
        SignalRate(
            signal=signal,
            measurable=signal not in _LINK_SIGNALS,
            deceptive_hits=seen.get(signal, (0, 0))[0],
            deceptive_rows=deceptive_rows,
            truthful_hits=seen.get(signal, (0, 0))[1],
            truthful_rows=truthful_rows,
        )
        for signal in SignalName
    )


_LINK_SIGNALS = frozenset(
    {SignalName.DOMAIN_FREQUENCY, SignalName.DOMAIN_LOOKALIKE, SignalName.CATEGORY_CONFLICT}
)


def _placements(
    rows: Sequence[Labelled], *, deceptive_only: bool
) -> tuple[tuple[str, int], ...]:
    """The Scam Category tally, counted over one class of the external label.

    Placed by the same `place()` every command uses, so the tally cannot disagree with the
    Corpus composition about where a piece of text lands. `Other` is included and sorts
    last, for the reason `Association` gives: a post no list matched makes no claim about
    itself, and on real text most of it makes none.
    """
    tallies: dict[str, int] = {}
    for post, row in zip(as_corpus(rows), rows, strict=True):
        if row.deceptive is not deceptive_only:
            continue
        placed: Placed = place(post)
        tallies[placed.scam_category] = tallies.get(placed.scam_category, 0) + 1
    return tuple(
        sorted(
            tallies.items(),
            key=lambda entry: (entry[0] == OTHER.name, -entry[1], entry[0]),
        )
    )