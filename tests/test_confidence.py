"""The Confidence: a classifier's probability, held internally and never displayed.

Seam under test: `rfi confidence` and the pure functions behind it — the features read off
one post, the fit, the leave-one-campaign-out folds, the metrics, the two baselines, the
published file, and the report. Nothing here inspects the code that wrote them.

The load-bearing claims, in the order the ticket states them:

* Every Content Item has a Confidence, in a file of its own, whose vocabulary is not the
  Policy Score's vocabulary and cannot be made into it.
* The Confidence never appears in user-facing output. The Review Queue is rendered here and
  searched for the word, for the file, and for every published probability; the Campaign
  Candidate report is held structurally, because rendering one needs the stored vectors and
  `tests/test_confidence_store.py` is where this repository keeps the half that does.
* No arithmetic anywhere in `src/` combines the Confidence with the Policy Score, and the
  test that says so walks the AST of every module rather than grepping for a word.
* The model is a published choice over published features, and its performance is measured
  out of fold rather than assumed, beside a constant baseline and the Policy Score's own
  ordering of the same posts.
* Calibration is measured over a stated binning and reported as a finding in its own voice —
  which on this Corpus is a finding that the Confidence is not calibrated, and the report
  says so rather than printing a number that hides it.
* The calibration figure is the Confidence's own question. It is computed from the
  probabilities and the labels and nothing else, it is not evidence about whether the Policy
  Score is sensible, and neither number is corrected toward the other.
* The training labels come from the Planted Campaign membership, the generator wrote them,
  and the report says so in a section of its own.

This half needs no database. The features are read from the Corpus, the published Public
Suffix List, and the same Contact Point reader `rfi campaign-candidates` uses; the fold
assignment is arithmetic; the fit is arithmetic.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path

import pytest

from reddit_fraud_intelligence.campaigns import render_table as render_candidates
from reddit_fraud_intelligence.cli import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_SCORES_PATH,
    DEFAULT_SUFFIX_LIST_PATH,
    DEFAULT_TRUTH_PATH,
    main,
)
from reddit_fraud_intelligence.confidence import (
    CALIBRATION_BINS,
    CALIBRATION_METHOD,
    CALIBRATION_TOLERANCE,
    FEATURES,
    MODEL_NAME,
    _PRECISION,
    Bin,
    Confidence,
    ConfidenceFacts,
    Confidences,
    Feature,
    Figure,
    Fit,
    Sources,
    _is_better,
    calibration,
    scored,
    confident,
    confidence_features,
    folds,
    label_for,
    label_set,
    measure,
    read_confidences,
    render_report,
    render_table,
    train,
)
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.review_queue import build_queue, render_table as render_queue
from reddit_fraud_intelligence.truth import read_truth

REPO_ROOT = Path(__file__).parent.parent
SOURCE = REPO_ROOT / "src" / "reddit_fraud_intelligence"
COMMITTED_CANDIDATES = REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl"
COMMITTED_SHARED_HOSTS = REPO_ROOT / "data" / "infrastructure" / "shared-hosts.jsonl"
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_LIST = REPO_ROOT / "data" / "public-suffix" / "public_suffix_list.dat"
COMMITTED_REPORT = REPO_ROOT / "docs" / "confidence.md"
COMMITTED_SCORES = REPO_ROOT / "data" / "signals" / "policy-scores.jsonl"
COMMITTED_TRUTH = REPO_ROOT / "data" / "corpus" / "truth.jsonl"
COMMITTED_CONFIDENCES = REPO_ROOT / "data" / "model" / "confidences.jsonl"

Row = Mapping[str, object]


def rows(path: Path) -> list[Row]:
    """A committed JSON Lines file, as the objects it holds."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} should be text, is {value!r}"
    return value


def number(row: Row, field: str) -> float:
    value = row[field]
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"{field} should be a number, is {value!r}"
    )
    return float(value)


def items_by_id() -> Mapping[str, CorpusItem]:
    return {item.post_id: item for item in read_corpus(COMMITTED_CORPUS)}


def _committed() -> Sources:
    """The four committed inputs, at this file's absolute paths.

    Absolute because every comparison here is against a committed file by absolute path, and the
    figures under test are numbers rather than the bytes of a generated report — the byte
    comparisons use `run`, which takes the command's own relative defaults for exactly the
    reason `test_campaign_recovery.py` does.
    """
    return Sources(
        corpus=COMMITTED_CORPUS,
        suffix_list=COMMITTED_LIST,
        truth=COMMITTED_TRUTH,
        scores=COMMITTED_SCORES,
    )


def measured() -> Confidences:
    """One run over the committed inputs, which every test here reads."""
    return confident(_committed())


def published_rows() -> Sequence[Confidence]:
    return read_confidences(COMMITTED_CONFIDENCES)


def _views(run: Confidences) -> tuple[tuple[str, str], ...]:
    """The console output and the report, each with the name the assertions name it by.

    Both views of one run, in the order the console output is checked before the report, so
    that every test asking "do both views say this?" asks it the same way. The claim those
    tests make is about the two views agreeing with each other, and a loop that named them
    inline would be a third place to fix when a name changed.
    """
    return (
        (render_table(run), "the console output"),
        (render_report(run), "the report"),
    )


def run(
    directory: Path,
    corpus: Path = DEFAULT_CORPUS_PATH,
    list_path: Path = DEFAULT_SUFFIX_LIST_PATH,
    truth: Path = DEFAULT_TRUTH_PATH,
    scores: Path = DEFAULT_SCORES_PATH,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> tuple[Path, Path, str]:
    """Run `rfi confidence`, writing both outputs into `directory`.

    The four inputs default to the command's own defaults rather than to this file's absolute
    paths, because the report prints the paths it was given: a run over absolute paths would
    write a report with this machine's directory names in it, and that report could not be held
    to the committed bytes — which is the point of holding it.
    """
    confidences = directory / "confidences.jsonl"
    report = directory / "confidence.md"
    exit_code = main(
        [
            "confidence",
            "--corpus", str(corpus),
            "--list", str(list_path),
            "--truth", str(truth),
            "--scores", str(scores),
            "--confidences", str(confidences),
            "--report", str(report),
        ]
    )
    assert exit_code == 0
    return confidences, report, capsys.readouterr().out if capsys is not None else ""


# --- the vocabulary ------------------------------------------------------------------


def test_every_content_item_has_a_confidence_in_a_file_of_its_own() -> None:
    """One row per post, in the Confidence vocabulary and no other.

    The separation ADR-0003 asks for is a field list, so this checks the field list: a
    published row is the post, the account, the probability, the model and the recipe
    digest, and `read_confidences` refuses a row holding anything else. That refusal is
    what would stop a Policy Score being added to this file later and the two numbers
    starting to sit beside each other in one record.
    """
    written = rows(COMMITTED_CONFIDENCES)
    vocabulary = {field.name for field in fields(Confidence)}

    assert len(written) == len(read_corpus(COMMITTED_CORPUS))
    assert {text(row, "post_id") for row in written} == set(items_by_id())
    for row in written:
        assert set(row) == vocabulary, f"a Confidence row holds {sorted(row)}"
    assert not vocabulary & {"score", "points", "weight", "published_points", "signals"}


def test_the_confidence_is_a_probability_and_never_a_figure_out_of_a_hundred() -> None:
    """Strictly between 0 and 1, which is what a probability is and what 0-100 is not.

    Both bounds are strict because a fitted logistic model reaches neither, so a row that
    did would be a fitted number that had gone somewhere it should not. Printing one as an
    integer out of a hundred would be a severity wearing the model's name, which is the
    failure ADR-0003 records.
    """
    for row in published_rows():
        assert 0.0 < row.confidence < 1.0


def test_the_file_records_the_model_and_the_digest_of_its_recipe() -> None:
    """The same facts the Content Embedding record carries, for the same reason.

    A probability from one model is not comparable with a probability from another, so the
    name and the recipe digest travel on every row. The digest is the half a version number
    cannot be on its own: editing the recipe without bumping the name stops the run rather
    than quietly reusing numbers the new recipe would not have produced.
    """
    written = rows(COMMITTED_CONFIDENCES)
    digests = {text(row, "recipe") for row in written}

    assert {text(row, "model") for row in written} == {MODEL_NAME}
    assert len(digests) == 1
    assert len(digests.pop()) == 64


def test_nothing_but_the_probability_and_its_provenance_is_published() -> None:
    """No fold, no label, no membership and no feature values beside each probability.

    A label is the reason most of all: a Confidence row carrying `planted: true` would be
    the membership in the pipeline's own output (ADR-0008), and a fold number would be the
    same information one step removed, since a fold partitions the posts by the campaign
    they came from.
    """
    forbidden = {"fold", "label", "planted", "campaign", "campaign_id", "features"}

    for row in published_rows():
        assert not forbidden & {field.name for field in fields(Confidence)}


def test_the_file_publishes_no_order_because_a_queue_would_be_a_second_priority() -> None:
    """Post-id order and no rank column.

    A Confidence sorted into a Review Queue would be Triage Priority by another name, and
    Triage Priority is the Policy Score and nothing else (ADR-0020). So the file publishes
    probabilities in post-id order, which is stability, and publishes no judgement about
    which of them a reviewer should read.
    """
    written = [row.post_id for row in published_rows()]

    assert written == sorted(written)
    assert not {"rank", "order", "priority"} & {field.name for field in fields(Confidence)}


# --- the features --------------------------------------------------------------------


def test_every_feature_is_a_count_a_reader_can_recount_from_the_post() -> None:
    """The whole of what the model reads, published, and each figure checked against the Corpus.

    The enum is the contract: a feature name this build does not have cannot be fitted, and
    the order it declares is the order the coefficients print in. What the assertions add is
    agreement with the Corpus rather than self-consistency — the first post of this Corpus
    links one URL, so its `links` is one, and it names no Contact Point, so that is zero.
    """
    first = items_by_id()["syn_p_0005"]
    values = confidence_features(read_corpus(COMMITTED_CORPUS), COMMITTED_LIST)[first.post_id]

    assert FEATURES == tuple(member.value for member in Feature)
    assert len(FEATURES) == 7
    assert len(set(FEATURES)) == len(FEATURES)
    assert set(values) == set(FEATURES)
    assert all(isinstance(count, int) and count >= 0 for count in values.values())
    assert values[Feature.LINKS.value] == len(first.links)
    assert values[Feature.REGISTRABLE_DOMAINS.value] == 1
    assert values[Feature.CONTACT_POINTS.value] == 1, "the handle this post's body names"
    assert values[Feature.WORDS_WITH_DIGITS.value] == 3, "420, 48, and the digits in the handle"

    # The two word counts are checked against the corpus's own definition of a word, which
    # splits on an underscore. `whitespace` disagrees with it by exactly one on this post,
    # because its body names the handle `@syn_northwindhire` and a word is a run of letters
    # and digits with the underscore as a separator — the same rule the embeddings use
    # (ADR-0024), so the two modules count the same word the same way.
    assert values[Feature.TITLE_WORDS.value] == len(re.findall(r"[^\W_]+", first.title))
    assert values[Feature.BODY_WORDS.value] == len(re.findall(r"[^\W_]+", first.body))
    assert values[Feature.BODY_WORDS.value] == len(first.body.split()) + 1, (
        "the one word whitespace counts that this does not is the underscore in the handle"
    )


def test_renaming_every_account_leaves_every_feature_identical(tmp_path: Path) -> None:
    """Structural rather than promised: the features cannot read an account.

    The whole Corpus is re-read with every account renamed, every subreddit swapped and
    every timestamp moved, and the features have to come back identical. They are read from
    the post's own title, its own body and its own links and from the links the Corpus
    resolves, so a model fitted on them cannot be reading the generator's membership through
    a side door (ADR-0007, ADR-0008).
    """
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(
                {
                    "account": f"other_{index}",
                    "body": item.body,
                    "created_at": "2027-02-02T02:02:02Z",
                    "links": list(item.links),
                    "post_id": item.post_id,
                    "subreddit": f"elsewhere_{index}",
                    "title": item.title,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for index, item in enumerate(read_corpus(COMMITTED_CORPUS), start=1)
        ),
        encoding="utf-8",
    )

    assert confidence_features(read_corpus(corpus), COMMITTED_LIST) == confidence_features(
        read_corpus(COMMITTED_CORPUS), COMMITTED_LIST
    )


def test_a_feature_is_never_read_off_another_post() -> None:
    """Each row of the feature map is a function of its own post alone.

    Checked by dropping a post and requiring every other row to be unchanged: a feature that
    counted anything across the Corpus — how many posts reach a registration, say — would
    move every row when one post left, and the Policy Score's `domain_frequency` is exactly
    the Signal that does that (ADR-0007). This model has no such feature, so it cannot
    re-derive the Policy Score from outside.
    """
    items = read_corpus(COMMITTED_CORPUS)
    full = confidence_features(items, COMMITTED_LIST)
    fewer = confidence_features(items[1:], COMMITTED_LIST)

    assert set(fewer) == set(full) - {items[0].post_id}
    assert all(fewer[post_id] == full[post_id] for post_id in fewer)


# --- the model -----------------------------------------------------------------------


def test_the_fit_is_deterministic_and_holds_one_coefficient_per_feature() -> None:
    """Same rows, same labels, same numbers twice over, and a coefficient per feature.

    The model is fitted here by gradient descent at a published step count and a published
    penalty, so it has no seed and nothing to tune at run time. A run whose numbers moved
    between two identical runs would leave a published file that nothing could hold.
    """
    features = confidence_features(read_corpus(COMMITTED_CORPUS), COMMITTED_LIST)
    labelled = label_set(read_truth(COMMITTED_TRUTH))
    training = tuple(
        (tuple(features[item.post_id][name] for name in FEATURES), label_for(labelled, item))
        for item in read_corpus(COMMITTED_CORPUS)
    )

    once = train(training)
    twice = train(training)

    assert once == twice
    assert len(once.coefficients) == len(FEATURES)
    assert once.steps > 0
    assert once.rows == len(training)
    assert once.positives == 7


def test_a_fit_over_one_label_is_refused_rather_than_diverging() -> None:
    """A logistic regression over a single class has no finite answer, so the run stops.

    The coefficients would grow until the step count ran out and the number that came back
    would be published as a Confidence with nothing wrong with it visible. The refusal names
    what happened, because "one label" is the fact a reader needs.
    """
    with pytest.raises(ValueError, match="one label"):
        train((((1, 2), True), ((3, 4), True)))


def test_an_empty_training_set_is_refused_rather_than_reported_as_zero() -> None:
    """A fit over no rows has no Confidence in it, and 0 of 0 is not a figure."""
    with pytest.raises(ValueError):
        train(())


def test_rows_of_two_widths_are_refused() -> None:
    """Two different feature widths in one fit, which no coefficient vector could score."""
    with pytest.raises(ValueError, match="width"):
        train((((1, 2), True), ((3,), False)))


def test_the_fit_refuses_a_row_of_the_wrong_width() -> None:
    """A row that cannot be scored against these coefficients, refused by name.

    Scoring it anyway would pad it with zeros or truncate it, and both produce a number that
    looks like a Confidence and is not one.
    """
    fit = Fit(
        coefficients=(0.1, 0.2),
        intercept=0.0,
        centre=(0.0, 0.0),
        scale=(1.0, 1.0),
        steps=1,
        rows=1,
        positives=1,
    )

    with pytest.raises(ValueError, match="width"):
        fit.confidence((0.1,))


def test_a_positive_coefficient_raises_the_confidence_and_a_negative_one_lowers_it() -> None:
    """The model's own arithmetic, checked against the fit rather than a quoted number.

    A logistic regression whose output moved the other way from its own published
    coefficient would mean the coefficients were not the model's, and a reader checking a
    Confidence against them has no other way to tell.
    """
    fit = Fit(
        coefficients=(0.5, -0.25),
        intercept=0.0,
        centre=(0.0, 0.0),
        scale=(1.0, 1.0),
        steps=1,
        rows=1,
        positives=1,
    )

    assert fit.confidence((1.0, 0.0)) > fit.confidence((0.0, 0.0)) > fit.confidence((0.0, 1.0))


# --- the labels and the folds --------------------------------------------------------


def test_the_labels_are_the_planted_campaign_membership_and_nothing_else() -> None:
    """A post's label is its account's membership, and the membership is the generator's.

    This is the fact the whole page is measured against, so it is asserted where it is read:
    the label set is keyed by campaign over accounts, the seven planted posts are exactly
    the seven the membership names, and every other post of the Corpus is negative.
    """
    items = read_corpus(COMMITTED_CORPUS)
    campaigns = read_truth(COMMITTED_TRUTH)
    labelled = label_set(campaigns)
    planted = {post for campaign in campaigns for post in campaign.posts}

    assert len(labelled) == 2, "two Planted Campaigns, so two folds to hold one out at a time"
    assert sum(1 for item in items if label_for(labelled, item)) == 7
    assert {item.post_id for item in items if label_for(labelled, item)} == planted


def test_every_confidence_is_held_out_of_its_own_campaign() -> None:
    """Leave-one-campaign-out, and the fold assignment is arithmetic rather than a seed.

    Not merely never saw the post — never saw any post of that post's Planted Campaign, so a
    model that had memorised one campaign's vocabulary could not carry it into the other.
    This is the whole measurement claim: every published probability comes from a fit
    trained without its post and without its campaign.

    A negative is assigned to a fold rather than left out of them, so every post is held out
    exactly once and every published Confidence is out of fold. The assignment is by position
    in post-id order over the negatives, which is why it needs no seed.
    """
    items = read_corpus(COMMITTED_CORPUS)
    labelled = label_set(read_truth(COMMITTED_TRUTH))
    plan = folds(items, labelled)

    assert set(plan) == {row.post_id for row in published_rows()}
    assert set(plan.values()) == {0, 1}
    for account in {item.account for item in items}:
        held = {plan[item.post_id] for item in items if item.account == account}
        assert len(held) == 1, "two posts by one account in two folds, so neither is held out"


def test_the_two_baselines_are_a_constant_and_the_policy_scores_own_ordering() -> None:
    """What a classifier has to beat: the base rate, and the rules engine beside it.

    A constant equal to the training base rate is the trivial baseline — it predicts the
    rate and orders nothing, and a classifier that cannot order posts better than it is not
    worth reading. The Policy Score is the second, measured over the same held-out posts
    with the same metric, so the model's contribution is a difference a reader can see
    rather than two numbers from two runs.
    """
    evaluation = measured().evaluation

    assert [figure.name for figure in evaluation.baselines] == [
        "constant base rate",
        "policy score",
    ]
    constant, policy = evaluation.baselines
    assert constant.auc == pytest.approx(0.5)
    assert policy.auc != constant.auc, "the two baselines must not be the same number"
    assert policy.log_loss is None, (
        "a Policy Score has no probability reading, so no proper scoring rule applies to it, "
        "and reporting one would be turning an editorial judgement into a likelihood"
    )
    assert constant.log_loss is not None and constant.brier is not None


def test_the_model_is_measured_out_of_fold_over_every_post_at_once() -> None:
    """One fold per Planted Campaign, every post predicted once, and the report says so.

    The distinction the report has to hold is the one ADR-0022 makes for recovery: a
    training fit over seven positives is perfect by construction and says nothing, so the
    only figure published is the held-out one and the report names which it is rather than
    leaving a reader to infer it from the word `evaluated`.
    """
    measured_out = measured()
    report = COMMITTED_REPORT.read_text(encoding="utf-8")

    assert measured_out.evaluation.held_out is True
    assert measured_out.evaluation.model.name == MODEL_NAME
    assert measured_out.evaluation.verdict, "the three rows are compared, not just printed"
    assert measured_out.evaluation.posts == len(read_corpus(COMMITTED_CORPUS))
    assert measured_out.evaluation.positives == 7
    assert measured_out.evaluation.folds == 2
    assert "held out" in report
    assert "out of fold" in report


def test_the_three_published_metrics_are_the_ones_a_reader_can_recompute() -> None:
    """AUC, log loss and Brier, recomputed here from the published rows rather than quoted.

    Each is a piece of arithmetic over the held-out rows and the labels, so the test does it
    again from the committed Confidence file and requires the report's figures to match. A
    figure no reader could redo would be a claim, and this repository's whole position is
    that its figures can be redone.
    """
    measured_out = measured()
    published = published_rows()
    items = items_by_id()
    labelled = label_set(read_truth(COMMITTED_TRUTH))
    truth = [
        (row.confidence, float(label_for(labelled, items[row.post_id]))) for row in published
    ]
    positives = [probability for probability, actual in truth if actual == 1.0]
    negatives = [probability for probability, actual in truth if actual == 0.0]
    pairs = sum(
        (high > low) + 0.5 * (high == low) for high in positives for low in negatives
    )

    assert measured_out.evaluation.model.auc == pytest.approx(
        pairs / (len(positives) * len(negatives))
    )
    assert measured_out.evaluation.model.brier == pytest.approx(
        sum((probability - actual) ** 2 for probability, actual in truth) / len(truth)
    )
    assert measured_out.evaluation.model.log_loss == pytest.approx(
        sum(
            -(actual * math.log(probability) + (1 - actual) * math.log(1 - probability))
            for probability, actual in truth
        )
        / len(truth)
    )
    assert measured_out.mean_confidence == pytest.approx(
        sum(row.confidence for row in published) / len(published)
    )


# --- the calibration --------------------------------------------------------------


def test_the_calibration_is_measured_over_a_binning_the_output_states() -> None:
    """Ten equal-width bins over the open unit interval, and a post in the bin it falls in.

    The binning is named in the output rather than left in the code, because a calibration
    figure with no binning beside it is a claim: the same probabilities binned five ways
    give five different figures, and a reader cannot redo one they were not told how to
    build. Every bin's own numbers are the arithmetic of the posts inside it, so the whole
    table is checkable against the published file.
    """
    figure = calibration([(0.05, False), (0.09, False), (0.11, True), (0.95, True)])

    assert len(figure.bins) == CALIBRATION_BINS == 10
    assert CALIBRATION_METHOD.startswith("10 equal-width bins over the open unit interval")

    edges = [(one.lower, one.upper) for one in figure.bins]
    assert edges[0] == (0.0, 0.1) and edges[-1] == (0.9, 1.0)
    assert all(high - low == pytest.approx(0.1) for low, high in edges)

    low, high = figure.bins[0], figure.bins[9]
    assert (low.posts, low.planted) == (2, 0)
    assert low.mean_confidence == pytest.approx(0.07)
    assert low.observed == 0.0
    assert low.gap == pytest.approx(0.07)
    assert (high.posts, high.planted) == (1, 1)
    assert high.observed == 1.0

    middle = figure.bins[1]
    assert (middle.posts, middle.planted, middle.observed) == (1, 1, 1.0), (
        "a post at 0.11 belongs to the second bin and nothing else decides that"
    )
    assert figure.posts == 4
    assert figure.positives == 2
    assert figure.base_rate == pytest.approx(0.5)
    assert figure.mean_confidence == pytest.approx((0.05 + 0.09 + 0.11 + 0.95) / 4)


def test_the_calibration_error_is_each_bins_gap_weighted_by_the_posts_in_it() -> None:
    """The expected calibration error, done again here over the same rows.

    A mean over bins weighted by nothing would be a mean over ten numbers, three of which
    are gaps between one post and its own prediction. The weighting is the whole of the
    definition, so the test reproduces it rather than quoting the module's figure.
    """
    rows_in = [(0.02, False), (0.08, False), (0.12, True), (0.42, False), (0.94, True)]
    figure = calibration(rows_in)
    weight = sum(
        one.posts / len(rows_in) * abs((one.mean_confidence or 0.0) - (one.observed or 0.0))
        for one in figure.bins
    )

    assert figure.error == pytest.approx(weight)
    assert figure.error > 0, "these rows are not calibrated, so the error cannot be zero"


def test_an_empty_bin_publishes_its_count_and_no_figures() -> None:
    """No posts means no mean and no observed share, and a dash rather than a zero.

    A bin holding nothing has no average in it, and `0.0000` beside an empty bin is a
    number a reader would add into the column above it. The count is still printed, because
    where the posts are not is half of what the table is for.
    """
    figure = calibration([(0.05, False), (0.95, True)])
    empty = figure.bins[4]

    assert empty.posts == 0
    assert empty.planted == 0
    assert empty.mean_confidence is None
    assert empty.observed is None
    assert empty.gap is None, (
        "an empty bin has no gap to print and contributes none to the error; a 0.0000 there "
        "is a number a reader would add into the column above it"
    )
    assert sum(one.posts for one in figure.bins) == 2


def test_a_calibrated_predictor_is_reported_as_calibrated() -> None:
    """The verdict is computed from the figures, so it can say the opposite of what it says here.

    Four posts the model calls 0.25 and one of which is planted: the bin says 0.25 and the
    bin is 0.25 planted, so the gap is nothing and the figure is calibrated. A verdict
    written into the template rather than computed would say the sentence this page's own
    finding contradicts.
    """
    figure = calibration(
        [(0.25, True), (0.25, False), (0.25, False), (0.25, False)]
    )

    assert figure.error == pytest.approx(0.0)
    assert figure.calibrated is True
    assert "is calibrated" in figure.verdict
    assert "is not calibrated" not in figure.verdict


def test_a_miscalibrated_predictor_is_reported_as_miscalibrated_in_plain_words() -> None:
    """The finding is stated, with the figures that make it and the bin that carries it.

    The acceptance criterion the report exists for: a miscalibrated Confidence has to be
    reported as one rather than as a number a reader has to interpret. So the verdict names
    the verdict, the error against the bar it fails, the direction of the average, and the
    single worst bin — because an average gap is the form in which a bad bin hides.
    """
    figure = calibration([(0.81, False), (0.95, True), (0.04, False), (0.07, True)])

    assert figure.calibrated is False
    assert figure.error > CALIBRATION_TOLERANCE
    assert "is not calibrated" in figure.verdict
    assert f"{figure.error:.4f}" in figure.verdict, "the figure that failed is in the sentence"
    assert f"{CALIBRATION_TOLERANCE:.4f}" in figure.verdict, "so is the bar it failed"
    assert "under" in figure.verdict or "over" in figure.verdict, (
        "a reader is told which way the model is wrong, not only that it is"
    )
    assert figure.worst.span in figure.verdict, (
        "the bin that carries the error is named: an average gap can hide it"
    )


def test_the_worst_bin_is_published_beside_the_average_gap() -> None:
    """Two figures, because one of them is an average and averages hide.

    A model whose bins are mostly right and whose top bin is badly wrong has a small
    expected calibration error and a large worst-bin gap, and publishing only the first
    would be publishing the half that reads well. Both are properties of the object rather
    than of one renderer, so both views print them.
    """
    rows_in = [(0.25, True), (0.25, False), (0.25, False), (0.25, False), (0.90, False)]
    figure = calibration(rows_in)

    assert figure.worst.posts == 1
    assert figure.worst.gap == pytest.approx(0.90)
    assert figure.error == pytest.approx(0.18), (
        "the weighting is what makes the average a fifth of the worst bin rather than the "
        "worst bin itself; a different number means the error is not the weighted mean it "
        "claims to be"
    )
    assert figure.error < (figure.worst.gap or 0.0), (
        "the weighting is what makes the average smaller than the worst bin; if these are "
        "equal, the error is not the weighted mean it claims to be"
    )
    assert figure.worst.gap == max(one.gap or 0.0 for one in figure.bins), (
        "the worst bin is the one furthest from its own prediction and not merely one of them"
    )


def test_the_confidence_is_not_calibrated_on_this_corpus_and_the_run_says_so() -> None:
    """The finding on the committed Corpus, recomputed from the published file.

    Every figure here is arithmetic over `data/model/confidences.jsonl` and the membership,
    so a reader can do it with the two files and a pencil. What the test adds is the
    finding: the model's probabilities run from 1.3e-05 to 0.9981 with a mean of 0.1425
    against a base rate of 0.2059, and its worst bin is a negative it is 0.81 sure about.
    That is a miscalibrated model and the run has to say so in those words.
    """
    measured_out = measured()
    figure = measured_out.evaluation.calibration
    published = published_rows()
    items = items_by_id()
    labelled = label_set(read_truth(COMMITTED_TRUTH))
    truth = [(row.confidence, label_for(labelled, items[row.post_id])) for row in published]

    expected = []
    for index in range(CALIBRATION_BINS):
        inside = [
            (value, label)
            for value, label in truth
            if index / CALIBRATION_BINS <= value < (index + 1) / CALIBRATION_BINS
        ]
        expected.append(
            len(inside) / len(truth)
            * abs(
                sum(value for value, _ in inside) / len(inside)
                - sum(1 for _, label in inside if label) / len(inside)
            )
            if inside
            else 0.0
        )

    assert figure.error == pytest.approx(sum(expected))
    assert figure.mean_confidence == pytest.approx(
        sum(row.confidence for row in published) / len(published)
    )
    assert figure.base_rate == pytest.approx(7 / 34)
    assert figure.calibrated is False, (
        "on this Corpus the Confidence is miscalibrated; if that has changed, the finding the "
        "page leads with is stale and the prose around it needs rewriting rather than the test"
    )
    assert "is not calibrated" in figure.verdict


def test_the_calibration_is_computed_without_reading_the_policy_scores(tmp_path: Path) -> None:
    """Scrambling every Policy Score moves the baseline and moves no calibration figure.

    Calibration is the Confidence's own question and the Policy Score is not an input to it,
    so a run whose scores are in reverse order has to publish the same bins, the same error
    and the same verdict while the baseline that scores feed moves. It is the same
    separation the features are held to, asked again at the measurement: a calibration
    figure fitted to the Policy Score would be the Policy Score's shape reported as a
    probability, arrived at by the other door.
    """
    scrambled = tmp_path / "policy-scores.jsonl"
    scrambled.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "score": 100 - int(number(row, "score")),
                    "points": 130 - int(number(row, "points")),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in reversed(rows(COMMITTED_SCORES))
        ),
        encoding="utf-8",
    )

    honest = measured()
    flipped = confident(replace(_committed(), scores=scrambled))

    assert flipped.evaluation.calibration == honest.evaluation.calibration
    assert [figure.auc for figure in flipped.evaluation.baselines] != [
        figure.auc for figure in honest.evaluation.baselines
    ], "the Policy Score baseline must be the one thing the scores file moves"


def test_both_views_print_the_same_row_for_every_bin() -> None:
    """The two views are one run rendered twice, so no cell of one table may differ from the other.

    A reliability table is a figure a reader recomputes bin by bin, and the console output and
    the report are two renderings of it. A cell one view prints as a dash and the other as a
    zero is two answers to the same cell — and it is exactly the drift a test checking the
    spans and the headline figures cannot see, which is how it survived a first reading.
    """
    one_run = measured()
    lines = render_table(one_run).splitlines() + render_report(one_run).splitlines()

    for one in one_run.evaluation.calibration.bins:
        rows = [
            [token.strip("`") for token in re.split(r"[|\s]+", line) if token]
            for line in lines
            if line.strip().startswith(one.span) or line.strip().startswith(f"| `{one.span}`")
        ]
        assert len(rows) == 2, f"{one.span} is printed {len(rows)} times, not once in each view"
        assert rows[0] == rows[1], (
            f"the console output and the report print different rows for {one.span}: "
            f"{rows[0]} and {rows[1]}"
        )


def test_both_views_print_the_calibration_over_the_binning_they_state() -> None:
    """The figure, the table it is computed from, and the method — in both views.

    A metric printed without the binning behind it is a claim, and a metric printed in one
    view and not the other is two commands' worth of drift. So both the console output and
    the report carry the method, the expected calibration error, the worst bin, and the
    verdict sentence, and each bin's row of the table beside them.
    """
    one_run = measured()
    figure = one_run.evaluation.calibration

    for view, where in _views(one_run):
        joined = " ".join(view.split())
        assert "expected calibration error" in joined, where
        assert f"{figure.error:.4f}" in joined, where
        assert f"{figure.worst.gap:.4f}" in joined, where
        assert "equal-width bins" in joined, f"{where} does not state the binning"
        assert one_run.evaluation.calibration.verdict in joined, where
        for one in figure.bins:
            if one.posts:
                assert one.span in joined, f"{where} prints no row for the bin {one.span}"


def test_both_views_keep_calibration_apart_from_the_policy_score() -> None:
    """The two questions are named as two, and neither is evidence for the other.

    The Policy Score is not a probability and has no calibration to measure: whether it is
    *sensible* is an editorial judgement about published weights, which this figure says
    nothing about. So a miscalibrated Confidence is not an argument against the Policy Score,
    and the Policy Score's AUC of 1.0000 is not an argument that the Confidence is calibrated.
    A report that put the two side by side without saying this is how a miscalibrated model
    gets read as vindicated by a good severity.
    """
    one_run = measured()

    for view, where in _views(one_run):
        joined = " ".join(view.split())
        assert "not a probability" in joined, f"{where} does not say what the Policy Score is"
        assert "calibration" in joined and "Policy Score" in joined, where
        assert "says nothing about" in joined, (
            f"{where} puts the two figures together without saying that neither is evidence "
            "for the other"
        )


def test_both_views_say_the_confidence_is_still_displayed_nowhere() -> None:
    """The calibration figure changes where the number is displayed: it does not.

    A miscalibrated model is an argument for not showing it, never for showing it, and the
    one thing this measurement must not do is move the number onto a screen. So both views
    say the Confidence stays in its own file whatever the figure above it says — and the
    structural half of the claim, the import graph that keeps `rfi review-queue` from
    reaching this module, is unchanged by anything here.
    """
    one_run = measured()

    for view, where in _views(one_run):
        joined = " ".join(view.split())
        assert "displayed nowhere" in joined, where
        assert "whatever the" in joined, (
            f"{where} does not tie the calibration figure to where the number is displayed"
        )
    assert render_queue(build_queue(COMMITTED_SCORES, COMMITTED_CANDIDATES, 50)).lower().count(
        "calibrat"
    ) == 0, "the Review Queue is not where a finding about the Confidence belongs"


def test_every_published_probability_falls_in_exactly_one_bin() -> None:
    """Every published Confidence is inside the open unit interval rather than on an edge.

    A probability that landed on a bin boundary would be in two bins or none, and a bin the
    run and the page counted differently would be a figure neither could be checked on. The
    published file is read here rather than the run's own rows, so what is checked is the
    file the report's table is derived from.
    """
    published = published_rows()
    figure = calibration([(row.confidence, False) for row in published])
    edges = [one.lower for one in figure.bins] + [figure.bins[-1].upper]

    assert all(0.0 < row.confidence < 1.0 for row in published)
    assert all(edges[index] < edges[index + 1] for index in range(len(edges) - 1))
    assert sum(one.posts for one in figure.bins) == len(published)
    assert edges[0] == 0.0 and edges[-1] == 1.0


# --- the three refusals --------------------------------------------------------------


def test_the_review_queue_never_prints_a_confidence() -> None:
    """The user-facing queue names no Confidence, in any form it could be printed in.

    Rendered from the committed published files and then searched: for the word, for the
    file, and for every published probability in the two formats a reviewer would see — four
    decimal places as a probability, and as a whole number out of a hundred as a severity. A
    queue is the one output a reviewer acts on, and a probability sitting in it beside the
    Policy Score is the fused score ADR-0003 rules out.

    The severity form is checked as a whole number rather than a bare digit, because a bare
    digit would collide with the counts every table on the page carries and the test would
    fail on a coincidence rather than on anything about the Confidence.
    """
    queue = build_queue(COMMITTED_SCORES, COMMITTED_CANDIDATES, 50)
    printed = render_queue(queue)

    _refuses_every_confidence(printed, "the Review Queue")

    # The stronger form of "never displayed as a severity": every figure the queue prints out
    # of a hundred is a published Policy Score, and nothing else. Checked against the scores
    # file rather than against the Confidence file, because the claim is about what the severities
    # on this page are — a Confidence printed as a severity could not be spotted by looking for
    # Confidence-shaped numbers, only by establishing that every severity is accounted for.
    severities = {int(found) for found in re.findall(r"(\d+)/100", printed)}
    published_scores = {int(number(row, "score")) for row in rows(COMMITTED_SCORES)}

    assert severities == published_scores, (
        f"the queue prints severities {sorted(severities)}, and the Policy Scores publish "
        f"{sorted(published_scores)}: a figure on the page that is neither is a number whose "
        "provenance the page does not state"
    )


def test_the_campaign_candidate_report_never_prints_a_confidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second of the two reports the ticket names, rendered and searched like the first.

    The grouping needs the stored vectors and so a database, which is why this half of the check
    lives beside `tests/test_campaign_candidates.py` rather than here: a test that skips would
    leave the acceptance criterion resting on the Review Queue alone.
    """
    from reddit_fraud_intelligence.campaigns import group as group_accounts

    try:
        grouping = group_accounts(
            COMMITTED_CORPUS,
            COMMITTED_LIST,
            COMMITTED_SHARED_HOSTS,
        )
    except ValueError as refusal:
        # The grouping's own refusals about a database are the two cases this check cannot be
        # made over: no vectors stored, or no extension. Both mean the report could not be
        # rendered here, and skipping with the reason says so rather than passing quietly.
        if "no vectors" in str(refusal) or "not activated" in str(refusal):
            pytest.skip(f"the stored vectors are not readable on this machine: {refusal}")
        raise
    printed = render_candidates(grouping)
    capsys.readouterr()

    _refuses_every_confidence(printed, "the Campaign Candidate report")


def test_the_campaign_candidate_report_cannot_reach_a_confidence_at_all() -> None:
    """And structurally, so the check above cannot be the only thing standing between them.

    `campaigns.py` cannot import this module, so there is no code path by which a Confidence
    could reach that report even if a future edit wanted one. Checked over both spellings of an
    import — `from ... import confidence` and `import ...confidence` — because checking one is a
    test that passes vacuously against the other.
    """
    reached = sorted(
        path.name
        for path in sorted(SOURCE.glob("*.py"))
        if path.name in {"campaigns.py", "review_queue.py", "signals.py", "evaluation.py"}
        and any(
            name.endswith("confidence") for name in _imported_modules(path)
        )
    )

    assert reached == [], (
        f"a module that renders user-facing output imports the Confidence: {reached}"
    )


def test_no_arithmetic_in_this_repository_combines_the_two_numbers() -> None:
    """The strongest form of "never summed": no expression in `src/` adds the two.

    Walked as an AST rather than grepped for a word, because the failure being ruled out is
    `score + confidence` or `confidence * 0.1 + score` — arithmetic mentioning both names —
    and a grep would pass over `points + probability` just as happily. Every `BinOp` in
    every module is inspected, and none may carry a confidence-named thing on one side and a
    Policy-Score-named thing on the other.
    """
    a_confidence = re.compile(r"confiden|probabilit|likelihood", re.IGNORECASE)
    a_policy_score = re.compile(r"score|points|weight|policy", re.IGNORECASE)
    offenders: list[str] = []
    expressions = 0

    for path in sorted(SOURCE.glob("*.py")):
        for node in ast.walk(ast.parse(_source(path))):
            if isinstance(node, ast.BinOp):
                expressions += 1
                expression = ast.unparse(node)
                if a_confidence.search(expression) and a_policy_score.search(expression):
                    offenders.append(f"{path.name}:{node.lineno} {expression}")

    assert not offenders, "arithmetic combining the two numbers: " + "; ".join(offenders)
    # The walk is only worth anything if it reaches expressions, and a walk that has silently
    # stopped matching would report the same clean result. This Corpus has hundreds of them,
    # and at least one carries a Confidence-named thing in it, so a run that finds neither is a
    # walk that has broken rather than a repository that is clean.
    assert expressions > 100, f"the walk inspected only {expressions} expressions; it is broken"


def test_the_features_do_not_depend_on_the_policy_scores(tmp_path: Path) -> None:
    """Scrambling every Policy Score changes the baseline and changes nothing else.

    The scores file is opened for one reason: to order the same posts as a baseline to read the
    model's figure against. If a score could reach a feature, the Confidence would be the Policy
    Score with a decimal point on it, and this is what says it cannot — a run whose scores are
    in reverse order publishes the same probabilities, while the baseline the scores feed moves.
    """
    scrambled = tmp_path / "policy-scores.jsonl"
    scrambled.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "score": 100 - int(number(row, "score")),
                    "points": 130 - int(number(row, "points")),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in reversed(rows(COMMITTED_SCORES))
        ),
        encoding="utf-8",
    )

    honest = confident(_committed())
    flipped = confident(replace(_committed(), scores=scrambled))

    assert [row.confidence for row in flipped.rows] == [row.confidence for row in honest.rows]
    assert flipped.evaluation.model == honest.evaluation.model
    assert [figure.auc for figure in flipped.evaluation.baselines] != [
        figure.auc for figure in honest.evaluation.baselines
    ], "the Policy Score baseline must be the one thing the scores file moves"
    assert flipped.fits == honest.fits, "the fitted weights must not read a score either"


def test_the_confidence_module_reads_no_policy_score_rule() -> None:
    """The Policy Score is not an input, checked as an import rather than as a habit.

    Not merely "no Signal is used": `confidence.py` does not import `signals` at all, so the
    Policy Score cannot become an input by accident either. A model given the Policy Score as a
    feature returns the Policy Score with a decimal point on it, and ADR-0003's separation would
    become a matter of arithmetic happening not to fire.
    """
    imports = _imported_modules(SOURCE / "confidence.py")

    assert not [name for name in imports if name.endswith("signals")]
    assert not [
        name
        for name in imports
        if name.rpartition(".")[2] in {"read_policy_scores", "score_corpus", "PostScore"}
    ]


def test_only_the_cli_reaches_the_confidence_so_no_other_command_can_print_one() -> None:
    """The import graph says it, and a reader can check it without running anything.

    A module that renders user-facing output and cannot import this one has no way to print
    a Confidence, whatever the ticket asks of it later.
    """
    importers = sorted(
        path.name
        for path in sorted(SOURCE.glob("*.py"))
        if path.name != "confidence.py"
        and any(module.endswith("confidence") for module in _imported_modules(path))
    )

    assert importers == ["cli.py"], f"only cli.py may reach the Confidence; found {importers}"


def test_the_report_states_that_the_labels_come_from_the_generator() -> None:
    """The report says where the labels came from, in a section of its own.

    This is the acceptance criterion the page exists to protect: a reader who took one of
    these figures for a rate of fraud found in the world would have been told something
    false about how it came about. So the words are checked for, not the intention behind
    them.
    """
    report = COMMITTED_REPORT.read_text(encoding="utf-8")

    # Joined, because the report wraps at its own width and the sentence a reader reads is the
    # sentence regardless of where the line fell.
    joined = " ".join(report.split())

    assert "## What these figures are not" in report
    assert "Planted Campaign membership" in joined
    assert "recovery of planted structure" in joined
    assert "not a rate of fraud found in anything" in joined
    assert "Nothing here has been reviewed by a person" in joined, (
        "the generator assigned every label in this Corpus, and a page that did not say so "
        "would leave a reader believing a person had"
    )


# --- the published file --------------------------------------------------------------


def test_a_confidences_file_carrying_a_policy_score_field_is_refused(tmp_path: Path) -> None:
    """A row with `score` beside `confidence` is refused by name rather than ignored.

    Silently dropping it would leave the file still publishing the Policy Score through a
    field nothing checks, which is the failure `read_corpus` exists to prevent on the Corpus
    side (ADR-0008).
    """
    path = tmp_path / "confidences.jsonl"
    path.write_text(json.dumps(_row(extra={"score": 65})) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="score"):
        read_confidences(path)


def test_a_confidence_outside_the_unit_interval_is_refused(tmp_path: Path) -> None:
    """A row claiming 0, 1 or 65 is refused; one claiming 0.5 is not.

    The interval is a fact about the model and about this file's purpose at once: a number
    outside it is not a Confidence, and a file publishing one would be publishing a severity
    under the model's name.
    """
    path = tmp_path / "confidences.jsonl"

    path.write_text(json.dumps(_row(extra={"confidence": 0.5})) + "\n", encoding="utf-8")
    assert read_confidences(path)[0].confidence == 0.5

    for value in (0.0, 1.0, 65, -0.1):
        path.write_text(json.dumps(_row(extra={"confidence": value})) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="confidence"):
            read_confidences(path)


def test_two_rows_for_one_post_are_refused(tmp_path: Path) -> None:
    """One post on two rows, so a later reader takes whichever it read last.

    The same refusal as every other published file in this project: a duplicate is not a
    formatting matter, it is two answers to one question with nothing to say which is which.
    """
    path = tmp_path / "confidences.jsonl"
    path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="on two rows each"):
        read_confidences(path)


def test_rows_disagreeing_about_the_model_are_refused(tmp_path: Path) -> None:
    """Two models' probabilities in one file, and a comparison between them means nothing.

    The same refusal `read_content_records` makes for two models' vectors: a probability is
    comparable only with probabilities from the same model under the same recipe.
    """
    path = tmp_path / "confidences.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_row(post_id=f"syn_p_{index:04d}", extra={"model": model}))
            for index, model in enumerate(("model-a", "model-b"), start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models"):
        read_confidences(path)


def test_rows_disagreeing_about_the_recipe_are_refused(tmp_path: Path) -> None:
    """Two recipes under one model name, which is the near miss a name cannot catch.

    Two rows fitted under different rules and published under one version number is exactly
    what the recipe digest exists to stop, and a reader comparing the two probabilities would
    have nothing in the file to notice it.
    """
    path = tmp_path / "confidences.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_row(post_id=f"syn_p_{index:04d}", extra={"recipe": recipe}))
            for index, recipe in enumerate(("a" * 64, "b" * 64), start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recipes"):
        read_confidences(path)


def test_a_recipe_that_is_not_a_digest_is_refused(tmp_path: Path) -> None:
    """A recipe field holding prose rather than a SHA-256, refused as the shape it should be.

    Same check the Content Embedding record makes, for the same reason: the digest is a
    promise that can be checked, and a sentence in its place is a promise that cannot.
    """
    path = tmp_path / "confidences.jsonl"
    path.write_text(json.dumps(_row(extra={"recipe": "the recipe"})) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recipe"):
        read_confidences(path)


# --- the two views -------------------------------------------------------------------


def test_the_console_output_fits_the_width_and_the_ascii_it_claims() -> None:
    """The two claims this command's own docstrings make about its console output.

    Its `render_table` docstring says ASCII only, so it prints the same way on a console that
    cannot encode anything else, and `_WIDTH` says every line fits a hundred columns. Both are
    checkable, so both are checked — a width nobody checks is a width that drifts, and the first
    thing to drift it is a table of coefficients with one column too many.

    **Lines naming a path are excluded, as `test_review_queue.py` excludes them.** The figures
    block prints the path each file was read from, so the width of those lines is the length of
    whatever path the caller passed. A check that included them would be a check on the length
    of this machine's directory: it passed over a 25-character root and failed over CI's 78,
    with nothing about the command having changed. That is not hypothetical — it is what this
    test did before the exclusion was added.
    """
    printed = render_table(measured())
    lines = printed.splitlines()

    not_ascii = [line for line in lines if not line.isascii()]
    too_wide = [
        (len(line), line)
        for line in lines
        if len(line) > 100 and not _names_a_path(line)
    ]

    assert not not_ascii, f"the console output is not ASCII: {not_ascii}"
    assert not too_wide, f"the console output is wider than the width it claims: {too_wide[:2]}"


def _names_a_path(line: str) -> bool:
    """Whether a line's width is the length of a filesystem path rather than of this module.

    Anything holding a path this run was given, or the model's own name, is as wide as the
    caller's directory. Everything else is this module's own text and its width is a claim
    about the module.
    """
    return any(
        token in line
        for token in (
            "corpus.jsonl",
            "public_suffix_list.dat",
            "truth.jsonl",
            "policy-scores.jsonl",
            "logistic-content-link-features-v1",
        )
    )


def test_the_console_output_names_the_model_the_folds_and_the_base_rate() -> None:
    """The method is printed, because a figure whose method is hidden is a claim.

    A reader who cannot see that the numbers are out of fold, over how many positives, in how
    many folds, has four numbers and no way to weigh them.
    """
    printed = render_table(measured())

    assert printed.startswith("Confidence")
    assert MODEL_NAME in printed
    assert "held out" in printed
    joined = " ".join(printed.split())
    assert (
        f"of which {measured().evaluation.positives} are posts of a Planted Campaign" in joined
    ), (
        "the positive class has to be in the header, because a probability measured over an "
        "unstated number of positives says nothing about which class it was measured on"
    )


def test_the_console_output_quotes_no_scoring_rule_for_the_policy_score_baseline() -> None:
    """The Policy Score appears beside the Confidence as a baseline, and only as a baseline.

    It is printed in the baseline table with the reason it has no log loss and no Brier score,
    so no proper scoring rule is quoted for it. Printing one would be the fusion ADR-0003
    rules out, arrived at by the other door.
    """
    printed = render_table(measured())
    baseline = printed.split("baselines")[1]

    assert "policy score" in baseline.lower()
    assert "no probability reading" in baseline


def test_the_report_is_generated_and_says_so() -> None:
    """`docs/confidence.md` is present, starts with its heading, and names its generator."""
    report = COMMITTED_REPORT.read_text(encoding="utf-8")

    assert report.startswith("# The Confidence")
    assert "`rfi confidence`" in report
    assert "Do not edit it by hand" in report


def test_the_report_publishes_the_baselines_the_model_is_read_against() -> None:
    """Both baselines appear in the page, or the model's figure is a number with no scale."""
    report = COMMITTED_REPORT.read_text(encoding="utf-8")

    assert "constant base rate" in report
    assert "policy score" in report.lower()
    assert "AUC" in report


def test_both_views_say_outright_that_the_model_loses() -> None:
    """The comparison is stated as a finding, not left in a table for a reader to subtract.

    On this Corpus the model orders the planted posts worse than the Policy Score and is scored
    worse than the constant, and both views of the run say so in a sentence of their own. The
    check is the shape of the claim rather than its wording: it fails if either view carries the
    three figures and no sentence comparing them, which is the failure a reader would meet —
    three numbers, no verdict, and a conclusion left to whoever is looking.
    """
    one_run = measured()
    model, policy = one_run.evaluation.model, one_run.evaluation.baselines[1]

    assert model.auc < policy.auc, (
        "on this Corpus the model is expected to order the planted posts worse than the Policy "
        "Score; if that has changed, the finding the page leads with is stale and the prose "
        "around it needs rewriting rather than the test"
    )
    # Compared with whitespace joined, because both views wrap to their own width and the
    # sentence a reader reads is the sentence regardless of where a line fell. Comparing the
    # raw string would fail on a line break rather than on a missing finding.
    for view, where in _views(one_run):
        assert one_run.evaluation.verdict in " ".join(view.split()), (
            f"{where} gives three figures and no sentence comparing them, so a reader has to "
            "work out the finding themselves"
        )


def test_the_report_and_the_console_output_print_the_same_confidence_range() -> None:
    """The two views of one run cannot disagree about what the probabilities came out as.

    Built from the same object, as every other command in this repository does, so the
    committed report and the committed file are two views of one pass rather than two
    hand-written accounts of it.
    """
    one_run = measured()
    printed = render_table(one_run)
    report = render_report(one_run)

    for value in (one_run.lowest, one_run.highest):
        assert measure(value) in printed
        assert measure(value) in report, (
            "the report has to carry the range the console output does, or a reader of one "
            "and not the other has no way to tell they describe the same run"
        )


def test_confidences_carries_the_rows_and_the_facts_the_two_views_print() -> None:
    """One object holds the published rows and the figures, so the file and the report agree.

    This is the shape every other command here uses for the same reason: three views of one
    pass over the inputs, rather than three passes a reader has to reconcile.
    """
    one_run = measured()

    assert isinstance(one_run, Confidences)
    assert isinstance(one_run.facts, ConfidenceFacts)
    assert len(one_run.rows) == one_run.facts.posts
    assert one_run.evaluation.model.name == MODEL_NAME
    assert all(isinstance(row, Confidence) for row in one_run.rows)


def test_a_figure_holds_its_metrics_and_says_which_it_has_none_of() -> None:
    """A figure is a named row of the table, and a metric it cannot have is absent.

    `None` rather than zero for the Policy Score's missing log loss, because zero is a number
    a reader would compare against the model's and a missing one is not a claim at all.
    """
    model = Figure(name=MODEL_NAME, auc=1.0, log_loss=0.1, brier=0.1, detail="34 posts")
    policy = Figure(name="policy score", auc=0.8, log_loss=None, brier=None, detail="34 posts")

    assert model.log_loss is not None and model.brier is not None
    assert policy.log_loss is None and policy.brier is None
    assert model.detail == policy.detail
    assert scored(policy.log_loss) == "-", (
        "the dash in the baseline table is how a metric a predictor has none of is printed; a "
        "zero there would be a number to compare against the model's"
    )
    assert scored(model.log_loss) == "0.1000"


def test_comparing_two_proper_scoring_rules_needs_two_figures_that_have_them() -> None:
    """The comparison refuses a figure with no scoring rule, by name.

    The two rows this is ever called with — the model and the constant — both carry both rules.
    A third row that does not, passed in by a later edit, has to stop the run rather than have
    its absent metric compared against a number: that is the fusion ADR-0003 rules out, arrived
    at by scoring a 0-100 editorial figure as though it were a likelihood.
    """
    policy = Figure(name="policy score", auc=0.8, log_loss=None, brier=None, detail="34 posts")
    model = Figure(name=MODEL_NAME, auc=1.0, log_loss=0.1, brier=0.1, detail="34 posts")

    with pytest.raises(ValueError, match="policy score"):
        _is_better(policy, model)
    with pytest.raises(ValueError, match=MODEL_NAME):
        _is_better(Figure(name=MODEL_NAME, auc=1.0, log_loss=None, brier=None, detail=""), policy)
    assert _is_better(model, Figure(name="constant", auc=0.5, log_loss=0.2, brier=0.2, detail=""))


def test_two_runs_over_the_same_inputs_publish_the_same_figures() -> None:
    """The reproducibility claim, and the reason the file can be held to its bytes."""
    assert measured().rows == measured().rows
    assert measured().evaluation == measured().evaluation


def test_the_evaluation_moves_when_the_membership_does(tmp_path: Path) -> None:
    """A different membership gives different figures, so the evaluation is not a constant.

    One campaign reduced to a single post, and then both, so the positive class goes from 7 to
    2 and every figure with it. A report that printed the same numbers over any membership
    would be a page of prose with figures in it.

    The first case is a refusal rather than a figure: one campaign leaves two folds, so one of
    them would be fitted against a training set that holds no positive at all. That is a
    membership this method cannot answer, and saying so is the honest result rather than a
    figure over whatever came out.
    """
    campaigns = read_truth(COMMITTED_TRUTH)
    thinned = tmp_path / "truth.jsonl"
    thinned.write_text(
        json.dumps(
            {
                "accounts": list(campaigns[0].accounts[:1]),
                "campaign_id": campaigns[0].campaign_id,
                "posts": list(campaigns[0].posts[:1]),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    honest = measured()
    with pytest.raises(ValueError):
        confident(replace(_committed(), truth=thinned))

    # And with both campaigns reduced to a single post each, so the positive class is two and
    # the folds are still two. A membership change the run cannot answer is a refusal; one it
    # can answer must move the figures, or the evaluation is a constant the report could print
    # over any membership at all.
    both = tmp_path / "both.jsonl"
    both.write_text(
        "".join(
            json.dumps(
                {
                    "accounts": list(campaign.accounts[:1]),
                    "campaign_id": campaign.campaign_id,
                    "posts": list(campaign.posts[:1]),
                }
            )
            + "\n"
            for campaign in campaigns
        ),
        encoding="utf-8",
    )
    fewer = confident(replace(_committed(), truth=both))

    assert fewer.evaluation.folds == 2
    assert 0 < fewer.evaluation.positives < honest.evaluation.positives
    assert fewer.evaluation.model.auc != honest.evaluation.model.auc
    assert [row.confidence for row in fewer.rows] != [row.confidence for row in honest.rows]


# --- the command ---------------------------------------------------------------------


def test_the_command_writes_the_two_files_it_names(tmp_path: Path) -> None:
    """`rfi confidence` writes the published file and the report, held to the committed bytes.

    Every generated file in this repository is committed and held to what its command produces,
    and this one is no exception: a run over the committed inputs has to produce exactly the
    committed probabilities and exactly the committed page, or one of them is stale and nothing
    else would say so.

    No database is involved, so this runs on a machine that has never had pgvector built.
    """
    confidences, report, _ = run(tmp_path)

    assert confidences.read_bytes() == COMMITTED_CONFIDENCES.read_bytes()
    assert report.read_bytes() == COMMITTED_REPORT.read_bytes()
    assert read_confidences(confidences) == published_rows()


def test_running_twice_writes_byte_identical_bytes(tmp_path: Path) -> None:
    """Two runs over the same Corpus on one machine, the same bytes.

    The reproducibility the rest of this repository rests on, asserted for the one command that
    fits a model: a run whose numbers moved between two identical runs would leave a published
    file nothing could hold, and a reader would have no way to tell a Corpus change from a
    rerun.

    This is the within-machine half. The cross-machine half is the rounding, and the test that
    holds it is the one below.
    """
    first = run(tmp_path / "first")
    second = run(tmp_path / "second")

    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()


def test_the_published_precision_sits_far_below_the_smallest_difference_it_has_to_express() -> None:
    """Why twelve decimal places, checked rather than asserted in a comment.

    Two things have to be true at once for the rounding to be honest, and this checks both.
    **It must not collapse two posts.** The closest two published Confidences have to stay
    further apart than the rounding can move them, so a reader can still tell every post's
    probability from every other post's. **It must be well above what two interpreters
    disagree by**, which is what it is there for: the committed file was written on Windows and
    a Linux run disagreed with it in the seventeenth significant digit of one post, and the
    margin between a rounding at twelve places and a disagreement of that size is six orders of
    magnitude.

    A future Corpus whose probabilities bunch together could close the first gap, and this test
    is what would say so — before the rounding started merging two posts into one figure.
    """
    published = sorted(row.confidence for row in published_rows())
    closest = min(
        high - low for low, high in zip(published, published[1:], strict=False)
    )

    assert closest > 10 ** -_PRECISION, (
        f"the closest two published Confidences differ by {closest}, which the rounding to "
        f"{_PRECISION} places could merge: two posts would publish one figure between them"
    )
    assert _PRECISION <= 12, (
        "the rounding is coarser than the disagreement between two interpreters would survive, "
        "so a Windows run and a Linux run can publish different bytes again"
    )


def test_the_command_refuses_to_write_two_outputs_to_one_path(tmp_path: Path) -> None:
    """The confidences file and the report asked for at one path, refused by name.

    A run that wrote the report over its own data file would leave the next reader with a
    Markdown table where a JSON Lines file belongs and nothing to say so.
    """
    with pytest.raises(SystemExit, match="one path"):
        main(
            [
                "confidence",
                "--corpus", str(COMMITTED_CORPUS),
                "--list", str(COMMITTED_LIST),
                "--truth", str(COMMITTED_TRUTH),
                "--confidences", str(tmp_path / "same"),
                "--report", str(tmp_path / "same"),
            ]
        )


def test_the_command_names_a_missing_input_rather_than_failing_inside(
    tmp_path: Path,
) -> None:
    """A Corpus that is not there, refused by name with the command that writes it.

    Every command that reads the Corpus and the published list refuses the same way, because
    a bare `FileNotFoundError` from three frames down says nothing about which of the four
    inputs was missing.
    """
    with pytest.raises(SystemExit, match="the Corpus is not there"):
        main(
            [
                "confidence",
                "--corpus", str(tmp_path / "absent.jsonl"),
                "--list", str(COMMITTED_LIST),
                "--truth", str(COMMITTED_TRUTH),
                "--confidences", str(tmp_path / "confidences.jsonl"),
                "--report", str(tmp_path / "confidence.md"),
            ]
        )


# --- the vocabulary, and the model behind it --------------------------------------------------


def test_the_confidence_record_carries_no_field_a_signal_would_use() -> None:
    """A Signal is a published, weighted contribution to the Policy Score; this is neither.

    The two vocabularies are kept apart so a later edit cannot quietly turn a probability into
    a weighted contribution, which is the one shape the fusion ADR-0003 rejects would take here.
    """
    assert not {field.name for field in fields(Confidence)} & {
        "weight",
        "points",
        "signals",
        "evidence",
        "published_points",
        "published_signals",
    }


def test_the_feature_enum_is_the_whole_of_what_the_model_reads() -> None:
    """A feature name this build does not have cannot be fitted, because there is no row for it."""
    assert set(FEATURES) == {member.value for member in Feature}
    assert isinstance(FEATURES, tuple)


def _refuses_every_confidence(printed: str, where: str) -> None:
    """Whether a printed output names no Confidence in any form it could be printed in.

    Four forms rather than one, because this command prints a probability to four places above a
    thousandth and six below it, and the published file holds it at full precision — so a check
    over one form leaves the rest unchecked on exactly the values where they differ from each
    other. The severity form is deliberately absent as a bare number: every output in this
    repository carries counts, so a bare `1` would fail on a coincidence in a count rather than
    on anything about the Confidence. The severity claim is checked where it can be checked
    properly, by asking what severities the output prints at all.
    """
    assert "confidence" not in printed.lower(), f"{where} prints the word confidence"
    assert "confidences.jsonl" not in printed, f"{where} names the Confidence file"

    for row in published_rows():
        for rendered in (
            f"{row.confidence:.4f}",
            f"{row.confidence:.2f}",
            measure(row.confidence),
            repr(row.confidence),
        ):
            assert rendered not in printed, (
                f"{where} prints {rendered}, which is the Confidence of {row.post_id}"
            )


# --- helpers -------------------------------------------------------------------------------------


def _row(post_id: str = "syn_p_0001", extra: Mapping[str, object] | None = None) -> Row:
    record: dict[str, object] = {
        "account": "syn_example_0001",
        "confidence": 0.5,
        "model": MODEL_NAME,
        "post_id": post_id,
        "recipe": "b" * 64,
    }
    record.update(extra or {})
    return record


def _source(path: Path) -> str:
    """A module's text, with any byte-order mark removed.

    `utf-8-sig` rather than `utf-8` because two modules in this repository carry a BOM, and
    `ast.parse` refuses one outright — so a check that walks every module would fail on a
    character no code in the file has anything to do with, and the failure would read as
    something about the Confidence rather than about encoding.
    """
    return path.read_text(encoding="utf-8-sig")


def _imported_modules(path: Path) -> set[str]:
    """Every module a file imports, by the module's name, over both spellings of an import.

    `ast.ImportFrom` alone would miss `import reddit_fraud_intelligence.confidence` and
    `from reddit_fraud_intelligence import confidence`, and a test that only catches one of the
    three is a test that passes vacuously against the other two — which is the failure mode a
    negative test is most prone to. The module rather than the imported name for the same
    reason: `cli.py` brings this one in as `confident as measure_confidences`, and checking for
    the bare name would miss the one importer that matters.
    """
    imported: set[str] = set()
    for node in ast.walk(ast.parse(_source(path))):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            if node.level == 0:
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported

