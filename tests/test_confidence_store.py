"""The Confidence on a Corpus of the test's own, and the committed files held to this code.

The offline half of ticket #25 is `tests/test_confidence.py`: the features, the fit, the folds,
the metrics, the baselines, and the three refusals. This file is the half that needs the rest of
the machine — the command run end to end over a Corpus written out by hand, and the committed
`data/model/confidences.jsonl` and `docs/confidence.md` held to what the command produces.

The Corpus of the test's own is what makes the fit testable at all. The committed Corpus has
seven positives in two campaigns, which is the smallest positive class a two-fold evaluation can
run on, so the interesting cases — a fold that fits nothing, a Corpus with one campaign, a Corpus
where the planted posts are the short ones and nothing else distinguishes them — are all cases the
committed file cannot contain. They are written out here instead, at four or five posts each, and
the figures are checked rather than the prose.

Nothing here needs a database. The features are read from the Corpus, the published Public Suffix
List and the same Contact Point reader the grouping uses, so this file runs on a machine that has
never had pgvector built, which is the same split `test_content_embeddings.py` and
`test_embedding_store.py` make for the embeddings.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from reddit_fraud_intelligence.confidence import (
    MODEL_NAME,
    Confidence,
    Sources,
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
from reddit_fraud_intelligence.domains import post_domains
from reddit_fraud_intelligence.suffixes import PublicSuffixes
from reddit_fraud_intelligence.truth import read_truth

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CONFIDENCES = REPO_ROOT / "data" / "model" / "confidences.jsonl"
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_LIST = REPO_ROOT / "data" / "public-suffix" / "public_suffix_list.dat"
COMMITTED_REPORT = REPO_ROOT / "docs" / "confidence.md"
COMMITTED_SCORES = REPO_ROOT / "data" / "signals" / "policy-scores.jsonl"
COMMITTED_TRUTH = REPO_ROOT / "data" / "corpus" / "truth.jsonl"

# The command's own defaults rather than this file's absolute paths, wherever a run is compared
# against a committed file. The report prints the paths it was given, so a run over absolute
# paths writes this machine's directory names into the page and could not be held to the
# committed bytes — which is the whole point of holding it.
DEFAULT_CORPUS_PATH = Path("data/corpus/corpus.jsonl")
DEFAULT_LIST_PATH = Path("data/public-suffix/public_suffix_list.dat")
DEFAULT_SCORES_PATH = Path("data/signals/policy-scores.jsonl")
DEFAULT_TRUTH_PATH = Path("data/corpus/truth.jsonl")

Row = Mapping[str, object]

# A Corpus of the test's own: two planted campaigns of two posts each, and four negatives. The
# planted posts are longer and name a Contact Point; the negatives are short and do not. That is
# the shape the model can learn, and writing it out by hand is what makes the cases below
# reachable at all — the committed Corpus's figures are over seven positives and cannot be made
# to fail on purpose.
PLANTED_LONG = (
    "The monthly return is fixed and the payout has never once been late, and you can read the "
    "statement yourself before you commit a single dollar of it. Message @syn_example_alpha or "
    "open the page and the intake is the same form either way.",
    "The return is fixed rather than promised, the payout has not been late, and the statement "
    "is there to be read before anyone commits money to it. Reach @syn_example_alpha and the "
    "intake form is the one the page carries too.",
)
PLANTED_LINKED = (
    "Equipment is provided but there is a refundable materials deposit for the workstation, and "
    "you have to finish the intake within 48 hours or the slot goes to someone else. Message "
    "@syn_example_beta if you would rather not wait for me.",
    "A refundable deposit for the workstation is what the intake asks for, and the slot goes to "
    "whoever finishes the intake within 48 hours. Message @syn_example_beta if you would rather "
    "not wait for me for it.",
)
NEGATIVE_SHORT = (
    "Anyone else find the new sorting on this sub actively worse than the old one?",
    "Reminder that the recycling move is on Thursday, not Friday, as the sign outside says.",
    "The bakery on the corner has stopped doing the rye at all, which is a genuine loss.",
    "Lost my cat on Tuesday and she came back on Friday like nothing had happened at all.",
)


def post(post_id: str, account: str, body: str, link: str | None = None) -> CorpusItem:
    """One post of the test's own Corpus, in the Corpus file's own vocabulary."""
    return CorpusItem(
        post_id=post_id,
        account=account,
        subreddit="test",
        title=body.split(".")[0],
        body=body,
        created_at="2026-01-05T09:00:00Z",
        links=(link,) if link else (),
    )


def write(path: Path, items: tuple[CorpusItem, ...]) -> Path:
    """The Corpus as the command reads it: the file, not the generator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "account": item.account,
                    "body": item.body,
                    "created_at": item.created_at,
                    "links": list(item.links),
                    "post_id": item.post_id,
                    "subreddit": item.subreddit,
                    "title": item.title,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    return path


def write_truth(path: Path, campaigns: tuple[Mapping[str, object], ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(campaign, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for campaign in campaigns
        ),
        encoding="utf-8",
    )
    return path


def a_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A Corpus of the test's own, its membership, and its Policy Scores.

    The scores are hand-written rather than produced by `rfi policy-score`, which is legitimate
    for one reason and only that one: this file is checking what the Confidence does with a
    scores file, and a scores file it cannot reason about would be the one thing that could
    change the answer for a reason unrelated to the Confidence. The scores rank the four planted
    posts above the four negatives, which is the case the baseline comparison is about.
    """
    items = (
        post("syn_p_0101", "syn_alpha_0001", PLANTED_LONG[0]),
        post("syn_p_0102", "syn_alpha_0002", PLANTED_LONG[1]),
        post("syn_p_0103", "syn_beta_0001", PLANTED_LINKED[0]),
        post("syn_p_0104", "syn_beta_0002", PLANTED_LINKED[1]),
        *(post(f"syn_p_01{index:02d}", f"syn_quiet_{index:04d}", body)
          for index, body in enumerate(NEGATIVE_SHORT, start=5)),
    )
    corpus = write(tmp_path / "corpus.jsonl", items)
    truth = write_truth(
        tmp_path / "truth.jsonl",
        (
            {
                "accounts": ["syn_alpha_0001", "syn_alpha_0002"],
                "campaign_id": "syn-campaign-alpha",
                "posts": ["syn_p_0101", "syn_p_0102"],
            },
            {
                "accounts": ["syn_beta_0001", "syn_beta_0002"],
                "campaign_id": "syn-campaign-beta",
                "posts": ["syn_p_0103", "syn_p_0104"],
            },
        ),
    )
    scores = tmp_path / "policy-scores.jsonl"
    scores.write_text(
        "".join(
            json.dumps(
                {
                    "account": item.account,
                    "points": 60 if index < 4 else 0,
                    "post_id": item.post_id,
                    "published_points": 130,
                    "published_signals": 6,
                    "score": 46 if index < 4 else 0,
                    "signals": [],
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for index, item in enumerate(items)
        ),
        encoding="utf-8",
    )
    return corpus, truth, scores


# --- the fit on a Corpus of the test's own -----------------------------------------------


def test_the_model_learns_a_corpus_whose_planted_posts_are_its_long_ones(
    tmp_path: Path,
) -> None:
    """The end-to-end shape: every post gets a Confidence, and the planted ones rank higher.

    Not "the model works" — the point is that the fit is a function of the inputs and the
    published figures describe what it did. Four positives over two campaigns is the smallest
    Corpus this method can be run on at all, and it is here so the published Confidence file can
    be checked against a run whose answer is known rather than against a figure nobody can
    derive.
    """
    corpus, truth, scores = a_corpus(tmp_path)

    measured = confident(Sources(corpus=corpus, suffix_list=COMMITTED_LIST, truth=truth, scores=scores))

    assert len(measured.rows) == 8
    assert all(0.0 < row.confidence < 1.0 for row in measured.rows)
    ranked = sorted(measured.rows, key=lambda row: row.confidence, reverse=True)
    assert {row.post_id for row in ranked[:4]} == {
        "syn_p_0101", "syn_p_0102", "syn_p_0103", "syn_p_0104",
    }, "the four planted posts should rank above the four negatives"
    assert measured.evaluation.model.auc == 1.0
    assert measured.evaluation.held_out is True


def test_a_corpus_of_one_campaign_is_refused_because_one_fold_would_fit_nothing(
    tmp_path: Path,
) -> None:
    """The case the committed Corpus cannot contain, and the most likely one in practice.

    One Planted Campaign means one fold, and the fold holds out every planted post — so the fit
    has no positives at all and logistic regression over one class has no finite answer. The
    refusal is the honest result. Publishing a figure over the Corpus anyway would mean either
    dropping the fold or fitting on everything, and both of those are the two things this ticket
    exists to avoid: a probability with no out-of-sample reading, and a measurement over rows the
    model has seen.
    """
    items = read_corpus(a_corpus(tmp_path)[0])
    truth = write_truth(
        tmp_path / "one.jsonl",
        (
            {
                "accounts": ["syn_alpha_0001", "syn_alpha_0002"],
                "campaign_id": "syn-campaign-alpha",
                "posts": ["syn_p_0101", "syn_p_0102"],
            },
        ),
    )

    with pytest.raises(ValueError, match="holds every post in the Corpus"):
        confident(
            Sources(
                corpus=tmp_path / "corpus.jsonl",
                suffix_list=COMMITTED_LIST,
                truth=truth,
                scores=tmp_path / "policy-scores.jsonl",
            )
        )


def test_a_corpus_whose_planted_posts_are_the_short_ones_still_measures(
    tmp_path: Path,
) -> None:
    """The same Corpus with the labels on the short posts, and the figure drops.

    The committed Corpus's positives are its longer posts and the ones naming a Contact Point, so
    a model reading either would score well there. Reversing which posts are planted leaves only
    the other direction available: now the distinguishing features belong to the *negatives*, and
    a model can only reach a good figure by inverting them. Whether it does is the interesting
    part, and the check is that the figures move — a run whose AUC did not change when every
    label flipped would be printing figures from something other than the fit.
    """
    _, _, scores = a_corpus(tmp_path)
    swapped = write(
        tmp_path / "swapped.jsonl",
        (
            post("syn_p_0101", "syn_alpha_0001", PLANTED_LONG[0]),
            post("syn_p_0102", "syn_alpha_0002", PLANTED_LONG[1]),
            post("syn_p_0103", "syn_beta_0001", PLANTED_LINKED[0]),
            post("syn_p_0104", "syn_beta_0002", PLANTED_LINKED[1]),
            *(post(f"syn_p_01{index:02d}", f"syn_quiet_{index:04d}", body)
              for index, body in enumerate(NEGATIVE_SHORT, start=5)),
        ),
    )
    reversed_truth = write_truth(
        tmp_path / "reversed.jsonl",
        (
            {
                "accounts": ["syn_quiet_0005", "syn_quiet_0006"],
                "campaign_id": "syn-campaign-alpha",
                "posts": ["syn_p_0105", "syn_p_0106"],
            },
            {
                "accounts": ["syn_quiet_0007", "syn_quiet_0008"],
                "campaign_id": "syn-campaign-beta",
                "posts": ["syn_p_0107", "syn_p_0108"],
            },
        ),
    )
    honest_corpus, honest_truth, _ = a_corpus(tmp_path)
    honest = confident(
        Sources(
            corpus=honest_corpus,
            suffix_list=COMMITTED_LIST,
            truth=honest_truth,
            scores=scores,
        )
    )

    measured = confident(
        Sources(
            corpus=swapped,
            suffix_list=COMMITTED_LIST,
            truth=reversed_truth,
            scores=scores,
        )
    )

    assert measured.evaluation.positives == 4
    assert measured.evaluation.held_out is True
    assert measured.evaluation.model.log_loss != honest.evaluation.model.log_loss, (
        "every label has flipped and the model can still order the posts, because the features "
        "that separate them belong to what is now the negative class; what must change is the "
        "sign of the weights and the confidence of each post"
    )
    assert [row.confidence for row in measured.rows] != [row.confidence for row in honest.rows]


def test_a_corpus_whose_features_are_all_identical_measures_nothing_and_says_so(
    tmp_path: Path,
) -> None:
    """Four posts with the same seven counts, and a feature vector of no spread.

    The standardisation divides by each feature's standard deviation, so a feature every row
    agrees on has no spread and would divide by zero. It is standardised to zero instead, which
    leaves that feature flat and the remaining ones to carry the fit. The run must finish and
    publish figures rather than dividing by zero, and the figures must be the ones a predictor
    with no information produces.
    """
    same = "The very same words in every one of them, with nothing to tell them apart at all."
    accounts = [f"syn_same_{index:04d}" for index in range(1, 9)]
    items = tuple(
        post(f"syn_p_0{index:03d}", account, same) for index, account in enumerate(accounts, start=1)
    )
    corpus = write(tmp_path / "corpus.jsonl", items)
    # Four of the eight are planted and four are not, in two campaigns of two — so each fold
    # holds out one campaign and half the negatives and is fitted against a training set with
    # both labels in it. Every row's seven counts are identical, so there is nothing in any of
    # them to tell the two apart, which is the case this Corpus exists for.
    truth = write_truth(
        tmp_path / "truth.jsonl",
        (
            {
                "accounts": accounts[:2],
                "campaign_id": "syn-campaign-alpha",
                "posts": [item.post_id for item in items[:2]],
            },
            {
                "accounts": accounts[2:4],
                "campaign_id": "syn-campaign-beta",
                "posts": [item.post_id for item in items[2:4]],
            },
        ),
    )
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        "".join(
            json.dumps(
                {
                    "account": item.account,
                    "points": 0,
                    "post_id": item.post_id,
                    "published_points": 130,
                    "published_signals": 6,
                    "score": 0,
                    "signals": [],
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )

    measured = confident(Sources(corpus=corpus, suffix_list=COMMITTED_LIST, truth=truth, scores=scores))

    assert len(measured.rows) == len(items)
    assert measured.evaluation.model.auc == 0.5, (
        "eight identical posts with two campaigns hold nothing to learn from, so the ordering "
        "figure must be the one a predictor with no information reaches"
    )
    assert len({row.confidence for row in measured.rows}) == 1, (
        "identical features must give identical probabilities; anything else means the "
        "standardisation is reading something other than the post"
    )


def test_a_fit_over_a_feature_of_no_spread_leaves_it_flat() -> None:
    """The same rule at the seam, so it is the fit and not the Corpus that carries it.

    Two rows differing in one feature only: the other six have no spread, so their coefficients
    come out zero rather than as a division by zero, and the fit still produces a probability.
    """
    fit = train((((0.0, 1.0), True), ((0.0, 0.0), False)))

    assert fit.confidence((0.0, 1.0)) > fit.confidence((0.0, 0.0))
    assert fit.coefficients[0] == 0.0, "a feature of no spread cannot carry any weight"
    assert math_is_finite(fit.intercept)


def math_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


# --- the published files -----------------------------------------------------------------


def _committed_relative() -> Sources:
    """The command's own default paths rather than this file's absolute ones.

    Used by every check that compares against a generated file, because the report prints the
    paths it was given: a run over absolute paths writes this machine's directory names into the
    page, and that page could not be held to the committed bytes — which is the whole point of
    holding it. The figures under test are the numbers; the bytes are checked through the
    relative paths, and through `run` in `tests/test_confidence.py`.
    """
    return Sources(
        corpus=DEFAULT_CORPUS_PATH,
        suffix_list=DEFAULT_LIST_PATH,
        truth=DEFAULT_TRUTH_PATH,
        scores=DEFAULT_SCORES_PATH,
    )


def test_the_committed_confidences_are_the_rows_the_command_writes() -> None:
    """The committed file is a projection of the Corpus and the model's own output.

    Every row is re-derived rather than trusted: the post and the account from the Corpus, the
    probability from a fit over the same inputs, and the model and the recipe from the module
    that fitted them. A published row that disagreed with any of those would be a figure no
    reader could trace to the run that produced it.

    Absolute paths on purpose, since this compares numbers rather than the bytes of a generated
    report — and the byte comparisons below are the ones that need the command's own defaults.
    """
    written = read_confidences(COMMITTED_CONFIDENCES)
    measured = confident(
        Sources(
            corpus=COMMITTED_CORPUS,
            suffix_list=COMMITTED_LIST,
            truth=COMMITTED_TRUTH,
            scores=COMMITTED_SCORES,
        )
    )

    assert written == measured.rows
    assert {row.model for row in written} == {MODEL_NAME}


def test_the_committed_confidences_hold_one_row_per_post_of_the_committed_corpus() -> None:
    """The Corpus and the Confidence file agree on which posts exist, in both directions.

    The file is a projection of the Corpus, so a row for a post nobody can read is a number with
    no subject and a post with no row is a Content Item this project claims a Confidence for and
    does not have. Both directions are checked rather than the count, because equal counts with
    one of each wrong is the case a count would pass.
    """
    posts = {item.post_id for item in read_corpus(COMMITTED_CORPUS)}
    written = {row.post_id for row in read_confidences(COMMITTED_CONFIDENCES)}

    assert written == posts


def test_the_committed_report_holds_every_figure_the_files_carry() -> None:
    """The page's headline figures are recomputed from the committed data, not read off it.

    Two runs over the committed inputs and a check that the report is byte-identical to what
    they produce — which is the general form of the claim, and the one that catches a stale page
    as well as a wrong number.
    """
    measured = confident(_committed_relative())
    report = COMMITTED_REPORT.read_text(encoding="utf-8")

    assert f"{measured.evaluation.model.auc:.4f}" in report
    assert f"{measured.evaluation.baselines[0].auc:.4f}" in report
    assert f"{measured.evaluation.baselines[1].auc:.4f}" in report
    assert report == render_report(measured)
    assert render_table(measured).count("\n") > 30, (
        "the console output has to be more than a headline: the method, the features and the "
        "limits are the parts a pasted-into-an-issue figure arrives without"
    )


def test_the_committed_report_names_the_fold_count_and_the_positive_class() -> None:
    """The two figures that make every other figure on the page legible.

    An AUC over an unstated number of positives, in an unstated number of folds, is a number a
    reader cannot weigh. Both are in the prose rather than only in a table row, because the prose
    is what a reader reaches the page for.
    """
    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    measured = confident(_committed_relative())

    assert f"in {measured.evaluation.folds} folds" in report
    assert f"{measured.evaluation.positives} positives is a very small positive class" in report
    assert f"{measured.evaluation.posts} posts" in report


def test_the_committed_report_prints_every_fold_s_coefficients() -> None:
    """One table per fold, every feature in it, and the weight that fold's fit gave.

    The page's claim is that the numbers came from those seven counts, and a reader who cannot
    see the weights cannot check that. **Every** fold, because there is no single fit behind the
    published probabilities: publishing one table for the run would be publishing one fold's
    weights beside everybody's numbers, and a reader checking a Confidence against the wrong
    fold's table would find a mismatch and blame the model.
    """
    measured = confident(_committed_relative())
    report = COMMITTED_REPORT.read_text(encoding="utf-8")

    assert len(measured.fits) == measured.evaluation.folds
    for folded in measured.fits:
        assert f"### Fold {folded.fold}" in report, f"the report has no table for fold {folded.fold}"
        assert f"| intercept | — | {folded.intercept:.4f} |" in report
        for name, weight in folded.coefficients:
            assert f"| `{name}` |" in report, f"the report does not print the feature {name}"
            assert f"| {weight:.4f} |" in report, (
                f"the report does not print fold {folded.fold}'s weight for {name}"
            )


# --- the reading of a link ------------------------------------------------------------------


def test_the_registration_features_come_from_the_published_suffix_list() -> None:
    """A host three labels deep under the reserved TLD is one registration, not three.

    The same truncation `rfi post-domains` publishes, taken from the same Public Suffix List, so
    a model reading `registrable_domains` counts registrations the way every other command
    counts them rather than counting hosts. Checked here against `post_domains` itself rather
    than against a transcribed figure, which is what makes it a comparison and not a copy.
    """
    items = (
        post("syn_p_0301", "syn_one_0001", "A post with a mirror beside it.",
             link="https://mirror.syn-shop.example/mirror/page"),
        post("syn_p_0302", "syn_two_0002", "Another post from the same desk.",
             link="https://syn-shop.example/main"),
        post("syn_p_0303", "syn_three_0003", "A post naming no registration at all.",
             link="mailto:nobody@somewhere.invalid"),
    )

    values = confidence_features(items, COMMITTED_LIST)
    resolved = {row.post_id: row.domains for row in post_domains(items, PublicSuffixes.read(COMMITTED_LIST))}

    for item in items:
        assert values[item.post_id]["registrable_domains"] == len(resolved[item.post_id])
    assert values["syn_p_0301"]["registrable_domains"] == 1
    assert values["syn_p_0303"]["registrable_domains"] == 0
    assert values["syn_p_0303"]["unresolved_links"] == 1, (
        "a mailto: link names no registration, and the count of those is a feature: a post "
        "whose every link resolves to nothing is a post linking nothing anybody registered"
    )


# --- the fold plan ------------------------------------------------------------------------------


def test_two_posts_by_one_account_land_in_one_fold(tmp_path: Path) -> None:
    """The half of the fold rule that is easy to get wrong, and the reason it is here.

    Seven posts by one account split across two folds would leave the model free to memorise
    that account's vocabulary and then be scored on it — the same leak as holding out a campaign
    by post while calling it a campaign. So the negatives are dealt out by account: two posts by
    one account always land in the same fold, and the published probabilities of a single voice
    come from a fit that has never seen that voice.
    """
    items = read_corpus(a_corpus(tmp_path)[0])
    noisy = write(
        tmp_path / "noisy.jsonl",
        items
        + tuple(
            post(f"syn_p_01{index:02d}", "syn_quiet_0005", NEGATIVE_SHORT[0])
            for index in range(9, 13)
        ),
    )
    items = read_corpus(noisy)
    plan = folds(items, label_set(read_truth(a_corpus(tmp_path)[1])))

    for account in {item.account for item in items}:
        assert len({plan[item.post_id] for item in items if item.account == account}) == 1, (
            f"the posts of {account} are split across folds, so one of them was fitted against "
            "the other's vocabulary"
        )
    assert len({plan[item.post_id] for item in items}) == 2


def test_every_post_is_in_exactly_one_fold_and_the_folds_partition_the_corpus() -> None:
    """A fold plan that misses a post would leave that post with no Confidence at all.

    Checked over the committed Corpus rather than the test's own, because this is the property
    the published file depends on: every published row has a fold behind it, and every post in
    the Corpus has a row.
    """
    items = read_corpus(COMMITTED_CORPUS)
    plan = folds(items, label_set(read_truth(COMMITTED_TRUTH)))
    published = {row.post_id for row in read_confidences(COMMITTED_CONFIDENCES)}

    assert set(plan) == published
    assert set(plan.values()) == {0, 1}
    assert len(plan) == len(items)


def test_a_membership_naming_no_campaign_is_refused_by_name(tmp_path: Path) -> None:
    """No campaigns, no folds, and a Confidence with nothing to be held out of.

    A fit over every post with no fold to leave anything out of is the in-sample figure ADR-0003
    warns about, so it is refused rather than produced.
    """
    with pytest.raises(ValueError, match="no fold"):
        folds(read_corpus(a_corpus(tmp_path)[0]), {})


# --- the printed range ------------------------------------------------------------------------------


def test_the_low_end_of_the_range_is_printed_at_a_precision_that_is_not_zero() -> None:
    """A probability of 2.4e-05 printed as `0.0000` is a figure of zero, and it is not one.

    The published file's lowest Confidence is well below a thousandth, and printing it to four
    decimal places would report a probability of exactly zero for a post the model is only
    confident it did not plant. Every figure a reader cannot check is a claim, and a rounded-away
    probability is the quietest way to make one.
    """
    published = read_confidences(COMMITTED_CONFIDENCES)
    lowest = min(row.confidence for row in published)

    assert lowest < 0.001, "this Corpus's lowest Confidence should be below a thousandth"
    assert float(measure(lowest)) > 0.0
    assert measure(lowest) != "0.0000"
    assert len(measure(lowest)) == len("0.000000")


def test_measure_rounds_consistently_at_the_boundary() -> None:
    """Four places above a thousandth, six at and below it, and never a rounded-away zero."""
    assert measure(1.0 / 3) == "0.3333"
    assert measure(0.001) == "0.0010"
    assert measure(0.000999) == "0.000999"
    assert measure(0.0000004) == "0.000000"
    assert re.fullmatch(r"0\.\d+", measure(0.0000004)), (
        "a probability this small rounds to zero at six places, and the honest answer is to "
        "print it at more rather than to print a zero the model did not produce"
    )


def test_confidence_is_a_frozen_record_named_for_what_it_is() -> None:
    """A named type rather than a tuple, because the file's vocabulary is this one.

    The reader's vocabulary check is `fields(Confidence)`, so a field added here without a
    writer and a reader to match it would be a field a row could carry and nothing would notice.
    """
    row = Confidence(
        post_id="syn_p_0001",
        account="syn_example_0001",
        confidence=0.5,
        model=MODEL_NAME,
        recipe="a" * 64,
    )

    assert row.as_row() == {
        "post_id": "syn_p_0001",
        "account": "syn_example_0001",
        "confidence": 0.5,
        "model": MODEL_NAME,
        "recipe": "a" * 64,
    }
    assert label_for(
        {"syn-campaign-alpha": frozenset({"syn_example_0001"})},
        post("syn_p_0001", "syn_example_0001", "A post."),
    )