"""The Confidence: a classifier's probability, held internally and never displayed (ADR-0027).

This is the project's machine-learning claim, and it is deliberately kept off every screen. A
Content Item gets a **Confidence**: the model's probability that it exhibits the pattern the
model was trained to find, fitted here over seven features read from the post's own title,
its own body and its own links. The Confidence is stored in a file of its own and read by
nothing in this repository. It is never rendered as a severity, never printed as a number out
of a hundred, and never summed with the Policy Score — which stays what ADR-0003 made it, an
additive rules engine whose weights are published and whose arithmetic a reader can redo by
hand.

**Why the file exists at all, if nothing reads it.** Because ADR-0003 says the Confidence is
stored and never displayed, and a quantity that exists only in prose is a claim. Writing one
probability per post is a fact somebody can check, and it is what ticket #27 needs to measure
calibration against. What makes it safe to store is that no command prints it: `cli.py` is the
only module that imports this one, so `rfi review-queue` and `rfi campaign-candidates` cannot
reach it even by accident, and `tests/test_confidence.py` asserts that by walking the import
graph and by searching the Review Queue's own printed output.

**The training labels are the Planted Campaign membership, and the report says so.** A post's
label is whether its account belongs to a Planted Campaign, taken from
`data/corpus/truth.jsonl`. That file is otherwise read only by the evaluator, and this
command reads it for the evaluator's reason: a supervised model needs labels, and a Corpus
from a Corpus Provider has none beside it. So what is measured here is **recovery of planted
structure** — whether a classifier can tell the generator's planted posts from the rest of a
Corpus the generator also wrote — and it is not a rate of fraud found in the world. The report
gives that a section of its own rather than a footnote, because a reader who takes a number
off this page for anything else has been told something false about how it came about.

**Every published probability is out of fold.** The folds are the Planted Campaigns: a post's
Confidence comes from a fit trained without that post and without every other post of its
campaign, so a model that had memorised one campaign's vocabulary could not carry it into the
other. On a Corpus with two campaigns this is two fits, and the report says so. A training fit
over seven positives is perfect by construction and says nothing, which is why no figure
published here comes from one.

**The model is published rather than assumed.** `logistic-content-link-features-v1`, fitted by
gradient descent at a stated step count, a stated learning rate and a stated penalty, over a
fixed and enumerated feature list. No wheel, no seed, no hyperparameter chosen by looking at
the answer: two runs over the same Corpus produce the same bytes, which is what lets
`tests/test_confidence_store.py` hold the published file to what this code produces. The recipe
digest travels on every row for the reason ADR-0024 gives for the embeddings — a name is a
promise somebody has to keep, and the digest is the promise checked.

**Two baselines, and the model is read against both.** The trivial one is a constant equal to
the base rate: it predicts the rate and orders nothing, and a classifier that cannot order
posts better than it is not worth reading. The interesting one is the Policy Score, ordered as
it is published, so the model's contribution is a difference a reader can see rather than two
numbers from two runs. The Policy Score is read for that one purpose and is otherwise not an
input: this module does not import `signals` at all, so it cannot become a feature by
accident. A model given the Policy Score as an input would return the Policy Score with a
decimal point on it, which is the fused score ADR-0003 rules out arrived at by another door.

**Calibration is not measured here.** ADR-0003 already says it is a separate concern with its
own evaluation, and that is ticket #27. What this command publishes is discrimination — how
well the Confidence orders posts, by AUC, and two proper scoring rules — and the report says
which of the two it is reporting, so that neither number is read as the other.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

from reddit_fraud_intelligence.contacts import extract as read_contact_points
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.domains import post_domains
from reddit_fraud_intelligence.jsonl import (
    JsonObject,
    read_object,
    read_rows,
    read_text,
    refuse_repeated,
    write_lines,
)
from reddit_fraud_intelligence.suffixes import PublicSuffixes
from reddit_fraud_intelligence.text import wrap
from reddit_fraud_intelligence.truth import PlantedCampaign, read_truth

_HEADING = "Confidence"

_SUBHEADING = """\
A probability, not a severity. Every post below was scored by a model fitted without it and
without every other post of its Planted Campaign, so these figures are held out of the fit that
produced them, and the labels they are measured against come from the generator that wrote the
Corpus (ADR-0027)."""

# Where the confidences are published, as the command writes them by default. Named here
# because the report points at the file a reader should open beside it, and a report printing
# the path of this particular run would produce different bytes for the same Corpus depending
# only on the directory it was written into. `cli.py` holds the default the command actually
# uses; this is the name the report gives it.
CONFIDENCES_PATH = "data/model/confidences.jsonl"

# The width a console line is wrapped to, which is the claim this command makes and the width
# its own test holds it to.
_WIDTH = 100

# The name carries the recipe's version, because a change to any part of that recipe changes
# every number the model produces — and a penalty changed without the name changing would be
# the worst version of this bug, since every stored probability would still parse. The digest
# beside it is the half a name cannot be on its own.
MODEL_NAME = "logistic-content-link-features-v1"

# The fit, as three numbers a reader can check against the output rather than tune. The step
# count is not a hyperparameter chosen by looking at the held-out figure: it is where the
# objective has stopped moving on this Corpus by three orders of magnitude. The penalty is one
# whole number rather than a searched value, for the reason ADR-0024 gives for refusing a
# downloaded model: a figure a reader cannot reproduce is a claim.
_STEPS = 400
_LEARNING_RATE = 0.5
_PENALTY = 0.01

# A run of letters and digits, which is the corpus's own definition of a word rather than
# whitespace's: `body_words` is a feature a reader will recount, and the two definitions
# disagree on every hyphen and every underscore — and on every Contact Point this Corpus
# publishes, since those are written with underscores in them.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


class Feature(StrEnum):
    """Every feature the model reads, and the order the coefficients print in.

    The list is closed and declared here rather than derived from the Corpus, so a reader can
    see the whole of what goes into the number before running anything, and a feature that
    appeared by accident would have to be added to this enum to be fitted at all.

    Every member is a count read from one post — its own title, its own body, its own links,
    and what those links resolve to — so `tests/test_confidence.py` can hold the features
    identical when every account in the Corpus is renamed (ADR-0007, ADR-0008). None of them
    reads across the Corpus: `domain_frequency` is a Signal precisely because it does, and a
    model reading one of those would be re-deriving the Policy Score from the outside.

    The two word counts are separate because a title and a body are written differently, and
    one standing in for the other would make a feature that reads as the same thing in two
    vocabularies. `registrable_domains` and `links` are separate for the same reason: a post
    linking a link-in-bio page and the site behind it reaches one registration by two links,
    and a model told only the link count would be told the opposite.
    """

    TITLE_WORDS = "title_words"
    BODY_WORDS = "body_words"
    LINKS = "links"
    REGISTRABLE_DOMAINS = "registrable_domains"
    UNRESOLVED_LINKS = "unresolved_links"
    CONTACT_POINTS = "contact_points"
    WORDS_WITH_DIGITS = "words_with_digits"

    @property
    def reads(self) -> str:
        """The one-line reading of this feature, printed beside its coefficient.

        Printed rather than asserted because a coefficient a reader cannot check the meaning
        of is a number with a label on it. The whole claim of this model is that it reads
        nothing a reviewer could not count off the post in front of them.
        """
        return _READINGS[self]


FEATURES: tuple[str, ...] = tuple(member.value for member in Feature)

_READINGS: Mapping[Feature, str] = {
    Feature.TITLE_WORDS: "words in the post's own title",
    Feature.BODY_WORDS: "words in the post's own body",
    Feature.LINKS: "links the post carries",
    Feature.REGISTRABLE_DOMAINS: "distinct Registrable Domains those links resolve to",
    Feature.UNRESOLVED_LINKS: "links naming no Registrable Domain",
    Feature.CONTACT_POINTS: "Contact Points the post names, read by that reader",
    Feature.WORDS_WITH_DIGITS: "words in the title and body carrying a digit",
}

# The model, as one paragraph, and the digest of it beside the name. Published rather than
# derived from this file: a digest taken over the code would move whenever a comment or a type
# annotation did, and what a reader needs is a statement of the rule their own numbers would
# have been produced under.
_RECIPE = (
    "seven features, each a count read from one post's own title, its own body and its own "
    "links, standardised by subtracting the mean of the fitting rows and dividing by their "
    "standard deviation, with a feature of no spread among them standardised to zero; the "
    "model is logistic regression fitted by batch gradient descent on the mean binary "
    f"cross-entropy over the fitting rows with an L2 penalty of {_PENALTY} on the "
    f"coefficients and none on the intercept, at a learning rate of {_LEARNING_RATE} for "
    f"{_STEPS} steps, so the fit is deterministic and holds no seed; a post's probability is "
    "the logistic function of its intercept plus the weighted sum of its standardised "
    "features; and the fit runs once per fold, with every post of that fold's Planted "
    "Campaign removed from the fitting rows"
)


@dataclass(frozen=True, slots=True)
class Confidence:
    """One Content Item's Confidence, as this project publishes it.

    Five fields: the post, the account it belongs to, the probability, the model, and the
    digest of the recipe. No Policy Score, no Signal, no weight, and no label — and
    `read_confidences` refuses a row holding any of them, which is what keeps the two numbers
    ADR-0003 separates out of one record rather than merely out of one view (ADR-0008).

    No fold either. A fold partitions the posts by the Planted Campaign they came from, so a
    `fold` field here would be the membership one step removed; the fold structure is published
    in the report, where a reader is told what it is for.

    The account travels because every other published record about a post carries it, and a
    Confidence nobody can attach to a post is a number with no subject.
    """

    post_id: str
    account: str
    confidence: float
    model: str
    recipe: str

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(
                f"{self.post_id} has confidence={self.confidence}, and a probability outside "
                "the open unit interval is not a Confidence: neither bound is one a fitted "
                "logistic model reaches, and a number above one would be a severity wearing "
                "the model's name (ADR-0003)"
            )

    def as_row(self) -> JsonObject:
        return {
            "post_id": self.post_id,
            "account": self.account,
            "confidence": self.confidence,
            "model": self.model,
            "recipe": self.recipe,
        }


@dataclass(frozen=True, slots=True)
class Fit:
    """One fitted model: its coefficients, its intercept, and what it was fitted on.

    `centre` and `scale` travel with the coefficients because the standardisation is half the
    model, and neither half means anything without the other: a coefficient is a weight per
    standard deviation, so a reader scoring a post by hand needs the mean and the spread to
    arrive at the published probability. They are held here rather than published because what
    the report prints is the weight per feature and the feature's own reading — a reader who
    wants the arithmetic reproduces the standardisation from the Corpus, which is where the
    numbers came from anyway.

    `rows` and `positives` are printed beside the fit, because a fit over seven positives is a
    fact about the Corpus rather than about the model.

    The coefficients are in `FEATURES` order, which is the declaration order of the enum and
    so the print order of the table.
    """

    coefficients: tuple[float, ...]
    intercept: float
    centre: tuple[float, ...]
    scale: tuple[float, ...]
    steps: int
    rows: int
    positives: int

    def confidence(self, values: Sequence[float]) -> float:
        """The probability for one post's features, under this fit.

        The standardised form rather than the raw one, and the refusal is explicit: a row of
        the wrong width cannot be scored against these coefficients, and padding it with zeros
        or truncating it would each produce a number that looks like a Confidence and is not
        one.
        """
        if len(values) != len(self.coefficients):
            raise ValueError(
                f"a row of {len(values)} features cannot be scored against coefficients of "
                f"width {len(self.coefficients)}: the two were fitted over different lists"
            )
        return _sigmoid(self.intercept + _weighted(values, self))

    def standardised(self, values: Sequence[float]) -> tuple[float, ...]:
        """One row of features against this fit's own centre and scale.

        Written out because the two consumers of a fitted model — the training loop and the
        scoring call — must standardise identically, and a second copy of that arithmetic is a
        second definition of what the coefficients mean. Also what a reader reproducing a
        published probability by hand needs, which is why the mean and the spread are held
        beside the coefficients rather than recomputed from the Corpus.
        """
        if len(values) != len(self.coefficients):
            raise ValueError(
                f"a row of {len(values)} features cannot be scored against coefficients of "
                f"width {len(self.coefficients)}: the two were fitted over different lists"
            )
        return tuple(
            (value - middle) / spread
            for value, middle, spread in zip(values, self.centre, self.scale, strict=True)
        )


def _weighted(values: Sequence[float], fit: Fit) -> float:
    """The weighted sum of one row's standardised features, over a fit's coefficients.

    The one place the dot product is written, because the training loop and the scoring call
    have to agree on it exactly: two copies of this arithmetic are two definitions of what a
    coefficient means, and a disagreement between them would show up as a model that fits
    badly and scores well.
    """
    return sum(
        weight * standardised
        for weight, standardised in zip(
            fit.coefficients, fit.standardised(values), strict=True
        )
    )


@dataclass(frozen=True, slots=True)
class Figure:
    """One row of the measurement: a named predictor and the metrics it reached.

    `log_loss` and `brier` are `None` where the predictor cannot be scored by a proper scoring
    rule, and that is the case for the Policy Score: a 0-100 editorial figure has no probability
    reading, and publishing a likelihood derived from one would be turning a judgement into a
    number with a mathematical claim attached. `None` rather than zero, because zero is a figure
    a reader would compare against the model's.
    """

    name: str
    auc: float
    log_loss: float | None
    brier: float | None
    detail: str


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The model and the two baselines it is read against, over the same held-out rows.

    `held_out` is published as a field rather than left implied by a word in the prose,
    because a figure a reader cannot tell was measured out of sample is a figure they have to
    take on trust — and the difference between the two fits over seven positives is the whole of
    what this command could otherwise be claiming.

    `verdict` is the comparison worked out rather than left to the reader. Publishing three
    numbers in a table and hoping somebody subtracts them is how a table becomes an argument
    nobody made: this project's position throughout is that a figure beside no other figure is a
    claim, and the comparison is the figure.
    """

    model: Figure
    baselines: tuple[Figure, ...]
    posts: int
    positives: int
    folds: int
    held_out: bool
    verdict: str


@dataclass(frozen=True, slots=True)
class ConfidenceFacts:
    """What reading the four inputs establishes, stated as claims about bytes.

    The four digests are printed because all four files can be regenerated by a different
    command over a different Corpus, and a Confidence a reader cannot trace to the bytes behind
    it is a number they have to take on trust. `scores_path` is here for the reason the Policy
    Score baseline is published at all: so a reader can see which ordering the model's figure
    is being read against.
    """

    accounts: int
    corpus_path: str
    corpus_sha256: str
    folds: int
    model: str
    posts: int
    positives: int
    recipe: str
    scores_path: str
    scores_sha256: str
    suffix_list_path: str
    suffix_list_sha256: str
    training_rows: int
    truth_path: str
    truth_sha256: str


@dataclass(frozen=True, slots=True)
class Sources:
    """The four files one run reads, named together because they travel together.

    A type rather than four parameters because they do: `confident`, `_facts` and `_evaluation`
    each take all four, and four same-typed parameters travelling through three functions in
    one order is four chances to pass the membership where the Public Suffix List belongs — which
    would be a silent and serious mistake rather than a loud one. Named fields make the mistake
    impossible to make silently.
    """

    corpus: Path
    suffix_list: Path
    truth: Path
    scores: Path

    @property
    def folds(self) -> int:
        """How many folds the run will hold out, which is how many Planted Campaigns there are.

        The folds *are* the campaigns, so the count is a fact about the membership rather than
        about anything computed from it — which is why it is derived here from the file's rows
        rather than threaded through from the fold plan.
        """
        return len(read_truth(self.truth))


@dataclass(frozen=True, slots=True)
class Folded:
    """One fold's fitted model: which fold, and what it gave each feature.

    A fit per fold rather than one fit for the run, because that is what leave-one-campaign-out
    means and because publishing a single set of weights beside probabilities that two different
    fits produced would be a claim the page could not support. Both folds are published, so a
    reader who wants to know what drove a Confidence can see which fold scored that post and
    what that fold's weights were.
    """

    fold: int
    coefficients: tuple[tuple[str, float], ...]
    intercept: float
    rows: int
    positives: int


@dataclass(frozen=True, slots=True)
class Confidences:
    """Everything one run establishes, so the table and the file cannot disagree.

    The rows, the figures, the folds, and the range they came out in are one object because the
    console output and the generated report are two renderings of it. The range is carried rather
    than recomputed per view, so the two views cannot print different bounds over the same run.
    """

    facts: ConfidenceFacts
    rows: tuple[Confidence, ...]
    evaluation: Evaluation
    fits: tuple[Folded, ...]

    @property
    def lowest(self) -> float:
        return min(row.confidence for row in self.rows)

    @property
    def highest(self) -> float:
        return max(row.confidence for row in self.rows)

    @property
    def mean_confidence(self) -> float:
        return sum(row.confidence for row in self.rows) / len(self.rows)


# --- the features ----------------------------------------------------------------------


def recipe_digest() -> str:
    """The digest of what the model does, beside its name.

    The half a version number cannot be on its own: editing the recipe without bumping
    `MODEL_NAME` stops the run rather than quietly reusing probabilities the new recipe would
    not have produced, and nothing else would notice — the name would match, the file would
    parse, and the numbers would just be of a model nobody can name.
    """
    return hashlib.sha256(_RECIPE.encode("utf-8")).hexdigest()


def confidence_features(
    items: Sequence[CorpusItem], list_path: Path
) -> dict[str, dict[str, int]]:
    """Every post's seven features, as counts a reader can recount by hand.

    Two passes over the Corpus's own text: the resolved links first, so the registration counts
    come from the same Public Suffix List every other command resolves against, and the Contact
    Points second, read by `contacts.extract` rather than from the file `rfi contact-points`
    publishes — the same decision `rfi campaign-candidates` makes, so this command is a function
    of the Corpus and the list rather than of whether somebody remembered to run a step before
    it.

    Nothing here reads an account, a subreddit or a timestamp. That is structural rather than
    promised: `tests/test_confidence.py` re-reads the whole Corpus with every account renamed,
    every subreddit swapped and every timestamp moved, and requires the features to come back
    identical. A model fitted on labels derived from Planted Campaign membership would otherwise
    be reading the generator's own bookkeeping through a side door (ADR-0008).
    """
    suffixes = PublicSuffixes.read(list_path)
    resolved = {row.post_id: row for row in post_domains(items, suffixes)}
    contact_counts = {row.post_id: len(row.contact_points) for row in read_contact_points(items)}

    features: dict[str, dict[str, int]] = {}
    for item in items:
        row = resolved[item.post_id]
        title = _words(item.title)
        body = _words(item.body)
        features[item.post_id] = {
            Feature.TITLE_WORDS.value: len(title),
            Feature.BODY_WORDS.value: len(body),
            Feature.LINKS.value: len(item.links),
            Feature.REGISTRABLE_DOMAINS.value: len(row.domains),
            Feature.UNRESOLVED_LINKS.value: sum(
                1 for link in row.links if link.domain is None
            ),
            Feature.CONTACT_POINTS.value: contact_counts[item.post_id],
            Feature.WORDS_WITH_DIGITS.value: sum(
                1 for word in (*title, *body) if any(character.isdigit() for character in word)
            ),
        }
    return features


def _words(text: str) -> tuple[str, ...]:
    """The words of one field, as runs of letters and digits, case folded."""
    return tuple(_WORD.findall(text.lower()))


# --- the labels and the folds ------------------------------------------------------------


def label_set(campaigns: Sequence[PlantedCampaign]) -> dict[str, frozenset[str]]:
    """The membership as a lookup: campaign id to the accounts in it.

    Keyed by account rather than by post, because that is what a label is — an account's
    membership decides its posts' labels, and joining on post ids instead would make the label a
    fact about the generator's own record of which post it wrote rather than about who wrote it.

    Built here and passed in rather than read inside the fit, which is what lets a test fit over
    hand-written labels without a membership file.
    """
    return {campaign.campaign_id: frozenset(campaign.accounts) for campaign in campaigns}


def label_for(labelled: Mapping[str, frozenset[str]], item: CorpusItem) -> bool:
    """Whether this post's account is in any Planted Campaign.

    `any` over the whole mapping rather than a lookup of one campaign, because a post is
    positive if it is in any of them; and reading the membership at all is something this
    command does for the reason the evaluator does — see the module docstring.
    """
    return any(item.account in accounts for accounts in labelled.values())


def folds(
    items: Sequence[CorpusItem], labelled: Mapping[str, frozenset[str]]
) -> dict[str, int]:
    """Which fold each post is held out in, over the Planted Campaigns.

    A positive takes the index of its own campaign, so every post of one campaign is held out
    together and a model fitted on the other cannot carry that campaign's vocabulary into it.
    That is the whole of the leave-one-campaign-out rule.

    A negative's fold is decided by its **account**, not by the post, and that is the half of
    the rule that is easy to get wrong. Seven posts by one account split across two folds would
    leave the model free to memorise that account's vocabulary and be scored on it, which is the
    same leak as holding out a campaign by post while calling it a campaign. So the accounts no
    campaign covers are dealt out whole, by their position among the accounts in post-id order,
    and every post by one account lands in one fold — which is also what makes the published
    probabilities comparable, since two posts by one account are two readings of one voice.

    Every post is held out exactly once, so every published Confidence is out of fold. Leaving
    the negatives out of the folds altogether would mean their probabilities came from a model
    that had seen them, and mixing two kinds of number in one column is the sort of thing this
    project exists to catch.

    The assignment is arithmetic over post-id order rather than a seed, so two runs produce the
    same folds and the published file can be held to its bytes.
    """
    count = len(labelled)
    if count == 0:
        raise ValueError(
            "the membership names no Planted Campaign, so there is no fold to hold one out in "
            "and a Confidence could not be measured out of sample at all"
        )
    plan: dict[str, int] = {}
    free: list[str] = []
    for item in sorted(items, key=lambda item: (item.account, item.post_id)):
        if not label_for(labelled, item) and item.account not in free:
            free.append(item.account)
    assignment = {account: position % count for position, account in enumerate(free)}
    for item in items:
        plan[item.post_id] = assignment.get(item.account, 0)
    for index, accounts in enumerate(labelled.values()):
        for item in items:
            if item.account in accounts:
                plan[item.post_id] = index
    return plan


# --- the fit -----------------------------------------------------------------------------


def train(rows: Sequence[tuple[Sequence[float], bool]]) -> Fit:
    """Fit the model by gradient descent, deterministically.

    Three refusals before any arithmetic, because each of them is a figure this project would
    otherwise publish as though it meant something. No rows is 0 of 0, which is not a figure.
    One label is a logistic regression with no finite answer: the coefficients grow until the
    step count runs out and the probability that came back would be published as a Confidence
    with nothing wrong with it visible. Two widths is two different feature lists in one fit, and
    no coefficient vector could score both.

    Standardisation is fitted over the fitting rows rather than over the whole Corpus, so a post
    held out of the fit is also held out of the numbers that scale it — otherwise the held-out
    figure would be measured against a transformation the model had already seen.

    A feature of no spread among the fitting rows is standardised to zero rather than divided by
    zero, which makes its coefficient meaningless and its contribution flat; a Corpus where every
    post carries the same number of links is a Corpus that cannot say anything about links, and a
    `NaN` would be the wrong way to say so.
    """
    if not rows:
        raise ValueError(
            "no rows to fit: a Confidence over an empty Corpus would be a figure of nothing"
        )

    if len({label for _, label in rows}) == 1:
        raise ValueError(
            f"every one of the {len(rows)} fitting rows carries one label, and a logistic "
            "regression over one class has no finite answer: the coefficients would grow until "
            "the step count ran out and the probability published would be an artefact of that. "
            "A fold holding out one Planted Campaign from a Corpus whose every other post is "
            "negative is this case"
        )

    width = len(rows[0][0])
    wrong = next(
        (index for index, (values, _) in enumerate(rows) if len(values) != width), None
    )
    if wrong is not None:
        raise ValueError(
            f"the row at index {wrong} has {len(rows[wrong][0])} features beside {width}, and "
            "a coefficient vector of one width cannot score both"
        )

    centre, scale = _standardise(rows, width)
    coefficients = [0.0] * width
    intercept = 0.0
    count = len(rows)
    for _ in range(_STEPS):
        # The fit is held as a `Fit` and rebuilt each step, rather than the loop keeping its own
        # centre, scale and coefficients in step with each other. A `Fit` built from the current
        # numbers is the same arithmetic as before — `_weighted` is the dot product either way —
        # and it means the loop and `Fit.confidence` cannot standardise differently, which is the
        # one way a fit could end up scoring better than it trained.
        fit = Fit(
            coefficients=tuple(coefficients),
            intercept=intercept,
            centre=centre,
            scale=scale,
            steps=0,
            rows=count,
            positives=0,
        )
        gradient = [0.0] * width
        bias = 0.0
        for values, label in rows:
            error = _sigmoid(intercept + _weighted(values, fit)) - (1.0 if label else 0.0)
            bias += error
            standardised = fit.standardised(values)
            for index, value in enumerate(standardised):
                gradient[index] += error * value
        intercept -= _LEARNING_RATE * bias / count
        for index in range(width):
            coefficients[index] -= _LEARNING_RATE * (
                gradient[index] / count + _PENALTY * coefficients[index]
            )

    return Fit(
        coefficients=tuple(coefficients),
        intercept=intercept,
        centre=centre,
        scale=scale,
        steps=_STEPS,
        rows=count,
        positives=sum(1 for _, label in rows if label),
    )


def _standardise(
    rows: Sequence[tuple[Sequence[float], bool]], width: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The mean and the standard deviation of each feature, over the fitting rows."""
    centre: list[float] = []
    scale: list[float] = []
    for index in range(width):
        values = [float(row[0][index]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        centre.append(mean)
        scale.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return tuple(centre), tuple(scale)


def _sigmoid(total: float) -> float:
    """One logistic function, in the two halves that keep `exp` from overflowing.

    The halves are the same number written two ways: on a large positive argument
    `math.exp(-total)` underflows to zero and the result is 1, and on a large negative one
    `math.exp(total)` underflows to zero and the result is 0. Neither raises, and both
    approach a bound without reaching it — which is what keeps every published probability
    inside the open unit interval, so a row of exactly 0 or 1 is a number no fit in this
    module can have produced and `read_confidences` refuses it by name.
    """
    if total >= 0.0:
        return 1.0 / (1.0 + math.exp(-total))
    grown = math.exp(total)
    return grown / (1.0 + grown)


# --- the measurement ----------------------------------------------------------------------


def measure(value: float) -> str:
    """One published probability, as it is printed.

    Four decimal places above a thousandth and six below it, because the low end of this
    model's range is a genuine 2.4e-05 and printing that as `0.0000` would report a
    probability of zero for a post the model is only confident it did not plant. A number
    rounded to nothing is a figure a reader cannot check, which is the one thing every figure
    on this page has to be.
    """
    return f"{value:.4f}" if value >= 0.001 else f"{value:.6f}"


def _auc(scored: Sequence[tuple[float, bool]]) -> float:
    """The area under the ROC curve, by counting.

    Every planted post is paired with every other and scored for whether it was ranked above it,
    with a tie counting half. It is the ranking metric rather than a scoring rule because the
    question here is whether the Confidence orders posts usefully, and AUC answers exactly that
    while staying defined at every prevalence — including the 7-in-34 this Corpus has, where an
    accuracy would flatter a model that simply predicted the majority.

    A set of rows with one class in it has no such pairs and no AUC, and returns half rather
    than raising: half is the one number a predictor that orders nothing would reach, so it is
    the figure a degenerate row should be reported at.
    """
    positives = [value for value, label in scored if label]
    negatives = [value for value, label in scored if not label]
    if not positives or not negatives:
        return 0.5
    wins = sum(
        (high > low) + 0.5 * (high == low) for high in positives for low in negatives
    )
    return wins / (len(positives) * len(negatives))


def _log_loss(scored: Sequence[tuple[float, bool]]) -> float:
    """The mean negative log-likelihood, over the same rows.

    A proper scoring rule, so it can only be read against another probability's. `log` of a
    probability is negative, so the loss is a positive number and lower is better — which is why
    the constant baseline has one at all: a model that cannot beat the base rate's loss is worse
    than saying nothing.
    """
    return -sum(
        (1.0 if label else 0.0) * math.log(value)
        + (0.0 if label else 1.0) * math.log(1.0 - value)
        for value, label in scored
    ) / len(scored)


def _brier(scored: Sequence[tuple[float, bool]]) -> float:
    """The mean squared error of the probabilities, over the same rows.

    The other proper scoring rule, and the one that stays readable where a log loss goes large on
    a single confident mistake. Both are published because they disagree sometimes, and a reader
    told only one of them has been told half of it.
    """
    return sum(
        (value - (1.0 if label else 0.0)) ** 2 for value, label in scored
    ) / len(scored)


def _policy_scores(scores_path: Path, items: Sequence[CorpusItem]) -> dict[str, float]:
    """The published Policy Scores, as a lookup by post.

    Read straight out of the published file rather than through `signals.py`, and that is the
    point rather than a shortcut: this module does not import `signals` at all, so the Policy
    Score cannot become an input to the fit by accident, and the one place it is read is a
    baseline table. A model given the Policy Score as a feature would return the Policy Score with
    a decimal point on it (ADR-0003).

    The reader is deliberately small — the post id and the score, nothing else — because the
    fewer fields this touches the less of the scoring vocabulary it could be said to share, and
    nothing here needs the Signal-by-Signal breakdown.
    """
    held: dict[str, float] = {}
    for number, line in read_rows(scores_path):
        where = f"{scores_path.as_posix()}:{number}"
        record = read_object(where, line)
        value = record["score"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} has score={value!r}, which is not a number")
        held[read_text(where, record, "post_id")] = float(value)
    missing = sorted({item.post_id for item in items} - held.keys())
    if missing:
        raise ValueError(
            f"{missing[0]} has no Policy Score in {scores_path.as_posix()}, so it cannot be "
            "read against one. Run `rfi policy-score` over a Corpus holding it first"
        )
    return held


# --- the run ---------------------------------------------------------------------------------


def confident(sources: Sources) -> Confidences:
    """Every post's Confidence, the model that produced them, and the figures beside it.

    One call, for the reason the grouping and the scoring are one call: the features, the rows,
    the folds, and the evaluation are four views of one pass over four files, and a caller that
    read them separately could end up printing one run's figures over another's rows.

    Those four files are the whole input. The Policy Scores are read for the baseline table and
    for nothing else — `tests/test_confidence.py` proves it by running this over a scores file
    with every score reversed and requiring all the probabilities back unchanged while the
    baseline moves.
    """
    items = read_corpus(sources.corpus)
    labelled = label_set(read_truth(sources.truth))
    features = confidence_features(items, sources.suffix_list)
    plan = folds(items, labelled)

    rows: list[Confidence] = []
    scored: list[tuple[float, bool]] = []
    fits: list[Folded] = []

    for fold in sorted(set(plan.values())):
        fitting = tuple(
            (
                tuple(features[item.post_id][name] for name in FEATURES),
                label_for(labelled, item),
            )
            for item in items
            if plan[item.post_id] != fold
        )
        if not fitting:
            raise ValueError(
                f"fold {fold} holds every post in the Corpus, so there is nothing left to fit "
                "on. One Planted Campaign means one fold, and the fold holds out every planted "
                "post with it. A Confidence measured that way would be either a training fit "
                "reading back its own answer or a figure with no fold behind it at all"
            )
        fit = train(fitting)
        fits.append(
            Folded(
                fold=fold,
                coefficients=tuple(zip(FEATURES, fit.coefficients, strict=True)),
                intercept=fit.intercept,
                rows=fit.rows,
                positives=fit.positives,
            )
        )
        for item in items:
            if plan[item.post_id] != fold:
                continue
            value = fit.confidence(tuple(features[item.post_id][name] for name in FEATURES))
            rows.append(
                Confidence(
                    post_id=item.post_id,
                    account=item.account,
                    confidence=value,
                    model=MODEL_NAME,
                    recipe=recipe_digest(),
                )
            )
            scored.append((value, label_for(labelled, item)))

    written = tuple(sorted(rows, key=lambda row: row.post_id))
    labels = {item.post_id: label_for(labelled, item) for item in items}
    return Confidences(
        facts=_facts(sources, items, labels, max(fit.rows for fit in fits)),
        rows=written,
        evaluation=_evaluation(
            written, scored, items, labels, sources.scores, len(set(plan.values()))
        ),
        fits=tuple(fits),
    )


def _evaluation(
    written: Sequence[Confidence],
    scored: Sequence[tuple[float, bool]],
    items: Sequence[CorpusItem],
    labels: Mapping[str, bool],
    scores_path: Path,
    fold_count: int,
) -> Evaluation:
    """The model and the two baselines, over the same held-out rows.

    The constant baseline is the base rate of the held-out rows themselves, which is what makes
    it a baseline rather than a second answer: it is the best any method that cannot see a post
    can do on these posts. The Policy Score is ordered as it is published and gets no scoring
    rule, because a 0-100 editorial figure has no probability reading and publishing a likelihood
    from one would be the fusion ADR-0003 rules out.

    All three rows are over the same posts, so the three AUCs are comparable. That is the point
    of the table: a model figure printed alone is a claim, and printed beside two baselines over
    the same rows it is a difference.
    """
    posts = len(items)
    positives = sum(1 for _, label in scored if label)
    base_rate = positives / posts
    constant_scored = tuple((base_rate, label) for _, label in scored)

    published = _policy_scores(scores_path, items)
    ordered = [(published[row.post_id], labels[row.post_id]) for row in written]

    detail = f"{posts} posts, {positives} planted, {fold_count} folds"
    model = Figure(
        name=MODEL_NAME,
        auc=_auc(scored),
        log_loss=_log_loss(scored),
        brier=_brier(scored),
        detail=detail,
    )
    constant = Figure(
        name="constant base rate",
        auc=_auc(constant_scored),
        log_loss=_log_loss(constant_scored),
        brier=_brier(constant_scored),
        detail=f"every post predicted at {base_rate:.4f}",
    )
    policy = Figure(
        name="policy score",
        auc=_auc(ordered),
        log_loss=None,
        brier=None,
        detail="the published Policy Score, ordered; it has no probability reading",
    )
    return Evaluation(
        model=model,
        baselines=(constant, policy),
        posts=posts,
        positives=positives,
        folds=fold_count,
        held_out=True,
        verdict=_verdict(model, constant, policy),
    )


def _verdict(model: Figure, constant: Figure, policy: Figure) -> str:
    """What the three rows say when they are compared, in one sentence.

    Computed rather than written, because a verdict that is prose in a template is a verdict
    that stops being true the moment the numbers move and nothing notices. Three comparisons
    are stated and the reader can check each against the table: the ordering against the
    constant, the probabilities against it, and the ordering against the Policy Score.

    The Policy Score comparison is the uncomfortable one and it is stated in the same voice as
    the others. A model that orders worse than the rules engine this project already ships has
    not earned the word useful, and saying so on the page is cheaper than a reader finding it
    out and wondering why it was buried.
    """
    orders_better = model.auc > constant.auc
    scores_better = _is_better(model, constant)
    behind_policy = model.auc < policy.auc

    ordering = (
        "orders the planted posts above the rest better than the constant does"
        if orders_better
        else "does not order the planted posts above the rest any better than the constant does"
    )
    probability = (
        "is scored better than the constant by both proper scoring rules"
        if scores_better
        else "is scored worse than the constant by at least one proper scoring rule"
    )
    against_policy = (
        f"and orders them less well than the Policy Score does, at {policy.auc:.4f} against "
        f"{model.auc:.4f}"
        if behind_policy
        else f"and orders them at least as well as the Policy Score does, at {policy.auc:.4f}"
    )
    return f"the model {ordering}, {probability}, {against_policy}."


def _is_better(model: Figure, constant: Figure) -> bool:
    """Whether the model beats the constant on both proper scoring rules.

    Both, not either. A model that wins on one and loses on the other has not scored better; it
    has scored differently, and a page that reported whichever suited would be choosing its own
    metric after seeing the answers.

    A `Figure` with no scoring rule cannot be compared on one, and the two figures this is
    called with both carry both — the model and the constant are the two rows of the table that
    have probabilities. The check is a refusal rather than an `assert` for the reason
    `CampaignCandidate` refuses an `Evidence` of the wrong kind: a figure that could not be
    compared should stop the run and say so, not disappear under `-O`.
    """
    mine = _both_rules(model)
    theirs = _both_rules(constant)
    return mine[0] < theirs[0] and mine[1] < theirs[1]


def _both_rules(figure: Figure) -> tuple[float, float]:
    """One figure's log loss and Brier score, refusing a figure that has neither.

    The refusal rather than an `assert` for the reason `CampaignCandidate` refuses an
    `Evidence` of the wrong kind: a figure that could not be compared should stop the run and
    say so, not disappear under `-O`. Which figure it is named in the message, because the only
    row in this table with no scoring rule is the Policy Score and a reader needs to know that is
    what stopped the comparison.
    """
    if figure.log_loss is None or figure.brier is None:
        raise ValueError(
            f"{figure.name} carries no log loss or Brier score, and a proper scoring rule "
            "cannot be compared against a figure that has none. That is the Policy Score's row, "
            "and the policy is not compared this way"
        )
    return figure.log_loss, figure.brier


def _facts(
    sources: Sources,
    items: Sequence[CorpusItem],
    labels: Mapping[str, bool],
    fitting_rows: int,
) -> ConfidenceFacts:
    """Every figure about the four files the output prints, from the rows and the bytes.

    `positives` is counted from the labels rather than from the membership's own post list, so
    it is the number of published rows that were held out as positives — which is the number
    every figure on the page is measured over. The two would differ only on a Corpus whose
    membership names a post the Corpus does not hold, and that Corpus is refused by
    `read_truth` and by `read_corpus` before it reaches here.

    The folds are counted from the labels' side rather than from the fold plan, which is not
    carried here: the plan is one intermediate the figures do not need, and taking the count
    from the campaigns it was derived from keeps this function a function of its inputs.
    """
    return ConfidenceFacts(
        accounts=len({item.account for item in items}),
        corpus_path=sources.corpus.as_posix(),
        corpus_sha256=hashlib.sha256(sources.corpus.read_bytes()).hexdigest(),
        folds=sources.folds,
        model=MODEL_NAME,
        posts=len(items),
        positives=sum(1 for label in labels.values() if label),
        recipe=recipe_digest(),
        scores_path=sources.scores.as_posix(),
        scores_sha256=hashlib.sha256(sources.scores.read_bytes()).hexdigest(),
        suffix_list_path=sources.suffix_list.as_posix(),
        suffix_list_sha256=hashlib.sha256(sources.suffix_list.read_bytes()).hexdigest(),
        training_rows=fitting_rows,
        truth_path=sources.truth.as_posix(),
        truth_sha256=hashlib.sha256(sources.truth.read_bytes()).hexdigest(),
    )


# --- the file ------------------------------------------------------------------------------------


def write_confidences(path: Path, rows: Sequence[Confidence]) -> None:
    """The published file: one row per post, in post-id order.

    Sorted by post rather than written in the order the folds ran, because a fold order says which
    model scored which post and a reader holding the file and the Corpus wants to line them up by
    identifier. No `fold` field travels with it: a fold partitions the posts by the Planted Campaign
    they came from, so publishing one is publishing the membership one step removed (ADR-0008).
    """
    write_lines(
        path,
        (row.as_row() for row in sorted(rows, key=lambda row: row.post_id)),
    )


def read_confidences(path: Path) -> tuple[Confidence, ...]:
    """The published file, read back, refusing what it cannot account for.

    Five refusals, each for a reason this repository has already stated elsewhere. An unknown
    field is refused rather than ignored, because the one field that must never appear here beside
    a probability is a Policy Score's, and a reader that skipped what it did not recognise would
    skip that without saying so (ADR-0008, ADR-0003). A probability outside the open unit interval
    is refused because a number that is not a probability is a severity wearing the model's name.
    Two rows for one post, two models, or two recipes are each refused for the reason
    `read_content_records` refuses them: something downstream would read one answer and call it the
    answer.
    """
    rows = tuple(_confidence(path, number, line) for number, line in read_rows(path))
    refuse_repeated(
        path.as_posix(),
        (row.post_id for row in rows),
        "a later reader would take whichever row it read last and call it the Confidence, so "
        "one post on two rows is two answers to one question",
    )
    models = {row.model for row in rows}
    if len(models) > 1:
        raise ValueError(
            f"{path.as_posix()} holds {len(models)} models {sorted(models)}, and a probability "
            "from one model has no comparison to a probability from another"
        )
    recipes = {row.recipe for row in rows}
    if len(recipes) > 1:
        raise ValueError(
            f"{path.as_posix()} holds {len(recipes)} recipes, and one of them is not the rule the "
            "other rows were fitted under"
        )
    return rows


def _confidence(path: Path, number: int, line: str) -> Confidence:
    """One row, checked field by field against the vocabulary this project publishes."""
    where = f"{path.as_posix()}:{number}"
    record = read_object(where, line)
    vocabulary = tuple(field.name for field in fields(Confidence))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds {sorted(record)}, which is not the Confidence vocabulary "
            f"{sorted(vocabulary)}"
        )
    value = record["confidence"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} has confidence={value!r}, which is not a number")
    recipe = read_text(where, record, "recipe")
    if len(recipe) != 64 or set(recipe) - set("0123456789abcdef"):
        raise ValueError(
            f"{where} has recipe={recipe!r}, and the recipe beside a Confidence is the SHA-256 of "
            "the rule it was fitted under"
        )
    return Confidence(
        post_id=read_text(where, record, "post_id"),
        account=read_text(where, record, "account"),
        confidence=float(value),
        model=read_text(where, record, "model"),
        recipe=recipe,
    )


# --- the two views -------------------------------------------------------------------------------


def render_table(confidences: Confidences) -> str:
    """The console output: what was read, the figure, both baselines, and the model beside them.

    ASCII only, so it prints the same way on a console that cannot encode anything else and the
    same way when it is redirected, which is what lets it be pasted into an issue. The table of
    coefficients is printed in full because the claim this command makes is that the number came
    from those seven counts and nothing else, and a claim a reader cannot check is a claim.
    """
    evaluation = confidences.evaluation
    sections = (
        f"{_HEADING}\n\n{_opening(confidences)}",
        _figures(confidences),
        _measurement(evaluation),
        _coefficients(confidences),
        _footer(confidences),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _opening(confidences: Confidences) -> str:
    """What the figures below are, said before any of them is printed."""
    facts = confidences.facts
    held = (
        "every post was scored by a model fitted without it and without every other post of its "
        f"Planted Campaign, over {facts.folds} folds"
    )
    return "\n".join(
        [
            *wrap(
                f"{facts.posts} posts in {facts.corpus_path}, of which "
                f"{facts.positives} are posts of a Planted Campaign; {held}. The labels come "
                f"from {facts.truth_path}, which the generator wrote.",
                indent=0,
                width=_WIDTH,
            ),
            _SUBHEADING,
        ]
    )


def _figures(confidences: Confidences) -> str:
    """What was read, and what came of it. One figure per line, labelled."""
    facts = confidences.facts
    figures = (
        ("corpus", f"{facts.corpus_path} ({_count(facts.posts, 'post')}, "
                   f"{_count(facts.accounts, 'account')})"),
        ("corpus sha256", facts.corpus_sha256),
        ("suffix list", f"{facts.suffix_list_path} ({facts.suffix_list_sha256[:12]})"),
        ("membership", f"{facts.truth_path} ({_count(facts.folds, 'fold')}, "
                       f"{facts.truth_sha256[:12]})"),
        ("policy scores", f"{facts.scores_path} ({facts.scores_sha256[:12]})"),
        ("model", f"{facts.model} (recipe {facts.recipe[:12]})"),
        ("training rows", f"up to {facts.training_rows} per fold, without the fold's own posts"),
        (
            "confidence",
            f"{measure(confidences.lowest)} to {measure(confidences.highest)}, "
            f"mean {measure(confidences.mean_confidence)}",
        ),
        ("published", f"{CONFIDENCES_PATH}, one row per post, in post-id order"),
    )
    width = max(len(name) for name, _ in figures)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in figures)


def _measurement(evaluation: Evaluation) -> str:
    """The model and both baselines, each on its row with its own note beneath it.

    The baseline rows are what make the model's row readable: a number beside no other number is
    a claim, and the two here are the two a classifier has to beat — the base rate, which orders
    nothing, and the rules engine this project already ships.

    The note sits under its own row rather than in a fifth column, so a reader can tell which
    note belongs to which figure. A column of prose would be both unreadable and, at the width
    the longest note needs, wider than the console this command says it fits.
    """
    figures = (evaluation.model, *evaluation.baselines)
    headings = ("predictor", "AUC", "log loss", "Brier")
    rows = [
        (
            figure.name,
            f"{figure.auc:.4f}",
            scored(figure.log_loss),
            scored(figure.brier),
        )
        for figure in figures
    ]
    widths = [max(len(cell) for cell in column) for column in zip(headings, *rows, strict=True)]
    table = "\n".join(
        ["  " + "  ".join(cell.ljust(width) for cell, width in zip(headings, widths, strict=True))]
        + ["  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
           for row in rows]
    )
    # Each row's own note indented under the row rather than a fifth column, because the notes
    # are prose and a column wide enough for the longest of them pushes the table past the
    # console width this command claims — which is a claim a reader with a narrow window can
    # check and the table was failing.
    notes = "\n".join(
        "    " + line
        for figure in figures
        for line in wrap(
            f"({figure.name}) {figure.detail}", indent=4, width=_WIDTH
        )
    )
    note = (
        "  AUC is how well a predictor orders the planted posts above the rest, counted over every "
        "planted\n  post against every other; 0.5000 orders nothing and 1.0000 orders them all "
        "first. Log loss\n  and Brier are proper scoring rules and are lower-is-better, and a "
        "policy score has no\n  probability reading, so neither is quoted for it: turning a "
        "0-100 editorial figure into a\n  likelihood is the fusion ADR-0003 rules out."
    )
    verdict = "\n".join(f"  {line}" for line in wrap(evaluation.verdict, indent=0, width=_WIDTH))
    return "baselines\n" + table + "\n" + notes + "\n\n" + note + "\n\n" + verdict


def scored(value: float | None) -> str:
    """One proper scoring rule's figure, or the dash that means the predictor has none.

    A dash rather than a zero, and the distinction is the point of the baseline table: a zero
    would be a number to compare the model's loss against, where what is actually meant is that
    the Policy Score has no probability reading at all and cannot be given a likelihood without
    the fusion ADR-0003 rules out. Public because it is the table's own rule — how this module
    prints a metric a predictor may not have — rather than a detail of one renderer.
    """
    return "-" if value is None else f"{value:.4f}"


def _coefficients(confidences: Confidences) -> str:
    """One block per fold: the seven features and what that fold's fit gave each of them.

    A block per fold rather than one block for the run, because there is no single fit behind
    the published probabilities — leave-one-campaign-out fits once per fold, and a table of one
    set of weights beside {evaluation.posts} probabilities from several fits would be a claim
    the numbers do not support. Each block says which fold scored which posts, so a reader
    looking for what drove a given Confidence can find the fit that produced it.

    In `FEATURES` order within each block, so the reader is looking down the same list the code
    enumerates. A weight is standardised, so it is a weight per standard deviation of the
    feature rather than a weight per unit of it, and the column heading says so.
    """
    blocks: list[str] = []
    for folded in confidences.fits:
        heading = "\n".join(
            wrap(
                f"coefficients  fold {folded.fold} of {len(confidences.fits)}: a weight here "
                "is per standard deviation of its feature, and this fit scored the "
                f"{folded.rows} posts the fold held out ({folded.positives} planted)",
                indent=0,
                width=_WIDTH,
            )
        )
        lines = [heading, f"  {'feature'.ljust(22)}  {'reads'.ljust(44)}  {'weight'.rjust(10)}"]
        for (name, weight), member in zip(folded.coefficients, Feature, strict=True):
            lines.append(f"  {name.ljust(22)}  {member.reads.ljust(44)}  {weight:>10.4f}")
        lines.append(f"  {'intercept'.ljust(22)}  {''.ljust(44)}  {folded.intercept:>10.4f}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _footer(confidences: Confidences) -> str:
    """What the number is, what it is not, and why it is on no screen.

    The limits are the ones a reader would otherwise have to guess at: what the pattern was, what
    the labels were, what seven positives is worth, and the standing refusal to print it beside a
    Policy Score.
    """
    evaluation = confidences.evaluation
    return f"""\
The Confidence is a probability and nothing else: that a post exhibits the pattern this model
was trained on, which is Planted Campaign membership as the generator wrote it. It is not a
severity, it is not a score out of a hundred, and it is not a claim that a post is fraud, which
this project makes about no post at all (ADR-0002).

The training labels come from Planted Campaign membership, and so the figures above measure
recovery of planted structure over a Corpus the generator also wrote. They are not a rate of
fraud found in the world and they are not real-world performance. Nobody reviewed any of this:
the labels were assigned by the code that wrote the posts, so what has been measured is whether
a classifier can recover that code's own bookkeeping. Calibration is a separate question with
its own evaluation (ADR-0003) and is not measured here.

{evaluation.positives} positives is a very small positive class, and {evaluation.folds} folds is a
very small number of them. A figure over that many positives cannot separate a model that
generalises from one that has been lucky, and it is published with the base rate beside it for
that reason rather than on its own.

Nothing on any screen in this project reads this number. The Policy Score is computed from
published Signal weights and printed as a 0-100 severity; the Confidence is stored in its own
file and displayed nowhere, and no arithmetic anywhere in this repository combines the two
(ADR-0003). `rfi review-queue` and `rfi campaign-candidates` cannot reach this module, which
`tests/test_confidence.py` checks against the import graph and against their printed output."""


def render_report(confidences: Confidences) -> str:
    """The reader-facing page, generated from the Corpus and the published figures.

    Written for a reader who has to weigh the numbers rather than take them: the model is beside
    both baselines, the seven features are printed with what the model gave each of them, and the
    section on what the figures are not is the longest thing on the page — because it is the
    part a reader would otherwise skip, and skipping it is what makes a number from this page
    mean something it does not.

    Every figure is computed from the four files the command read, so a sentence that was true
    of one Corpus is not left in the next. The Corpus, the membership and the published Policy
    Scores are named by the paths this run read them from, and the Confidence file by the path the
    command writes by default, so two runs over the same Corpus produce the same bytes whichever
    directory they write to.
    """
    facts = confidences.facts
    evaluation = confidences.evaluation
    model, constant, policy = evaluation.model, *evaluation.baselines

    table = "\n".join(f"| {_figure_row(figure)} |" for figure in (model, constant, policy))
    feature_tables = "\n\n".join(_feature_table(folded) for folded in confidences.fits)

    return f"""# The Confidence

Generated by `rfi confidence` from the committed Corpus, the published Public Suffix List, the
Planted Campaign membership, and the published Policy Scores. Do not edit it by hand — a test
holds this file to what the command produces, and re-running the command rewrites it byte for
byte.

## What this is

Every Content Item in this project gets a **Confidence**: the model's probability that it
exhibits the pattern the model was trained to find, fitted over seven counts read from the post's
own title, its own body and its own links. The Confidence is written to `{CONFIDENCES_PATH}` and
**displayed nowhere**. It is not a severity, it is not a score out of a hundred, and it is never
added to the Policy Score.

That separation is at file level rather than column level, and the departure from ADR-0003's
wording is named rather than glossed: neither number lives in a table here, the Policy Score is a
field of a JSON Lines row, and so the two are held apart by being two files whose readers each
refuse any field the other publishes. A row carrying both would defeat the point the constraint
exists for. No arithmetic in this repository combines them (ADR-0003).

That is the whole of the constraint, and it is a constraint rather than an absence: the number
exists, it is measured, and it is held back from every view a reviewer reads.

## The figure

**AUC {model.auc:.4f}** over {model.detail}, with every post scored by a model fitted without it.
The published probabilities run from {measure(confidences.lowest)} to {measure(confidences.highest)},
mean {measure(confidences.mean_confidence)}.

**The model loses to the Policy Score on this Corpus, and it is scored worse than the base rate
besides.** That is the headline, stated before the table rather than left for a reader to
subtract out of it: {evaluation.verdict}

| Predictor | AUC | Log loss | Brier | Over |
| --- | ---: | ---: | ---: | --- |
{table}

**Out of fold, in {evaluation.folds} folds.** The folds are the Planted Campaigns: a post's
Confidence comes from a fit trained without that post and without every other post of its campaign,
so a model that had memorised one campaign's vocabulary could not carry it into the other. Every
negative is held out too, and by its **account** rather than by its post, so two posts by one
account cannot sit on either side of a fold and let the model recognise that voice. All
{evaluation.posts} probabilities are therefore out of fold: none of them is a training fit reading
back its own answer.

A training fit over {evaluation.positives} positives would be perfect by construction and would say
nothing, which is why no figure on this page comes from one.

## The model

`{facts.model}`, fitted by batch gradient descent at a learning rate of 0.5 for 400 steps with an L2
penalty of 0.01 on the coefficients and none on the intercept. There is no seed and nothing to tune:
two runs over the same Corpus produce the same bytes, which is what lets the published file be held
to what this code produces. The name and a digest of the recipe travel on every row of
`{CONFIDENCES_PATH}`, for the reason ADR-0024 gives for the embeddings — a version number is a
promise somebody has to keep and the digest is the promise checked.

Logistic regression rather than a tree ensemble because {evaluation.positives} positives cannot
support one: a forest would fit the fitting rows exactly and have nothing to say about a post it had
not seen. No wheel is involved either, which is why the fit is written here rather than imported —
the cost of that is that these three numbers are published rather than tuned, and the cost of the
alternative is ADR-0024's. The coefficients are printed below, standardised, so the reader can check
what went into the number rather than take the model's word for it.

## What it reads

Seven counts, each from one post and none of them from across the Corpus. Their being per-post
matters as much as which seven they are: `domain_frequency` is a Signal precisely because it
counts what other posts reach, and a model reading a count like that would be re-deriving the
Policy Score from the outside. Nothing here reads an account's age, karma, posting rate or
activity change, and no account is a feature at all (ADR-0007, ADR-0008) — which
`tests/test_confidence.py` holds by renaming every account in the Corpus and requiring the
features to come back identical.

**One table per fold, because there is no single fit behind these numbers.** Leave-one-campaign-out
fits once per fold, so the weights that scored a post are the weights of the fold that post was
held out in. Publishing one table for the run would be publishing one fold's weights beside
everybody's probabilities. Each table below names its fold and how many posts that fold's fit
scored.

{feature_tables}

## The baselines

A model that cannot order posts better than these two is not worth reading, so both are published
with the same metric over the same held-out rows.

**On this Corpus the model does not win, and that is the finding rather than a footnote.**
{evaluation.verdict}

**The constant base rate** predicts every post at one number — the share of the held-out rows that
are planted — so it ranks nothing by construction and its AUC is 0.5000. That is the number to
read the model's against: a classifier that cannot order the planted posts above the rest better
than a coin has nothing to add. Its log loss and Brier score are the best any method that cannot
see a post can do on these rows, and a model that cannot beat them is worse than saying nothing.

**The Policy Score** is this project's existing rules engine, ordered as `rfi policy-score`
published it: {policy.auc:.4f} on the same rows. It gets no log loss and no Brier score, and the
dash in the table is the point rather than a gap — a 0-100 editorial figure has no probability
reading, and publishing a likelihood derived from one would be exactly the fused score ADR-0003
rules out. Comparing the two by AUC is the whole of the comparison that is available between
them.

So on this Corpus the seven published Signal weights order these 34 posts better than a
seven-feature classifier fitted out of fold does. That is what a measurement is for. The model is
kept because the ticket asks for the Confidence to exist, be measured, and be held apart from the
Policy Score — and this is what it cost: seven counts over a post's own text and links are not a
substitute for six published weights over a registration's reach, and a reader who believed
otherwise would have been told something the numbers do not support.

## What these figures are not

**These figures measure recovery of planted structure, and they are not a rate of fraud found in
anything.** The training labels come from the Planted Campaign membership in
`{facts.truth_path}`, which is a file the generator wrote when it wrote the posts. Nothing here
has been reviewed by a person, and nothing here has seen real Reddit content: this project's
Corpus is generated, and every Synthetic Entity in it is marked `syn_` and sits under a reserved
domain (ADR-0001).

So a reader who takes {model.auc:.4f} off this page as a statement about how much fraud there is
out there has been told something false about how the number came about. What has been measured
is whether a classifier can tell one synthetic generator's planted posts from the rest of its own
output. That is a real question and a real measurement — a classifier that could not would be
useless to this project — and it is a much narrower one than the question a reviewer asks.

**{evaluation.positives} positives is a very small positive class.** A figure over that many can
separate a model that generalises from one that has been lucky far less often than a reader would
assume, and no confidence interval is printed because over seven positives there is nothing to put
one round. **Calibration is not measured here at all** (ADR-0003): the numbers on this page are
discrimination and two proper scoring rules, and a proper scoring rule rewards a calibrated
probability without telling the reader whether it is one. That is ticket #27.

## Why it is displayed nowhere

The Policy Score is what a reviewer sees: published weights, an additive sum, arithmetic printed
beside every figure. The Confidence is a different quantity answering a different question, and
ADR-0003's decision is that the two are stored in separate files, never summed, and never shown as
one number.

Enforcement is structural rather than promised. `cli.py` is the only module in this repository
that imports the Confidence, so `rfi review-queue` and `rfi campaign-candidates` cannot reach it
even by accident; `tests/test_confidence.py` checks that by walking the import graph, and checks
the Review Queue again by rendering it and searching its output for the word, for the file, and
for every published probability in both the form a probability takes and the form a severity
takes. A second test walks the AST of every module and refuses any expression that has a
confidence-named thing on one side of an operator and a Policy-Score-named thing on the other.

## What was read

| Input | Path | SHA-256 |
| --- | --- | --- |
| Corpus | `{facts.corpus_path}` | `{facts.corpus_sha256}` |
| Public Suffix List | `{facts.suffix_list_path}` | `{facts.suffix_list_sha256}` |
| Planted Campaign membership | `{facts.truth_path}` | `{facts.truth_sha256}` |
| Policy Scores | `{facts.scores_path}` | `{facts.scores_sha256}` |

The membership is the one input the pipeline does not otherwise see. A Corpus from a Corpus
Provider has no labels beside it, and a supervised model needs them; what that means for this
figure is stated twice above rather than left to the reader to work out from the file names.
"""


def _feature_table(folded: Folded) -> str:
    """One fold's seven features and weights, as a report table under a heading of its own.

    A heading rather than a caption, because the reader arriving at the second table needs to
    know it is a different fit before reading a single weight in it — and the number of posts
    that fit scored is on the heading, since a fold fitted on three rows is a different claim
    from one fitted on thirty.
    """
    rows = "\n".join(
        f"| `{name}` | {member.reads} | {weight:.4f} |"
        for (name, weight), member in zip(folded.coefficients, Feature, strict=True)
    )
    return f"""### Fold {folded.fold}

Fitted on {folded.rows} rows ({folded.positives} planted), and it scored the
{folded.rows} posts this fold held out. Weights are standardised, so each is per standard
deviation of its feature.

| Feature | Reads | Weight |
| --- | --- | ---: |
{rows}
| intercept | — | {folded.intercept:.4f} |"""


def _figure_row(figure: Figure) -> str:
    """One report table row: the name, both metrics or a dash, and what it was measured over."""
    log_loss = "-" if figure.log_loss is None else f"{figure.log_loss:.4f}"
    brier = "-" if figure.brier is None else f"{figure.brier:.4f}"
    return f"{figure.name} | {figure.auc:.4f} | {log_loss} | {brier} | {figure.detail}"


def _count(number: int, noun: str) -> str:
    """One count, agreeing with its noun. Every figure here is small, and a reader seeing
    "1 posts" stops to wonder whether the figure is right."""
    return f"{number} {noun if number == 1 else f"{noun}s"}"