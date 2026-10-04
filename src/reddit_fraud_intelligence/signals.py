"""Link Signals, Content Signals, the published weights, and the Policy Score.

The Policy Score is an additive rules engine and deliberately nothing else: every
Signal that contributes is computable from a post's own text and links, the weights
are published, and no model output feeds the number (ADR-0007). The arithmetic is
the claim this project makes to a reviewer — take what is visible in front of you,
apply the weights, arrive at the displayed figure — so the weight set lives in
`data/signals/weights.jsonl` rather than beside the rules below. Adding a Signal is
then a published change a reader can argue with, and not a line of Python.

A Signal is present or absent. A post that links three shared registrations, or that
uses four phrases from one Content Signal's list, carries that Signal once rather than
once per match: the score is a subset-sum of the published weights, which is the
arithmetic a reader can hold in their head, and it does not reward a post for how much
of it there is. Every Signal that fires names what fired it, so a reader who doubts one
can look at those and find the evidence behind them.

Two Signals read the Corpus's links, and both are Link Signals because neither can be
computed from one post alone. That is not account history: no account's age, karma,
posting rate, or activity change is read anywhere in this path, and none exists in the
Corpus file to read (ADR-0008). What is read is which registrations other posts' links
resolve to, which is structure the reviewer can go and look at.

Three Signals read the post's own title and body against a list of phrases, and are
Content Signals because a single post is all they need. The list is printed beside the
weights, because a reviewer holding the post can only check a match if they can see
what the rule was looking for, and the sentence the phrase was found in is the evidence
rather than the phrase alone: it is what a reader reads to decide whether the post
really makes the claim.

Nothing here reads the truth file or the Nuisance Structure manifest. The Corpus, the
published Public Suffix List, the shared-infrastructure list, and the weight set are the
whole input.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.domains import PostDomains, post_domains
from reddit_fraud_intelligence.infrastructure import SharedInfrastructure
from reddit_fraud_intelligence.jsonl import (
    JsonObject,
    read_names,
    read_object,
    read_rows,
    read_text,
    read_vocabulary,
    refuse_repeated,
    write_lines,
)
from reddit_fraud_intelligence.suffixes import PublicSuffixes
from reddit_fraud_intelligence.text import matcher, sentences, spans, wrap

_HEADING = "Policy Score"
_SUBHEADING = """\
A rules engine, and nothing else. Every Signal below is computed from a post's own \
text and links and from the registrations those links resolve to, and the weights \
are the published ones. No model output feeds this number and none ever will \
(ADR-0007)."""

# The threshold `domain_frequency` turns on, and the reason it is this one rather
# than another: it is the same threshold ADR-0005 puts two accounts in a Campaign
# Candidate over. Below it a registration is one account's site, which is what a
# link in a post usually is. At it, the registration has stopped being a personal
# site and started being infrastructure somebody else can reach.
_MIN_ACCOUNTS = 2


class SignalName(StrEnum):
    """Every Signal this module can produce, and every name the weight set may hold.

    The enum is the contract between the rules below and the published weight file:
    `Weights.read` refuses a file naming a Signal that does not exist here, and
    refuses one that leaves a Signal here unweighted. Adding a Signal therefore
    cannot be done by adding an evaluator alone — the run stops and names it.

    Declaration order is the order Signals are printed in, which is stated here rather
    than derived from a set so a run over the same Corpus writes the same file.
    """

    DOMAIN_FREQUENCY = "domain_frequency"
    DOMAIN_LOOKALIKE = "domain_lookalike"
    GUARANTEED_RETURN = "guaranteed_return"
    PAYMENT_REQUEST = "payment_request"
    URGENCY_LANGUAGE = "urgency_language"


# Every phrase a Content Signal matches, and the whole of each Signal: nothing outside
# these strings fires one. Published by being printed rather than held in a file
# alongside the weights, because a phrase list is a rule and not a number — a reader who
# disagrees with one has to be able to see it and quote it, and `tests/test_policy_score.py`
# reads this table back out of the command's own output.
_PHRASES: Mapping[SignalName, tuple[str, ...]] = {
    SignalName.GUARANTEED_RETURN: (
        "assured",
        "cannot lose",
        "can't lose",
        "guaranteed",
        "no risk",
        "no way it fails",
        "no way to lose",
        "risk free",
        "risk-free",
        "safe return",
        "will not lose",
        "won't lose",
    ),
    SignalName.PAYMENT_REQUEST: (
        "buy up front",
        "deposit before",
        "deposit for",
        "equipment charge",
        "kit fee",
        "materials deposit",
        "money up front",
        "pay up front",
        "refundable deposit",
    ),
    SignalName.URGENCY_LANGUAGE: (
        "48-hour",
        "act now",
        "before it goes",
        "fills fast",
        "fills up fast",
        "last chance",
        "limited time",
        "place goes to",
        "slot goes to",
        "while it lasts",
        "within 48 hours",
    ),
}

_CONTENT_SIGNALS = tuple(signal for signal in SignalName if signal in _PHRASES)

# Words that cancel a match wherever they stand in the sentence before it. A bare `no`
# is not among them and is handled in `_negated` instead, because it is the one English
# word in the set that carries no negation of its own.
_NEGATORS = frozenset(
    {
        "aren't",
        "can't",
        "cannot",
        "didn't",
        "doesn't",
        "don't",
        "isn't",
        "neither",
        "never",
        "nor",
        "nothing",
        "not",
        "without",
        "won't",
        "wouldn't",
    }
)

# A run of letters and digits, with apostrophes inside a word rather than at its edges,
# so `don't` is one word and can be a negator.
_WORD = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class SignalWeight:
    """One published weight, and the one-line reason it is that number.

    The rationale is in the file rather than in this module so it travels with the
    number it argues for: a reader who wants to disagree with a weight should not
    have to find the argument in the source, and a weight whose reason lived in the
    code could be edited without its reason.
    """

    signal: SignalName
    weight: int
    rationale: str


@dataclass(frozen=True, slots=True)
class Weights:
    """The published weight set, as the scoring path reads it.

    `total` is the denominator of the score and is carried rather than looked up, so
    the figure a reader divides by is the one that was published.
    """

    path: str
    entries: tuple[SignalWeight, ...]

    @property
    def total(self) -> int:
        return sum(entry.weight for entry in self.entries)

    def of(self, signal: SignalName) -> int:
        return next(entry.weight for entry in self.entries if entry.signal is signal)

    @classmethod
    def read(cls, path: Path) -> Weights:
        """The weight set, checked row by row and as a whole.

        Every row is checked because a weight file is meant to be edited by somebody
        who disagrees with a number: a misspelt Signal name, a weight of zero, and a
        rationale left blank are all mistakes a reader could not see in the output, and
        each of them changes what the score means.

        The set is checked as a whole for the same reason from the other side, and the
        comparison is over the list rather than the set of names: a row repeated passes a
        set comparison and would count its weight twice, inflating the denominator every
        score is a share of while the breakdown still named one Signal per post. A Signal
        with no published weight would either be scored at zero — a Signal the output
        names and the arithmetic ignores — or dropped, which is the failure the
        published-weights claim is made of. Every direction is refused, so the file and
        this module cannot describe different sets of Signals.
        """
        entries = tuple(_weight(path, number, text) for number, text in read_rows(path))
        published = [entry.signal for entry in entries]
        known = set(SignalName)
        if sorted(published) != sorted(known):
            raise ValueError(
                f"{path.as_posix()} publishes {sorted(signal.value for signal in published)}, "
                f"and this module computes {sorted(signal.value for signal in known)}; "
                "every Signal must be published exactly once and no others"
            )
        return cls(path=path.as_posix(), entries=tuple(sorted(entries, key=_by_signal)))


def _by_signal(entry: SignalWeight) -> str:
    return entry.signal.value


@dataclass(frozen=True, slots=True)
class DomainFrequency:
    """One registration, and how much of the Corpus reaches it.

    Both figures are the Corpus's and not this post's: the Signal is about the
    registration rather than about the post, and a post that happens to link it
    inherits the figure. `accounts` is what the Signal turns on; `posts` is printed
    because a reader counting accounts is asking how many places it was reached from.
    """

    domain: str
    posts: int
    accounts: int

    @property
    def signal(self) -> SignalName:
        return SignalName.DOMAIN_FREQUENCY

    @property
    def registrations(self) -> tuple[str, ...]:
        return (self.domain,)

    def line(self) -> str:
        return f"{self.domain}: {self.posts} posts by {self.accounts} accounts"


@dataclass(frozen=True, slots=True)
class ConfusablePair:
    """Two registrations one edit apart under the same Public Suffix.

    `domain` is the one a post linked and `other` is the one it is one edit from.
    Neither is marked as the copy, because the spelling does not say which imitates
    which and nothing else in the Corpus does either: the Corpus plants a recruitment
    firm and a warehouse employer, each a character from a Planted Campaign's
    registration, and a rule that guessed the direction would be guessing.
    """

    domain: str
    other: str

    @property
    def signal(self) -> SignalName:
        return SignalName.DOMAIN_LOOKALIKE

    @property
    def registrations(self) -> tuple[str, ...]:
        return (self.domain, self.other)

    def line(self) -> str:
        return f"{self.domain}: one edit from {self.other}"


@dataclass(frozen=True, slots=True)
class PhraseMatch:
    """One sentence of a post's own text, and the phrases that matched inside it.

    The sentence is the evidence rather than the phrase, for two reasons that point the
    same way. It is what a reviewer reads to decide whether the post really makes the
    claim the Signal names, and it is the whole of what the negation guard looked at, so
    a reader who thinks a match was wrongly cancelled has everything needed to see why
    it was. `phrases` is every published phrase that fired in it, so the reader does not
    have to re-apply the list to find out which of them did — though they are free to,
    because the list is printed too.

    `field` is carried because a claim in the title is not the claim in the fourth
    paragraph, and a breakdown that did not say which would be making one of them.
    """

    signal: SignalName
    field: str
    phrases: tuple[str, ...]
    sentence: str

    @property
    def registrations(self) -> tuple[str, ...]:
        return ()

    def line(self) -> str:
        return f'{self.field}: "{self.sentence}"'


Evidence = DomainFrequency | ConfusablePair | PhraseMatch


@dataclass(frozen=True, slots=True)
class SignalHit:
    """One Signal a post carries, its published weight, and the evidence behind it.

    The weight travels with the hit rather than being looked up when printing, so a
    breakdown cannot be read against a weight set that has changed since, and the
    evidence is the figures a reader would use to disagree with it.
    """

    signal: SignalName
    weight: int
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError(f"{self.signal} is present with no evidence for it")
        wrong = sorted(
            {item.signal.value for item in self.evidence if item.signal is not self.signal}
        )
        if wrong:
            raise ValueError(
                f"{self.signal} cannot rest on {wrong}; each Signal's evidence has to be "
                "of its own kind, or the breakdown would show a Signal standing on figures "
                "that mean something else"
            )

    def line(self) -> str:
        return "; ".join(item.line() for item in self.evidence)


@dataclass(frozen=True, slots=True)
class PostScore:
    """One post, its Policy Score, and the arithmetic behind the number.

    `points` is the sum of the published weights of the Signals this post carries, and
    `published_points` and `published_signals` are the whole of the published set, so
    the displayed `score` can be recomputed from this row alone — which is what makes
    the row worth writing.
    """

    post_id: str
    account: str
    score: int
    points: int
    published_points: int
    published_signals: int
    signals: tuple[SignalHit, ...]


@dataclass(frozen=True, slots=True)
class Reach:
    """What the Corpus's links do with one registration, counted once each way."""

    posts: int
    accounts: int


@dataclass(frozen=True, slots=True)
class LinkIndex:
    """What the Corpus's links add up to, which is the whole of what a Link Signal
    reads.

    One pass over the rows rather than a walk per post, because every post asks the
    same question about its own registrations and the answer must not depend on how
    large the Corpus is. The withheld registrations are carried rather than dropped:
    the output prints what was kept out and why, and a registration that is in the
    resolved links, scores nothing, and is not explained anywhere would be a gap in
    the output rather than a decision in it.

    `confusable` is every registration beside the ones it is one edit away from, and it
    is built over all of them including the withheld ones. A registration one edit from
    a link shortener everybody uses is what somebody imitating that shortener looks
    like, which is a case the shortener's presence creates rather than one it excuses,
    so the filter applies to `domain_frequency` alone.
    """

    reach: Mapping[str, Reach]
    withheld: frozenset[str]
    confusable: Mapping[str, tuple[str, ...]]

    @classmethod
    def of(cls, rows: Iterable[PostDomains], withheld: frozenset[str]) -> LinkIndex:
        """What every registration the Corpus's links resolve to is reached by, and what
        each one is one edit away from.

        Counted per registration and not per link: a post that links a link-in-bio page
        and the site behind it touches one registration, and counting the links would
        report that registration as reached from twice as many places as it is.
        """
        posts: dict[str, set[str]] = {}
        accounts: dict[str, set[str]] = {}
        for row in rows:
            for domain in row.domains:
                posts.setdefault(domain, set()).add(row.post_id)
                accounts.setdefault(domain, set()).add(row.account)

        reach = {
            domain: Reach(posts=len(found), accounts=len(accounts[domain]))
            for domain, found in posts.items()
        }
        return cls(
            reach=reach,
            withheld=withheld,
            confusable=_confusable(sorted(reach)),
        )

    def reached_by(self, domain: str) -> Reach:
        """What one registration is reached by. Never zero: a caller only asks about
        registrations its own post links, so the index has already seen them."""
        return self.reach[domain]

    def pairs(self) -> tuple[ConfusablePair, ...]:
        """Every confusable pair the Corpus contains, printed once rather than twice.

        Both members of a pair carry the Signal, so iterating `confusable` would print
        each pair from both ends. A pair is held together on the side that sorts first,
        which is a stable rule rather than a guess at which of the two came first.
        """
        return tuple(
            ConfusablePair(domain=mine, other=theirs)
            for mine, theirs in sorted(
                {
                    (min(mine, other), max(mine, other))
                    for mine, others in self.confusable.items()
                    for other in others
                }
            )
        )


def _confusable(registrations: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Which of these registrations are one edit away from which.

    An edit is an insertion, a deletion, a substitution, or a transposition — the four
    ways a name somebody already owns gets changed into another one — and one of them
    is the whole of the rule. Two edits apart is a different name: in a real Corpus
    every pair of long domains is within two edits of something, so a rule that kept
    going would eventually fire on all of them.

    The Public Suffix has to match. `vantage-ledger.example` and
    `vantage-ledgtr.co.uk` are one edit apart in their registered labels and say
    different things about who registered what, so they are not two spellings of one
    name. Registrations are sorted first, so the pairing does not depend on the order
    the Corpus happens to be read in.
    """
    found: dict[str, list[str]] = {domain: [] for domain in registrations}
    for index, mine in enumerate(registrations):
        my_label, my_suffix = _label_and_suffix(mine)
        for other in registrations[index + 1 :]:
            their_label, their_suffix = _label_and_suffix(other)
            if my_suffix == their_suffix and _one_edit_apart(my_label, their_label):
                found[mine].append(other)
                found[other].append(mine)
    return {domain: tuple(sorted(others)) for domain, others in found.items()}


def _label_and_suffix(registration: str) -> tuple[str, str]:
    """A registration into the label somebody registered and the Public Suffix behind
    it. A Registrable Domain is exactly one label plus its Public Suffix, so the first
    dot is the join."""
    label, _, suffix = registration.partition(".")
    return label, suffix


def _one_edit_apart(mine: str, theirs: str) -> bool:
    """Whether two labels differ by exactly one edit.

    Damerau-Levenshtein bounded at one, walked row by row: a transposition is the one
    case an edit-distance function that only counts insertions, deletions, and
    substitutions misses, and `ledder` for `ledger` is the copy somebody makes by
    typing it in the wrong order, so it has to count. The walk stops as soon as a whole
    row is above one: every cell in a row one from the top of the matrix is built from
    a cell of the two rows above plus one, and the transposition term is one of those,
    so a row that has already lost cannot come back — and neither can any row after it.
    That is what makes the bound at one an answer rather than an approximation.
    """
    if abs(len(mine) - len(theirs)) > 1:
        return False
    previous_two: list[int] | None = None
    previous = list(range(len(theirs) + 1))
    for row, left in enumerate(mine, start=1):
        current = [row] + [0] * len(theirs)
        for column, right in enumerate(theirs, start=1):
            transposed = (
                previous_two[column - 2] + 1
                if previous_two is not None
                and column > 1
                and left == theirs[column - 2]
                else _IMPOSSIBLE
            )
            current[column] = min(
                previous[column] + 1,  # theirs gained a character
                current[column - 1] + 1,  # mine gained a character
                previous[column - 1] + (left != right),  # one of them changed
                transposed,  # the two swapped places, which is one edit rather than two
            )
        if min(current) > 1:
            return False
        previous_two, previous = previous, current
    return previous[-1] == 1


_IMPOSSIBLE = 99


@dataclass(frozen=True, slots=True)
class ScoreFacts:
    """What reading the four published files establishes, stated as claims about bytes."""

    accounts: int
    corpus_path: str
    corpus_sha256: str
    posts: int
    posts_with_signals: int
    registrations: int
    registrations_withheld: int
    rules_sha256: str
    suffix_list_path: str
    suffix_list_sha256: str


@dataclass(frozen=True, slots=True)
class Scored:
    """Everything one run establishes, so the table and the file cannot disagree.

    `index` is carried rather than reduced to a count because the table prints every
    registration with the reach that decided whether its posts carry a Signal, which
    is the audit the per-Signal breakdown claims to be checkable against.
    """

    facts: ScoreFacts
    weights: Weights
    shared: SharedInfrastructure
    index: LinkIndex
    scores: tuple[PostScore, ...]


def score_corpus(
    corpus_path: Path, list_path: Path, shared_path: Path, weights_path: Path
) -> Scored:
    """Read the four published files and return every post's Policy Score.

    One call, for the same reason the grouping is one call: the figures, the scores,
    and the evidence they are read out of are three views of one pass over the Corpus,
    and a caller that resolved the links and then asked for the scores separately could
    end up printing one run's figures over another's results.

    The weights are read first. A weight file that does not account for every Signal
    stops the run before a single post is scored, because a run that scored some
    Signals and dropped others would print a total that no row adds up to.

    The resolved links are an argument rather than something read here. They are
    produced by the same `post_domains` the first command publishes, so the two cannot
    disagree unless that function is wrong — which its own tests hold it to. The
    published file is not read: the score is a function of the Corpus alone, and
    re-reading a derived file would make the result depend on whether somebody had
    remembered to run the step before it.
    """
    weights = Weights.read(weights_path)
    suffixes = PublicSuffixes.read(list_path)
    shared = SharedInfrastructure.read(shared_path, suffixes)
    items = read_corpus(corpus_path)
    rows = post_domains(items, suffixes)
    index = LinkIndex.of(rows, shared.withheld())

    scores = tuple(
        PostScore(
            post_id=row.post_id,
            account=row.account,
            score=_normalise(points, weights.total),
            points=points,
            published_points=weights.total,
            published_signals=len(weights.entries),
            signals=hits,
        )
        for item, row in zip(items, rows, strict=True)
        for hits in (_hits(item, row, index, weights),)
        for points in (sum(hit.weight for hit in hits),)
    )

    return Scored(
        facts=_facts(corpus_path, list_path, rows, index, scores, suffixes),
        weights=weights,
        shared=shared,
        index=index,
        scores=scores,
    )


def _hits(
    item: CorpusItem, row: PostDomains, index: LinkIndex, weights: Weights
) -> tuple[SignalHit, ...]:
    """The Signals one post carries, each with the evidence a reader checks it with.

    Every Signal is asked for on every post, and the one that is not there is simply
    absent: nothing here can decide not to look. They come out in `SignalName` order,
    so the order a breakdown is written in is a property of the enum rather than of the
    order the rules happen to be written below.

    Both halves of the post are handed over rather than one of them: the resolved links
    carry no text, and the text carries no resolved links, and a Signal is allowed to
    read either. The map below is exhaustive over the enum by construction — two Link
    Signals written out, and every Signal that has a phrase list — so a Signal added to
    the enum without an evaluator here stops the run rather than scoring nothing.
    """
    by_signal: dict[SignalName, tuple[Evidence, ...]] = {
        SignalName.DOMAIN_FREQUENCY: _domain_frequency(row, index),
        SignalName.DOMAIN_LOOKALIKE: _domain_lookalike(row, index),
        **{signal: _content(item, signal) for signal in _CONTENT_SIGNALS},
    }
    return tuple(
        SignalHit(signal=signal, weight=weights.of(signal), evidence=evidence)
        for signal in SignalName
        if (evidence := by_signal[signal])
    )


def _domain_frequency(row: PostDomains, index: LinkIndex) -> tuple[Evidence, ...]:
    return tuple(
        DomainFrequency(
            domain=domain,
            posts=index.reached_by(domain).posts,
            accounts=index.reached_by(domain).accounts,
        )
        for domain in row.domains
        if domain not in index.withheld and index.reached_by(domain).accounts >= _MIN_ACCOUNTS
    )


def _domain_lookalike(row: PostDomains, index: LinkIndex) -> tuple[Evidence, ...]:
    return tuple(
        ConfusablePair(domain=domain, other=other)
        for domain in row.domains
        for other in index.confusable.get(domain, ())
    )


def _content(item: CorpusItem, signal: SignalName) -> tuple[Evidence, ...]:
    """What one Content Signal finds in a post's own text, one row per sentence.

    A sentence rather than a phrase, so the evidence a reviewer reads is the thing they
    would read, and so the whole of the negation guard is in front of them: the printed
    sentence either names the claim or denies it, and either way they can see which. The
    title is read before the body because it is what a reviewer sees first, and a claim
    made there is not the same claim as one made in the fourth paragraph.
    """
    found: list[PhraseMatch] = []
    for field, text in (("title", item.title), ("body", item.body)):
        for sentence in sentences(text):
            fired = tuple(
                sorted(
                    phrase
                    for phrase in _PHRASES[signal]
                    if any(not _negated(sentence, at) for at in spans(sentence, phrase))
                )
            )
            if fired:
                found.append(
                    PhraseMatch(
                        signal=signal,
                        field=field,
                        phrases=fired,
                        sentence=sentence,
                    )
                )
    return tuple(found)


def _negated(sentence: str, start: int) -> bool:
    """Whether a negator stands between this match and the start of its sentence.

    The sentence and not the phrase, because a negation is a claim about a whole
    statement: "we do not ask for a deposit, a kit fee, or any money up front" is one
    request denied three times, and stopping at the comma would read the last two as
    requests.

    A bare `no` is the exception and counts only when it stands directly before the
    phrase. `no experience needed` is in every job post ever written, so a `no` that
    counted anywhere in the sentence would cancel a deposit named in the same breath as
    it and leave the Signal silent on exactly the posts it exists for. Everywhere else it
    means nothing at all, which is why it is not in the list above.

    The cost is a post that asks for money in one sentence and denies asking in the next
    one. It is lost, and it is lost in the direction of not scoring a post, which is the
    cheaper of the two mistakes to make with a severity figure a reviewer will act on.
    """
    words: list[str] = _WORD.findall(sentence[:start].lower())
    if not words:
        return False
    if any(word in _NEGATORS for word in words):
        return True
    return words[-1] == "no"


def _normalise(points: int, published: int) -> int:
    """The Policy Score: earned points as a share of the published ones, out of 100.

    Rounded half up by integer arithmetic rather than by `round`, so that the rule a
    reader follows is stated in one sentence and cannot depend on the interpreter's
    tie-breaking. Dividing by the published total rather than by 100 means a Signal
    added later moves every score rather than pushing the existing ones over the top,
    which is the documented change the weight file exists to make.
    """
    return (200 * points + published) // (2 * published)


def _facts(
    corpus_path: Path,
    list_path: Path,
    rows: Sequence[PostDomains],
    index: LinkIndex,
    scores: Sequence[PostScore],
    suffixes: PublicSuffixes,
) -> ScoreFacts:
    """Every figure about the Corpus the table prints, from the rows, the index, and
    the bytes behind them."""
    return ScoreFacts(
        accounts=len({row.account for row in rows}),
        corpus_path=corpus_path.as_posix(),
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        posts=len(scores),
        posts_with_signals=sum(1 for score in scores if score.signals),
        registrations=len(index.reach),
        registrations_withheld=sum(1 for domain in index.reach if domain in index.withheld),
        rules_sha256=suffixes.rules_digest(),
        suffix_list_path=list_path.as_posix(),
        suffix_list_sha256=hashlib.sha256(list_path.read_bytes()).hexdigest(),
    )


def write_policy_scores(path: Path, scores: Sequence[PostScore]) -> None:
    """Every post's score, with the arithmetic beside it.

    The file holds the breakdown rather than the score alone, so a reader who wants
    the numbers rather than the argument reads the same claims in a different shape.
    A Signal appears once per post with every registration that fired it listed under
    it, rather than once per firing registration: one row per Signal carrying the
    weight twice would add up to twice the score, and a reader summing the column
    would be right about the file and wrong about the arithmetic.
    """

    def objects() -> Iterator[JsonObject]:
        for score in scores:
            yield {
                "post_id": score.post_id,
                "account": score.account,
                "score": score.score,
                "points": score.points,
                "published_points": score.published_points,
                "published_signals": score.published_signals,
                "signals": [
                    {
                        "signal": hit.signal.value,
                        "weight": hit.weight,
                        "evidence": [asdict(item) for item in hit.evidence],
                    }
                    for hit in score.signals
                ],
            }

    write_lines(path, objects())


def read_policy_scores(path: Path) -> tuple[PostScore, ...]:
    """Every post's score, read back out of the file the scoring command wrote.

    The reader lives beside the writer because they are one vocabulary: a field cannot be
    added to `PostScore` and left out of the file, and a field cannot be in the file that
    this does not check. An unknown field is refused rather than ignored, for the reason
    `read_corpus` gives — the one field that must never appear here is a Planted Campaign's
    identifier, and a reader that skipped what it did not recognise would skip that
    without saying so (ADR-0008).

    Three claims in a row are checked against the arithmetic before the row is returned,
    because the Review Queue orders posts by these numbers and prints them beside a
    breakdown. The points have to be what the Signals in the row are priced at, the score
    has to be those points as a share of the published total, and the published total has
    to be the same on every row. A row failing any of those describes a figure this
    project could not have published, and a queue built on it would rank posts by numbers
    that disagree with the arithmetic printed beside them.
    """
    scores = tuple(_score(path, number, text) for number, text in read_rows(path))
    refuse_repeated(
        path.as_posix(),
        (score.post_id for score in scores),
        "a queue would read one post's Signals into two entries and rank both of them",
    )
    published = {score.published_points for score in scores}
    if len(published) > 1:
        raise ValueError(
            f"{path.as_posix()} publishes {sorted(published)} points of weight across its "
            "rows, and every post has to be scored against the same published set for "
            "these scores to be ranked against each other at all"
        )
    return scores


def _score(path: Path, number: int, text: str) -> PostScore:
    """One row, checked against the arithmetic the writer wrote it from.

    Checked here rather than trusted because the numbers in this file are what a reader
    sorts a queue by, and every one of the four claims below is one a row could fail
    without any part of it looking wrong: a weight the weight set does not publish, a
    Signal priced twice, points that do not add up, and a score that is not the share of
    them the file publishes beside.
    """
    where = f"{path.as_posix()}:{number}"
    record = read_object(where, text)
    vocabulary = tuple(field.name for field in fields(PostScore))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds {sorted(record)}, which is not the score vocabulary "
            f"{sorted(vocabulary)}"
        )

    points = _figure(where, record, "points")
    published_points = _figure(where, record, "published_points")
    published_signals = _figure(where, record, "published_signals")
    if published_points == 0:
        raise ValueError(
            f"{where} publishes no points, and a share of nothing is not a Policy Score"
        )

    carried = record["signals"]
    if not isinstance(carried, list):
        raise ValueError(f"{where} has signals={carried!r}, which is not a list")
    hits = tuple(_published(where, carried[at], at) for at in range(len(carried)))
    refuse_repeated(
        f"{where} signals",
        (hit.signal.value for hit in hits),
        "a Signal is present or absent, so one of them is not in this row (ADR-0014)",
    )
    priced = sum(hit.weight for hit in hits)
    if priced != points:
        raise ValueError(
            f"{where} has points={points}, and its Signals are priced at {priced}"
        )
    if len(hits) > published_signals:
        raise ValueError(
            f"{where} carries {len(hits)} Signals, and the published set holds "
            f"{published_signals}"
        )

    score = _figure(where, record, "score")
    earned = _normalise(points, published_points)
    if score != earned:
        raise ValueError(
            f"{where} has score={score}, and {points} of {published_points} published "
            f"points is {earned}"
        )
    return PostScore(
        post_id=read_text(where, record, "post_id"),
        account=read_text(where, record, "account"),
        score=score,
        points=points,
        published_points=published_points,
        published_signals=published_signals,
        signals=hits,
    )


def _published(where: str, record: object, at: int) -> SignalHit:
    """One Signal a post is recorded as carrying, with the weight and evidence it was.

    `SignalHit` checks the rest from here: a Signal present with no evidence for it, or
    resting on evidence of another Signal's kind, is refused in its own constructor. What
    is left here is the weight, which the weight file has already refused to publish as
    zero and this file has no such guarantee about.
    """
    row = f"{where} signals[{at}]"
    if not isinstance(record, dict):
        raise ValueError(f"{row} is not a row: {record!r}")
    vocabulary = tuple(field.name for field in fields(SignalHit))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{row} holds {sorted(record)}, which is not the Signal vocabulary "
            f"{sorted(vocabulary)}"
        )
    weight = _figure(row, record, "weight")
    if weight == 0:
        raise ValueError(
            f"{row} prices {read_vocabulary(row, 'signal', record['signal'], SignalName)} at "
            "nothing, and a Signal that adds nothing is a Signal a reader cannot weigh"
        )
    evidence = record["evidence"]
    if not isinstance(evidence, list):
        raise ValueError(f"{row} has evidence={evidence!r}, which is not a list")
    return SignalHit(
        signal=read_vocabulary(row, "signal", record["signal"], SignalName),
        weight=weight,
        evidence=tuple(
            _evidence(f"{row} evidence[{position}]", entry)
            for position, entry in enumerate(evidence)
        ),
    )


def _evidence(where: str, record: object) -> Evidence:
    """One piece of evidence, told apart by the fields it holds rather than by a name.

    None of the three kinds says which it is: two of them are keyed on `domain`, and only
    the sentence names a Signal. The field set is therefore what identifies it, and it is
    the whole of what the writer wrote — a row holding anything else would have to be
    either dropped or guessed at, and evidence that is dropped is evidence a reviewer
    cannot check the score against.
    """
    if not isinstance(record, dict):
        raise ValueError(f"{where} is not a row: {record!r}")
    for kind, read in _EVIDENCE.items():
        if set(record) == {field.name for field in fields(kind)}:
            return read(where, record)
    shapes = ", ".join(
        f"{{{', '.join(sorted(field.name for field in fields(kind)))}}}" for kind in _EVIDENCE
    )
    raise ValueError(f"{where} holds evidence {sorted(record)}, which is none of {shapes}")


def _frequency(where: str, record: JsonObject) -> DomainFrequency:
    return DomainFrequency(
        domain=read_text(where, record, "domain"),
        posts=_figure(where, record, "posts"),
        accounts=_figure(where, record, "accounts"),
    )


def _pair(where: str, record: JsonObject) -> ConfusablePair:
    return ConfusablePair(
        domain=read_text(where, record, "domain"),
        other=read_text(where, record, "other"),
    )


def _sentence(where: str, record: JsonObject) -> PhraseMatch:
    phrases = read_names(where, record, "phrases")
    if not phrases:
        raise ValueError(f"{where} matches no phrase, and a match on nothing is not a match")
    return PhraseMatch(
        signal=read_vocabulary(where, "signal", record["signal"], SignalName),
        field=read_text(where, record, "field"),
        phrases=phrases,
        sentence=read_text(where, record, "sentence"),
    )


_EVIDENCE: Mapping[type[Evidence], Callable[[str, JsonObject], Evidence]] = {
    DomainFrequency: _frequency,
    ConfusablePair: _pair,
    PhraseMatch: _sentence,
}


def _figure(where: str, record: JsonObject, field: str) -> int:
    """One whole number of points, refused when it is anything else.

    A boolean passes for an integer in Python and would make a weight of `true` read as
    one, and a negative figure would have to be explained by something this file does not
    print. Both are named rather than left to a reader of the file.
    """
    value = record[field]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"{where} has {field}={value!r}, and a figure is a whole number of points"
        )
    return value


def render_table(scored: Scored) -> str:
    """The console output: what was read, then the arithmetic, then what drove it.

    ASCII only, so it prints the same way on a console that cannot encode anything
    else and the same way when it is redirected, which is what lets it be pasted into
    an issue or diffed between runs.

    The index and the blocks are in the same order on purpose, both highest score
    first: a reader who has just read a line of the index scrolls down looking for
    that post's block. Ties fall through to the post id here rather than to the points
    earned, so this is a stable table and not the Review Queue: `rfi review-queue` is
    where the whole Corpus is ordered and where a tie is broken.
    """
    scored_posts = _by_score([score for score in scored.scores if score.signals])
    sections = (
        f"{_HEADING}\n\n{_SUBHEADING}",
        _figures(scored),
        _index(scored_posts),
        "\n\n".join(_block(score) for score in scored_posts),
        _registrations(scored),
        _confusable_table(scored),
        _weights_table(scored),
        _phrases_table(),
        _footer(scored),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _figures(scored: Scored) -> str:
    """What was read, and what came of it. One figure per line, labelled."""
    facts = scored.facts
    fields_out = (
        ("corpus", f"{facts.corpus_path} ({_count(facts.posts, 'post')} across "
                   f"{_count(facts.accounts, 'account')})"),
        ("sha256", facts.corpus_sha256),
        ("suffix list", facts.suffix_list_path),
        ("rules sha256", facts.rules_sha256),
        ("shared list", f"{scored.shared.path} ({_count(len(scored.shared.hosts), 'host')})"),
        (
            "weights",
            f"{scored.weights.path} ({_count(len(scored.weights.entries), 'Signal')}, "
            f"{scored.weights.total} points published)",
        ),
        (
            "scored",
            f"{_count(facts.posts_with_signals, 'post')} carrying a Signal, "
            f"{_count(facts.posts - facts.posts_with_signals, 'post')} carrying none",
        ),
    )
    width = max(len(name) for name, _ in fields_out)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in fields_out)


def _count(number: int, noun: str) -> str:
    """One count, agreeing with its noun. Every figure here is small, and a reader
    seeing "1 accounts" stops to wonder whether the figure is right."""
    return f"{number} {noun if number == 1 else f"{noun}s"}"


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """One line of a table. A cell of nothing but digits is a count, and counts are
    right-aligned so the digits line up down the column; the rest is text of no fixed
    width and reads better against the left edge."""
    return "  ".join(
        cell.rjust(width) if cell.isdigit() else cell.ljust(width)
        for cell, width in zip(cells, widths, strict=True)
    )


def _index(scores: Sequence[PostScore]) -> str:
    """One line per post carrying a Signal, in the order the blocks below are printed.

    A post carrying no Signal is not a line here: its score is zero and there is no
    arithmetic to show for it, so all of them would be. The count is in the figures
    above and every one of them is in the file, which is where the posts of this Corpus
    that carry nothing are.
    """
    if not scores:
        return "No post in this Corpus carries a Signal."

    headings = ("post", "account", "score", "signals")
    rows = [
        (
            score.post_id,
            score.account,
            str(score.score),
            ", ".join(hit.signal.value for hit in score.signals),
        )
        for score in scores
    ]
    widths = [max(len(cell) for cell in column) for column in zip(headings, *rows, strict=True)]
    return "\n".join([_row(headings, widths), *(_row(row, widths) for row in rows)])


def _by_score(scores: Sequence[PostScore]) -> list[PostScore]:
    """Highest score first, then by post id, so a fixed Corpus prints a fixed table."""
    return sorted(scores, key=lambda score: (-score.score, score.post_id))


def _block(score: PostScore) -> str:
    """One post's breakdown: every Signal, its published weight, and what fired it.

    The last line is the total, against the published total, so the division that
    produced the displayed score is written out rather than left to the reader.
    """
    name = max(len(hit.signal.value) for hit in score.signals)
    figures = max(len(str(hit.weight)) for hit in score.signals)
    lines = [f"{score.post_id}  {score.account}  {score.score}/100"]
    for hit in score.signals:
        lines.append(
            f"  {hit.signal.value.ljust(name)}  {str(hit.weight).rjust(figures)}  "
            f"{hit.line()}"
        )
    lines.append(
        f"  {'total'.ljust(name)}  {str(score.points).rjust(figures)}  "
        f"of {score.published_points} published points, "
        f"{len(score.signals)} of {score.published_signals} Signals"
    )
    return "\n".join(lines)


def _registrations(scored: Scored) -> str:
    """Every registration, what reaches it, and which Signals that decides.

    The table the per-Signal breakdown is checked against: a reader who thinks a
    Signal fired wrongly can see here the posts and accounts it was fired from, and a
    registration with one account — the floor — is printed with its Signals column
    empty rather than left out, so the reason it does not fire is visible too.
    """
    headings = ("domain", "posts", "accounts", "withheld", "signals")
    rows = []
    for domain in sorted(scored.index.reach):
        reach = scored.index.reach[domain]
        carried = sorted(
            {
                hit.signal.value
                for score in scored.scores
                for hit in score.signals
                for item in hit.evidence
                if domain in item.registrations
            }
        )
        rows.append(
            (
                domain,
                str(reach.posts),
                str(reach.accounts),
                "yes" if domain in scored.index.withheld else "-",
                ", ".join(carried),
            )
        )
    widths = [max(len(cell) for cell in column) for column in zip(headings, *rows, strict=True)]
    table = "\n".join([_row(headings, widths), *(_row(row, widths) for row in rows)])
    return f"registrations  {len(rows)} of which {scored.facts.registrations_withheld} withheld\n{table}"


def _confusable_table(scored: Scored) -> str:
    """Every confusable pair the Corpus's links resolve to, once per pair.

    Printed after the scores rather than inside them because it is a fact about the
    Corpus and not about any one post: the same pair is named by both members, and a
    reader looking at the score of one post wants to know the other name, not to be
    shown the pair twice from each end.
    """
    pairs = scored.index.pairs()
    if not pairs:
        return ""

    width = max(len(pair.domain) for pair in pairs)
    heading = (
        f"confusable  {_count(len(pairs), 'pair')} one edit apart under the same Public "
        "Suffix; no pair is named as the copy of the other"
    )
    return "\n".join(
        [
            heading,
            *(f"  {pair.domain.ljust(width)}  {pair.other}" for pair in pairs),
        ]
    )


def _weights_table(scored: Scored) -> str:
    """The published weights and their reasons, read out of the file rather than
    restated, so the table and the data cannot drift apart.

    The columns are as wide as the data rather than as wide as this weight set: a
    Signal added in ticket #10 with a longer name or a weight over three digits would
    otherwise run into its own reason.
    """
    entries = scored.weights.entries
    name = max(len(entry.signal.value) for entry in entries)
    figure = max(len(str(entry.weight)) for entry in entries)
    return "\n".join(
        [
            f"weights  {scored.weights.path}",
            *(
                f"  {entry.signal.value.ljust(name)}  {str(entry.weight).rjust(figure)}  "
                f"{entry.rationale}"
                for entry in entries
            ),
        ]
    )


def _phrases_table() -> str:
    """Every phrase a Content Signal matches, printed in full so the rule can be applied
    to the post text by hand.

    The weights above are data and can be argued with in a file; a phrase list is a rule
    and lives in the code, so publishing it is the only way a reader gets to see what the
    Signal was actually looking for. Without this a breakdown says a sentence fired
    `payment_request` and leaves the strings that decide it unsaid, which is half the
    auditability claim missing.
    """
    lines = [
        "phrases  the whole of each Content Signal's list, and nothing outside it fires "
        'one. A match is dropped where one of the negators below stands earlier in the '
        'same sentence, or where a bare "no" stands directly before it'
    ]
    lines.extend(
        wrap(f"negators  {', '.join(f'\"{word}\"' for word in sorted(_NEGATORS))}", indent=2)
    )
    for signal in _CONTENT_SIGNALS:
        phrases = _PHRASES[signal]
        lines.append(f"  {signal.value}  {_count(len(phrases), 'phrase')}")
        lines.extend(wrap(", ".join(f'"{phrase}"' for phrase in phrases)))
    return "\n".join(lines)


def _footer(scored: Scored) -> str:
    """What the score can and cannot claim, and what it is worth on its own.

    The limits are the ones a reader would otherwise have to guess at: the arithmetic is
    a subset-sum and no more, the frequency figures count other posts and not the
    reader's judgement of them, the phrase rules lose the posts that deny a request in
    the same sentence they make it, and the withheld registrations are out of the scoring
    path entirely rather than scored at zero.
    """
    return f"""\
A Signal is present or absent, so a post that links three shared registrations, or that
uses four phrases from one Content Signal's list, carries that Signal once and not four
times: the score is a subset-sum of the weights above and nothing else. It is a severity
judgement by this project, published so it can be argued with, and it is not a
probability that anything is what it appears to be.

The score is the points earned as a share of the published total, to the nearest whole
number out of 100 with halves going up, so it can differ from the share by half a point.
Dividing by the published total rather than by 100 is what lets a Signal be added later
without any existing score exceeding 100.

No account's age, karma, posting rate, or activity change is read on this path, and the
Corpus file holds no such field to read (ADR-0008). What the frequency Signal reads is
which registrations other posts' links resolve to, which is structure the reader can go
and look at; the figures are printed with the Signal for exactly that reason.

The Content Signals read the post and nothing else, and they read it literally: a Signal
fires where a published phrase appears in the post's own title or body and no negator
stands between it and the start of that sentence. The sentence is printed with the match
because that is the judgement a reader may want to overturn, and the cost of the guard
is stated rather than hidden: a post that asks for money in one sentence and denies
asking in the next carries no `payment_request`, and a bare "no" cancels only the phrase
directly after it, because "no experience needed" is in every job post ever written.

The registrations on the known-shared list are out of the scoring path rather than
scored at zero, and the list is named above so a reader who wants a different one can
change the file and watch every score move:

    withheld  {scored.facts.registrations_withheld} of {scored.facts.registrations} registrations

A Planted Campaign that leans on a shared host is lost with the host, so a score built
this way is a lower bound for the same reason a recovery figure is."""


def _weight(path: Path, number: int, line: str) -> SignalWeight:
    """One row, checked field by field.

    Checked here rather than trusted because the file is the thing this project
    publishes its weights in: a weight of zero would put a Signal in the breakdown
    that no arithmetic adds up to, and a blank rationale would publish a number
    nothing argues for. Both are invisible in the output, which is why they stop the
    run here instead.
    """
    where = f"{path.as_posix()}:{number}"
    try:
        record = json.loads(line)
    except json.JSONDecodeError as refusal:
        raise ValueError(f"{where} is not JSON: {line!r}") from refusal
    if not isinstance(record, dict):
        raise ValueError(f"{where} is not a row: {line!r}")

    vocabulary = tuple(field.name for field in fields(SignalWeight))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds {sorted(record)}, which is not the weight vocabulary "
            f"{sorted(vocabulary)}"
        )
    try:
        signal = SignalName(record["signal"])
    except ValueError as unknown:
        names = ", ".join(name.value for name in SignalName)
        raise ValueError(f"{where} names {record['signal']!r}, and the Signals are {names}") from unknown
    weight = record["weight"]
    if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
        raise ValueError(f"{where} has weight={weight!r}, and a weight is a whole number above zero")
    rationale = record["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{where} has no rationale, and a weight with no reason is an assertion")
    return SignalWeight(signal=signal, weight=weight, rationale=rationale)
