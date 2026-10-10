"""Campaign Candidates: accounts joined by shared infrastructure, either kind of it.

The first command that cuts a complete path from the Corpus to an output, and the whole
design rests on it. ADR-0005 puts two accounts in the same Campaign Candidate only if
they share a Registrable Domain or a Contact Point, and both edges are read here. Two
accounts with the same words and nothing shared are not grouped, because a converged
scam template must not merge two unrelated operators.

The rule is a union-find over accounts: every account is on its own, every shared thing
touches every account that reaches it, and two accounts end up together exactly when a
chain of shared registrations or shared Contact Points connects them. A component of
one is not a candidate, so an account that links only its own site, or publishes only
its own reach, is never reported.

The second edge is what makes a desk that pays for a domain per post visible. Its
registrations move every time the advert is posted and nothing joins the accounts that
share them; one Telegram handle or one intake address published throughout is the same
infrastructure by any other name. Which is also why every candidate is labelled with the
evidence that justified it. The two edges are not equally good: a registration is
somebody's property and this project can check it against a published Public Suffix
List, while a Contact Point is read out of a post by a rule with a measured recall
against the Labelled Set, and the run says which kind of claim it is making rather than
letting a handle count the same as a domain. The reading is the one `contacts.py` owns,
and it is called rather than re-read: the published file is not opened, so a grouping
that needed somebody to have run `rfi contact-points` first would not be a function of
the Corpus.

Known-shared infrastructure never enters that graph, on either edge. A link shortener, a
paste site, or a link-in-bio page is a registration like any other until it is withheld,
and it is withheld *before* the grouping rather than after it, so it cannot act as a
bridge between two accounts that share nothing else. An address at one of those
registrations is withheld with the registration it sits under: everybody who complains
into a paste site publishes that address, and joining two of them on it would rebuild
the component the filter just removed. The registrations to withhold come from
`data/infrastructure/shared-hosts.jsonl` — published data, not a list in this query, so
a domain-reputation feed can replace it without a line here changing (ADR-0009). The run
also groups the Corpus with the filter switched off, because a filter whose effect
cannot be measured is an assertion: the output says how many components withholding
these registrations removed.

What a candidate is shown with is the point of the whole module. The accounts, the
posts, and the shared registrations and Contact Points that justify the grouping are
printed together, because a grouping a reader cannot check is a claim rather than a
result: the things named under a candidate are the only reason its accounts are in it,
so a candidate that rests on a long chain of unrelated sites is visible the moment they
are printed beside the accounts. A registration or a Contact Point only one account in
the component reaches is left out of that list — it is context, not a reason, and the
resolved links and the published Contact Points already hold it. The withheld
registrations are printed on the same terms, since a registration in the resolved links
that no candidate names would otherwise be unexplained.

The clock is read here too, and it is read as corroboration and nothing else
(ADR-0025). Every candidate carries the temporal proximity of the things it rests on —
the span from the earliest post by any account reaching one to the latest, and the gap
between the nearest two of them — and a piece of evidence whose accounts were all active
inside a stated window is corroborated. That verdict does not join two accounts at any
window, because ADR-0005 says a candidate rests on shared infrastructure and on nothing
else, and a rule that could be argued about from a threshold would be a grouping by
timing wearing a grouping by registration's clothes. What it does is order the output and
say so: a candidate the window corroborates something under is printed ahead of every
candidate it corroborates nothing under, and one it corroborates nothing under is named
below the table with the gaps beside its evidence, because a miss nobody can see is a miss
nobody can diagnose. The window is a figure in the output and a parameter on the command,
not a rule buried here, and nothing is removed for want of it: a registration two
accounts reached months apart may be a registration that changed hands, and a campaign may
simply be a patient one.

The stored vectors are read here too, and they are read as corroboration and nothing else
(ADR-0026). Every candidate carries the content similarity of the posts inside it — for each
post, the cosine distance to its nearest post by a *different* account in the same candidate,
over the vectors `embeddings.py` stored — and a candidate is corroborated when every one of
its posts has a near-twin by another account inside a stated threshold. That verdict joins
nothing at any threshold, and the reason is worth more than the rule: a scam template
converges across unrelated operators, so near-identical text is what a converged template
looks like and says nothing about who runs it. What the threshold does is order the output as
a second key under the clock's, and say so: a candidate both corroborate is printed ahead of
every candidate the clock corroborates and the vectors do not, and one neither corroborates is
named below the table with the nearest distance beside its posts, because a miss nobody can
see is a miss nobody can diagnose. Nothing is removed for want of it either: a genuine
campaign that paraphrases its own advert reads as different words to a lexical model, which
is a limit to state rather than a campaign to discard. This is the one thing read from
outside the Corpus's own files, and it is read from the table `rfi content-embeddings`
wrote, so a Corpus a reader cannot embed cannot be grouped either — said here rather than
as a surprise at the first refusal.

What this cannot reach is stated here rather than left in the tickets. A campaign that
rotates both its registrations and its Contact Points shares nothing with itself, so no
grouping resting on either edge can propose it; the count of campaigns in that position
is the method's recall bound, and the evaluator reports it beside the recovery figure
(ADR-0023).

The three corroborations above are then read together, and their result is the cohesion
system ADR-0009 asks for (ADR-0028). A candidate is retained when two of the three hold
and removed when they do not, and the removal is the first one this module makes for any
reason but known-shared infrastructure: the clock and the vectors may order and
deprioritise, and this filters. It is also the reason the two tiers are published as a
pair rather than one of them replacing the other. The baseline — the union-find, with the
known-shared registrations withheld and no corroboration applied — stays in the file and
in the figures exactly as it was, because it is the number a reader has to see beside the
lower one to know that the lower one was a decision and not a bug. Every candidate names
which corroborations held and which went against it, and every candidate the system
removed is named below the table with the evidence that removed it, so a miss is
diagnosable rather than silent. The order of the two figures never changes: neither tier
is published as *the* result.

Nothing here reads the truth file. The Corpus, the published Public Suffix List, the
shared-infrastructure list, and the stored vectors are the whole input, and the truth file
is joined by the evaluator after this has finished (ADR-0008).
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from reddit_fraud_intelligence.contacts import (
    ContactKind,
    SharedContact,
    extract as read_contact_points,
    reach as shared_contact_points,
)
from reddit_fraud_intelligence.embeddings import (
    DEFAULT_TABLE,
    connection_string,
    open_store,
    stored_vectors,
)
from reddit_fraud_intelligence.categories import OTHER, EVERY_CATEGORY
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.domains import PostDomains, post_domains
from reddit_fraud_intelligence.infrastructure import SharedHost, SharedInfrastructure
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
from reddit_fraud_intelligence.placement import (
    MIN_PLACED,
    agreed_category,
    check_placements,
    in_print_order,
    place,
)
from reddit_fraud_intelligence.suffixes import PublicSuffixes

_HEADING = "Campaign Candidates"
_SUBHEADING = """\
Accounts joined by a shared registrable domain or a shared Contact Point.
Every entry is a proposal, not a finding about who is behind them."""

# The window temporal proximity is measured against, in hours. A parameter rather than a
# constant because what counts as "at the same time" is a judgment a reader may want to
# make differently, and because a figure whose rule is buried in the code cannot be
# disagreed with: the output names this number on every run and `--window-hours` sets it.
DEFAULT_WINDOW_HOURS = 24

# The cosine distance two posts may be apart and still count as the same text, stated
# rather than buried. A distance at most this wide reads as near-identical; the model's
# own near-twins on this Corpus sit at about 0.15-0.21 and its retellings of one offer at
# 0.53-0.74, so 0.5 separates "the same advert pasted" from "the same offer retold".
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# What justified a candidate, in the run's own words. Three and no fourth, because
# ADR-0005 permits two edges and a candidate of two or more accounts cannot have used
# neither. The names are the ones the index and the blocks print, so the table cannot
# describe a candidate one way and label it another.
#
# A Contact Point alone is called out rather than folded in with a registration,
# because the two are not the same claim. This project measures the reading behind the
# Contact Point against a Labelled Set and publishes the shortfall row by row, so a
# candidate resting on one handle rests on the weaker of the two readings and a reader
# is entitled to know that before deciding whether to act on it.
_WEAKER = "the weaker of the two readings; `rfi contact-points` has the measured recall"

# The three corroborations ADR-0009 names, in the order they are declared and in the
# order every verdict this run publishes counts them. One list rather than three
# vocabularies written out where each verdict is reached: the count of them that hold
# and the names of the ones that did not are the whole of the cohesion figure, and a
# reader who has to learn a fourth name from the code to check it cannot check it.
# Public rather than private because `rfi campaign-recovery` states the same rule in its
# own output and its own page, and a rule a figure rests on may not be written twice.
CORROBORATIONS = ("temporal proximity", "content similarity", "category agreement")

# How many of them a candidate has to hold for the cohesion system to keep it. A bare
# majority of the declared list rather than a literal, so a fourth corroboration would
# move the bar with the list instead of being counted and then outvoted by three. Two of
# three is the weakest bar that is not one: a candidate on the clock alone, or on the
# categories alone, is one reading agreeing with itself, and the third is there to catch
# the case where the other two are the same claim read twice.
QUORUM = len(CORROBORATIONS) // 2 + 1

# The fields one candidate's `cohesion` row carries, in the file's own order. A constant
# rather than two literals, because the writer and the reader are the same vocabulary and
# the reader's whole job is to refuse a row holding a field it does not know: a set of
# names written out at both ends is a set that will one day be written out at only one.
COHESION_FIELDS = ("against", "category", "corroborating", "retained")


class EvidenceKind(StrEnum):
    """Which of ADR-0005's two edges one piece of evidence belongs to.

    The singular of the two labels above, kept as its own vocabulary because a piece of
    evidence names exactly one edge and a candidate's label may name both or neither. The
    two are not written out twice anywhere: `Evidence.kind` reads this one, and the reader
    takes the kind off that rather than off a literal of its own.

    A closed vocabulary rather than the name of a thing, because the published file pairs
    each timing row with the evidence it describes and a row naming something this build
    does not publish is a row no figure beside it can be counted in.
    """

    REGISTRATION = "registration"
    CONTACT_POINT = "Contact Point"


class Evidence(StrEnum):
    """Which of ADR-0005's two edges put a candidate's accounts together.

    Derived from the candidate's two evidence lists rather than stored beside them, so
    the file and the table cannot hold different opinions about which edge a candidate
    rests on, and so a file can never carry a label nothing checks.

    `NONE` is the fourth value and it exists because the reader allows an empty list for
    either edge: a row naming neither is a candidate nobody justified, and saying
    `registrations` about it would be a guess. The grouping never writes one.
    """

    NONE = "nothing this file names"
    DOMAIN = "registrations"
    CONTACT = "Contact Points"
    BOTH = "registrations and Contact Points"

    @property
    def kind(self) -> EvidenceKind:
        """The one edge this label names.

        `NONE` and `BOTH` name no single edge and are refused rather than answered with
        one: a piece of evidence is a registration or a Contact Point and never both and
        never neither, so a label naming two or none has no answer to give. Answering with
        a guess is how two vocabularies for the same thing drift apart quietly, and this is
        the only place the reader learns which kind a piece of evidence is.
        """
        if self is Evidence.DOMAIN:
            return EvidenceKind.REGISTRATION
        if self is Evidence.CONTACT:
            return EvidenceKind.CONTACT_POINT
        raise ValueError(
            f"{self.value} names no single edge, so it has no kind: a piece of evidence is "
            "a registration or a Contact Point, and never both and never neither"
        )

    @classmethod
    def of(cls, shared_domains: Sequence[SharedDomain], points: Sequence[SharedContact]) -> Evidence:
        """The one label for whichever pair of lists is non-empty."""
        on_domains = bool(shared_domains)
        on_points = bool(points)
        if not on_domains and not on_points:
            return cls.NONE
        if on_domains and on_points:
            return cls.BOTH
        return cls.CONTACT if on_points else cls.DOMAIN

    def joined_on(self, names: Sequence[str]) -> str:
        """Every shared thing, named under the edge it rests on.

        The one place the two kinds are rendered as one list, because three commands print
        a candidate's evidence beside an identifier and a list of bare strings would not
        say which of ADR-0005's two edges produced it — a Contact Point and a
        Registrable Domain are the same shape and a different claim.
        """
        named = ", ".join(names)
        return f"{self.value}: {named}" if named else self.value


@dataclass(frozen=True, slots=True)
class SharedDomain:
    """One registration, and the accounts of a candidate that reach it.

    A registration is only named under a candidate when two or more of that candidate's
    accounts link it, so this is always the justification rather than the context.
    """

    domain: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WithheldContactPoint:
    """One Contact Point at a withheld registration, and what published it.

    Printed for the same reason a withheld registration is: an address nobody can act
    on is a gap in the output rather than a decision in it, and a reader who finds it in
    the published Contact Points needs to be able to see that it was filtered rather than
    missed.
    """

    kind: ContactKind
    value: str
    registration: str
    accounts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WithheldRegistration:
    """One registration the filter kept out of the graph, and what it reached.

    Printed for the same reason a shared registration is: a registration the resolved
    links hold and no candidate names is otherwise a gap in the output rather than a
    decision in it.
    """

    registration: str
    hosts: tuple[SharedHost, ...]
    accounts: int


@dataclass(frozen=True, slots=True)
class _Joined:
    """What one component rests on, and when the accounts behind each of it posted.

    The evidence and its timing in one value, because they are one answer: a candidate
    whose registrations and Contact Points were worked out in one pass and whose gaps were
    worked out in another could print a gap beside a piece of evidence it does not name, or
    leave one named piece of evidence with no gap beside it at all. Nothing here reaches
    outside the component, so every list is the justification rather than the context.
    """

    domains: tuple[SharedDomain, ...]
    points: tuple[SharedContact, ...]
    timing: tuple[EvidenceTiming, ...]
    similarity: tuple[SimilarityEvidence, ...]
    category: CategoryAgreement


@dataclass(frozen=True, slots=True)
class EvidenceTiming:
    """How close in time the accounts behind one piece of evidence posted.

    Two gaps, and they answer different questions. `span_seconds` runs from the earliest
    of those accounts' posts to the latest, and it is what the window is measured against:
    an operator running two accounts is active through both of them across the same days,
    and a registration two accounts reached months apart may be a domain that changed
    hands — which is the case the ticket is written for. `closest_seconds` is between the
    nearest post by one account reaching the evidence and the nearest by another. It never
    decides anything, and it is published because it is the figure a reader needs to
    disagree with the span: a candidate whose accounts came within a minute of each other
    and did nothing else together for three weeks is a different claim from one whose
    accounts posted together all morning.

    Both are published rather than summarised, and neither is allowed to group anything:
    see `Timing`.
    """

    kind: EvidenceKind
    value: str
    closest_seconds: int
    span_seconds: int

    def within(self, window_seconds: int) -> bool:
        """Whether these accounts were all active inside the stated window.

        A method rather than a stored column, for the reason ADR-0023 gives for not
        storing the evidence label: a verdict nothing recomputes is a second account of
        the same fact, and the file and the table beside it could then disagree about
        whether a candidate was corroborated.

        The span rather than the closest pair, which is the stronger of the two readings
        and the one the ticket asks for. Deciding on the closest pair would let a single
        coincidence corroborate a whole component: two of five accounts posting in the
        same minute by accident is enough to call the other three corroborated, and a
        group claim is not settled by the luckiest pair in it.
        """
        return self.span_seconds <= window_seconds


@dataclass(frozen=True, slots=True)
class Timing:
    """Temporal proximity across a whole candidate, in the run's own words.

    `pieces` is how many registrations and Contact Points the candidate rests on and
    `corroborated` is how many of them hold their accounts inside the window, so a
    candidate resting on one registration reads `1 of 1` or `0 of 1` and a candidate
    resting on both edges reads out of four rather than collapsing to a yes.
    `closest_seconds` is the nearest pair of accounts anywhere in the component, and it
    is published whatever the window says: it is the gap a reader needs to disagree with
    the window rather than the number the window produced.
    """

    window_seconds: int
    pieces: int
    corroborated: int
    closest_seconds: int

    @property
    def corroborates(self) -> bool:
        """Whether any piece of evidence under this candidate holds two accounts together.

        Anything at all, rather than all of it. A candidate whose registration puts two
        accounts in the same hour and whose Contact Point joins a third a fortnight later
        is a grouping timing agrees with and disagrees with in two places, and the second
        place is printed beside the third account rather than used to throw the candidate
        away.
        """
        return self.corroborated > 0


@dataclass(frozen=True, slots=True)
class SimilarityEvidence:
    """One post of a candidate, and its nearest post by another account.

    The same shape of figure `EvidenceTiming` is for timing: a per-member measurement
    that the candidate-level summary counts over rather than absorbs. The nearest post
    is measured across accounts rather than inside one, because two posts from one
    account are that account's own habits and not evidence about a group. It is
    published whole rather than summarised, and it is never allowed to group anything:
    see `ContentSimilarity`.
    """

    post: str
    account: str
    nearest: str
    nearest_account: str
    distance: float


@dataclass(frozen=True, slots=True)
class ContentSimilarity:
    """Within-component content similarity across a whole candidate.

    `pieces` is how many posts the candidate holds and `corroborated` is how many of
    them have a near-twin by a different account within the threshold, so a candidate
    reads `0 of 5` through `5 of 5` and never collapses to a yes. `closest_distance`
    is the smallest such gap in the candidate, published whatever the threshold says:
    it is the figure a reader needs to disagree with the threshold rather than the
    number the threshold produced.
    """

    threshold: float
    pieces: int
    corroborated: int
    closest_distance: float

    @property
    def corroborates(self) -> bool:
        """Whether every post in the candidate has a near-twin by another account.

        All of the posts rather than the luckiest pair: a candidate is a claim about a
        group, and one coincidence inside it is not the group. This is the same
        reading ADR-0025 defends for the window: the span across the whole piece,
        which decides on all of it, rather than the closest pair, which decides on the
        luckiest two.
        """
        return self.pieces > 0 and self.corroborated == self.pieces


@dataclass(frozen=True, slots=True)
class CategoryAgreement:
    """The Scam Category the posts inside one candidate agree on, and the tally behind it.

    The same reading ADR-0021 gives a Registrable Domain, at the scope of a component
    rather than a registration, and decided by `placement.agreed_category` for that
    reason: a strict majority of the postings a list placed, out of the two at least that
    have to be placed before any of them counts. What it says is that the accounts inside
    the candidate are making the same pitch, which is a claim about the group the
    component is a claim about — the third corroboration ADR-0009 names, and the one
    that decides the staggered-paraphrase case the vectors cannot corroborate (ADR-0028).

    The tally is carried whole rather than reduced to the class, because the class is
    what a rule produced and the tally is what a reader counts. Other is in the tally and
    out of the agreement, as everywhere else: a post no list matched makes no claim.
    """

    tally: tuple[tuple[str, int], ...]

    @property
    def placed(self) -> int:
        """The postings that were placed in one of the ten, which is what decides."""
        return sum(count for name, count in self.tally if name != OTHER.name)

    @property
    def scam_category(self) -> str | None:
        """The class the placed postings agree on, or none where the tally decides nothing."""
        return agreed_category(self.tally)

    @property
    def corroborates(self) -> bool:
        """Whether this candidate's posts agree on one class.

        A method rather than a stored column, for the reason `EvidenceTiming.within`
        gives: a verdict nothing recomputes is a second account of the same fact, and the
        file and the table beside it could then disagree about whether a candidate was
        corroborated.
        """
        return self.scam_category is not None

    def why_not(self) -> str:
        """Why the tally decides nothing, in the words the registrations table uses.

        Two figures rather than an adjective: the count of placements is what decides,
        and a reader who is told only that there was no majority cannot tell a component
        whose posts split evenly from one where two were placed and two matched nothing.
        """
        return "too few placed" if self.placed < MIN_PLACED else "no majority"


@dataclass(frozen=True, slots=True)
class Cohesion:
    """One candidate's three corroborations and the count they add up to.

    The count and the names are worked out rather than carried, and the names are the
    reason: a cohesion score a reader cannot take apart is a number to be trusted, and
    this project has spent three tickets refusing to publish numbers of that kind. So the
    score is how many of `CORROBORATIONS` hold, and `corroborating` and `against` say
    which — the same count the file publishes beside the tally it was counted from, and
    the reader checks one against the other rather than taking either on trust.

    `QUORUM` is the whole of the filtering rule, and it is deliberately not one: a
    candidate on the clock alone is one reading agreeing with itself, and the same is
    true of the categories alone. What the third buys is the case where the other two
    would both have said yes for the same reason — two registrations worked months apart
    by the same three people are not two corroborations, and the categories are what
    catch it.
    """

    timing: Timing
    similarity: ContentSimilarity
    category: CategoryAgreement

    @property
    def corroborating(self) -> tuple[str, ...]:
        """The corroborations that hold, in the order they are declared."""
        return tuple(
            name for name, holds in zip(CORROBORATIONS, self._holds, strict=True) if holds
        )

    @property
    def against(self) -> tuple[str, ...]:
        """The corroborations that do not, in the order they are declared."""
        return tuple(
            name for name, holds in zip(CORROBORATIONS, self._holds, strict=True) if not holds
        )

    @property
    def retained(self) -> bool:
        """Whether the cohesion system keeps this candidate.

        Two of the three, and the count is a majority of the declared list rather than a
        literal three, so a fourth corroboration added later would move the bar with the
        list instead of being counted and then outvoted.
        """
        return len(self.corroborating) >= QUORUM

    @property
    def _holds(self) -> tuple[bool, ...]:
        """The three verdicts, in the order `CORROBORATIONS` declares them.

        Asked of the three values rather than stored beside them, so the count and the
        names cannot come to disagree with each other: both are the same tuple read two
        ways, and the verdicts they come from are the ones the candidate publishes on its
        own rows.
        """
        return (
            self.timing.corroborates,
            self.similarity.corroborates,
            self.category.corroborates,
        )


@dataclass(frozen=True, slots=True)
class CampaignCandidate:
    """A proposed grouping of accounts, and everything a reader needs to check it.

    `accounts` is what the run proposes; `shared_domains` and `shared_contact_points`
    are why, and `evidence` says which of the two the candidate rests on. `first_seen`
    is the earliest post by any account in the component, which is the only date in the
    Corpus that says anything about when the group was active. `corroboration` and
    `timing` are what the clock adds, and `similarity` and `content_similarity` what
    the stored vectors add: both are corroboration and never the reason the candidate
    exists (ADR-0025, ADR-0026). `cohesion` is what those two and the categories add
    together, and it is a filter rather than a verdict on the candidate: the candidate
    is in this file and in the table whatever it says (ADR-0028).
    """

    candidate_id: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]
    shared_domains: tuple[SharedDomain, ...]
    shared_contact_points: tuple[SharedContact, ...]
    corroboration: tuple[EvidenceTiming, ...]
    timing: Timing
    similarity: tuple[SimilarityEvidence, ...]
    content_similarity: ContentSimilarity
    cohesion: Cohesion
    first_seen: str

    @property
    def evidence(self) -> Evidence:
        """Which of ADR-0005's two edges joined these accounts, worked out not stored."""
        return Evidence.of(self.shared_domains, self.shared_contact_points)

    def joined_on(self) -> str:
        """Everything this candidate is joined on, under the edge it rests on.

        The same phrasing `rfi campaign-recovery` and `rfi review-queue` print beside an
        identifier, so a candidate is not one claim in the grouping's own table and
        another in the two commands that read its output.
        """
        return self.evidence.joined_on(
            [
                *(shared.domain for shared in self.shared_domains),
                *(point.value for point in self.shared_contact_points),
            ]
        )


@dataclass(frozen=True, slots=True)
class GroupingFacts:
    """What reading the Corpus and the published Public Suffix List establishes, as
    claims about bytes.

    The four account counts are a partition of the Corpus's accounts: grouped, alone,
    silenced, and unreachable. They are printed as four lines rather than one because
    each answers a different question — how much of the Corpus grouped, how much reaches
    something nobody else reaches, how much reached nothing but known-shared
    infrastructure, and how much reached nothing the grouping could have used at all. The
    partition is over both edges rather than over registrations, because an account
    grouped by a shared Contact Point has been reached: counting it as unreachable would
    put it in two of the four lines at once and the four would stop adding up to the
    Corpus's accounts.
    """

    accounts: int
    accounts_alone: int
    accounts_in_candidates: int
    accounts_unreachable: int
    candidates: int
    corpus_path: str
    corpus_sha256: str
    posts: int
    shared_contact_points: int
    shared_domains: int
    suffix_list_path: str
    suffix_list_sha256: str
    rules_sha256: str


@dataclass(frozen=True, slots=True)
class SharedFilter:
    """What withholding the known-shared registrations did to this run.

    Both groupings come from the same rows and the same union-find, so the difference
    between them is the filter and nothing else. `components_before` is the direct
    adjacency baseline ADR-0009 asks for, and `components_removed` counts the
    components that are no longer components: one the filter splits into two is counted
    once here, because the component itself is gone, and its parts are counted as
    candidates instead.

    `withheld` is what left the graph, `withheld_points` is the Contact Points that left
    it with the registration under them, and `silenced` is the accounts that reached
    nothing else — the ones the filter made unproposable rather than the ones it
    merely touched. Keeping the last two apart is the difference between "the filter
    grouped nothing on this" and "these accounts are in no candidate at all", which are
    not the same claim: an account can reach a shortener and still share a registration
    or a handle with another account.
    """

    published: SharedInfrastructure
    withheld: tuple[WithheldRegistration, ...]
    withheld_points: tuple[WithheldContactPoint, ...]
    silenced: frozenset[str]
    accounts_reaching_withheld: int
    registrations: int
    contact_points: int
    components_before: int
    components_removed: int


@dataclass(frozen=True, slots=True)
class Corroboration:
    """What the clock added to this run, which is evidence for and against and never a
    grouping edge (ADR-0005).

    Beside `SharedFilter`, which removes accounts from the graph, this one removes
    nothing. `deprioritised` is every candidate the window corroborates nothing under, and
    it is carried whole rather than counted so the output can name each one: a grouping
    that got weaker for want of a second account in the same hour is a decision, and a
    decision a reader cannot see is the same as a defect nobody noticed.

    The two candidate counts and the two piece counts are published side by side because
    they can disagree, and when they do the disagreement is the finding — a candidate
    corroborated by one registration and not by the Contact Point that joined its third
    account is the shape this Corpus actually produces.
    """

    window_seconds: int
    candidates: int
    corroborated: int
    pieces: int
    corroborated_pieces: int
    deprioritised: tuple[CampaignCandidate, ...]


@dataclass(frozen=True, slots=True)
class SimilarityCorroboration:
    """What the stored vectors added to this run, which is evidence for and against
    and never a grouping edge (ADR-0005, ADR-0026).

    Mirrors `Corroboration`: beside `SharedFilter`, which removes accounts from the
    graph, this one removes nothing. `deprioritised` is every candidate the threshold
    corroborates nothing under, and it is carried whole rather than counted so the
    output can name each one: a grouping that got weaker for want of a second account
    retelling the same words is a decision, and a decision a reader cannot see is the
    same as a defect nobody noticed. The heading it is printed under says `filtered`,
    which is the Ticket's word, and the sentence under that heading says that none of
    them is removed for it.
    """

    threshold: float
    candidates: int
    corroborated: int
    pieces: int
    corroborated_pieces: int
    deprioritised: tuple[CampaignCandidate, ...]


@dataclass(frozen=True, slots=True)
class CohesionSystem:
    """What the cohesion system retained of the baseline, and what it removed.

    The two tiers, as figures over one another rather than as two runs. `candidates` is
    the baseline whole — every component the union-find produced, which is exactly what
    the file holds and exactly what it held before this system existed — and every other
    figure here is counted off it, so the two numbers cannot drift and neither can be
    published without the other being computable from the same tuple.

    Accounts are counted across candidates rather than per candidate: two candidates
    sharing an account would otherwise report one account twice, and the difference
    between the two tiers is the whole of what corroboration bought, so a figure that
    over-counted would over-state it.
    """

    candidates: tuple[CampaignCandidate, ...]

    @property
    def baseline(self) -> int:
        """What direct adjacency found, which is the number this run must not lose."""
        return len(self.candidates)

    @property
    def accounts(self) -> int:
        """The accounts in the baseline, over both edges and counted once each."""
        return len({account for candidate in self.candidates for account in candidate.accounts})

    @property
    def retained(self) -> int:
        """What the cohesion system kept of the baseline."""
        return sum(1 for candidate in self.candidates if candidate.cohesion.retained)

    @property
    def retained_accounts(self) -> int:
        """The accounts in what the cohesion system kept."""
        return len(
            {
                account
                for candidate in self.candidates
                if candidate.cohesion.retained
                for account in candidate.accounts
            }
        )

    @property
    def filtered(self) -> tuple[CampaignCandidate, ...]:
        """Every candidate the cohesion system removed, in the order the table prints them.

        Carried whole rather than counted, for the same reason `Corroboration` carries its
        deprioritised candidates: a filtering decision a reader cannot see is the same as
        a defect nobody noticed, and each of these has to be nameable with the evidence
        that removed it.
        """
        return tuple(candidate for candidate in self.candidates if not candidate.cohesion.retained)

    @property
    def removed(self) -> int:
        """What the system removed, which is the whole difference between the two tiers."""
        return self.baseline - self.retained


@dataclass(frozen=True, slots=True)
class Grouping:
    """Everything one run establishes, so the table and the file cannot disagree.

    `temporal` is what the clock added across the run, `content` what the stored vectors
    added, and `cohesion` what those two and the Scam Categories add together (ADR-0028).
    Each is one value over the whole run rather than a figure recomputed at each place it
    is printed. The Corpus is carried because the printed table quotes its posts: the
    claims are about content in that file, and the evidence is read from it rather than
    transcribed.
    """

    facts: GroupingFacts
    filtered: SharedFilter
    temporal: Corroboration
    content: SimilarityCorroboration
    cohesion: CohesionSystem
    candidates: tuple[CampaignCandidate, ...]
    corpus: tuple[CorpusItem, ...] = field(repr=False)


def group(
    corpus_path: Path,
    list_path: Path,
    shared_path: Path,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    table: str = DEFAULT_TABLE,
) -> Grouping:
    """Read the three files and the stored vectors, and return the candidates, the
    figures, and the Corpus.

    One call, because the figures, the candidates, and the printed evidence are three
    views of one pass over the Corpus. A caller that resolved the links and then asked
    for the candidates separately would do the work twice and could end up printing
    one run's figures over another's results.

    The rows are grouped twice: once with the known-shared registrations withheld,
    which is the result, and once with nothing withheld, which is the adjacency
    baseline ADR-0009 asks for and the thing the filter is measured against. Both
    come from the same rows and the same union-find, so the difference between the two
    is the filter and nothing else.

    The baseline is what the file holds and what it has always held, and the cohesion
    system filters it rather than replacing it: `CohesionSystem` carries the whole of
    the baseline, so the two tiers are figures over one tuple rather than two runs that
    could only be compared against each other by eye (ADR-0028).

    The resolved links and the Contact Points are arguments rather than something read
    here. They are produced by the same `post_domains` and the same `contacts.extract`
    that the earlier commands publish, so the two cannot disagree unless those
    functions are wrong — which their own tests hold them to. The published files are
    not read: the grouping is a function of the Corpus alone, and re-reading a derived
    file would make the result depend on whether somebody had remembered to run the
    step before it.

    The Scam Categories are placed here rather than read from
    `data/corpus/composition.jsonl`, for the same reason the links are resolved rather
    than read: the grouping is a function of the Corpus, and a fifth published file would
    make it a function of whether somebody had run a command first. `placement.place` is
    the function the other two commands place with, so the three cannot disagree about
    what a post's class is.

    The vectors are the exception, and they are read from the table rather than
    recomputed: they are the one input this call cannot have another answer to, and
    reading them is what makes the similarity figures the ones a similarity query over
    that table would return (ADR-0026). A post the table does not hold refuses the run
    by name, because a figure about a candidate that quietly skipped one of its posts is
    a figure about a different candidate.

    Candidates come out largest first, then by the accounts' names, so a fixed Corpus
    produces a fixed list under fixed identifiers — under the clock's key and then the
    vectors', because that is the whole of what either is allowed to do here
    (ADR-0005). The cohesion system does not reorder them, deliberately: the baseline and
    its identifiers are what the file has published since before the system existed, and
    moving them would make the pre-corroboration number a reconstruction rather than a
    record. What the system decided is said beside each candidate instead. The identifier
    says where a candidate sits in the output and nothing else: it is not an identity that
    survives the Corpus changing underneath it, or the threshold moving.
    """
    window_seconds = _window_seconds(window_hours)
    _require_threshold(threshold)
    suffixes = PublicSuffixes.read(list_path)
    corpus = read_corpus(corpus_path)
    shared = SharedInfrastructure.read(shared_path, suffixes)
    rows = post_domains(corpus, suffixes)
    points = shared_contact_points(read_contact_points(corpus))
    created_at = {item.post_id: item.created_at for item in corpus}
    items = {item.post_id: item for item in corpus}
    check_placements()
    categories = {item.post_id: place(item).scam_category for item in corpus}

    connection = open_store(connection_string())
    try:
        vectors = stored_vectors(connection, table, items.keys())
    finally:
        connection.close()

    withheld = shared.withheld()
    kept = _Reach.of(rows, points, created_at, withheld, items, vectors, categories)
    unfiltered = _Reach.of(rows, points, created_at, frozenset(), items, vectors, categories)
    candidates = tuple(
        kept.candidate(f"cc-{number:02d}", accounts, window_seconds, threshold)
        for number, accounts in enumerate(kept.ordered(window_seconds, threshold), start=1)
    )
    shared_filter = _filter(rows, unfiltered, points, candidates, shared, withheld)
    return Grouping(
        facts=_facts(
            corpus_path,
            list_path,
            rows,
            suffixes,
            points,
            candidates,
            shared_filter.silenced,
        ),
        filtered=shared_filter,
        temporal=_corroboration(window_seconds, candidates),
        content=_similarity_corroboration(threshold, candidates),
        cohesion=CohesionSystem(candidates),
        candidates=candidates,
        corpus=corpus,
    )


def _window_seconds(window_hours: int) -> int:
    """The window in seconds, refusing a window that could never corroborate anything.

    Zero would make the closest pair of accounts fall outside it however close the two
    posts are, and a negative window is the same rule reached from the other side. The
    run refuses it rather than publishing a figure about a threshold that cannot be met,
    which is what makes `--window-hours 0` a mistake somebody is told about rather than a
    Corpus every candidate of which is uncorroborated for no reason.
    """
    if window_hours < 1:
        raise ValueError(
            f"--window-hours is {window_hours}, and a window has to be at least one hour: "
            "nothing falls inside a window of less than that"
        )
    return window_hours * 3_600


def _require_threshold(threshold: float) -> None:
    """The threshold, refused when it cannot discriminate anything.

    Zero would corroborate nothing whatever this corpus holds, and past one cosine
    distance the model calls every text near-identical: a threshold nothing can fail
    is a Signal that fires on nothing, which is a claim. This is the same shape of
    refusal `_window_seconds` makes for a window of nothing.
    """
    if threshold <= 0 or threshold > 1.0:
        raise ValueError(
            f"--similarity-threshold is {threshold}, and a cosine distance between two "
            "posts counts as near-identical only when it is above zero and at most one: "
            "at or below zero nothing corroborates, and above one everything would"
        )


def _similarity_corroboration(
    threshold: float, candidates: Sequence[CampaignCandidate]
) -> SimilarityCorroboration:
    """What the vectors say across the whole run, and which candidates they say
    nothing for. Counted off the candidates themselves, so the figure and the rows
    under each candidate cannot disagree."""
    return SimilarityCorroboration(
        threshold=threshold,
        candidates=len(candidates),
        corroborated=sum(1 for candidate in candidates if candidate.content_similarity.corroborates),
        pieces=sum(len(candidate.similarity) for candidate in candidates),
        corroborated_pieces=sum(
            candidate.content_similarity.corroborated for candidate in candidates
        ),
        deprioritised=tuple(
            candidate for candidate in candidates if not candidate.content_similarity.corroborates
        ),
    )


def _corroboration(
    window_seconds: int, candidates: Sequence[CampaignCandidate]
) -> Corroboration:
    """What the clock says across the whole run, and which candidates it says nothing about.

    Counted off the candidates themselves rather than recomputed, so the figure and the
    lines under each candidate cannot disagree: one candidate's timing is one count and
    this sums them.
    """
    return Corroboration(
        window_seconds=window_seconds,
        candidates=len(candidates),
        corroborated=sum(1 for candidate in candidates if candidate.timing.corroborates),
        pieces=sum(len(candidate.corroboration) for candidate in candidates),
        corroborated_pieces=sum(candidate.timing.corroborated for candidate in candidates),
        deprioritised=tuple(
            candidate for candidate in candidates if not candidate.timing.corroborates
        ),
    )


@dataclass(frozen=True, slots=True)
class _Reach:
    """Who reaches what: the whole of what the grouping is read out of.

    Three indexes per question rather than a scan, because every candidate asks the same
    questions about its own accounts and about the registrations and Contact Points they
    share, and answering each by walking the Corpus would make the cost of a candidate
    depend on the size of the Corpus rather than on the size of the candidate.

    `points` is the Contact Points that survived the filter, keyed as `contacts.py`
    keys them. It is carried whole rather than re-indexed by accounts, because the
    spellings and the post list of each one are evidence the candidate has to publish
    and there is nothing to be gained by taking them apart to put them back together.

    `moments` is each account's posts as sorted instants, which is the whole of what the
    clock needs: one index rather than a re-read of the Corpus per piece of evidence,
    and it is built from the same `posts_by_account` the grouping was built from, so a
    post that reached no registration still counts towards when its account was active.
    That matters — the complaint a victim writes about a desk arrives weeks after the
    desk's own posts, and it is the timestamp of both that says so.

    `categories` is each post's Scam Category, placed once by `group` and handed to both
    of this class's graphs, so the two are reading one pass rather than placing the
    Corpus twice. The category tally is counted per component like everything else here,
    because the question it answers — do these accounts agree on a pitch — is a question
    about the component and not about the Corpus.
    """

    accounts_by_domain: dict[str, set[str]]
    posts_by_domain: dict[str, set[str]]
    points: Mapping[tuple[ContactKind, str], SharedContact]
    posts_by_account: dict[str, set[str]]
    first_seen: dict[str, str]
    moments: dict[str, tuple[datetime, ...]]
    items: Mapping[str, CorpusItem]
    vectors: Mapping[str, Sequence[float]]
    categories: Mapping[str, str]

    @classmethod
    def of(
        cls,
        rows: Iterable[PostDomains],
        points: Mapping[tuple[ContactKind, str], SharedContact],
        created_at: dict[str, str],
        withheld: frozenset[str],
        items: Mapping[str, CorpusItem],
        vectors: Mapping[str, Sequence[float]],
        categories: Mapping[str, str],
    ) -> _Reach:
        """What the rows reach, with `withheld` kept out of the graph entirely.

        A withheld registration contributes nothing here — no edge, no accounts, no
        posts — which is what makes the filter a filter rather than a check afterwards.
        A Contact Point at a withheld registration is withheld with it, for the same
        reason and stated in the same terms: the address of a paste site is reached by
        every account that pastes into one.

        The posts of both still count towards the accounts that wrote them, because a
        post is why an account is silent, not why its other posts do not exist.
        """
        reach = cls({}, {}, {}, {}, {}, {}, items, vectors, categories)
        for row in rows:
            reach.posts_by_account.setdefault(row.account, set()).add(row.post_id)
            _earliest(reach.first_seen, row.account, created_at[row.post_id])
            for domain in row.domains:
                if domain in withheld:
                    continue
                reach.accounts_by_domain.setdefault(domain, set()).add(row.account)
                reach.posts_by_domain.setdefault(domain, set()).add(row.post_id)
        return replace(
            reach,
            points={
                key: point for key, point in points.items() if _kept(point, withheld)
            },
            moments={
                account: tuple(sorted(_moment(created_at[post]) for post in posts))
                for account, posts in reach.posts_by_account.items()
            },
        )

    def order_key(self, accounts: frozenset[str]) -> tuple[int, int, tuple[str, ...]]:
        """Largest first, then most posts, then by name, so identifiers are stable.

        The names are the last tiebreak because they are the only part of a component
        that does not depend on how many posts the Corpus happens to hold for it.
        """
        return (-len(accounts), -len(self.posts_of(accounts)), tuple(sorted(accounts)))

    def ordered(self, window_seconds: int, threshold: float) -> list[frozenset[str]]:
        """The components, the corroborated ones first, then the usual order.

        Two keys, in that order of precedence: whether the clock corroborates the
        component, then whether the stored vectors do. The clock keeps its precedence
        because it had this job first (ADR-0025) and nothing here retires it, and the
        vectors get the second key because a threshold is also only allowed to
        deprioritise (ADR-0026) - never to remove, and never to promote anything the
        clock has put behind a candidate it corroborates nothing under. Within both,
        the usual order: largest first, then most posts, then by name.

        Sorting on it at all is the difference between the ordering being the run's
        opinion and a figure being a column of a table somebody can ignore.
        """
        return sorted(
            self.components(),
            key=lambda accounts: (
                not self.timing(accounts, window_seconds).corroborates,
                not self.content_similarity(accounts, threshold).corroborates,
                *self.order_key(accounts),
            ),
        )

    def posts_of(self, accounts: frozenset[str]) -> tuple[str, ...]:
        """Every post written by a set of accounts, sorted."""
        return tuple(
            sorted({post for account in accounts for post in self.posts_by_account[account]})
        )

    def reaching(self, registration: str) -> frozenset[str]:
        """The accounts that reach one registration, or none.

        Asked of the unfiltered graph, so it answers what the Corpus does with a
        registration whether or not the filter still lets it group anything.
        """
        return frozenset(self.accounts_by_domain.get(registration, ()))

    def timing(self, accounts: frozenset[str], window_seconds: int) -> Timing:
        """What the clock says about one component, against a stated window.

        Worked out from the same gaps the candidate publishes, by the same function, so the
        ordering and the line printed under a candidate cannot disagree about which
        candidates are corroborated. It is worked out twice — once to sort the components
        and once to build the candidate they become — because the order has to be settled
        before any identifier is assigned, and the gaps are a pure function of the
        component, so there is nothing for the two to disagree about.
        """
        return _summary(self.corroboration(accounts), window_seconds)

    def content_similarity(self, accounts: frozenset[str], threshold: float) -> ContentSimilarity:
        """What the stored vectors say about one component, against a stated threshold.

        Worked out from the same rows the candidate publishes, by the same function, so
        the ordering and the figure under a candidate cannot disagree about which
        candidates the vectors corroborate. Worked out twice - once to sort the
        components and once to build the candidate they become - because the order has
        to be settled before any identifier is assigned, and the rows are a pure
        function of the component, so there is nothing for the two to disagree about.
        """
        return _similarity_summary(self.similarity(accounts), threshold)

    def category_agreement(self, accounts: frozenset[str]) -> CategoryAgreement:
        """What the Scam Categories say about one component, against ADR-0021's rule.

        Counted over every post the component holds rather than the posts carrying shared
        evidence, because the question is whether the accounts inside it are making the
        same pitch — and an account that published a handle once and posts about something
        else for a year was still part of the pitch when it did.
        """
        return _agreement([self.categories[post_id] for post_id in self.posts_of(accounts)])

    def similarity(self, accounts: frozenset[str]) -> tuple[SimilarityEvidence, ...]:
        """The similarity rows of every post of one component."""
        return within_component_similarity(
            [self.items[post_id] for post_id in self.posts_of(accounts)], self.vectors
        )

    def corroboration(self, accounts: frozenset[str]) -> tuple[EvidenceTiming, ...]:
        """The timing of every piece of evidence a component is joined on."""
        return self._joined(accounts).timing

    def _joined(self, accounts: frozenset[str]) -> _Joined:
        """What one component rests on, and how far apart those accounts posted.

        Only something two of the component's accounts reach is named: it is the reason
        they are together, and every registration is in the resolved links and every
        Contact Point in the file the reading published. A registration one account of the
        component reaches is context rather than a reason, and a gap measured over the
        accounts that joined nothing would be a number about the Corpus rather than about
        this candidate.
        """
        domains = tuple(
            SharedDomain(
                domain=domain,
                accounts=tuple(sorted(accounts & self.accounts_by_domain[domain])),
                posts=tuple(sorted(self.posts_by_domain[domain])),
            )
            for domain in sorted(self.accounts_by_domain)
            if len(accounts & self.accounts_by_domain[domain]) > 1
        )
        points = tuple(
            replace(point, accounts=tuple(sorted(accounts & set(point.accounts))))
            for _, point in sorted(self.points.items())
            if len(accounts & set(point.accounts)) > 1
        )
        pieces = [
            *(
                (EvidenceKind.REGISTRATION, shared.domain, shared.accounts)
                for shared in domains
            ),
            *(
                (EvidenceKind.CONTACT_POINT, point.value, point.accounts)
                for point in points
            ),
        ]
        return _Joined(
            domains=domains,
            points=points,
            timing=tuple(self._gap(kind, value, reaching) for kind, value, reaching in pieces),
            similarity=within_component_similarity(
                [self.items[post_id] for post_id in self.posts_of(accounts)], self.vectors
            ),
            category=self.category_agreement(accounts),
        )

    def _gap(self, kind: EvidenceKind, value: str, reaching: Collection[str]) -> EvidenceTiming:
        """One piece of evidence, and how far apart the accounts that reach it posted.

        The nearest pair of accounts is found by walking the posts in time order and
        looking at the neighbouring pairs that belong to different accounts, rather than
        by pairing every account with every other: the closest two posts by different
        accounts are always neighbours in that order, because anything between them would
        be nearer to one of them. So the walk is linear in the posts of the piece rather
        than quadratic in them, and a candidate over a thousand posts costs no more than
        one over ten.

        The span is the first post to the last, over every account reaching the piece and
        every post any of them wrote — not only the posts carrying it. An account that
        published a handle once and posts about something else for a year was still
        active for a year, and that is the span a reader wants beside the gap.

        An account that reached the evidence with no post in the Corpus is refused rather
        than skipped, because there is no moment for it and a skipped one would leave the
        span measuring fewer accounts than the evidence names.
        """
        silent = sorted(set(reaching) - self.moments.keys())
        if silent:
            raise ValueError(
                f"{value} is reached by {silent}, and an account with no post in the Corpus "
                "has no moment to be timed at"
            )
        ordered = sorted(
            (moment, account) for account in reaching for moment in self.moments[account]
        )
        gaps = (
            (later[0] - earlier[0]).total_seconds()
            for earlier, later in zip(ordered, ordered[1:], strict=False)
            if earlier[1] != later[1]
        )
        return EvidenceTiming(
            kind=kind,
            value=value,
            closest_seconds=int(min(gaps, default=0.0)),
            span_seconds=int((ordered[-1][0] - ordered[0][0]).total_seconds()),
        )

    def candidate(
        self,
        candidate_id: str,
        accounts: frozenset[str],
        window_seconds: int,
        threshold: float,
    ) -> CampaignCandidate:
        """One component, with the shared registrations and Contact Points that join it
        rather than every one it touches, and the timing of each.

        Every post of every account in the component is listed, not only the posts that
        carry shared evidence — an account reached through one link or one handle may have
        posts that show what else it does, and that is the material a reviewer reads the
        grouping against.

        A Contact Point's spellings are the Corpus's rather than the component's, and
        that is deliberate: the spellings are what tells sharing from folding, and
        leaving out the one that made a second account's post read as the same value
        would hide the only evidence that it was a fold rather than a repetition.

        The evidence and its timing come out of one pass, so a candidate cannot print a gap
        beside a piece of evidence it does not name or name one it has no gap for.
        """
        joined = self._joined(accounts)
        timing = _summary(joined.timing, window_seconds)
        similarity = _similarity_summary(joined.similarity, threshold)
        return CampaignCandidate(
            candidate_id=candidate_id,
            accounts=tuple(sorted(accounts)),
            posts=self.posts_of(accounts),
            shared_domains=joined.domains,
            shared_contact_points=joined.points,
            corroboration=joined.timing,
            timing=timing,
            similarity=joined.similarity,
            content_similarity=similarity,
            cohesion=Cohesion(timing=timing, similarity=similarity, category=joined.category),
            first_seen=min(self.first_seen[account] for account in accounts),
        )

    def components(self) -> list[frozenset[str]]:
        """The connected components of the accounts-over-shared-infrastructure graph,
        two or more wide.

        Union-find over both edges at once, because the question is reachability and not
        adjacency: three accounts where the first and second share a registration and the
        second and third share a Contact Point are one component, and a rule that only
        paired accounts off per thing would report two overlapping groups with no way to
        say what either one was.
        """
        parent: dict[str, str] = {}

        def find(account: str) -> str:
            parent.setdefault(account, account)
            root = account
            while parent[root] != root:
                root = parent[root]
            while parent[account] != root:
                parent[account], account = root, parent[account]
            return root

        for linked in sorted(
            [sorted(accounts) for accounts in self.accounts_by_domain.values()]
            + [sorted(point.accounts) for point in self.points.values()]
        ):
            for other in linked[1:]:
                first, second = find(linked[0]), find(other)
                if first != second:
                    parent[second] = first

        grouped: dict[str, set[str]] = {}
        for account in parent:
            grouped.setdefault(find(account), set()).add(account)
        return [frozenset(component) for component in grouped.values() if len(component) > 1]


def _kept(point: SharedContact, withheld: frozenset[str]) -> bool:
    """Whether a Contact Point survives the filter.

    An address under a withheld registration is withheld with it; a handle is not
    filtered by anything, because there is no published list of handles that everybody
    uses and this project is not going to invent one. The filter's claim is about
    services rather than about posts, and a handle is not a service anybody registered.
    """
    if point.kind is not ContactKind.EMAIL:
        return True
    _, _, host = point.value.rpartition("@")
    return host not in withheld


def _earliest(first_seen: dict[str, str], account: str, created_at: str) -> None:
    """Keep the earliest post per account. RFC 3339 UTC sorts lexically as it reads."""
    if account not in first_seen or created_at < first_seen[account]:
        first_seen[account] = created_at


def _summary(pieces: Sequence[EvidenceTiming], window_seconds: int) -> Timing:
    """One candidate's timing, from the gaps it publishes.

    The one place the four figures are counted, so the ordering decision, the line under
    the candidate and the line in the file are the same arithmetic. `closest_seconds` is
    published whatever the window says: it is the gap a reader needs in order to disagree
    with the window rather than the number the window produced.
    """
    return Timing(
        window_seconds=window_seconds,
        pieces=len(pieces),
        corroborated=sum(1 for piece in pieces if piece.within(window_seconds)),
        closest_seconds=min((piece.closest_seconds for piece in pieces), default=0),
    )


def within_component_similarity(
    posts: Sequence[CorpusItem], vectors: Mapping[str, Sequence[float]]
) -> tuple[SimilarityEvidence, ...]:
    """Each post's nearest post by another account, by cosine distance over the stored vectors.

    The unit vectors the store holds make distance a dot product away: `1 - dot(u, v)` is
    the cosine distance the database computes itself, so the figure a candidate publishes
    is the one a similarity query over the same table would return. Rows come out in post
    order, so the published figure and the file carrying it move together.

    A post missing from `vectors` is refused rather than skipped: the figure would say
    "every post has a twin this close" while one post was never asked, and a candidate
    flattered by a question nothing asked is the shape of miss this command exists to
    make visible. Two vectors of different widths are refused rather than left to the
    zip, which would silently compare the shared prefix of two different kinds of number.
    """
    widths = {len(vectors.get(post.post_id, ())) for post in posts}
    if 0 in widths:
        missing = sorted(post.post_id for post in posts if post.post_id not in vectors)
        raise ValueError(
            f"{missing[0]} has no stored vector, so its posts cannot be compared: run "
            "`rfi content-embeddings` over the Corpus first"
        )
    if len(widths) > 1:
        raise ValueError(
            f"the stored vectors have {sorted(widths)} widths, and a vector of one "
            "width has no distance to a vector of another"
        )
    rows: list[SimilarityEvidence] = []
    for post in sorted(posts, key=lambda item: item.post_id):
        nearest: SimilarityEvidence | None = None
        for other in posts:
            if other.account == post.account:
                continue
            distance = 1.0 - sum(
                left * right
                for left, right in zip(
                    vectors[post.post_id], vectors[other.post_id], strict=True
                )
            )
            if nearest is None or distance < nearest.distance:
                nearest = SimilarityEvidence(
                    post=post.post_id,
                    account=post.account,
                    nearest=other.post_id,
                    nearest_account=other.account,
                    distance=distance,
                )
        if nearest is None:
            raise ValueError(
                f"{post.post_id} has no post by another account to be compared with, and a "
                "candidate of one account is not a candidate"
            )
        rows.append(nearest)
    return tuple(rows)


def _similarity_summary(
    rows: Sequence[SimilarityEvidence], threshold: float
) -> ContentSimilarity:
    """One candidate's similarity, counted from the rows it publishes.

    The one place the four figures are counted, so the ordering decision, the line
    under the candidate and the line in the file are the same arithmetic.
    `closest_distance` is published whatever the threshold says: it is the gap a
    reader needs in order to disagree with the threshold rather than the number the
    threshold produced.

    An empty row list is refused rather than counted as zero. A component of two or
    more accounts always holds two or more posts, so an empty one means the rows were
    never worked out — and publishing "nearest pair 0.00 apart" for a candidate with
    no pair at all would be the figure that reads like the strongest agreement there
    is.
    """
    if not rows:
        raise ValueError(
            "a Campaign Candidate of two or more accounts always holds posts to "
            "compare, so a candidate with no similarity rows has had none worked out"
        )
    return ContentSimilarity(
        threshold=threshold,
        pieces=len(rows),
        corroborated=sum(1 for row in rows if row.distance <= threshold),
        closest_distance=min(row.distance for row in rows),
    )


def _moment(created_at: str) -> datetime:
    """One post's timestamp as an instant, so two of them can be subtracted.

    The Corpus writes RFC 3339 in UTC and the writer sorts on it lexically, which is
    chronological order as long as the offset is always the same one — a Corpus carrying
    both `Z` and `+02:00` would sort wrongly there, and parsing here is what makes the
    gaps right anyway. A timestamp that will not parse is refused by name rather than
    quietly read as zero: a gap of no length at all is the one figure in this module that
    would corroborate everything.

    A timestamp with no offset is refused too, and separately, because it is not a failure
    of parsing but of arithmetic: subtracting it from one that carries an offset raises a
    `TypeError` from inside the subtraction rather than anything a reader of this output
    could act on. The Corpus generator writes `Z` throughout; a Corpus Provider that wrote
    a local time would be refused here and named, which is the point of the refusal.
    """
    try:
        moment = datetime.fromisoformat(created_at)
    except ValueError as refusal:
        raise ValueError(
            f"created_at={created_at!r} is not an RFC 3339 timestamp, and the timing of a "
            "post cannot be read from it"
        ) from refusal
    if moment.tzinfo is None:
        raise ValueError(
            f"created_at={created_at!r} carries no offset, and a gap between it and a "
            "timestamp that carries one is not a length"
        )
    return moment


def _facts(
    corpus_path: Path,
    list_path: Path,
    rows: Sequence[PostDomains],
    suffixes: PublicSuffixes,
    points: Mapping[tuple[ContactKind, str], SharedContact],
    candidates: Sequence[CampaignCandidate],
    silenced: frozenset[str],
) -> GroupingFacts:
    """Every figure about the Corpus the table prints, from the rows, the candidates,
    and the bytes behind them.

    The four account counts are a partition, so each is decided per account rather
    than per post: an account is unreachable when none of its own posts names a
    registration or a Contact Point, and alone when it names one that no other account
    names. Counting a post instead would call an account unreachable while another of
    its posts was sitting in a candidate, and the four figures would no longer add up
    to the Corpus's accounts.

    `points` is every Contact Point the Corpus published rather than the graph the
    filter left standing, because "reaches nothing" is a claim about the Corpus and not
    about the filter. An account whose only Contact Point sat under a withheld
    registration does reach something, and what the filter did about it is `silenced`'s
    to say, which is why that is passed in rather than worked out here. Reading the kept
    graph instead would put such an account in two of the four lines at once, and the
    four would then describe more accounts than the Corpus holds.

    `silenced` is the filter's count and not this function's, so it is passed in rather
    than worked out again: two ways of asking whether an account reached anything
    usable is two ways to disagree about the partition.
    """
    accounts = {row.account for row in rows}
    reaching = {row.account for row in rows if row.domains} | {
        account for point in points.values() for account in point.accounts
    }
    grouped = {account for candidate in candidates for account in candidate.accounts}
    unreachable = accounts - reaching
    return GroupingFacts(
        accounts=len(accounts),
        accounts_alone=len(accounts - grouped - unreachable - silenced),
        accounts_in_candidates=len(grouped),
        accounts_unreachable=len(unreachable),
        candidates=len(candidates),
        corpus_path=corpus_path.as_posix(),
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        posts=len(rows),
        shared_contact_points=len(
            {
                (joining.kind, joining.value)
                for candidate in candidates
                for joining in candidate.shared_contact_points
            }
        ),
        shared_domains=len(
            {
                joining.domain
                for candidate in candidates
                for joining in candidate.shared_domains
            }
        ),
        suffix_list_path=list_path.as_posix(),
        suffix_list_sha256=hashlib.sha256(list_path.read_bytes()).hexdigest(),
        rules_sha256=suffixes.rules_digest(),
    )


def _filter(
    rows: Sequence[PostDomains],
    unfiltered: _Reach,
    points: Mapping[tuple[ContactKind, str], SharedContact],
    candidates: Sequence[CampaignCandidate],
    shared: SharedInfrastructure,
    withheld: frozenset[str],
) -> SharedFilter:
    """What the list took out of the graph, and what difference that made.

    The withheld registrations are read off the unfiltered graph rather than off the
    list, so a host nobody in this Corpus reaches is not counted as withheld and the
    account figures are the number of accounts that actually reach one. `silenced` is
    the accounts that reached nothing usable, which is a smaller set than the accounts
    reaching something withheld: an account can share a registration or a handle with
    another account and paste the same advert, and the filter says nothing about it.

    The Contact Points that left the graph are read the same way, off the values the
    Corpus published, so the same argument holds for them: the figure is the number of
    points somebody actually published, not the number of ways this build could have
    filtered one.
    """
    rows_out: list[WithheldRegistration] = []
    reaching: set[str] = set()
    for published in shared.registrations:
        accounts = unfiltered.reaching(published.registration)
        if not accounts:
            continue
        rows_out.append(
            WithheldRegistration(
                registration=published.registration,
                hosts=published.hosts,
                accounts=len(accounts),
            )
        )
        reaching |= accounts

    withheld_points = tuple(
        WithheldContactPoint(
            kind=point.kind,
            value=point.value,
            registration=point.value.rpartition("@")[2],
            accounts=point.accounts,
        )
        for _, point in sorted(points.items())
        if not _kept(point, withheld)
    )
    for point in withheld_points:
        reaching |= set(point.accounts)

    usable = {
        row.account for row in rows if any(domain not in withheld for domain in row.domains)
    } | {
        account
        for _, point in points.items()
        if _kept(point, withheld)
        for account in point.accounts
    }
    baseline = unfiltered.components()
    proposed = {frozenset(candidate.accounts) for candidate in candidates}
    return SharedFilter(
        published=shared,
        withheld=tuple(rows_out),
        withheld_points=withheld_points,
        silenced=frozenset(reaching - usable),
        accounts_reaching_withheld=len(reaching),
        registrations=len(unfiltered.accounts_by_domain),
        contact_points=len(points),
        components_before=len(baseline),
        components_removed=sum(1 for component in baseline if component not in proposed),
    )


def write_campaign_candidates(path: Path, candidates: Sequence[CampaignCandidate]) -> None:
    """The candidates, in the order the table prints them.

    The file holds the same evidence the console prints, so a reader who wants the
    list rather than the argument reads the same claims in a different shape.
    """

    def objects() -> Iterator[JsonObject]:
        for candidate in candidates:
            yield {
                "candidate_id": candidate.candidate_id,
                "accounts": list(candidate.accounts),
                "posts": list(candidate.posts),
                "shared_domains": [
                    {
                        "domain": shared.domain,
                        "accounts": list(shared.accounts),
                        "posts": list(shared.posts),
                    }
                    for shared in candidate.shared_domains
                ],
                "shared_contact_points": [
                    {
                        "kind": point.kind.value,
                        "value": point.value,
                        "accounts": list(point.accounts),
                        "posts": list(point.posts),
                        "spellings": list(point.spellings),
                    }
                    for point in candidate.shared_contact_points
                ],
                "corroboration": [
                    {
                        "kind": piece.kind.value,
                        "value": piece.value,
                        "closest_seconds": piece.closest_seconds,
                        "span_seconds": piece.span_seconds,
                    }
                    for piece in candidate.corroboration
                ],
                "timing": {
                    "window_seconds": candidate.timing.window_seconds,
                    "pieces": candidate.timing.pieces,
                    "corroborated": candidate.timing.corroborated,
                    "closest_seconds": candidate.timing.closest_seconds,
                },
                "similarity": [
                    {
                        "post": row.post,
                        "account": row.account,
                        "nearest": row.nearest,
                        "nearest_account": row.nearest_account,
                        "distance": row.distance,
                    }
                    for row in candidate.similarity
                ],
                "content_similarity": {
                    "threshold": candidate.content_similarity.threshold,
                    "pieces": candidate.content_similarity.pieces,
                    "corroborated": candidate.content_similarity.corroborated,
                    "closest_distance": candidate.content_similarity.closest_distance,
                },
                "cohesion": {
                    "against": list(candidate.cohesion.against),
                    "category": {
                        "tally": [
                            [name, count] for name, count in candidate.cohesion.category.tally
                        ]
                    },
                    "corroborating": list(candidate.cohesion.corroborating),
                    "retained": candidate.cohesion.retained,
                },
                "first_seen": candidate.first_seen,
            }

    write_lines(path, objects())


def read_campaign_candidates(path: Path) -> tuple[CampaignCandidate, ...]:
    """The candidates this project published, checked row by row.

    The reader lives beside the writer because they are one vocabulary: a field
    cannot be added to `CampaignCandidate` and left out of the file, and a field
    cannot be in the file that the reader does not check. That is the same
    arrangement `CorpusItem` has, and it is checked in the same way — an unknown
    field is refused rather than ignored, because the one field that must never
    appear in this file is a Planted Campaign's identifier, and a reader that
    skipped what it did not recognise would skip that without saying so
    (ADR-0008). The evaluator is what depends on this: it is the only command
    allowed to read the truth file, so the file it measures against has to be one
    it can prove carries no membership of its own.

    A row naming one account is refused as well. A component of one is not a
    Campaign Candidate (ADR-0005), so a file holding one is a file describing a
    grouping this project does not produce, and the recovery figure would then be
    joining against something other than the grouping's output.

    Every row is also checked against the arithmetic it publishes beside itself, which
    is the arrangement `read_policy_scores` has: the timing on a row has to name exactly
    the evidence the row rests on, to count as many pieces as there are of them, and to
    carry the nearest gap among them. A file whose rows disagree with themselves about
    when their accounts posted is a file a reader cannot use to weigh a candidate, and
    the two commands that read it would print the disagreement as though it were a fact.

    One window across the whole file, for the same reason `read_policy_scores` demands
    one published weight set: `corroborated` means nothing without the threshold it was
    counted against, so a file written by two runs at two windows is refused rather than
    read as one figure.

    The cohesion row is checked the same way and with more behind it, because it is
    counted from three verdicts rather than from one list: the tally has to name Scam
    Categories the projection publishes, has to account for every post the row holds, and
    the count beside it has to be the number of the three verdicts that actually hold on
    this row. The third is the one that cannot be recomputed from anything else here — the
    evaluator cannot place posts without opening the phrase lists, and opening them would
    put a fifth file into a command that does no inference — so it is checked for what it
    must be consistent with and taken as published for the rest.
    """
    candidates = tuple(_candidate(path, number, text) for number, text in read_rows(path))
    refuse_repeated(
        path.as_posix(),
        (candidate.candidate_id for candidate in candidates),
        "a join would count a candidate twice",
    )
    windows = {candidate.timing.window_seconds for candidate in candidates}
    if len(windows) > 1:
        raise ValueError(
            f"{path.as_posix()} publishes {sorted(windows)} seconds of timing window across "
            "its rows, and a corroborated count means nothing without the window it was "
            "counted against"
        )
    thresholds = {candidate.content_similarity.threshold for candidate in candidates}
    if len(thresholds) > 1:
        raise ValueError(
            f"{path.as_posix()} publishes {sorted(thresholds)} of similarity threshold "
            "across its rows, and a corroborated count means nothing without the "
            "threshold it was counted against"
        )
    return candidates


def _candidate(path: Path, number: int, text: str) -> CampaignCandidate:
    where = f"{path.as_posix()}:{number}"
    record = read_object(where, text)
    vocabulary = tuple(field.name for field in fields(CampaignCandidate))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds {sorted(record)}, which is not the candidate vocabulary "
            f"{sorted(vocabulary)}"
        )
    accounts = read_names(where, record, "accounts")
    if len(accounts) < 2:
        raise ValueError(
            f"{where} names {_count(len(accounts), 'account')}, and a component of one is "
            "not a Campaign Candidate (ADR-0005)"
        )
    shared = _entries(where, record, "shared_domains")
    points = _entries(where, record, "shared_contact_points")
    names = _joined_on(where, shared, points)
    pieces = _gaps(where, record, names)
    posts = read_names(where, record, "posts")
    similarity_rows = _similarity(where, record, accounts, posts)
    timing = _timing(where, record, names, pieces)
    content_similarity = _content_similarity(where, record, similarity_rows)
    return CampaignCandidate(
        candidate_id=read_text(where, record, "candidate_id"),
        accounts=accounts,
        posts=posts,
        shared_domains=tuple(
            _shared(where, entry, accounts) for entry in shared if isinstance(entry, dict)
        ),
        shared_contact_points=tuple(
            _point(where, entry, accounts) for entry in points if isinstance(entry, dict)
        ),
        corroboration=pieces,
        timing=timing,
        similarity=similarity_rows,
        content_similarity=content_similarity,
        cohesion=_cohesion(where, record, posts, timing, content_similarity),
        first_seen=read_text(where, record, "first_seen"),
    )


def _joined_on(
    where: str, shared: Sequence[object], points: Sequence[object]
) -> frozenset[tuple[str, str]]:
    """The `(kind, value)` pairs a candidate's own two evidence lists name.

    Read off the evidence rather than off the timing rows, so the checks below run the
    other way round: what the timing claims to describe has to be something the candidate
    is actually joined on, rather than what the candidate happens to name happening to be
    timed. One kind of pair rather than two so the comparison below is one comparison, and
    the kinds come off `Evidence.kind` so this reader cannot drift from the label the table
    prints.
    """
    named = [
        _named_evidence(where, entry, kind, field)
        for kind, field, entries in (
            (Evidence.DOMAIN, "domain", shared),
            (Evidence.CONTACT, "value", points),
        )
        for entry in entries
        if isinstance(entry, dict)
    ]
    return frozenset(named)


def _named_evidence(
    where: str, record: JsonObject, label: Evidence, field: str
) -> tuple[str, str]:
    """One piece of evidence a candidate names, as `(kind, value)`.

    Checked here rather than left to the two list readers below, because this runs first
    and a field it cannot find has to be refused with the line it is on: a refusal with no
    location is a complaint a reader of the file cannot act on, which is the whole reason
    `read_object` hands the line number down.
    """
    if field not in record:
        raise ValueError(
            f"{where} holds {label.value} with {sorted(record)}, and it names no {field}"
        )
    return (label.kind.value, read_text(where, record, field))


def _gaps(
    where: str, record: JsonObject, names: frozenset[tuple[str, str]]
) -> tuple[EvidenceTiming, ...]:
    """One row's timing, checked against the evidence that row rests on.

    Both directions, because both are ways the file could lie about a candidate: a timing
    row for something the candidate is not joined on is a gap the reader would weigh
    against a grouping it does not justify, and a piece of evidence with no timing row is
    a claim about a grouping that nothing measured.
    """
    entries = _entries(where, record, "corroboration")
    pieces = tuple(_gap_row(where, entry) for entry in entries if isinstance(entry, dict))
    refuse_repeated(
        f"{where} corroboration",
        (f"{piece.kind.value} {piece.value}" for piece in pieces),
        "one piece of evidence has two gaps, and a reader cannot tell which is the run's",
    )
    named = {(piece.kind.value, piece.value) for piece in pieces}
    if named != names:
        raise ValueError(
            f"{where} times {sorted(named)} and is joined on {sorted(names)}, so its "
            "corroboration is about something other than this candidate"
        )
    return pieces


def _gap_row(where: str, record: JsonObject) -> EvidenceTiming:
    """One timing row, checked as a row of its own before it is read."""
    vocabulary = tuple(field.name for field in fields(EvidenceTiming))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds a timing row with {sorted(record)}, which is not the timing "
            f"vocabulary {sorted(vocabulary)}"
        )
    closest = _seconds(where, record, "closest_seconds")
    span = _seconds(where, record, "span_seconds")
    if span < closest:
        raise ValueError(
            f"{where} has closest_seconds={closest} and span_seconds={span}, and the span "
            "across a piece of evidence cannot be shorter than the gap inside it"
        )
    return EvidenceTiming(
        kind=read_vocabulary(where, "kind", record["kind"], EvidenceKind),
        value=read_text(where, record, "value"),
        closest_seconds=closest,
        span_seconds=span,
    )


def _timing(
    where: str,
    record: JsonObject,
    names: frozenset[tuple[str, str]],
    pieces: Sequence[EvidenceTiming],
) -> Timing:
    """One row's timing summary, checked against the gaps the row publishes above it.

    Four checks, all the same claim in four shapes: the row says how many pieces of
    evidence it is timing, the list above it says how many there are, the nearest gap it
    publishes has to be the smallest one in that list, and the count of corroborated
    pieces has to be the count of those gaps that fall inside the window this row carries.
    That last one is the figure both the ordering of this project and the heading below the
    table rest on, so a row asserting it without bearing it out would be read as though the
    clock had agreed with a candidate the clock disagrees about.
    """
    carried = record["timing"]
    if not isinstance(carried, dict):
        raise ValueError(f"{where} has timing={carried!r}, which is not a row")
    vocabulary = tuple(field.name for field in fields(Timing))
    if set(carried) != set(vocabulary):
        raise ValueError(
            f"{where} holds timing with {sorted(carried)}, which is not the timing "
            f"vocabulary {sorted(vocabulary)}"
        )
    window_seconds = _seconds(where, carried, "window_seconds")
    counted = _seconds(where, carried, "pieces")
    corroborated = _seconds(where, carried, "corroborated")
    closest_seconds = _seconds(where, carried, "closest_seconds")
    if window_seconds < 3_600:
        raise ValueError(
            f"{where} was counted at a window of {window_seconds} seconds, and a window has "
            "to be at least one hour: nothing this project can produce falls inside less"
        )
    if counted != len(names):
        raise ValueError(
            f"{where} says it times {counted} pieces of evidence and is joined on "
            f"{len(names)} of them"
        )
    if corroborated > counted:
        raise ValueError(
            f"{where} corroborates {corroborated} of its {counted} pieces of evidence, and "
            "there are only that many to corroborate"
        )
    gaps = [piece.closest_seconds for piece in pieces]
    if gaps and min(gaps) != closest_seconds:
        raise ValueError(
            f"{where} says its accounts came closest {closest_seconds} seconds apart and "
            f"publishes gaps of {sorted(gaps)}"
        )
    inside = sum(1 for piece in pieces if piece.within(window_seconds))
    if inside != corroborated:
        raise ValueError(
            f"{where} says {corroborated} of its pieces of evidence fall inside its "
            f"{window_seconds} second window and publishes spans of "
            f"{sorted(piece.span_seconds for piece in pieces)}, of which {inside} do"
        )
    return Timing(
        window_seconds=window_seconds,
        pieces=counted,
        corroborated=corroborated,
        closest_seconds=closest_seconds,
    )


def _similarity(
    where: str, record: JsonObject, accounts: tuple[str, ...], posts: tuple[str, ...]
) -> tuple[SimilarityEvidence, ...]:
    """One row's similarity rows, checked against the posts that row lists.

    Both directions, the same two ways the file could lie that `_gaps` checks the
    timing rows for: a similarity row for a post the candidate does not hold is a twin
    a reader would weigh against a grouping that does not contain it, and a held post
    with no row is a figure the reader cannot check. Every row also names accounts the
    candidate's own holds, for the same reason `shared_domains` rows are asked to, and
    its twin by another one, because a twin by the same account is that account's own
    text.
    """
    entries = _entries(where, record, "similarity")
    rows = tuple(_similarity_row(where, entry) for entry in entries if isinstance(entry, dict))
    refuse_repeated(
        f"{where} similarity",
        (row.post for row in rows),
        "one post has two twins, and a reader cannot tell which is the run's",
    )
    named = {row.post for row in rows}
    if named != set(posts):
        raise ValueError(
            f"{where} holds similarity rows for {sorted(named)} and lists posts "
            f"{sorted(posts)}, so its similarity is about something other than this "
            "candidate"
        )
    for row in rows:
        if row.nearest not in posts:
            raise ValueError(
                f"{where} names {row.nearest} as a twin, and that post is not in the "
                "candidate the row is printed under"
            )
        for account in (row.account, row.nearest_account):
            _named_by(where, (account,), accounts, "an account")
        if row.account == row.nearest_account:
            raise ValueError(
                f"{where} names {row.account}'s twin by {row.nearest_account}, and a "
                "twin by the same account is that account's own text, not corroboration"
            )
    return rows


def _similarity_row(where: str, record: JsonObject) -> SimilarityEvidence:
    """One similarity row, checked as a row of its own before it is read."""
    vocabulary = tuple(field.name for field in fields(SimilarityEvidence))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds a similarity row with {sorted(record)}, which is not the "
            f"similarity vocabulary {sorted(vocabulary)}"
        )
    distance = record["distance"]
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or distance < 0
        or distance > 2
    ):
        raise ValueError(
            f"{where} has distance={distance!r}, and a cosine distance lies in [0, 2]"
        )
    post = read_text(where, record, "post")
    nearest = read_text(where, record, "nearest")
    account = read_text(where, record, "account")
    nearest_account = read_text(where, record, "nearest_account")
    if nearest == post:
        raise ValueError(
            f"{where} names {post} as its own twin, and the twin has to be a different "
            "post"
        )
    return SimilarityEvidence(
        post=post,
        account=account,
        nearest=nearest,
        nearest_account=nearest_account,
        distance=float(distance),
    )


def _content_similarity(
    where: str, record: JsonObject, rows: Sequence[SimilarityEvidence]
) -> ContentSimilarity:
    """One row's similarity summary, checked against the twins the row publishes.

    The checks are the ones `Timing` gets: the counts the row says it holds have to
    be the counts the rows above it hold, the nearest gap has to be the smallest one
    among them, the corroborated count has to be the number of rows inside the
    threshold, and the threshold cannot be a distance nothing can fail.
    """
    carried = record["content_similarity"]
    if not isinstance(carried, dict):
        raise ValueError(f"{where} has content_similarity={carried!r}, which is not a row")
    vocabulary = tuple(field.name for field in fields(ContentSimilarity))
    if set(carried) != set(vocabulary):
        raise ValueError(
            f"{where} holds content_similarity with {sorted(carried)}, which is not "
            f"the similarity vocabulary {sorted(vocabulary)}"
        )
    threshold = carried["threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or threshold <= 0
        or threshold > 1.0
    ):
        raise ValueError(
            f"{where} has threshold={threshold!r}, and a cosine-distance threshold "
            "counts as near-identical only above zero and at most one"
        )
    pieces = _seconds(where, carried, "pieces")
    corroborated = _seconds(where, carried, "corroborated")
    closest = carried["closest_distance"]
    if isinstance(closest, bool) or not isinstance(closest, (int, float)) or closest < 0:
        raise ValueError(
            f"{where} has closest_distance={closest!r}, which is not a non-negative "
            "cosine distance"
        )
    if pieces != len(rows):
        raise ValueError(
            f"{where} says it has {pieces} similarity rows and publishes {len(rows)}"
        )
    if corroborated > pieces:
        raise ValueError(
            f"{where} corroborates {corroborated} of its {pieces} posts, and there are "
            "only that many to corroborate"
        )
    gaps = [row.distance for row in rows]
    if gaps and min(gaps) != closest:
        raise ValueError(
            f"{where} says its closest pair is {closest} apart and publishes twins of "
            f"{sorted(gaps)}"
        )
    inside = sum(1 for row in rows if row.distance <= threshold)
    if inside != corroborated:
        raise ValueError(
            f"{where} says {corroborated} of its posts fall inside the {threshold} "
            f"threshold and publishes twins of {sorted(gaps)}, of which {inside} do"
        )
    return ContentSimilarity(
        threshold=float(threshold),
        pieces=pieces,
        corroborated=corroborated,
        closest_distance=float(closest),
    )


def _agreement(scams: Sequence[str]) -> CategoryAgreement:
    """One candidate's Scam Category tally, from the classes its posts were placed in.

    A pure function of the classes rather than of the posts, so the grouping and the
    reader check it over the same input and neither of them has to place anything again:
    the file holds the tally and the reader decides the majority from it, the same way
    `read_policy_scores` decides a Signal's arithmetic from the rows beside it.

    The order is a printing order rather than an input — widest class first and Other
    last, so the leading entry of the tally is never a bucket that cannot be associated
    with anything — and it is `placement.in_print_order`, the same one a registration's
    tally is printed in. `placement.agreed_category` finds the widest class itself, so a
    reader counting the tally by eye and the code deciding the majority cannot reach the
    same answer by two different routes.
    """
    counted: dict[str, int] = {}
    for scam in scams:
        counted[scam] = counted.get(scam, 0) + 1
    return CategoryAgreement(tally=in_print_order(tuple(counted.items())))


def _cohesion(
    where: str,
    record: JsonObject,
    posts: Sequence[str],
    timing: Timing,
    similarity: ContentSimilarity,
) -> Cohesion:
    """One row's cohesion score, checked against the three verdicts the row publishes.

    Five checks, all of them ways this file could lie about the one figure it exists to
    measure. The tally has to name classes the projection publishes, or the majority
    below it would be a majority of something this build has never heard of. It has to
    account for every post the row holds, or the candidate would be scored on a subset of
    its own content. It has to hold no class twice, or the counts beside it would add up
    to more posts than there are. The two name lists have to partition the corroborations
    between them — one name twice, one missing, one invented — because the names are what
    a reader weighs and a reader cannot weigh a list that does not add up. And the list
    has to be the one the clock, the vectors and the tally on this same row produce, which
    is the check the whole ticket rests on: a figure the evaluator publishes without being
    able to bear it out would be the claim the recovery report is built to avoid.
    """
    carried = record["cohesion"]
    if not isinstance(carried, dict):
        raise ValueError(f"{where} has cohesion={carried!r}, which is not a row")
    if set(carried) != set(COHESION_FIELDS):
        raise ValueError(
            f"{where} holds cohesion with {sorted(carried)}, and the cohesion vocabulary "
            f"is {sorted(COHESION_FIELDS)}"
        )
    cohesion = Cohesion(
        timing=timing,
        similarity=similarity,
        category=_tally(where, carried, posts),
    )
    named = read_names(where, carried, "corroborating")
    against = read_names(where, carried, "against")
    for name, field in ((named, "corroborating"), (against, "against")):
        unknown = sorted(set(name) - set(CORROBORATIONS))
        if unknown:
            raise ValueError(
                f"{where} names {unknown} in {field}, and the corroborations are "
                f"{', '.join(CORROBORATIONS)}"
            )
    if set(named) | set(against) != set(CORROBORATIONS):
        raise ValueError(
            f"{where} names {sorted(named)} as corroborated and {sorted(against)} as "
            f"against, which is not the {len(CORROBORATIONS)} corroborations between them"
        )
    if set(named) != set(cohesion.corroborating):
        raise ValueError(
            f"{where} names {sorted(named)} as corroborated and publishes "
            f"{sorted(cohesion.corroborating)} beside a tally of "
            f"{list(cohesion.category.tally)}"
        )
    retained = carried["retained"]
    if not isinstance(retained, bool):
        raise ValueError(f"{where} has retained={retained!r}, which is not a yes or a no")
    if retained is not cohesion.retained:
        raise ValueError(
            f"{where} says retained={retained} and names {len(named)} of the "
            f"{len(CORROBORATIONS)} corroborations, which retains a candidate on "
            f"{QUORUM}"
        )
    return cohesion


def _tally(where: str, record: JsonObject, posts: Sequence[str]) -> CategoryAgreement:
    """One row's category tally, checked as a tally of the row's own posts.

    The refusals are the ways this list could make the majority beside it a claim: a
    class the projection does not publish, a count that is not a count of postings, a
    class counted twice, and a list that does not add up to the posts the row holds. None
    of them is a figure a reader could check by counting, which is the whole of what this
    row is for. An empty list is refused rather than read as a candidate making no claim
    at all: a Campaign Candidate always holds posts, so a tally of none is a tally of
    something other than this row.
    """
    carried = record["category"]
    if not isinstance(carried, dict):
        raise ValueError(f"{where} has cohesion.category={carried!r}, which is not a row")
    vocabulary = tuple(field.name for field in fields(CategoryAgreement))
    if set(carried) != set(vocabulary):
        raise ValueError(
            f"{where} holds category with {sorted(carried)}, which is not the category "
            f"vocabulary {sorted(vocabulary)}"
        )
    entries = carried["tally"]
    if not isinstance(entries, list):
        raise ValueError(f"{where} has category.tally={entries!r}, which is not a list")
    if not entries:
        raise ValueError(
            f"{where} publishes no posts in any Scam Category and lists {len(posts)} "
            "posts, so its Scam Categories are about something other than this candidate"
        )
    known = {scam.name for scam in EVERY_CATEGORY}
    tally: list[tuple[str, int]] = []
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(
                f"{where} has a tally row {entry!r}, which is not a class and a count"
            )
        name, count = entry
        if name not in known:
            raise ValueError(
                f"{where} tallies {name!r}, which is a category this build does not publish: "
                f"the names are {', '.join(sorted(known))}"
            )
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(
                f"{where} tallies {name!r} at {count!r}, which is not a count of postings"
            )
        tally.append((name, count))
    refuse_repeated(
        f"{where} category tally",
        (name for name, _ in tally),
        "the counts beside it would then add up to more posts than the candidate holds",
    )
    if sum(count for _, count in tally) != len(posts):
        raise ValueError(
            f"{where} tallies {sum(count for _, count in tally)} postings and lists "
            f"{len(posts)} posts, so its Scam Categories are about something other than "
            "this candidate"
        )
    return CategoryAgreement(tally=in_print_order(tally))


def _seconds(where: str, record: JsonObject, field_name: str) -> int:
    """One duration in seconds, refused when it is anything else.

    A boolean passes for an integer in Python, so `true` would read as one second and
    corroborate everything; a negative duration would have to be explained by something
    this file does not print. Both are named rather than left to a reader of the file.
    """
    value = record[field_name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"{where} has {field_name}={value!r}, and a duration is a whole number of "
            "seconds"
        )
    return value


def _entries(where: str, record: JsonObject, field_name: str) -> list[object]:
    """One evidence list of a candidate row, checked as a list before it is walked.

    An empty list is allowed, because a candidate resting on one edge has an empty list
    for the other and the run prints that absence rather than treating it as a mistake.
    """
    entries = record[field_name]
    if not isinstance(entries, list):
        raise ValueError(f"{where} has {field_name}={entries!r}, which is not a list")
    return entries


def _shared(where: str, record: JsonObject, accounts: tuple[str, ...]) -> SharedDomain:
    """One registration a candidate is joined on, checked as a row of its own.

    The accounts named here have to be the candidate's own, because a registration
    under a candidate naming an account outside it would be evidence for a grouping
    nobody proposed. The file has no column saying so, so it is checked.
    """
    vocabulary = tuple(field.name for field in fields(SharedDomain))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds a shared registration with {sorted(record)}, which is not "
            f"the registration vocabulary {sorted(vocabulary)}"
        )
    named = read_names(where, record, "accounts")
    _named_by(where, named, accounts, "a registration")
    return SharedDomain(
        domain=read_text(where, record, "domain"),
        accounts=named,
        posts=read_names(where, record, "posts"),
    )


def _point(where: str, record: JsonObject, accounts: tuple[str, ...]) -> SharedContact:
    """One Contact Point a candidate is joined on, checked as a row of its own.

    Checked on the same terms as a registration, and for the same reason: the evidence
    printed under a candidate has to be evidence for that candidate, and the spelling
    list has to be there rather than dropped, because a fold a reader cannot see is
    indistinguishable from two accounts publishing the same thing twice.
    """
    vocabulary = tuple(field.name for field in fields(SharedContact))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds a shared Contact Point with {sorted(record)}, which is not "
            f"the Contact Point vocabulary {sorted(vocabulary)}"
        )
    named = read_names(where, record, "accounts")
    _named_by(where, named, accounts, "a Contact Point")
    return SharedContact(
        kind=read_vocabulary(where, "kind", record["kind"], ContactKind),
        value=read_text(where, record, "value"),
        accounts=named,
        posts=read_names(where, record, "posts"),
        spellings=read_names(where, record, "spellings"),
    )


def _named_by(
    where: str, named: tuple[str, ...], accounts: tuple[str, ...], what: str
) -> None:
    """The accounts named under one piece of evidence, against the candidate's own.

    One check for both kinds of evidence because the rule is the same and the file has
    no column saying so: evidence printed under a candidate has to be evidence for that
    candidate, or the reader is being shown a reason for a grouping that was not
    proposed.
    """
    outside = sorted(set(named) - set(accounts))
    if outside:
        raise ValueError(
            f"{where} names {what} reached by {outside}, which is not in the candidate "
            "it is printed under"
        )


def render_table(grouping: Grouping) -> str:
    """The console output: the figures, an index of the candidates, then the evidence.

    ASCII only, so it prints the same way on a console that cannot encode anything
    else and the same way when it is redirected, which is what lets it be pasted into
    an issue or diffed between runs. Sections are joined by one blank line, and an
    absent section leaves no gap behind it.
    """
    by_post = _posts_by_id(grouping.corpus)
    sections = (
        f"{_HEADING}\n\n{_SUBHEADING}",
        _figures(grouping.facts, grouping.filtered, grouping.temporal, grouping.content,
                 grouping.cohesion),
        _index(grouping.candidates),
        "\n\n".join(_block(candidate, by_post) for candidate in grouping.candidates),
        _uncorroborated(grouping.temporal),
        _deprioritised_by_similarity(grouping.content),
        _filtered_by_cohesion(grouping.cohesion),
        _withheld(grouping.filtered),
        _footer(grouping.filtered, grouping.temporal, grouping.content, grouping.cohesion),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _figures(
    facts: GroupingFacts,
    shared: SharedFilter,
    temporal: Corroboration,
    similarity: SimilarityCorroboration,
    cohesion: CohesionSystem,
) -> str:
    """What was read, and what came of it. One figure per line, labelled.

    The four account counts are a partition of the accounts, so a reader can add them
    up and get the number the second line prints. Each of the two edges gets its own
    count on the `grouped` line rather than being added into one number, because a
    candidate resting on one handle and a candidate resting on one domain are not the
    same claim and the reader is entitled to know which of them is which.

    The window and the threshold are lines of their own and the two counts are lines
    of their own, because a corroborated count without the threshold it was counted
    against is not a figure: `3 of 4` beside a threshold nobody can see is a score
    out of an unknown number.

    The last two lines are the two tiers ADR-0009 asks for, and they are adjacent on
    purpose: the baseline says what direct adjacency found and the line under it says
    what the cohesion system retained of it and how many it removed. Printing the second
    alone would leave a reader with a number and no way to tell a filtering decision from
    a bug, and printing the first alone would make the corroboration step an assertion.
    """
    fields = (
        ("corpus", facts.corpus_path),
        ("posts", _count(facts.posts, "post") + f" across {_count(facts.accounts, 'account')}"),
        ("sha256", facts.corpus_sha256),
        ("suffix list", facts.suffix_list_path),
        ("rules sha256", facts.rules_sha256),
        ("shared list", _shared_list(shared)),
        (
            "grouped",
            f"{_count(facts.accounts_in_candidates, 'account')} in "
            f"{_count(facts.candidates, 'candidate')}, on "
            f"{_count(facts.shared_domains, 'registration')} and "
            f"{_count(facts.shared_contact_points, 'Contact Point')}",
        ),
        ("alone", f"{_count(facts.accounts_alone, 'account')} reaching something nobody else reaches"),
        (
            "silenced",
            f"{_count(len(shared.silenced), 'account')} reaching nothing but known-shared "
            "infrastructure",
        ),
        ("unreachable", f"{_count(facts.accounts_unreachable, 'account')} reaching nothing this grouping can use"),
        ("filtered", _withheld_counts(shared)),
        ("window", f"{window_of(temporal.window_seconds)} between two accounts on one "
                   "piece of evidence"),
        ("timing", _timing_counts(temporal)),
        ("threshold", f"cosine distance at most {_distance(similarity.threshold)} between two posts in one candidate"),
        ("similarity", _similarity_counts(similarity)),
        ("baseline", _baseline_counts(cohesion)),
        ("cohesion", _cohesion_counts(cohesion)),
    )
    width = max(len(name) for name, _ in fields)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in fields)


def _baseline_counts(cohesion: CohesionSystem) -> str:
    """What direct adjacency found, named as such rather than as the result.

    The candidates and the accounts are `Grouping`'s own partition rather than this
    system's, and the two lines below and above this one are what make them checkable: the
    `grouped` line says how many candidates there are over the same Corpus, so a reader
    who adds them up is adding up two accounts of the same partition and not comparing
    two runs.
    """
    return (
        f"{_count(cohesion.baseline, 'candidate')} over "
        f"{_count(cohesion.accounts, 'account')}, by direct adjacency alone"
    )


def _cohesion_counts(cohesion: CohesionSystem) -> str:
    """What the cohesion system retained, and the difference between the two tiers.

    Both halves of the comparison rather than the retained count alone: a reader cannot
    see what corroboration bought from `3 of 4` without the four, and cannot see that it
    bought anything at all without being told how many were removed. The removed count is
    stated even when it is nothing, because a run that removed nothing and a run that
    never asked are two different things and the section below says which this was.
    """
    return (
        f"{count_of(cohesion.retained, cohesion.baseline, 'candidate')} and "
        f"{count_of(cohesion.retained_accounts, cohesion.accounts, 'account')} retained by the "
        f"cohesion system, {cohesion.removed} filtered"
    )


def _similarity_counts(similarity: SimilarityCorroboration) -> str:
    """What the stored vectors corroborated, and the rule that it removed nothing.

    The trailing clause is on every run rather than only on one where it mattered,
    because a reader looking for the candidates the vectors dropped and finding none
    has to be able to tell that from a store that was never asked. The candidates it
    did not corroborate are below the table under their own heading.
    """
    return (
        f"{count_of(similarity.corroborated, similarity.candidates, 'candidate')} and "
        f"{count_of(similarity.corroborated_pieces, similarity.pieces, 'piece')} of evidence "
        "corroborated; no candidate removed"
    )


def _distance(value: float) -> str:
    """The threshold as a figure a reader can compare two posts with."""
    return f"{value:.2f}"


def _timing_counts(temporal: Corroboration) -> str:
    """What the clock corroborated, and the rule that it removed nothing.

    The trailing clause is on every run rather than only on one where it mattered,
    because a reader looking for the candidates the clock dropped and finding none has
    to be able to tell that from a clock that was never asked. The candidates it did not
    corroborate are below the table under their own heading.
    """
    return (
        f"{count_of(temporal.corroborated, temporal.candidates, 'candidate')} and "
        f"{count_of(temporal.corroborated_pieces, temporal.pieces, 'piece')} of evidence "
        "corroborated; no candidate removed"
    )


def _withheld_counts(shared: SharedFilter) -> str:
    """What the filter took out, out of what there was, and what it cost.

    The Contact Points are counted in the same sentence as the registrations whenever the
    Corpus has any, because the filter reaches both edges and a figure naming only one
    of them would read as though it did not. A Corpus with no Contact Point at all says
    nothing rather than saying `0 of 0`.
    """
    parts = [count_of(len(shared.withheld), shared.registrations, "registration")]
    if shared.contact_points:
        parts.append(count_of(len(shared.withheld_points), shared.contact_points, "Contact Point"))
    return (
        " and ".join(parts)
        + " withheld, removing "
        f"{shared.components_removed} of {shared.components_before} components"
    )


def count_of(number: int, total: int, noun: str) -> str:
    """How many of how many, with the total agreeing with its noun.

    The total is the one that reads as a count of things in the Corpus, so it is the
    one that has to agree: "0 of 1 Contact Points" would leave a reader wondering
    whether the figure is wrong. Public because `rfi campaign-recovery` prints the same
    shape against its own nouns, and a figure a reader learns to read in one place
    should not read differently in the next.
    """
    return f"{number} of {_count(total, noun)}"


def _shared_list(shared: SharedFilter) -> str:
    """The list, how much of it there is, and when it was last updated.

    The date is the newest one any row carries rather than a date of its own, so the
    figure cannot drift from the file: change a row and the line moves with it.
    """
    published = shared.published
    hosts = _count(len(published.hosts), "host")
    updated = (
        f", last updated {published.last_updated}" if published.last_updated else ""
    )
    return f"{published.path} ({hosts}{updated})"


def _count(number: int, noun: str) -> str:
    """One count, agreeing with its noun. Every figure here is small, and a reader
    seeing "1 accounts" stops to wonder whether the figure is right."""
    return f"{number} {noun if number == 1 else f"{noun}s"}"


def _index(candidates: Sequence[CampaignCandidate]) -> str:
    """One line per candidate, so the output says how big the run was before the detail."""
    if not candidates:
        return (
            "No candidate: no two accounts in this Corpus share a registrable domain or "
            "a Contact Point."
        )

    headings = (
        "candidate",
        "accounts",
        "posts",
        "first seen",
        "justified by",
        "shared registrations",
        "timing",
        "similarity",
    )
    rows = [
        (
            candidate.candidate_id,
            str(len(candidate.accounts)),
            str(len(candidate.posts)),
            candidate.first_seen,
            candidate.evidence.value,
            ", ".join(shared.domain for shared in candidate.shared_domains),
            f"{candidate.timing.corroborated} of {candidate.timing.pieces} corroborated",
            f"{candidate.content_similarity.corroborated} of {candidate.content_similarity.pieces} corroborated",
        )
        for candidate in candidates
    ]
    widths = [max(len(cell) for cell in column) for column in zip(headings, *rows, strict=True)]
    return "\n".join([_row(headings, widths), *(_row(row, widths) for row in rows)])


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """One line of the index. A cell of nothing but digits is a count, and counts are
    right-aligned so the digits line up down the column; the rest is text of no fixed
    width and reads better against the left edge."""
    return "  ".join(
        cell.rjust(width) if cell.isdigit() else cell.ljust(width)
        for cell, width in zip(cells, widths, strict=True)
    )


def _block(candidate: CampaignCandidate, posts: dict[str, CorpusItem]) -> str:
    """One candidate and the evidence for it: why it exists, who, and on what.

    The label comes first and the evidence follows it, so the block reads in the order a
    reader has to judge it in: which kind of claim this is, then what the clock made of
    it, then what the vectors made of it, then what the three add up to, then the
    registrations, then the Contact Points, then the accounts and every post they wrote.
    The timing sits beside each piece of evidence rather than in one list at the end,
    because the claim it bears on is about that piece and nobody else.
    """
    timing = _gaps_by_value(candidate)
    window_seconds = candidate.timing.window_seconds
    lines = [
        f"{candidate.candidate_id}  {len(candidate.accounts)} accounts, "
        f"{len(candidate.posts)} posts, first seen {candidate.first_seen}",
        f"  justified by  {_justified(candidate)}",
        f"  timing        {_candidate_timing(candidate)}",
        f"  similarity    {_candidate_similarity(candidate)}",
        f"  cohesion      {cohesion_verdict(candidate)}",
        f"  categories    {_candidate_categories(candidate)}",
    ]
    if candidate.shared_domains:
        lines.append("  shared registrations")
        width = max(len(shared.domain) for shared in candidate.shared_domains)
        for shared in candidate.shared_domains:
            lines.append(
                f"    {shared.domain.ljust(width)}  {len(shared.accounts)} accounts, "
                f"{len(shared.posts)} posts  {_within(timing[shared.domain], window_seconds)}"
            )
    if candidate.shared_contact_points:
        lines.append("  shared contact points")
        width = max(len(point.value) for point in candidate.shared_contact_points)
        for point in candidate.shared_contact_points:
            spellings = ", ".join(point.spellings)
            lines.append(
                f"    {point.value.ljust(width)}  {point.kind.value}, "
                f"{_count(len(point.accounts), 'account')}, "
                f"{_count(len(point.posts), 'post')}  "
                f"{_within(timing[point.value], window_seconds)}  written {spellings}"
            )
    lines.append("  accounts")
    lines.extend(f"    {account}" for account in candidate.accounts)
    lines.append("  posts")
    for post_id in candidate.posts:
        item = posts[post_id]
        lines.append(f"    {post_id}  {item.account}  {item.created_at}  {item.subreddit}")
        lines.append(f"      {item.title}")
    return "\n".join(lines)


def _gaps_by_value(candidate: CampaignCandidate) -> dict[str, EvidenceTiming]:
    """The timing of each piece of evidence, keyed by what it names.

    Keyed by the value alone because the two edges name different things: a Contact
    Point value is a handle or an address, never a registration, so one key cannot hold
    two pieces and a lookup that found the wrong one would be a lookup that failed.
    """
    return {piece.value: piece for piece in candidate.corroboration}


def _candidate_timing(candidate: CampaignCandidate) -> str:
    """One candidate against the window, and the gap that decided it.

    The count of pieces rather than a yes or a no, because a candidate can be
    corroborated by one edge and not the other and both halves of that are findings: on
    this Corpus the registration puts three accounts in the same morning and the Contact
    Point brings in a fourth account two weeks later.
    """
    timing = candidate.timing
    pieces = "piece" if timing.pieces == 1 else "pieces"
    return (
        f"{timing.corroborated} of {timing.pieces} {pieces} of evidence within the "
        f"{_window_of(timing.window_seconds)} window; nearest pair "
        f"{gap_of(timing.closest_seconds)} apart"
    )


def _candidate_similarity(candidate: CampaignCandidate) -> str:
    """One candidate against the threshold, and the gap that decided it.

    The count of posts rather than a yes or a no, because a candidate can be
    corroborated by one Signal and not the other and both halves of that are
    findings: the registration is corroborated by the clock and its posts are a
    fortnight apart, but the texts in it are the same playbook read aloud, so
    the two figures together say more than either one alone.
    """
    summary = candidate.content_similarity
    pieces = "post" if summary.pieces == 1 else "posts"
    return (
        f"{summary.corroborated} of {summary.pieces} {pieces} within the "
        f"{_distance(summary.threshold)} cosine-distance threshold; nearest pair "
        f"{_distance(summary.closest_distance)} apart"
    )


def cohesion_verdict(candidate: CampaignCandidate) -> str:
    """One candidate's cohesion score, the names behind it, and the verdict it carries.

    Both halves, in this order, and never one without the other: the count says how much
    corroboration the candidate has and the names say which corroboration, so a reader who
    disagrees with the verdict has the evidence in front of them rather than a number to
    take on trust. `retained` and `filtered` are the run's own words rather than softer
    ones, because this is the one verdict in the module that removes a candidate from
    what the system proposes and a reader has to be able to see that it happened.

    The same line is printed under the candidate above, in the section below the table,
    and on the graph page `rfi campaign-graph` draws, so no reader meets two accounts of
    one decision in two vocabularies.
    """
    cohesion = candidate.cohesion
    verdict = "retained" if cohesion.retained else "filtered"
    return (
        f"{verdict} on {len(cohesion.corroborating)} of {len(CORROBORATIONS)}: "
        f"{names_of(cohesion.corroborating)}; against: {names_of(cohesion.against)}"
    )


def _candidate_categories(candidate: CampaignCandidate) -> str:
    """One candidate's Scam Category tally, and the class its posts agree on.

    The tally is printed whole and beside the count of placed postings, for the reason
    `Association` prints one registration's: a majority a reader cannot count is a
    majority they have to accept. `no majority` and `too few placed` are the words the
    registrations table uses for the same two outcomes, so a reader who has learned them
    there needs not learn them again here (ADR-0021, ADR-0028).
    """
    agreement = candidate.cohesion.category
    tally = ", ".join(f"{name} {count}" for name, count in agreement.tally)
    verdict = (
        f"agreeing on {agreement.scam_category}"
        if agreement.corroborates
        else agreement.why_not()
    )
    return (
        f"{verdict}, from {agreement.placed} of {_count(len(candidate.posts), 'post')} "
        f"placed: {tally}"
    )


def names_of(named: Sequence[str]) -> str:
    """A list of corroboration names, or what to print when there are none.

    `none` rather than an empty cell, because a blank after a colon reads as a figure the
    run forgot to print rather than as the claim that nothing corroborated it — which is
    exactly the candidate a reader most needs to be told about.
    """
    return ", ".join(named) if named else "none"


def window_of(window_seconds: int) -> str:
    """The window, in hours, because that is the unit `--window-hours` sets it in.

    Rounded down rather than to the nearest hour, so a figure that says "24 hours" is
    the window the run was given and not an hour either side of it.

    Public rather than private because `rfi campaign-graph` prints the same window on
    each page it draws, and two spellings of one published figure is how a reader ends up
    with two of them.
    """
    return f"{window_seconds // 3_600} hour" + ("" if window_seconds == 3_600 else "s")


def _window_of(window_seconds: int) -> str:
    """The same window as it reads in front of a noun: `24-hour`, never `24 hours`.

    Two spellings of one figure rather than one spelling used in two places, because
    "the 24 hours window" is the sort of thing a reader reads past without noticing and
    then quotes back at somebody.
    """
    return f"{window_seconds // 3_600}-hour"


def gap_of(seconds: int) -> str:
    """One gap, in the largest unit that still says something about it.

    `30m`, `4h30m`, `14d9h16m`, and a seconds part only where the gap is not a whole
    minute — a reader comparing this with the `created_at` printed above it has to be able
    to check it by eye, and that means no decimals, no unit they would have to look up, and
    no figure that is quietly rounded. A gap of nothing reads as the minute it was rather
    than as a zero, because two accounts posted in the same minute is the case the ticket
    is about and `0m apart` reads like a missing figure.

    Public rather than private for the reason `window_of` is: the graph pages print these
    gaps beside their evidence, and a page that rounded them where the table did not would
    be showing the same figure in two spellings.
    """
    if seconds == 0:
        return "the same minute"
    days, rest = divmod(seconds, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, rest = divmod(rest, 60)
    parts = [f"{days}d"] if days else []
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return "".join(parts)


def _justified(candidate: CampaignCandidate) -> str:
    """The evidence label, carrying the warning where a reader will meet the candidate.

    Only the Contact Point edge carries the recall warning, and a candidate resting on a
    registration as well has nothing to warn about. What the clock makes of the candidate
    is on the `timing` line two below this one rather than here: it is a second claim on
    the same candidate, and a line carrying two of them is a line where a reader has to
    work out which caveat belongs to which piece of evidence. A candidate naming no
    evidence at all carries the fourth label, which is what the reader allows and the
    grouping never writes.
    """
    if candidate.evidence is not Evidence.CONTACT:
        return candidate.evidence.value
    return f"{candidate.evidence.value}, {_WEAKER}"


def _uncorroborated(temporal: Corroboration) -> str:
    """Every candidate the window corroborates nothing under, named.

    Printed on every run, including one where there are none: a section that only appears
    when it is bad is a section a reader cannot tell from a missing one, and a candidate
    the clock had no time for and a clock never asked the question are two different
    things. Nothing is dropped — this is a deprioritising, so the candidate is in the
    table above with its span beside its evidence and named again here with the nearest
    pair of accounts, which is the one figure a reader needs to disagree with the window.
    """
    window = window_of(temporal.window_seconds)
    if not temporal.deprioritised:
        return (
            "uncorroborated  none: every candidate above holds its evidence's accounts "
            f"inside {window}"
        )
    lines = [
        f"uncorroborated  {count_of(len(temporal.deprioritised), temporal.candidates, 'candidate')} "
        f"deprioritised: no evidence under them spans less than {window}, and none of "
        "them is removed for it"
    ]
    for candidate in temporal.deprioritised:
        lines.append(
            f"  {candidate.candidate_id}  {len(candidate.accounts)} accounts, "
            f"{len(candidate.posts)} posts, nearest pair of accounts "
            f"{gap_of(candidate.timing.closest_seconds)} apart"
        )
    return "\n".join(lines)


def _deprioritised_by_similarity(similarity: SimilarityCorroboration) -> str:
    """Every candidate the threshold corroborates nothing under, named.

    Printed on every run, including one where there are none: a section that only
    appears when it is bad is a section a reader cannot tell from a missing one, and
    a candidate the vectors had nothing to say about and vectors never asked are two
    different things. Nothing is dropped - this is a deprioritising, so the candidate
    is in the table above with its nearest pair beside its posts and named again here
    with the gap a reader needs to disagree with the threshold.

    The heading says `filtered` because the Ticket says it, and the word is a
    deprecising here rather than a removing one: the sentence under it says that none
    of them is removed for it, and the code beside it is called `deprioritised`,
    because `filtered` already means removal in this module and a reader of the file
    should not have to work out which of the two it is looking at.
    """
    threshold = _distance(similarity.threshold)
    if not similarity.deprioritised:
        return (
            f"filtered by similarity  none: every candidate above holds a near-twin for "
            f"every post within {threshold}"
        )
    lines = [
        f"filtered by similarity  {count_of(len(similarity.deprioritised), similarity.candidates, 'candidate')} "
        f"deprioritised: in none of them does every post have a near-twin by another "
        f"account within {threshold}, and none of them is removed for it"
    ]
    for candidate in similarity.deprioritised:
        lines.append(
            f"  {candidate.candidate_id}  {len(candidate.accounts)} accounts, "
            f"{len(candidate.posts)} posts, nearest pair of posts "
            f"{_distance(candidate.content_similarity.closest_distance)} apart"
        )
    return "\n".join(lines)


def _within(piece: EvidenceTiming, window_seconds: int) -> str:
    """One piece of evidence against the window, in the block under its own name.

    The span leads because that is the figure the verdict is about, and the window is
    named rather than referred to: `inside the window` in a block with four evidence rows
    on it does not say which window.
    """
    return (
        f"{gap_of(piece.span_seconds)} across, "
        + ("inside" if piece.within(window_seconds) else "outside")
        + f" the {_window_of(window_seconds)} window"
    )


def _filtered_by_cohesion(cohesion: CohesionSystem) -> str:
    """Every candidate the cohesion system removed, named with what removed it.

    Printed on every run, including one where there are none, for the reason the two
    sections above it are: a section that appears only when it is bad is a section a
    reader cannot tell from a missing one. This one says `filtered` rather than
    `deprioritised`, because unlike them it removes — that is the difference ADR-0009
    asks for and the reason the baseline is published beside it rather than replaced by
    it, so a candidate listed here is still in the file and still in the table above with
    its own evidence under it.

    Each line is the same sentence the block above the table printed for that candidate,
    so the two views cannot be read as two accounts of one decision.
    """
    total = len(CORROBORATIONS)
    if not cohesion.filtered:
        return (
            f"filtered by cohesion  none: every candidate above holds at least {QUORUM} "
            f"of the {total}"
        )
    lines = [
        f"filtered by cohesion  {count_of(len(cohesion.filtered), cohesion.baseline, 'candidate')} "
        f"removed: fewer than {QUORUM} of the {total} corroborations under "
        f"{_plural(len(cohesion.filtered), 'it', 'them')} hold"
    ]
    for candidate in cohesion.filtered:
        lines.append(
            f"  {candidate.candidate_id}  {_count(len(candidate.accounts), 'account')}, "
            f"{_count(len(candidate.posts), 'post')}  {cohesion_verdict(candidate)}"
        )
    return "\n".join(lines)


def _plural(number: int, one: str, many: str) -> str:
    """The word that agrees with a count: one candidate holds, none hold."""
    return one if number == 1 else many


def _withheld(shared: SharedFilter) -> str:
    """Everything the filter kept out of the graph, on either edge, one block each.

    Two blocks rather than one table because the two kinds are decided differently and a
    reader comparing them needs to know which is which: a registration is withheld by the
    published list, and a Contact Point is withheld because the registration it sits under
    is.
    """
    return "\n\n".join(
        section for section in (_withheld_hosts(shared), _withheld_points(shared)) if section
    )


def _withheld_hosts(shared: SharedFilter) -> str:
    """The registrations the filter kept out of the graph, and what each one reached.

    Printed because a registration the resolved links hold and no candidate names is
    otherwise unexplained: a reader would find a shortener's registration in
    `post-domains.jsonl` and have no way to tell a decision from an omission.
    Withholding is a judgment about a service rather than about a post, so each row
    says which kind of service it is as well as how many accounts reached it.

    The heading claims only what is true of every run: no candidate is joined on one of
    these, whatever else the accounts it reached went on to do. It does not claim the
    accounts are in no candidate, because an account can reach a shortener and share a
    registration or a handle with another account, and that pairing survives the filter
    untouched.
    """
    if not shared.withheld:
        return ""

    names = max(len(item.registration) for item in shared.withheld)
    kinds = max(len(kind) for item in shared.withheld for kind in _kinds(item))
    lines = [
        f"withheld  {_count(len(shared.withheld), 'registration')}, reached by "
        f"{_count(shared.accounts_reaching_withheld, 'account')}; no candidate is "
        "joined on one of them"
    ]
    for item in shared.withheld:
        lines.append(
            f"  {item.registration.ljust(names)}  {', '.join(_kinds(item)).ljust(kinds)}  "
            f"{_count(item.accounts, 'account')}"
        )
    return "\n".join(lines)


def _withheld_points(shared: SharedFilter) -> str:
    """The Contact Points that left the graph with the registration under them.

    Every one of them, whether one account published it or several, because the figure
    above counts the whole set: a point filtered out of the graph and left out of this
    list is a decision the output does not account for. Named for the same reason the
    registrations are named — an address a reader can find in the published Contact
    Points and in no candidate has to be a decision rather than a gap.
    """
    if not shared.withheld_points:
        return ""

    names = max(len(point.value) for point in shared.withheld_points)
    lines = [
        "withheld Contact Points at those registrations, as published in the Corpus; "
        "no candidate is joined on one of them either"
    ]
    for point in shared.withheld_points:
        lines.append(
            f"  {point.value.ljust(names)}  {point.kind.value} at "
            f"{point.registration}  {_count(len(point.accounts), 'account')}"
        )
    return "\n".join(lines)


def _kinds(item: WithheldRegistration) -> tuple[str, ...]:
    """The kinds of service that publish one registration, distinct and in host order."""
    return tuple(dict.fromkeys(host.kind.value.replace("_", " ") for host in item.hosts))


def _footer(
    shared: SharedFilter,
    temporal: Corroboration,
    similarity: SimilarityCorroboration,
    cohesion: CohesionSystem,
) -> str:
    """What the run can and cannot claim, and where the filter comes from.

    The list is named rather than described, because the claim that the junk is gone
    is only worth as much as the data behind it, and a reader who wants to disagree
    with the filter needs the file rather than a promise.
    """
    return f"""\
Every candidate above rests on a shared registrable domain, a shared Contact Point, or
both, and on nothing else. Timing and content similarity are read as corroboration
and neither could produce a candidate on its own, which is what ADR-0005 and
ADR-0026 require.

The two tiers are published as a pair because ADR-0009 asks for both and neither alone
(ADR-0028). The baseline is what direct adjacency found: {_count(cohesion.baseline, 'candidate')}
over {_count(cohesion.accounts, 'account')}, decided by nothing but the links and the Contact
Points. The cohesion system is what it retained of that, {_count(cohesion.retained, 'candidate')}
over {_count(cohesion.retained_accounts, 'account')}, once temporal proximity, content similarity
and agreement on Scam Category have each had their say. A candidate is retained on
{QUORUM} of the {len(CORROBORATIONS)}, and every candidate says which held and which went against
it. Nothing here is published as the result on its own, because a reader shown the lower
number with nothing beside it cannot tell a filtering decision from a bug, and one shown
the higher number cannot tell what corroboration bought. This run removes
{cohesion.removed} of {cohesion.baseline}, each named below the table with the evidence that
removed it.

The first removal this module makes for any reason but known-shared infrastructure is
this one. The clock and the vectors may order and deprioritise, and the categories join
them in that, but none of them may create a candidate at any window or any threshold; the
cohesion system filters candidates that shared infrastructure already proposed, and it
publishes each one it removes rather than deleting it. The baseline is left in the file
exactly as it was, with its identifiers where they were, so the pre-corroboration number
is a record rather than something a reader has to reconstruct.

Timing corroborates a grouping and never creates one. Two accounts that post in the same
minute and share nothing are not grouped, and this run would not group them whatever the
window is set to. What the window does is order this list and say so: a candidate some
piece of evidence under it holds two accounts inside the window is printed ahead of every
candidate it does not, and one it corroborates nothing under is printed in full with the
gap beside its evidence and named again below the table. Nothing is dropped for want of
timing. A domain two accounts reached months apart may be a domain that changed hands and
a campaign may simply be a patient one, so the gap is published as a figure a reviewer can
weigh rather than acted on as a decision, and the window that judged it is named above.

Content similarity corroborates a grouping and never creates one either (ADR-0026). Two
accounts that paste the same advert, hour for hour, and share no registrable domain and
no Contact Point are not grouped, whatever the threshold is set to: the Corpus's own copy
of that case is the decoy cluster, whose text is the nearest on the Corpus and whose
accounts reach only a link shortener between them. What the threshold does is order this
list and say so, the same way the window does. The reason content similarity can never
establish a grouping is written here because it is the question every reader will ask:
a scam template converges across unrelated operators, and near-identical text is what
such a convergence looks like when it crosses. An advert that two unrelated desks both
paste is evidence about the template, not about the operator behind it, and an account
sharing nothing but that text cannot be told from the unrelated operator two hours and
a different country away. The model reads words and not meaning, which the vectors' own
footer states: two posts making the same pitch in different words come out distant, as
this Corpus's own Planted Campaign does, and a signal that corroborates nothing would be
no signal at all. So the threshold is stated beside every figure it was counted against,
the candidates it corroborates nothing under are named with the nearest distance rather
than discarded, and the only candidate-level verdict it can carry is corroboration.

Agreement on Scam Category corroborates a grouping and never establishes one either. A
candidate whose posts all landed in one class is a group of accounts making the same
pitch, which is a stronger claim than shared infrastructure and a weaker one than
identity: there are only ten classes, two desks running two different pitches share a
class by coincidence, and the placing itself is a phrase list read off a post's own words
(ADR-0017), so a candidate of complaints about one desk agrees with the desk it is
complaining about. The tally is printed beside every candidate for the same reason the
distances are: a class a reader cannot count is a class they have to accept. `Other` is
never agreed on, because a post no list matched makes no claim at all, and a component
whose posts split evenly is reported as `no majority` rather than settled by a tie-break.

The two edges are not equally good, and each candidate says which one it rests on. A
registration is somebody's property and this project resolves it against the published
Public Suffix List. A Contact Point is read out of a post by a rule with a measured
recall against the Labelled Set: of the Contact Points this Corpus publishes, some are
written out in disguise and one is only a picture of one, and this reading finds the
plain ones and some of the rest. A candidate built on a Contact Point alone therefore rests on
the weaker of the two, and a campaign that rotates both its registrations and its Contact
Points cannot be recovered here at all. The count of campaigns in that position is the
recall bound, and `rfi campaign-recovery` reports it beside the recovery figure rather
than leaving it in the tickets (ADR-0023).

Known-shared infrastructure was filtered out of the graph before the grouping rather
than out of the result afterwards, so a link shortener, a paste site, or a link-in-bio
page joins nothing at all: the registrations they name are withheld whole, an address at
one of them is withheld with it, and no withheld registration can bridge two accounts
either. The list is published as data rather than written into this query, which is what
lets a domain-reputation feed replace it (ADR-0009):

    {shared.published.path}

What that costs is stated rather than hidden. A Planted Campaign that leans on a shared
host is lost with the host, and an account that reaches nothing else cannot be proposed
at all, so a recall figure measured this way is a lower bound.

Whether the accounts in one candidate belong together is a judgment for a
reviewer. This output does not make it."""


def _posts_by_id(corpus: Sequence[CorpusItem]) -> dict[str, CorpusItem]:
    return {item.post_id: item for item in corpus}
