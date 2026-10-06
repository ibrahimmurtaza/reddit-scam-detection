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

What this cannot reach is stated here rather than left in the tickets. A campaign that
rotates both its registrations and its Contact Points shares nothing with itself, so no
grouping resting on either edge can propose it; the count of campaigns in that position
is the method's recall bound, and the evaluator reports it beside the recovery figure
(ADR-0023).

Nothing here reads the truth file. The Corpus, the published Public Suffix List, and the
shared-infrastructure list are the whole input, and the truth file is joined by the
evaluator after this has finished (ADR-0008).
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
    def kind(self) -> EvidenceKind | None:
        """The one edge this label names, and `None` for the two that name none or both.

        `NONE` and `BOTH` have no single edge and say so rather than picking one, which is
        the same reason they exist: a timing row always names exactly one kind, so a
        candidate resting on both edges cannot be rendered as one.
        """
        if self is Evidence.DOMAIN:
            return EvidenceKind.REGISTRATION
        if self is Evidence.CONTACT:
            return EvidenceKind.CONTACT_POINT
        return None

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
class CampaignCandidate:
    """A proposed grouping of accounts, and everything a reader needs to check it.

    `accounts` is what the run proposes; `shared_domains` and `shared_contact_points`
    are why, and `evidence` says which of the two the candidate rests on. `first_seen`
    is the earliest post by any account in the component, which is the only date in the
    Corpus that says anything about when the group was active. `corroboration` and
    `timing` are what the clock adds, which is corroboration and never the reason the
    candidate exists (ADR-0025).
    """

    candidate_id: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]
    shared_domains: tuple[SharedDomain, ...]
    shared_contact_points: tuple[SharedContact, ...]
    corroboration: tuple[EvidenceTiming, ...]
    timing: Timing
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
class Grouping:
    """Everything one run establishes, so the table and the file cannot disagree.

    The Corpus is carried because the printed table quotes its posts: the claims are
    about content in that file, and the evidence is read from it rather than
    transcribed.
    """

    facts: GroupingFacts
    filtered: SharedFilter
    temporal: Corroboration
    candidates: tuple[CampaignCandidate, ...]
    corpus: tuple[CorpusItem, ...] = field(repr=False)


def group(
    corpus_path: Path,
    list_path: Path,
    shared_path: Path,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> Grouping:
    """Read the three files and return the candidates, the figures, and the Corpus.

    One call, because the figures, the candidates, and the printed evidence are three
    views of one pass over the Corpus. A caller that resolved the links and then asked
    for the candidates separately would do the work twice and could end up printing
    one run's figures over another's results.

    The rows are grouped twice: once with the known-shared registrations withheld,
    which is the result, and once with nothing withheld, which is the adjacency
    baseline ADR-0009 asks for and the thing the filter is measured against. Both
    come from the same rows and the same union-find, so the difference between the two
    is the filter and nothing else.

    The resolved links and the Contact Points are arguments rather than something read
    here. They are produced by the same `post_domains` and the same `contacts.extract`
    that the earlier commands publish, so the two cannot disagree unless those
    functions are wrong — which their own tests hold them to. The published files are
    not read: the grouping is a function of the Corpus alone, and re-reading a derived
    file would make the result depend on whether somebody had remembered to run the
    step before it.

    Candidates come out largest first, then by the accounts' names, so a fixed Corpus
    produces a fixed list under fixed identifiers — with every candidate the timing
    corroborates ahead of every candidate it does not, because that is the whole of
    what timing is allowed to do here (ADR-0005). The identifier says where a
    candidate sits in the output and nothing else: it is not an identity that survives
    the Corpus changing underneath it.
    """
    suffixes = PublicSuffixes.read(list_path)
    corpus = read_corpus(corpus_path)
    shared = SharedInfrastructure.read(shared_path, suffixes)
    rows = post_domains(corpus, suffixes)
    points = shared_contact_points(read_contact_points(corpus))
    created_at = {item.post_id: item.created_at for item in corpus}

    window_seconds = _window_seconds(window_hours)
    withheld = shared.withheld()
    kept = _Reach.of(rows, points, created_at, withheld)
    unfiltered = _Reach.of(rows, points, created_at, frozenset())
    candidates = tuple(
        kept.candidate(f"cc-{number:02d}", accounts, window_seconds)
        for number, accounts in enumerate(kept.ordered(window_seconds), start=1)
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
    """

    accounts_by_domain: dict[str, set[str]]
    posts_by_domain: dict[str, set[str]]
    points: Mapping[tuple[ContactKind, str], SharedContact]
    posts_by_account: dict[str, set[str]]
    first_seen: dict[str, str]
    moments: dict[str, tuple[datetime, ...]]

    @classmethod
    def of(
        cls,
        rows: Iterable[PostDomains],
        points: Mapping[tuple[ContactKind, str], SharedContact],
        created_at: dict[str, str],
        withheld: frozenset[str],
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
        reach = cls({}, {}, {}, {}, {}, {})
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

    def ordered(self, window_seconds: int) -> list[frozenset[str]]:
        """The components, the corroborated ones first, then the usual order.

        The only thing the clock is allowed to do to the output. Corroborated ahead of
        uncorroborated is a deprioritising and not a filter: both are reported, and
        which is which is under every candidate's name. Sorting on it at all is the
        difference between the ordering being the run's opinion and the clock being a
        column of a table somebody can ignore.
        """
        return sorted(
            self.components(),
            key=lambda accounts: (
                not self.timing(accounts, window_seconds).corroborates,
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
        self, candidate_id: str, accounts: frozenset[str], window_seconds: int
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
        return CampaignCandidate(
            candidate_id=candidate_id,
            accounts=tuple(sorted(accounts)),
            posts=self.posts_of(accounts),
            shared_domains=joined.domains,
            shared_contact_points=joined.points,
            corroboration=joined.timing,
            timing=_summary(joined.timing, window_seconds),
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
    return CampaignCandidate(
        candidate_id=read_text(where, record, "candidate_id"),
        accounts=accounts,
        posts=read_names(where, record, "posts"),
        shared_domains=tuple(
            _shared(where, entry, accounts) for entry in shared if isinstance(entry, dict)
        ),
        shared_contact_points=tuple(
            _point(where, entry, accounts) for entry in points if isinstance(entry, dict)
        ),
        corroboration=pieces,
        timing=_timing(where, record, names, pieces),
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
    kind = label.kind
    assert kind is not None, f"{label} names no single edge"
    if field not in record:
        raise ValueError(
            f"{where} holds {label.value} with {sorted(record)}, and it names no {field}"
        )
    return (kind.value, read_text(where, record, field))


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
        _figures(grouping.facts, grouping.filtered, grouping.temporal),
        _index(grouping.candidates),
        "\n\n".join(_block(candidate, by_post) for candidate in grouping.candidates),
        _uncorroborated(grouping.temporal),
        _withheld(grouping.filtered),
        _footer(grouping.filtered, grouping.temporal),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _figures(facts: GroupingFacts, shared: SharedFilter, temporal: Corroboration) -> str:
    """What was read, and what came of it. One figure per line, labelled.

    The four account counts are a partition of the accounts, so a reader can add them
    up and get the number the second line prints. Each of the two edges gets its own
    count on the `grouped` line rather than being added into one number, because a
    candidate resting on one handle and a candidate resting on one domain are not the
    same claim and the reader is entitled to know which of them is which.

    The window is a line of its own and the two timing counts are a line of their own,
    because a corroborated count without the threshold it was counted against is not a
    figure: `3 of 4` beside a window nobody can see is a score out of an unknown number.
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
        ("window", f"{_window(temporal.window_seconds)} between two accounts on one piece of evidence"),
        ("timing", _timing_counts(temporal)),
    )
    width = max(len(name) for name, _ in fields)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in fields)


def _timing_counts(temporal: Corroboration) -> str:
    """What the clock corroborated, and the rule that it removed nothing.

    The trailing clause is on every run rather than only on one where it mattered,
    because a reader looking for the candidates the clock dropped and finding none has
    to be able to tell that from a clock that was never asked. The candidates it did not
    corroborate are below the table under their own heading.
    """
    return (
        f"{_of(temporal.corroborated, temporal.candidates, 'candidate')} and "
        f"{_of(temporal.corroborated_pieces, temporal.pieces, 'piece')} of evidence "
        "corroborated; no candidate removed"
    )


def _withheld_counts(shared: SharedFilter) -> str:
    """What the filter took out, out of what there was, and what it cost.

    The Contact Points are counted in the same sentence as the registrations whenever the
    Corpus has any, because the filter reaches both edges and a figure naming only one
    of them would read as though it did not. A Corpus with no Contact Point at all says
    nothing rather than saying `0 of 0`.
    """
    parts = [_of(len(shared.withheld), shared.registrations, "registration")]
    if shared.contact_points:
        parts.append(_of(len(shared.withheld_points), shared.contact_points, "Contact Point"))
    return (
        " and ".join(parts)
        + " withheld, removing "
        f"{shared.components_removed} of {shared.components_before} components"
    )


def _of(number: int, total: int, noun: str) -> str:
    """How many of how many, with the total agreeing with its noun.

    The total is the one that reads as a count of things in the Corpus, so it is the
    one that has to agree: "0 of 1 Contact Points" would leave a reader wondering
    whether the figure is wrong.
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
    it, then the registrations, then the Contact Points, then the accounts and every post
    they wrote. The timing sits beside each piece of evidence rather than in one list at
    the end, because the claim it bears on is about that piece and nobody else.
    """
    timing = _gaps_by_value(candidate)
    window_seconds = candidate.timing.window_seconds
    lines = [
        f"{candidate.candidate_id}  {len(candidate.accounts)} accounts, "
        f"{len(candidate.posts)} posts, first seen {candidate.first_seen}",
        f"  justified by  {_justified(candidate)}",
        f"  timing        {_candidate_timing(candidate)}",
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
        f"{_gap(timing.closest_seconds)} apart"
    )


def _within(piece: EvidenceTiming, window_seconds: int) -> str:
    """One piece of evidence against the window, in the block under its own name.

    The span leads because that is the figure the verdict is about, and the window is
    named rather than referred to: `inside the window` in a block with four evidence rows
    on it does not say which window.
    """
    return (
        f"{_gap(piece.span_seconds)} across, "
        + ("inside" if piece.within(window_seconds) else "outside")
        + f" the {_window_of(window_seconds)} window"
    )


def _window(window_seconds: int) -> str:
    """The window, in hours, because that is the unit `--window-hours` sets it in.

    Rounded down rather than to the nearest hour, so a figure that says "24 hours" is
    the window the run was given and not an hour either side of it.
    """
    return f"{window_seconds // 3_600} hour" + ("" if window_seconds == 3_600 else "s")


def _window_of(window_seconds: int) -> str:
    """The same window as it reads in front of a noun: `24-hour`, never `24 hours`.

    Two spellings of one figure rather than one spelling used in two places, because
    "the 24 hours window" is the sort of thing a reader reads past without noticing and
    then quotes back at somebody.
    """
    return f"{window_seconds // 3_600}-hour"


def _gap(seconds: int) -> str:
    """One gap, in the largest unit that still says something about it.

    `30m`, `4h30m`, `14d9h16m`, and a seconds part only where the gap is not a whole
    minute — a reader comparing this with the `created_at` printed above it has to be able
    to check it by eye, and that means no decimals, no unit they would have to look up, and
    no figure that is quietly rounded. A gap of nothing reads as the minute it was rather
    than as a zero, because two accounts posted in the same minute is the case the ticket
    is about and `0m apart` reads like a missing figure.
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
    window = _window(temporal.window_seconds)
    if not temporal.deprioritised:
        return (
            "uncorroborated  none: every candidate above holds its evidence's accounts "
            f"inside {window}"
        )
    lines = [
        f"uncorroborated  {_of(len(temporal.deprioritised), temporal.candidates, 'candidate')} "
        f"deprioritised: no evidence under them spans less than {window}, and none of "
        "them is removed for it"
    ]
    for candidate in temporal.deprioritised:
        lines.append(
            f"  {candidate.candidate_id}  {len(candidate.accounts)} accounts, "
            f"{len(candidate.posts)} posts, nearest pair of accounts "
            f"{_gap(candidate.timing.closest_seconds)} apart"
        )
    return "\n".join(lines)


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


def _footer(shared: SharedFilter, temporal: Corroboration) -> str:
    """What the run can and cannot claim, and where the filter comes from.

    The list is named rather than described, because the claim that the junk is gone
    is only worth as much as the data behind it, and a reader who wants to disagree
    with the filter needs the file rather than a promise.
    """
    return f"""\
Every candidate above rests on a shared registrable domain, a shared Contact Point, or
both, and on nothing else. Timing is read as corroboration and content similarity is not
read at all, and neither could produce a candidate on its own, which is what ADR-0005
requires.

Timing corroborates a grouping and never creates one. Two accounts that post in the same
minute and share nothing are not grouped, and this run would not group them whatever the
window is set to. What the window does is order this list and say so: a candidate some
piece of evidence under it holds two accounts inside the window is printed ahead of every
candidate it does not, and one it corroborates nothing under is printed in full with the
gap beside its evidence and named again below the table. Nothing is dropped for want of
timing. A domain two accounts reached months apart may be a domain that changed hands and
a campaign may simply be a patient one, so the gap is published as a figure a reviewer can
weigh rather than acted on as a decision, and the window that judged it is named above.

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
