"""Recovery of Planted Campaigns: X of N, joined after inference has finished.

This is the project's one substantive claim, so the join is deliberately dull and
deliberately separate. A Campaign Candidate counts as a recovery only when it holds a
Planted Campaign's whole membership: two accounts of a three-account campaign grouped on
their shared registration is a real grouping and not a recovery, and a candidate holding
a campaign and one account of its own is an over-grouping rather than half a recovery.
Anything short of the whole membership is reported beside the figure with the accounts
held, missing, and unexpected named, so `X of N` is a number a reader can take apart
rather than take on trust (ADR-0004).

The two steps are separate commands with a committed file between them. This one reads
the Corpus, the Planted Campaign membership, the Nuisance Structure manifest,
`data/campaigns/campaign-candidates.jsonl`, which `rfi campaign-candidates` wrote, and
`data/signals/policy-scores.jsonl`, which `rfi policy-score` wrote; it never imports,
calls, or re-runs the grouping, and nothing in the grouping path opens the membership
or the manifest. The boundary is the file, so a reader can check it: run the grouping
over the Corpus and grep for the membership, and run this command and watch it open the
file the grouping published (ADR-0018).

The report names the Nuisance Structure the figure was measured against, because X of N
over a Corpus that held nothing else would be a figure about a generator that planted two
campaigns and said so nowhere. The rate at which the grouping is wrong is measured here
too, against the same manifest: recovery and false-grouping are reported together, and
neither appears alone. The Review Queue's precision at several depths is measured here as
well, because a recovery number without the queue's hit rate beside it is a claim without
its shape. All three are measured here rather than in a second evaluator command, so the
membership has one reader (ADR-0022).

It also says what the figure is not. Every label in this Corpus was assigned by the
generator that wrote the posts, so a share of the posts called right or wrong would
measure agreement with the generator rather than anything about fraud, and no such figure
is published here in any view. Nobody reviewed any of it: a Campaign Candidate is a
proposal, and what it is scored against is planted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from reddit_fraud_intelligence.campaigns import (
    CampaignCandidate,
    Evidence,
    read_campaign_candidates,
)
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.jsonl import JsonObject, write_lines
from reddit_fraud_intelligence.nuisance import (
    HardNegativeCharacter,
    NuisanceKind,
    NuisanceRecord,
    read_nuisance,
)
from reddit_fraud_intelligence.review_queue import QueueEntry, build_queue
from reddit_fraud_intelligence.truth import PlantedCampaign, read_truth
from reddit_fraud_intelligence.text import wrap

_HEADING = "Campaign recovery"
_SUBHEADING = """\
How many Planted Campaigns the grouping recovered, and what it did with the ones it did
not. A candidate counts as a recovery only when it holds a campaign's whole membership,
and anything less is reported beside the figure rather than counted into it."""

# Where the join is published, as the command writes it by default. Named here rather
# than read out of the run, because the report points at the file a reader should open
# beside it, and a report naming the directory this particular run wrote into would
# produce different bytes for the same Corpus. `cli.py` holds the default the command
# uses; this is the name the report gives it, which is what `composition.py` does with
# the placements beside `docs/corpus-composition.md`.
RECOVERY_PATH = "data/evaluation/recovery.jsonl"

# The depths the Review Queue's precision is measured at by default. Named once, in
# the evaluator that prints them, because a figure carrying a depth and a default
# that named another one would be a quiet disagreement between the command and the
# report.
DEFAULT_DEPTHS: tuple[int, ...] = (5, 10, 20, 50)

# The widest kind name in the enum, so the nuisance table's labels line up whatever a
# future kind is called rather than reflowing under it.
_KIND_WIDTH = max(len(kind.value) for kind in NuisanceKind)


class Outcome(StrEnum):
    """What the run did with one Planted Campaign's membership.

    Three outcomes and no fourth, because they partition the campaigns: a candidate
    holding the whole membership, a candidate holding part of it, and no candidate at
    all. A campaign that cannot be in one of the three would be a figure a reader could
    not take apart.
    """

    RECOVERED = "recovered"
    PARTIAL = "partial"
    MISSED = "missed"


@dataclass(frozen=True, slots=True)
class Match:
    """One Campaign Candidate set against one Planted Campaign's membership.

    The three name lists are the join spelled out, so the outcome is checkable from the
    row: `held` is what the candidate has of the membership, `missing` is what it does
    not, and `extra` is what it holds that the membership does not name. A candidate
    whose `missing` and `extra` are both empty is the whole membership and only that.
    """

    candidate_id: str
    accounts: tuple[str, ...]
    held: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def whole(self) -> bool:
        """Whether this candidate holds the membership and nothing else.

        Asked of the match rather than worked out again by the callers, because the
        figure, the console, and the report all need the same answer and three
        definitions of it would be three chances to disagree.
        """
        return not self.missing and not self.extra


@dataclass(frozen=True, slots=True)
class CampaignRecovery:
    """One Planted Campaign, its membership, and every candidate that reached it.

    `matches` holds a candidate that names at least one account of the membership, in
    candidate order, and is empty exactly when the campaign is missed. The outcome is
    worked out here rather than passed in, so the file, the table, and the figure cannot
    disagree about which of the three a campaign is.

    `within` is what the membership shares with itself: every shared registration and
    every shared Contact Point that two or more of the campaign's own accounts reach.
    It is empty exactly when the campaign is in the recall bound, and it is what makes
    that figure checkable rather than asserted — a reader can look at the memberships
    and see for themselves which of them share anything.
    """

    campaign_id: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]
    matches: tuple[Match, ...]
    within: tuple[tuple[Evidence, str], ...] = ()  # each as (which edge, what it names)

    @property
    def outcome(self) -> Outcome:
        """Recovered on whole membership alone.

        A candidate holding every account and one more is not a recovery: the membership
        is what was planted, and a grouping that reached past it is grouping wrongly, so
        it is reported as the partial it is rather than counted into the figure.
        """
        if any(match.whole for match in self.matches):
            return Outcome.RECOVERED
        if self.matches:
            return Outcome.PARTIAL
        return Outcome.MISSED

    @property
    def partial_matches(self) -> tuple[Match, ...]:
        """The candidates holding only part of this membership, however the outcome came
        out.

        Asked of every campaign rather than only the partial ones, because a campaign
        recovered whole by one candidate can still have a second candidate holding two
        of its three accounts. That match is a real grouping of real accounts and it is
        not part of the recovery, so the views that report it have to report it whatever
        the campaign's own outcome was.
        """
        return tuple(match for match in self.matches if not match.whole)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(match.candidate_id for match in self.matches)

    @property
    def bounded(self) -> bool:
        """Whether no edge this method rests on reaches inside this membership.

        A campaign two of whose accounts share nothing that reaches a candidate has no
        edge within its own membership, so no candidate can hold two of them without
        also holding an account from outside it — which is not a recovery. Asked of the
        membership rather than of the outcome, because a campaign can be `recovered` and
        still be outside the method's reach on a Corpus where the memberships overlap,
        and conflating the two questions would report a bound as a result.
        """
        return not self.within

    def shared_within(self) -> str:
        """What the membership shares that reaches a candidate, grouped by which edge.

        Grouped rather than one edge per item, because the enum's names are plural —
        they are the labels the candidates table prints — and "Contact Points
        syn_vantageledger" beside "registrations vantage-ledger.example" is a list of two
        headings rather than a claim about two things.
        """
        return "; ".join(
            f"{edge.value}: {', '.join(names)}"
            for edge, names in self.shared_by_edge()
            if names
        )

    def shared_by_edge(self) -> tuple[tuple[Evidence, tuple[str, ...]], ...]:
        """The names this membership shares that reach a candidate, per edge.

        In `Evidence`'s own order, so the two views print the same list in the same
        sequence without either of them sorting a vocabulary the other does not share.
        """
        return tuple(
            (
                edge,
                tuple(name for other, name in self.within if other is edge),
            )
            for edge in Evidence
        )


@dataclass(frozen=True, slots=True)
class Unmatched:
    """A Campaign Candidate that names no account of any Planted Campaign.

    Printed rather than counted against N. The Corpus plants one — a shop's three
    accounts on one domain — and ADR-0005 says they group, so the grouping is right to
    produce them and calling that a miss would report correct behaviour as a failure.
    The evidence is carried rather than only printed, because a candidate with no campaign
    behind it is only diagnosable if the reader can see what it was joined on, and
    `joined_on` renders both edges through the grouping's own vocabulary.
    """

    candidate_id: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]
    evidence: Evidence
    shared_domains: tuple[str, ...]
    shared_contact_points: tuple[str, ...]

    def joined_on(self) -> str:
        """Everything it is joined on, under the edge, in the grouping's own words."""
        return self.evidence.joined_on((*self.shared_domains, *self.shared_contact_points))


@dataclass(frozen=True, slots=True)
class FalseGrouping:
    """One Campaign Candidate that is a false grouping, spelled out.

    The two name lists are the definition made visible: `campaigns` is every
    Planted Campaign one of its accounts belongs to, and `outside` is every
    account that belongs to no one. A candidate is false when `campaigns`
    holds two or more, or `outside` holds anything. `nuisance` is the
    Nuisance Structure records its accounts appear in, because the rate is
    only readable against that manifest.
    """

    candidate_id: str
    accounts: tuple[str, ...]
    campaigns: tuple[str, ...]
    outside: tuple[str, ...]
    nuisance: tuple[tuple[str, str], ...]  # each as (record_id, kind)

    def why(self) -> str:
        """The reason a reader can check against the definition."""
        reasons = []
        if len(self.campaigns) > 1:
            reasons.append("holds accounts of more than one Planted Campaign")
        if self.outside:
            reasons.append("holds accounts that belong to no Planted Campaign")
        return "; ".join(reasons)


@dataclass(frozen=True, slots=True)
class QueuePrecision:
    """Precision at one review depth.

    `true` of the top `of` entries of the Review Queue hold an account of a
    Planted Campaign. The depth is carried with the figure rather than
    implied, because a precision number without its depth is not reported.
    """

    depth: int
    true: int
    of: int


@dataclass(frozen=True, slots=True)
class NuisanceCount:
    """One kind of Nuisance Structure, and how much of the Corpus it covers.

    `accounts` and `posts` are counted once per kind rather than summed over the
    records of it: an account that reaches a link shortener and a paste site appears in
    two records, and adding it up would report more accounts than the Corpus holds.
    """

    kind: NuisanceKind
    records: int
    accounts: tuple[str, ...]
    posts: tuple[str, ...]
    characters: tuple[tuple[HardNegativeCharacter, int], ...]

    def character_summary(self) -> str:
        """The Hard Negative characters this kind holds, each with its count."""
        return ", ".join(f"{character.value} {count}" for character, count in self.characters)


@dataclass(frozen=True, slots=True)
class NuisanceBaseline:
    """The Nuisance Structure the figure was measured against, read from the manifest.

    `records` is the whole manifest rather than the sum of the kinds, so a kind the
    reader adds later and this run has not seen shows up as a difference between the two
    rather than in neither.
    """

    path: str
    sha256: str
    records: int
    by_kind: tuple[NuisanceCount, ...]


@dataclass(frozen=True, slots=True)
class RecoveryFacts:
    """What reading the five files establishes, as claims about bytes.

    Every figure the table prints lives here rather than being worked out again where it
    is printed, so the figures block reads one object and the two views cannot disagree
    about a count. The digests are here so the figure can be traced to the files it was
    measured against, and the seed is carried rather than looked up: it decides nothing
    here, and it is printed so a reader can regenerate the Corpus this figure is about
    rather than trust the claim that they can.
    """

    accounts: int
    candidate_accounts: int
    candidates: int
    candidates_path: str
    candidates_sha256: str
    campaigns: int
    campaign_posts: int
    corpus_path: str
    corpus_sha256: str
    nuisance_kinds: int
    nuisance_path: str
    nuisance_records: int
    nuisance_sha256: str
    posts: int
    seed: int
    scores_path: str
    scores_sha256: str
    truth_path: str
    truth_sha256: str


@dataclass(frozen=True, slots=True)
class Recovery:
    """Everything one run establishes, so the table and the file cannot disagree.

    `joined` is the set of candidates that reached at least one Planted Campaign and
    `unmatched` is computed as what is left of them, so the two views of the candidates
    cannot add up differently: every candidate either reached a campaign or is printed
    as one that reached none.
    """

    facts: RecoveryFacts
    recoveries: tuple[CampaignRecovery, ...]
    joined: frozenset[str]
    unmatched: tuple[Unmatched, ...]
    nuisance: NuisanceBaseline
    false_groupings: tuple[FalseGrouping, ...]
    precision: tuple[QueuePrecision, ...]
    depths: tuple[int, ...]

    def count(self, outcome: Outcome) -> int:
        return sum(1 for recovery in self.recoveries if recovery.outcome is outcome)

    @property
    def recovered(self) -> int:
        return self.count(Outcome.RECOVERED)

    @property
    def partial(self) -> int:
        return self.count(Outcome.PARTIAL)

    @property
    def missed(self) -> int:
        return self.count(Outcome.MISSED)

    @property
    def partial_candidates(self) -> int:
        """The candidates holding only part of some campaign's membership.

        Counted over every campaign rather than only the partial ones, because a partial
        match is a property of a candidate's reach rather than of a campaign's outcome:
        a campaign recovered whole by one candidate can still have a second candidate
        holding two of its three accounts, and that match is exactly what the figure
        must not quietly drop. The campaign's outcome says the recovery happened; this
        says something else also happened.
        """
        return len(
            {
                match.candidate_id
                for recovery in self.recoveries
                for match in recovery.matches
                if not match.whole
            }
        )

    @property
    def beyond(self) -> tuple[CampaignRecovery, ...]:
        """The campaigns this method cannot reach, in membership order.

        The method's recall bound, stated as the membership rather than as a count, so
        the figure it produces can be taken apart: each of these is a campaign the
        generator planted that no shared thing inside it reaches a candidate.
        """
        return tuple(recovery for recovery in self.recoveries if recovery.bounded)

    @property
    def ceiling(self) -> int:
        """The most campaigns this method could recover, whatever the grouping did.

        An upper bound on the method rather than a prediction of the run: a campaign
        inside the bound can still be missed, because the component holding it may also
        hold an account from outside it — which is what happened to the alpha Planted
        Campaign on this Corpus.
        """
        return len(self.recoveries) - len(self.beyond)


def recover(
    corpus_path: Path,
    candidates_path: Path,
    truth_path: Path,
    nuisance_path: Path,
    seed: int,
    scores_path: Path,
    depths: Sequence[int] = DEFAULT_DEPTHS,
) -> Recovery:
    """Read the five files and join the membership against the published candidates.

    One call, for the same reason the grouping is one call: the figures, the join, and
    the evidence the join is read out of are three views of one pass over five files,
    and a caller that read them twice could end up printing one run's figures over
    another's join.

    The Corpus is read for two reasons and no others: the digests, so the figure can be
    traced to the bytes it is about, and the refusal below. The membership and the
    candidates are checked against it before anything is joined, because a join against a
    membership or a candidate naming an account the Corpus does not hold is
    arithmetically fine and meaningless — the figure would be measuring against files
    that do not describe the same Corpus.

    The resolved links are an argument rather than something read here. This command
    opens neither the Public Suffix List nor the shared-infrastructure list, because it
    does no grouping of its own: it joins what the grouping wrote, and cannot reach
    inference even by accident.
    """
    items = read_corpus(corpus_path)
    candidates = read_campaign_candidates(candidates_path)
    campaigns = read_truth(truth_path)
    manifest = read_nuisance(nuisance_path)
    _check(items, candidates, campaigns)

    recoveries = tuple(_recovery(campaign, candidates) for campaign in campaigns)
    joined = {
        candidate_id
        for recovery in recoveries
        for candidate_id in recovery.candidate_ids
    }
    if not depths:
        raise ValueError("precision needs at least one review depth to be measured at")
    for depth in depths:
        if depth < 1:
            raise ValueError(
                f"a review depth of {depth} orders nothing, and an empty queue reads "
                "like a Corpus with nothing in it worth reviewing"
            )
    queue = build_queue(scores_path, candidates_path, max(depths))
    return Recovery(
        facts=_facts(
            corpus_path,
            candidates_path,
            truth_path,
            nuisance_path,
            items,
            candidates,
            campaigns,
            manifest,
            seed,
            scores_path,
        ),
        recoveries=recoveries,
        joined=frozenset(joined),
        unmatched=tuple(
            _unmatched(candidate)
            for candidate in candidates
            if candidate.candidate_id not in joined
        ),
        nuisance=_baseline(nuisance_path, manifest),
        false_groupings=_false_groupings(candidates, campaigns, manifest),
        precision=_precision(queue.entries, campaigns, depths),
        depths=tuple(depths),
    )


def _check(
    items: Sequence[CorpusItem],
    candidates: Sequence[CampaignCandidate],
    campaigns: Sequence[PlantedCampaign],
) -> None:
    """Refuse files that do not describe the same Corpus.

    Four checks, all of them about a membership or a candidate naming something the
    Corpus does not hold. Each of them would produce a figure that reads well and means
    nothing: N counting a campaign nobody planted, a candidate credited with an account
    it never reached, or a campaign's post attributed to an account outside it, which is
    not a membership at all. Stated as refusals rather than as a lower bound because a
    lower bound is the honest answer to a case the system genuinely cannot do, and these
    are not cases like that.
    """
    authors = {item.post_id: item.account for item in items}
    accounts = set(authors.values())
    named = (
        *((c.campaign_id, c.accounts, c.posts, "membership") for c in campaigns),
        *((c.candidate_id, c.accounts, c.posts, "grouping") for c in candidates),
    )
    for named_by, of_accounts, posts, what in named:
        _check_accounts(named_by, of_accounts, accounts, what)
        _check_posts(named_by, posts, of_accounts, authors)


def _check_accounts(
    named_by: str, named: Sequence[str], accounts: set[str], what: str
) -> None:
    outside = sorted(set(named) - accounts)
    if outside:
        raise ValueError(
            f"{named_by} names {outside}, which no post in the Corpus was written by, "
            f"so it is not a {what} of this Corpus"
        )


def _check_posts(
    named_by: str, posts: Sequence[str], members: Sequence[str], authors: dict[str, str]
) -> None:
    """Posts one membership or one candidate names, against the Corpus's own author.

    A post the Corpus does not hold and a post held under another account's name are
    different mistakes with the same consequence — the membership cannot be recovered
    from what the Corpus actually contains — so they are named apart, because only one
    of the two is fixed by regenerating the file.
    """
    absent = sorted(post for post in posts if post not in authors)
    if absent:
        raise ValueError(f"{named_by} names {absent}, which the Corpus does not hold")
    strangers = sorted(post for post in posts if authors[post] not in set(members))
    if strangers:
        raise ValueError(
            f"{named_by} names {strangers}, which the Corpus holds as another account's "
            "post, so the membership is not one thing"
        )


def _recovery(
    campaign: PlantedCampaign, candidates: Sequence[CampaignCandidate]
) -> CampaignRecovery:
    """One campaign's membership against every candidate that names part of it.

    Membership is compared as a set of accounts rather than as a set of posts. A
    campaign's accounts write other posts, and a candidate's posts are every post its
    accounts wrote, so the two sets differ legitimately and a post comparison would
    report a mismatch on a campaign recovered exactly. The accounts are sorted on both
    sides, which is what lets the outcome be one equality rather than a rule about
    which side came first.
    """
    members = set(campaign.accounts)
    matches = tuple(
        Match(
            candidate_id=candidate.candidate_id,
            accounts=tuple(sorted(candidate.accounts)),
            held=tuple(sorted(members & set(candidate.accounts))),
            missing=tuple(sorted(members - set(candidate.accounts))),
            extra=tuple(sorted(set(candidate.accounts) - members)),
        )
        for candidate in sorted(candidates, key=lambda candidate: candidate.candidate_id)
        if members & set(candidate.accounts)
    )
    return CampaignRecovery(
        campaign_id=campaign.campaign_id,
        accounts=tuple(sorted(campaign.accounts)),
        posts=tuple(sorted(campaign.posts)),
        matches=matches,
        within=_within(members, candidates),
    )


def _within(
    members: set[str], candidates: Sequence[CampaignCandidate]
) -> tuple[tuple[Evidence, str], ...]:
    """What the membership shares that reaches a candidate, read off the candidates file.

    Two of the campaign's own accounts reached by one registration or one Contact Point
    is what makes the campaign reachable by this method at all, and the candidates file
    is where that is recorded: any two accounts reaching the same thing are in one
    component, so the thing is named under the candidate holding them. This is why the
    bound can be measured here without the evaluator doing any grouping of its own, and
    why the evaluator still opens neither the Public Suffix List nor the
    shared-infrastructure list.

    Read off what the grouping published rather than off the Corpus, so it measures what
    the method did rather than what the Corpus holds: a membership whose only shared
    registration was withheld has nothing here, which is exactly why it is beyond the
    method and belongs in the bound.

    The edge travels with the name because a registration and a Contact Point can be the
    same string, and a reader told only `copperlantern.example` could not tell which one
    put a candidate together.
    """
    found: set[tuple[Evidence, str]] = set()
    for candidate in candidates:
        for shared in candidate.shared_domains:
            if len(members & set(shared.accounts)) > 1:
                found.add((Evidence.DOMAIN, shared.domain))
        for point in candidate.shared_contact_points:
            if len(members & set(point.accounts)) > 1:
                found.add((Evidence.CONTACT, point.value))
    return tuple(sorted(found, key=lambda item: (item[0].value, item[1])))


def _false_groupings(
    candidates: Sequence[CampaignCandidate],
    campaigns: Sequence[PlantedCampaign],
    manifest: Sequence[NuisanceRecord],
) -> tuple[FalseGrouping, ...]:
    """Every candidate that is a false grouping, in candidate order.

    The rule, stated plainly: a grouping is false when its accounts belong to
    more than one Planted Campaign, or when any of them belongs to none. The
    two cases together cover accounts grouped together that belong to
    different Planted Campaigns, or to none — the definition this report
    states in its own words — and a candidate whose accounts all sit in one
    campaign is a true grouping whatever the membership looks like.
    """
    owner: dict[str, set[str]] = {}
    for campaign in campaigns:
        for account in campaign.accounts:
            owner.setdefault(account, set()).add(campaign.campaign_id)
    nuisance_of: dict[str, set[tuple[str, str]]] = {}
    for record in manifest:
        for account in record.accounts:
            nuisance_of.setdefault(account, set()).add(
                (record.nuisance_id, record.kind.value)
            )

    false = []
    for candidate in candidates:
        campaign_ids = sorted(
            {
                campaign_id
                for account in candidate.accounts
                if account in owner
                for campaign_id in owner[account]
            }
        )
        outside = sorted(account for account in candidate.accounts if account not in owner)
        if len(campaign_ids) > 1 or outside:
            false.append(
                FalseGrouping(
                    candidate_id=candidate.candidate_id,
                    accounts=tuple(sorted(candidate.accounts)),
                    campaigns=tuple(campaign_ids),
                    outside=tuple(outside),
                    nuisance=tuple(
                        sorted(
                            {
                                held
                                for account in candidate.accounts
                                for held in nuisance_of.get(account, ())
                            }
                        )
                    ),
                )
            )
    return tuple(false)


def _precision(
    entries: Sequence[QueueEntry], campaigns: Sequence[PlantedCampaign], depths: Sequence[int]
) -> tuple[QueuePrecision, ...]:
    """True findings of the top entries, per depth, of the published queue order."""
    campaign_accounts = {account for campaign in campaigns for account in campaign.accounts}
    return tuple(
        QueuePrecision(
            depth=depth,
            true=sum(
                1
                for entry in entries[:depth]
                if entry.score.account in campaign_accounts
            ),
            of=len(entries[:depth]),
        )
        for depth in depths
    )


def _unmatched(candidate: CampaignCandidate) -> Unmatched:
    return Unmatched(
        candidate_id=candidate.candidate_id,
        accounts=tuple(sorted(candidate.accounts)),
        posts=tuple(sorted(candidate.posts)),
        shared_domains=tuple(
            sorted(shared.domain for shared in candidate.shared_domains)
        ),
        shared_contact_points=tuple(
            sorted(point.value for point in candidate.shared_contact_points)
        ),
        evidence=candidate.evidence,
    )


def _baseline(path: Path, manifest: Sequence[NuisanceRecord]) -> NuisanceBaseline:
    """The Nuisance Structure, counted by kind, from the records the manifest holds.

    Kinds are the enum's rather than the manifest's, so a kind the manifest has lost
    prints with a count of zero instead of vanishing: a recovery figure whose baseline
    has quietly shrunk is the failure ADR-0011 is about, and it is only visible if the
    report prints what the baseline should have held.
    """
    by_kind = []
    for kind in NuisanceKind:
        of_kind = [record for record in manifest if record.kind is kind]
        characters = tuple(
            sorted(
                (
                    character,
                    sum(
                        1
                        for record in of_kind
                        for held in record.characters
                        if held is character
                    ),
                )
                for character in HardNegativeCharacter
            )
        )
        by_kind.append(
            NuisanceCount(
                kind=kind,
                records=len(of_kind),
                accounts=tuple(
                    sorted({account for record in of_kind for account in record.accounts})
                ),
                posts=tuple(sorted({post for record in of_kind for post in record.posts})),
                characters=tuple(
                    (character, count) for character, count in characters if count
                ),
            )
        )
    return NuisanceBaseline(
        path=path.as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        records=len(manifest),
        by_kind=tuple(by_kind),
    )


def _facts(
    corpus_path: Path,
    candidates_path: Path,
    truth_path: Path,
    nuisance_path: Path,
    items: Sequence[CorpusItem],
    candidates: Sequence[CampaignCandidate],
    campaigns: Sequence[PlantedCampaign],
    manifest: Sequence[NuisanceRecord],
    seed: int,
    scores_path: Path,
) -> RecoveryFacts:
    """Every figure about the five files the table prints, computed rather than
    transcribed, so a figure and the bytes behind it cannot drift apart."""
    return RecoveryFacts(
        accounts=len({item.account for item in items}),
        candidate_accounts=len(
            {account for candidate in candidates for account in candidate.accounts}
        ),
        candidates=len(candidates),
        candidates_path=candidates_path.as_posix(),
        candidates_sha256=hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        campaigns=len(campaigns),
        campaign_posts=sum(len(campaign.posts) for campaign in campaigns),
        corpus_path=corpus_path.as_posix(),
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        nuisance_kinds=len(NuisanceKind),
        nuisance_path=nuisance_path.as_posix(),
        nuisance_records=len(manifest),
        nuisance_sha256=hashlib.sha256(nuisance_path.read_bytes()).hexdigest(),
        posts=len(items),
        seed=seed,
        scores_path=scores_path.as_posix(),
        scores_sha256=hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        truth_path=truth_path.as_posix(),
        truth_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest(),
    )


def write_recovery(path: Path, recoveries: Sequence[CampaignRecovery]) -> None:
    """One row per Planted Campaign: its membership, its outcome, and the join.

    The file holds the join rather than a count, because a count cannot be taken apart
    and this figure has to be. Every campaign gets a row including the ones that were
    missed, so the outcome column partitions the membership and a reader can add it up
    rather than take the numerator on trust.

    The candidates that matched no campaign are not rows here. They are the other half
    of the figure rather than part of it, they are in `{RECOVERY_PATH}`'s own input,
    and a row per candidate would put two kinds of row in one file where each would be
    read as the other. They are in the console output and the report, named with the
    registrations that join them.
    """

    def objects() -> Iterator[JsonObject]:
        for recovery in recoveries:
            yield {
                "campaign_id": recovery.campaign_id,
                "accounts": list(recovery.accounts),
                "posts": list(recovery.posts),
                "outcome": recovery.outcome.value,
                "candidates": [
                    {
                        "candidate_id": match.candidate_id,
                        "held": list(match.held),
                        "missing": list(match.missing),
                        "extra": list(match.extra),
                    }
                    for match in recovery.matches
                ],
            }

    write_lines(path, objects())


def render_table(recovery: Recovery) -> str:
    """The console output: what was read, the join, what is left over, what it sat on.

    ASCII only, so it prints the same way on a console that cannot encode anything else
    and the same way when it is redirected, which is what lets it be pasted into an
    issue or diffed between runs.
    """
    sections = (
        f"{_HEADING}\n\n{_SUBHEADING}",
        _figures(recovery),
        _campaigns_table(recovery),
        _bound(recovery),
        _unmatched_table(recovery),
        _false_groupings_table(recovery),
        _precision_table(recovery),
        _nuisance_table(recovery),
        _footer(recovery),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _figures(recovery: Recovery) -> str:
    """What was read, and what came of it. One figure per line, labelled."""
    facts = recovery.facts
    fields_out = (
        (
            "corpus",
            f"{facts.corpus_path} ({_count(facts.posts, 'post')} across "
            f"{_count(facts.accounts, 'account')})",
        ),
        ("corpus sha256", facts.corpus_sha256),
        ("seed", str(facts.seed)),
        (
            "candidates",
            f"{facts.candidates_path} ({_count(facts.candidates, 'candidate')}, "
            f"{_count(facts.candidate_accounts, 'account')})",
        ),
        ("candidates sha256", facts.candidates_sha256),
        (
            "truth",
            f"{facts.truth_path} ({_count(facts.campaigns, 'Planted Campaign')}, "
            f"{_count(facts.campaign_posts, 'post')})",
        ),
        ("truth sha256", facts.truth_sha256),
        (
            "nuisance",
            f"{facts.nuisance_path} ({_count(facts.nuisance_records, 'record')} over "
            f"{facts.nuisance_kinds} kinds)",
        ),
        ("nuisance sha256", facts.nuisance_sha256),
        (
            "recovered",
            f"{recovery.recovered} of {facts.campaigns} Planted Campaigns",
        ),
        (
            "partial",
            f"{recovery.partial} Planted Campaigns, {recovery.partial_candidates} of "
            f"{facts.candidates} candidates hold only part of one",
        ),
        ("missed", f"{recovery.missed} Planted Campaigns"),
        (
            "bound",
            f"{len(recovery.beyond)} of {facts.campaigns} Planted Campaigns have nothing "
            "inside them reaching a candidate",
        ),
        (
            "false groupings",
            f"{len(recovery.false_groupings)} of {facts.candidates} Campaign Candidates, "
            "named below with what joins them",
        ),
        (
            "precision",
            f"the Review Queue's entries at depths {_depths(recovery)}, below",
        ),
        (
            "unmatched",
            f"{len(recovery.unmatched)} of {recovery.facts.candidates} candidates, none of "
            f"them a Planted Campaign, "
            f"{_count(sum(len(candidate.accounts) for candidate in recovery.unmatched), 'account')} "
            "in them",
        ),
    )
    width = max(len(name) for name, _ in fields_out)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in fields_out)


def _false_groupings_table(recovery: Recovery) -> str:
    """Every Campaign Candidate that is a false grouping, and what joins it.

    The rate beside the figure is only checkable if each one is named with the
    accounts and the Nuisance Structure records that make it one, because the
    definition is a rule about accounts rather than about candidates.
    """
    if not recovery.false_groupings:
        return (
            "false groupings  no Campaign Candidate; every grouping holds the\n"
            "accounts of one Planted Campaign and no other"
        )
    lines = [
        f"false groupings  {len(recovery.false_groupings)} of "
        f"{recovery.facts.candidates} Campaign Candidates hold accounts that belong to",
        "  different Planted Campaigns, or to none.",
    ]
    for grouping in recovery.false_groupings:
        lines.append(
            f"  {grouping.candidate_id}  {_count(len(grouping.accounts), 'account')}  "
            f"{grouping.why()}"
        )
        for record_id, kind in grouping.nuisance:
            lines.append(f"    {record_id} ({kind})")
    return "\n".join(lines)


def _depths(recovery: Recovery) -> str:
    """The depths the precision figures were measured at, named in the header."""
    return ", ".join(str(depth) for depth in recovery.depths)


def _bound(recovery: Recovery) -> str:
    """The recall bound, and what each membership shares that reaches a candidate.

    Printed for every run, including a run where nothing is bounded, because a figure
    that appears only when it is bad is indistinguishable from a missing one. The rows
    under it are the check: a reader can take each membership and see for themselves what
    reaches a candidate, so the count above is arithmetic rather than an assertion.

    Wording, not decoration: the count is measured against the edges the grouping
    published, so the claim is that nothing inside the membership reaches a candidate
    rather than that the accounts share nothing at all. A membership whose only shared
    registration was withheld shares plenty and is still beyond the method, and it is
    this wording that keeps the count honest about which of the two it found.
    """
    count = len(recovery.beyond)
    campaigns = len(recovery.recoveries)
    lines = [
        "recall bound",
        *wrap(
            f"{count} of {campaigns} Planted Campaigns have nothing inside their own "
            "membership that reaches a Campaign Candidate, so no edge this method rests "
            f"on can hold {_plural(count, 'it', 'them')}: the figure above sits under a "
            f"ceiling of {recovery.ceiling} of {campaigns}, which is a limit on the "
            "method rather than a prediction of the run",
            indent=2,
        ),
    ]
    for entry in recovery.recoveries:
        lines.append(
            f"  {entry.campaign_id}  "
            + (entry.shared_within() if entry.within else "nothing reaches a candidate")
        )
    return "\n".join(lines)


def _count(number: int, noun: str) -> str:
    """One count, agreeing with its noun. Every figure here is small, and a reader
    seeing "1 accounts" stops to wonder whether the figure is right."""
    return f"{number} {noun if number == 1 else f"{noun}s"}"


def _plural(number: int, one: str, many: str) -> str:
    """The word that agrees with a count: one campaign shares, none share.

    The one agreement helper, so a console line and a report line naming the same figure
    cannot disagree about the word that goes with it — which is how the two views of the
    recall bound came to saying "them" and "it" for the same count of one.
    """
    return one if number == 1 else many


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """One line of a table. A cell of nothing but digits is a count, and counts are
    right-aligned so the digits line up down the column; the rest is text of no fixed
    width and reads better against the left edge."""
    return "  ".join(
        cell.rjust(width) if cell.isdigit() else cell.ljust(width)
        for cell, width in zip(cells, widths, strict=True)
    )


def _table(headings: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(cell) for cell in column) for column in zip(headings, *rows, strict=True)]
    return "\n".join([_row(headings, widths), *(_row(row, widths) for row in rows)])


def _campaigns_table(recovery: Recovery) -> str:
    """Every Planted Campaign, its outcome, and the candidates that reached it.

    Every match that is not a whole-membership match is printed under its campaign,
    whatever the campaign's own outcome was, because "partial" and "missed" on their
    own are not diagnosable: a reader who wants to know whether the system was nearly
    right needs to see which accounts it held, which it did not, and which it reached
    for that the membership never named. A campaign recovered by one candidate and
    half-reached by another has both facts to show, and printing only the recovery
    would be the output agreeing with itself.
    """
    headings = ("campaign", "outcome", "accounts", "posts", "candidates")
    rows = [
        (
            campaign.campaign_id,
            campaign.outcome.value,
            str(len(campaign.accounts)),
            str(len(campaign.posts)),
            ", ".join(campaign.candidate_ids) or "-",
        )
        for campaign in recovery.recoveries
    ]
    lines = [_table(headings, rows)]
    for campaign in recovery.recoveries:
        partials = campaign.partial_matches
        if not partials and campaign.matches:
            continue
        lines.append(f"  {campaign.campaign_id}")
        lines.extend(_match(match, campaign) for match in partials)
        if not campaign.matches:
            lines.append(f"    no candidate names {', '.join(campaign.accounts)}")
    return "\n".join(lines)


def _match(match: Match, campaign: CampaignRecovery) -> str:
    """One candidate's reach over one membership, and what it got wrong."""
    lines = [
        f"    {match.candidate_id}  holds {_count(len(match.held), 'account')} of "
        f"{len(campaign.accounts)}: {', '.join(match.held)}"
    ]
    if match.missing:
        lines.append(f"      not held  {', '.join(match.missing)}")
    if match.extra:
        lines.append(
            f"      also holds  {', '.join(match.extra)}, which the membership does not name"
        )
    return "\n".join(lines)


def _unmatched_table(recovery: Recovery) -> str:
    """The candidates that reached no campaign, and what they were joined on.

    A candidate a reader cannot account for is the thing that makes X of N
    uninterpretable: `2 of 2` beside three candidates is a claim about one third of the
    output. The shared registrations and Contact Points are printed for the same reason
    the grouping prints its own - the shared thing is the whole of the reason those
    accounts are together, so the reason is what a reader needs, and a candidate resting
    on a Contact Point names the Contact Point rather than nothing at all.
    """
    if not recovery.unmatched:
        return "unmatched  no candidate; every one of them reached a Planted Campaign"

    unmatched = len(recovery.unmatched)
    heading = (
        f"unmatched  {_count(recovery.facts.candidates, 'candidate')} published, of which "
        f"{unmatched} {_plural(unmatched, 'reaches', 'reach')} no Planted Campaign."
    )
    lines = [heading, "  not one of them is counted against the figure above"]
    for candidate in recovery.unmatched:
        lines.append(
            f"  {candidate.candidate_id}  {_count(len(candidate.accounts), 'account')}, "
            f"{_count(len(candidate.posts), 'post')}  {candidate.joined_on()}"
        )
    return "\n".join(lines)


def _precision_table(recovery: Recovery) -> str:
    """Precision at each depth, one row per depth, so the shape shows."""
    rows = [
        (str(measure.depth), str(measure.true), str(measure.of))
        for measure in recovery.precision
    ]
    table = _table(("depth", "true", "of"), rows)
    return (
        "precision in the Review Queue, at several depths\n\n"
        "True findings of the top entries. A precision number is not printed\n"
        "without its depth.\n\n" + table
    )


def _nuisance_table(recovery: Recovery) -> str:
    """The Nuisance Structure the figure was measured against, counted by kind.

    Every kind of the enum is printed, including any the manifest has lost, because a
    baseline that quietly shrank is invisible unless the page says what it should have
    held. Accounts and posts are counted once per kind: an account reaching a shortener
    and a paste site is in two records and is one account.
    """
    facts = recovery.facts
    lines = [
        f"nuisance  {facts.nuisance_path} "
        f"({_count(facts.nuisance_records, 'record')} over {facts.nuisance_kinds} kinds)"
    ]
    for count in recovery.nuisance.by_kind:
        lines.append(
            f"  {count.kind.value.ljust(_KIND_WIDTH)}  {_count(count.records, 'record')}  "
            f"{_count(len(count.accounts), 'account')}  "
            f"{_count(len(count.posts), 'post')}"
        )
        if count.characters:
            lines.append(f"    characters  {count.character_summary()}")
    return "\n".join(lines)


def _footer(recovery: Recovery) -> str:
    """Who measured this, what the figure is not, and what it is bounded by.

    The reviewer in this project is a fiction and the output has to say so: a figure a
    reader could take for a person's judgment about synthetic posts would misrepresent
    the system, and so would one that read as a rate of finding fraud in the world. The
    bounds are the ones a reader would otherwise have to guess at, and they are why the
    figure is a lower bound rather than an estimate.

    Every sentence carrying figures is worked out above the block and dropped in on a
    line of its own, because a line that is 200 columns wide is not a paragraph and
    wrapping it by hand inside the string would mean re-wrapping it whenever a figure
    changes length.
    """
    facts = recovery.facts
    records_line = (
        f"The manifest holds {facts.nuisance_records} records over "
        f"{facts.nuisance_kinds} kinds."
    )
    unmatched = len(recovery.unmatched)
    candidates_line = (
        f"Of the {facts.candidates} candidates, {unmatched} "
        f"{_plural(unmatched, 'reaches', 'reach')} no Planted Campaign."
    )
    return f"""\
Measured by `rfi campaign-recovery`, against the Planted Campaign membership in
{facts.truth_path}. No person reviewed any of it: there is no reviewer behind this
number, nothing was clicked, and a Campaign Candidate is a proposal rather than a
finding.

No figure over the whole Corpus is published here or in the page beside this one. Every
label in this Corpus was assigned by the generator that wrote the posts, so a share of
the posts called right or wrong would measure agreement with the generator rather than
anything about fraud, and publishing it would be a number nobody could falsify
(ADR-0004).

What the figure is bounded by is stated rather than left in the tickets, and the bound
is a count rather than an adjective. {recovery.recovered} of {len(recovery.recoveries)}
Planted Campaigns were recovered and {len(recovery.beyond)} {_plural(len(recovery.beyond), 'has', 'have')}
nothing inside {_plural(len(recovery.beyond), 'its', 'their')} own membership that reaches a candidate, so this
method cannot hold {_plural(len(recovery.beyond), 'it', 'them')} by construction: the figure sits under a
ceiling of {recovery.ceiling} of {len(recovery.recoveries)}, which is a limit on the method rather than a
prediction of the run. A Planted Campaign that leans on a known-shared host is lost with
the host, and a campaign that rotates both its registrations and its Contact Points is
invisible to any method resting on these two edges. Recovery measured this way is a
statement about planted structure in a synthetic Corpus, and it is not an estimate of
fraud found in the world.

The Nuisance Structure above is the other half of the claim.
{records_line}
{candidates_line}
The false-grouping rate above is measured against the same Nuisance Structure, and both
of those numbers are why the baseline is printed here rather than left to be found. Of
the two rates on this page, recovery and precision, the lower of the two numbers is the
more trustworthy, and this is deliberate: either one can flatter the system while the
other quietly fails.

Nothing in this command reads anything the grouping reads, and nothing in the grouping
reads the membership or the manifest: this is a separate command over the candidates file
named above, which `rfi campaign-candidates` wrote (ADR-0018). The digests above are what
a reader would check those files against."""

def render_report(recovery: Recovery) -> str:
    """The reader-facing page, generated from the five files the run read.

    Written for a reader who has to see the join rather than the total: every campaign
    gets a row with its outcome, every outcome that is not a recovery gets its join
    spelled out beneath it, the candidates that reached no campaign are named with the
    registrations that join them, and the Nuisance Structure the figure was measured
    against has a section rather than a footnote. Every figure in the prose is computed
    from the files — nothing about which campaign was recovered is written by hand,
    because the page is generated and a sentence that was true of one Corpus would
    quietly be untrue of the next. The join is named by the path the command writes by
    default, so two runs over the same five files produce the same bytes whichever
    directory they write to.
    """
    facts = recovery.facts
    table = "\n".join(
        f"| {campaign.campaign_id} | {campaign.outcome.value} | {len(campaign.accounts)} | "
        f"{len(campaign.posts)} | {', '.join(campaign.candidate_ids) or '—'} |"
        for campaign in recovery.recoveries
    )
    joins = "\n".join(
        _join(campaign) for campaign in recovery.recoveries if campaign.matches
    )
    misses = "\n".join(
        f"- **{campaign.campaign_id}** — no candidate names "
        f"{', '.join(f'`{account}`' for account in campaign.accounts)}."
        for campaign in recovery.recoveries
        if not campaign.matches
    )
    left_over = _unmatched_report(recovery)
    nuisance = "\n".join(_nuisance_row(count) for count in recovery.nuisance.by_kind)
    characters = "\n".join(
        f"- `{count.kind.value}` — {count.character_summary()}."
        for count in recovery.nuisance.by_kind
        if count.characters
    )
    widest = max(
        recovery.nuisance.by_kind,
        key=lambda count: (len(count.accounts), count.records),
    )
    headline = (
        f"**{recovery.recovered} of {len(recovery.recoveries)} Planted Campaigns "
        f"recovered**, with {recovery.partial} partial and {recovery.missed} missed."
    )
    largest = (
        f"The largest piece of it is `{widest.kind.value}`, holding "
        f"{len(widest.accounts)} accounts across {_count(widest.records, 'record')}."
    )
    false_count = len(recovery.false_groupings)
    false_rows = "\n".join(
        f"| `{grouping.candidate_id}` | {len(grouping.accounts)} | {grouping.why()} | "
        f"{', '.join(f'{record_id} ({kind})' for record_id, kind in grouping.nuisance) or '—'} |"
        for grouping in recovery.false_groupings
    ) or "| — | — | — | — |"
    precision_rows = "\n".join(
        f"| {measure.depth} | {measure.true} | {measure.of} |"
        for measure in recovery.precision
    )
    bound_rows = "\n".join(
        f"| `{entry.campaign_id}` | "
        + (
            "; ".join(
                f"{edge.value}: {', '.join(f'`{name}`' for name in names)}"
                for edge, names in entry.shared_by_edge()
                if names
            )
            if entry.within
            else "**nothing reaches a candidate**"
        )
        + " |"
        for entry in recovery.recoveries
    )

    return f"""# Recovery of Planted Campaigns

Generated by `rfi campaign-recovery` from the committed Corpus, the candidates
`rfi campaign-candidates` published, the Planted Campaign membership, the Nuisance
Structure manifest, and the Policy Scores `rfi policy-score` published. Do not edit it
by hand — a test holds this file to what the command produces, and re-running the
command rewrites it byte for byte.

## The figure

{headline}

The three outcomes partition the membership in `{facts.truth_path}`, so a reader can add
them up and get N rather than take the numerator on trust.

| Planted Campaign | Outcome | Accounts | Posts | Candidates |
| --- | --- | ---: | ---: | --- |
{table}

A Campaign Candidate counts as a recovery only when it holds a Planted Campaign's whole
membership. That is the whole of the rule, and it is stricter than it sounds: two
accounts of a three-account campaign grouped on their shared registration is a real
grouping and not a recovery, and a candidate holding a campaign's three accounts and one
of its own is an over-grouping rather than half a recovery. Both are reported apart from
the figure, with the accounts named, because a number a reader cannot take apart is a
number a reader has to take on trust (ADR-0004).

### The joins behind it

{joins or "- Every Planted Campaign was recovered on a candidate holding its whole membership."}

{misses or "- No Planted Campaign went ungrouped."}

## Candidates that are not a recovery

{left_over}

None of these is counted against the figure above, and none of them is a miss. ADR-0005
puts accounts in a Campaign Candidate when they share a registrable domain or a shared
Contact Point, so both of the shapes the Corpus plants are groupings the system is
*right* to produce: a small business whose three accounts share its own domain, and one
desk whose three accounts publish one Telegram Contact Point three ways. Neither is a
Planted Campaign, and counting either against N would report correct behaviour as a
failure and would put N above the number of things that were planted. The rate at which
the grouping is wrong is reported below: it is counted against the same manifest these
candidates were measured against.

## The recall bound

{_bound_line(recovery)}

| Planted Campaign | Shares within its own membership |
| --- | --- |
{bound_rows}

{_bound_prose(recovery)}

## False groupings

A false grouping is a Campaign Candidate whose accounts belong to different Planted
Campaigns, or to none. Stated as a rule rather than left to be inferred from the
figure, so a reader can check the number against the sentence rather than the other
way round. One of the candidates above rests on a Contact Point alone, and a grouping
built on the weaker of the two readings is the one this page should be read with more
care: `rfi campaign-candidates` prints the measured recall behind that reading
beside the candidate, and it is not the same kind of evidence as a registration.

**{false_count} of {facts.candidates} Campaign Candidates are false groupings.**
Measured against the Nuisance Structure in `{facts.nuisance_path}`, because a
false-grouping rate over a clean sweep is a number nobody can falsify (ADR-0022).

| Candidate | Accounts | Why it is a false grouping | Nuisance records |
| --- | ---: | --- | --- |
{false_rows}

## Precision in the Review Queue at several depths

Precision at a depth is the share of the top entries of the Review Queue whose posts
hold an account of a Planted Campaign. Several depths rather than one, so the shape
of the trade-off is visible rather than a single number.

| Depth | True findings | Entries |
| ---: | ---: | ---: |
{precision_rows}

Measured at depths {_depths(recovery)}, against the Planted Campaign membership by
this command, not by a person reviewing the queue (ADR-0022). Of the two rates on this
page — recovery and precision — the lower of the two numbers is the more trustworthy,
and this is deliberate: either one can flatter the system while the other quietly
fails. Recovery and the false-grouping rate are reported together on this page;
neither number appears alone.

## What it was measured against

`{facts.nuisance_path}` holds {recovery.nuisance.records} records, and they are the
baseline this figure sits on: a recovery rate over a Corpus that held nothing else would
be a figure about a generator that planted two campaigns and said so nowhere. Every kind
is printed below, including any the manifest has lost, because a baseline that quietly
shrank is invisible unless the page says what it should have held. Accounts and posts are
counted once per kind: an account reaching a link shortener and a paste site appears in
two records and is one account.

| Kind | Records | Accounts | Posts |
| --- | ---: | ---: | ---: |
{nuisance}

{characters}

{largest}

Every row of the manifest carries a note in prose saying what it is there to test, so the
file reads without running anything, and the kinds are defined in `GLOSSARY.md`.

## Who measured it

Measured by `rfi campaign-recovery`, against the Planted Campaign membership in
`{facts.truth_path}`. **No person reviewed any of it**: there is no reviewer behind this
number, nothing was clicked, and a Campaign Candidate is a proposal rather than a
finding. What was measured is what the generator planted, which is membership by
construction rather than a judgement anybody made about the content.

The measurement is a separate command over a separate file. This one reads
`{facts.candidates_path}`, which `rfi campaign-candidates` wrote, and the Policy
Scores `{facts.scores_path}`, and opens neither the Public Suffix List nor the
shared-infrastructure list because it does no grouping of its own. Nothing in the
grouping path reads the membership or the manifest, so the two steps cannot be
reordered into one process that has both (ADR-0008, ADR-0018).

## What this number cannot say

**No figure over the whole Corpus is published here or in the console output.** Every
label in this Corpus was assigned by the generator that wrote the posts, so a share of
the posts called right or wrong would measure agreement with the generator rather than
anything about fraud, and it would be a number nobody could falsify. That is why the
figure is a count of recovered campaigns and not a rate over labelled posts (ADR-0004).

**The figure is a lower bound, and the bound is stated as a count.** A Planted Campaign
that leans on a known-shared host is lost with the host, because the registration is
withheld before the grouping; and a campaign that rotates both its registrations and its
Contact Points leaves nothing inside itself that reaches a candidate, so no edge this
method rests on can hold it however well the grouping works. That last case is the recall
bound, counted in its own section above. It is published rather than described, because a
rate a reader has to take on trust about its own ceiling is not a measurement.

**The figure says nothing about fraud in the world.** Everything above is a statement
about planted structure in a synthetic Corpus, measured against membership the generator
wrote. It is not an estimate of fraud found in the world, and no figure over the whole
Corpus is published to suggest otherwise.

**Recovery and the false-grouping rate are measured together.** Recovery alone can be
produced by a grouping that also merges unrelated accounts, so the rate of false
groupings above is the companion this figure has always needed: the two numbers are
reported together, and neither appears alone.

## Reproducing it

`uv run rfi generate-corpus --seed {facts.seed}` writes `{facts.corpus_path}`,
`{facts.truth_path}`, `{facts.nuisance_path}` and the shared-infrastructure list again
byte for byte; `uv run rfi campaign-candidates` writes `{facts.candidates_path}` again
byte for byte; `uv run rfi policy-score` writes `{facts.scores_path}` again byte for
byte; this command writes `{RECOVERY_PATH}` and this page. Nothing in that chain reads
a seed at run time — the seed decides what was generated, and it is printed here so a
reader can regenerate rather than take the claim on trust.

The digests below are what the five files the figure was measured against were when
this page was written:

| File | SHA-256 |
| --- | --- |
| `{facts.corpus_path}` | `{facts.corpus_sha256}` |
| `{facts.candidates_path}` | `{facts.candidates_sha256}` |
| `{facts.truth_path}` | `{facts.truth_sha256}` |
| `{facts.nuisance_path}` | `{facts.nuisance_sha256}` |
| `{facts.scores_path}` | `{facts.scores_sha256}` |
"""


def _bound_line(recovery: Recovery) -> str:
    """The bound as one sentence, used by the report and by nothing else.

    Both views say it, and they say it in the same words because it is one figure: a
    console and a page that named different ceilings would be two claims about the same
    run, and the page is generated so the agreement is the default rather than a
    coincidence.
    """
    count = len(recovery.beyond)
    campaigns = len(recovery.recoveries)
    return (
        f"**{count} of the {campaigns} Planted Campaigns have nothing inside their own "
        "membership that reaches a Campaign Candidate**, so no edge this method rests on "
        f"can hold {_plural(count, 'it', 'them')} at all: the figure above sits under a "
        f"ceiling of {recovery.ceiling} of {campaigns}, which is a limit on the method "
        "rather than a prediction of the run."
    )


def _bound_prose(recovery: Recovery) -> str:
    """What the bound is and is not, in the words a reader has to be able to check.

    The check is in the table above rather than in this paragraph: a membership with
    nothing reaching a candidate inside it cannot be recovered by one, because holding
    two of its accounts means also holding whatever connects them, and nothing does.

    The wording counts what reaches a candidate rather than what the accounts share at
    all, and the difference matters: a membership whose one shared registration was
    withheld shares plenty, is still beyond this method, and belongs in this count. The
    evaluator cannot tell the two apart without opening the shared-infrastructure list,
    which it must not do (ADR-0018).
    """
    count = len(recovery.beyond)
    if count:
        return (
            "Every one of those is a miss this method could not have avoided: two accounts "
            "of it reach nothing in common that a candidate could hold them on, so a "
            "candidate that held them would have had to reach outside the membership, "
            "which is an over-grouping rather than a recovery. The other "
            f"{recovery.ceiling} {_plural(recovery.ceiling, 'campaign is', 'campaigns are')} "
            "in the table above share something that reaches a candidate, so they were "
            "within reach — and within reach is not recovered: a campaign can be inside "
            "this bound and still be missed, because the component holding it may also "
            "hold an account from outside it, which is what happened to the alpha Planted "
            "Campaign on this Corpus."
        )
    return (
        "Every membership in this Corpus shares something that reaches a candidate, so "
        "nothing here is out of the method's reach by construction and the ceiling above "
        "is all of them. That is a fact about this Corpus and not a general one: a "
        "campaign that rotates both its registrations and its Contact Points is invisible "
        "to a method resting on those two edges, and one that leans on a known-shared host "
        "is lost with the host."
    )


def _join(campaign: CampaignRecovery) -> str:
    """One campaign's membership, and every candidate's reach over it."""
    lines = [f"- **{campaign.campaign_id}** — {_count(len(campaign.accounts), 'account')}, "
             f"{_count(len(campaign.posts), 'post')}, outcome `{campaign.outcome.value}`."]
    for match in campaign.matches:
        detail = f"  - `{match.candidate_id}` holds {', '.join(f'`{account}`' for account in match.held)}"
        if match.missing:
            detail += f"; not held: {', '.join(f'`{account}`' for account in match.missing)}"
        if match.extra:
            detail += (
                f"; also holds "
                f"{', '.join(f'`{account}`' for account in match.extra)}, which the "
                "membership does not name"
            )
        lines.append(detail)
    return "\n".join(lines)


def _unmatched_report(recovery: Recovery) -> str:
    """The candidates that reached no campaign, one line each, with what joins them."""
    if not recovery.unmatched:
        return "Every Campaign Candidate this run produced reached a Planted Campaign."

    unmatched = len(recovery.unmatched)
    lines = [
        f"Of the {_count(recovery.facts.candidates, 'candidate')} published, "
        f"{unmatched} {_plural(unmatched, 'reaches', 'reach')} no account of any "
        "Planted Campaign."
    ]
    lines.append(
        "Each is printed with the registrations and Contact Points that join it, and with "
        "the edge they rest on, because that is the whole of the reason its accounts are "
        "together:"
    )
    for candidate in recovery.unmatched:
        lines.append(
            f"- `{candidate.candidate_id}` — {_count(len(candidate.accounts), 'account')}, "
            f"{_count(len(candidate.posts), 'post')}, {candidate.evidence.value}: "
            f"{_named(candidate)}."
        )
    return "\n".join(lines)


def _named(candidate: Unmatched) -> str:
    """A candidate's evidence in a report, each name in code formatting.

    One helper rather than a join at the call site, so the empty case says what it means
    instead of rendering an empty set of backticks and so the phrase comes from the same
    `Evidence` the grouping printed it with.
    """
    return ", ".join(
        f"`{name}`" for name in (*candidate.shared_domains, *candidate.shared_contact_points)
    ) or candidate.evidence.value


def _nuisance_row(count: NuisanceCount) -> str:
    return (
        f"| `{count.kind.value}` | {count.records} | {len(count.accounts)} | "
        f"{len(count.posts)} |"
    )
