"""Deterministic Corpus generation.

A fixed seed fixes the Corpus: two runs with the same seed write byte-identical
files. The Corpus, the truth file, and the Nuisance Structure manifest are projections
of one plan, and the Corpus projection carries nothing the others carry — a
`CorpusItem` has no membership field and no nuisance field, and the writer that
serialises it takes nothing but items, so there is no field for either to leak
through. A reader does not have to trust that: they can grep the file (ADR-0008).
The known-shared infrastructure list is written alongside them rather than projected
from the plan, because it is an input to the grouping step and does not vary with the
seed.

What the seed controls: the window the Corpus covers, the minute each post was made,
and which of the spare Nuisance Structure material the Corpus carries. Which
Synthetic Entities exist is fixed, because every test in the repository reads the
Corpus by name.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import count

from reddit_fraud_intelligence.content import (
    PLANTED,
    UNGROUPED,
    PlantedScript,
    SyntheticPost,
    link_hosts,
)
from reddit_fraud_intelligence.corpus import CorpusItem
from reddit_fraud_intelligence.infrastructure import SHARED_HOSTS
from reddit_fraud_intelligence.nuisance import (
    SPARES,
    STRUCTURES,
    NuisanceKind,
    NuisanceRecord,
    NuisanceStructure,
)
from reddit_fraud_intelligence.truth import PlantedCampaign

_ANCHOR = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
_CORPUS_WINDOW_DAYS = 365
_UNGROUPED_SPREAD_MINUTES = 30 * 24 * 60


@dataclass(frozen=True, slots=True)
class _Plan:
    posts: tuple[CorpusItem, ...]
    campaigns: tuple[PlantedCampaign, ...]
    nuisance: tuple[NuisanceRecord, ...]


@dataclass(frozen=True, slots=True)
class _Planted:
    """A structure the plan has filled in: each post with the identifier it was given."""

    structure: NuisanceStructure
    planted: tuple[tuple[str, SyntheticPost], ...]


def corpus_items(seed: int) -> list[CorpusItem]:
    """The Corpus, and nothing about how it was planted."""
    return list(_plan(seed).posts)


def planted_campaigns(seed: int) -> list[PlantedCampaign]:
    """The membership record the evaluator joins, after inference has finished."""
    return list(_plan(seed).campaigns)


def nuisance_records(seed: int) -> list[NuisanceRecord]:
    """What the Corpus holds that is not a Planted Campaign, and why it is there."""
    return list(_plan(seed).nuisance)


def _plan(seed: int) -> _Plan:
    window_start = _ANCHOR + timedelta(days=seed % _CORPUS_WINDOW_DAYS)
    post_ids = count(1)
    posts: list[CorpusItem] = []
    campaigns: list[PlantedCampaign] = []
    paraphrases: list[NuisanceRecord] = []

    for script in PLANTED:
        member_posts = []
        for index, post in enumerate(script.posts):
            member_post_id = _next_post_id(post_ids)
            stagger = script.stagger_minutes
            posts.append(
                _item(
                    post,
                    member_post_id,
                    _staggered_at(seed, member_post_id, window_start, index, stagger),
                )
            )
            member_posts.append(member_post_id)
        campaigns.append(
            PlantedCampaign(
                campaign_id=script.campaign_id,
                accounts=script.accounts,
                posts=tuple(member_posts),
            )
        )
        if script.paraphrase_note is not None:
            paraphrases.append(_paraphrase(script, tuple(member_posts), script.paraphrase_note))

    for post in UNGROUPED:
        post_id = _next_post_id(post_ids)
        posts.append(
            _item(
                post,
                post_id,
                _scattered_at(seed, post_id, window_start, _UNGROUPED_SPREAD_MINUTES),
            )
        )

    structures: list[_Planted] = []
    for structure in _structures(seed):
        spread = structure.spread_hours * 60
        planted_posts = []
        for post in structure.posts:
            post_id = _next_post_id(post_ids)
            posts.append(
                _item(post, post_id, _scattered_at(seed, post_id, window_start, spread))
            )
            planted_posts.append((post_id, post))
        structures.append(_Planted(structure, tuple(planted_posts)))

    return _Plan(
        posts=tuple(posts),
        campaigns=tuple(campaigns),
        nuisance=(
            *paraphrases,
            *(
                planted.structure.record(tuple(post_id for post_id, _ in planted.planted))
                for planted in structures
            ),
            *_shared_infrastructure(structures),
        ),
    )


def _paraphrase(
    script: PlantedScript, post_ids: tuple[str, ...], note: str
) -> NuisanceRecord:
    """The staggered-paraphrase record for a Planted Campaign whose posts reword one
    text. It is not planted material: the posts are already in the Corpus as campaign
    members, so this records why they are one thing rather than four."""
    return NuisanceRecord(
        kind=NuisanceKind.STAGGERED_PARAPHRASE,
        nuisance_id=f"syn-nuisance-paraphrase-{script.campaign_id.rsplit('-', 1)[-1]}",
        accounts=script.accounts,
        posts=post_ids,
        hosts=tuple(
            sorted({h for post in script.posts for h in link_hosts(post.links)})
        ),
        characters=(),
        note=note,
    )


def _structures(seed: int) -> tuple[NuisanceStructure, ...]:
    """What this seed plants.

    The declared structures are always there — they are what guarantees every kind of
    Nuisance Structure is present whatever the seed — and the seed draws from the spare
    pool. The draw may be empty: a seed that picks nothing still produces a Corpus with
    the whole Nuisance Structure in it.
    """
    rng = _rng(seed, "nuisance-spares")
    spares = rng.sample(SPARES, rng.randrange(len(SPARES) + 1))
    return (*STRUCTURES, *spares)


def _shared_infrastructure(structures: list[_Planted]) -> tuple[NuisanceRecord, ...]:
    """One record per known-shared host, naming the posts and accounts that touch it.

    Computed from the plan rather than declared: the set of posts linking a shortener
    is not something a hand-written list survives contact with the next seed, and a
    record that disagrees with the Corpus is worse than no record.
    """
    records = []
    for host in SHARED_HOSTS:
        post_ids = []
        accounts: set[str] = set()
        for planted in structures:
            for post_id, post in planted.planted:
                if host.host in link_hosts(post.links):
                    post_ids.append(post_id)
                    accounts.add(post.account)
        records.append(
            NuisanceRecord(
                kind=NuisanceKind.KNOWN_SHARED_INFRASTRUCTURE,
                nuisance_id=f"syn-nuisance-shared-{host.host.split('.')[0]}",
                accounts=tuple(sorted(accounts)),
                posts=tuple(post_ids),
                hosts=(host.host,),
                characters=(),
                note=(
                    f"{host.host} is a {host.kind.value.replace('_', ' ')}, used by "
                    f"{len(accounts)} accounts in this Corpus that have nothing else in "
                    "common. It cannot form a Campaign Candidate on its own, and any "
                    "component it appears in is a false grouping until it is filtered."
                ),
            )
        )
    return tuple(records)


def _item(post: SyntheticPost, post_id: str, created_at: str) -> CorpusItem:
    """One post, at a time the plan has already chosen."""
    return CorpusItem(
        post_id=post_id,
        account=post.account,
        subreddit=post.subreddit,
        title=post.title,
        body=post.body,
        created_at=created_at,
        links=post.links,
    )


def _next_post_id(post_ids: count[int]) -> str:
    return f"syn_p_{next(post_ids):04d}"


def _scattered_at(
    seed: int, post_id: str, window_start: datetime, spread_minutes: int
) -> str:
    """One post anywhere inside a window. How the Nuisance Structure arrives: real
    accounts post over weeks, with nothing linking one week's post to the next."""
    return _stamp(window_start, _rng(seed, _key(post_id)).randrange(spread_minutes))


def _staggered_at(
    seed: int,
    post_id: str,
    window_start: datetime,
    index: int,
    stagger_minutes: int,
) -> str:
    """A group of posts in order, a random interval apart, so a Planted Campaign's
    accounts appear across hours rather than all at once."""
    rng = _rng(seed, _key(post_id))
    minutes = index * stagger_minutes + rng.randrange(stagger_minutes)
    return _stamp(window_start, minutes)


def _stamp(window_start: datetime, minutes: int) -> str:
    return (window_start + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _key(post_id: str) -> str:
    """Timestamps are keyed on the post, so adding a post does not move the others."""
    return f"{post_id}:created_at"


def _rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))
