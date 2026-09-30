"""Deterministic Corpus generation.

A fixed seed fixes the Corpus: two runs with the same seed write byte-identical
files. The Corpus and the truth file are two projections of one plan, and the
Corpus projection carries nothing the membership projection carries — a
`CorpusItem` has no membership field, and the writer that serialises it takes
nothing but items, so there is no field for membership to leak through. A reader
does not have to trust that: they can grep the file (ADR-0008).

What the seed controls at this stage: the window the Corpus covers, and the
minute each post was made. Later work grows the seed to scale the Corpus and to
vary its Nuisance Structure; until then it is honest to say the seed does not
change which entities are planted.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import count

from reddit_fraud_intelligence.content import PLANTED, UNGROUPED, SyntheticPost
from reddit_fraud_intelligence.corpus import CorpusItem
from reddit_fraud_intelligence.truth import PlantedCampaign

_ANCHOR = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
_CORPUS_WINDOW_DAYS = 365
_UNGROUPED_SPREAD_HOURS = 30 * 24


@dataclass(frozen=True, slots=True)
class _Plan:
    posts: tuple[CorpusItem, ...]
    campaigns: tuple[PlantedCampaign, ...]


def corpus_items(seed: int) -> list[CorpusItem]:
    """The Corpus, and nothing about how it was planted."""
    return list(_plan(seed).posts)


def planted_campaigns(seed: int) -> list[PlantedCampaign]:
    """The membership record the evaluator joins, after inference has finished."""
    return list(_plan(seed).campaigns)


def _plan(seed: int) -> _Plan:
    window_start = _ANCHOR + timedelta(days=seed % _CORPUS_WINDOW_DAYS)
    post_ids = count(1)
    posts: list[CorpusItem] = []
    campaigns: list[PlantedCampaign] = []

    for script in PLANTED:
        member_posts = []
        for post in script.posts:
            member_post_id = _next_post_id(post_ids)
            posts.append(
                _item(seed, window_start, post, member_post_id, script.spread_hours)
            )
            member_posts.append(member_post_id)
        campaigns.append(
            PlantedCampaign(
                campaign_id=script.campaign_id,
                accounts=script.accounts,
                posts=tuple(member_posts),
            )
        )

    for post in UNGROUPED:
        posts.append(
            _item(seed, window_start, post, _next_post_id(post_ids), _UNGROUPED_SPREAD_HOURS)
        )

    return _Plan(posts=tuple(posts), campaigns=tuple(campaigns))


def _item(
    seed: int,
    window_start: datetime,
    post: SyntheticPost,
    post_id: str,
    spread_hours: int,
) -> CorpusItem:
    return CorpusItem(
        post_id=post_id,
        account=post.account,
        subreddit=post.subreddit,
        title=post.title,
        body=post.body,
        created_at=_created_at(seed, post_id, window_start, spread_hours),
        links=post.links,
    )


def _next_post_id(post_ids: count[int]) -> str:
    return f"syn_p_{next(post_ids):04d}"


def _created_at(
    seed: int, post_id: str, window_start: datetime, spread_hours: int
) -> str:
    """A timestamp keyed on the post, so adding a post does not move the others."""
    rng = _rng(seed, f"{post_id}:created_at")
    return (window_start + timedelta(minutes=rng.randrange(spread_hours * 60))).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))
