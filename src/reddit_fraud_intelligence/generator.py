"""Deterministic Corpus generation.

A fixed seed fixes the Corpus: two runs with the same seed write byte-identical
files. The Corpus, the truth file, the Labelled Set, and the Nuisance Structure
manifest are projections of one plan, and the Corpus projection carries nothing
the others carry — a `CorpusItem` has no membership field, no published Contact
Point, and no nuisance field, and the writer that serialises it takes nothing but
items, so there is no field for either to leak through. A reader does not have to trust
that: they can grep the file (ADR-0008).
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

from reddit_fraud_intelligence.contacts import PublishedPost
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

# What a background post says, and where it says it. Drawn from rather than composed, for
# the reasons `_background` gives, and deliberately ordinary: these are the posts that are
# neither evidence nor nuisance, so there is nothing in them to measure and nothing in
# them that should be measured. None carries an `@` or an address, and none names a
# category the projection would score it into — they land in `Other`, which is where an
# ordinary post belongs, and a reader comparing a scaled Corpus against a 34-post one can
# see exactly which posts the extra population added.
_BACKGROUND_SUBREDDITS = ("OffTopic", "dailychat", "Misc", "general", "casual")
_BACKGROUND_TEXT = (
    (
        "Has anyone else had this slow this week?",
        "Nothing has changed on my end as far as I can tell. Posting mainly to see if it "
        "is just me.",
    ),
    (
        "Reminder to back up your own files",
        "Mine was three years old and I had forgotten what was on it. A weekly copy would "
        "have saved me an afternoon.",
    ),
    (
        "What are you all reading at the moment?",
        "Looking for something that does not need a second week to get into. I have a "
        "pile I keep starting and abandoning.",
    ),
    (
        "The weather finally broke",
        "Two days of it and I had forgotten what the street looks like in the light. That "
        "is the whole update.",
    ),
    (
        "Small win: finished the thing I had been putting off",
        "Took most of the year. Writing it down so that next year has something to point "
        "at when I feel like I have achieved nothing.",
    ),
    (
        "Anyone else reorganising their notes this month?",
        "Every spring I decide the system is wrong and rebuild it, and every year I find "
        "the previous version in a folder and use it instead.",
    ),
    (
        "Public transport was on time for once",
        "I am aware this is not a popular opinion here, but it happened and it should be "
        "recorded somewhere.",
    ),
    (
        "Looking for a mechanic I can trust",
        "Somewhere that will actually tell me what is wrong rather than replacing four "
        "parts and hoping. Recommendations welcome.",
    ),
    (
        "The building work next door finished early",
        "Six weeks of noise and then nothing. I did not realise how much I had been dreading "
        "it until it stopped.",
    ),
    (
        "First proper cold of the year",
        "Slept fourteen hours and woke up to a different season. The heating has an opinion "
        "about it too.",
    ),
    (
        "Lost my keys, found them in the fridge",
        "No idea how long. I am going to stop writing things down at two in the morning.",
    ),
    (
        "Anyone know a decent printer for occasional use?",
        "Not scanning, not copying, not doing anything clever. Just printing, occasionally, "
        "without a fight.",
    ),
)


@dataclass(frozen=True, slots=True)
class _Plan:
    posts: tuple[CorpusItem, ...]
    campaigns: tuple[PlantedCampaign, ...]
    nuisance: tuple[NuisanceRecord, ...]
    published: tuple[PublishedPost, ...]


@dataclass(frozen=True, slots=True)
class _Planted:
    """A structure the plan has filled in: each post with the identifier it was given."""

    structure: NuisanceStructure
    planted: tuple[tuple[str, SyntheticPost], ...]


def corpus_items(seed: int, background: int = 0) -> list[CorpusItem]:
    """The Corpus, and nothing about how it was planted."""
    return list(_plan(seed, background).posts)


def planted_campaigns(seed: int, background: int = 0) -> list[PlantedCampaign]:
    """The membership record the evaluator joins, after inference has finished."""
    return list(_plan(seed, background).campaigns)


def nuisance_records(seed: int, background: int = 0) -> list[NuisanceRecord]:
    """What the Corpus holds that is not a Planted Campaign, and why it is there."""
    return list(_plan(seed, background).nuisance)


def published_posts(seed: int, background: int = 0) -> list[PublishedPost]:
    """The Labelled Set: what every post publishes, and the posts that publish nothing.

    One row per post, built from what the post declared rather than from reading its
    text back for an `@`, so a label is something somebody wrote down and a disagreement
    between the two is something the measurement reports rather than something the
    projection smooths over. A post that publishes nothing is labelled as publishing
    nothing, which is the row the false-positive rate is counted against.
    """
    return list(_plan(seed, background).published)


def _plan(seed: int, background: int = 0) -> _Plan:
    window_start = _ANCHOR + timedelta(days=seed % _CORPUS_WINDOW_DAYS)
    post_ids = count(1)
    posts: list[CorpusItem] = []
    campaigns: list[PlantedCampaign] = []
    paraphrases: list[NuisanceRecord] = []
    published: list[PublishedPost] = []

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
            published.append(PublishedPost(post_id=member_post_id, published=post.published))
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
        published.append(PublishedPost(post_id=post_id, published=post.published))

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
            published.append(PublishedPost(post_id=post_id, published=post.published))
        structures.append(_Planted(structure, tuple(planted_posts)))

    # Last, so that growing the background cannot renumber anything above it: post ids
    # and the timestamps keyed on them are what the committed Corpus is held to.
    for post in _background(seed, background):
        post_id = _next_post_id(post_ids)
        posts.append(
            _item(
                post,
                post_id,
                _scattered_at(seed, post_id, window_start, _UNGROUPED_SPREAD_MINUTES),
            )
        )
        published.append(PublishedPost(post_id=post_id, published=post.published))

    return _Plan(
        posts=tuple(posts),
        campaigns=tuple(campaigns),
        published=tuple(published),
        nuisance=(
            *paraphrases,
            *(
                planted.structure.record(tuple(post_id for post_id, _ in planted.planted))
                for planted in structures
            ),
            *_shared_infrastructure(structures),
        ),
    )


def _background(seed: int, count: int) -> tuple[SyntheticPost, ...]:
    """Ordinary posts from ordinary accounts, for a Corpus larger than the committed one.

    This is what "scaling past the committed file is a matter of running it" means. The
    Planted Campaigns and the Nuisance Structure are hand-written evidence — each one is
    planted because a particular failure would otherwise go unmeasured — so they are a
    fixed set and cannot be scaled by a number. The background population is the opposite:
    it exists to make the figures harder to win by accident, and growing it is arithmetic.

    Every post here is safe by construction rather than by inspection. It carries no link,
    so it reaches no registrable domain; it names no Contact Point, so it joins no account
    to another; and its account appears once, so it cannot be a shared identifier. A
    background post therefore cannot form or join a Campaign Candidate on its own, and the
    recovery figure over a larger Corpus is measuring what it measures at 34 posts rather
    than a larger population of planted evidence that was never there. Content similarity
    is the one thing these posts can be alike in, and it corroborates and never
    establishes (ADR-0026), so a thousand identical background posts still form no group.

    The text is drawn from a fixed set of ordinary sentences rather than generated, so a
    reader can read what a scaled Corpus contains instead of trusting a template. None of
    them carries an `@` or anything address-shaped, because a background post that
    published a Contact Point would be a false positive planted by the scale parameter
    rather than by the Nuisance Structure, and would move the invented figure that
    `rfi contact-points` measures.
    """
    if count < 0:
        raise ValueError(f"a background population cannot be {count} posts")
    return tuple(_background_post(seed, index) for index in range(count))


def _background_post(seed: int, index: int) -> SyntheticPost:
    """One background post, at an index the seed decides everything else about.

    Keyed on the index rather than on a running total, so growing the population does
    not rewrite the posts that were already there — the same property `_key` gives the
    timestamps, and the reason `--background 200` is the same Corpus as `--background 0`
    with more of it rather than a different Corpus.
    """
    rng = _rng(seed, f"background:{index}")
    title, body = _BACKGROUND_TEXT[rng.randrange(len(_BACKGROUND_TEXT))]
    return SyntheticPost(
        account=f"syn_background_{index:04d}",
        subreddit=_BACKGROUND_SUBREDDITS[rng.randrange(len(_BACKGROUND_SUBREDDITS))],
        title=title,
        body=body,
        links=(),
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
