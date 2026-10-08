"""Within-component content similarity, computed from the stored vectors.

The testable seam of ticket #24: the pure figures, taken off a mapping of post id to
the vector the store holds. Nothing here touches the database or the grouping; the
command-level shape of the Signal is covered in tests/test_campaign_candidates.py.

Two of the four reference cases on this Corpus are the Ticket's own: the staggered
paraphrases of one offer (`syn-campaign-alpha`), which must stay readable as a
candidate even though this lexical model reads them as different words, and the decoy
cluster posting one advert word for word with nothing to group on, whose text is the
nearest on the Corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reddit_fraud_intelligence.campaigns import (
    ContentSimilarity,
    within_component_similarity,
)
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.embeddings import compute_embeddings

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"

Posts = tuple[CorpusItem, ...]


def vectors_of(corpus_path: Path = COMMITTED_CORPUS) -> dict[str, tuple[float, ...]]:
    """The vectors `rfi content-embeddings` stored, recomputed the same way here.

    The model is deterministic, so the vectors the pure function sees in a test are the
    ones the store holds: the offline test and the database-backed command read the
    same numbers.
    """
    return {
        item.record.post_id: item.embedding
        for item in compute_embeddings(read_corpus(corpus_path))
    }


def posts_by_id(corpus_path: Path = COMMITTED_CORPUS) -> dict[str, CorpusItem]:
    return {item.post_id: item for item in read_corpus(corpus_path)}


def test_each_post_is_matched_to_its_nearest_post_by_a_different_account() -> None:
    """One row per post, and the row is honest about who its twin is.

    The near-twin is always a post by a different account in the same component: a
    candidate is a claim about a group, and two posts from one account are the
    account's own habits, not corroboration of a grouping.
    """
    posts = {
        item.post_id: item
        for item in read_corpus(COMMITTED_CORPUS)
        if item.post_id in ("syn_p_0012", "syn_p_0013", "syn_p_0014")
    }
    rows = within_component_similarity(
        tuple(posts[post_id] for post_id in ("syn_p_0012", "syn_p_0013", "syn_p_0014")),
        vectors_of(),
    )

    assert {row.post for row in rows} == set(posts)
    for row in rows:
        assert row.nearest != row.post
        assert row.nearest_account != row.account
        assert posts[row.nearest].account == row.nearest_account
        assert 0.0 <= row.distance <= 2.0


def test_the_decoy_cluster_text_is_the_nearest_on_the_corpus() -> None:
    """The pasted advert: the same words, three accounts, no infrastructure.

    This is the Ticket's test case at the seam: the text is near-identical, and no
    shared registrable domain or Contact Point joins it to a grouping. The similarity
    function never groups anything itself, so what it reports here is a figure a
    grouping would get wrong to act on, and never a grouping.
    """
    ids = ("syn_p_0012", "syn_p_0013", "syn_p_0014")
    rows = within_component_similarity(
        tuple(posts_by_id()[post_id] for post_id in ids), vectors_of()
    )

    assert all(row.distance < 0.25 for row in rows)


def test_the_staggered_paraphrases_do_not_read_as_the_same_words() -> None:
    """The same offer in four different words is not the same text.

    The model reads words, not meaning: a Planted Campaign that paraphrases is read as
    distant, and that is a limit to state rather than a failure to fix. The candidate
    is still recovered from its registration - this figure only deprioritises, and the
    Ticket asks that the genuine case survive the Signal, not that every genuine case
    flatter the Signal.
    """
    ids = ("syn_p_0001", "syn_p_0002", "syn_p_0003", "syn_p_0004")
    rows = within_component_similarity(
        tuple(posts_by_id()[post_id] for post_id in ids), vectors_of()
    )

    assert all(row.distance > 0.55 for row in rows)
    summary = ContentSimilarity(
        threshold=0.5,
        pieces=len(rows),
        corroborated=sum(1 for row in rows if row.distance <= 0.5),
        closest_distance=min(row.distance for row in rows),
    )
    assert summary.corroborated == 0
    assert not summary.corroborates


def test_a_candidate_whose_accounts_recycle_the_same_text_is_corroborated() -> None:
    """The one Corpus shape similarity exists for: the shop's accounts reuse the
    same updates, so every post in the candidate has a near-identical twin."""
    ids = ("syn_p_0015", "syn_p_0016", "syn_p_0017")
    rows = within_component_similarity(
        tuple(posts_by_id()[post_id] for post_id in ids), vectors_of()
    )

    assert all(row.distance <= 0.5 for row in rows)
    summary = ContentSimilarity(
        threshold=0.5,
        pieces=len(rows),
        corroborated=sum(1 for row in rows if row.distance <= 0.5),
        closest_distance=min(row.distance for row in rows),
    )
    assert summary.corroborated == 3
    assert summary.corroborates


def test_unknown_post_id_is_refused_by_name() -> None:
    """A post with no stored vector cannot be compared, and a silent skip would read
    as a candidate the vectors agreed with when they were never asked."""
    posts = tuple(posts_by_id()[post_id] for post_id in ("syn_p_0012", "syn_p_0013"))
    vectors = vectors_of()
    del vectors["syn_p_0013"]

    with pytest.raises(ValueError, match="syn_p_0013"):
        within_component_similarity(posts, vectors)


def test_content_similarity_verdict_is_a_method_of_the_threshold() -> None:
    """Corroborated is a recomputation, never a stored opinion, of the threshold."""
    rows = within_component_similarity(
        tuple(posts_by_id()[post_id] for post_id in ("syn_p_0012", "syn_p_0013")),
        vectors_of(),
    )
    wide = ContentSimilarity(
        threshold=0.5,
        pieces=len(rows),
        corroborated=sum(1 for row in rows if row.distance <= 0.5),
        closest_distance=min(row.distance for row in rows),
    )
    narrow = ContentSimilarity(
        threshold=0.01,
        pieces=len(rows),
        corroborated=sum(1 for row in rows if row.distance <= 0.01),
        closest_distance=min(row.distance for row in rows),
    )

    assert wide.corroborates
    assert not narrow.corroborates
