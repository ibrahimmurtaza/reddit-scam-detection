"""The truth file: Planted Campaign membership, at a path of its own (ADR-0008).

No analysis module imports this one. The generator writes it; the evaluator will
join it, and only after inference has finished.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from reddit_fraud_intelligence.jsonl import JsonObject, write_lines


@dataclass(frozen=True, slots=True)
class PlantedCampaign:
    """A grouping written into the Corpus deliberately, whose membership is known
    by construction."""

    campaign_id: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]


def write_truth(path: Path, campaigns: Iterable[PlantedCampaign]) -> None:
    write_lines(path, [_object(campaign) for campaign in campaigns])


def _object(campaign: PlantedCampaign) -> JsonObject:
    return {
        "campaign_id": campaign.campaign_id,
        "accounts": list(campaign.accounts),
        "posts": list(campaign.posts),
    }
