"""The truth file: Planted Campaign membership, at a path of its own (ADR-0008).

No analysis module imports this one except the evaluator, and the evaluator runs
after inference has finished. The generator writes it; `rfi campaign-recovery` joins
it against the candidates the grouping published, and reads it nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

from reddit_fraud_intelligence.jsonl import (
    JsonObject,
    read_names,
    read_object,
    read_rows,
    read_text,
    refuse_repeated,
    write_lines,
)


@dataclass(frozen=True, slots=True)
class PlantedCampaign:
    """A grouping written into the Corpus deliberately, whose membership is known
    by construction."""

    campaign_id: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]


def write_truth(path: Path, campaigns: Iterable[PlantedCampaign]) -> None:
    write_lines(path, [_object(campaign) for campaign in campaigns])


def read_truth(path: Path) -> tuple[PlantedCampaign, ...]:
    """The membership, checked row by row, because it is the answer key.

    The evaluator joins this file against what the grouping produced, and a join
    against a membership that is not the one the generator planted measures nothing.
    Four things are refused for that reason: a row holding a field nothing here knows, a
    file holding no campaign at all, which would make the figure 0 of 0, two rows
    claiming one campaign id, which would count it in the denominator twice and hide
    whatever the first one said, and a campaign naming no post.

    A membership with no post is refused rather than left to the join. A Campaign
    Candidate is built from the links its accounts' posts carry, so a campaign naming
    posts nobody wrote could never be recovered by any grouping, and a figure with a
    guaranteed zero in its denominator is a figure about the membership file rather
    than about the system.

    An unknown field is an error rather than something dropped, for the reason
    `read_corpus` gives: a field appearing here is a field somebody added to the file
    the pipeline is measured against, and ignoring it quietly is the first way to stop
    being able to say what the file holds.
    """
    campaigns = tuple(_campaign(path, number, text) for number, text in read_rows(path))
    if not campaigns:
        raise ValueError(
            f"{path.as_posix()} holds no Planted Campaign, so there is nothing to measure"
        )
    refuse_repeated(
        path.as_posix(),
        (campaign.campaign_id for campaign in campaigns),
        "the figure would count it twice",
    )
    return campaigns


def _campaign(path: Path, number: int, text: str) -> PlantedCampaign:
    where = f"{path.as_posix()}:{number}"
    record = read_object(where, text)
    vocabulary = tuple(field.name for field in fields(PlantedCampaign))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds {sorted(record)}, which is not the membership vocabulary "
            f"{sorted(vocabulary)}"
        )
    posts = read_names(where, record, "posts")
    if not posts:
        raise ValueError(f"{where} names no post, so no grouping could ever recover it")
    return PlantedCampaign(
        campaign_id=read_text(where, record, "campaign_id"),
        accounts=read_names(where, record, "accounts"),
        posts=posts,
    )


def _object(campaign: PlantedCampaign) -> JsonObject:
    return {
        "campaign_id": campaign.campaign_id,
        "accounts": list(campaign.accounts),
        "posts": list(campaign.posts),
    }
