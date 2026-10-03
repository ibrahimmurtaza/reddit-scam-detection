"""The truth file: Planted Campaign membership, at a path of its own (ADR-0008).

No analysis module imports this one except the evaluator, and the evaluator runs
after inference has finished. The generator writes it; `rfi campaign-recovery` joins
it against the candidates the grouping published, and reads it nowhere else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

from reddit_fraud_intelligence.jsonl import JsonObject, read_rows, write_lines


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
    Three things are refused for that reason: a row holding a field nothing here
    knows, a file holding no campaign at all, which would make the figure 0 of 0,
    and two rows claiming one campaign id, which would count it in the denominator
    twice and hide whatever the first one said.

    A membership with no posts is refused as well. A Campaign Candidate is built
    from the links its accounts' posts carry, so a campaign naming accounts that
    wrote nothing could never be recovered by any grouping, and a figure with a
    guaranteed zero in its denominator is a figure about the membership file rather
    than about the system.

    An unknown field is an error rather than something dropped, for the reason
    `read_corpus` gives: a field appearing here is a field somebody added to the
    file the pipeline is measured against, and ignoring it quietly is the first way
    to stop being able to say what the file holds.
    """
    campaigns = tuple(_campaign(path, number, text) for number, text in read_rows(path))
    if not campaigns:
        raise ValueError(
            f"{path.as_posix()} holds no Planted Campaign, so there is nothing to measure"
        )
    names = [campaign.campaign_id for campaign in campaigns]
    if len(set(names)) != len(names):
        repeated = sorted(name for name in set(names) if names.count(name) > 1)
        raise ValueError(
            f"{path.as_posix()} names {repeated} on two rows each, so the figure would "
            "count them twice"
        )
    return campaigns


def _campaign(path: Path, number: int, text: str) -> PlantedCampaign:
    where = f"{path.as_posix()}:{number}"
    record = _record(where, text)
    vocabulary = tuple(field.name for field in fields(PlantedCampaign))
    if set(record) != set(vocabulary):
        raise ValueError(
            f"{where} holds {sorted(record)}, which is not the membership vocabulary "
            f"{sorted(vocabulary)}"
        )
    accounts = _names(where, record, "accounts")
    posts = _names(where, record, "posts")
    if not posts:
        raise ValueError(f"{where} names no post, so no grouping could ever recover it")
    campaign_id = record["campaign_id"]
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError(f"{where} has no campaign_id")
    return PlantedCampaign(campaign_id=campaign_id, accounts=accounts, posts=posts)


def _record(where: str, text: str) -> JsonObject:
    try:
        record = json.loads(text)
    except json.JSONDecodeError as refusal:
        raise ValueError(f"{where} is not JSON: {text!r}") from refusal
    if not isinstance(record, dict):
        raise ValueError(f"{where} is not a row: {text!r}")
    return record


def _names(where: str, record: JsonObject, field_name: str) -> tuple[str, ...]:
    """One field of names, as a tuple in the file's own order and checked.

    Order is kept rather than sorted, so a refusal about a membership that has been
    written down twice in one row names the file's own row rather than a version of
    it this code rearranged.
    """
    value = record[field_name]
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{where} has {field_name}={value!r}, which is not a list of names")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} names one thing twice in {field_name}")
    return tuple(value)


def _object(campaign: PlantedCampaign) -> JsonObject:
    return {
        "campaign_id": campaign.campaign_id,
        "accounts": list(campaign.accounts),
        "posts": list(campaign.posts),
    }