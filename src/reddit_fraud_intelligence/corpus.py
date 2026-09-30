"""The Corpus file: content, accounts, and links, and nothing else (ADR-0008).

The fields on `CorpusItem` are the whole vocabulary the pipeline gets to see.
There is no membership field to leak, no generator bookkeeping, and no way to
express "this post belongs to a planted group" without adding one here — which
would be visible in the file and greppable by any reader.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from reddit_fraud_intelligence.jsonl import JsonObject, write_lines


@dataclass(frozen=True, slots=True)
class CorpusItem:
    """One post, its account, and its links — the whole of what the pipeline sees.

    `created_at` is RFC 3339 in UTC, so lexical order is chronological order,
    which is what the writer below sorts on.
    """

    post_id: str
    account: str
    subreddit: str
    title: str
    body: str
    created_at: str
    links: tuple[str, ...]


def write_corpus(path: Path, items: Iterable[CorpusItem]) -> None:
    """Write the Corpus. Takes items and nothing else, in any order."""

    def objects() -> Iterable[JsonObject]:
        for item in sorted(items, key=lambda item: (item.created_at, item.post_id)):
            yield asdict(item)

    write_lines(path, objects())


