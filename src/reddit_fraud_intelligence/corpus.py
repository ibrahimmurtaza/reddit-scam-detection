"""The Corpus file: content, accounts, and links, and nothing else (ADR-0008).

The fields on `CorpusItem` are the whole vocabulary the pipeline gets to see.
There is no membership field to leak, no generator bookkeeping, and no way to
express "this post belongs to a planted group" without adding one here — which
would be visible in the file and greppable by any reader.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
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


def read_corpus(path: Path) -> tuple[CorpusItem, ...]:
    """The Corpus as the pipeline receives it: items, in the file's own order.

    Reading the file rather than the generator, because the file is the boundary.
    ADR-0001 puts ingestion behind a Corpus Provider so that nothing downstream
    assumes where content came from; until a real provider is authorised, this is
    the file the portfolio build supplies, and a reader can check that what the
    pipeline saw is what is in the repository by reading the file.

    A field the pipeline does not know is an error rather than something dropped.
    The whole claim of ADR-0008 is that the file holds no membership field, and
    silently ignoring a field that has appeared would be the first way to stop
    being able to say that. `links` is put back into a tuple because JSON has no
    such type and the field is declared as one: an item read back has to be equal
    to the item written.
    """
    vocabulary = {field.name for field in fields(CorpusItem)}
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != vocabulary:
            raise ValueError(
                f"{path}:{number} holds {sorted(record)}, which is not the Corpus "
                f"vocabulary {sorted(vocabulary)}"
            )
        items.append(CorpusItem(**{**record, "links": tuple(record["links"])}))
    return tuple(items)


