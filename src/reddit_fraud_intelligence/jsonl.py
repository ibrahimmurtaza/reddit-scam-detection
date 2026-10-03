"""Canonical JSON Lines.

Both files the generator writes share this, so a fixed seed yields byte-identical
output: keys sorted, no whitespace to argue about, ASCII only so the bytes do
not depend on an encoding, and LF line endings on every platform.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

JsonObject = Mapping[str, object]


def line(obj: JsonObject) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_lines(path: Path, objects: Iterable[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for obj in objects:
            handle.write(f"{line(obj)}\n")


def read_rows(path: Path) -> Iterator[tuple[int, str]]:
    """Every row of the file with the line number a reader would call it by.

    The number is the physical line rather than the ordinal of the non-blank ones,
    because the refusals every reader below it prints have to point at the line a
    reader sees in their editor: counting only the rows that parsed would move
    every complaint up a line as soon as somebody left a blank one above it. The
    text is handed over rather than parsed so each reader can say what is wrong
    with a row in the terms of its own vocabulary.
    """
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if text.strip():
            yield number, text
