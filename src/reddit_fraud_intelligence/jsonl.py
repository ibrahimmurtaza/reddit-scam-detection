"""Canonical JSON Lines.

Both files the generator writes share this, so a fixed seed yields byte-identical
output: keys sorted, no whitespace to argue about, ASCII only so the bytes do
not depend on an encoding, and LF line endings on every platform.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

JsonObject = Mapping[str, object]


def line(obj: JsonObject) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_lines(path: Path, objects: Iterable[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for obj in objects:
            handle.write(f"{line(obj)}\n")
