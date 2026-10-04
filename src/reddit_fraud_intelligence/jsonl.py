"""Canonical JSON Lines, read and written.

Both files the generator writes share this, so a fixed seed yields byte-identical
output: keys sorted, no whitespace to argue about, ASCII only so the bytes do
not depend on an encoding, and LF line endings on every platform.

Reading is here too, and for the same reason. Every file the project reads is a
JSON Lines file and every one of them is checked row by row, so the shape of a
check — walk the rows, parse the row, refuse what this file's own vocabulary cannot
account for — is written once. What is left for each caller is the vocabulary itself:
which fields a candidate row holds, which fields are names, which value is a date.
Those differ per file and belong beside the file they describe; the walking and the
parsing do not.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from enum import StrEnum
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
    because the refusals the readers below it print have to point at the line a reader
    sees in their editor: counting only the rows that parsed would move every
    complaint up a line as soon as somebody left a blank one above it. The text is
    handed over rather than parsed so each reader can say what is wrong with a row in
    the terms of its own vocabulary.
    """
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if text.strip():
            yield number, text


def read_object(where: str, text: str) -> JsonObject:
    """One row, parsed, refusing a line that is not JSON and one that is not a row.

    A JSON Lines file that has grown a line holding an array or a bare string is a file
    the reader below this call cannot account for, and saying so is cheaper than a
    `TypeError` from three lines further down.
    """
    try:
        record = json.loads(text)
    except json.JSONDecodeError as refusal:
        raise ValueError(f"{where} is not JSON: {text!r}") from refusal
    if not isinstance(record, dict):
        raise ValueError(f"{where} is not a row: {text!r}")
    return record


def read_names(where: str, record: JsonObject, field: str) -> tuple[str, ...]:
    """One field of names, in the file's own order, and checked three ways.

    The order is the file's rather than sorted, so a refusal quotes a membership as it
    was written rather than as this code rearranged it. A value that is not a list of
    text and a name written twice are both refused: the first because the reader cannot
    tell what the row meant, and the second because every count built from the field
    would double one of its entries — which is exactly how a duplicate membership or a
    duplicated account would quietly become two things.
    """
    value = record[field]
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{where} has {field}={value!r}, which is not a list of names")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} names one thing twice in {field}")
    return tuple(value)


def read_text(where: str, record: JsonObject, field: str) -> str:
    """One field that has to be there and has to say something.

    Blank is refused as much as absent: an empty identifier or a blank date is a
    mistake no reader could see in the output, because the row still parses.
    """
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} has no {field}")
    return value


def read_vocabulary[Name: StrEnum](
    where: str, field: str, value: object, vocabulary: type[Name]
) -> Name:
    """One value of a closed vocabulary, refused by name rather than by a `ValueError`.

    A kind a build does not read, or a way of writing a measurement it does not know, is
    a row the figure beside it cannot count, so the refusal names every name that would
    have worked. That is the difference between a file somebody can fix and a file
    somebody has to guess about, and it is why the vocabulary is a parameter rather than
    written here: the members are each file's own, the shape of the check is this one's.
    """
    if not isinstance(value, str):
        raise ValueError(f"{where} has {field}={value!r}, which is not a name")
    try:
        return vocabulary(value)
    except ValueError as unknown:
        names = ", ".join(member.value for member in vocabulary)
        raise ValueError(
            f"{where} has {field}={value!r}, and the names are {names}"
        ) from unknown


def refuse_repeated(where: str, values: Iterable[str], consequence: str) -> None:
    """Refuse a file holding one name on two rows, saying what that would cost.

    The consequence is the caller's because it is the caller's: a duplicated candidate
    identifier would be counted twice by a join, a duplicated campaign identifier twice
    by the figure's denominator, and a duplicated nuisance identifier twice by the
    baseline printed beside it. Three files, three counts, one rule.
    """
    names = list(values)
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f"{where} names {twice} on two rows each, and {consequence}")
