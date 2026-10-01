"""The CAFC extract: the real-world base rates the Corpus is compared against.

ADR-0006 chose the Canadian Anti-Fraud Centre because it is the one source of
real, analyst-reviewed fraud reports that is freely downloadable under a licence
permitting this use. This module fetches that extract, caches it, and computes
the base rate of each of its thematic categories. Deciding where a category lands
in a Scam Category is `categories.py`, and nothing here does that.

Every figure reported is computed by reading the cached extract, so a figure and
the bytes it came from cannot drift apart. Nothing is transcribed by hand, and
re-running the command over an unchanged cache rewrites identical files.

CAFC's own filename calls this an extract, which is the word used here. It is
one of a reporting system's output, not something the pipeline is fed: the
Corpus remains synthetic, and this file only ever supplies a prior.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import urllib.request
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from reddit_fraud_intelligence.jsonl import JsonObject, write_lines

RECORD_URL = (
    "https://open.canada.ca/data/en/dataset/6a09c998-cddb-4a22-beff-4dca67ab892f"
)
SOURCE_URL = (
    "https://open.canada.ca/data/dataset/6a09c998-cddb-4a22-beff-4dca67ab892f"
    "/resource/43c67af5-e598-4a9b-a484-fe1cb5d775b5/download"
    "/cafc-open-gouv-database-2021-01-01-to-2025-09-30-extracted-2025-10-01.csv"
)
USER_AGENT = (
    "reddit-fraud-intelligence/0.1 (base rates for a documented projection)"
)
LICENCE = "Open Government Licence - Canada"
LICENCE_URL = "https://open.canada.ca/data/en/open-government-licence-canada"
ATTRIBUTION = (
    "Canadian Anti-Fraud Centre, Fraud Reporting System reporting extract, Royal "
    "Canadian Mounted Police (Government of Canada), used unchanged under the "
    "Open Government Licence - Canada."
)

# CAFC publishes its thematic categories in two places — the annex and the
# values in a release — and the two do not agree on a count. This module measures
# one of them and says so; reconciling the two, and correcting the figure ADR-0006
# asserted, belongs to `categories.py` and the report it renders.
ANNEX_NOT_MEASURED_HERE = (
    "CAFC's annex defines its own set of headings, at a different granularity "
    "again, and the two sets do not contain one another."
)

# CAFC's longest published label is 62 characters — the French expansion of
# "Emergency (Jail, Accident, Hospital, Help)". A bound of 64 therefore separates an
# enumerated label, possibly translated, from prose. Nothing here claims to know what
# CAFC meant by a column: it claims only that no value in the extract is long enough to
# be a sentence. Crossing the bound withdraws the "no free-text field" statement rather
# than inverting it, because a longer translated label is a likelier cause than a new
# text column and must not be reported as one.
MAX_ENUMERATED_VALUE_CHARS = 64

CATEGORY_COLUMN = "Fraud and Cybercrime Thematic Categories"
DATE_COLUMN = "Date Received"
_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class ExtractFacts:
    """What reading the cached extract establishes, stated as a claim about bytes."""

    attribution: str
    categories: int
    date_from: str
    date_to: str
    free_text_ruled_out: bool
    licence: str
    licence_url: str
    longest_value_chars: int
    record_url: str
    reports: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BaseRate:
    """One thematic category's share of every report in the extract."""

    category: str
    reports: int
    base_rate: float


def download_extract(destination: Path) -> int:
    """Cache the extract from CAFC, compressed. Returns the uncompressed size.

    Streamed rather than read whole, since the payload is 72 MiB. Written through
    `mtime=0` so that re-fetching unchanged bytes rewrites an identical file and
    leaves the working tree clean.
    """
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with urllib.request.urlopen(request) as response, destination.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
        ) as cache:
            while chunk := response.read(_CHUNK_BYTES):
                written += len(chunk)
                cache.write(chunk)
    return written


def digest(path: Path) -> str:
    """SHA-256 of the uncompressed CSV, which is what the figures were read from.

    Not of the cached file: gzip output is not stable across compression settings,
    so a digest of it would report a change where nothing about the extract moved.
    """
    checksum = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def survey(path: Path) -> tuple[ExtractFacts, tuple[BaseRate, ...]]:
    """Read the cached extract once and derive every figure the project records."""
    counts: Counter[str] = Counter()
    dates: list[str] = []
    longest = 0
    reports = 0

    with _open(path) as text:
        reader = csv.DictReader(text)
        category_field, date_field = _locate(reader.fieldnames)
        for record in reader:
            reports += 1
            counts[_value(record, category_field)] += 1
            dates.append(_value(record, date_field))
            for value in record.values():
                if value is not None:
                    longest = max(longest, len(value))

    if not reports:
        raise ValueError(f"the cached extract holds no reports: {path}")

    facts = ExtractFacts(
        attribution=ATTRIBUTION,
        categories=len(counts),
        date_from=min(dates),
        date_to=max(dates),
        free_text_ruled_out=longest <= MAX_ENUMERATED_VALUE_CHARS,
        licence=LICENCE,
        licence_url=LICENCE_URL,
        longest_value_chars=longest,
        record_url=RECORD_URL,
        reports=reports,
        sha256=digest(path),
    )
    base_rates = tuple(
        BaseRate(category=category, reports=count, base_rate=round(count / reports, 6))
        for category, count in sorted(counts.items(), key=_by_size_then_name)
    )
    return facts, base_rates


def write_provenance(path: Path, facts: ExtractFacts) -> None:
    write_lines(path, (asdict(facts),))


def write_base_rates(path: Path, base_rates: Sequence[BaseRate]) -> None:
    def objects() -> Iterator[JsonObject]:
        for base_rate in base_rates:
            yield asdict(base_rate)

    write_lines(path, objects())


def render_report(facts: ExtractFacts, base_rates: Sequence[BaseRate]) -> str:
    """The reader-facing report, generated from the figures rather than written."""
    width = max(len(base_rate.category) for base_rate in base_rates)
    table = "\n".join(
        f"| {base_rate.category:<{width}} | {base_rate.reports:>9,} | "
        f"{base_rate.base_rate:>8.2%} |"
        for base_rate in base_rates
    )
    if facts.free_text_ruled_out:
        limitation = (
            "**There is no free-text field.** No column of this extract can hold a\n"
            f"sentence: the longest value anywhere in it is {facts.longest_value_chars} characters, against a bound\n"
            f"of {MAX_ENUMERATED_VALUE_CHARS} that an enumerated label — even one translated into French — fits\n"
            "inside. Every column is therefore a label, a number, or a date. **That is why\n"
            "it cannot validate a text classifier.** It fixes what the Scam Categories\n"
            "are built on and what the priors are, and it says nothing whatever about\n"
            "whether text is scored correctly. Held-out validation on labelled text is a\n"
            "separate thing this extract cannot supply."
        )
    else:
        limitation = (
            "**No claim is made here that this extract has no free-text field.** A value\n"
            f"reaches {facts.longest_value_chars} characters, past the {MAX_ENUMERATED_VALUE_CHARS}-character bound an\n"
            "enumerated label fits inside, so the shape of the columns can no longer be\n"
            "read off their length. That may be a longer translated label rather than a text\n"
            "column, and nothing here distinguishes the two. What this release can and\n"
            "cannot establish about a text classifier is therefore unstated, not assumed."
        )

    return f"""# CAFC base rates

Generated by `rfi cafc-report` from the cached extract. Do not edit it by hand —
a test holds this file to what the cache produces, and re-running the command
rewrites it byte for byte.

## Provenance

| | |
|---|---|
| Reports | {facts.reports:,} |
| Date received | {facts.date_from} to {facts.date_to} |
| Thematic categories present | {facts.categories} |
| Longest value in any column | {facts.longest_value_chars} characters |
| SHA-256 of the uncompressed CSV | `{facts.sha256}` |

Attribution: {facts.attribution}

Licence: [{facts.licence}]({facts.licence_url})

Open Government Portal record: <{facts.record_url}>

CAFC publishes this extract under a licence that permits the use made of it here.
It is redistributed compressed and otherwise unchanged, and the digest above is of
the uncompressed file every figure below was read from. Re-fetching from the record
and comparing digests is how a reader confirms the cache is the release these figures
come from.

## What this cannot do

{limitation}

This release enumerates **{facts.categories} thematic category values**, and that is the whole
of what it measures. {ANNEX_NOT_MEASURED_HERE}

The projection in `docs/scam-categories.md` accounts for every name CAFC has
published across both sources, states how the counts reconcile, and corrects the
figure ADR-0006 asserted. The base rate of the one name this release does not
carry is neither zero nor known: it is absent from the release, not observed at
zero. A category CAFC dropped and one CAFC merely had no reports of are not the
same thing, and the projection keeps them apart.

## Base rate of each thematic category

A base rate here is a category's share of all {facts.reports:,} reports in the
extract, not its share of reported losses. It answers "how often does this kind of
fraud get reported here", which is the only question a prior can be read off.

| Category | Reports | Base rate |
| --- | ---: | ---: |
{table}
"""


def _open(path: Path) -> io.TextIOWrapper:
    """The cached extract as text. CAFC writes UTF-8 with a byte order mark."""
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8-sig", newline="")


def _locate(fieldnames: Sequence[str] | None) -> tuple[str, str]:
    """Resolve the columns we read, failing on a rename rather than reading nothing.

    CAFC heads each column with its English name, then a slash, then its French
    translation, so the header line is not the name to look up by.
    """
    names = fieldnames or ()
    return _field(names, CATEGORY_COLUMN), _field(names, DATE_COLUMN)


def _field(names: Sequence[str], wanted: str) -> str:
    index = next((i for i, name in enumerate(names) if name.startswith(wanted)), None)
    if index is None:
        raise ValueError(f"the extract has no {wanted!r} column: {list(names)}")
    return names[index]


def _value(record: Mapping[str, str | None], field: str) -> str:
    value = record.get(field)
    if value is None:
        raise ValueError(f"a report is missing its {field!r} value")
    return value


def _by_size_then_name(item: tuple[str, int]) -> tuple[int, str]:
    """Largest first, ties by name, so the ordering never depends on the input."""
    category, count = item
    return -count, category
