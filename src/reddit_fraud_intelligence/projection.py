"""The projection as a reader can argue with, and as data the next step can read.

`categories.py` holds the argument: ten Scam Categories, the CAFC label each one
lands, the Annex E heading CAFC defines it under, and a one-line reason. This
module renders that argument beside the base rates CAFC's extract produces, so the
projection is arguable in the sense the ticket means — a reader can name a specific
merge, see the one line that justifies it, and see how many real reports sit on
either side of it.

Both outputs are generated and neither is edited by hand. The mapping is written
as JSON Lines because the next step needs the mapping itself and a Markdown table
parsed back out of a page is not an interface; the report is written because the
argument is only published if a person can read it. `rfi scam-categories` is the
only producer of both.

Every figure here is computed from the committed base rates. The projection is
not: it is a judgement, and keeping the two apart in the output is deliberate. A
reader who thinks a merge is wrong is disagreeing with a line of prose, and can
say so without having to reconstruct the arithmetic first. The reasons are quoted,
never restated, so the argument has one home.

Nothing here reads the network, and nothing here reads the annex. The annex is a
PDF whose headings are transcribed in `categories.py`; the extract is the
enumeration, and a category the projection does not account for stops the command
rather than being summarised away.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reddit_fraud_intelligence.cafc import (
    ATTRIBUTION,
    LICENCE,
    LICENCE_URL,
    RECORD_URL,
    BaseRate,
)
from reddit_fraud_intelligence.categories import (
    ANNEX_CATEGORIES,
    ANNEX_URL,
    ASSERTED_IN_ADR_0006,
    CAFC_CATEGORIES,
    EVERY_CATEGORY,
    OTHER,
    RECOVERY_PITCH,
    TOP_LEVEL_COUNT,
    CafcCategory,
    ScamCategory,
)
from reddit_fraud_intelligence.jsonl import JsonObject, write_lines


@dataclass(frozen=True, slots=True)
class Placed:
    """One CAFC category, joined to the figures the extract gives it.

    `reports` and `base_rate` are `None` for a name CAFC has published and this
    window's extract does not carry. They are not zero, and neither the report
    nor the published mapping says otherwise: a name observed at zero and a name
    never observed are different claims.
    """

    category: CafcCategory
    reports: int | None
    base_rate: float | None


def read_base_rates(path: Path) -> dict[str, BaseRate]:
    """The committed base rates, keyed by CAFC's label.

    Read rather than recomputed from the extract: the extract is 72 MiB and the
    figures in it are already held to the bytes that produced them. Raises rather
    than returns nothing, because an empty projection would render a page that
    looks like an answer.
    """
    counts: dict[str, BaseRate] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        base_rate = BaseRate(
            category=str(row["category"]),
            reports=int(row["reports"]),
            base_rate=float(row["base_rate"]),
        )
        counts[base_rate.category] = base_rate
    if not counts:
        raise ValueError(f"the base rates hold no categories: {path}")
    return counts


def place(base_rates: Mapping[str, BaseRate]) -> tuple[Placed, ...]:
    """Join the projection to the base rates, refusing a label it does not hold.

    A category added upstream lands here absent from the projection. Raising beats
    rendering a page that looks complete, because the page is the artefact a
    reader trusts to be the whole of the argument.
    """
    unplaced = sorted(set(base_rates) - {category.category for category in CAFC_CATEGORIES})
    if unplaced:
        plural = "y" if len(unplaced) == 1 else "ies"
        raise ValueError(
            f"the base rates hold {len(unplaced)} categor{plural} the projection does "
            f"not account for: {', '.join(unplaced)}. Add each to CAFC_CATEGORIES with "
            "a Scam Category or a reason for dropping it, then re-run."
        )

    def joined(category: CafcCategory) -> Placed:
        base_rate = base_rates.get(category.category)
        return Placed(
            category=category,
            reports=None if base_rate is None else base_rate.reports,
            base_rate=None if base_rate is None else base_rate.base_rate,
        )

    return tuple(joined(category) for category in CAFC_CATEGORIES)


def total(placed: Sequence[Placed], name: str) -> int:
    """The reports landing in one Scam Category, or in the Other bucket.

    A dropped category contributes nothing: its reports are counted apart and the
    report prints that count, rather than being quietly added to a Scam Category
    that does not describe them.
    """
    return sum(
        item.reports or 0
        for item in placed
        if item.category.scam_category == name and not item.category.dropped
    )


def unplaced_reports(placed: Sequence[Placed]) -> int:
    """The reports that land in no Scam Category because their category was dropped."""
    return sum(item.reports or 0 for item in placed if item.category.dropped)


def write_mapping(path: Path, placed: Sequence[Placed]) -> None:
    """The mapping as data, sorted by CAFC's label so the bytes are stable.

    Each row carries the `covers` line of the Scam Category it lands in, so a row
    read on its own says what its class means. The repetition costs a little
    space and buys a file that needs no companion and no Markdown table beside it.
    """

    def objects() -> Iterator[JsonObject]:
        for item in sorted(placed, key=lambda item: item.category.category):
            yield {
                "category": item.category.category,
                "annex": item.category.annex,
                "scam_category": item.category.scam_category,
                "covers": _covers(item.category.scam_category),
                "dropped": item.category.dropped,
                "rationale": item.category.rationale,
                "reports": item.reports,
                "base_rate": item.base_rate,
            }

    write_lines(path, objects())


def render_report(placed: Sequence[Placed]) -> str:
    """The reader-facing page, generated from the projection and the figures."""
    reports = sum(item.reports or 0 for item in placed)
    ordered = _by_size(placed)
    summary = "\n".join(_summary_row(scam, placed, reports) for scam in ordered)
    sections = "\n\n".join(_section(scam, placed, reports) for scam in ordered)
    dropped_items = [item for item in placed if item.category.dropped]
    dropped = "\n".join(
        f"- **{item.category.category}** — {item.category.rationale}"
        for item in dropped_items
    )

    other = total(placed, OTHER.name)
    unplaced = unplaced_reports(placed)
    recovery = _find(placed, RECOVERY_PITCH)
    observed = [item for item in placed if item.reports is not None]
    absent = sorted(item.category.category for item in placed if item.reports is None)
    undefined = sorted(item.category.category for item in placed if item.category.annex is None)

    return f"""# Scam Categories

Generated by `rfi scam-categories` from the committed CAFC base rates. Do not edit it by hand —
a test holds this file to what the projection produces, and re-running the command
rewrites it byte for byte.

## What this is

{TOP_LEVEL_COUNT} Scam Categories, projected from the {len(CAFC_CATEGORIES)} thematic categories
CAFC has published, plus an Other bucket for content that fits none of them. ADR-0006
asked for a documented top-level projection rather than a hand-picked list, so the
projection is the whole of the page: every CAFC category is placed in a Scam Category,
sits in Other, or is dropped with a reason, and every placement carries a one-line
reason. A reader who thinks a particular merge is wrong can find it below, read the
line that justifies it, and see how many real reports sit on either side of it.

The judgement is written once, in `categories.py`, and read from there; the figures are
computed from the committed base rates, and the report quotes the reasons rather than
restating them. The same mapping is published as `data/cafc/scam-categories.jsonl` for
whatever reads it next. Nothing here renames CAFC: a CAFC category is a pitch a reporter
recognised in a form they were handed to fill in, a Scam Category is a class a pipeline
can rank content under, and the projection is many-to-one in that direction without
inventing a label.

## Where the {ASSERTED_IN_ADR_0006} in ADR-0006 went

**{ASSERTED_IN_ADR_0006} is not a number CAFC publishes, and it is corrected here rather than
reproduced.** CAFC publishes its thematic categories in two places, and they do not agree
on how many there are. This release of the extract enumerates **{len(observed)} values**.
CAFC's Annex E defines **{len(ANNEX_CATEGORIES)} headings**, and those are not the same
granularity: the annex defines Grant and Loan together, and defines Identity Theft and
Identity Fraud as two halves of one pitch, while the extract carries each as a value a
reporter can pick.

The two sets overlap, and neither contains the other. Of the {len(observed)} values the
extract enumerates, {len(observed) - len(undefined)} are defined in the annex and
{len(undefined)} are not — {", ".join(undefined)}. The annex defines one name the extract
does not carry: {", ".join(absent)}. The union of names CAFC has published is therefore
{len(CAFC_CATEGORIES)}, and CAFC has published no {ASSERTED_IN_ADR_0006}st.

Two things follow, and they are why the numbers above are not one number. The extract is
the enumeration this project consumes, so it is what the mapping is guaranteed against: a
new value in a new release stops the command until the projection accounts for it. The
annex headings are read by hand from a cited PDF, because nothing here parses a PDF, so
they supply the *definitions* the reasons argue from and are a transcription rather than
a measurement. {", ".join(absent)} is the one published name this release does not carry.
Its base rate is neither zero nor known: it is absent from the release, not observed at
zero, and it is placed below on the strength of the annex rather than on a measurement.

Annex E: <{ANNEX_URL}>

Open Government Portal record: <{RECORD_URL}>

Attribution: {ATTRIBUTION}

Licence: [{LICENCE}]({LICENCE_URL})

## What each Scam Category lands

A base rate here is a share of the {reports:,} reports in the extract, and of reports rather
than of losses. It answers "how often does this kind of fraud get reported in Canada",
which is the only thing a prior can be read off. The ten are listed by what they land;
Other is listed with them because its size is the finding, not because it is one of the ten.

| Scam Category | Reports | Base rate |
| --- | ---: | ---: |
{summary}

## Why the count is {TOP_LEVEL_COUNT}

The count is argued, not hit. Nine would have had to fold Work and Payroll into a
neighbour, losing the fact that the victim is between posts rather than shopping,
investing, or being phished. Eleven would have had to give Extortion a neighbour too,
which is the one CAFC category with no con in it, or give Charity / Donation a class of
its own at well under a tenth of a percent of reports. Twelve would have separated the
two things CAFC's own annex calls subsections — Spoofing within Spear Phishing, and
Directory within False Billing — and a projection that undoes CAFC's own structure is not
simpler, only longer.

The merge a reader is most likely to contest is Charity / Donation sitting with the
prizes. Its line is under that heading and the argument is that a donation pitch is the
same shape as a prize pitch with the direction reversed, and too small to carry a class of
its own.

## What lands where, and why

{sections}

## Dropped, with the reason

{len(dropped_items)} of CAFC's categories are not kinds of fraud, and dropping them into a
neighbour would put reports into a class that does not describe them. They are counted
here instead, because {unplaced / reports:.2%} of real reports landing in no Scam Category at all is
a finding about the projection and not a rounding error.

{dropped}

## What falls into no Scam Category

**Other holds {other:,} reports, {other / reports:.2%} of the extract.** CAFC files it for a
report its own analysts could not describe any other way, which is the same position
this projection is in when it cannot place content. ADR-0006 makes the bucket's size a
standing measure of what the system fails to represent, so it is reported rather than
absorbed into a neighbour.

Adding the dropped categories, {other + unplaced:,} reports — {(other + unplaced) / reports:.2%}
— land in no Scam Category. Every other report is accounted for exactly once.

## The one category that exists because a victim is targeted twice

**{recovery.category.category} holds {recovery.reports:,} reports, {recovery.base_rate:.2%} of the
extract.** CAFC documents it as a category of its own because after someone has been
victimised, a fraudster may approach them again offering to recover the money. Folding it
into the con-based category that most resembles it would erase the only record CAFC keeps
that the second contact is a pattern rather than an accident, and it is the reason
ADR-0006 rejected a hand-picked list of ten. It is placed in
{recovery.category.scam_category}.

{recovery.category.rationale}
"""


def _by_size(placed: Sequence[Placed]) -> list[ScamCategory]:
    """The ten and the bucket, largest share first, ties by name."""
    return sorted(EVERY_CATEGORY, key=lambda scam: (-total(placed, scam.name), scam.name))


def _summary_row(scam: ScamCategory, placed: Sequence[Placed], reports: int) -> str:
    width = max(len(each.name) for each in EVERY_CATEGORY)
    landing = total(placed, scam.name)
    return f"| {scam.name:<{width}} | {landing:>9,} | {landing / reports:>8.2%} |"


def _section(scam: ScamCategory, placed: Sequence[Placed], reports: int) -> str:
    members = sorted(
        (item for item in placed if item.category.scam_category == scam.name),
        key=lambda item: (-(item.reports or 0), item.category.category),
    )
    width = max(len(item.category.category) for item in members)
    table = "\n".join(
        f"| {item.category.category:<{width}} | {_reports(item.reports)} | "
        f"{_base_rate(item, reports)} |"
        for item in members
    )
    reasons = "\n".join(
        f"- **{item.category.category}** — {item.category.rationale}" for item in members
    )
    heading = (
        f"{scam.name}, the bucket rather than one of the ten" if scam is OTHER else scam.name
    )
    return f"""### {heading}

{scam.covers}

| CAFC category | Reports | Base rate |
| --- | ---: | ---: |
{table}

{reasons}"""


def _covers(name: str | None) -> str | None:
    """What the named Scam Category covers, or `None` for a dropped row.

    Raises on a name that is neither: `place` has already refused a base rate the
    projection does not hold, so an unrecognised landing is a mistake in the
    projection rather than something to publish as an empty string.
    """
    if name is None:
        return None
    for scam in EVERY_CATEGORY:
        if scam.name == name:
            return scam.covers
    raise ValueError(f"the projection names no Scam Category called {name!r}")


def _reports(reports: int | None) -> str:
    return "absent" if reports is None else f"{reports:>7,}"


def _base_rate(item: Placed, reports: int) -> str:
    if item.reports is None:
        return "not in this release"
    return f"{item.reports / reports:>8.2%}"


def _find(placed: Sequence[Placed], category: str) -> Placed:
    """The one category the report singles out by name rather than by count."""
    found = [item for item in placed if item.category.category == category]
    if not found:
        raise ValueError(f"the projection holds no {category!r} to single out")
    return found[0]
