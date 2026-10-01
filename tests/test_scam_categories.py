"""The projection is a published argument, so the tests read the argument.

Seam under test: the `scam-categories` command, observed through the two files it
writes, and the projection it is generated from, read as the module's own
constants. Nothing here inspects the code that wrote them.

Three kinds of test live here. Most run the command over the committed base rates
and read what comes out, because the point is that a reader can argue with a
specific merge from the page. Some read the projection directly, because the claim
being defended is about the shape of the argument rather than about the rendering
of it: that every name CAFC has published is accounted for, that a name added
upstream cannot slip through, and that nothing is dropped quietly. And one
compares the committed files to a fresh run, so a hand-edited page fails.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from reddit_fraud_intelligence.categories import (
    ANNEX_CATEGORIES,
    ASSERTED_IN_ADR_0006,
    CAFC_CATEGORIES,
    OTHER,
    SCAM_CATEGORIES,
    TOP_LEVEL_COUNT,
)
from reddit_fraud_intelligence.cli import main

REPO_ROOT = Path(__file__).parent.parent

COMMITTED_BASE_RATES = REPO_ROOT / "data" / "cafc" / "base_rates.jsonl"
COMMITTED_MAPPING = REPO_ROOT / "data" / "cafc" / "scam-categories.jsonl"
COMMITTED_REPORT = REPO_ROOT / "docs" / "scam-categories.md"

# The category CAFC's annex documents as a pitch for a second approach, and the
# one the base rate of 2,060 reports belongs to. The ticket named the count and
# not the category, so the test names both: a projection that quietly folds this
# into something else has lost the one CAFC category that exists because a victim
# is targeted twice. Spelled out here rather than imported, so the test does not
# pass by agreeing with the module about which category it means.
RECOVERY_PITCH = "Recovery Pitch"
RECOVERY_PITCH_REPORTS = 2_060


def run_report(directory: Path, base_rates: Path = COMMITTED_BASE_RATES) -> tuple[Path, Path]:
    """Run the command, writing both outputs into `directory`.

    Both paths are given explicitly: the defaults point into the working tree, and
    a test that rewrites a committed artefact is a test that can hide a drift.
    """
    mapping = directory / "scam-categories.jsonl"
    report = directory / "scam-categories.md"
    exit_code = main(
        [
            "scam-categories",
            "--base-rates",
            str(base_rates),
            "--mapping",
            str(mapping),
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0
    return mapping, report


def _mapping_rows() -> list[Mapping[str, object]]:
    return rows(COMMITTED_MAPPING)


def rows(path: Path) -> list[Mapping[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def number(row: Mapping[str, object], field: str) -> int:
    value = row[field]
    assert isinstance(value, int), f"{field} should be a whole number, is {value!r}"
    return value


def text(row: Mapping[str, object], field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} should be text, is {value!r}"
    return value


def maybe_text(row: Mapping[str, object], field: str) -> str | None:
    """A field that is text or nothing at all, rather than text or a number."""
    value = row[field]
    if value is not None:
        assert isinstance(value, str), f"{field} should be text or nothing, is {value!r}"
    return value


def committed() -> dict[str, int]:
    return {
        text(row, "category"): number(row, "reports") for row in rows(COMMITTED_BASE_RATES)
    }


def test_every_category_in_the_extract_is_mapped_or_dropped_with_a_reason() -> None:
    """The acceptance criterion, checked against the file the project consumes.

    ADR-0006 and the ticket both said CAFC publishes 41 thematic categories. It
    does not: this release of the extract enumerates 39 values and CAFC's annex
    defines 35 headings at a different granularity, so the union of names CAFC
    has published is 40. The extract is the enumeration the mapping is guaranteed
    against, and these assertions hold for the release committed here — adopting a
    new one is a deliberate act, because `fetch-cafc` refuses to replace a cache
    that is already committed, and the projection is revisited at that point.
    """
    published = {category.category for category in CAFC_CATEGORIES}
    observed = set(committed())

    assert observed <= published, (
        f"the extract holds categories the projection does not account for: "
        f"{sorted(observed - published)}"
    )
    assert published - observed == {"Identity Theft"}, (
        "the projection must hold every name CAFC has published, and name the ones "
        "this window's extract does not carry"
    )
    assert len(published) == 40, f"the projection holds {len(published)} names, not 40"
    assert len(ANNEX_CATEGORIES) == 35, f"the annex holds {len(ANNEX_CATEGORIES)} headings"
    assert ASSERTED_IN_ADR_0006 not in {len(published), len(observed), len(ANNEX_CATEGORIES)}


def test_a_name_this_release_lacks_is_never_given_a_figure() -> None:
    """Absent is not zero, and the published mapping must not print it as one.

    The report already says "not in this release" for the one published name the
    extract does not carry. The data file is read by whatever comes next, so it
    has to say the same thing: `null` for the reports and the base rate, not `0`.
    """
    absent = "Identity Theft"
    assert absent not in committed()

    written = {text(row, "category"): row for row in _mapping_rows()}
    assert written[absent]["reports"] is None
    assert written[absent]["base_rate"] is None


def test_every_heading_the_annex_defines_is_named_by_a_category() -> None:
    """The annex is read, not summarised, so a heading cannot go missing quietly.

    Checked both ways: nothing is named that the annex does not define, and
    nothing the annex defines goes unnamed. A projection that quietly stopped
    consulting the annex would fail the second assertion rather than the first.
    """
    named = {category.annex for category in CAFC_CATEGORIES}
    assert None in named, "the projection should record the two labels the annex omits"
    assert named - {None} == ANNEX_CATEGORIES

    undefined = sorted(c.category for c in CAFC_CATEGORIES if c.annex is None)
    assert undefined == ["Credit Card", "Telecom Fraud"]


def test_a_category_added_upstream_cannot_be_silently_ignored(tmp_path: Path) -> None:
    """The failure this test exists to prevent, exercised rather than described.

    A new release brings a new label. It is absent from the projection, so the
    command refuses to render a page that would quietly leave it out — rather
    than rendering one that looks complete and is not.
    """
    base_rates = tmp_path / "base-rates.jsonl"
    base_rates.write_text(
        COMMITTED_BASE_RATES.read_text(encoding="utf-8")
        + '{"base_rate":0.000001,"category":"Sunset Timeshare","reports":1}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "scam-categories",
                "--base-rates",
                str(base_rates),
                "--mapping",
                str(tmp_path / "mapping.jsonl"),
                "--report",
                str(tmp_path / "scam-categories.md"),
            ]
        )

    assert "Sunset Timeshare" in str(refusal.value)
    assert not (tmp_path / "scam-categories.md").exists(), "a refusal must write nothing"


def test_every_placement_carries_a_one_line_rationale_and_a_drop_carries_a_reason() -> None:
    """A CAFC category that exists is either placed or explained, never lost.

    Both halves of the ticket's second and third criteria in one place, because
    they are one property: a placement states why, and a drop states why not. A
    rationale too short to argue from fails here too — the ticket asks for a
    written rationale a reader can disagree with, not a label.
    """
    for category in CAFC_CATEGORIES:
        rationale = category.rationale.strip()
        assert rationale, f"{category.category} carries no rationale"
        assert len(rationale.splitlines()) == 1, (
            f"{category.category} explains itself in more than one line"
        )
        assert len(rationale) >= 40, (
            f"{category.category} has a rationale too short to argue from: {rationale!r}"
        )
        if category.dropped:
            assert category.scam_category is None, (
                f"{category.category} is dropped and also placed in a Scam Category"
            )
        else:
            assert category.scam_category is not None, (
                f"{category.category} is placed but names no Scam Category"
            )


def test_every_merge_carries_a_one_line_rationale() -> None:
    for category in CAFC_CATEGORIES:
        rationale = category.rationale.strip()
        assert rationale, f"{category.category} carries no rationale"
        assert len(rationale.splitlines()) == 1, (
            f"{category.category} explains itself in more than one line"
        )
        assert len(rationale) >= 40, (
            f"{category.category} has a rationale too short to argue from: {rationale!r}"
        )


def test_the_recovery_pitch_is_placed_and_says_why_it_stands_alone() -> None:
    """2,060 real reports, named in the ticket, kept visible rather than absorbed."""
    found = next(c for c in CAFC_CATEGORIES if c.category == RECOVERY_PITCH)
    assert committed()[RECOVERY_PITCH] == RECOVERY_PITCH_REPORTS
    assert found.dropped is False
    assert found.scam_category is not None

    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    assert RECOVERY_PITCH in report
    assert f"{RECOVERY_PITCH_REPORTS:,}" in report


def test_the_projection_is_about_ten_categories_and_says_why() -> None:
    assert TOP_LEVEL_COUNT == len(SCAM_CATEGORIES)
    assert len(SCAM_CATEGORIES) == 10, f"{len(SCAM_CATEGORIES)} Scam Categories, not 10"
    names = [scam_category.name for scam_category in SCAM_CATEGORIES]
    assert OTHER.name not in names, "Other is a bucket, not one of the ten"
    assert OTHER.covers, "the Other bucket does not say what it is for"
    assert f"## Why the count is {TOP_LEVEL_COUNT}" in COMMITTED_REPORT.read_text(
        encoding="utf-8"
    )


def test_other_holds_content_that_fits_no_category_and_is_measured() -> None:
    """ADR-0006 makes the bucket's size a standing finding, so it is measured here.

    CAFC's own residual category is the one name that legitimately lands in it:
    CAFC filed it for a report its analysts could not describe any other way,
    which is exactly the content the project cannot place either.
    """
    placed = [c for c in CAFC_CATEGORIES if c.scam_category == OTHER.name]
    assert [c.category for c in placed] == ["Other"]

    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    assert OTHER.name in report
    assert f"{committed()['Other']:,}" in report


def test_the_base_rates_of_a_scam_category_add_up_to_the_reports_it_lands() -> None:
    """Every mapped report is counted once, and counted where it lands.

    Checked by reading the page's own table back, so an aggregation that
    double-counts a category or loses one cannot pass on the strength of the
    figure the code computed for itself.
    """
    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    counts = committed()
    totals = _totals_from(report)

    mapped = {
        category.scam_category
        for category in CAFC_CATEGORIES
        if category.scam_category is not None and not category.dropped
    }
    assert mapped == {scam_category.name for scam_category in SCAM_CATEGORIES} | {OTHER.name}
    assert set(totals) == mapped

    for name, total in totals.items():
        members = [
            counts[c.category]
            for c in CAFC_CATEGORIES
            if c.scam_category == name and not c.dropped and c.category in counts
        ]
        assert total == sum(members), f"{name} reports {total}, its members hold {sum(members)}"

    dropped = sum(
        counts[c.category] for c in CAFC_CATEGORIES if c.dropped and c.category in counts
    )
    assert sum(totals.values()) + dropped == sum(counts.values()), (
        "the projection accounts for every report in the extract"
    )


def test_the_report_states_the_count_reconciliation_rather_than_a_bare_number() -> None:
    """The 41 in ADR-0006 is corrected in the open, with the evidence for it."""
    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    for expected in (
        f"{len(ANNEX_CATEGORIES)}",
        f"{len(committed())}",
        f"{len(CAFC_CATEGORIES)}",
        "ADR-0006",
        "annex",
    ):
        assert expected in report, f"the report does not mention {expected!r}"


def test_the_report_names_every_category_and_its_rationale() -> None:
    """A reader who disagrees with one merge has to be able to find it."""
    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    for category in CAFC_CATEGORIES:
        assert category.category in report, f"{category.category} is not in the report"
        assert category.rationale in report, f"{category.category} has no stated reason"


def test_the_report_leaves_nothing_because_of_a_rewrite_of_the_figures() -> None:
    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    assert "Generated by `rfi scam-categories`" in report
    assert "Do not edit it by hand" in report


def test_the_committed_outputs_are_what_the_command_writes(tmp_path: Path) -> None:
    mapping, report = run_report(tmp_path)
    assert report.read_bytes() == COMMITTED_REPORT.read_bytes()
    assert mapping.read_bytes() == COMMITTED_MAPPING.read_bytes()


def test_running_twice_writes_byte_identical_bytes(tmp_path: Path) -> None:
    first = run_report(tmp_path / "first")
    second = run_report(tmp_path / "second")
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]


def test_the_command_needs_no_network_and_reads_only_its_one_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same argument as the CAFC figures: reproducible offline, from bytes."""
    import urllib.request

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the scam-categories command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert run_report(tmp_path)[1].exists()


def test_missing_base_rates_points_at_the_command_that_writes_them(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as refusal:
        main(["scam-categories", "--base-rates", str(tmp_path / "absent.jsonl")])

    assert "cafc-report" in str(refusal.value)


def test_no_base_rates_is_refused_rather_than_producing_an_empty_projection(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "base_rates.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as refusal:
        main(["scam-categories", "--base-rates", str(empty), "--report", str(tmp_path / "s.md")])

    assert "hold no categories" in str(refusal.value)


def test_the_argument_is_written_in_one_module() -> None:
    """A projection duplicated into a second module is an argument with two versions.

    The reasons are the argument, so this greps `src/` for them. It does not
    grep for CAFC's labels: one-word labels like Other and Job are ordinary
    English, and the multi-word ones legitimately appear in the renderer as it
    explains which merge a reader is most likely to contest. Quoting a label in
    order to argue about it is the point; restating a placement is the risk.
    """
    source = REPO_ROOT / "src"
    quoting = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*.py")
        if any(
            category.rationale in path.read_text(encoding="utf-8")
            for category in CAFC_CATEGORIES
        )
    }
    assert quoting == {"reddit_fraud_intelligence/categories.py"}, (
        f"the reasons are also written in {sorted(quoting)}"
    )


def test_the_mapping_is_written_out_as_data_for_the_rest_of_the_pipeline() -> None:
    """One artefact, readable without running the renderer.

    A reader, and later a command, needs the mapping itself rather than a
    Markdown page parsed back out of a table. Checked against the committed file,
    which `test_the_committed_outputs_are_what_the_command_writes` already holds
    to a fresh run, so this is about the shape of the rows rather than the bytes.
    """
    written = _mapping_rows()
    assert [text(row, "category") for row in written] == [
        c.category for c in sorted(CAFC_CATEGORIES, key=lambda c: c.category)
    ]
    covers = {scam.name: scam.covers for scam in SCAM_CATEGORIES + (OTHER,)}
    for row in written:
        source = next(c for c in CAFC_CATEGORIES if c.category == text(row, "category"))
        assert text(row, "rationale") == source.rationale
        assert maybe_text(row, "annex") == source.annex
        assert row["dropped"] is source.dropped
        if source.dropped:
            assert row["scam_category"] is None
            assert row["covers"] is None
        else:
            assert text(row, "scam_category") == source.scam_category
            assert text(row, "covers") == covers[source.scam_category], (
                f"{source.category} does not carry what its Scam Category means"
            )


def _totals_from(report: str) -> dict[str, int]:
    """Read the page's own totals table back, rather than trusting the renderer."""
    totals: dict[str, int] = {}
    reading = False
    for line in report.splitlines():
        if line.startswith("| Scam Category |"):
            reading = True
            continue
        if reading:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != 3 or not cells[1].replace(",", "").isdigit():
                continue
            totals[cells[0]] = int(cells[1].replace(",", ""))
    return totals
