"""The cached CAFC extract is a claim about real fraud, so it is checked against bytes.

Seam under test: the `cafc-report` command, observed through the files it writes,
and `fetch-cafc`, observed through the refusal it makes before touching the
network. Nothing here inspects the code that wrote them.

Two kinds of test live here. Most read the committed artefacts, because the point
is that a real 72 MB extract is described accurately. A few run the command over a
handful of rows written to a temporary directory, because behaviours that only
show up on the real file — determinism, or what happens when a column does hold
prose — cannot be provoked by 350,361 rows.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import main

REPO_ROOT = Path(__file__).parent.parent

COMMITTED_EXTRACT = REPO_ROOT / "data" / "cafc" / "cafc-extract.csv.gz"
COMMITTED_PROVENANCE = REPO_ROOT / "data" / "cafc" / "provenance.jsonl"
COMMITTED_BASE_RATES = REPO_ROOT / "data" / "cafc" / "base_rates.jsonl"
COMMITTED_REPORT = REPO_ROOT / "docs" / "cafc-base-rates.md"

# The header of the real extract is bilingual, English column then its French
# translation. These two rows reproduce that shape without the accented text, so
# the tests fail if the code starts depending on the exact French wording.
HEADER = (
    "Number ID,Date Received / Date recue,Fraud and Cybercrime Thematic Categories,"
    "Fraud Categories,Solicitation Method"
)

ROWS = (
    ("350308", "2025-09-29", "Identity Fraud", "Direct call"),
    ("350309", "2025-09-29", "Extortion", "Direct call"),
    ("350310", "2025-09-28", "Identity Fraud", "Internet-social network"),
    ("350311", "2025-09-28", "Extortion", "Direct call"),
)


def run_report(directory: Path, extract: Path) -> tuple[Path, Path, Path]:
    """Run the command over `extract`, writing into `directory`."""
    provenance = directory / "provenance.jsonl"
    base_rates = directory / "base_rates.jsonl"
    report = directory / "cafc-base-rates.md"
    exit_code = main(
        [
            "cafc-report",
            "--extract",
            str(extract),
            "--provenance",
            str(provenance),
            "--base-rates",
            str(base_rates),
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0
    return provenance, base_rates, report


def write_extract(path: Path, rows: Sequence[tuple[str, str, str, str]]) -> Path:
    """Write a compressed extract in the shape the real one has."""
    body = "".join(
        f"{row[0]},{row[1]},{row[2]},Fraud Categories,{row[3]}\r\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(f"{HEADER}\r\n{body}".encode("utf-8"))
    return path


def rows(path: Path) -> list[Mapping[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def one(path: Path) -> Mapping[str, object]:
    found = rows(path)
    assert len(found) == 1, f"{path.name} should hold a single line, holds {len(found)}"
    return found[0]


def number(row: Mapping[str, object], field: str) -> int:
    value = row[field]
    assert isinstance(value, int), f"{field} should be a whole number, is {value!r}"
    return value


def decimal(row: Mapping[str, object], field: str) -> float:
    value = row[field]
    assert isinstance(value, float), f"{field} should be a fraction, is {value!r}"
    return value


def text(row: Mapping[str, object], field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} should be text, is {value!r}"
    return value


@pytest.fixture(scope="module")
def reported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """The command run once over the committed extract.

    Module scope because the extract is 72 MB uncompressed and reading it back is
    the slow part of this file; every test that needs a fresh reading shares it.
    """
    return run_report(tmp_path_factory.mktemp("committed"), COMMITTED_EXTRACT)


def test_the_cached_extract_holds_the_number_of_reports_the_recorded_figure_names() -> None:
    """A download cut short fails here, loudly, rather than quietly halving a base rate.

    Counted line by line rather than by parsing, deliberately: the count has to be
    independent of the code that recorded it, and the extract holds no value with a
    newline in it.
    """
    recorded = number(one(COMMITTED_PROVENANCE), "reports")

    with gzip.open(COMMITTED_EXTRACT, "rb") as handle:
        lines = iter(handle)

        header = next(lines).decode("utf-8")
        assert header.count("Number ID") == 1, "the first line is the header, not a report"
        assert sum(1 for line in lines if line.strip()) == recorded


def test_the_cached_extract_is_the_bytes_the_recorded_digest_names() -> None:
    recorded = text(one(COMMITTED_PROVENANCE), "sha256")

    checksum = hashlib.sha256()
    with gzip.open(COMMITTED_EXTRACT, "rb") as handle:
        while chunk := handle.read(1 << 20):
            checksum.update(chunk)

    assert checksum.hexdigest() == recorded


def test_the_base_rates_are_what_the_cached_extract_produces(
    reported: tuple[Path, Path, Path],
) -> None:
    assert (reported[1].read_bytes(), reported[2].read_bytes()) == (
        COMMITTED_BASE_RATES.read_bytes(),
        COMMITTED_REPORT.read_bytes(),
    )


def test_the_provenance_is_what_the_cached_extract_produces(
    reported: tuple[Path, Path, Path],
) -> None:
    assert reported[0].read_bytes() == COMMITTED_PROVENANCE.read_bytes()


def test_every_category_in_the_extract_has_a_base_rate_and_they_add_up_to_one() -> None:
    base_rates = rows(COMMITTED_BASE_RATES)
    assert len(base_rates) > 0

    reports = sum(number(entry, "reports") for entry in base_rates)
    assert reports == number(one(COMMITTED_PROVENANCE), "reports")

    categories = [text(entry, "category") for entry in base_rates]
    assert len(set(categories)) == len(categories), "a category is listed twice"
    assert abs(sum(decimal(entry, "base_rate") for entry in base_rates) - 1.0) <= 1e-4


def test_the_report_states_the_licence_the_attribution_and_the_source() -> None:
    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    for expected in (
        "Open Government Licence",
        "Royal Canadian Mounted Police",
        "open.canada.ca/data/en/dataset/6a09c998-cddb-4a22-beff-4dca67ab892f",
    ):
        assert expected in report, f"the report does not name {expected!r}"


def test_the_report_says_plainly_that_there_is_no_free_text_to_learn_from() -> None:
    provenance = one(COMMITTED_PROVENANCE)
    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    assert provenance["free_text_ruled_out"] is True
    assert "no free-text field" in report
    assert "cannot validate" in report
    assert str(number(provenance, "longest_value_chars")) in report


def test_the_report_does_not_claim_to_have_counted_everything_cafc_publishes() -> None:
    """This file measures one release; the projection reconciles both of CAFC's sources.

    The figure of 41 that ADR-0006 asserted is not one CAFC publishes, and it is
    neither repeated here nor replaced by another total: the report says what it
    measured and points at the page that reconciles the two counts.
    """
    observed = number(one(COMMITTED_PROVENANCE), "categories")
    assert observed == len(rows(COMMITTED_BASE_RATES))
    report = COMMITTED_REPORT.read_text(encoding="utf-8")
    assert f"enumerates **{observed} thematic category values**" in report
    assert "that is the whole\nof what it measures" in report
    assert "docs/scam-categories.md" in report
    assert "41" not in report


def test_reading_the_same_extract_twice_writes_the_same_bytes(tmp_path: Path) -> None:
    extract = write_extract(tmp_path / "extract.csv.gz", ROWS)
    first = run_report(tmp_path / "first", extract)
    second = run_report(tmp_path / "second", extract)

    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_the_base_rate_of_a_category_is_its_share_of_the_reports(tmp_path: Path) -> None:
    extract = write_extract(tmp_path / "extract.csv.gz", ROWS)
    _, base_rates, _ = run_report(tmp_path / "out", extract)

    found = {str(entry["category"]): entry for entry in rows(base_rates)}
    assert set(found) == {"Identity Fraud", "Extortion"}
    assert found["Identity Fraud"]["reports"] == 2
    assert found["Identity Fraud"]["base_rate"] == 0.5


def test_a_value_too_long_to_be_an_enumerated_label_withdraws_the_no_free_text_claim(
    tmp_path: Path,
) -> None:
    """The bound has two ways to fail and neither may be reported as the other.

    A caller writing prose into a column and CAFC publishing a longer translated
    label both cross it. The report must then decline to say anything, rather than
    claim the extract has free text — which is what a boolean would have let it do.
    """
    long_value = "A caller offered me a crypto investment and then asked for my SIN."
    extract = write_extract(
        tmp_path / "extract.csv.gz",
        (*ROWS, ("350312", "2025-09-27", "Extortion", long_value)),
    )
    provenance, base_rates, report_path = run_report(tmp_path / "out", extract)

    report = report_path.read_text(encoding="utf-8")
    assert one(provenance)["free_text_ruled_out"] is False
    assert "No claim is made here that this extract has no free-text field" in report
    assert "**There is no free-text field.**" not in report
    assert base_rates.exists(), "the base rates do not depend on the claim"


def test_fetching_refuses_to_replace_a_cache_that_is_already_there(tmp_path: Path) -> None:
    """Checked before any network call, so the test needs none."""
    extract = tmp_path / "cafc-extract.csv.gz"
    extract.write_bytes(b"already here")

    with pytest.raises(SystemExit) as refusal:
        main(["fetch-cafc", "--extract", str(extract)])

    assert "already there" in str(refusal.value)
    assert extract.read_bytes() == b"already here"
