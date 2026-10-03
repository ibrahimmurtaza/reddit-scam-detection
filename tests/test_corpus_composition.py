"""The Corpus's own distribution beside CAFC's published base rates.

Seam under test: the `corpus-composition` command, observed through the two files it
writes and the table it prints. Nothing here inspects the code that wrote them.

The claim under test is that a distortion in the synthetic Corpus is visible rather
than assumed away. That claim is asked as a question about the output: every Scam
Category has to appear with the share of the Corpus landing in it *and* CAFC's base
rate for the same class, on one line, with a difference to read. The Other bucket is
asked for by name in every view, because a bucket that is only ever the remainder of
a table is a bucket whose size means nothing.

The rest of the file checks the things that make those figures checkable: that every
share in the table is a share of the whole and the shares add up, that every post is
placed once, that a placement rests on a published phrase inside a sentence of the
post's own text, and that CAFC's own limits are stated rather than left for a reader
to discover.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import pytest

from reddit_fraud_intelligence.categories import (
    CAFC_CATEGORIES,
    EVERY_CATEGORY,
    OTHER,
    SCAM_CATEGORIES,
)
from reddit_fraud_intelligence.cli import (
    DEFAULT_BASE_RATES_PATH,
    DEFAULT_CORPUS_PATH,
    DEFAULT_SEED,
    main,
)
from reddit_fraud_intelligence.corpus import CorpusItem

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_COMPOSITION = REPO_ROOT / "data" / "corpus" / "composition.jsonl"
COMMITTED_REPORT = REPO_ROOT / "docs" / "corpus-composition.md"
COMMITTED_BASE_RATES = REPO_ROOT / "data" / "cafc" / "base_rates.jsonl"
COMMITTED_NUISANCE = REPO_ROOT / "data" / "corpus" / "nuisance.jsonl"

Row = Mapping[str, object]


class _Opened:
    """Records every path a run opens, so a claim about what it read can be checked.

    An audit hook rather than a monkeypatch, because the point is to catch a read from
    anywhere at all - including from inside the standard library, which a patched
    `open` would miss. It cannot be uninstalled, so it records only while `recording`
    is set and does nothing for the rest of the session.
    """

    def __init__(self) -> None:
        self.recording = False
        self.paths: list[str] = []

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        if self.recording and event == "open":
            self.paths.append(str(arguments[0]))

    def record(self) -> list[str]:
        return list(self.paths)


def run(
    directory: Path,
    corpus: Path = DEFAULT_CORPUS_PATH,
    base_rates: Path = DEFAULT_BASE_RATES_PATH,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> tuple[Path, Path, str]:
    """Run the command over the committed Corpus, writing both outputs into `directory`.

    The inputs default to the command's own defaults rather than to this file's absolute
    paths, because the report prints the path it read: a report generated from an
    absolute path could not be held to the committed one.
    """
    composition = directory / "composition.jsonl"
    report = directory / "corpus-composition.md"
    exit_code = main(
        [
            "corpus-composition",
            "--corpus",
            str(corpus),
            "--base-rates",
            str(base_rates),
            "--composition",
            str(composition),
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out if capsys is not None else ""
    return composition, report, printed


def rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} should be text, is {value!r}"
    return value


def number(row: Row, field: str) -> int:
    value = row[field]
    assert isinstance(value, int), f"{field} should be a whole number, is {value!r}"
    return value


def texts(row: Row, field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    return [str(entry) for entry in value]


def records(row: Row, field: str) -> list[Row]:
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    return [entry for entry in value if isinstance(entry, dict)]


def _table(printed: str) -> dict[str, dict[str, str]]:
    """The comparison table, read back out of the console output.

    Read back rather than trusted, because the claim is that the two distributions are
    directly comparable: a figure a test recomputed from the module that printed it
    would pass even if the table dropped one of the two columns.
    """
    cells: dict[str, dict[str, str]] = {}
    reading = False
    for line in printed.splitlines():
        if line.strip().startswith("category") and "CAFC" in line:
            reading = True
            continue
        if not reading:
            continue
        if not line.strip():
            break
        parts = line.split()
        if len(parts) < 5:
            continue
        cells[" ".join(parts[:-4])] = {
            "posts": parts[-4],
            "corpus": parts[-3],
            "cafc": parts[-2],
            "difference": parts[-1],
        }
    return cells


def test_every_scam_category_carries_the_corpus_share_and_cafcs_base_rate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ticket's first two criteria, asked as one line per category.

    Every one of the ten and the bucket is a row, and every row carries four figures: how
    many posts landed in it, what share of the Corpus that is, CAFC's base rate for the
    same class, and the difference between the two. A row carrying only the Corpus
    figure would leave the reader to look the other half up.
    """
    _, _, printed = run(tmp_path, capsys=capsys)
    table = _table(printed)

    assert set(table) == {scam.name for scam in EVERY_CATEGORY}, (
        f"the table holds {sorted(table)}, which is not the ten and the bucket"
    )
    for name, row in table.items():
        assert int(row["posts"]) >= 0
        assert row["corpus"].endswith("%"), f"{name} carries no Corpus share: {row}"
        assert row["cafc"].endswith("%"), f"{name} carries no CAFC base rate: {row}"
        assert row["difference"].endswith("pp"), (
            f"{name} carries no difference to read: {row}"
        )


def test_cafcs_base_rate_beside_each_category_is_the_share_the_projection_lands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CAFC column is computed from the committed figures, not written down here.

    Read the landing off the projection and the reports off the committed base rates,
    divide by every report in the extract, and require the table to agree. That is the
    figure `docs/scam-categories.md` already publishes, so the two pages cannot disagree
    about the same class.
    """
    _, _, printed = run(tmp_path, capsys=capsys)
    table = _table(printed)

    counts = {
        text(row, "category"): number(row, "reports")
        for row in rows(COMMITTED_BASE_RATES)
    }
    reports = sum(counts.values())
    assert reports == 350_361, f"the extract holds {reports:,} reports, not 350,361"

    for name, row in table.items():
        landing = sum(
            counts[category.category]
            for category in CAFC_CATEGORIES
            if category.scam_category == name
            and not category.dropped
            and category.category in counts
        )
        assert row["cafc"] == _percent(landing, reports), (
            f"{name} is shown against {row['cafc']} and {landing:,} of {reports:,} "
            "reports land in it"
        )


def test_the_corpus_shares_are_shares_of_the_corpus_and_add_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two things in one, because they are one property: a share of something.

    Every category is a share of the whole Corpus rather than of the posts that were
    placed, and the shares add up to all of it. A share of the placed posts alone would
    quietly shrink the Other bucket by exactly the size of the bucket, which is the one
    figure the ticket asks to be unable to lose.
    """
    composition, _, printed = run(tmp_path, capsys=capsys)
    table = _table(printed)

    posts = rows(COMMITTED_CORPUS)
    assert sum(int(row["posts"]) for row in table.values()) == len(posts), (
        "the categories do not account for every post in the Corpus"
    )
    assert len(rows(composition)) == len(posts)

    for name, row in table.items():
        assert row["corpus"] == _percent(int(row["posts"]), len(posts)), (
            f"{name} shows {row['corpus']} for {row['posts']} of {len(posts)} posts"
        )


def _percent(count: int, of: int) -> str:
    """A share to the tenth of a percent, as the table prints it.

    Written out here rather than imported so the test cannot pass by agreeing with the
    renderer about how a share is rounded. Half up, in tenths of a percent, which is
    what a figure of this size needs: CAFC's smallest landing is 3,897 reports.
    """
    tenths = (1000 * count + of // 2) // of
    return f"{tenths // 10}.{tenths % 10}%"


def test_the_other_bucket_is_a_row_and_cannot_be_dropped_as_a_remainder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The criterion the ticket calls out by name, checked in every view at once.

    The bucket is a row in the table beside the ten, a line in the report, and a share
    of the whole Corpus rather than of what is left over after the ten are counted. It
    is also held up against CAFC's own residual category, because CAFC filed that one
    for a report its analysts could not describe any other way: the two figures measure
    the same kind of gap and are worth having side by side.
    """
    composition, report, printed = run(tmp_path, capsys=capsys)
    table = _table(printed)

    assert OTHER.name in table, "the Other bucket is not a row in the comparison"
    assert OTHER.name in report.read_text(encoding="utf-8"), "the Other bucket is not in the report"

    placed = [
        row for row in rows(composition) if text(row, "scam_category") == OTHER.name
    ]
    assert table[OTHER.name]["posts"] == str(len(placed)), (
        "the bucket's share in the table is not the number of posts in it"
    )

    cafc_other = next(
        number(row, "reports") for row in rows(COMMITTED_BASE_RATES) if text(row, "category") == OTHER.name
    )
    reports = sum(number(row, "reports") for row in rows(COMMITTED_BASE_RATES))
    assert table[OTHER.name]["cafc"] == _percent(cafc_other, reports), (
        "CAFC's own residual category is not the figure the bucket is held against"
    )


def test_the_scam_categories_the_corpus_does_not_cover_are_named_with_their_share(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A distortion the reader has to compute is a distortion nobody looks at.

    CAFC's reports do not arrive evenly spread across the ten: most of them land in
    classes this Corpus has nothing in at all, because the generator writes the two
    pitches it writes. The report has to name those classes and print the share of real
    reports that lands in them, rather than leaving the reader to add up the zeroes.
    """
    _, report, printed = run(tmp_path, capsys=capsys)
    table = _table(printed)
    text_of = report.read_text(encoding="utf-8")

    uncovered = [name for name, row in table.items() if int(row["posts"]) == 0]
    assert uncovered, "this Corpus covers every one of the ten, which nothing here expects"
    for name in uncovered:
        assert name in text_of, f"{name} holds no post in the Corpus and is not named as such"

    total = sum(
        number(row, "reports")
        for row in rows(COMMITTED_BASE_RATES)
        if row["category"]
        in {
            category.category
            for category in CAFC_CATEGORIES
            if category.scam_category in uncovered and not category.dropped
        }
    )
    reports = sum(number(row, "reports") for row in rows(COMMITTED_BASE_RATES))
    assert _percent(total, reports) in text_of, (
        "the share of real reports landing in the classes the Corpus has nothing in is "
        "not printed"
    )


def test_the_largest_distortion_is_named_with_both_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The distortion the ticket describes, asked of whichever one is largest.

    "A Corpus that is 22% crypto investment content against a real-world base rate of
    5.8% should look like a problem". The report has to name the class where the gap is
    widest and print both figures of it, so a reader is told what to look at rather than
    left to find it. Which class that is comes out of the table rather than being
    written into the test, so the assertion cannot drift away from the run.
    """
    _, report, printed = run(tmp_path, capsys=capsys)
    table = _table(printed)

    def gap(row: Mapping[str, str]) -> float:
        return abs(_share(row["corpus"]) - _share(row["cafc"]))

    name, row = max(table.items(), key=lambda pair: (gap(pair[1]), pair[0]))
    text_of = report.read_text(encoding="utf-8")

    assert gap(row) > 0, "no category in this Corpus differs from its base rate"
    assert name in text_of, f"{name} carries the largest gap and is not named as such"
    for figure in (row["corpus"], row["cafc"]):
        assert figure in text_of, f"the largest gap is reported without {figure}"


def test_the_output_states_cafc_carries_no_free_text_and_cannot_validate_a_classifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The criterion that keeps this comparison honest about what it is.

    CAFC's extract has no column that can hold a sentence, so it constrains which
    Scam Categories exist and what their priors are, and it cannot check a single
    placement made from a post's own words. Both views have to say so: the report is
    where a reader looks for the limits, and the console is where they are met first.

    Named as "the Scam Categories" rather than as a classification scheme, because
    `GLOSSARY.md` reserves that other word against this concept and a report that used
    it would be the one file here arguing with the vocabulary.
    """
    _, report, printed = run(tmp_path, capsys=capsys)
    for view in (report.read_text(encoding="utf-8"), printed):
        lowered = view.lower()
        assert "free text" in lowered or "free-text" in lowered, view[:200]
        assert "cannot validate" in lowered, (
            f"neither view says the extract cannot validate a classifier:\n{view[:400]}"
        )
        assert "scam categor" in lowered, (
            f"neither view says which classes the extract constrains:\n{view[:400]}"
        )
        assert "prior" in lowered, f"neither view says the extract constrains the priors"


def test_the_run_needs_no_network_and_reads_the_corpus_and_the_base_rates_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two files in, and the truth file and the Nuisance Structure manifest out of reach.

    The placement is read from a post's own text, so the whole input about content is
    the Corpus file; the membership of the Planted Campaigns sits next to it holding the
    answer to what these posts are. Reading either would make the distribution a
    statement about the generator rather than about the Corpus a reviewer can read.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the corpus-composition command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    opened = _Opened()
    sys.addaudithook(opened)

    opened.recording = True
    try:
        run(tmp_path)
    finally:
        opened.recording = False

    read = {
        Path(path).name
        for path in opened.record()
        if Path(path).suffix in {".jsonl", ".dat", ".gz"} and "out" not in Path(path).parent.name
    }
    assert {"corpus.jsonl", "base_rates.jsonl"} <= read, sorted(read)
    assert not [name for name in read if name.startswith(("truth", "nuisance"))]


def test_the_committed_outputs_are_what_the_command_writes(tmp_path: Path) -> None:
    """Held to the command the same way the Corpus is held to its generator."""
    composition, report, _ = run(tmp_path)

    assert composition.read_bytes() == COMMITTED_COMPOSITION.read_bytes()
    assert report.read_bytes() == COMMITTED_REPORT.read_bytes()


def test_running_twice_writes_byte_identical_bytes(tmp_path: Path) -> None:
    first = run(tmp_path / "first")
    second = run(tmp_path / "second")

    assert [path.read_bytes() for path in first[:2]] == [path.read_bytes() for path in second[:2]]


def test_the_command_refuses_to_write_the_corpus_away(tmp_path: Path) -> None:
    """The Corpus is the input, and the distribution is about that exact file."""
    with pytest.raises(SystemExit):
        main(
            [
                "corpus-composition",
                "--corpus",
                str(COMMITTED_CORPUS),
                "--composition",
                str(COMMITTED_CORPUS),
                "--report",
                str(tmp_path / "report.md"),
            ]
        )

    assert rows(COMMITTED_CORPUS)


def test_missing_base_rates_points_at_the_command_that_writes_them(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as refusal:
        main(["corpus-composition", "--base-rates", str(tmp_path / "absent.jsonl")])

    assert "cafc-report" in str(refusal.value)


def _share(figure: str) -> float:
    """One printed percentage, read back as a number for comparing two of them."""
    return float(figure.removesuffix("%"))


# --- how a post is placed ------------------------------------------------------------


def write_corpus(path: Path, items: tuple[CorpusItem, ...]) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "post_id": item.post_id,
                    "account": item.account,
                    "subreddit": item.subreddit,
                    "title": item.title,
                    "body": item.body,
                    "created_at": item.created_at,
                    "links": list(item.links),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    return path


def post(post_id: str, body: str = "a body", title: str = "a title") -> CorpusItem:
    return CorpusItem(
        post_id=post_id,
        account=f"syn_account_{post_id}",
        subreddit="test",
        title=title,
        body=body,
        created_at="2026-01-05T09:00:00Z",
        links=(),
    )


def test_a_placement_rests_on_a_published_phrase_inside_the_posts_own_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The figure has to be recomputable by hand, or the table is an assertion.

    Every piece of evidence is checked against the post it is attached to rather than
    against a list of which Scam Category is meant to be which kind: the sentence is a
    substring of that post's own title or body, every phrase in it is a phrase the run
    published, and it appears in that sentence. A placement resting on anything else
    cannot be written without failing this.
    """
    composition, _, printed = run(tmp_path, capsys=capsys)
    published = published_phrases(printed)
    posts = {text(row, "post_id"): row for row in rows(COMMITTED_CORPUS)}

    placed = 0
    for row in rows(composition):
        post_id = text(row, "post_id")
        assert post_id in posts, f"{post_id} is not in the Corpus the run read"
        fields = {text(posts[post_id], "title"), text(posts[post_id], "body")}
        evidence = records(row, "evidence")
        if not evidence:
            assert text(row, "scam_category") == OTHER.name, (
                f"{post_id} has no evidence and is not in {OTHER.name}"
            )
            continue
        placed += 1
        assert text(row, "scam_category") != OTHER.name, (
            f"{post_id} carries evidence and is also in {OTHER.name}"
        )
        for item in evidence:
            sentence = text(item, "sentence")
            assert text(item, "field") in {"title", "body"}
            assert any(sentence in field for field in fields), (
                f"{post_id} carries a sentence that is not in its own post"
            )
            for phrase in texts(item, "phrases"):
                assert phrase in published, f"{phrase!r} was used and never published"
                assert phrase in sentence.lower(), (
                    f"{post_id} carries {phrase!r} on a sentence that does not contain it"
                )
    assert placed, "no post in this Corpus was placed by any phrase at all"


def test_a_post_matching_two_classes_lands_on_the_first_declared_and_records_the_other(
    tmp_path: Path,
) -> None:
    """The tie-break, exercised rather than described.

    A post can carry two pitches; this one is written out by hand for the case, because
    the Corpus happens to hold none. What is asserted is what the declaration order
    decides and what a reader is owed for the decision: the class the post did *not*
    land in goes in the file beside it, so the tie is visible rather than resolved
    silently by an order nothing else in the output explains.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_0001",
                body="Minimum ticket 500 USDT, and no experience needed, kit provided.",
            ),
        ),
    )
    composition, _, _ = run(tmp_path / "out", corpus=corpus)
    written = rows(composition)

    assert len(written) == 1
    assert text(written[0], "scam_category") == "Investment and Money Offers", (
        "the tie went to the class the projection declares later"
    )
    assert texts(written[0], "also_matched") == ["Work and Payroll"]
    evidence = texts(records(written[0], "evidence")[0], "phrases")
    assert "minimum ticket" in evidence
    assert "no experience" not in evidence, (
        "the evidence carries a phrase of the class the post did not land in"
    )


def test_both_views_publish_the_same_phrase_list_for_every_class(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A phrase list is a rule, so publishing it is the only way a reader gets it.

    Checked across the two published views rather than against the lists the code holds,
    so the assertion cannot pass by the test and the module agreeing about what the
    rules are: every class has a list in the console output, the same list in the report,
    and every phrase the run actually used is in both.
    """
    composition, report, printed = run(tmp_path, capsys=capsys)
    text_of = report.read_text(encoding="utf-8")

    for scam in SCAM_CATEGORIES:
        console = published_phrases(printed, scam.name)
        published = report_phrases(text_of, scam.name)
        assert console, f"{scam.name} has no phrase list in the console output"
        assert console == published, (
            f"{scam.name} publishes {len(console)} phrases on the console and "
            f"{len(published)} in the report"
        )

    used = {
        phrase
        for row in rows(composition)
        for item in records(row, "evidence")
        for phrase in texts(item, "phrases")
    }
    everything = published_phrases(printed)
    assert used <= everything, f"{sorted(used - everything)} were used and never published"
    assert len(everything) > 100, f"only {len(everything)} phrases are published"


def published_phrases(printed: str, name: str | None = None) -> set[str]:
    """The phrases the console output publishes, for one class or for all of them.

    Read back out of the output rather than from the module's table, because the claim is
    that a reader can see the rule: a test that compares the output with the code proves
    only that the code prints itself. Each class's block is kept apart before it is read,
    because the last phrase of one list and the first of the next are the same line of
    output as far as anything outside the run is concerned.
    """
    blocks: list[str] = []
    current: list[str] = []
    inside = name is None
    for line in printed.splitlines():
        stripped = line.strip()
        if stripped.startswith("phrases"):
            continue
        heading = (
            line.startswith("  ")
            and not line.startswith("    ")
            and stripped.endswith(("phrase", "phrases"))
        )
        if heading:
            if inside and current:
                blocks.append("\n".join(current))
            current = []
            inside = name is None or stripped.split("  ")[0].strip() == name
            continue
        if inside and line.startswith("    ") and '"' in line:
            current.append(line)
    if inside and current:
        blocks.append("\n".join(current))
    return {
        re.sub(r"\s+", " ", found).strip()
        for block in blocks
        for found in re.findall(r'"([^"]+)"', block)
    }


def report_phrases(text: str, name: str) -> set[str]:
    """The phrases one class publishes in the report, read back out of its own section.

    Whitespace is folded because the page wraps its lists too, and a phrase may straddle
    two lines: a reader sees one phrase and a test that saw two halves would be reading a
    wrapping artefact rather than a difference.
    """
    section = next(part for part in text.split("\n### ") if part.startswith(f"{name}\n"))
    return {
        re.sub(r"\s+", " ", found).strip() for found in re.findall(r'"([^"]+)"', section)
    }


def test_a_genuine_job_post_is_placed_by_the_pitch_it_advertises(
    tmp_path: Path,
) -> None:
    """What the reading can and cannot tell, read against the planted material.

    The Nuisance Structure manifest is the evaluator's file, and this is the evaluator
    reading it, which is the only direction ADR-0011 allows it in. Every post the Corpus
    plants as a genuine job advert lands in Work and Payroll beside the planted ones,
    because the lists read the pitch and a pitch about work is a pitch about work; a
    parody of a pitch lands in the class it parodies rather than in Other. Both are the
    honest limit of reading text for what it talks about, and the report names them
    rather than leaving a reader to infer that a class is a judgement.
    """
    composition, report, _ = run(tmp_path / "out")
    placed = {text(row, "post_id"): text(row, "scam_category") for row in rows(composition)}

    genuine = _posts_with_character("genuine_job_post")
    satire = _posts_with_character("satire")
    assert genuine and satire, "the Corpus plants neither of these characters"

    for post_id in genuine:
        assert placed[post_id] == "Work and Payroll", (
            f"{post_id} is a declared genuine job advert and landed elsewhere"
        )
    for post_id in satire:
        assert placed[post_id] != OTHER.name, (
            f"{post_id} parodies a pitch and landed in {OTHER.name}"
        )

    assert "satire" in report.read_text(encoding="utf-8"), (
        "the report does not name the case that shows what the reading cannot separate"
    )


def _posts_with_character(character: str) -> list[str]:
    """Every post the manifest records as carrying one Hard Negative character."""
    return [
        post_id
        for record in rows(COMMITTED_NUISANCE)
        if character in texts(record, "characters")
        for post_id in texts(record, "posts")
    ]


def test_the_run_covers_every_seed_without_a_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Corpus is a function of its seed, so a different seed must need no edit here.

    A seed draws different Nuisance Structure material and plants different spares, so
    which posts land in which class moves. What must hold for every seed is that every
    post is placed once, that every placement rests on a sentence of that post's own
    text, and that the figures in the table are shares of the whole Corpus.
    """
    for seed in (DEFAULT_SEED, DEFAULT_SEED + 1):
        corpus = tmp_path / f"corpus-{seed}.jsonl"
        assert main(
            [
                "generate-corpus",
                "--seed",
                str(seed),
                "--corpus",
                str(corpus),
                "--truth",
                str(tmp_path / f"truth-{seed}.jsonl"),
                "--nuisance",
                str(tmp_path / f"nuisance-{seed}.jsonl"),
                "--shared-infrastructure",
                str(tmp_path / f"shared-{seed}.jsonl"),
            ]
        ) == 0

        composition, report, printed = run(tmp_path / f"out-{seed}", corpus=corpus, capsys=capsys)
        written = rows(composition)
        posts = rows(corpus)
        table = _table(printed)

        assert len(written) == len(posts), f"seed {seed}: one row per post"
        assert len({text(row, "post_id") for row in written}) == len(written)
        assert sum(int(row["posts"]) for row in table.values()) == len(posts), (
            f"seed {seed}: the categories do not account for every post"
        )
        assert f"{len(posts)} posts" in report.read_text(encoding="utf-8")

        visible = {text(row, "title") for row in posts} | {text(row, "body") for row in posts}
        for row in written:
            for item in records(row, "evidence"):
                assert text(item, "field") in {"title", "body"}
                assert any(text(item, "sentence") in field for field in visible), (
                    f"seed {seed}: {text(row, 'post_id')} carries a sentence the Corpus "
                    "does not hold"
                )
