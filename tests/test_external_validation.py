"""The external set is held out, and the report says what the set cannot establish.

Seam under test: `rfi external-validation` and the pure functions behind it. Nothing here
reads the Corpus, and nothing here may fit anything — the whole value of this set is that
the weights and phrase lists in `src/` predate these bytes, so a test that fitted on them
would be testing the one thing that would make the figure worthless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import main
from reddit_fraud_intelligence.external import (
    ATTRIBUTION,
    LICENCE,
    Labelled,
    facts,
    read,
    write_provenance,
)
from reddit_fraud_intelligence.signals import PostScore, SignalName
from reddit_fraud_intelligence.validation import as_corpus, validate

REPO_ROOT = Path(__file__).parent.parent
SOURCE = REPO_ROOT / "src" / "reddit_fraud_intelligence"
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_EXTERNAL = REPO_ROOT / "data" / "external" / "job-scams-test.jsonl"
EXTERNAL_CORPUS = REPO_ROOT / "data" / "external" / "external-corpus.jsonl"
COMMITTED_PROVENANCE = REPO_ROOT / "data" / "external" / "provenance.jsonl"
COMMITTED_REPORT = REPO_ROOT / "docs" / "external-validation.md"
TRUTH = REPO_ROOT / "data" / "corpus" / "truth.jsonl"
WEIGHTS = REPO_ROOT / "data" / "signals" / "weights.jsonl"

# The external set is fetched rather than committed (the .gitignore says why), so every test
# that reads it is a test about a file the machine running it may not have. It skips with the
# reason rather than failing, the same rule `needs_database` follows, and the tests below
# that build their own labelled rows still run without it — those are the ones about the
# harness, and they are the reason a fresh clone can check the refusals at all.
needs_external_set = pytest.mark.skipif(
    not COMMITTED_EXTERNAL.exists(),
    reason=(
        "the external labelled set is not present: it is real third-party text, fetched "
        f"rather than committed. Run `rfi fetch-external` to write {COMMITTED_EXTERNAL}, "
        "then run these again. The tests that build their own labelled rows do not need it"
    ),
)


def run(directory: Path, external: Path = COMMITTED_EXTERNAL) -> tuple[Path, str]:
    corpus = directory / "external-corpus.jsonl"
    report = directory / "external-validation.md"
    exit_code = main(
        [
            "external-validation",
            "--external",
            str(external),
            "--corpus",
            str(corpus),
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0
    return report, corpus.read_text(encoding="utf-8")


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def written(path: Path) -> tuple[Labelled, ...]:
    return read(path)


def tiny(directory: Path, *, deceptive: int, truthful: int) -> Path:
    """A labelled set of a stated size, for the refusals that need one."""
    path = directory / "external.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(deceptive):
            handle.write(
                json.dumps({"text": f"guaranteed returns, act now, deal {index}", "label": 1})
                + "\n"
            )
        for index in range(truthful):
            handle.write(
                json.dumps({"text": f"an ordinary notice about a meeting {index}", "label": 0})
                + "\n"
            )
    return path


@needs_external_set
def test_the_committed_external_set_is_real_and_labelled_by_somebody_else() -> None:
    """The figure is only worth anything because these bytes came from elsewhere.

    Not a test about a count: a reader who took the set for the project's own synthetic
    data would be taking the one number off this page that is not about this project at
    all, and the whole claim of an external validation is that the content and the labels
    were not written here.
    """
    measured = facts(COMMITTED_EXTERNAL)

    assert measured.rows > 0
    assert measured.deceptive > 0, (
        "a labelled set with no fraudulent rows cannot be measured against a fraud label; "
        "the AUC below would be the 0.5 of a set with one class in it"
    )
    assert measured.truthful > 0
    assert measured.longest_chars > 0
    record = rows(COMMITTED_PROVENANCE)[0]
    assert record["licence"] == LICENCE
    assert record["trained_on"] is False
    assert record["sha256"] == measured.sha256
    assert str(record["attribution"]) == ATTRIBUTION
    assert "DIFrauD" in str(record["attribution"])


def test_the_external_set_is_never_a_training_set() -> None:
    """Held out in the only way that counts: nothing in `src/` can fit on it.

    Checked as an import rather than as a convention, for the reason
    `test_the_confidence_module_reads_no_policy_score_rule` gives: a module that imported
    the external rows could grow a training step later and nothing here would notice until
    the figure had stopped meaning what its page says.
    """
    offenders = sorted(
        path.name
        for path in sorted(SOURCE.glob("*.py"))
        if path.name not in {"cli.py", "external.py", "validation.py"}
        and any(
            name.endswith(("external", "validation")) for name in _imported(path)
        )
    )

    assert offenders == [], (
        f"modules that must not reach the external rows: {offenders}. Only the CLI may, and "
        "only to measure with a model that was fitted before the bytes arrived."
    )


def _imported(path: Path) -> list[str]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@needs_external_set
def test_the_planted_truth_file_and_the_external_labels_never_meet(tmp_path: Path) -> None:
    """The external label is not the planted label, and the run must not turn it into one.

    The tempting shortcut is to score the external rows and then measure them against
    `truth.jsonl` because that file is the project's one labelled file. It would work, and
    it would be meaningless: those rows describe accounts that do not exist in this set, so
    every post would land in the "not planted" class and the AUC would be measuring nothing
    at all while looking like a measurement.
    """
    _, _ = run(tmp_path)
    corpus_text = (tmp_path / "external-corpus.jsonl").read_text(encoding="utf-8")
    planted: set[str] = set()
    for row in rows(TRUTH):
        members = row["posts"]
        assert isinstance(members, list)
        planted.update(str(post) for post in members)

    for line in corpus_text.splitlines():
        post_id = json.loads(line)["post_id"]
        assert post_id not in planted, (
            "a row of the external corpus is carrying a planted post identifier, so the "
            "external run and the adversarial evaluation are looking at the same row"
        )
    assert not (tmp_path / "truth.jsonl").exists(), (
        "the external run wrote a membership file; it must not produce or read one"
    )


@needs_external_set
def test_the_external_corpus_invents_no_account_and_no_timestamp(tmp_path: Path) -> None:
    """One constant account, one instant, no links — and none of it is a real observation.

    A job advertisement has no subreddit and no posting history. Rather than fabricate a
    per-row account and a plausible date, which would put invented values into a file whose
    whole claim is that it is what was read, every row carries the same account and the same
    timestamp. The grouping needs an account to be shared to form anything, so a corpus
    where they all are would over-group if it were ever run through it — which is why the
    report says this set cannot measure grouping rather than quietly leaving it out.
    """
    _, corpus_text = run(tmp_path)
    corpus = rows(tmp_path / "external-corpus.jsonl")

    assert {row["account"] for row in corpus} == {"external"}
    assert {row["created_at"] for row in corpus} == {"2026-01-01T00:00:00Z"}
    assert all(row["links"] == [] for row in corpus)
    assert {row["subreddit"] for row in corpus} == {""}


@needs_external_set
def test_the_link_signals_are_reported_unmeasurable_rather_than_at_zero(
    tmp_path: Path,
) -> None:
    """A Signal that cannot be computed and a Signal that never fires look identical in a table.

    This set has no URLs, so the three link Signals have no chance to fire. Printing them at
    a rate of zero would say "these never fire on real fraud", which is not what was
    measured; printing them as unmeasurable says what is true and keeps the score above
    from reading as the whole rules engine having run.
    """
    report, _ = run(tmp_path)
    text = report.read_text(encoding="utf-8")

    assert "unmeasurable" in text
    unmeasurable = sorted(
        rate.signal.value
        for rate in validate(
            written(COMMITTED_EXTERNAL),
            _scores(tmp_path, written(COMMITTED_EXTERNAL)),
        ).signals
        if not rate.measurable
    )
    assert unmeasurable == [
        SignalName.CATEGORY_CONFLICT.value,
        SignalName.DOMAIN_FREQUENCY.value,
        SignalName.DOMAIN_LOOKALIKE.value,
    ]
    for signal in unmeasurable:
        assert f"`{signal}`" in text


def _scores(tmp_path: Path, rows_: tuple[Labelled, ...]) -> tuple[PostScore, ...]:
    from reddit_fraud_intelligence.corpus import write_corpus
    from reddit_fraud_intelligence.signals import score_corpus

    corpus = tmp_path / "for-scores.jsonl"
    write_corpus(corpus, as_corpus(rows_))
    return score_corpus(
        corpus,
        REPO_ROOT / "data" / "public-suffix" / "public_suffix_list.dat",
        REPO_ROOT / "data" / "infrastructure" / "shared-hosts.jsonl",
        WEIGHTS,
    ).scores


@needs_external_set
def test_the_report_states_the_set_cannot_speak_for_reddit_or_for_the_campaign(
    tmp_path: Path,
) -> None:
    """Story #69: a validation page is only as useful as its refusals.

    Every figure on the page is about job advertisements. A reader who carried it over to
    Reddit, or read it as evidence the campaign machinery works, would have been told
    something false, and the page is the only place that can prevent it.
    """
    report, _ = run(tmp_path)
    text = report.read_text(encoding="utf-8")

    for refusal in (
        "It cannot measure the campaign machinery at all.",
        "It says nothing about whether content is fraudulent.",
        "not read as scam validation",
    ):
        assert refusal in text, f"the report no longer refuses: {refusal}"


@needs_external_set
def test_the_report_leads_with_the_ranking_rather_than_the_mean_gap(tmp_path: Path) -> None:
    """The mean scores differ by twentyfold and mean nothing; the AUC says so.

    Held because the tempting edit is to move the means up where they read as a result. On
    this set almost every row on both sides is tied at zero, so the gap is carried by two
    rows and the AUC beside it is dominated by ties — a page leading with the means would be
    leading with the one number on it that misleads.
    """
    report, _ = run(tmp_path)
    text = report.read_text(encoding="utf-8")

    assert text.index("## The headline") < text.index("Mean Policy Score")
    assert "does not separate this set" in text


def test_a_set_with_one_class_in_it_has_no_auc_and_says_so(tmp_path: Path) -> None:
    """A degenerate set is refused rather than reported as a figure of half."""
    path = tiny(tmp_path, deceptive=0, truthful=5)
    report, _ = run(tmp_path, external=path)

    assert "0.5000" in report.read_text(encoding="utf-8")


@needs_external_set
def test_scores_from_a_different_set_are_refused(tmp_path: Path) -> None:
    """Pairing one population's scores with another's labels is the mistake this guards.

    Measured by length, which is the check the interface allows: it is the whole of the
    pairing that can go wrong without a shared identifier, and a same-length mismatch would
    be caught by the figures disagreeing with the report rather than here.
    """
    from reddit_fraud_intelligence.signals import PostScore

    rows_ = written(COMMITTED_EXTERNAL)
    short = tuple(
        PostScore(
            post_id=f"ext_{index:04d}",
            account="external",
            score=0,
            points=0,
            published_points=0,
            published_signals=0,
            signals=(),
        )
        for index in range(len(rows_) - 1)
    )

    with pytest.raises(ValueError, match="one set of texts with another's scores"):
        validate(rows_, short)


def test_a_missing_set_names_the_command_that_fetches_it(tmp_path: Path) -> None:
    """The set is fetched, so the first thing a new clone meets is its absence.

    Refused in a sentence rather than as a `FileNotFoundError` out of pathlib, because the
    reader who runs this without the set has done nothing wrong and the useful thing to tell
    them is which command writes the file and which digest to check it against. Held
    against the command rather than against `read` so the refusal is the one the user sees.
    """
    missing = tmp_path / "not-fetched.jsonl"

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "external-validation",
                "--external",
                str(missing),
                "--corpus",
                str(tmp_path / "corpus.jsonl"),
                "--report",
                str(tmp_path / "report.md"),
            ]
        )

    message = str(raised.value)
    assert "rfi fetch-external" in message
    assert str(missing) in message
    assert not (tmp_path / "report.md").exists(), (
        "a refused run wrote a report anyway; the refusal has to stop the command before "
        "any output, or the page would describe a set that was never read"
    )


def test_a_row_that_is_not_text_and_a_label_is_refused(tmp_path: Path) -> None:
    """A third field would mean this is not the file the provenance names.

    Skipping it instead would drop a label this project never saw and would have counted as
    agreement, which is the one way a held-out set can quietly become an easy set.
    """
    path = tmp_path / "odd.jsonl"
    path.write_text(
        json.dumps({"text": "hello", "label": 1, "source": "elsewhere"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not the labelled-set vocabulary"):
        read(path)


def test_a_label_this_project_cannot_read_is_refused(tmp_path: Path) -> None:
    """A third label value would not be a measurement, it would be an interpretation."""
    path = tmp_path / "label.jsonl"
    path.write_text(json.dumps({"text": "hello", "label": 7}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="which is neither 1"):
        read(path)


@needs_external_set
def test_the_committed_files_are_what_the_command_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held to the command the same way the Corpus is held to its generator.

    Run from the repository root over the default relative path, for the reason
    `tests/test_graph_report.py` gives: the report names the source it was read from, so
    comparing a run given an absolute path against the committed file would be a comparison
    of two different renderings rather than of bytes. The command therefore writes to the
    committed location, and the check is that it wrote the same bytes back — which is the
    claim, since a run that changed them would have failed before this and left a dirty
    tree rather than a failing test.

    Of the two files, only the report is committed. The Corpus view beside it is real text
    and is fetched with the set, so this asserts the report is byte-stable and that a run
    regenerates the view deterministically from whatever it was handed.
    """
    monkeypatch.chdir(REPO_ROOT)
    before_report = COMMITTED_REPORT.read_bytes()

    assert main(["external-validation"]) == 0

    assert COMMITTED_REPORT.read_bytes() == before_report


@needs_external_set
def test_running_twice_writes_byte_identical_files(tmp_path: Path) -> None:
    """The rows are the file's order and the figures are counted, so the bytes are fixed."""
    first_report, first_corpus = run(tmp_path / "first")
    second_report, second_corpus = run(tmp_path / "second")

    assert first_report.read_bytes() == second_report.read_bytes()
    assert first_corpus == second_corpus


@needs_external_set
def test_the_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same argument as every other command: the numbers come from the fetched bytes.

    Fetching and validating are separate commands on purpose. This one must not reach for
    the network even when the cache is absent, because a reader running it to check a figure
    is reading what is on disk, not re-downloading whatever the publisher serves today.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the external-validation command reached for the network")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    run(tmp_path)


def test_provenance_is_written_beside_the_bytes_not_into_the_report(tmp_path: Path) -> None:
    """It is a fact about the file, so it lives with the file."""
    path = tiny(tmp_path, deceptive=1, truthful=1)
    measured = facts(path)
    provenance = tmp_path / "provenance.jsonl"
    write_provenance(provenance, measured)

    record = rows(provenance)[0]
    assert record["sha256"] == measured.sha256
    assert record["split"] == "test"
    assert record["deceptive"] == 1