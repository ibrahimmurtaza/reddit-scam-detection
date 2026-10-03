"""Recovery of Planted Campaigns: X of N, joined against what the grouping produced.

This is the project's one substantive claim, and the test file is built around the
two things that make it worth anything: the join counts a recovery only on whole
membership, and the number is computed by a command that runs after the grouping
rather than inside it.

Seam under test: the `campaign-recovery` command, observed through the two files it
writes and the table it prints. Nothing here inspects the code that wrote them.

The Corpus cases are written inside the tests, small enough to read in full, because
the interesting outcomes are the ones the shipped Corpus does not happen to produce:
two accounts of a three-account campaign grouped on their registration, a campaign
with an extra account inside its candidate, a campaign no candidate names at all. The
committed Corpus is then asked for its own figure, which is two of two with one shop
beside it, and the whole chain is regenerated at the seed the report names to show the
number is a function of bytes rather than of a run.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import (
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_CORPUS_PATH,
    DEFAULT_NUISANCE_PATH,
    DEFAULT_SEED,
    DEFAULT_TRUTH_PATH,
    main,
)
from reddit_fraud_intelligence.nuisance import NuisanceKind

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_CANDIDATES = REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl"
COMMITTED_TRUTH = REPO_ROOT / "data" / "corpus" / "truth.jsonl"
COMMITTED_NUISANCE = REPO_ROOT / "data" / "corpus" / "nuisance.jsonl"
COMMITTED_SHARED_HOSTS = REPO_ROOT / "data" / "infrastructure" / "shared-hosts.jsonl"
COMMITTED_RECOVERY = REPO_ROOT / "data" / "evaluation" / "recovery.jsonl"
COMMITTED_REPORT = REPO_ROOT / "docs" / "campaign-recovery.md"

Row = Mapping[str, object]

# The word ADR-0004 rules out of every view this project publishes. It is named here
# because the tests directory is where the project's own prose is allowed to use the
# phrases the prose it publishes may not use.
BANNED_FIGURE = "accuracy"


class _Opened:
    """Records every path a run opens, so a claim about what it read can be checked.

    An audit hook rather than a monkeypatch, because the point is to catch a read from
    anywhere at all — including from inside the standard library, which a patched
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

    def names(self) -> set[str]:
        """Every data file opened, by name, which is how a claim is checked."""
        return {
            Path(path).name
            for path in self.paths
            if Path(path).suffix in {".jsonl", ".dat"}
        }


def run(
    directory: Path,
    corpus: Path = DEFAULT_CORPUS_PATH,
    candidates: Path = DEFAULT_CANDIDATES_PATH,
    truth: Path = DEFAULT_TRUTH_PATH,
    nuisance: Path = DEFAULT_NUISANCE_PATH,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> tuple[Path, Path, str]:
    """Run the command, writing both outputs into `directory`.

    The inputs default to the command's own defaults rather than to this file's
    absolute paths, because the report prints the path it read: a report generated
    from an absolute path could not be held to the committed one.
    """
    recovery = directory / "recovery.jsonl"
    report = directory / "campaign-recovery.md"
    exit_code = main(
        [
            "campaign-recovery",
            "--corpus",
            str(corpus),
            "--candidates",
            str(candidates),
            "--truth",
            str(truth),
            "--nuisance",
            str(nuisance),
            "--recovery",
            str(recovery),
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out if capsys is not None else ""
    return recovery, report, printed


def rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} should be text, is {value!r}"
    return value


def texts(row: Row, field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    return [str(entry) for entry in value]


def records(row: Row, field: str) -> list[Row]:
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    return [entry for entry in value if isinstance(entry, dict)]


def write_rows(path: Path, written: Iterable[Row]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for row in written
        ),
        encoding="utf-8",
    )
    return path


def post(post_id: str, account: str) -> Row:
    return {
        "post_id": post_id,
        "account": account,
        "subreddit": "test",
        "title": f"title of {post_id}",
        "body": "body",
        "created_at": "2026-01-05T09:00:00Z",
        "links": [f"https://{post_id}.example/note"],
    }


def campaign(campaign_id: str, accounts: Iterable[str], posts: Iterable[str]) -> Row:
    return {
        "campaign_id": campaign_id,
        "accounts": sorted(accounts),
        "posts": sorted(posts),
    }


def candidate(
    candidate_id: str, accounts: Iterable[str], posts: Iterable[str], *domains: str
) -> Row:
    accounts = sorted(accounts)
    return {
        "candidate_id": candidate_id,
        "accounts": accounts,
        "posts": sorted(posts),
        "shared_domains": [
            {"domain": domain, "accounts": accounts, "posts": sorted(posts)}
            for domain in domains
        ],
        "first_seen": "2026-01-05T09:00:00Z",
    }


def nuisance(nuisance_id: str, kind: NuisanceKind, *accounts: str) -> Row:
    return {
        "kind": kind.value,
        "nuisance_id": nuisance_id,
        "accounts": sorted(accounts),
        "posts": [],
        "hosts": [],
        "characters": [],
        "note": "Written by this test.",
    }


def corpus_of(directory: Path, *written: Row) -> Path:
    return write_rows(directory / "corpus.jsonl", written)


def figure(printed: str) -> str:
    """The figure line of the table, as the sentence beside the label rather than the
    label and whatever padding the widest label in the block happens to need."""
    line = next(line for line in printed.splitlines() if line.strip().startswith("recovered"))
    return line.split(maxsplit=1)[1].strip()


def headline(page: str) -> str:
    """The one bold line of the report: the figure as the page states it."""
    return next(line for line in page.splitlines() if line.startswith("**"))


def digests(page: str) -> list[str]:
    """Every SHA-256 the report quotes, in the order it quotes them."""
    return re.findall(r"\b[0-9a-f]{64}\b", page)


# --- what counts as a recovery ----------------------------------------------------


def test_a_candidate_holding_a_campaigns_whole_membership_is_a_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The figure the ticket asks for, over a Corpus small enough to check by hand.

    Two accounts, one registration, one candidate, one Planted Campaign. Every account
    of the campaign is in the candidate and the candidate holds nothing else, which is
    what matching the membership means. The Recovery row says so and names the candidate
    it was matched on, so the claim is checkable rather than a count.
    """
    corpus = corpus_of(
        tmp_path,
        post("syn_p_0001", "syn_alpha_0001"),
        post("syn_p_0002", "syn_beta_0002"),
    )
    candidates = write_rows(
        tmp_path / "campaign-candidates.jsonl",
        [candidate("cc-01", ("syn_alpha_0001", "syn_beta_0002"), ("syn_p_0001", "syn_p_0002"))],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [campaign("syn-campaign-one", ("syn_alpha_0001", "syn_beta_0002"), ("syn_p_0001",))],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )

    recovery, _, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=truth,
        nuisance=nuisance_path,
        capsys=capsys,
    )
    recovered = rows(recovery)

    assert figure(printed) == "1 of 1 Planted Campaigns"
    assert len(recovered) == 1
    assert text(recovered[0], "outcome") == "recovered"
    assert texts(recovered[0], "accounts") == [
        "syn_alpha_0001",
        "syn_beta_0002",
    ]
    match = records(recovered[0], "candidates")[0]
    assert text(match, "candidate_id") == "cc-01"
    assert texts(match, "missing") == []
    assert texts(match, "extra") == []


def test_a_candidate_holding_only_some_of_a_campaigns_accounts_is_a_partial_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The figure a system could report if it were flattering itself.

    Three accounts in the membership, two of them in one candidate, and a third in
    another campaign's candidate. Two of the three are genuinely grouped on their own
    registration, which is a real grouping and not a recovery: the campaign is split,
    so it is neither recovered nor missed. Counting it either way is what makes an X of
    N figure mean something, and the missing account is named so the split is visible.
    """
    corpus = corpus_of(
        tmp_path,
        post("syn_p_0001", "syn_alpha_0001"),
        post("syn_p_0002", "syn_beta_0002"),
        post("syn_p_0003", "syn_gamma_0003"),
        post("syn_p_0004", "syn_delta_0004"),
    )
    candidates = write_rows(
        tmp_path / "campaign-candidates.jsonl",
        [
            candidate("cc-01", ("syn_alpha_0001", "syn_beta_0002"), ("syn_p_0001", "syn_p_0002")),
            candidate("cc-02", ("syn_gamma_0003", "syn_delta_0004"), ("syn_p_0003", "syn_p_0004")),
        ],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [
            campaign(
                "syn-campaign-one",
                ("syn_alpha_0001", "syn_beta_0002", "syn_gamma_0003"),
                ("syn_p_0001", "syn_p_0002", "syn_p_0003"),
            )
        ],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )

    recovery, report, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=truth,
        nuisance=nuisance_path,
        capsys=capsys,
    )
    recovered = rows(recovery)
    page = report.read_text(encoding="utf-8")

    assert figure(printed) == "0 of 1 Planted Campaigns"
    assert text(recovered[0], "outcome") == "partial"
    matches = records(recovered[0], "candidates")
    assert [text(match, "candidate_id") for match in matches] == ["cc-01", "cc-02"]
    assert texts(matches[0], "held") == ["syn_alpha_0001", "syn_beta_0002"]
    assert texts(matches[0], "missing") == ["syn_gamma_0003"]
    assert texts(matches[1], "held") == ["syn_gamma_0003"]
    # Named in the reader-facing page, not only in the file.
    assert "syn_gamma_0003" in page
    assert "partial" in page


def test_a_candidate_holding_a_campaign_and_an_account_of_its_own_is_not_a_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One account too many is an over-grouping, and it is not half a recovery either.

    The campaign's two accounts are together and a third account nobody planted with
    them. Every account of the campaign is in the candidate, so a rule that only asked
    whether the membership was covered would call this a recovery — and the extra
    account is exactly what a reviewer needs to see, because it is the grouping being
    wrong in the direction that matters.
    """
    corpus = corpus_of(
        tmp_path,
        post("syn_p_0001", "syn_alpha_0001"),
        post("syn_p_0002", "syn_beta_0002"),
        post("syn_p_0003", "syn_gamma_0003"),
    )
    candidates = write_rows(
        tmp_path / "campaign-candidates.jsonl",
        [
            candidate(
                "cc-01",
                ("syn_alpha_0001", "syn_beta_0002", "syn_gamma_0003"),
                ("syn_p_0001", "syn_p_0002", "syn_p_0003"),
            )
        ],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [
            campaign(
                "syn-campaign-one",
                ("syn_alpha_0001", "syn_beta_0002"),
                ("syn_p_0001", "syn_p_0002"),
            )
        ],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )

    recovery, report, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=truth,
        nuisance=nuisance_path,
        capsys=capsys,
    )
    recovered = rows(recovery)
    match = records(recovered[0], "candidates")[0]

    assert figure(printed) == "0 of 1 Planted Campaigns"
    assert text(recovered[0], "outcome") == "partial"
    assert texts(match, "held") == ["syn_alpha_0001", "syn_beta_0002"]
    assert texts(match, "missing") == []
    assert texts(match, "extra") == ["syn_gamma_0003"]
    assert "syn_gamma_0003" in report.read_text(encoding="utf-8")


def test_a_campaign_no_candidate_names_is_a_miss_that_names_its_accounts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A miss has to be diagnosable, so the accounts it lost are printed with it.

    A campaign whose accounts each sit on registrations nobody else reaches is not in
    any candidate at all. That is a miss against N, and it is the one outcome a reader
    can act on, so the report says which accounts were not grouped rather than counting
    a miss and moving on.
    """
    corpus = corpus_of(
        tmp_path,
        post("syn_p_0001", "syn_alpha_0001"),
        post("syn_p_0002", "syn_beta_0002"),
    )
    candidates = write_rows(tmp_path / "campaign-candidates.jsonl", [])
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [
            campaign(
                "syn-campaign-one",
                ("syn_alpha_0001", "syn_beta_0002"),
                ("syn_p_0001", "syn_p_0002"),
            )
        ],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )

    recovery, report, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=truth,
        nuisance=nuisance_path,
        capsys=capsys,
    )
    recovered = rows(recovery)
    page = report.read_text(encoding="utf-8")

    assert figure(printed) == "0 of 1 Planted Campaigns"
    assert text(recovered[0], "outcome") == "missed"
    assert records(recovered[0], "candidates") == []
    assert texts(recovered[0], "accounts") == ["syn_alpha_0001", "syn_beta_0002"]
    for account in ("syn_alpha_0001", "syn_beta_0002"):
        assert account in page


def test_the_three_outcomes_partition_the_planted_campaigns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Recovered, partial, and missed add up to N, because every campaign gets one.

    The partition is what makes X of N readable: a campaign that is neither recovered
    nor reported is the failure mode where a figure reads well and the arithmetic
    behind it does not add up. Three campaigns, three outcomes, one row each.
    """
    corpus = corpus_of(
        tmp_path,
        post("syn_p_0001", "syn_alpha_0001"),
        post("syn_p_0002", "syn_beta_0002"),
        post("syn_p_0003", "syn_gamma_0003"),
        post("syn_p_0004", "syn_delta_0004"),
        post("syn_p_0005", "syn_epsilon_0005"),
        post("syn_p_0006", "syn_zeta_0006"),
    )
    candidates = write_rows(
        tmp_path / "campaign-candidates.jsonl",
        [
            candidate("cc-01", ("syn_alpha_0001", "syn_beta_0002"), ("syn_p_0001", "syn_p_0002")),
            candidate(
                "cc-02",
                ("syn_gamma_0003", "syn_epsilon_0005"),
                ("syn_p_0003", "syn_p_0005"),
            ),
        ],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [
            campaign("syn-campaign-one", ("syn_alpha_0001", "syn_beta_0002"), ("syn_p_0001",)),
            campaign(
                "syn-campaign-two",
                ("syn_gamma_0003", "syn_epsilon_0005", "syn_zeta_0006"),
                ("syn_p_0003", "syn_p_0005", "syn_p_0006"),
            ),
            campaign("syn-campaign-three", ("syn_delta_0004",), ("syn_p_0004",)),
        ],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )

    recovery, _, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=truth,
        nuisance=nuisance_path,
        capsys=capsys,
    )
    recovered = rows(recovery)
    counted = Counter(text(row, "outcome") for row in recovered)

    assert len(recovered) == 3
    assert counted == {"recovered": 1, "partial": 1, "missed": 1}
    assert figure(printed) == "1 of 3 Planted Campaigns"


# --- what the report says ---------------------------------------------------------


def test_the_report_names_no_figure_over_the_whole_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR-0004's rule, asked as a question about the output rather than the prose.

    Every label in this Corpus was assigned by the generator that wrote the posts, so a
    share of the posts called right or wrong would measure agreement with the generator.
    The figure is asserted absent from both views, and the committed page is included so
    the file a reader actually opens is the one checked.
    """
    _, report, printed = run(tmp_path, capsys=capsys)

    assert BANNED_FIGURE not in printed.lower()
    assert BANNED_FIGURE not in report.read_text(encoding="utf-8").lower()
    assert BANNED_FIGURE not in COMMITTED_REPORT.read_text(encoding="utf-8").lower()


def test_the_report_says_who_measured_it_and_that_no_person_did(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reviewer in this project is a fiction, and the page has to say so.

    A reader who opens the report has to be able to tell that the figure was computed by
    a command against membership the generator wrote, and not by somebody clicking
    Confirm on synthetic posts. It also has to keep the two apart in the other direction:
    a Campaign Candidate is a proposal, and the membership it is scored against is
    planted.
    """
    _, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    for view in (page, printed):
        assert "Measured by `rfi campaign-recovery`" in view
        assert "No person reviewed any of it" in view
        assert "Planted Campaign membership" in view


def test_the_report_states_the_nuisance_structure_it_was_measured_against(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """X of N beside nothing is a figure about a generator that planted two campaigns.

    The manifest is read, every kind it holds is named, and each kind's accounts and
    posts are counted, so the baseline the recovery rate sits on is in the output rather
    than in a ticket. The rate of groupings that are not recoveries is measured
    separately, and the page says where it comes from rather than implying the figure
    stands alone.
    """
    _, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")
    manifest = rows(COMMITTED_NUISANCE)

    for view in (page, printed):
        assert "data/corpus/nuisance.jsonl" in view
        for record in manifest:
            assert text(record, "kind") in view

    for kind in NuisanceKind:
        of_kind = [row for row in manifest if text(row, "kind") == kind.value]
        assert of_kind, f"the manifest holds no {kind.value}"
        assert f"| `{kind.value}` | {len(of_kind)} |" in page
    assert "companion figure" in page


def test_a_partial_match_beside_a_recovery_is_reported_in_every_view(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The case where the campaign was recovered and something else also happened.

    One candidate holds the whole membership, so the figure is 1 of 1, and a second
    candidate holds two of the three accounts. The recovery is the outcome, and the
    second candidate is a real grouping of real accounts that is not part of it — so the
    console has to name it as well. A console that printed only the outcome would say
    `partial 0` beside two candidates, which is the one combination of the two views
    that could hide a grouping being wrong.
    """
    corpus = corpus_of(
        tmp_path,
        post("syn_p_0001", "syn_alpha_0001"),
        post("syn_p_0002", "syn_beta_0002"),
        post("syn_p_0003", "syn_gamma_0003"),
    )
    candidates = write_rows(
        tmp_path / "campaign-candidates.jsonl",
        [
            candidate(
                "cc-01",
                ("syn_alpha_0001", "syn_beta_0002", "syn_gamma_0003"),
                ("syn_p_0001", "syn_p_0002", "syn_p_0003"),
            ),
            candidate(
                "cc-02",
                ("syn_alpha_0001", "syn_beta_0002"),
                ("syn_p_0001", "syn_p_0002"),
            ),
        ],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [
            campaign(
                "syn-campaign-one",
                ("syn_alpha_0001", "syn_beta_0002", "syn_gamma_0003"),
                ("syn_p_0001", "syn_p_0002", "syn_p_0003"),
            )
        ],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )

    recovery, report, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=truth,
        nuisance=nuisance_path,
        capsys=capsys,
    )
    recovered = rows(recovery)
    page = report.read_text(encoding="utf-8")

    assert figure(printed) == "1 of 1 Planted Campaigns"
    assert text(recovered[0], "outcome") == "recovered"
    partial = next(
        line for line in printed.splitlines() if line.strip().startswith("partial")
    )
    assert partial.split(maxsplit=1)[1] == (
        "0 Planted Campaigns, 1 of 2 candidates hold only part of one"
    )
    # The console names the second candidate and what it failed to hold.
    assert "cc-02" in printed
    assert "syn_gamma_0003" in printed
    assert "cc-02" in page


def test_the_console_output_is_ascii_and_wrapped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The console is pasted into issues and diffed between runs, so it has to be one
    thing on every terminal.

    The claim every table in this project makes about its own output: ASCII only, so a
    console that cannot encode anything else prints it the same way as one that can, and
    wrapped at the width a terminal gives you, so nothing is one line of three hundred
    columns that has to be re-wrapped by hand before it goes in an issue. Asserted on
    the output rather than on the source, because the claim is about what prints.
    """
    recovery, report, printed = run(tmp_path, capsys=capsys)

    assert printed.isascii(), "the console output carries a character a console may not have"
    # The two lines the command echoes name the paths the caller chose, which it does
    # not control; everything else is the table's own text.
    echoed = (str(recovery), str(report))
    too_wide = [
        line
        for line in printed.splitlines()
        if len(line) > 100 and not any(path in line for path in echoed)
    ]
    assert not too_wide, f"{len(too_wide)} lines over 100 columns: {too_wide[:2]}"


# --- the boundary between the two steps -------------------------------------------


def test_the_evaluation_reads_the_membership_and_the_grouping_never_does(
    tmp_path: Path,
) -> None:
    """The two steps are separate commands, checked by watching what each one opens.

    The grouping opens the Corpus and the two published lists and writes the candidates;
    it opens no membership and no manifest, which is what makes the membership an answer
    key rather than an input. The evaluation reads the membership and the manifest, and
    reads the candidates the grouping published — and opens neither the Public Suffix
    List nor the shared-infrastructure list, because it does no grouping of its own and
    cannot reach inference even by accident.
    """
    opened = _Opened()
    sys.addaudithook(opened)
    grouped = tmp_path / "campaign-candidates.jsonl"

    opened.recording = True
    try:
        assert main(
            [
                "campaign-candidates",
                "--corpus",
                str(DEFAULT_CORPUS_PATH),
                "--candidates",
                str(grouped),
            ]
        ) == 0
    finally:
        opened.recording = False
    grouping = opened.names()

    opened.paths.clear()
    opened.recording = True
    try:
        run(tmp_path / "out", candidates=grouped)
    finally:
        opened.recording = False
    evaluation = opened.names()

    assert not [name for name in grouping if name.startswith(("truth", "nuisance"))]
    assert "campaign-candidates.jsonl" in grouping

    assert {"truth.jsonl", "nuisance.jsonl", "campaign-candidates.jsonl"} <= evaluation
    assert "public_suffix_list.dat" not in evaluation
    assert "shared-hosts.jsonl" not in evaluation


def test_the_evaluation_refuses_to_run_before_the_grouping_has(
    tmp_path: Path,
) -> None:
    """The step the measurement comes after has to have happened, and the refusal says so.

    Recovery is measured against what the grouping produced, so a candidates file that
    is not there is a measurement of nothing. The run stops and names the command that
    produces it, rather than reporting zero of two and calling it a result.
    """
    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "campaign-recovery",
                "--corpus",
                str(DEFAULT_CORPUS_PATH),
                "--candidates",
                str(tmp_path / "absent.jsonl"),
                "--recovery",
                str(tmp_path / "recovery.jsonl"),
                "--report",
                str(tmp_path / "campaign-recovery.md"),
            ]
        )

    assert "rfi campaign-candidates" in str(refusal.value)
    assert not (tmp_path / "recovery.jsonl").exists()


def test_the_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same argument as every other command here: the figures come from committed bytes."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the campaign-recovery command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    run(tmp_path)


# --- the Corpus the project actually ships ---------------------------------------


def test_the_two_planted_campaigns_are_recovered_and_the_shop_is_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shipped figure, which a reader can check against three candidates by hand.

    `cc-01` holds all of alpha's three accounts and `cc-03` all of beta's two, so the
    figure is 2 of 2. `cc-02` is three accounts of one shop sharing one domain: ADR-0005
    says they group, the grouping is right to produce them, and they are not planted, so
    they are printed as a candidate holding no Planted Campaign rather than counted
    against N — which would be the system's correct behaviour reported as a failure.
    """
    recovery, report, printed = run(tmp_path, capsys=capsys)
    recovered = rows(recovery)
    page = report.read_text(encoding="utf-8")

    assert figure(printed) == "2 of 2 Planted Campaigns"
    assert [text(row, "campaign_id") for row in recovered] == [
        "syn-campaign-alpha",
        "syn-campaign-beta",
    ]
    assert {text(row, "outcome") for row in recovered} == {"recovered"}
    assert [
        text(records(row, "candidates")[0], "candidate_id") for row in recovered
    ] == ["cc-01", "cc-03"]

    assert "1 of 3 candidates" in printed
    assert "cc-02" in printed
    assert "rivermill-bikes.example" in page
    assert "2 of 2 Planted Campaigns" in page


def test_the_figure_is_reproducible_from_the_seed_the_report_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The number is a function of the seed, checked by regenerating everything.

    The seed is read out of the report rather than out of the module, so a report that
    named the wrong seed would fail here rather than agree with itself. The whole chain
    is then rebuilt from it: the generator writes four files, the grouping writes the
    candidates, and this command joins them again. Same figure, and the same bytes.
    """
    first_recovery, first_report, first_printed = run(tmp_path / "first", capsys=capsys)
    named = re.search(r"seed\s+(\d+)", first_report.read_text(encoding="utf-8"))
    assert named is not None, "the report does not name the seed it was measured at"
    seed = int(named.group(1))
    assert seed == DEFAULT_SEED

    generated = tmp_path / "generated"
    assert main(
        [
            "generate-corpus",
            "--seed",
            str(seed),
            "--corpus",
            str(generated / "corpus.jsonl"),
            "--truth",
            str(generated / "truth.jsonl"),
            "--nuisance",
            str(generated / "nuisance.jsonl"),
            "--shared-infrastructure",
            str(generated / "shared-hosts.jsonl"),
        ]
    ) == 0

    for written, committed in (
        (generated / "corpus.jsonl", COMMITTED_CORPUS),
        (generated / "truth.jsonl", COMMITTED_TRUTH),
        (generated / "nuisance.jsonl", COMMITTED_NUISANCE),
        (generated / "shared-hosts.jsonl", COMMITTED_SHARED_HOSTS),
    ):
        assert written.read_bytes() == committed.read_bytes(), committed.name

    rebuilt = tmp_path / "rebuilt"
    assert main(
        [
            "campaign-candidates",
            "--corpus",
            str(generated / "corpus.jsonl"),
            "--candidates",
            str(rebuilt / "campaign-candidates.jsonl"),
        ]
    ) == 0
    rebuilt_recovery, rebuilt_report, rebuilt_printed = run(
        tmp_path / "out",
        corpus=generated / "corpus.jsonl",
        candidates=rebuilt / "campaign-candidates.jsonl",
        truth=generated / "truth.jsonl",
        nuisance=generated / "nuisance.jsonl",
        capsys=capsys,
    )

    assert figure(rebuilt_printed) == figure(first_printed)
    assert rebuilt_recovery.read_bytes() == first_recovery.read_bytes()
    # The two reports name the four paths they read, and those differ by construction
    # here — the same bytes at a different path. The figure they carry does not, so
    # that is what is compared; the digests inside are compared too, and they do match,
    # because the files are byte-identical.
    assert headline(rebuilt_report.read_text(encoding="utf-8")) == headline(
        first_report.read_text(encoding="utf-8")
    )
    assert digests(rebuilt_report.read_text(encoding="utf-8")) == digests(
        first_report.read_text(encoding="utf-8")
    )


# --- the shape of the output ------------------------------------------------------


def test_the_committed_outputs_are_what_the_command_writes(tmp_path: Path) -> None:
    """Held to the command the same way the Corpus is held to its generator."""
    recovery, report, _ = run(tmp_path)

    assert recovery.read_bytes() == COMMITTED_RECOVERY.read_bytes()
    assert report.read_bytes() == COMMITTED_REPORT.read_bytes()


def test_running_twice_writes_byte_identical_bytes(tmp_path: Path) -> None:
    first = run(tmp_path / "first")
    second = run(tmp_path / "second")

    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()


def test_the_console_names_the_candidate_that_recovered_each_campaign(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table is the first thing a reader meets, so it has to carry the claim.

    Index of the Planted Campaigns with their outcome, then the candidate each one was
    matched on or no candidate at all. The Recovery file is the same join with the same
    rows, so a reader can check the table against the file rather than trust either.
    """
    recovery, _, printed = run(tmp_path, capsys=capsys)
    recovered = {text(row, "campaign_id"): row for row in rows(recovery)}

    assert "syn-campaign-alpha" in printed
    for campaign_id, row in recovered.items():
        assert campaign_id in printed
        match = records(row, "candidates")[0]
        assert text(match, "candidate_id") in printed


# --- refusals ---------------------------------------------------------------------


def test_the_run_refuses_a_membership_the_corpus_does_not_hold(
    tmp_path: Path,
) -> None:
    """X of N against a membership the Corpus does not contain is not a measurement.

    A truth file naming an account no post was written by is a membership that cannot be
    recovered by any grouping, and a truth file naming a post written by an account
    outside the campaign is not a membership at all. Either way the figure would be
    arithmetically fine and meaningless, so the run stops and says which file is wrong.
    """
    corpus = corpus_of(tmp_path, post("syn_p_0001", "syn_alpha_0001"))
    candidates = write_rows(tmp_path / "campaign-candidates.jsonl", [])
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [campaign("syn-campaign-one", ("syn_ghost_9999",), ("syn_p_0001",))],
    )

    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "campaign-recovery",
                "--corpus",
                str(corpus),
                "--candidates",
                str(candidates),
                "--truth",
                str(truth),
                "--nuisance",
                str(nuisance_path),
                "--recovery",
                str(tmp_path / "recovery.jsonl"),
                "--report",
                str(tmp_path / "campaign-recovery.md"),
            ]
        )

    assert "syn_ghost_9999" in str(refusal.value)
    assert not (tmp_path / "recovery.jsonl").exists()


def test_the_run_refuses_a_candidates_file_carrying_a_field_it_does_not_know(
    tmp_path: Path,
) -> None:
    """A membership field in the pipeline's output would undo ADR-0008 from the far side.

    The evaluator is the only command that may read the truth file, so the published
    candidates must not carry membership of its own. A row with a `campaign_id` on it is
    refused rather than read and ignored: quietly ignoring a field that has appeared is
    the first way to stop being able to say the file does not carry one.
    """
    corpus = corpus_of(
        tmp_path,
        post("syn_p_0001", "syn_alpha_0001"),
        post("syn_p_0002", "syn_beta_0002"),
    )
    candidates = write_rows(
        tmp_path / "campaign-candidates.jsonl",
        [
            {
                **candidate(
                    "cc-01",
                    ("syn_alpha_0001", "syn_beta_0002"),
                    ("syn_p_0001", "syn_p_0002"),
                ),
                "campaign_id": "syn-campaign-one",
            }
        ],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [campaign("syn-campaign-one", ("syn_alpha_0001", "syn_beta_0002"), ("syn_p_0001",))],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )

    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "campaign-recovery",
                "--corpus",
                str(corpus),
                "--candidates",
                str(candidates),
                "--truth",
                str(truth),
                "--nuisance",
                str(nuisance_path),
                "--recovery",
                str(tmp_path / "recovery.jsonl"),
                "--report",
                str(tmp_path / "campaign-recovery.md"),
            ]
        )

    assert "campaign_id" in str(refusal.value)


def test_the_command_refuses_to_write_the_report_over_the_membership(
    tmp_path: Path,
) -> None:
    """The membership is the input the measurement rests on, and a page written over it
    would leave the next run measuring a report."""
    truth = tmp_path / "truth.jsonl"

    with pytest.raises(SystemExit):
        main(
            [
                "campaign-recovery",
                "--corpus",
                str(DEFAULT_CORPUS_PATH),
                "--candidates",
                str(DEFAULT_CANDIDATES_PATH),
                "--truth",
                str(truth),
                "--nuisance",
                str(DEFAULT_NUISANCE_PATH),
                "--recovery",
                str(tmp_path / "recovery.jsonl"),
                "--report",
                str(truth),
            ]
        )

    assert not truth.exists()