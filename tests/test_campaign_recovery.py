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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pytest

from conftest import SCRATCH_TABLE, needs_database, seed_vectors
from reddit_fraud_intelligence.cli import (
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_CORPUS_PATH,
    DEFAULT_NUISANCE_PATH,
    DEFAULT_SEED,
    DEFAULT_SCORES_PATH,
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
    scores: Path = DEFAULT_SCORES_PATH,
    depths: Sequence[int] | None = None,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> tuple[Path, Path, str]:
    """Run the command, writing both outputs into `directory`.

    The inputs default to the command's own defaults rather than to this file's
    absolute paths, because the report prints the path it read: a report generated
    from an absolute path could not be held to the committed one.
    """
    recovery = directory / "recovery.jsonl"
    report = directory / "campaign-recovery.md"
    argv = [
        "campaign-recovery",
        "--corpus",
        str(corpus),
        "--candidates",
        str(candidates),
        "--truth",
        str(truth),
        "--nuisance",
        str(nuisance),
        "--scores",
        str(scores),
        "--recovery",
        str(recovery),
        "--report",
        str(report),
    ]
    if depths is not None:
        argv += ["--depths", ",".join(str(depth) for depth in depths)]
    exit_code = main(argv)
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


def nested(row: Row, field: str) -> Row:
    value = row[field]
    assert isinstance(value, dict), f"{field} should be a row, is {value!r}"
    return value


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
    candidate_id: str,
    accounts: Iterable[str],
    posts: Iterable[str],
    *domains: str,
    points: Iterable[str] = (),
    closest_seconds: int = 0,
) -> Row:
    """One row of `campaign-candidates.jsonl`, hand-written.

    `domains` and `points` are the two edges ADR-0005 allows and the reader refuses a
    row with neither, so a test that wanted a candidate resting on nothing would have to
    say so in the source rather than leave a row the command turns down. The timing is
    written too, and is checked by the reader against the evidence the row names — one
    gap per piece, a count of the pieces, and the nearest of the gaps — so a test cannot
    hand the evaluator a candidate whose clock disagrees with itself. Every piece here is
    written with a gap of nothing, which is what a Corpus of posts written at one instant
    gives; `closest_seconds` moves the whole row when a test needs a gap.
    """
    accounts = sorted(accounts)
    posts = sorted(posts)
    gaps = [{"kind": "registration", "value": domain,
             "closest_seconds": closest_seconds, "span_seconds": closest_seconds}
            for domain in domains]
    gaps += [{"kind": "Contact Point", "value": value,
              "closest_seconds": closest_seconds, "span_seconds": closest_seconds}
             for value in points]
    twins = [
        {
            "post": post,
            "account": accounts[index % len(accounts)],
            "nearest": posts[(index + 1) % len(posts)],
            "nearest_account": accounts[(index + 1) % len(accounts)],
            "distance": 0.0,
        }
        for index, post in enumerate(posts)
    ]
    return {
        "candidate_id": candidate_id,
        "accounts": accounts,
        "posts": posts,
        "shared_domains": [
            {"domain": domain, "accounts": accounts, "posts": posts} for domain in domains
        ],
        "shared_contact_points": [
            {
                "kind": "telegram",
                "value": value,
                "accounts": accounts,
                "posts": posts,
                "spellings": [f"@{value}"],
            }
            for value in points
        ],
        "corroboration": gaps,
        "timing": {
            "window_seconds": 86_400,
            "pieces": len(gaps),
            "corroborated": len(gaps),
            "closest_seconds": closest_seconds,
        },
        "similarity": twins,
        "content_similarity": {
            "threshold": 0.5,
            "pieces": len(posts),
            "corroborated": len(posts),
            "closest_distance": 0.0,
        },
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


# --- the recall bound --------------------------------------------------------------


def bound(printed: str) -> str:
    """The bound line of the figures block, without its label or the block's padding."""
    line = next(line for line in printed.splitlines() if line.strip().startswith("bound"))
    return line.split(maxsplit=1)[1].strip()


def rotator(
    tmp_path: Path,
    *,
    points: Iterable[str] = (),
    domains: Iterable[str] = ("rotator-free.example",),
) -> tuple[Path, Path, Path]:
    """Two campaigns, one of which shares nothing with itself and cannot be reached.

    `syn-campaign-shared` is a candidate holding two accounts on one registration or on
    one Contact Point; `syn-campaign-rotor` is two accounts that share neither, so no
    edge this method rests on joins them however hard the grouping tries. Written out
    rather than generated because the point of the test is the shape of the bound and a
    generator would be the wrong place to plant it.

    `domains` and `points` both default to something, so a caller wanting a candidate on
    one edge alone has to pass the other as empty rather than inherit both and claim the
    figure it sees came from the edge it meant to exercise.
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
            candidate(
                "cc-01",
                ("syn_alpha_0001", "syn_beta_0002"),
                ("syn_p_0001", "syn_p_0002"),
                *domains,
                points=points,
            )
        ],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [
            campaign(
                "syn-campaign-shared",
                ("syn_alpha_0001", "syn_beta_0002"),
                ("syn_p_0001", "syn_p_0002"),
            ),
            campaign(
                "syn-campaign-rotor",
                ("syn_gamma_0003", "syn_delta_0004"),
                ("syn_p_0003", "syn_p_0004"),
            ),
        ],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )
    return corpus, candidates, nuisance_path


def test_a_campaign_whose_accounts_share_nothing_is_in_the_recall_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bound is a count of campaigns, and it names the campaign it counts.

    `syn-campaign-rotor` rotates its registrations and publishes no Contact Point, so
    nothing about it appears in the candidates file at all — which is what makes it the
    floor rather than a miss the grouping could have caught. The figure beside it is a
    bound on what any method resting on these two edges could have recovered, so a
    reader can tell the difference between "the grouping was nearly right" and "no
    grouping of this shape could ever have reached it".
    """
    corpus, candidates, nuisance_path = rotator(tmp_path)

    _, report, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=tmp_path / "truth.jsonl",
        nuisance=nuisance_path,
        capsys=capsys,
    )
    page = report.read_text(encoding="utf-8")

    assert figure(printed) == "1 of 2 Planted Campaigns"
    assert bound(printed) == (
        "1 of 2 Planted Campaigns have nothing inside them reaching a candidate"
    )
    assert "syn-campaign-rotor" in printed
    assert "syn-campaign-shared" in printed
    assert "syn-campaign-rotor" in page
    assert "recall bound" in page
    assert "recall bound" in printed
    assert "ceiling of 1 of 2" in " ".join(printed.split())


def test_a_campaign_shared_on_a_contact_point_alone_is_outside_the_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bound counts what neither edge reaches, so one edge is enough to escape it.

    The same Corpus with the shared campaign resting on a Telegram handle and on no
    registration at all. Nothing about the bound changes: the campaign is still reachable,
    and the point of the test is that a reader checking the bound is not asked to know
    which edge did the work.

    The line the two edges are named on is asserted as well as the bound, because a
    candidate carrying both edges would escape the bound just the same and the test would
    pass while exercising nothing: the point of this one is the edge the second run adds.
    """
    corpus, candidates, nuisance_path = rotator(
        tmp_path, points=("syn_shared_desk",), domains=()
    )

    _, _, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=tmp_path / "truth.jsonl",
        nuisance=nuisance_path,
        capsys=capsys,
    )

    assert figure(printed) == "1 of 2 Planted Campaigns"
    assert bound(printed) == (
        "1 of 2 Planted Campaigns have nothing inside them reaching a candidate"
    )
    assert "syn-campaign-shared  Contact Points: syn_shared_desk" in printed


def test_the_bound_is_stated_as_a_bound_rather_than_as_a_rate_of_fraud_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC5: the ceiling is named as a ceiling, in both views.

    A recovery rate read as an estimate of how much fraud a system finds in the world
    would be a much larger claim than the number is. The bound is the honest form of it:
    however good the grouping gets, these campaigns cannot be recovered by this method,
    and the figure says how many they are rather than leaving a reader to infer the
    ceiling.
    """
    corpus, candidates, nuisance_path = rotator(tmp_path)

    _, report, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=tmp_path / "truth.jsonl",
        nuisance=nuisance_path,
        capsys=capsys,
    )
    page = report.read_text(encoding="utf-8")

    for view in (page, printed):
        flat = " ".join(view.split())
        assert "not an estimate of fraud found in the world" in flat
        assert "a limit on the method rather than a prediction of the run" in flat
    assert "ceiling of 1 of 2" in " ".join(printed.split())
    assert "ceiling of 1 of 2" in page


def test_the_two_views_of_the_bound_agree_on_the_word_goes_with_its_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One figure, two views, one pronoun — checked at the count where it breaks.

    The console and the page each render the bound, and each used to name the count's
    pronoun its own way: the page agreed it with the count and the console said "them"
    whatever the count was. At a bound of one, which is the case a small Corpus reaches
    first, the two views then made two grammatical claims about one run and neither was
    a figure a reader could check by adding.

    Both sentences are found by the clause they share rather than by exact text, because
    the views differ in their markdown and their line breaking and the claim here is about
    the word, not about the wrapping.
    """
    corpus, candidates, nuisance_path = rotator(tmp_path)

    _, report, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=tmp_path / "truth.jsonl",
        nuisance=nuisance_path,
        capsys=capsys,
    )
    page = report.read_text(encoding="utf-8")

    clause = "so no edge this method rests on can hold "

    def pronoun(view: str) -> str:
        """The word the sentence gives the count, stripped of what is joined to it."""
        flat = " ".join(view.split())
        after = flat.index(clause) + len(clause)
        return "".join(c for c in flat[after:].split(" ")[0] if c.isalpha())

    assert pronoun(printed) == pronoun(page)
    assert pronoun(printed) == "it"


def test_no_campaign_in_the_committed_corpus_is_beyond_the_method(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shipped figure, with the bound beside it and every campaign named.

    Zero of two: both Planted Campaigns share a registration and a Contact Point among
    their own accounts, so nothing in this Corpus is out of the method's reach by
    construction. The bound is printed even at zero, because a figure that appears only
    when it is bad is a figure a reader cannot tell apart from a missing one.
    """
    recovery, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    assert bound(printed) == (
        "0 of 2 Planted Campaigns have nothing inside them reaching a candidate"
    )
    assert "vantage-ledger.example" in printed
    assert "signal-harbor.example" in printed
    assert "syn_vantageledger" in printed
    assert "syn_northwindhire" in printed
    assert "0 of the 2 Planted Campaigns have nothing inside" in " ".join(page.split())
    assert {text(row, "campaign_id") for row in rows(recovery)} == {
        "syn-campaign-alpha",
        "syn-campaign-beta",
    }


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
    assert "false groupings" in page


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


@needs_database
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

    The grouping is run here rather than read from the committed file so the two halves
    of the boundary are read off one run, and it is run against a scratch vector table
    because the grouping reads the stored vectors (ADR-0026) — which is a database and
    not a file, and so changes nothing about which files this test watches.
    """
    opened = _Opened()
    sys.addaudithook(opened)
    grouped = tmp_path / "campaign-candidates.jsonl"

    seed_vectors(DEFAULT_CORPUS_PATH)
    opened.recording = True
    try:
        assert main(
            [
                "campaign-candidates",
                "--corpus",
                str(DEFAULT_CORPUS_PATH),
                "--candidates",
                str(grouped),
                "--table",
                SCRATCH_TABLE,
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

    assert {"truth.jsonl", "nuisance.jsonl", "campaign-candidates.jsonl", "policy-scores.jsonl"} <= evaluation
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


def test_the_two_planted_campaigns_come_out_and_one_of_them_is_no_longer_a_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shipped figure, which a reader can check against four candidates by hand.

    `cc-03` holds all of beta's two accounts, so beta is recovered. `cc-01` holds alpha's
    three accounts and one more: `syn_greyloch_6612`, a Hard Negative that published the
    desk's Telegram handle while complaining about the desk. ADR-0005 makes a shared
    Contact Point a grouping edge and nothing in this project can tell an operator from
    a customer, so the candidate holds four accounts, the membership is no longer held
    alone, and alpha is reported as `partial` rather than counted as a recovery. The
    figure falls from what it was, and the extra account is named beside the figure so a
    reader can see exactly which claim cost it.

    `cc-01` is three accounts of one shop sharing one domain: ADR-0005 says they group,
    the grouping is right to produce them, and they are not planted, so they are printed
    as a candidate holding no Planted Campaign rather than counted against N.
    """
    recovery, report, printed = run(tmp_path, capsys=capsys)
    recovered = rows(recovery)
    page = report.read_text(encoding="utf-8")

    assert figure(printed) == "1 of 2 Planted Campaigns"
    assert [text(row, "campaign_id") for row in recovered] == [
        "syn-campaign-alpha",
        "syn-campaign-beta",
    ]
    assert [text(row, "outcome") for row in recovered] == ["partial", "recovered"]
    assert [
        text(records(row, "candidates")[0], "candidate_id") for row in recovered
    ] == ["cc-02", "cc-03"]
    assert texts(records(recovered[0], "candidates")[0], "extra") == ["syn_greyloch_6612"]
    assert texts(records(recovered[1], "candidates")[0], "extra") == []

    assert "1 of 4 candidates" in printed
    assert "cc-02" in printed
    assert "rivermill-bikes.example" in page
    assert "1 of 2 Planted Campaigns" in page
    assert "syn_greyloch_6612" in printed and "syn_greyloch_6612" in page


def test_the_corpus_holds_a_desk_the_second_edge_finds_and_the_first_could_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the ticket is for, asked of the shipped Corpus rather than of a fixture.

    `syn-nuisance-obfuscated-copperlantern` is one desk publishing one Telegram handle
    three ways. Two of its three accounts reach no registration at all and the third
    reaches one only it reaches, so nothing about the desk is in the resolved links. The
    candidate names the handle and every spelling the Corpus wrote it in, which is what
    lets a reader see that this is one identifier rather than two that happen to fold
    together — and it rests on the Contact Point edge alone, so the output says so.

    It is not a Planted Campaign, so it is counted as a false grouping under the
    definition in ADR-0022: accounts of no campaign, grouped correctly, never a recovery.

    It is also `cc-04` rather than `cc-02`, because it is the one candidate the 24-hour
    timing window corroborates nothing under: three accounts holding one handle across
    three weeks is not what a tight window calls one operator, so the clock has put it at
    the end of the table and the identifier with it (ADR-0025). The candidate is still
    proposed, still printed in full and still counted; only its place in the list moved.
    """
    _, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    assert "syn_copperlantern" in printed
    assert "cc-04" in printed
    assert "3 of 4 Campaign Candidates" in page
    assert "syn_copperlantern" in page
    # The edge is named beside the value in both views, because a Contact Point and a
    # Registrable Domain are the same shape and a different kind of claim.
    assert "Contact Points: `syn_copperlantern`" in page
    assert "Contact Points: syn_copperlantern" in printed
    assert "weaker of the two readings" in page


@needs_database
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
    seed_vectors(generated / "corpus.jsonl")
    assert main(
        [
            "campaign-candidates",
            "--corpus",
            str(generated / "corpus.jsonl"),
            "--candidates",
            str(rebuilt / "campaign-candidates.jsonl"),
            "--table",
            SCRATCH_TABLE,
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


def score_row(
    post_id: str,
    account: str,
    points: int,
    published_points: int = 100,
    published_signals: int = 6,
) -> Row:
    """One row of `policy-scores.jsonl`, hand-written.

    `points` of zero is a row carrying no Signal, which the reader allows: a post
    that fires nothing is scored out at 0 and still takes its place in the queue.
    The nonzero row carries one Content Signal priced at the points, so the
    breakdown sums to the points beside it.
    """
    score = (200 * points + published_points) // (2 * published_points)
    return {
        "post_id": post_id,
        "account": account,
        "score": score,
        "points": points,
        "published_points": published_points,
        "published_signals": published_signals,
        "signals": (
            [
                {
                    "signal": "payment_request",
                    "weight": points,
                    "evidence": [
                        {
                            "signal": "payment_request",
                            "field": "body",
                            "phrases": ["deposit for"],
                            "sentence": (
                                "There is a refundable deposit for the materials, "
                                "and that is the whole of it."
                            ),
                        }
                    ],
                }
            ]
            if points
            else []
        ),
    }


def precision_at(printed: str) -> dict[int, tuple[int, int]]:
    """The precision table of the console output: depth to (true, entries)."""
    lines = printed.splitlines()
    start = next(
        number
        for number, line in enumerate(lines)
        if line.split() == ["depth", "true", "of"]
    )
    found: dict[int, tuple[int, int]] = {}
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        depth, true, of = line.split()
        found[int(depth)] = (int(true), int(of))
    return found


# --- the false-grouping rate ------------------------------------------------------


def test_the_report_defines_what_a_false_grouping_is(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The definition has to be in the output itself, not in a ticket."""
    _, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    for view in (page, printed):
        assert "different Planted Campaigns, or to none" in " ".join(view.split())
        # The figure the report must never publish, checked here as well as on the
        # recovery sections: new prose is new ways to say it by accident.
        assert BANNED_FIGURE not in view.lower()


def test_the_false_grouping_rate_is_reported_against_the_nuisance_structure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rate has to be said, and said against the manifest.

    On the committed Corpus the figure is three false groupings of four candidates:
    `cc-01`, which holds the alpha Planted Campaign and the Hard Negative that named its
    handle; `cc-04`, the desk behind the obfuscated Telegram handles, which belongs to no
    campaign at all; and `cc-02`, the shop's accounts, grouped because ADR-0005 puts them
    together and counted as false because they belong to no Planted Campaign. `cc-03`
    recovers beta's whole membership and is not false.
    """
    _, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    assert "3 of 4 Campaign Candidates" in page
    for candidate_id in ("cc-01", "cc-02", "cc-04"):
        assert candidate_id in page and candidate_id in printed
    assert "no Planted Campaign" in page
    # The rate is named beside the Nuisance Structure it was measured against.
    assert "data/corpus/nuisance.jsonl" in page
    assert "decoy_account_cluster" in page


def test_precision_in_the_review_queue_is_reported_at_several_depths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Several depths, not one, and every figure named beside its depth."""
    _, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    assert "precision" in printed.lower()
    rows = precision_at(printed)
    assert set(rows) == {5, 10, 20, 50}, rows
    for depth, (true, of) in rows.items():
        assert true <= of
        assert f"| {depth} | {true} | {of} |" in page
        # The same fraction is stated beside its depth in prose, in both views.
        assert f"{true} of {of} at depth {depth}" in page
        # The console wraps the same figure inside one wrapped sentence.
        assert f"{true} of {of} at depth {depth}" in " ".join(printed.split())


def test_precision_matches_a_hand_computed_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The figure has to be checkable by hand, so it is one on a small Corpus.

    Five posts, five scores: the two that belong to a Planted Campaign sit at
    ranks 2 and 3. At depth 1 the top entry is a post from no campaign, so the
    figure is 0 of 1; by depth 4 it is 2 of 4. The candidate holding rank 2's
    account and one from no campaign is a false grouping.
    """
    corpus = corpus_of(
        tmp_path,
        post("syn_p_0001", "syn_alpha_0001"),
        post("syn_p_0002", "syn_nowhere_0002"),
        post("syn_p_0003", "syn_beta_0003"),
        post("syn_p_0004", "syn_elsewhere_0004"),
        post("syn_p_0005", "syn_alsogone_0005"),
    )
    candidates = write_rows(
        tmp_path / "campaign-candidates.jsonl",
        [
            candidate(
                "cc-01",
                ("syn_alpha_0001", "syn_alsogone_0005"),
                ("syn_p_0001", "syn_p_0005"),
                "alsogone.example",
            )
        ],
    )
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [
            campaign("syn-campaign-one", ("syn_alpha_0001",), ("syn_p_0001",)),
            campaign("syn-campaign-two", ("syn_beta_0003",), ("syn_p_0003",)),
        ],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )
    scores = write_rows(
        tmp_path / "policy-scores.jsonl",
        [
            score_row("syn_p_0002", "syn_nowhere_0002", 100),
            score_row("syn_p_0001", "syn_alpha_0001", 80),
            score_row("syn_p_0003", "syn_beta_0003", 60),
            score_row("syn_p_0004", "syn_elsewhere_0004", 40),
            score_row("syn_p_0005", "syn_alsogone_0005", 0),
        ],
    )

    _, report, printed = run(
        tmp_path / "out",
        corpus=corpus,
        candidates=candidates,
        truth=truth,
        nuisance=nuisance_path,
        scores=scores,
        depths=(1, 2, 3, 4),
        capsys=capsys,
    )
    page = report.read_text(encoding="utf-8")

    assert precision_at(printed) == {1: (0, 1), 2: (1, 2), 3: (2, 3), 4: (2, 4)}
    assert "| 1 | 0 | 1 |" in page
    assert "| 2 | 1 | 2 |" in page
    assert "| 3 | 2 | 3 |" in page
    assert "| 4 | 2 | 4 |" in page
    assert "cc-01" in page and "no Planted Campaign" in page
    # The figure itself is the fraction of the top entries, stated with its depth;
    # a precision number never appears without its depth, in either view.
    expected = "0 of 1 at depth 1, 1 of 2 at depth 2, 2 of 3 at depth 3, 2 of 4 at depth 4"
    assert expected in " ".join(printed.split())
    for share in ("0 of 1 at depth 1", "1 of 2 at depth 2", "2 of 3 at depth 3", "2 of 4 at depth 4"):
        assert share in page


def test_the_report_states_the_lower_of_the_two_is_the_more_trustworthy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report says, plainly, which of the two numbers to believe."""
    _, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    for view in (page, printed):
        assert "the lower of the two" in view.lower()
        assert "deliberate" in view.lower()


def test_the_report_states_who_measured_precision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Precision is the evaluator's figure, not a reviewer clicking."""
    _, report, _ = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    assert "not by a person reviewing the queue" in page


def test_recovery_and_the_false_grouping_rate_are_in_the_same_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Neither number appears alone: both headline figures are on one page."""
    _, report, printed = run(tmp_path, capsys=capsys)
    page = report.read_text(encoding="utf-8")

    for view in (page, printed):
        assert "1 of 2 Planted Campaigns" in view
        assert "3 of 4 Campaign Candidates" in view
        assert "recall bound" in view


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


def timing_disagreement(
    tmp_path: Path, **overrides: object
) -> tuple[Path, Path]:
    """A candidates file whose timing disagrees with itself, one way at a time.

    The evaluator reads what the grouping published and does no timing of its own, so a
    row that claims something about when its accounts posted that the rest of the row does
    not bear out is a row it would print as though the clock had said so. The refusals are
    checked here rather than in the grouping's own tests because this is the command that
    has to stop rather than measure, and it reads the file the other reader reads.
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
                    "vantage-ledger.example",
                    closest_seconds=3_600,
                ),
                **overrides,
            }
        ],
    )
    return corpus, candidates


def recovery_argv(tmp_path: Path, corpus: Path, candidates: Path) -> list[str]:
    """The evaluator over a candidates file this test wrote, refusing nothing itself.

    The membership and the manifest are written into the same temporary directory and the
    refusal has to come from the reader rather than from a file that was never there,
    because a test that fails because it forgot a fixture is a test about fixtures.
    """
    truth = write_rows(
        tmp_path / "truth.jsonl",
        [campaign("syn-campaign-one", ("syn_alpha_0001", "syn_beta_0002"), ("syn_p_0001",))],
    )
    nuisance_path = write_rows(
        tmp_path / "nuisance.jsonl",
        [nuisance("syn-nuisance-single", NuisanceKind.SINGLE_ACCOUNT_DOMAIN)],
    )
    return [
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


@pytest.mark.parametrize(
    ("overrides", "said"),
    (
        pytest.param(
            {"corroboration": []}, "is joined on", id="no gaps beside its evidence"
        ),
        pytest.param(
            {
                "corroboration": [
                    {
                        "kind": "registration",
                        "value": "signal-harbor.example",
                        "closest_seconds": 3_600,
                        "span_seconds": 3_600,
                    }
                ]
            },
            "is joined on",
            id="a gap for something else",
        ),
        pytest.param(
            {
                "timing": {
                    "window_seconds": 86_400,
                    "pieces": 2,
                    "corroborated": 1,
                    "closest_seconds": 3_600,
                }
            },
            "says it times 2 pieces of evidence",
            id="more pieces than it carries",
        ),
        pytest.param(
            {
                "timing": {
                    "window_seconds": 86_400,
                    "pieces": 1,
                    "corroborated": 1,
                    "closest_seconds": 60,
                }
            },
            "publishes gaps of",
            id="a nearest pair none of its gaps show",
        ),
        pytest.param(
            {
                "corroboration": [
                    {
                        "kind": "registration",
                        "value": "vantage-ledger.example",
                        "closest_seconds": 7_200,
                        "span_seconds": 3_600,
                    }
                ]
            },
            "cannot be shorter than the gap inside it",
            id="a span shorter than its gap",
        ),
        pytest.param(
            {
                "corroboration": [
                    {
                        "kind": "url",
                        "value": "vantage-ledger.example",
                        "closest_seconds": 3_600,
                        "span_seconds": 3_600,
                    }
                ]
            },
            "the names are",
            id="a kind this build does not read",
        ),
        pytest.param(
            {
                "corroboration": [
                    {
                        "kind": "registration",
                        "value": "vantage-ledger.example",
                        "closest_seconds": 3_600,
                        "span_seconds": 172_800,
                    }
                ]
            },
            "of which 0 do",
            id="a count of corroborated pieces its spans deny",
        ),
        pytest.param(
            {
                "timing": {
                    "window_seconds": 60,
                    "pieces": 1,
                    "corroborated": 0,
                    "closest_seconds": 3_600,
                }
            },
            "a window has to be at least one hour",
            id="a window the grouping refuses to run at",
        ),
    ),
)
def test_the_run_refuses_a_candidates_file_whose_timing_disagrees_with_itself(
    tmp_path: Path, overrides: Mapping[str, object], said: str
) -> None:
    """A gap nothing recomputes is a claim, and the two readers of this file need it to be
    one the grouping can be held to.

    Six ways for a row to say something about when its accounts posted that the rest of the
    row does not bear out, and each is refused by name: a candidate carrying no gaps beside
    the evidence it is joined on, a gap for a piece of evidence it is not joined on, a
    count of pieces it does not carry, a nearest pair smaller than any gap beside it, a
    span shorter than the gap inside it, a kind of evidence this build does not publish, a
    count of corroborated pieces its own spans deny, and a window so short the grouping
    would have refused to run at it. None of them looks wrong on its own line, which is the
    reason each is checked rather than assumed — the corroborated count in particular is
    the figure the ordering of this project and the heading below the table both rest on.
    """
    corpus, candidates = timing_disagreement(tmp_path, **overrides)

    with pytest.raises(SystemExit) as refusal:
        main(recovery_argv(tmp_path, corpus, candidates))

    assert said in str(refusal.value)
    assert not (tmp_path / "recovery.jsonl").exists()


def test_the_run_refuses_a_candidates_file_counted_at_two_windows(
    tmp_path: Path,
) -> None:
    """A corroborated count means nothing without the window it was counted against.

    Two rows, each internally consistent, each holding a different window — which is what
    a file written by two runs at two `--window-hours` settings looks like. Read as one
    figure it would say `1 of 2 candidates`, and no reader of that sentence could tell
    which two candidates or which two windows, so the file is refused instead. The same
    rule `read_policy_scores` applies to one published weight set.
    """
    corpus, candidates = timing_disagreement(tmp_path)
    published = rows(candidates)
    second = {
        **published[0],
        "candidate_id": "cc-02",
        "timing": {**nested(published[0], "timing"), "window_seconds": 3_600},
    }
    two_windows = write_rows(candidates, [*published, second])

    with pytest.raises(SystemExit) as refusal:
        main(recovery_argv(tmp_path, corpus, two_windows))

    assert "seconds of timing window" in str(refusal.value)
    assert not (tmp_path / "recovery.jsonl").exists()


def similarity_disagreement(
    tmp_path: Path, **overrides: object
) -> tuple[Path, Path]:
    """A candidates file whose similarity disagrees with itself, one way at a time.

    The same argument as `timing_disagreement`, about the other Signal: the evaluator
    does no vector arithmetic of its own, so a row claiming something about how close
    its accounts' words are that the rest of the row does not bear out would be
    printed as though the vectors had said so.
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
                    "vantage-ledger.example",
                    closest_seconds=3_600,
                ),
                **overrides,
            }
        ],
    )
    return corpus, candidates


@pytest.mark.parametrize(
    ("overrides", "said"),
    (
        pytest.param(
            {"similarity": []},
            "holds similarity rows for [] and lists posts",
            id="no twins beside its posts",
        ),
        pytest.param(
            {
                "similarity": [
                    {
                        "post": "syn_p_0001",
                        "account": "syn_alpha_0001",
                        "nearest": "syn_p_0002",
                        "nearest_account": "syn_beta_0002",
                        "distance": 0.1,
                    },
                    {
                        "post": "syn_p_0002",
                        "account": "syn_beta_0002",
                        "nearest": "syn_p_0001",
                        "nearest_account": "syn_alpha_0001",
                        "distance": 0.1,
                    },
                ],
                "content_similarity": {
                    "threshold": 0.5,
                    "pieces": 3,
                    "corroborated": 3,
                    "closest_distance": 0.1,
                },
            },
            "says it has 3 similarity rows and publishes 2",
            id="a count of posts its twins deny",
        ),
        pytest.param(
            {
                "content_similarity": {
                    "threshold": 0.5,
                    "pieces": 2,
                    "corroborated": 1,
                    "closest_distance": 0.0,
                }
            },
            "says 1 of its posts fall inside the 0.5 threshold",
            id="a corroborated count its twins deny",
        ),
        pytest.param(
            {
                "content_similarity": {
                    "threshold": 0.5,
                    "pieces": 2,
                    "corroborated": 2,
                    "closest_distance": 0.9,
                }
            },
            "says its closest pair is 0.9 apart",
            id="a nearest pair none of its twins show",
        ),
        pytest.param(
            {
                "content_similarity": {
                    "threshold": 0,
                    "pieces": 2,
                    "corroborated": 0,
                    "closest_distance": 0.0,
                }
            },
            "counts as near-identical only above zero and at most one",
            id="a threshold the grouping refuses to run at",
        ),
        pytest.param(
            {
                "similarity": [
                    {
                        "post": "syn_p_0001",
                        "account": "syn_alpha_0001",
                        "nearest": "syn_p_0001",
                        "nearest_account": "syn_beta_0002",
                        "distance": 0.0,
                    },
                    {
                        "post": "syn_p_0002",
                        "account": "syn_beta_0002",
                        "nearest": "syn_p_0001",
                        "nearest_account": "syn_alpha_0001",
                        "distance": 0.0,
                    },
                ]
            },
            "names syn_p_0001 as its own twin",
            id="a post that is its own twin",
        ),
        pytest.param(
            {
                "similarity": [
                    {
                        "post": "syn_p_0001",
                        "account": "syn_alpha_0001",
                        "nearest": "syn_p_0002",
                        "nearest_account": "syn_alpha_0001",
                        "distance": 0.0,
                    },
                    {
                        "post": "syn_p_0002",
                        "account": "syn_beta_0002",
                        "nearest": "syn_p_0001",
                        "nearest_account": "syn_beta_0002",
                        "distance": 0.0,
                    },
                ]
            },
            "a twin by the same account is that account's own text",
            id="a twin by the same account",
        ),
    ),
)
def test_the_run_refuses_a_candidates_file_whose_similarity_disagrees_with_itself(
    tmp_path: Path, overrides: Mapping[str, object], said: str
) -> None:
    """The same argument as the timing rows, about the distances.

    Seven ways for a row to say something about how close its accounts' words are that
    the rest of the row does not bear out: no twins beside posts it holds, a count of
    posts the twins deny, a corroborated count the twins deny, a nearest pair smaller
    than any twin shows, a threshold nothing can fail, a post standing as its own twin,
    and a twin by the very account it belongs to - which is an account's habits rather
    than corroboration of a grouping. None of them looks wrong on its own line.
    """
    corpus, candidates = similarity_disagreement(tmp_path, **overrides)

    with pytest.raises(SystemExit) as refusal:
        main(recovery_argv(tmp_path, corpus, candidates))

    assert said in str(refusal.value), str(refusal.value)
    assert not (tmp_path / "recovery.jsonl").exists()


def test_the_run_refuses_a_candidates_file_counted_at_two_thresholds(
    tmp_path: Path,
) -> None:
    """The same rule the two-window refusal applies, to the other threshold.

    Two rows, each internally consistent, each counted against a different cosine
    distance — which is what a file written by two runs at two `--similarity-threshold`
    settings looks like. Read as one figure it would say how many posts were near
    identical without saying what near-identical meant.
    """
    corpus, candidates = similarity_disagreement(tmp_path)
    published = rows(candidates)
    second = {
        **published[0],
        "candidate_id": "cc-02",
        "accounts": ["syn_gamma_0003", "syn_delta_0004"],
        "shared_domains": [
            {
                "domain": "vantage-ledger.example",
                "accounts": ["syn_gamma_0003", "syn_delta_0004"],
                "posts": ["syn_p_0001", "syn_p_0002"],
            }
        ],
        "similarity": [
            {
                "post": "syn_p_0001",
                "account": "syn_gamma_0003",
                "nearest": "syn_p_0002",
                "nearest_account": "syn_delta_0004",
                "distance": 0.2,
            },
            {
                "post": "syn_p_0002",
                "account": "syn_delta_0004",
                "nearest": "syn_p_0001",
                "nearest_account": "syn_gamma_0003",
                "distance": 0.2,
            },
        ],
        "content_similarity": {
            "threshold": 0.25,
            "pieces": 2,
            "corroborated": 2,
            "closest_distance": 0.2,
        },
    }
    two_thresholds = write_rows(candidates, [*published, second])

    with pytest.raises(SystemExit) as refusal:
        main(recovery_argv(tmp_path, corpus, two_thresholds))

    assert "of similarity threshold" in str(refusal.value)
    assert not (tmp_path / "recovery.jsonl").exists()


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