"""The Review Queue: every post ordered by Triage Priority, at a stated review depth.

A queue is only worth a reviewer's attention if the number at the top of it is the number
the scoring command published for that post, so that is the claim under test rather than
the ordering as such: nothing reorders the Policy Score, a tie is broken without ever
putting a lower-scoring post above a higher-scoring one, and a row whose score is not the
arithmetic of its own points cannot reach the queue at all. Every entry carries the
Signal-by-Signal breakdown behind its score rather than a summary of one, names the
Campaign Candidates the post sits in, and the depth the run was asked for is in the
header, because a queue pasted into an issue without its depth says nothing about what
was left below it.

No figure is computed by this ticket, so there is nothing here to check about how good
the ordering is. What is checked is that the ordering is the Policy Score, that it says
so, and that it holds to the two published files it reads.

Seam under test: the `review-queue` command, observed through the table it prints.
Nothing here inspects the code that printed it, and no input is written by the module
that reads it: a scores file this project produced could not be used to argue that the
command listens to one.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_REVIEW_DEPTH, main

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_WEIGHTS = REPO_ROOT / "data" / "signals" / "weights.jsonl"
COMMITTED_SCORES = REPO_ROOT / "data" / "signals" / "policy-scores.jsonl"
COMMITTED_CANDIDATES = REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl"

Row = Mapping[str, object]


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


# --- reading the published files and what the command printed -------------------------


def rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} should be text, is {value!r}"
    return value


def count(row: Row, field: str) -> int:
    value = row[field]
    assert isinstance(value, int), f"{field} should be a whole number, is {value!r}"
    return value


def records(row: Row, field: str) -> list[Row]:
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    assert all(isinstance(entry, dict) for entry in value), f"{field} should hold rows"
    return [entry for entry in value if isinstance(entry, dict)]


# The published weight set, read out of the committed file rather than typed in, so a
# weight this project changes moves the fixtures with it and a fixture cannot end up
# describing a weight file that no longer exists.
PUBLISHED = {text(row, "signal"): count(row, "weight") for row in rows(COMMITTED_WEIGHTS)}
TOTAL_PUBLISHED = sum(PUBLISHED.values())
PUBLISHED_SIGNALS = len(PUBLISHED)

# The evidence each Signal kind fires on, in the shape the scoring command writes it.
# Three kinds, and none of them names itself: `domain` with `posts` and `accounts` is the
# reach, `domain` with `other` is the lookalike pair, and the row naming a Signal is the
# sentence a Content Signal matched.
REACH = {"domain": "signal-harbor.example", "posts": 3, "accounts": 2}
LOOKALIKE = {"domain": "signal-harbor.example", "other": "signal-harbour.example"}
SENTENCE = (
    "There is a refundable deposit for the materials, and you have to finish the intake "
    "within 48 hours or the slot goes to someone else."
)
PHRASE = {
    "guaranteed_return": "cannot lose",
    "payment_request": "deposit for",
    "urgency_language": "within 48 hours",
}


def evidence_of(signal: str) -> Row:
    if signal == "domain_frequency":
        return dict(REACH)
    if signal == "domain_lookalike":
        return dict(LOOKALIKE)
    return {
        "signal": signal,
        "field": "body",
        "phrases": [PHRASE[signal]],
        "sentence": SENTENCE,
    }


def share(points: int, published: int = TOTAL_PUBLISHED) -> int:
    """The Policy Score, worked out the way the scoring command works it out.

    Rounded half up by integer arithmetic, and typed in rather than imported: a fixture
    computed by the module under test could not be used to argue that the module checks
    what the file claims about itself.
    """
    return (200 * points + published) // (2 * published)


# --- reading what the command printed --------------------------------------------------


def index(printed: str) -> list[list[str]]:
    """The index as cells per row: rank, post, account, score, Signals, candidates.

    Read back out of the printed table rather than off the domain objects, because the
    claim is about the order a reviewer reads.
    """
    lines = printed.splitlines()
    start = next(number for number, line in enumerate(lines) if line.startswith("rank "))
    cells: list[list[str]] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        cells.append(line.split())
    return cells


def ranked(printed: str) -> list[str]:
    """The post ids in the order the queue prints them."""
    return [cells[1] for cells in index(printed)]


def scores(printed: str) -> list[int]:
    return [int(cells[3]) for cells in index(printed)]


def entries(printed: str) -> list[str]:
    """The post ids in the order the per-entry blocks are printed."""
    return [
        line.split()[1]
        for line in printed.splitlines()
        if line.endswith("/100") and len(line.split()) >= 3
    ]


def opening(printed: str) -> str:
    """The paragraph under the heading, which is where the depth is stated.

    Joined back into one string rather than returned line by line, because the depth
    sentence is wrapped to the console's width and the width depends on how long the path
    the caller chose is. `text.wrap` breaks only at spaces, so the sentence is the same
    sentence either way, and a claim about what it says must not depend on where a path
    of this machine's length happens to fall. That is not a hypothetical: the assertion
    held over a 110-character temporary path and failed over a 79-character one, where
    "3 of them sit below the cut" landed either side of a line break.
    """
    lines = printed.splitlines()
    start = next(
        number for number, line in enumerate(lines) if line.startswith("Review depth")
    )
    paragraph: list[str] = []
    for line in lines[start:]:
        if not line.strip() or line.startswith("  "):
            break
        paragraph.append(line)
    return " ".join(paragraph)


def depth_figure(printed: str) -> str:
    return next(line for line in printed.splitlines() if line.strip().startswith("depth "))


def closing(printed: str) -> list[str]:
    """The closing section: everything printed after the last entry's block."""
    lines = printed.splitlines()
    last = max(
        number
        for number, line in enumerate(lines)
        if line.endswith("/100") and len(line.split()) >= 3
    )
    return lines[last + 1 :]


def block(printed: str, post_id: str) -> list[str]:
    """One entry's block: the lines under its own heading, to the blank line after."""
    lines = printed.splitlines()
    start = next(
        number
        for number, line in enumerate(lines)
        if line.endswith("/100") and line.split()[1:2] == [post_id]
    )
    body: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        body.append(line)
    return body


def refusing(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> str:
    """One refusal, with whatever the console printed before it."""
    with pytest.raises(SystemExit) as refusal:
        main(list(argv))
    return str(refusal.value) + capsys.readouterr().out


# --- the two published files, hand-written ---------------------------------------------


def write(path: Path, entries: Sequence[Row]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for row in entries
        ),
        encoding="utf-8",
    )
    return path


def score_row(
    post_id: str,
    account: str,
    *signals: tuple[str, int],
    points: int | None = None,
    score: int | None = None,
    published_points: int = TOTAL_PUBLISHED,
    published_signals: int = PUBLISHED_SIGNALS,
) -> Row:
    """One row of `policy-scores.jsonl`, hand-written.

    Each Signal is named with what it is worth in that row rather than looked up, because
    a row the scoring command could not have written is the only way to reach the ties
    the ordering has to break: with the published weights every sum is a multiple of five
    and no two of them round to the same score. `points` and `score` default to what the
    Signals say and can be overridden with a figure the arithmetic does not support,
    which is what the refusals are tested against.
    """
    earned = sum(weight for _, weight in signals) if points is None else points
    return {
        "post_id": post_id,
        "account": account,
        "score": share(earned, published_points) if score is None else score,
        "points": earned,
        "published_points": published_points,
        "published_signals": published_signals,
        "signals": [
            {"signal": signal, "weight": weight, "evidence": [evidence_of(signal)]}
            for signal, weight in signals
        ],
    }


def candidate_row(
    candidate_id: str,
    accounts: Sequence[str],
    posts: Sequence[str],
    *domains: str,
    first_seen: str = "2026-01-05T09:00:00Z",
) -> Row:
    """One row of `campaign-candidates.jsonl`, hand-written."""
    return {
        "candidate_id": candidate_id,
        "accounts": list(accounts),
        "posts": list(posts),
        "shared_domains": [
            {"domain": domain, "accounts": list(accounts), "posts": list(posts)}
            for domain in domains
        ],
        "first_seen": first_seen,
    }


def published(
    tmp_path: Path,
    *scores: Row,
    candidates: Sequence[Row] = (),
) -> tuple[Path, Path]:
    """A scores file and a candidates file for the command to read."""
    return (
        write(tmp_path / "policy-scores.jsonl", scores),
        write(tmp_path / "campaign-candidates.jsonl", candidates),
    )


def run(
    scores_path: Path,
    candidates_path: Path,
    capsys: pytest.CaptureFixture[str] | None = None,
    depth: int | None = None,
) -> str:
    argv = ["review-queue", "--scores", str(scores_path), "--candidates", str(candidates_path)]
    if depth is not None:
        argv += ["--depth", str(depth)]
    exit_code = main(argv)
    assert exit_code == 0
    return capsys.readouterr().out if capsys is not None else ""


def refusing_run(
    scores_path: Path, candidates_path: Path, capsys: pytest.CaptureFixture[str]
) -> str:
    return refusing(
        ["review-queue", "--scores", str(scores_path), "--candidates", str(candidates_path)],
        capsys,
    )


# --- the queue is ordered by the Policy Score --------------------------------------------


def test_the_queue_is_ordered_by_the_policy_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Policy Score is the ordering, and nothing else is consulted to produce it.

    The rows go in shuffled and the queue comes out by score. A run that ordered by post
    id, by the file's own order, or by anything read off the Corpus would print these same
    three posts in a different sequence, which is the whole difference between a queue and
    a list.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0003", "syn_c_0003", ("urgency_language", PUBLISHED["urgency_language"])),
        score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", PUBLISHED["domain_frequency"])),
        score_row(
            "syn_p_0002",
            "syn_c_0002",
            ("urgency_language", PUBLISHED["urgency_language"]),
            ("payment_request", PUBLISHED["payment_request"]),
        ),
    )

    printed = run(scores_path, candidates_path, capsys, depth=50)

    by_score = sorted(rows(scores_path), key=lambda row: -count(row, "score"))
    assert ranked(printed) == [text(row, "post_id") for row in by_score]
    assert scores(printed) == [count(row, "score") for row in by_score]


def test_a_tie_is_broken_on_the_points_earned_and_not_on_the_post_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Policy Score is points rounded, so two posts can display the same one.

    Fifty points and forty-nine both display as 45 of 110 published, so the queue has to
    order them by something, and the post ids are chosen so that the post id would put
    them the other way round: a stable table is worth having, and a stable table that
    ranks the lower of two equal scores first is not a queue. Points are that same
    arithmetic one step finer, so nothing here is a second opinion about the post.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0009", "syn_c_0009", ("guaranteed_return", 50)),
        score_row("syn_p_0002", "syn_c_0002", ("guaranteed_return", 49)),
        score_row("syn_p_0007", "syn_c_0007", ("guaranteed_return", 48)),
    )

    printed = run(scores_path, candidates_path, capsys, depth=50)

    assert scores(printed) == [45, 45, 44]
    assert ranked(printed) == ["syn_p_0009", "syn_p_0002", "syn_p_0007"]


def test_the_order_does_not_depend_on_the_order_of_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Corpus read in another order is the same Corpus, so it is the same queue.

    Three rows in one sequence and the same three in another: the printed index and the
    blocks under it have to come out identical. Otherwise every commit that reordered a
    file would rewrite the queue, and a reviewer could not tell a change in the ordering
    from a change in the file.
    """
    rows_one = (
        score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", PUBLISHED["domain_frequency"])),
        score_row(
            "syn_p_0002",
            "syn_c_0002",
            ("urgency_language", PUBLISHED["urgency_language"]),
            ("payment_request", PUBLISHED["payment_request"]),
        ),
        score_row("syn_p_0003", "syn_c_0003", ("payment_request", PUBLISHED["payment_request"])),
    )
    first, first_candidates = published(tmp_path / "first", *rows_one)
    second, second_candidates = published(tmp_path / "second", *reversed(rows_one))

    printed = run(first, first_candidates, capsys, depth=50)
    against = run(second, second_candidates, capsys, depth=50)

    assert ranked(printed) == ranked(against)
    assert entries(printed) == entries(against)


def test_a_queue_holds_every_post_the_scores_file_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A post carrying no Signal is ordered rather than dropped.

    Dropping it would put a filter in front of the score, decided by a rule no weight file
    publishes, and it would make a review depth unable to bind: how many posts carry a
    Signal is a property of the Corpus rather than of the depth, so "the top 50" could
    never mean fifty. It sits at the tail with an empty breakdown, which says plainly that
    there was nothing to show.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0002", "syn_c_0002", ("payment_request", PUBLISHED["payment_request"])),
        score_row("syn_p_0001", "syn_c_0001"),
        score_row("syn_p_0003", "syn_c_0003"),
    )

    printed = run(scores_path, candidates_path, capsys, depth=50)

    assert ranked(printed) == ["syn_p_0002", "syn_p_0001", "syn_p_0003"]
    assert scores(printed) == [18, 0, 0]
    assert "no Signal" in "".join(block(printed, "syn_p_0001"))


# --- the review depth -----------------------------------------------------------------------


def test_the_depth_truncates_the_queue_and_the_header_says_where_it_was_cut(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A depth is a parameter, and a printed queue always states the one it was run at.

    Five posts in the file, two asked for, and the header names both figures and the three
    left below the cut. A queue stating its depth nowhere would be read as a claim about
    the whole Corpus, which is what "the top 50" has to be able to mean without anybody
    opening the file.
    """
    scores_path, candidates_path = published(
        tmp_path,
        *(
            score_row(f"syn_p_000{number}", f"syn_c_000{number}", ("payment_request", 20))
            for number in range(1, 6)
        ),
    )

    printed = run(scores_path, candidates_path, capsys, depth=2)

    assert len(index(printed)) == 2
    assert "2 of the 5" in depth_figure(printed), depth_figure(printed)
    assert "3 of them sit below the cut" in opening(printed), opening(printed)


def test_a_depth_deeper_than_the_file_is_the_whole_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asking for more than there is prints everything and says nothing was cut.

    The committed Corpus holds fewer posts than the default depth, so this is the case
    every reader of this repository hits first. A header claiming the top 50 over a
    34-post file without saying so would read as a claim that 50 posts were worth
    reading.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0001", "syn_c_0001", ("payment_request", 20)),
        score_row("syn_p_0002", "syn_c_0002"),
    )

    printed = run(scores_path, candidates_path, capsys, depth=50)

    assert ranked(printed) == ["syn_p_0001", "syn_p_0002"]
    assert "50 of the 2" in depth_figure(printed), depth_figure(printed)
    assert "nothing was cut" in opening(printed), opening(printed)


def test_the_default_depth_is_used_when_the_caller_states_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The depth is a parameter with a stated default, and the run says which it used.

    Read off the output rather than off the parser, so a run that quietly used a depth
    other than the one its own help text promises would fail here.
    """
    scores_path, candidates_path = published(
        tmp_path, score_row("syn_p_0001", "syn_c_0001", ("payment_request", 20))
    )

    printed = run(scores_path, candidates_path, capsys)

    assert f"{DEFAULT_REVIEW_DEPTH} of the 1" in depth_figure(printed), depth_figure(printed)


@pytest.mark.parametrize("depth", [0, -1, -50], ids=["zero", "negative", "far negative"])
def test_a_depth_that_is_not_a_depth_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], depth: int
) -> None:
    """A depth below one is refused rather than printing an empty queue.

    An empty queue under a header saying so is indistinguishable at a glance from a Corpus
    in which nothing is worth reading, and the second reading is a claim about the Corpus
    that this command has no business making. The refusal quotes the figure back.
    """
    scores_path, candidates_path = published(
        tmp_path, score_row("syn_p_0001", "syn_c_0001", ("payment_request", 20))
    )

    refusal = refusing(
        [
            "review-queue",
            "--scores",
            str(scores_path),
            "--candidates",
            str(candidates_path),
            "--depth",
            str(depth),
        ],
        capsys,
    )

    assert f"review depth of {depth}" in refusal, refusal
    assert "Review Queue" not in refusal, "the run printed a queue and then refused"


# --- what an entry shows ----------------------------------------------------------------------


def test_every_entry_shows_the_score_the_scores_file_published(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The number on an entry is the number in the file, read back rather than assumed.

    Compared post by post against the row the file holds, so a rendering that rounded,
    rescaled, or recomputed the score would fail here even though it looked right.
    """
    written = (
        score_row("syn_p_0002", "syn_c_0002", ("payment_request", PUBLISHED["payment_request"])),
        score_row(
            "syn_p_0001",
            "syn_c_0001",
            ("domain_frequency", PUBLISHED["domain_frequency"]),
            ("urgency_language", PUBLISHED["urgency_language"]),
        ),
    )
    scores_path, candidates_path = published(tmp_path, *written)

    printed = run(scores_path, candidates_path, capsys, depth=50)

    for row in written:
        heading = next(
            line.split()
            for line in printed.splitlines()
            if line.endswith("/100") and line.split()[1:2] == [text(row, "post_id")]
        )
        assert heading[1] == text(row, "post_id")
        assert heading[2] == text(row, "account")
        assert heading[3] == f"{count(row, 'score')}/100"


def test_every_entry_shows_the_signals_behind_the_score_with_their_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The breakdown survives the move from the scoring command to the queue.

    An entry that printed "2 Signals" and left it there would ask a reviewer to take the
    severity on trust at the exact point where they are deciding what to read first, so
    the Signal names, the weights they were earned at, and every figure and sentence of
    the evidence behind them are all checked to be on the entry. The evidence is checked
    by value rather than by shape: what matters is that the registration, the reach, and
    the whole sentence are there, not that the line is formatted the way one module
    happens to format it.
    """
    written = (
        score_row(
            "syn_p_0001",
            "syn_c_0001",
            ("domain_frequency", PUBLISHED["domain_frequency"]),
            ("payment_request", PUBLISHED["payment_request"]),
        ),
    )
    scores_path, candidates_path = published(tmp_path, *written)

    printed = run(scores_path, candidates_path, capsys, depth=50)

    body = block(printed, "syn_p_0001")
    for signal in records(written[0], "signals"):
        carrying = next(line for line in body if line.strip().startswith(text(signal, "signal")))
        assert carrying.split()[1] == str(count(signal, "weight")), carrying
        for piece in records(signal, "evidence"):
            for field, value in piece.items():
                for entry in value if isinstance(value, list) else [value]:
                    assert str(entry) in carrying, f"{field}={entry!r} is not on the entry"
    total = next(line for line in body if line.strip().startswith("total"))
    assert f"of {TOTAL_PUBLISHED} published points" in total, total
    assert f"2 of {PUBLISHED_SIGNALS} Signals" in total, total


def test_every_entry_names_the_campaign_candidates_it_sits_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A candidate id on its own means nothing to a reviewer who has not opened the file.

    Two candidates hold this post and both are named, each with how many accounts it holds
    and the registration it is joined on, which is the one fact that says whether the
    grouping is worth a look. Naming one of the two would leave a reader unable to tell a
    post in one candidate from a post in two, and the index column carries the same list
    so the two views of the queue cannot disagree about it.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", PUBLISHED["domain_frequency"])),
        score_row("syn_p_0002", "syn_c_0002", ("payment_request", PUBLISHED["payment_request"])),
        candidates=(
            candidate_row(
                "syn_c_001",
                ("syn_c_0001", "syn_c_0002"),
                ("syn_p_0001",),
                "signal-harbor.example",
            ),
            candidate_row(
                "syn_c_002",
                ("syn_c_0001", "syn_c_0003"),
                ("syn_p_0001",),
                "harbour-yards.example",
            ),
        ),
    )

    printed = run(scores_path, candidates_path, capsys, depth=50)

    body = block(printed, "syn_p_0001")
    named = [line for line in body if line.strip().startswith("candidates")]
    assert len(named) == 2, named
    assert "syn_c_001" in named[0] and "signal-harbor.example" in named[0], named[0]
    assert "syn_c_002" in named[1] and "harbour-yards.example" in named[1], named[1]
    assert "2 accounts" in named[0], named[0]
    cell = next(cells for cells in index(printed) if cells[1] == "syn_p_0001")
    assert " ".join(cell[5:]) == "syn_c_001, syn_c_002", cell


def test_a_post_no_candidate_holds_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An entry with no candidate says so, rather than printing nothing.

    A missing line is indistinguishable from a run that joined nothing, and this one joined
    a file holding a candidate. The dash is the project's own mark for a cell with nothing
    in it, the same one the scoring command prints for a registration it neither withheld
    nor scored.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", PUBLISHED["domain_frequency"])),
        score_row("syn_p_0002", "syn_c_0002", ("payment_request", PUBLISHED["payment_request"])),
        candidates=(
            candidate_row(
                "syn_c_002",
                ("syn_c_0002", "syn_c_0003"),
                ("syn_p_0002",),
                "harbour-yards.example",
            ),
        ),
    )

    printed = run(scores_path, candidates_path, capsys, depth=50)

    cell = next(cells for cells in index(printed) if cells[1] == "syn_p_0001")
    assert cell[5] == "-", cell
    alone = [
        line for line in block(printed, "syn_p_0001") if line.strip().startswith("candidates")
    ]
    assert len(alone) == 1, alone
    assert "no Campaign Candidate" in alone[0], alone[0]


# --- the join between the two published files -----------------------------------------------


def test_a_candidate_naming_a_post_the_scores_file_does_not_hold_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two published files that disagree about which posts exist stop the run.

    The refusal is the point: a queue that quietly left out the candidate whose post it
    could not find would hide the disagreement behind a shorter table, and the entries it
    did print would be fewer than the files describe with nothing saying so.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", PUBLISHED["domain_frequency"])),
        candidates=(
            candidate_row(
                "syn_c_001",
                ("syn_c_0001", "syn_c_0002"),
                ("syn_p_0001", "syn_p_0099"),
                "signal-harbor.example",
            ),
        ),
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "syn_p_0099" in refusal, refusal
    assert "Review Queue" not in refusal, "the run printed a queue and then refused"


def test_a_candidate_file_holding_nothing_still_produces_a_queue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The grouping is a separate step, and a run before it has to produce a queue.

    `rfi campaign-candidates` writes an empty file when no two accounts share a
    registration, so an empty candidates file is a real state of the repository and not a
    missing step. The queue is the Policy Score and the score stands on its own, so the
    candidates column is a column of dashes rather than a refusal.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", PUBLISHED["domain_frequency"])),
    )

    printed = run(scores_path, candidates_path, capsys, depth=50)

    assert ranked(printed) == ["syn_p_0001"]
    assert index(printed)[0][5] == "-"


# --- the scores file is checked where it is read ---------------------------------------------


def test_a_row_holding_a_field_this_project_does_not_publish_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A membership label in the scores file stops the run rather than being ignored.

    This is ADR-0008's refusal on the one file the pipeline reads after the Corpus: a
    `planted` field would put Planted Campaign membership into a file the Review Queue is
    built from, and quietly ignoring a field that had appeared is the first way to stop
    being able to say the file does not carry one.
    """
    scores_path, candidates_path = published(
        tmp_path,
        {**score_row("syn_p_0001", "syn_c_0001", ("payment_request", 20)), "planted": "syn_k_0001"},
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "planted" in refusal, refusal


def test_a_row_naming_one_post_twice_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A duplicated post id would be read twice and read twice again by a depth.

    A file holding one post on two rows can fill a depth of two by itself, and the queue
    would then claim to have given a reviewer two posts to read.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0001", "syn_c_0001", ("payment_request", 20)),
        score_row("syn_p_0001", "syn_c_0001", ("payment_request", 20)),
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "syn_p_0001" in refusal, refusal


def test_a_row_whose_score_is_not_the_arithmetic_of_its_points_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A score its own points do not support cannot be ordered against scores they do.

    This is what makes "the ordering and the displayed number cannot disagree"
    structural rather than a promise: the only way a number could contradict the queue is
    by not being the Policy Score of the arithmetic printed beside it, and that row never
    gets in.
    """
    scores_path, candidates_path = published(
        tmp_path, score_row("syn_p_0001", "syn_c_0001", ("payment_request", 20), score=90)
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "90" in refusal, refusal


def test_a_row_whose_points_are_not_the_sum_of_its_signal_weights_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The breakdown has to add up to the number printed beside it.

    A Signal priced at 25 and points earned of 30 leave five points no Signal in the row
    accounts for, so the entry would show a severity its own lines do not reach.
    """
    scores_path, candidates_path = published(
        tmp_path, score_row("syn_p_0001", "syn_c_0001", ("payment_request", 25), points=30)
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "30" in refusal and "25" in refusal, refusal


def test_rows_publishing_different_totals_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every row has to be scored against the same published set to be ordered together.

    Two rows taken from two runs over two weight sets carry two denominators, and a queue
    ordering them against each other would be comparing two scales. The header prints one
    published total beside every entry, so a queue holding both kinds of row would print
    a figure that is wrong for one of them.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row("syn_p_0001", "syn_c_0001", ("payment_request", 50)),
        score_row(
            "syn_p_0002",
            "syn_c_0002",
            ("payment_request", 40),
            published_points=80,
            score=share(40, 80),
        ),
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "110" in refusal and "80" in refusal, refusal


def test_a_signal_priced_twice_in_one_row_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Signal is present or absent, so a row carrying it twice is not a row of this
    project's arithmetic (ADR-0014)."""
    scores_path, candidates_path = published(
        tmp_path,
        score_row(
            "syn_p_0001",
            "syn_c_0001",
            ("payment_request", 20),
            ("payment_request", 20),
            points=40,
        ),
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "payment_request" in refusal, refusal


def test_a_signal_priced_at_nothing_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Signal the weight file could not have published is not one this file can claim.

    `Weights.read` refuses a weight of zero for the same reason: a Signal in a breakdown
    that adds nothing is a Signal a reader is asked to weigh without being able to.
    """
    scores_path, candidates_path = published(
        tmp_path, score_row("syn_p_0001", "syn_c_0001", ("payment_request", 0))
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "payment_request" in refusal, refusal


def test_evidence_the_project_does_not_write_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Evidence is one of three shapes, and an unknown one is not printed as though it
    were one of them.

    The three kinds share a field name and only one of them names a Signal, so a row
    cannot say what it is: the field set is what identifies it, and a row holding
    something else would have to be either dropped or guessed at.
    """
    row = score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", 30))
    scores_path, candidates_path = published(
        tmp_path,
        {
            **row,
            "signals": [
                {
                    "signal": "domain_frequency",
                    "weight": 30,
                    "evidence": [{"domain": "harbor.example"}],
                }
            ],
        },
    )

    refusal = refusing_run(scores_path, candidates_path, capsys)

    assert "evidence" in refusal, refusal


# --- what the run is allowed to read ----------------------------------------------------------


def test_the_run_reads_the_two_published_files_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The queue cannot reach inference, and the way to know is to watch it.

    A command that re-scored would have to open the Corpus, the Public Suffix List, and
    the weight set; one that measured anything would have to open the membership. It opens
    the scores and the candidates and nothing else. Asserted on the paths a run opens
    rather than on its imports, because what a reader can check is what the process
    touched.
    """
    scores_path, candidates_path = published(
        tmp_path, score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", 30))
    )
    opened = _Opened()
    sys.addaudithook(opened)

    opened.recording = True
    try:
        run(scores_path, candidates_path, capsys, depth=50)
    finally:
        opened.recording = False

    data_files = {
        Path(path).name for path in opened.record() if Path(path).suffix in {".jsonl", ".dat"}
    }
    assert data_files == {"policy-scores.jsonl", "campaign-candidates.jsonl"}
    assert not [path for path in opened.record() if Path(path).name.startswith("truth")]
    assert not [path for path in opened.record() if Path(path).name == "corpus.jsonl"]


def test_the_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same argument as every other command here: the numbers come from committed bytes."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the review-queue command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    scores_path, candidates_path = published(
        tmp_path, score_row("syn_p_0001", "syn_c_0001", ("domain_frequency", 30))
    )

    run(scores_path, candidates_path, depth=50)


# --- the committed pair ------------------------------------------------------------------------


def test_the_queue_is_the_committed_scores_file_in_policy_score_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Held to the two committed files the way the Corpus is held to its seed.

    The queue is a projection of the scores and the candidates, so the ordering is
    recomputed here out of the committed rows and has to be the order the command printed.
    Nothing about the Corpus is regenerated: the two files are the input, and a change to
    either moves this queue without a line of code changing.
    """
    printed = run(COMMITTED_SCORES, COMMITTED_CANDIDATES, capsys)

    expected = sorted(
        rows(COMMITTED_SCORES),
        key=lambda row: (-count(row, "score"), -count(row, "points"), text(row, "post_id")),
    )
    assert ranked(printed) == [text(row, "post_id") for row in expected]
    assert scores(printed) == [count(row, "score") for row in expected]
    assert len(index(printed)) == len(expected)


def test_the_index_and_the_entries_are_printed_in_the_same_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two views of one queue, and a reader who scrolls has to find the post they read.

    The index summarises every entry and the blocks under it are the detail. If the two
    disagreed, a reviewer who had just read a line of the index would scroll past the entry
    it named and have no way to tell which block was the one they meant.
    """
    printed = run(COMMITTED_SCORES, COMMITTED_CANDIDATES, capsys, depth=12)

    assert ranked(printed) == entries(printed)
    assert [cells[0] for cells in index(printed)] == [
        str(number) for number in range(1, len(index(printed)) + 1)
    ]


def test_running_twice_prints_the_same_table(capsys: pytest.CaptureFixture[str]) -> None:
    """Nothing in a queue is assigned an identifier, so the order has to be fixed by
    something else: the score, then the points earned, then the post id.

    A run that ordered by a set would print a different queue whenever the Corpus changed,
    and every other commit would be noise.
    """
    first = run(COMMITTED_SCORES, COMMITTED_CANDIDATES, capsys)
    second = run(COMMITTED_SCORES, COMMITTED_CANDIDATES, capsys)

    assert first == second


# --- the shape of the output ---------------------------------------------------------------------


def test_the_console_output_is_ascii_and_wrapped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The console is pasted into issues and diffed between runs, so it has to be one
    thing on every terminal.

    The claim every table in this project makes about its own output: ASCII only, so a
    console that cannot encode anything else prints it the same way as one that can, and
    wrapped at the width a terminal gives you. Two kinds of line are allowed past that
    width, and the assertion says why: the paths the caller chose, which the command does
    not control, and a sentence of a post's own text, which is quoted whole because the
    sentence rather than the phrase is the evidence a reviewer has to read.
    """
    scores_path, candidates_path = published(
        tmp_path,
        score_row(
            "syn_p_0001",
            "syn_c_0001",
            ("domain_frequency", PUBLISHED["domain_frequency"]),
            ("payment_request", PUBLISHED["payment_request"]),
            ("urgency_language", PUBLISHED["urgency_language"]),
        ),
    )

    printed = run(scores_path, candidates_path, capsys, depth=50)

    assert printed.isascii(), "the console output carries a character a console may not have"
    too_wide = [
        line
        for line in printed.splitlines()
        if len(line) > 100
        and '"' not in line
        and scores_path.as_posix() not in line
        and candidates_path.as_posix() not in line
    ]
    assert not too_wide, f"{len(too_wide)} lines over 100 columns: {too_wide[:2]}"


def test_the_queue_reports_no_figure_about_itself(capsys: pytest.CaptureFixture[str]) -> None:
    """No figure is computed by this ticket, and the output does not hint at one.

    The headline figure for a Review Queue is the fraction of true findings within its top
    D entries, and it needs a judgement about what is a finding that no command here can
    make: every label in this Corpus was assigned by the generator that wrote the posts, so
    a figure over the queue would measure agreement with the generator (ADR-0004). The
    words that would name such a figure appear nowhere in the queue, which is asserted
    rather than trusted, in the way ADR-0018's whole-Corpus refusal is.
    """
    printed = run(COMMITTED_SCORES, COMMITTED_CANDIDATES, capsys)

    prose = "\n".join([opening(printed), *closing(printed)])
    for word in ("precision", "accuracy", "recall", "recovered"):
        assert word not in prose.lower(), f"the queue prints {word!r}"


def test_a_queue_of_no_posts_says_so_rather_than_printing_an_empty_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty scores file is a run over nothing, and it says which of the two it is.

    The figures above the table carry the count either way, so a reader is never left
    holding a header and no rows; but the absence of the rows is stated where the table
    would have been, because a heading followed by a silence is the one shape an empty run
    takes by accident.
    """
    scores_path, candidates_path = published(tmp_path)

    printed = run(scores_path, candidates_path, capsys, depth=50)

    assert "No post" in printed, printed
    assert "rank " not in printed, printed