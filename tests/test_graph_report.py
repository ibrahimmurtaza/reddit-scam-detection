"""One Campaign Candidate as one self-contained HTML file, drawn with inline SVG.

The visual is the one thing this project cannot check by counting, so what is checked
here is that it is a rendering of the run's own output and of nothing else: every node
and every edge is a row of `data/campaigns/campaign-candidates.jsonl`, every candidate
in that file gets one, and no candidate's picture is a hand-picked example.

The self-containment is checked by reading for what is not there rather than by opening a
browser. A file that names a stylesheet, a font, a script or an image has to reach
outside itself to render, and an outside reach is exactly what a report that has to open
with no network, no build step and no `node_modules` cannot have. So the drawing is
parsed as markup rather than eyeballed: an SVG a browser cannot parse is a page that
cannot render itself, and nothing else here would notice.

Nothing here is measured about the picture. The layout is a drawing, not a figure, and
the only thing it may decide for itself is where to put things. What it may not do is
drop a candidate, name an account the file does not hold, or draw an edge the file does
not state, and those are what these tests are about.

Seam under test: the `campaign-graph` command, observed through the files it writes.
Nothing here inspects the code that drew them, and no candidates file is written by the
module that reads it: a file this project produced could not be used to argue that the
graph listens to one.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import main

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CANDIDATES = REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl"
COMMITTED_GRAPHS = REPO_ROOT / "docs" / "campaign-graph"

# The four kinds of thing a Campaign Candidate is made of, and the four ways one reaches
# another. A picture missing one of the eight is not the argument the file claims to make.
KINDS = ("account", "post", "registration", "Contact Point")
RELATIONS = ("wrote", "links", "publishes", "shares")

# What a file that has to render with nothing but itself cannot contain. Checked as text
# rather than by resolving anything: the claim is that these are absent, and a browser
# reaching for one of them is what the check exists to stop.
OUTWARD = ("http", "src=", "href", "@import", "url(", "<script", "<link", "<img")

# The claims a picture must not make for this project, on a page and in the console
# alike. `fraud` is a word only a reviewer may apply; the other two are the glossary's.
OVERSTATES = ("fraud", "confirmed", "detection")

Row = Mapping[str, object]


class _Opened:
    """Records every path a run opens, so a claim about what it read can be checked.

    An audit hook rather than a monkeypatch, because the point is to catch a read from
    anywhere at all — including from inside the standard library, which a patched `open`
    would miss. It cannot be uninstalled, so it records only while `recording` is set and
    does nothing for the rest of the session.
    """

    def __init__(self) -> None:
        self.recording = False
        self.paths: list[str] = []

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        if self.recording and event == "open":
            self.paths.append(str(arguments[0]))

    def record(self) -> list[str]:
        return list(self.paths)


# --- reading the published file back ------------------------------------------------------------


def rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} should be text, is {value!r}"
    return value


def records(row: Row, field: str) -> list[Row]:
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    assert all(isinstance(entry, dict) for entry in value), f"{field} should hold rows"
    return [entry for entry in value if isinstance(entry, dict)]


def texts(row: Row, field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    return [str(entry) for entry in value]


# --- reading what the command drew ----------------------------------------------------------------


DRAWING = re.compile(r'<svg class="graph".*?</svg>', re.DOTALL)


def drawing(markup: str) -> ElementTree.Element:
    """The graph as a parsed tree, which is also the check that it is well formed.

    Parsed rather than searched, because the acceptance is that the file opens in a
    browser and draws itself: an unclosed element or an unescaped ampersand is markup a
    browser renders differently from the way it reads, and no other assertion here would
    notice either.
    """
    found = DRAWING.search(markup)
    assert found is not None, "the file holds no drawing"
    return ElementTree.fromstring(found.group())


def nodes_of(picture: ElementTree.Element, kind: str) -> list[str]:
    """The names of the nodes of one kind, in the order they are drawn."""
    return [
        node.get("data-name") or ""
        for node in picture.iter("g")
        if node.get("data-kind") == kind
    ]


def _stroke(markup: str, class_name: str) -> dict[str, str]:
    """One rule of the page's own stylesheet, as the declarations it holds.

    Read out of the file rather than out of the module, because the claim is about what a
    browser will do with the page: a stylesheet that named four classes and gave them the
    same stroke would still pass a test that only counted the classes.
    """
    found = re.search(rf"\.{re.escape(class_name)}\s*\{{([^}}]*)\}}", markup)
    assert found is not None, f"the stylesheet has no rule for .{class_name}"
    declarations = (part.strip() for part in found.group(1).split(";"))
    return dict(part.split(":", 1) for part in declarations if ":" in part)


def edges_of(picture: ElementTree.Element, relation: str) -> list[tuple[str, str]]:
    """The `from` and `to` of every edge of one relation."""
    return [
        (node.get("data-from") or "", node.get("data-to") or "")
        for node in picture.iter("path")
        if node.get("data-edge") == relation
    ]


def graphs_in(directory: Path) -> list[str]:
    """The files a run left in the directory, by name."""
    return sorted(path.name for path in directory.glob("*.html"))


def written(directory: Path, candidate_id: str) -> str:
    return (directory / f"{candidate_id}.html").read_text(encoding="utf-8")


def one_line(markup: str) -> str:
    """The page as one string, because the sentences are wrapped for a browser."""
    return " ".join(markup.split())


def refusing(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> str:
    """One refusal, with whatever the console printed before it."""
    with pytest.raises(SystemExit) as refusal:
        main(list(argv))
    return str(refusal.value) + capsys.readouterr().out


# --- the published candidates file, hand-written --------------------------------------------------


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


def candidate_row(
    candidate_id: str,
    accounts: Sequence[str],
    posts: Sequence[str],
    *domains: str,
    points: Sequence[str] = (),
    corroborating: Sequence[str] = ("temporal proximity", "content similarity",
                                    "category agreement"),
    first_seen: str = "2026-05-30T11:04:00Z",
) -> Row:
    """One row of `campaign-candidates.jsonl`, hand-written.

    The reader refuses anything it cannot check against the row beside it: a timing count
    that does not match the evidence the row names, a similarity row for a post the
    candidate does not hold, a cohesion count that is not the three verdicts this row
    produces. So the corroborations are named here and the figures are worked out from
    them, which is the only way a hand-written row can be one the grouping could really
    have published.

    `points` with no domain is the candidate resting on one Contact Point, which is the
    case the drawing has to get right: ADR-0005 lets a handle join two accounts on the
    same footing as a registration, and a layout that assumed every candidate has a
    registration would either drop this one or invent a registration for it.

    The gaps are a span outside the window whenever the clock is not among the
    corroborations holding, and the twins are further apart than the threshold whenever the
    vectors are not, because the reader counts both verdicts off the rows beside them
    rather than taking the row's word for either.
    """
    held = list(corroborating)
    span = 0 if "temporal proximity" in held else 2 * 86_400
    distance = 0.12 if "content similarity" in held else 0.72
    gaps = [{"kind": "registration", "value": domain,
             "closest_seconds": 0, "span_seconds": span}
            for domain in domains]
    gaps += [{"kind": "Contact Point", "value": value,
              "closest_seconds": 0, "span_seconds": span}
             for value in points]
    twins = [
        {
            "post": post,
            "account": accounts[index % len(accounts)],
            "nearest": posts[(index + 1) % len(posts)],
            "nearest_account": accounts[(index + 1) % len(accounts)],
            "distance": distance,
        }
        for index, post in enumerate(posts)
    ]
    return {
        "candidate_id": candidate_id,
        "accounts": list(accounts),
        "posts": list(posts),
        "shared_domains": [
            {"domain": domain, "accounts": list(accounts), "posts": list(posts)}
            for domain in domains
        ],
        "shared_contact_points": [
            {
                "kind": "telegram",
                "value": value,
                "accounts": list(accounts),
                "posts": list(posts),
                "spellings": [f"@{value}"],
            }
            for value in points
        ],
        "corroboration": gaps,
        "timing": {
            "window_seconds": 86_400,
            "pieces": len(gaps),
            "corroborated": len(gaps) if "temporal proximity" in held else 0,
            "closest_seconds": 0,
        },
        "similarity": twins,
        "content_similarity": {
            "threshold": 0.5,
            "pieces": len(posts),
            "corroborated": len(posts) if "content similarity" in held else 0,
            "closest_distance": distance,
        },
        "cohesion": {
            "against": sorted(
                {"temporal proximity", "content similarity", "category agreement"} - set(held)
            ),
            "category": {"tally": [["Work and Payroll", len(posts)]]},
            "corroborating": held,
            "retained": len(held) >= 2,
        },
        "first_seen": first_seen,
    }


ALL_THREE = ("temporal proximity", "content similarity", "category agreement")
CATEGORY_ONLY = ("category agreement",)


def on_a_registration(
    candidate_id: str, corroborating: Sequence[str] = ALL_THREE
) -> Row:
    """A candidate resting on one registration, which is the ordinary shape."""
    return candidate_row(
        candidate_id,
        ("syn_clearpathwork_3184", "syn_northwindhire_7736"),
        ("syn_p_0001", "syn_p_0002", "syn_p_0003"),
        "signal-harbor.example",
        corroborating=corroborating,
    )


def on_a_handle(
    candidate_id: str, corroborating: Sequence[str] = ALL_THREE
) -> Row:
    """A candidate resting on one Contact Point and on no registration at all."""
    return candidate_row(
        candidate_id,
        ("syn_halberdmoor_8823", "syn_thrushmoot_5524"),
        ("syn_p_0004", "syn_p_0005"),
        points=("syn_copperlantern",),
        corroborating=corroborating,
    )


def published(tmp_path: Path, *candidates: Row) -> tuple[Path, Path]:
    """A candidates file and the directory the command draws into."""
    return (
        write(tmp_path / "campaign-candidates.jsonl", candidates),
        tmp_path / "graph",
    )


def run(
    candidates_path: Path,
    graphs_path: Path,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> str:
    exit_code = main(
        ["campaign-graph", "--candidates", str(candidates_path), "--graphs", str(graphs_path)]
    )
    assert exit_code == 0
    return capsys.readouterr().out if capsys is not None else ""


def refusing_run(
    candidates_path: Path, graphs_path: Path, capsys: pytest.CaptureFixture[str]
) -> str:
    return refusing(
        ["campaign-graph", "--candidates", str(candidates_path), "--graphs", str(graphs_path)],
        capsys,
    )


# --- what the run writes ------------------------------------------------------------------------------


def test_the_run_draws_one_file_for_every_candidate_in_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every Campaign Candidate the run published gets a file of its own.

    Not the biggest, and not a chosen example. A graph of the clearest candidate is a
    demonstration; a graph for each candidate is the output, and the difference is the
    whole acceptance: a reader who disagrees with the grouping can open the one they
    disagree with. Each file is named by the candidate's own identifier, so the picture
    and the row it was drawn from are found by the same name.
    """
    candidates_path, graphs_path = published(
        tmp_path,
        on_a_registration("cc-01"),
        on_a_handle("cc-02"),
        candidate_row("cc-03", ("syn_a_0001", "syn_b_0002"), ("syn_p_0006", "syn_p_0007"),
                      "vantage-ledger.example", points=("syn_vantageledger",)),
    )

    printed = run(candidates_path, graphs_path, capsys)

    assert graphs_in(graphs_path) == ["cc-01.html", "cc-02.html", "cc-03.html"]
    assert "3 files, one per candidate" in printed, printed


def test_a_candidates_file_holding_nothing_draws_nothing_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No candidates is a state of the repository, not a missing step.

    `rfi campaign-candidates` writes an empty file when no two accounts share anything,
    so an empty candidates file is a real run's output. What it is not is a reason to
    draw an empty page: a directory holding one blank graph reads as a candidate that
    exists and is empty, which is a different claim from either of the two true ones.
    """
    candidates_path, graphs_path = published(tmp_path)

    printed = run(candidates_path, graphs_path, capsys)

    assert graphs_in(graphs_path) == []
    assert "no Campaign Candidate" in printed, printed


def test_a_graph_left_behind_by_an_earlier_run_is_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The directory cannot keep a picture of a candidate the file no longer holds.

    The candidates file is committed and a graph is committed beside it, so a run over a
    changed candidates file that only added and overwrote would leave a candidate's
    picture in the repository that nothing produces and nothing holds to its bytes — the
    one way this output could drift from the run without a single test failing. The
    removal is printed rather than done quietly, because a file this project deletes is
    a file somebody may have had open.
    """
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"))
    graphs_path.mkdir(parents=True)
    stale = graphs_path / "cc-09.html"
    stale.write_text("<p>a candidate the file no longer holds</p>", encoding="utf-8")

    printed = run(candidates_path, graphs_path, capsys)

    assert graphs_in(graphs_path) == ["cc-01.html"]
    assert not stale.exists()
    assert "cc-09.html" in printed, printed


def test_the_missing_candidates_file_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file that is not there is named, and so is the command that writes it.

    The same refusal every command reading a published file makes, and for the same
    reason: a bare `FileNotFoundError` from inside the reader says nothing about which
    step of the pipeline has not been run yet.
    """
    graphs_path = tmp_path / "graph"

    refusal = refusing(
        ["campaign-graph", "--candidates", str(tmp_path / "absent.jsonl"),
         "--graphs", str(graphs_path)],
        capsys,
    )

    assert "absent.jsonl" in refusal, refusal
    assert "rfi campaign-candidates" in refusal, refusal


# --- what the drawing shows -------------------------------------------------------------------------------


def test_a_graph_draws_accounts_posts_registrations_and_contact_points(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The four kinds of thing a candidate is made of are all on the picture.

    A candidate is an argument about a relationship between four kinds of thing, and a
    picture holding three of them is not that argument in a prettier shape: drop the posts
    and the registrations look arbitrary, drop the registrations and there is no claim
    being made at all. This one rests on both edges of ADR-0005, so all four kinds have
    to be there and each node has to carry the name the row gives it.
    """
    candidates_path, graphs_path = published(
        tmp_path,
        candidate_row(
            "cc-01",
            ("syn_clearpathwork_3184", "syn_northwindhire_7736"),
            ("syn_p_0001", "syn_p_0002", "syn_p_0003"),
            "signal-harbor.example",
            points=("syn_northwindhire",),
        ),
    )

    run(candidates_path, graphs_path, capsys)
    picture = drawing(written(graphs_path, "cc-01"))

    assert nodes_of(picture, "account") == ["syn_clearpathwork_3184", "syn_northwindhire_7736"]
    assert nodes_of(picture, "post") == ["syn_p_0001", "syn_p_0002", "syn_p_0003"]
    assert nodes_of(picture, "registration") == ["signal-harbor.example"]
    assert nodes_of(picture, "Contact Point") == ["syn_northwindhire"]


def test_a_graph_draws_the_relationships_and_the_stylesheet_tells_them_apart(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Four relations, drawn as four edges a reader can tell apart.

    The ticket asks for the relationship type to be visually distinguishable, and the
    edge from an account to the registration it shares is a different claim from the
    edge from a post to that same registration: one is why the accounts were put together
    and the other is how they got there. Drawing both as one line would make the
    strongest claim in the file the least visible thing on the page, so each relation is
    a class of its own and the stylesheet gives all four a different stroke.
    """
    candidates_path, graphs_path = published(
        tmp_path,
        candidate_row(
            "cc-01",
            ("syn_a_0001", "syn_b_0002"),
            ("syn_p_0001", "syn_p_0002"),
            "signal-harbor.example",
            points=("syn_copperlantern",),
        ),
    )

    run(candidates_path, graphs_path, capsys)
    markup = written(graphs_path, "cc-01")
    picture = drawing(markup)

    assert edges_of(picture, "wrote") == [
        ("syn_a_0001", "syn_p_0001"),
        ("syn_b_0002", "syn_p_0002"),
    ]
    assert edges_of(picture, "links") == [
        ("syn_p_0001", "signal-harbor.example"),
        ("syn_p_0002", "signal-harbor.example"),
    ]
    assert edges_of(picture, "publishes") == [
        ("syn_p_0001", "syn_copperlantern"),
        ("syn_p_0002", "syn_copperlantern"),
    ]
    assert edges_of(picture, "shares") == [
        ("syn_a_0001", "signal-harbor.example"),
        ("syn_b_0002", "signal-harbor.example"),
        ("syn_a_0001", "syn_copperlantern"),
        ("syn_b_0002", "syn_copperlantern"),
    ]
    patterns = {
        relation: _stroke(markup, f"edge-{relation}").get("stroke-dasharray")
        for relation in RELATIONS
    }
    assert len(set(patterns.values())) == len(RELATIONS), (
        f"two of the relations are drawn with the same pattern: {patterns}"
    )


def test_a_candidate_resting_on_one_handle_draws_that_edge_and_no_registration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A candidate on one Contact Point is drawn as a candidate on one Contact Point.

    ADR-0005 lets two accounts reach the same candidate on a handle as well as on a
    registration, and the two are not the same claim: a handle is read out of a post by a
    rule with a measured recall against a Labelled Set, while a registration is
    somebody's property and this project can check it against a published list. So the
    drawing carries whatever edges the row states and no others, and this row states
    three of the four relations and not the fourth.
    """
    candidates_path, graphs_path = published(tmp_path, on_a_handle("cc-01"))

    run(candidates_path, graphs_path, capsys)
    picture = drawing(written(graphs_path, "cc-01"))

    assert nodes_of(picture, "registration") == []
    assert nodes_of(picture, "Contact Point") == ["syn_copperlantern"]
    assert edges_of(picture, "links") == []
    assert edges_of(picture, "publishes") == [
        ("syn_p_0004", "syn_copperlantern"),
        ("syn_p_0005", "syn_copperlantern"),
    ]
    assert edges_of(picture, "shares") == [
        ("syn_halberdmoor_8823", "syn_copperlantern"),
        ("syn_thrushmoot_5524", "syn_copperlantern"),
    ]


def test_every_candidate_the_committed_run_produced_is_drawn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not one chosen example out of the four this repository publishes.

    The committed file holds a candidate on a registration alone, one on both edges and
    one on a handle alone, and one the cohesion system removed. All four are drawn, each
    with its own accounts, its own posts and its own evidence — and no candidate's file
    names anything belonging to another's, which is the half of the claim that catches a
    renderer carrying state from one candidate into the next.
    """
    candidates = rows(COMMITTED_CANDIDATES)
    graphs_path = tmp_path / "graph"

    run(COMMITTED_CANDIDATES, graphs_path, capsys)

    assert graphs_in(graphs_path) == [f"{text(row, 'candidate_id')}.html" for row in candidates]
    for row in candidates:
        picture = drawing(written(graphs_path, text(row, "candidate_id")))
        assert nodes_of(picture, "account") == texts(row, "accounts"), row["candidate_id"]
        assert nodes_of(picture, "post") == texts(row, "posts"), row["candidate_id"]
        assert nodes_of(picture, "registration") == [
            text(entry, "domain") for entry in records(row, "shared_domains")
        ], row["candidate_id"]
        assert nodes_of(picture, "Contact Point") == [
            text(entry, "value") for entry in records(row, "shared_contact_points")
        ], row["candidate_id"]

    everyone = {
        value
        for row in candidates
        for field in ("accounts", "posts")
        for value in texts(row, field)
    }
    for row in candidates:
        mine = set(texts(row, "accounts")) | set(texts(row, "posts"))
        markup = written(graphs_path, text(row, "candidate_id"))
        for stranger in sorted(everyone - mine):
            assert stranger not in markup, (
                f"{text(row, 'candidate_id')} draws {stranger}, which is in another candidate"
            )


def test_the_graph_states_the_cohesion_verdict_the_file_publishes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The verdict is on the page, with the corroborations it was counted from.

    The cohesion system retains a candidate on two of three named corroborations
    (ADR-0028), and a picture showing the accounts without the verdict would show a
    grouping a reader could not tell from one the system threw away. So each page names
    which corroborations held, which went against it, what that decided, and the window
    and the threshold it read those figures at — and the candidate the system removed
    says so on its own page rather than being left out of the picture altogether.
    """
    candidates_path, graphs_path = published(
        tmp_path,
        on_a_registration("cc-01"),
        on_a_handle("cc-02", corroborating=CATEGORY_ONLY),
    )

    run(candidates_path, graphs_path, capsys)
    kept = one_line(written(graphs_path, "cc-01"))
    removed = one_line(written(graphs_path, "cc-02"))

    assert "retained on 3 of 3" in kept, kept
    assert "temporal proximity" in kept and "category agreement" in kept, kept
    assert "against: none" in kept, kept
    assert "filtered on 1 of 3" in removed, removed
    assert "against: temporal proximity, content similarity" in removed, removed
    for page in (kept, removed):
        assert "24 hours" in page, page
        assert "0.5" in page, page


def test_every_node_and_its_label_is_inside_the_drawing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing is drawn past the edge of the page it is drawn on.

    A viewBox that stops above the last label is a clipped name, and a clipped name is the
    one thing on the page a reader cannot read at all. It is checked over the committed
    candidates rather than over a two-node example, because the height is decided by the
    tallest column and by the bowed lines underneath it: `cc-04` has seven posts and its
    arcs sit shallow, so it is the one page where the two disagree, and the three shorter
    pages would have hidden the bug.
    """
    graphs_path = tmp_path / "graph"

    run(COMMITTED_CANDIDATES, graphs_path, capsys)

    for candidate in rows(COMMITTED_CANDIDATES):
        name = text(candidate, "candidate_id")
        picture = drawing(written(graphs_path, name))
        _, _, width, height = _view_box(picture)
        assert (width, height) != (0.0, 0.0), f"{name} declares no viewBox"
        for label in picture.iter("text"):
            assert label.text is not None, f"{name} draws a label with no name in it"
            x, y = float(label.get("x", "nan")), float(label.get("y", "nan"))
            assert 0 <= x <= width, f"{name} draws {label.text} at x={x} of {width}"
            assert 0 <= y <= height, f"{name} draws {label.text} at y={y} of {height}"


def _view_box(picture: ElementTree.Element) -> tuple[float, float, float, float]:
    """The drawing's own coordinate space: the `min-x`, `min-y`, width and height it declares."""
    box = picture.get("viewBox")
    assert box is not None, "the drawing declares no viewBox"
    parts = box.split()
    assert len(parts) == 4, f"viewBox={box!r} is not four numbers"
    return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))


def test_the_graph_names_the_file_it_came_from_and_that_file_s_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The source path and the SHA-256 of the bytes it was drawn from are on the page.

    The candidates file is regenerable by another command over another Corpus, so a
    picture carrying nothing about which bytes produced it is a claim a reader has to
    take on trust — and this project has spent three tickets refusing to publish numbers
    of that kind. It is the same digest `rfi review-queue` prints beside the files it
    reads, and a reader can check it against the file with one command.
    """
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"))

    run(candidates_path, graphs_path, capsys)
    markup = one_line(written(graphs_path, "cc-01"))

    assert hashlib.sha256(candidates_path.read_bytes()).hexdigest() in markup, markup
    assert candidates_path.as_posix() in markup, markup


def test_the_graph_is_labelled_a_candidate_grouping_and_not_a_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The page says what it is on its face, before the picture is read.

    A Campaign Candidate is a proposal, not a finding about who is behind the accounts
    (ADR-0005), and a picture is the shape of that claim most likely to be believed past
    what it says: nodes joined by lines look like a structure somebody established. So the
    label is in the title of the page, in the line under the heading, and in the closing
    section — and no figure about fraud is named anywhere on it, because there is none to
    name (ADR-0004).
    """
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"))

    run(candidates_path, graphs_path, capsys)
    markup = written(graphs_path, "cc-01")

    assert "<title>" in markup and "Campaign Candidate cc-01" in markup, markup[:400]
    assert "a proposed grouping" in markup.lower(), markup[:400]
    assert "not a finding" in markup.lower(), markup[:400]
    for word in OVERSTATES:
        assert word not in markup.lower(), f"the page prints {word!r}"


# --- what the file may reach for ---------------------------------------------------------------------------


def test_the_styles_and_the_graphics_are_inline_so_the_file_opens_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing in the file points outside itself.

    That is the whole acceptance: a report a reader needs a toolchain, a build step or a
    `node_modules` for is a report most readers will never open, and the Next.js
    dashboard was deferred on exactly that ground. The file is checked for the outward
    reach rather than through a browser, because every one of these needles is something
    a renderer emits silently — a webfont link, a script tag, an image source — and
    none of them is something a browser would fail loudly over.
    """
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"))

    run(candidates_path, graphs_path, capsys)
    markup = written(graphs_path, "cc-01").lower()

    assert "<style>" in markup, "the styles are not in the file"
    assert "<svg" in markup, "the graphics are not in the file"
    for needle in OUTWARD:
        assert needle not in markup, f"the file reaches outside itself: {needle!r}"


def test_the_run_reads_the_candidates_file_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The picture is a function of the published file, and can reach nothing else.

    A renderer that re-ran the grouping could draw a Corpus this repository has no file
    for, and one that read the membership would put Planted Campaign answers into the one
    output a reader is most likely to believe on sight (ADR-0008). The graph is a
    projection, like the Review Queue, so it opens the candidates file and stops. Asserted
    on the paths a run opens, because what a reader can check is what the process touched.
    """
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"))
    opened = _Opened()
    sys.addaudithook(opened)

    opened.recording = True
    try:
        run(candidates_path, graphs_path, capsys)
    finally:
        opened.recording = False

    data_files = {
        Path(path).name for path in opened.record() if Path(path).suffix in {".jsonl", ".dat"}
    }
    assert data_files == {"campaign-candidates.jsonl"}
    assert not [path for path in opened.record() if Path(path).name.startswith("truth")]
    assert not [path for path in opened.record() if Path(path).name == "corpus.jsonl"]


def test_the_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same argument as every other command here: the numbers come from committed bytes."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the campaign-graph command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"))

    run(candidates_path, graphs_path)


# --- the committed graphs -------------------------------------------------------------------------------


def test_the_committed_graphs_are_the_ones_this_command_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graphs in this repository are the ones a run over the committed file produces.

    Every generated file here is committed and held to its bytes, and this one matters
    more than most: a picture is read as an artefact of the analysis rather than as a
    generated file, so a renderer that changed without the pages being redrawn would
    leave four files in the repository describing a graph this code does not draw. The
    run is made from the repository root over the default path, because each page names
    the source it was drawn from and an absolute path would make the comparison one of
    two different renderings rather than of bytes.
    """
    monkeypatch.chdir(REPO_ROOT)
    graphs_path = tmp_path / "graph"

    run(Path("data/campaigns/campaign-candidates.jsonl"), graphs_path, capsys)

    committed = graphs_in(COMMITTED_GRAPHS)
    assert committed, "no graph is committed"
    assert graphs_in(graphs_path) == committed
    # And the other way round: a page in the repository that no run over this candidates
    # file writes is a picture of a candidate the file no longer holds. Nothing generates
    # one, so only a hand-edited commit or a deleted page can leave it there, and the run
    # itself would not have noticed.
    assert graphs_in(COMMITTED_GRAPHS) == [
        f"{text(row, 'candidate_id')}.html" for row in rows(COMMITTED_CANDIDATES)
    ], "the repository holds a page for a candidate the candidates file does not name"
    for name in committed:
        assert (graphs_path / name).read_bytes() == (COMMITTED_GRAPHS / name).read_bytes(), (
            f"{name} is not what this command writes"
        )


def test_running_twice_draws_the_same_picture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The layout is a function of the row, so the same row draws the same picture.

    Coordinates are chosen by this command rather than measured, which is the one thing
    it is allowed to decide — but only that. A layout depending on a set's order or on a
    hash would rewrite every graph in the repository on an unrelated commit, and a diff
    of four pages of coordinates tells a reader nothing about the analysis.
    """
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"))

    run(candidates_path, graphs_path, capsys)
    first = written(graphs_path, "cc-01")
    run(candidates_path, graphs_path, capsys)

    assert first == written(graphs_path, "cc-01")


# --- the shape of the console output --------------------------------------------------------------------------


def test_the_console_output_is_ascii_and_wrapped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The console is pasted into issues and diffed between runs, so it has to be one
    thing on every terminal.

    The claim every table in this project makes about its own output: ASCII only, so a
    console that cannot encode anything else prints it the same way as one that can, and
    wrapped at the width a terminal gives you. The one kind of line allowed past that
    width is a path the caller chose, which the command does not control and this test
    asserts by name rather than by measuring.
    """
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"),
                                            on_a_handle("cc-02"))

    printed = run(candidates_path, graphs_path, capsys)

    assert printed.isascii(), "the console output carries a character a console may not have"
    too_wide = [
        line
        for line in printed.splitlines()
        if len(line) > 100
        and candidates_path.as_posix() not in line
        and graphs_path.as_posix() not in line
    ]
    assert not too_wide, f"{len(too_wide)} lines over 100 columns: {too_wide[:2]}"


def test_the_console_output_says_what_was_drawn_and_where_it_came_from(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The console names the file it read, its digest, and every page it wrote.

    The command's own output is where a reader looks before opening a browser, so it
    carries the provenance the page carries: which candidates file, what its bytes are,
    and where the pages are. It says what a graph is as well, because a file called
    `campaign-graph` that nobody has opened yet is a file whose claim has not been read.
    """
    candidates_path, graphs_path = published(tmp_path, on_a_registration("cc-01"),
                                            on_a_handle("cc-02"))

    printed = run(candidates_path, graphs_path, capsys)

    assert candidates_path.as_posix() in printed, printed
    assert hashlib.sha256(candidates_path.read_bytes()).hexdigest() in printed, printed
    assert graphs_path.as_posix() in printed, printed
    assert "cc-01.html" in printed and "cc-02.html" in printed, printed
    assert "a proposed grouping" in printed.lower(), printed
    for word in OVERSTATES:
        assert word not in printed.lower(), f"the console prints {word!r}"
