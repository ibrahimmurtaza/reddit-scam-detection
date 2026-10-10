"""One Campaign Candidate as one self-contained HTML file, drawn with inline SVG.

The visual argument is the one thing this project publishes that a reader cannot check by
counting, so this module is written to be the dullest possible rendering of what the
grouping already produced. It reads `data/campaigns/campaign-candidates.jsonl` and
nothing else — no Corpus, no Public Suffix List, no vector table, and no Planted Campaign
membership — and every node and every edge on the picture is a row of that file. A
drawing that re-ran the grouping could show a Corpus this repository has no file for, and
one that read the membership would put the evaluation's answers into the output a reader
is most likely to believe on sight (ADR-0008). So the graph is a projection, like the
Review Queue, and its test holds it to that by watching what a run opens.

One file per candidate, rather than one file for all of them. A graph of the largest
candidate would be a demonstration, and this is the output: a reader who disagrees with a
grouping has to be able to open the one they disagree with. Each file is named by the
candidate's own identifier, so a picture and the row it was drawn from are found by the
same name.

The file is one file. Styles, graphics and data are inline, there is no script and no
external resource of any kind, and it opens from a thumb drive with no network, no build
step and no `node_modules`. That is not a preference: the Next.js dashboard was deferred
precisely because the toolchain costs disk this project does not have, and a report that
needs a package manager is a report most readers will never open. The test holds the files
to that by reading them for the outward reach — a link, a script, an image, a webfont —
rather than by opening a browser, because those are the things a renderer emits silently
and a test is the only thing here that will notice.

What the picture shows is the whole of ADR-0005's two edges and no more: an account, a
post, a shared registration, a shared Contact Point, and the four ways one reaches
another. The edge from an account to the thing it shares is why the accounts were put
together and the edge from a post to the same thing is how they got there; drawing them
as one line would make the strongest claim on the page the least visible thing in it, so
each of the four is drawn as a class of its own, each of the four nodes is a shape of its
own, and the key is built from the same two functions as the drawing so it cannot drift
from it.

A candidate resting on one Contact Point is drawn with no registration on it, because a
Contact Point alone is a claim of its own: a handle is read out of a post by a rule whose
measured recall `rfi contact-points` publishes, while a registration is somebody's
property and this project can check it against a published list. The page says which
corroborations held, which went against the candidate, and whether the cohesion system
kept it or removed it (ADR-0028) — a picture of accounts without the verdict beside it is
a grouping a reader cannot tell from one the system threw away.

The layout is arithmetic over the row's own order: accounts in a column, posts in a
column, the things they share in a third, and the account-to-shared edges bowed beneath
so they cannot be mistaken for the post edges passing through. Nothing here is measured
and nothing is inferred. The coordinates are the only thing this command decides for
itself, and it decides them the same way for the same row every time, because a
repository of pictures that reorder themselves on an unrelated commit is a repository in
which every diff is noise.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from html import escape
from pathlib import Path

from reddit_fraud_intelligence.campaigns import (
    CORROBORATIONS,
    CampaignCandidate,
    CategoryAgreement,
    EvidenceKind,
    EvidenceTiming,
    cohesion_verdict,
    gap_of,
    names_of,
    read_campaign_candidates,
    window_of,
)

# The two headings, kept apart because they name two different things: the console prints
# the one the command is and every page prints the one the page is about.
_COMMAND = "Campaign Candidate graphs"
_PAGE = "Campaign Candidate"

_SUBHEADING = """\
One self-contained HTML file per Campaign Candidate, drawn from the candidates file alone.
Each one is a proposed grouping, not a finding about who is behind the accounts, and none
of them reaches outside itself to render: open one in a browser with no network attached."""


class NodeKind(StrEnum):
    """The four kinds of thing a Campaign Candidate is made of.

    Closed, and named in the glossary's own words, because a node of some fifth kind would
    be a claim the grouping does not make. `slug` is the name a stylesheet can use: a CSS
    class is not a place to put two words, and the Contact Point's glossary spelling is
    two of them.
    """

    ACCOUNT = "account"
    POST = "post"
    REGISTRATION = "registration"
    CONTACT_POINT = "Contact Point"

    @property
    def slug(self) -> str:
        """The name this kind takes in a class attribute."""
        return self.value.replace(" ", "-").lower()


class Relation(StrEnum):
    """The four ways one thing in a candidate reaches another.

    Closed, because these are the four relationships the row states and the picture is a
    rendering of them: an account writes a post, a post links a registration, a post
    publishes a Contact Point, and an account shares a registration or a Contact Point
    with another account. The fourth is the reason the candidate exists at all, so it is
    named on its own rather than folded in with the other three.
    """

    WROTE = "wrote"
    LINKS = "links"
    PUBLISHES = "publishes"
    SHARES = "shares"


@dataclass(frozen=True, slots=True)
class Node:
    """One thing in the picture, and what a reader hovering over it is told.

    The note is not decoration: the picture is where a reader looks first, and a node
    carrying a bare identifier is a node they have to go and look up. What it holds is
    what the row already publishes beside that identifier — an account's share of the
    candidate's posts, a post's nearest twin by another account, the accounts behind a
    registration — so nothing on the page is a figure this command worked out.
    """

    kind: NodeKind
    name: str
    note: str


@dataclass(frozen=True, slots=True)
class Edge:
    """One line: what it joins, and which of the four relationships it is."""

    relation: Relation
    source_kind: NodeKind
    source_name: str
    target_kind: NodeKind
    target_name: str

    @property
    def key(self) -> tuple[NodeKind, str, NodeKind, str]:
        """The two ends as one lookup key, for the drawing's tables of positions."""
        return (self.source_kind, self.source_name, self.target_kind, self.target_name)


@dataclass(frozen=True, slots=True)
class Arc:
    """One bowed line: where its control point sits, and how far down the curve really goes.

    The two are not the same number, and the difference is the empty band under the picture:
    a quadratic curve passes well short of its control point, so reserving the full depth
    of the control would leave a hundred-odd pixels of blank page under a drawing with no
    node in them.
    """

    middle: float
    control: float
    deepest: float


@dataclass(frozen=True, slots=True)
class SharedThing:
    """One shared thing, under every name the row and the picture know it by.

    The relation is carried because a post reaches a registration and a post reaches a
    Contact Point by different edges; the kind is carried because the picture draws the two
    as different shapes and the table names them differently; and the accounts and the
    posts are carried because they are the join itself rather than a count of it. One
    value rather than a five-tuple re-split at each of the four places that need it.
    """

    relation: Relation
    kind: NodeKind
    name: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphFacts:
    """What reading the candidates file establishes, as claims about bytes.

    The same figures `rfi review-queue` prints, over the one file it reads. The digest is
    here so a reader can tell which bytes a picture was drawn from, because the candidates
    file is regenerable by another command over another Corpus and a picture naming
    nothing about its own source is a claim to be taken on trust.
    """

    accounts: int
    candidates: int
    candidates_path: str
    candidates_sha256: str
    posts: int


@dataclass(frozen=True, slots=True)
class CandidateGraph:
    """One candidate, and the file it was read out of.

    The facts are carried rather than passed in, so the page and the console cannot quote
    two different digests for the same run — which is what would happen if the renderer
    took them as arguments and a caller passed the wrong one.
    """

    candidate: CampaignCandidate
    facts: GraphFacts


@dataclass(frozen=True, slots=True)
class Graphs:
    """Every candidate the file holds, and the figures about the file itself.

    The facts are here rather than derived from the candidates so that a file holding
    none still has a source, a digest and a count to print: an empty candidates file is a
    real state of this repository and not a missing step.
    """

    facts: GraphFacts
    candidates: tuple[CandidateGraph, ...]


@dataclass(frozen=True, slots=True)
class Written:
    """The pages a run wrote, and the ones it took out of the directory.

    Both are carried because both are decisions: a directory that only ever grows can
    keep a picture of a candidate the file no longer holds, which is the one way this
    output could drift from the run without a single test failing.
    """

    directory: Path
    pages: tuple[Path, ...]
    removed: tuple[Path, ...]


def read_graphs(candidates_path: Path) -> Graphs:
    """Read the published candidates and pair each one with what the file establishes.

    One call, for the same reason the grouping and the scoring are one call: the figures
    and the drawings are two views of one pass over one file, and a caller that read it
    twice could end up printing one run's digest over another's pictures.

    The candidates are not re-derived and the grouping is not re-run. Both are functions
    of the Corpus, both are held to the bytes their own command writes, and a picture of a
    grouping nobody published would describe a Corpus this repository has no file for.
    """
    candidates = read_campaign_candidates(candidates_path)
    facts = GraphFacts(
        accounts=len({account for row in candidates for account in row.accounts}),
        candidates=len(candidates),
        candidates_path=candidates_path.as_posix(),
        candidates_sha256=hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        posts=len({post for row in candidates for post in row.posts}),
    )
    return Graphs(
        facts=facts,
        candidates=tuple(CandidateGraph(candidate=row, facts=facts) for row in candidates),
    )


def write_graphs(graphs: Graphs, directory: Path) -> Written:
    """Draw every candidate into the directory, and clear out what this run did not draw.

    A file this project publishes into a directory it owns, where the previous run's output
    stays behind unless it is deliberately removed. That is the arrangement
    `rfi content-embeddings` needs `--replace` for, and it is not optional here: a
    candidate that disappears from the candidates file would otherwise leave a picture of
    it committed in the repository, describing a grouping nothing produces and nothing
    holds to its bytes.

    Only `.html` is removed, because only `.html` is what this command writes, and only
    files — a directory somebody put there by hand is left alone.
    """
    directory.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    for graph in graphs.candidates:
        page = directory / f"{graph.candidate.candidate_id}.html"
        page.write_text(render_graph(graph), encoding="utf-8", newline="\n")
        pages.append(page)

    written = {page.resolve() for page in pages}
    removed = tuple(
        sorted(path for path in directory.glob("*.html") if path.resolve() not in written)
    )
    for path in removed:
        path.unlink()
    return Written(directory=directory, pages=tuple(pages), removed=removed)


def render_graph(graph: CandidateGraph) -> str:
    """One candidate as one HTML page: the picture, the evidence, and what it is not.

    The three are in that order because that is the order a reader needs them in. The
    claim is on the page's face before the drawing is reached — a picture of joined nodes
    looks like a structure somebody established, and the label is what stops it being read
    past what it says (ADR-0005). The evidence is beside the picture rather than behind a
    link because there are no links: the whole page is the report.
    """
    candidate = graph.candidate
    heading = f"{_PAGE} {candidate.candidate_id}"
    return "\n".join(
        (
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_text(heading)}: {_text(_claim(candidate))}</title>",
            f"<style>{_STYLES}</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{_text(heading)}</h1>",
            f'<p class="claim">{_text(_claim(candidate))}</p>',
            _figures_section(graph),
            _figure(candidate),
            _evidence_section(candidate),
            _corroboration_section(candidate),
            _page_footer(graph),
            "</main>",
            "</body>",
            "</html>",
            "",
        )
    )


def _claim(candidate: CampaignCandidate) -> str:
    """What the page is, in one sentence, in the project's own words.

    The same sentence on every page whatever the candidate holds, because the claim is
    about the kind of thing the file is and not about how much evidence any one of them
    has behind it: a candidate with four corroborations is a proposal in exactly the way a
    candidate with one is.
    """
    return (
        f"A proposed grouping of {_count(len(candidate.accounts), 'account')}, not a finding "
        "about who is behind them"
    )


# --- the console output ---------------------------------------------------------------------------


def render_table(graphs: Graphs, written: Written) -> str:
    """The console output: what was read, what was drawn, and what that is.

    ASCII only, so it prints the same way on a console that cannot encode anything else
    and the same way when it is redirected, which is what lets it be pasted into an issue
    or diffed between runs.
    """
    sections = (
        f"{_COMMAND}\n\n{_SUBHEADING}",
        _figures(graphs, written),
        _index(graphs),
        _console_footer(),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _figures(graphs: Graphs, written: Written) -> str:
    """What was read, and what came of it. One figure per line, labelled."""
    facts = graphs.facts
    lines = [
        (
            "candidates",
            f"{facts.candidates_path} ({_count(facts.candidates, 'candidate')}, "
            f"{_count(facts.accounts, 'account')} and {_count(facts.posts, 'post')})",
        ),
        ("sha256", facts.candidates_sha256),
        (
            "graphs",
            (
                f"{written.directory.as_posix()} "
                f"({len(written.pages)} files, one per candidate)"
                if written.pages
                else f"{written.directory.as_posix()} (nothing to draw)"
            ),
        ),
        *(
            (
                "removed",
                f"{path.name} (the candidates file does not hold that candidate any more)",
            )
            for path in written.removed
        ),
    ]
    width = max(len(name) for name, _ in lines)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in lines)


def _index(graphs: Graphs) -> str:
    """One line per candidate, in the order the pages are named.

    The corroborations column counts rather than names them, because the names are three
    and the page beside this line carries all of them: a reader scanning the console wants
    to see at a glance which candidates the system retained and which it filtered, and a
    reader who wants to know why opens the file whose name is on the row. `retained` and
    `filtered` are the grouping's own words for that verdict rather than softer ones, so
    the column and the page beside it say the same thing in the same terms.
    """
    if not graphs.candidates:
        return (
            f"That file holds no Campaign Candidate, so no page was drawn into "
            f"{graphs.facts.candidates_path}."
        )

    headings = (
        "candidate",
        "accounts",
        "posts",
        "registrations",
        "Contact Points",
        "corroborations",
        "cohesion",
        "page",
    )
    rows = [
        (
            candidate.candidate_id,
            str(len(candidate.accounts)),
            str(len(candidate.posts)),
            str(len(candidate.shared_domains)),
            str(len(candidate.shared_contact_points)),
            f"{len(candidate.cohesion.corroborating)} of {len(CORROBORATIONS)}",
            "retained" if candidate.cohesion.retained else "filtered",
            f"{candidate.candidate_id}.html",
        )
        for candidate in (graph.candidate for graph in graphs.candidates)
    ]
    widths = [max(len(cell) for cell in column) for column in zip(headings, *rows, strict=True)]
    return "\n".join(
        ["  " + _row(headings, widths), *("  " + _row(row, widths) for row in rows)]
    )


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """One line of the index, counts right-aligned so the digits line up down the column."""
    return "  ".join(
        cell.rjust(width) if cell.isdigit() else cell.ljust(width)
        for cell, width in zip(cells, widths, strict=True)
    )


def _console_footer() -> str:
    """What a graph is, and what a reader must not take from one.

    The same two every command in this project states: a Campaign Candidate is a proposal
    rather than a statement about anybody (ADR-0005), and a Contact Point is the weaker of
    the two readings ADR-0005 permits, with the measured recall sitting in another
    command's report rather than here.
    """
    return """\
Every page above is a proposed grouping, never a statement about what the accounts behind
it are doing (ADR-0005). Each one names the candidate it drew, the registrations and
Contact Points that put the accounts together, the three corroborations and what each of
them read, and whether the cohesion system kept the candidate or removed it.

These pages are drawn from the published candidates file and from nothing else. The Corpus
is not re-read, the grouping is not re-run, the Planted Campaign membership is not opened
(ADR-0008), and the Confidence is displayed nowhere (ADR-0003, ADR-0027). What a reviewer
should read first is `rfi review-queue`, and what the grouping recovered of the planted
structure is `rfi campaign-recovery`."""


# --- the page -----------------------------------------------------------------------------------------


def _figures_section(graph: CandidateGraph) -> str:
    """What this candidate is, over the bytes it was drawn from."""
    candidate = graph.candidate
    facts = graph.facts
    cohesion = candidate.cohesion
    rows = (
        ("accounts", _names(candidate.accounts)),
        ("posts", _names(candidate.posts)),
        ("first seen", candidate.first_seen),
        ("joined on", candidate.joined_on()),
        (
            "corroborating",
            f"{names_of(cohesion.corroborating)}; against it: {names_of(cohesion.against)}",
        ),
        ("cohesion", _verdict(candidate)),
        ("candidates file", facts.candidates_path),
        ("candidates sha256", facts.candidates_sha256),
        ("this candidate", f"{candidate.candidate_id} of {facts.candidates} in that file"),
    )
    return '<dl class="facts">\n' + "\n".join(
        f"<dt>{_text(name)}</dt><dd>{_text(value)}</dd>" for name, value in rows
    ) + "\n</dl>"


def _evidence_kind(kind: NodeKind) -> EvidenceKind:
    """The grouping's own name for one kind of shared thing.

    Two vocabularies for the same two things, and this is the one place they meet: the
    grouping times its evidence and calls the kind `EvidenceKind`, while this module draws
    it and calls the same kind a node. They are the same two names under ADR-0005's two
    edges, so the conversion happens once here rather than a second spelling of each kind
    being written out wherever an edge meets a gap.
    """
    return EvidenceKind(kind.value)


def _figure(candidate: CampaignCandidate) -> str:
    """The drawing, and the key to it, built from the same two functions.

    The key is not a second description of the picture: each swatch is the shape function
    and each line is the edge class, so a swatch cannot stop matching what it stands for
    without the page's own tests failing.
    """
    return "\n".join(
        [
            "<figure>",
            _drawing(candidate),
            "<figcaption>",
            "<h2>What the shapes and the lines are</h2>",
            '<ul class="key-list">',
            *(_shape_key(kind, label) for kind, label in _SHAPE_KEY),
            *(_edge_key(relation, label) for relation, label in _RELATION_KEY),
            "</ul>",
            "</figcaption>",
            "</figure>",
        ]
    )


def _evidence_section(candidate: CampaignCandidate) -> str:
    """The shared things, with the accounts and posts that reach them, and their timing.

    The table is the picture's argument written out, because a drawing can show that two
    accounts reach one registration and cannot show how close in time they posted. The
    gaps are the row's own figures and the window is the one they were counted against
    (ADR-0025), read through `EvidenceTiming.within` rather than restated, so this column
    cannot disagree with the timing the candidates file publishes.
    """
    gaps = {(piece.kind, piece.value): piece for piece in candidate.corroboration}
    window = candidate.timing.window_seconds
    return "\n".join(
        [
            "<section>",
            "<h2>The shared things this grouping rests on</h2>",
            "<table>",
            "<thead><tr><th>Shared thing</th><th>Accounts reaching it</th>"
            "<th>Posts reaching it</th><th>Closest gap</th><th>Span</th>"
            f"<th>Within the {window_of(window)} window</th></tr></thead>",
            "<tbody>",
            *(_evidence_row(gaps, thing, window) for thing in _shared_things(candidate)),
            "</tbody>",
            "</table>",
            "</section>",
        ]
    )


def _evidence_row(
    gaps: Mapping[tuple[EvidenceKind, str], EvidenceTiming],
    thing: SharedThing,
    window: int,
) -> str:
    """One piece of evidence: who reaches it, how close in time, and whether that is inside
    the window.

    The account and post lists are the join itself rather than a count, because the
    reader's question about a shared thing is which accounts reach it, and a count does not
    answer it.
    """
    piece = gaps[(_evidence_kind(thing.kind), thing.name)]
    within = piece.within(window)
    cells = (
        _text(thing.name if thing.kind is NodeKind.REGISTRATION else f"{thing.name} ({piece.kind.value})"),
        _text(_names(thing.accounts)),
        _text(_names(thing.posts)),
        _text(gap_of(piece.closest_seconds)),
        _text(gap_of(piece.span_seconds)),
        f'<span class="{"verdict-holds" if within else "verdict-against"}">'
        f'{"yes" if within else "no"}</span>',
    )
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _corroboration_section(candidate: CampaignCandidate) -> str:
    """The three verdicts, what each read, and what the count of them decided.

    Named rather than summarised (ADR-0028): the cohesion system is a count of named
    verdicts, and a page carrying the count without the names would ask a reader to take
    on trust the one figure this page exists to make checkable.
    """
    cohesion = candidate.cohesion
    reads = {
        "temporal proximity": _timing_read(candidate),
        "content similarity": _similarity_read(candidate),
        "category agreement": _category_read(cohesion.category),
    }
    return "\n".join(
        [
            "<section>",
            "<h2>The corroborations</h2>",
            f'<p class="verdict">Cohesion: {_text(_verdict(candidate))}</p>',
            "<table>",
            "<thead><tr><th>Corroboration</th><th>Verdict</th><th>What it read</th></tr>"
            "</thead>",
            "<tbody>",
            *(
                f"<tr><td>{_text(name)}</td>"
                f'<td><span class="verdict-{"holds" if name in cohesion.corroborating else "against"}">'
                f'{"holds" if name in cohesion.corroborating else "against"}</span></td>'
                f"<td>{_text(reads[name])}</td></tr>"
                for name in CORROBORATIONS
            ),
            "</tbody>",
            "</table>",
            "</section>",
        ]
    )


def _verdict(candidate: CampaignCandidate) -> str:
    """What the cohesion system decided, in the grouping's own words.

    `cohesion_verdict` rather than a sentence of this module's own: `retained` and
    `filtered` are the run's words for the one verdict that removes a candidate from what
    the system proposes, and this page says the same thing the console says under the
    candidate it was drawn from. A reader who has read one and then opened the other has
    not met two accounts of one decision.
    """
    return cohesion_verdict(candidate)


def _page_footer(graph: CandidateGraph) -> str:
    """What the page can and cannot claim, and where it came from.

    The three limits a reader of a picture would otherwise have to guess at: it is a
    proposal, a Contact Point is the weaker of the two readings, and nothing on the page
    was worked out here — every node and edge is a row of the file named above it, and the
    only thing this command decided for itself was where to draw them.
    """
    facts = graph.facts
    others = facts.candidates - 1
    paragraphs = (
        "A Campaign Candidate is a hypothesis produced by cohesion analysis, never a "
        "statement about what the accounts behind it are doing (ADR-0005). Nothing on this "
        "page says the content is anything in particular, and no person has reviewed any "
        "of it.",
        "A shared Contact Point is the weaker of the two readings ADR-0005 permits: a "
        "registration is somebody's property and this project checks it against a published "
        "Public Suffix List, while a Contact Point is read out of a post by a rule whose "
        "measured recall against a Labelled Set is published by "
        "<code>rfi contact-points</code>. A candidate resting on a Contact Point alone "
        "rests on that reading alone, and the table above says so.",
        f"Every node and every edge on the picture is a row of "
        f"<code>{_text(facts.candidates_path)}</code>, whose SHA-256 is above. The Corpus "
        "is not re-read, the grouping is not re-run, the Planted Campaign membership is not "
        "opened (ADR-0008), and the Confidence is displayed nowhere (ADR-0003, ADR-0027). "
        "The only thing this command decided for itself was where to draw each thing, and "
        "it decides it the same way for the same row every time.",
        (
            f"The other {_count(others, 'candidate')} in that file "
            f"{'is a page of its own beside this one' if others == 1 else 'are pages of their own beside this one'}."
        )
        if others
        else "This is the only Campaign Candidate in that file.",
    )
    return "<footer>\n" + "\n".join(f"<p>{paragraph}</p>" for paragraph in paragraphs) + "\n</footer>"


# --- the drawing -------------------------------------------------------------------------------------------


def _drawing(candidate: CampaignCandidate) -> str:
    """The picture: the edges beneath the nodes, in a grid of three columns.

    Edges first and the nodes over them, so a line stops at the edge of a shape rather
    than running under a label: the join between two centres is a straight line and the
    node covers whatever is inside it. The account-to-shared edges are bowed underneath
    rather than drawn straight, because those are the ones crossing the whole picture — a
    straight line from an account to a registration would pass behind every post between
    them and read as though it joined one.
    """
    nodes = _nodes(candidate)
    edges = _edges(candidate)
    columns = _columns(nodes)
    placed = _place(columns, _centres_x(columns))
    arcs = _arcs(candidate, edges, placed)
    width, height = _extent(columns, arcs)

    lines = [
        f'<svg class="graph" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-labelledby="graph-title graph-desc">',
        f'<title id="graph-title">{_text(_summary(candidate))}</title>',
        '<desc id="graph-desc">'
        + _text(
            "Accounts in the left column, the posts they wrote in the middle, and the "
            "registrations and Contact Points they share in the right. Every account is "
            "drawn joined to each thing it reaches, which is the whole reason those "
            "accounts are in one candidate."
        )
        + "</desc>",
        '<g class="edges">',
        *(_edge_markup(edge, placed, arcs) for edge in edges),
        "</g>",
        '<g class="nodes">',
        *(_node_markup(node, placed[(node.kind, node.name)]) for node in nodes),
        "</g>",
        "</svg>",
    ]
    return "\n".join(lines)


def _summary(candidate: CampaignCandidate) -> str:
    """What the picture holds, for the title a screen reader and a browser tab read."""
    return (
        f"{candidate.candidate_id}: {_count(len(candidate.accounts), 'account')}, "
        f"{_count(len(candidate.posts), 'post')}, "
        f"{_count(len(candidate.shared_domains), 'shared registration')} and "
        f"{_count(len(candidate.shared_contact_points), 'shared Contact Point')}"
    )


def _nodes(candidate: CampaignCandidate) -> tuple[Node, ...]:
    """Every node in the picture, in the order the columns are laid out.

    Accounts, then posts, then the shared registrations, then the shared Contact Points.
    The row's own order and the drawing's column order at once, so a picture and the table
    it was drawn from are read in the same sequence.
    """
    authors, twins = _authors(candidate)
    posts_of = _posts_by_account(candidate)
    return (
        *(
            Node(
                kind=NodeKind.ACCOUNT,
                name=account,
                note=(
                    f"wrote {len(posts_of.get(account, ()))} of this candidate's "
                    f"{_count(len(candidate.posts), 'post')}"
                ),
            )
            for account in candidate.accounts
        ),
        *(
            Node(kind=NodeKind.POST, name=post, note=twins[post])
            for post in candidate.posts
        ),
        *(
            Node(
                kind=NodeKind.REGISTRATION,
                name=shared.domain,
                note=(
                    f"a Registrable Domain reached by "
                    f"{_count(len(shared.accounts), 'account')} of this candidate, from "
                    f"{_count(len(shared.posts), 'post')}"
                ),
            )
            for shared in candidate.shared_domains
        ),
        *(
            Node(
                kind=NodeKind.CONTACT_POINT,
                name=point.value,
                note=(
                    f"a {point.kind.value} Contact Point reached by "
                    f"{_count(len(point.accounts), 'account')} of this candidate; the "
                    f"Corpus writes it as {_names(point.spellings)}"
                ),
            )
            for point in candidate.shared_contact_points
        ),
    )


def _edges(candidate: CampaignCandidate) -> tuple[Edge, ...]:
    """Every edge in the picture, each one a relationship the row states.

    The account a post belongs to is read off the similarity rows rather than off a second
    input file, because the candidates file publishes one row per post and the grouping
    computed it from the Corpus: asking the page to re-read the Corpus to learn who wrote a
    post would make the graph a function of two files rather than one.

    A piece of evidence names every post in the Corpus reaching it, and some of those
    belong to accounts outside this candidate. They are left off the drawing: an edge to a
    post nothing else in the picture names is a line into nowhere, and the candidate's own
    posts are what this page is about.
    """
    authors, _ = _authors(candidate)
    held = set(candidate.posts)
    shared = _shared_things(candidate)
    return (
        *(
            Edge(
                relation=Relation.WROTE,
                source_kind=NodeKind.ACCOUNT,
                source_name=authors[post],
                target_kind=NodeKind.POST,
                target_name=post,
            )
            for post in candidate.posts
        ),
        *(
            Edge(
                relation=thing.relation,
                source_kind=NodeKind.POST,
                source_name=post,
                target_kind=thing.kind,
                target_name=thing.name,
            )
            for thing in shared
            for post in thing.posts
            if post in held
        ),
        *(
            Edge(
                relation=Relation.SHARES,
                source_kind=NodeKind.ACCOUNT,
                source_name=account,
                target_kind=thing.kind,
                target_name=thing.name,
            )
            for thing in shared
            for account in thing.accounts
        ),
    )


def _shared_things(candidate: CampaignCandidate) -> tuple[SharedThing, ...]:
    """Every shared thing in the candidate, registrations first.

    One list rather than two, because both edges are drawn and both are tabulated as the
    same kind of thing: what tells them apart is which accounts reach it, and the two
    vocabularies are there so the reader can be told which is which.

    The picture and the table under it are both built off this one function, so a piece of
    evidence cannot be drawn and not tabulated.
    """
    return (
        *(
            SharedThing(
                relation=Relation.LINKS,
                kind=NodeKind.REGISTRATION,
                name=shared.domain,
                accounts=shared.accounts,
                posts=shared.posts,
            )
            for shared in candidate.shared_domains
        ),
        *(
            SharedThing(
                relation=Relation.PUBLISHES,
                kind=NodeKind.CONTACT_POINT,
                name=point.value,
                accounts=point.accounts,
                posts=point.posts,
            )
            for point in candidate.shared_contact_points
        ),
    )


def _authors(candidate: CampaignCandidate) -> tuple[Mapping[str, str], Mapping[str, str]]:
    """Who wrote each post, and how far it is from the nearest post by another account.

    Both off the similarity rows the file publishes, one row per post the candidate holds:
    the reading is the grouping's own and it is already beside the identifier, so the page
    carries it rather than working it out again from vectors this command cannot see.
    """
    authors = {row.post: row.account for row in candidate.similarity}
    twins = {
        row.post: (
            f"written by {row.account}; nearest post by another account: {row.nearest} by "
            f"{row.nearest_account}, {row.distance:.2f} apart"
        )
        for row in candidate.similarity
    }
    return authors, twins


def _posts_by_account(candidate: CampaignCandidate) -> Mapping[str, tuple[str, ...]]:
    """Which of the candidate's posts each account wrote."""
    authors, _ = _authors(candidate)
    grouped: dict[str, list[str]] = {}
    for post in candidate.posts:
        grouped.setdefault(authors[post], []).append(post)
    return {account: tuple(posts) for account, posts in grouped.items()}


def _node_markup(node: Node, centre: tuple[float, float]) -> str:
    """One node: its shape, its name under it, and what hovering over it says."""
    x, y = centre
    return "\n".join(
        [
            f'<g class="node" data-kind="{escape(node.kind.value)}" '
            f'data-name="{escape(node.name)}">',
            f"<title>{_text(node.name)}: {_text(node.note)}</title>",
            _shape(node.kind, x, y),
            f'<text class="node-label" x="{_coordinate(x)}" y="{_coordinate(y + _SHAPE / 2 + 16)}">'
            f"{_text(node.name)}</text>",
            "</g>",
        ]
    )


def _edge_markup(
    edge: Edge,
    placed: Mapping[tuple[NodeKind, str], tuple[float, float]],
    arcs: Mapping[tuple[NodeKind, str, NodeKind, str], Arc],
) -> str:
    """One line, carrying the two things it joins and the relationship it is.

    `data-from` and `data-to` are on every edge because the claim the picture makes is
    about which account reaches which shared thing, and a line that cannot be read back is
    a line nobody can check.
    """
    (x1, y1) = placed[(edge.source_kind, edge.source_name)]
    (x2, y2) = placed[(edge.target_kind, edge.target_name)]
    attributes = (
        f'class="edge edge-{edge.relation.value}" data-edge="{edge.relation.value}" '
        f'data-from="{escape(edge.source_name)}" data-to="{escape(edge.target_name)}"'
    )
    if edge.key in arcs:
        arc = arcs[edge.key]
        route = (
            f"M {_coordinate(x1)} {_coordinate(y1)} Q {_coordinate(arc.middle)} {_coordinate(arc.control)} {_coordinate(x2)} {_coordinate(y2)}"
        )
    else:
        route = f"M {_coordinate(x1)} {_coordinate(y1)} L {_coordinate(x2)} {_coordinate(y2)}"
    return f"<path {attributes} d=\"{route}\"/>"


def _shape(kind: NodeKind, x: float, y: float) -> str:
    """The shape one kind of node is drawn as.

    Four shapes rather than four colours, because a colour is the first thing to go on a
    page printed in black and white or read by somebody who cannot tell them apart. A
    circle for an account, a square for a post, a hexagon for a registration and a rounded
    bar for a Contact Point — the last two wider than they are tall, because a domain and a
    handle are the two longest labels on the page.

    Exhaustive rather than a cascade that ends in a default, so a fifth kind of node is
    refused instead of being drawn as the fourth: a shape that means another thing is a
    wrong claim rather than a missing one.
    """
    attribute = f'class="shape shape-{kind.slug}"'
    if kind is NodeKind.ACCOUNT:
        return (
            f'<circle {attribute} cx="{_coordinate(x)}" cy="{_coordinate(y)}" '
            f'r="{_coordinate(_SHAPE / 2 - 2)}"/>'
        )
    if kind is NodeKind.POST:
        half = _SHAPE / 2 - 4
        return (
            f'<rect {attribute} x="{_coordinate(x - half)}" y="{_coordinate(y - half)}" '
            f'width="{_coordinate(half * 2)}" height="{_coordinate(half * 2)}"/>'
        )
    if kind is NodeKind.REGISTRATION:
        wide, tall = _SHAPE / 2, _SHAPE / 2 - 4
        points = (
            f"{_coordinate(x - wide)},{_coordinate(y)} "
            f"{_coordinate(x - wide / 2)},{_coordinate(y - tall)} "
            f"{_coordinate(x + wide / 2)},{_coordinate(y - tall)} "
            f"{_coordinate(x + wide)},{_coordinate(y)} "
            f"{_coordinate(x + wide / 2)},{_coordinate(y + tall)} "
            f"{_coordinate(x - wide / 2)},{_coordinate(y + tall)}"
        )
        return f"<polygon {attribute} points=\"{points}\"/>"
    if kind is NodeKind.CONTACT_POINT:
        wide, tall = _SHAPE / 2 - 2, _SHAPE / 2 - 8
        return (
            f'<rect {attribute} x="{_coordinate(x - wide)}" y="{_coordinate(y - tall)}" '
            f'width="{_coordinate(wide * 2)}" height="{_coordinate(tall * 2)}" '
            f'rx="{_coordinate(tall)}"/>'
        )
    raise ValueError(
        f"{kind.value} is a node kind this drawing has no shape for, and drawing it as "
        "something else would put a shape on the page that means another thing"
    )


def _shape_key(kind: NodeKind, label: str) -> str:
    """One swatch in the key, drawn by the function that draws the node."""
    return (
        f'<li><svg class="key-shape" viewBox="0 0 {_SHAPE} {_SHAPE}" width="26" height="26" '
        f'aria-hidden="true">{_shape(kind, _SHAPE / 2, _SHAPE / 2)}</svg>'
        f"{_text(label)}</li>"
    )


def _edge_key(relation: Relation, label: str) -> str:
    """One line in the key, drawn in the class the edge itself is drawn in."""
    return (
        f'<li><svg class="key-edge" viewBox="0 0 {_SHAPE} 16" width="26" height="16" '
        f'aria-hidden="true"><path class="edge edge-{relation.value}" d="M 2 8 L '
        f'{_coordinate(_SHAPE - 2)} 8"/></svg>{_text(label)}</li>'
    )


def _columns(nodes: Sequence[Node]) -> tuple[tuple[Node, ...], ...]:
    """The three columns, in the order the drawing lays them out from left to right.

    Posts in the middle rather than beside their accounts, so each of the four kinds of
    node is a column of its own and the argument reads left to right: what the accounts
    are, what they wrote, and what they share.
    """
    return (
        tuple(node for node in nodes if node.kind is NodeKind.ACCOUNT),
        tuple(node for node in nodes if node.kind is NodeKind.POST),
        tuple(
            node
            for node in nodes
            if node.kind in (NodeKind.REGISTRATION, NodeKind.CONTACT_POINT)
        ),
    )


def _centres_x(columns: Sequence[Sequence[Node]]) -> tuple[float, ...]:
    """Where each column's centre sits, each column as wide as its own longest label.

    Widths are measured rather than fixed, because the labels are ids and domains this
    project does not choose: a fixed column either clips a long registration or wastes the
    width of a short handle, and the first is a graph that cannot be read.
    """
    left: float = _MARGIN
    centres: list[float] = []
    for column in columns:
        width = _column_width(column)
        centres.append(left + width / 2)
        left += width + _GUTTER
    return tuple(centres)


def _place(
    columns: Sequence[Sequence[Node]], centres_x: Sequence[float]
) -> Mapping[tuple[NodeKind, str], tuple[float, float]]:
    """Where every node is drawn, each column centred against the tallest one.

    Centred rather than topped up, so a candidate with two accounts and seven posts does
    not have its two accounts bunched against the top of the picture while the posts run
    down the middle of it.
    """
    rows = max(len(column) for column in columns)
    placed: dict[tuple[NodeKind, str], tuple[float, float]] = {}
    for x, column in zip(centres_x, columns, strict=True):
        first = _MARGIN + _ROW / 2 + (rows - len(column)) * _ROW / 2
        for index, node in enumerate(column):
            placed[(node.kind, node.name)] = (x, first + index * _ROW)
    return placed


def _arcs(
    candidate: CampaignCandidate,
    edges: Sequence[Edge],
    placed: Mapping[tuple[NodeKind, str], tuple[float, float]],
) -> Mapping[tuple[NodeKind, str, NodeKind, str], Arc]:
    """The bowed lines, and how deep each one is drawn.

    Only the account-to-shared edges, which are the ones crossing the whole picture. Each
    account's line is bowed deeper than the account above it, so two accounts reaching the
    same thing are two curves rather than one drawn twice.
    """
    depth = {account: index for index, account in enumerate(candidate.accounts)}
    arcs: dict[tuple[NodeKind, str, NodeKind, str], Arc] = {}
    for edge in edges:
        if edge.relation is not Relation.SHARES:
            continue
        (x1, y1) = placed[(edge.source_kind, edge.source_name)]
        (x2, y2) = placed[(edge.target_kind, edge.target_name)]
        control = max(y1, y2) + _ARC + _ARC_STEP * depth[edge.source_name]
        arcs[edge.key] = Arc(
            middle=(x1 + x2) / 2, control=control, deepest=_deepest(y1, y2, control)
        )
    return arcs


def _deepest(y1: float, y2: float, control: float) -> float:
    """The lowest point of one bowed line, which is not where its control point is.

    The curve turns where the two ends pull equally hard, which for a quadratic is at
    `t = (control - y1) / (2 * control - y1 - y2)` along the line. A control at or below
    both ends is not a bow at all and falls back to the ends themselves.
    """
    span = 2 * control - y1 - y2
    if span <= 0:
        return max(y1, y2)
    at = (control - y1) / span
    return (1 - at) ** 2 * y1 + 2 * (1 - at) * at * control + at**2 * y2


def _extent(
    columns: Sequence[Sequence[Node]],
    arcs: Mapping[tuple[NodeKind, str, NodeKind, str], Arc],
) -> tuple[int, int]:
    """The picture's width and height: wide enough for the labels, deep enough for the
    bowed lines underneath them."""
    width = (
        _MARGIN * 2
        + sum(_column_width(column) for column in columns)
        + _GUTTER * (len(columns) - 1)
    )
    rows = max(len(column) for column in columns)
    bottom = _MARGIN + (rows - 1) * _ROW + _ROW / 2 + _SHAPE / 2 + _LABEL
    bowed = max((arc.deepest for arc in arcs.values()), default=bottom) + _LABEL
    return round(width), round(max(bottom, bowed) + _MARGIN)


def _column_width(column: Sequence[Node]) -> float:
    """How wide a column is drawn: its longest label, and never narrower than a shape."""
    return max([_SHAPE, *(_text_width(node.name) for node in column)])


def _text_width(label: str) -> float:
    """How wide one label is drawn.

    The advance of the label font at the size the stylesheet sets it. A monospace font is
    what makes a number like this one rather than a measurement of text this command cannot
    make without shipping the font it would have to measure: a label wider than its column
    is a name cut in half, and a name cut in half is the one thing on the page that cannot
    be read at all.
    """
    return len(label) * _CHAR + _LABEL


def _coordinate(value: float) -> str:
    """One coordinate, without the trailing zeroes Python would otherwise print."""
    return f"{value:.1f}".removesuffix(".0")


def _text(value: str) -> str:
    """Text for the body of an element, where only the three markup characters matter.

    The attributes are escaped with the quoting escape as well, because a value inside one
    has to survive the quote that ends it; inside the body of an element a quote is an
    ordinary character, and escaping every apostrophe in a page about identifiers nobody
    reads as markup is noise in a file this project commits.
    """
    return escape(value, quote=False)


# --- the figures beside the picture ------------------------------------------------------------------------


def _timing_read(candidate: CampaignCandidate) -> str:
    """What the clock said about this candidate, in its own four figures (ADR-0025).

    The window and the gaps are the grouping's own renderings of them, imported rather
    than written here: the page prints the same figures the console prints under the
    candidate, and a gap rounded on the page and not in the table would be one published
    figure in two spellings.
    """
    timing = candidate.timing
    return (
        f"{timing.corroborated} of {timing.pieces} pieces of evidence hold their accounts "
        f"inside a {window_of(timing.window_seconds)} window; the closest pair of accounts "
        f"is {gap_of(timing.closest_seconds)} apart"
    )


def _similarity_read(candidate: CampaignCandidate) -> str:
    """What the stored vectors said about this candidate (ADR-0026)."""
    similarity = candidate.content_similarity
    return (
        f"{similarity.corroborated} of {similarity.pieces} posts have a near-twin by another "
        f"account within {similarity.threshold}; the closest pair is "
        f"{similarity.closest_distance:.2f} apart"
    )


def _category_read(category: CategoryAgreement) -> str:
    """What the Scam Categories said about this candidate, with the tally behind it.

    The tally is carried whole rather than reduced to the class, because the class is what
    a rule produced and the tally is what a reader counts.
    """
    tally = ", ".join(f"{name} {count}" for name, count in category.tally)
    if category.scam_category is None:
        return f"no majority of the placed posts ({category.why_not()}): {tally}"
    return f"{category.scam_category} ({tally})"


def _names(values: Sequence[str]) -> str:
    """A list of names as one string, with the dash for a list of nothing.

    The dash rather than `names_of`'s `none`, because this list is accounts and posts and
    an empty one would be a claim the run forgot to print; the corroboration names go
    through `names_of` instead, where `none` is exactly the claim.
    """
    return ", ".join(values) if values else "—"


def _count(number: int, noun: str) -> str:
    """One count, agreeing with its noun. Every figure here is small, and a reader seeing
    "1 posts" stops to wonder whether the figure is right."""
    return f"{number} {noun if number == 1 else f"{noun}s"}"


# --- the shapes and lines the key names -------------------------------------------------------------------------


_SHAPE_KEY: tuple[tuple[NodeKind, str], ...] = (
    (NodeKind.ACCOUNT, "an account in the candidate"),
    (NodeKind.POST, "a post that account wrote"),
    (NodeKind.REGISTRATION, "a Registrable Domain two accounts reach"),
    (NodeKind.CONTACT_POINT, "a Contact Point two accounts publish"),
)

_RELATION_KEY: tuple[tuple[Relation, str], ...] = (
    (Relation.WROTE, "wrote — the post's own account"),
    (Relation.LINKS, "links — the post's own link, resolved to a registration"),
    (Relation.PUBLISHES, "publishes — the post names the Contact Point"),
    (
        Relation.SHARES,
        "shares — why the accounts are in one candidate at all (ADR-0005), drawn bowed "
        "underneath the other three",
    ),
)


_STYLES = """\
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 2.5rem 1.5rem 4rem;
  background: #f8fafc;
  color: #111827;
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
}
main { max-width: 64rem; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 0 0 .3rem; letter-spacing: -.01em; }
h2 { font-size: 1.05rem; margin: 0 0 .6rem; }
section { margin: 0 0 1.75rem; }
.claim {
  margin: 0 0 1.5rem;
  padding: .65rem .9rem;
  background: #fef3c7;
  border-left: 4px solid #b45309;
  font-weight: 600;
}
code {
  font-family: ui-monospace, "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: .9em;
  background: #eef2f7;
  padding: 0 .25em;
  border-radius: .2em;
  word-break: break-all;
}
dl.facts {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: .3rem 1.25rem;
  margin: 0 0 1.5rem;
  padding: 1rem 1.25rem;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: .5rem;
}
dl.facts dt { font-weight: 600; color: #334155; }
dl.facts dd {
  margin: 0;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: .9rem;
  word-break: break-word;
}
figure {
  margin: 0 0 1.75rem;
  padding: 1.25rem;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: .5rem;
  overflow-x: auto;
}
.graph { display: block; max-width: 100%; height: auto; }
.node-label {
  font: 13px ui-monospace, "Cascadia Mono", Consolas, monospace;
  fill: #0f172a;
  text-anchor: middle;
}
.edge { fill: none; }
.edge-wrote { stroke: #64748b; stroke-width: 1.5; stroke-dasharray: 1 3; stroke-linecap: round; }
.edge-links { stroke: #b45309; stroke-width: 2; stroke-dasharray: 7 4; }
.edge-publishes { stroke: #6d28d9; stroke-width: 2; stroke-dasharray: 2 4; }
.edge-shares { stroke: #be123c; stroke-width: 3.5; }
.shape-account { fill: #0f766e; }
.shape-post { fill: #1d4ed8; }
.shape-registration { fill: #b45309; }
.shape-contact-point { fill: #6d28d9; }
figcaption { margin-top: 1rem; }
.key-list { display: flex; flex-wrap: wrap; gap: .5rem 1.75rem; margin: 0; padding: 0; list-style: none; font-size: .92rem; }
.key-list li { display: flex; align-items: center; gap: .5rem; }
.key-shape, .key-edge { flex: none; }
table { width: 100%; border-collapse: collapse; background: #fff; font-size: .93rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
th {
  background: #f1f5f9;
  color: #334155;
  font-size: .78rem;
  letter-spacing: .04em;
  text-transform: uppercase;
}
td:first-child { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; word-break: break-word; }
p.verdict { margin: 0 0 .75rem; font-weight: 600; }
.verdict-holds { color: #166534; font-weight: 600; }
.verdict-against { color: #9a3412; font-weight: 600; }
footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #cbd5e1; font-size: .93rem; color: #334155; }
footer p { margin: .6rem 0; }
"""

# --- the geometry -------------------------------------------------------------------------------------------

# The margin around the picture, and the gap between its columns: room for a label under a
# node, and room between two columns whose labels both need their full width.
_MARGIN = 40
_GUTTER = 90
_ROW = 88

# The room below a node for its own label, and the widest box any shape is drawn in, which
# is also the narrowest a column can be.
_LABEL = 16
_SHAPE = 48

# The advance of one character of the label font, at the size the stylesheet sets it. A
# monospace font is what makes a number like this one rather than a measurement of text
# this command cannot make without shipping the font it would have to measure.
_CHAR = 7.8

# How far the first account-to-shared line is bowed below the picture, and how much deeper
# each account's line is drawn than the account above it. Enough that a bowed line is never
# mistaken for a straight one crossing the columns.
_ARC = 52
_ARC_STEP = 20
