"""Link Signals, the Policy Score, and the arithmetic behind both.

Two Signals reach the Policy Score from the Corpus's links: one registration reached
by more than one account, and two registrations one edit apart. Both are here
because they can be checked by looking at the post, which is the whole claim
ADR-0007 makes about the Policy Score.

Seam under test: the `policy-score` command, observed through the file it writes and
the table it prints. Nothing here inspects the code that wrote them.

The auditability claim is the thing under test, so it is asked as a question about
the output rather than about the code: the published weights, applied to the Signals
a reviewer can see in a post's own links, have to produce the number the command
displays. A Signal that could not be recomputed by hand would pass a test asserting
that it fires, and would still break the claim the score is published under.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_CORPUS_PATH, DEFAULT_SEED, main
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_WEIGHTS = REPO_ROOT / "data" / "signals" / "weights.jsonl"
COMMITTED_SCORES = REPO_ROOT / "data" / "signals" / "policy-scores.jsonl"
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_NUISANCE = REPO_ROOT / "data" / "corpus" / "nuisance.jsonl"
COMMITTED_POST_DOMAINS = REPO_ROOT / "data" / "domains" / "post-domains.jsonl"


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

Row = Mapping[str, object]


def run(
    directory: Path,
    corpus: Path = DEFAULT_CORPUS_PATH,
    capsys: pytest.CaptureFixture[str] | None = None,
    weights: Path | None = None,
) -> tuple[Path, str]:
    scores_path = directory / "policy-scores.jsonl"
    argv = ["policy-score", "--corpus", str(corpus), "--scores", str(scores_path)]
    if weights is not None:
        argv += ["--weights", str(weights)]
    exit_code = main(argv)
    assert exit_code == 0
    printed = capsys.readouterr().out if capsys is not None else ""
    return scores_path, printed


def rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} is not text: {value!r}"
    return value


def texts(row: Row, field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [str(entry) for entry in value]


def records(row: Row, field: str) -> list[Row]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [entry for entry in value if isinstance(entry, dict)]


def by_post(path: Path) -> dict[str, Row]:
    return {text(row, "post_id"): row for row in rows(path)}


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


def post(
    post_id: str,
    account: str,
    *links: str,
    body: str = "body",
    created_at: str = "2026-01-05T09:00:00Z",
) -> CorpusItem:
    return CorpusItem(
        post_id=post_id,
        account=account,
        subreddit="test",
        title=f"title of {post_id}",
        body=body,
        created_at=created_at,
        links=links,
    )


def signals(row: Row) -> list[str]:
    """The Signals a post's row carries, by name, in the order the file writes them."""
    return [text(hit, "signal") for hit in records(row, "signals")]


def evidence(row: Row, signal: str) -> Row:
    """The hit of one Signal, checked to be the one asked for."""
    found = [hit for hit in records(row, "signals") if text(hit, "signal") == signal]
    assert len(found) == 1, f"{text(row, 'post_id')} carries {signal} {len(found)} times"
    return found[0]


def items(hit: Row) -> list[Row]:
    """Every piece of evidence one Signal was fired on."""
    return records(hit, "evidence")


def nested(row: Row, field: str) -> Row:
    value = row[field]
    assert isinstance(value, dict), f"{field} is not an object: {value!r}"
    return value


def domain(item: Row) -> str:
    return text(item, "domain")


def confusable_pairs(printed: str) -> set[tuple[str, str]]:
    """Every pair the output names as confusable, read out of the table."""
    section = printed.split("\nconfusable  ")[1].split("\n\n")[0]
    pairs = set()
    for line in section.splitlines()[1:]:
        names = line.split()
        assert len(names) == 2, f"a confusable line names {len(names)} registrations: {line!r}"
        pairs.add((names[0], names[1]))
    return pairs


# --- the domain-frequency Signal -------------------------------------------------


def test_a_registration_two_accounts_reach_carries_the_domain_frequency_signal(
    tmp_path: Path,
) -> None:
    """The Signal, in the shape the Corpus plants: two accounts, three posts, one
    registration, reached through two hostnames.

    Both figures are asserted rather than one of them. `3 posts by 2 accounts` is the
    claim a reviewer recomputes with, and a Signal that fired on the right post while
    counting the wrong posts would satisfy a test that only looked at its name â€” and
    the three posts are reached through `vantage-ledger.example` and a mirror of it,
    so a command that counted links rather than posts would report four here. The
    fourth post is the floor: an account that reaches a registration nobody else
    reaches is one account wide, which is what a link in a post usually is.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_0001", "syn_alpha_0001", "https://vantage-ledger.example/entry"),
            post(
                "syn_p_0002",
                "syn_beta_0002",
                "https://mirror.vantage-ledger.example/month-log",
                created_at="2026-01-05T10:30:00Z",
            ),
            post(
                "syn_p_0003",
                "syn_alpha_0001",
                "https://vantage-ledger.example/entry",
                created_at="2026-01-06T08:15:00Z",
            ),
            post("syn_p_0004", "syn_gamma_0003", "https://plainsaw.example/bench"),
        ),
    )

    scores_path, _ = run(tmp_path, corpus)
    written = by_post(scores_path)

    assert signals(written["syn_p_0001"]) == ["domain_frequency"]
    assert signals(written["syn_p_0002"]) == ["domain_frequency"]
    assert signals(written["syn_p_0003"]) == ["domain_frequency"]
    assert signals(written["syn_p_0004"]) == []

    for post_id in ("syn_p_0001", "syn_p_0002", "syn_p_0003"):
        shown = evidence(written[post_id], "domain_frequency")
        assert domain(items(shown)[0]) == "vantage-ledger.example", post_id
        assert count(items(shown)[0], "posts") == 3, post_id
        assert count(items(shown)[0], "accounts") == 2, post_id


def test_the_frequency_figures_are_the_corpus_reach_and_not_this_posts_share_of_it(
    tmp_path: Path,
) -> None:
    """One post of five, four accounts, one registration.

    The Signal is about the registration rather than about the post, so a post that
    links it inherits the Corpus's figure. Counting a post's own links instead would
    make every post that fired the Signal report `1 post by 1 account`, which is
    evidence of nothing and reads as a Signal that had not fired at all.
    """
    link = "https://vantage-ledger.example/entry"
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_0001", "syn_alpha_0001", link),
            post("syn_p_0002", "syn_alpha_0001", link, created_at="2026-01-05T10:00:00Z"),
            post("syn_p_0003", "syn_beta_0002", link, created_at="2026-01-05T11:00:00Z"),
            post("syn_p_0004", "syn_gamma_0003", link, created_at="2026-01-05T12:00:00Z"),
            post("syn_p_0005", "syn_delta_0004", link, created_at="2026-01-05T13:00:00Z"),
        ),
    )

    scores_path, _ = run(tmp_path, corpus)

    shown = evidence(by_post(scores_path)["syn_p_0001"], "domain_frequency")
    assert count(items(shown)[0], "posts") == 5
    assert count(items(shown)[0], "accounts") == 4


def test_a_registration_the_shared_list_withholds_carries_no_frequency_signal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shortener is four accounts wide, which is more accounts than any campaign
    in this Corpus has, and it is exactly what must not score.

    Four accounts pasting one advert through one shortener is the Nuisance Structure
    the grouping filter was written for (ADR-0009): the reach is real and it means
    nothing. If the Signal read the same list and did not, every one of those posts
    would carry the strongest evidence in the weight set on the strength of a service
    everybody uses. The list is the same published data the grouping reads, and the
    output names it, so a reader can see which registration was kept out.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_0011", "syn_draycott_5583", "https://hopcut.example/q4k"),
            post(
                "syn_p_0012",
                "syn_underhill_7420",
                "https://hopcut.example/q4k",
                created_at="2026-01-05T10:00:00Z",
            ),
            post(
                "syn_p_0013",
                "syn_pellworth_3196",
                "https://hopcut.example/q4k",
                created_at="2026-01-05T11:00:00Z",
            ),
            post(
                "syn_p_0014",
                "syn_mossgavel_2218",
                "https://hopcut.example/q4k",
                created_at="2026-01-05T12:00:00Z",
            ),
        ),
    )

    scores_path, printed = run(tmp_path, corpus, capsys)

    assert all(signals(row) == [] for row in rows(scores_path))
    assert "hopcut.example" in printed
    assert "data/infrastructure/shared-hosts.jsonl" in printed

# --- the lookalike Signal --------------------------------------------------------


def test_the_confusable_pairs_the_corpus_plants_are_all_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pairs, read out of the manifest rather than written out here.

    The Corpus plants two registrations one character apart on purpose, and the point
    of planting them is that eyeballing them is not a test. So the pairs come from
    `nuisance.jsonl`, where the generator recorded what it planted and why, and the
    assertion is that every one of them is found and that the run finds nothing else:
    a Signal that fired on most registrations would satisfy the first half.

    The manifest is not read by the command, only here. That is the boundary the
    grouping tests hold too: expectations may be stated from what was planted, and the
    run may not.
    """
    planted = {
        tuple(sorted(texts(record, "hosts")))
        for record in rows(COMMITTED_NUISANCE)
        if text(record, "kind") == "near_miss_domain_pair"
    }

    scores_path, printed = run(tmp_path, capsys=capsys)

    assert planted, "the Corpus plants no near-miss domain pair, so this would pass vacuously"
    assert confusable_pairs(printed) == planted

    written = by_post(scores_path)
    for mine, theirs in planted:
        named = {
            text(item, "other")
            for row in written.values()
            for hit in records(row, "signals")
            if text(hit, "signal") == "domain_lookalike"
            for item in items(hit)
            if domain(item) == mine
        }
        assert theirs in named, f"no post linking {mine} names {theirs} beside it"


def test_one_edit_apart_fires_and_two_edits_apart_does_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One edit, in each of the four shapes an edit takes, and nothing else.

    A substitution, a transposition, an insertion, and a deletion all count, because a
    deliberate copy is made by doing exactly one of those to a name somebody already
    owns. Two edits apart is a different name, and a rule that kept going would fire on
    every long domain in a real Corpus eventually. The Corpus plants only an insertion
    and a deletion, so the transposition would go untested by it.

    Two registrations one edit apart under different Public Suffixes are not a
    confusable pair: `example` and `co.uk` say different things about what somebody
    registered, so the two are not two spellings of one name.
    """
    pairs = (
        ("plainsaw.example", "plainsae.example"),  # substitution
        ("rimcure.example", "rmicure.example"),  # transposition
        ("single-run.example", "single-runs.example"),  # insertion
        ("signal-harbor.example", "signal-harbr.example"),  # deletion
    )
    too_far = (
        ("signal-harbor.example", "signal-harbxx.example"),  # two substitutions
        ("vantage-ledger.example", "vantage-ledgtr.co.uk"),  # one edit, other suffix
    )
    hosts = [host for pair in pairs + too_far for host in pair]
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        tuple(
            post(f"syn_p_{number:04d}", f"syn_account_{number:04d}", f"https://{host}/")
            for number, host in enumerate(hosts)
        ),
    )

    _, printed = run(tmp_path, corpus, capsys)

    found = confusable_pairs(printed)
    assert found == {tuple(sorted(pair)) for pair in pairs}
    for mine, theirs in too_far:
        assert tuple(sorted((mine, theirs))) not in found


def test_both_registrations_of_a_pair_carry_the_signal_and_neither_is_named_the_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The spelling does not say which of the two imitates which, so both carry it.

    A rule that guessed the direction would be guessing. `signal-harbour.example` is a
    British spelling a real employer arrived at independently and
    `vantage-ledgers.example` is a recruitment firm that took a similar name, and
    nothing in the Corpus says which of the four registrations is the copy. What the
    output can say is the pair, and which of the two is worth reading is a judgment a
    reviewer makes.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_0021", "syn_lanternrow_5548", "https://vantage-ledgers.example/apply"),
            post(
                "syn_p_0022",
                "syn_harborlight_5517",
                "https://vantage-ledger.example/entry",
                created_at="2026-01-05T10:00:00Z",
            ),
        ),
    )

    scores_path, printed = run(tmp_path, corpus, capsys)
    written = by_post(scores_path)

    mine = evidence(written["syn_p_0021"], "domain_lookalike")
    theirs = evidence(written["syn_p_0022"], "domain_lookalike")
    assert domain(items(mine)[0]) == "vantage-ledgers.example"
    assert text(items(mine)[0], "other") == "vantage-ledger.example"
    assert domain(items(theirs)[0]) == "vantage-ledger.example"
    assert text(items(theirs)[0], "other") == "vantage-ledgers.example"

    # The pair is printed once rather than twice, with the direction left open.
    assert confusable_pairs(printed) == {("vantage-ledger.example", "vantage-ledgers.example")}
    assert "no pair is named as the copy" in printed


def test_a_lookalike_to_a_shared_service_still_fires(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The withheld list keeps junk out of the score; it does not make it unreadable.

    A registration one edit from a link shortener everybody uses is what somebody
    imitating that shortener looks like, and that is a case the shortener's presence
    creates rather than one it excuses. So the comparison runs over every registration
    the Corpus's links resolve to, and the published filter applies to
    `domain_frequency` alone.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_0031", "syn_lanternrow_5548", "https://hopcut.example/q4k"),
            post("syn_p_0032", "syn_tideline_4471", "https://hopcuty.example/q4k"),
        ),
    )

    scores_path, printed = run(tmp_path, corpus, capsys)

    assert confusable_pairs(printed) == {("hopcut.example", "hopcuty.example")}
    carried = signals(by_post(scores_path)["syn_p_0031"])
    assert "domain_lookalike" in carried
    assert "domain_frequency" not in carried

# --- the Policy Score, recomputed by hand ---------------------------------------


def count(row: Row, field: str) -> int:
    value = row[field]
    assert isinstance(value, int), f"{field} is not a count: {value!r}"
    return value


def published_weights() -> dict[str, int]:
    """The weight of every published Signal, read out of the committed weight file."""
    return {text(row, "signal"): count(row, "weight") for row in rows(COMMITTED_WEIGHTS)}


# The Corpus the hand-recomputation below is worked out on. Four posts carry a Signal
# and two do not, and between them they use every combination the weight set can
# produce, so the arithmetic has no untested case left in it.
#
#   syn_p_0001  vantage-ledger.example      2 accounts share it, one edit from
#   syn_p_0002  mirror.vantage-ledger         vantage-ledgers.example  ->  30 + 20 = 50
#   syn_p_0003  vantage-ledgers.example     no other account reaches it, but it is
#                                             still one edit away            ->  20
#   syn_p_0004  hopcut.example, hopcuty.example
#                                         a known-shared registration one edit from
#                                         another, both withheld            ->  20
#   syn_p_0005  nothing                    no links at all               ->   0
#   syn_p_0006  plainsaw.example           one account, nothing near it  ->   0
#
# 50 of 50 published points is 100, 20 of 50 is 40, and 0 of 50 is 0.
HAND_CORPUS = (
    post("syn_p_0001", "syn_alpha_0001", "https://vantage-ledger.example/entry"),
    post(
        "syn_p_0002",
        "syn_beta_0002",
        "https://mirror.vantage-ledger.example/month-log",
        created_at="2026-01-05T10:30:00Z",
    ),
    post("syn_p_0003", "syn_gamma_0003", "https://vantage-ledgers.example/apply"),
    post(
        "syn_p_0004",
        "syn_delta_0004",
        "https://hopcut.example/q4k",
        "https://hopcuty.example/q4k",
        created_at="2026-01-05T11:00:00Z",
    ),
    post("syn_p_0005", "syn_epsilon_0005", created_at="2026-01-05T12:00:00Z"),
    post("syn_p_0006", "syn_zeta_0006", "https://plainsaw.example/bench",
         created_at="2026-01-05T13:00:00Z"),
)

# What a reviewer can see in each of those posts' own links, and therefore what each
# Signal is allowed to rest on. Written out rather than derived, because deriving it
# from the run's own output would let a Signal stand on something the run invented.
VISIBLE = {
    "syn_p_0001": {"vantage-ledger.example", "vantage-ledgers.example"},
    "syn_p_0002": {"vantage-ledger.example", "vantage-ledgers.example"},
    "syn_p_0003": {"vantage-ledgers.example", "vantage-ledger.example"},
    "syn_p_0004": {"hopcut.example", "hopcuty.example"},
    "syn_p_0005": set(),
    "syn_p_0006": {"plainsaw.example"},
}

EXPECTED = {
    "syn_p_0001": (["domain_frequency", "domain_lookalike"], 50, 100),
    "syn_p_0002": (["domain_frequency", "domain_lookalike"], 50, 100),
    "syn_p_0003": (["domain_lookalike"], 20, 40),
    "syn_p_0004": (["domain_lookalike"], 20, 40),
    "syn_p_0005": ([], 0, 0),
    "syn_p_0006": ([], 0, 0),
}


def test_the_published_weights_applied_to_what_a_reviewer_can_see_produce_the_score(
    tmp_path: Path,
) -> None:
    """The auditability claim, asked of the output rather than of the code.

    Three things have to hold at once, and none of them holds because the code says
    so. The Signals a post carries have to be the ones its own links justify, so no
    Signal can rest on anything the reviewer cannot look at. The weight beside each
    Signal has to be the weight published in the data file, so the arithmetic a reader
    does is the arithmetic the run did. And the score has to be the sum of those
    published weights as a share of all of them, out of one hundred — worked out by
    hand in the table above and written down there.

    A Signal needing account history would break the first of the three and is
    therefore not addable without failing this, which is what makes the constraint
    structural rather than a promise in a docstring (ADR-0007).
    """
    corpus = write_corpus(tmp_path / "corpus.jsonl", HAND_CORPUS)

    scores_path, _ = run(tmp_path, corpus)
    written = by_post(scores_path)
    weights = published_weights()

    assert weights == {"domain_frequency": 30, "domain_lookalike": 20}
    assert sorted(written) == sorted(EXPECTED)

    for post_id, (expected_signals, expected_points, expected_score) in EXPECTED.items():
        row = written[post_id]
        carried = signals(row)
        assert carried == expected_signals, post_id
        assert count(row, "points") == expected_points, post_id
        assert count(row, "score") == expected_score, post_id
        assert count(row, "published_points") == sum(weights.values()), post_id

        for hit in records(row, "signals"):
            assert count(hit, "weight") == weights[text(hit, "signal")], post_id
            assert items(hit), post_id
            for item in items(hit):
                assert domain(item) in VISIBLE[post_id], (
                    f"{post_id} carries {text(hit, 'signal')} on {domain(item)}, which is "
                    "not in its own links"
                )
                if text(hit, "signal") == "domain_lookalike":
                    assert text(item, "other") in VISIBLE[post_id], post_id


def test_a_signal_is_counted_once_however_many_registrations_fire_it(tmp_path: Path) -> None:
    """Two shared registrations in one post is one Signal, not two of them.

    The score is a subset-sum of the published weights, which is the arithmetic a
    reader can hold in their head and redo without a calculator. Counting a Signal
    once per firing registration would also reward a post for how many links it
    carries, which says something about link stuffing rather than about the
    infrastructure the post points at.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_0001",
                "syn_alpha_0001",
                "https://vantage-ledger.example/entry",
                "https://rivermill-bikes.example/parts",
            ),
            post(
                "syn_p_0002",
                "syn_beta_0002",
                "https://vantage-ledger.example/entry",
                created_at="2026-01-05T10:00:00Z",
            ),
            post(
                "syn_p_0003",
                "syn_gamma_0003",
                "https://rivermill-bikes.example/parts",
                created_at="2026-01-05T11:00:00Z",
            ),
        ),
    )

    scores_path, _ = run(tmp_path, corpus)
    row = by_post(scores_path)["syn_p_0001"]
    frequency = items(evidence(row, "domain_frequency"))

    assert signals(row) == ["domain_frequency"]
    assert count(row, "points") == 30
    assert {domain(item) for item in frequency} == {
        "vantage-ledger.example",
        "rivermill-bikes.example",
    }
    assert count(row, "score") == 60


def test_the_frequency_figures_can_be_counted_off_the_published_resolved_links(
    tmp_path: Path,
) -> None:
    """The evidence is recomputed here from a different file, which is what makes it
    evidence rather than an assertion.

    `domain_frequency` cannot be worked out from one post: it counts how much of the
    Corpus reaches a registration, so the two figures it prints are only auditable if a
    reader can go and count them. This counts them again from
    `data/domains/post-domains.jsonl` — the file `rfi post-domains` publishes, produced
    by a different command over the same Corpus — and requires the two counts to agree.

    That is the honest form of the auditability claim for this Signal. It is not "read
    the post and you have the score": it is "read the post, read the registration table
    beside the Signal, and find the posts and accounts that put it there". The withheld
    registrations are excluded here as they are there, which is what the withheld column
    of the printed table is for.
    """
    resolved = rows(COMMITTED_POST_DOMAINS)
    posts: dict[str, set[str]] = {}
    accounts: dict[str, set[str]] = {}
    for row in resolved:
        for registration in texts(row, "domains"):
            posts.setdefault(registration, set()).add(text(row, "post_id"))
            accounts.setdefault(registration, set()).add(text(row, "account"))

    scores_path, _ = run(tmp_path)
    written = rows(scores_path)
    checked = 0

    for row in written:
        if "domain_frequency" not in signals(row):
            continue
        for item in items(evidence(row, "domain_frequency")):
            registration = domain(item)
            assert count(item, "posts") == len(posts[registration]), registration
            assert count(item, "accounts") == len(accounts[registration]), registration
            checked += 1

    assert checked >= 9, f"only {checked} figures were checked off the resolved links"


def test_the_console_prints_the_arithmetic_beside_the_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table a reader scrolls back to has to carry the division, not just the
    answer.

    Four lines for one post: which Signals, what each is published at, what each one
    rests on, and the total against the published total. Without the last line the
    displayed score is a number with nothing to recompute it from, which is the one
    thing the published-weights claim needs it not to be.
    """
    corpus = write_corpus(tmp_path / "corpus.jsonl", HAND_CORPUS)

    _, printed = run(tmp_path, corpus, capsys)

    block = printed.split("syn_p_0001  syn_alpha_0001  100/100")[1].split("\n\n")[0]
    assert "domain_frequency  30  vantage-ledger.example: 2 posts by 2 accounts" in block
    assert (
        "domain_lookalike  20  vantage-ledger.example: one edit from "
        "vantage-ledgers.example" in block
    )
    assert "total             50  of 50 published points, 2 of 2 Signals" in block

    # The weights are quoted from the file, with the reason each one is that number.
    assert "data/signals/weights.jsonl" in printed
    weights = printed.split("\nweights  ")[1].split("\n\n")[0]
    for signal, weight in published_weights().items():
        assert f"{signal}" in weights and f"{weight}" in weights
    assert "A rules engine, and nothing else" in printed


def test_every_post_in_the_corpus_scores_the_published_weights_of_the_signals_it_carries(
    tmp_path: Path,
) -> None:
    """The same claim over the Corpus the project ships, every post of it.

    The expected score is worked out from the committed weight file — data somebody
    wrote down on purpose — rather than from the run's own figures, so a change to
    the normalisation or to a weight is a change this test sees. The division is exact
    for this weight set, which is asserted rather than assumed, so nothing here depends
    on how the run rounds.
    """
    scores_path, _ = run(tmp_path)

    weights = published_weights()
    published = sum(weights.values())
    written = rows(scores_path)
    corpus = rows(COMMITTED_CORPUS)

    assert len(written) == len(corpus)
    for row in written:
        earned = sum(weights[signal] for signal in set(signals(row)))
        assert earned == count(row, "points"), text(row, "post_id")
        assert 100 * earned % published == 0, (
            f"{text(row, 'post_id')} does not divide evenly, so the score would depend on "
            "how the run rounds"
        )
        assert count(row, "score") == 100 * earned // published, text(row, "post_id")
        assert count(row, "published_points") == published, text(row, "post_id")
# --- the weights are data --------------------------------------------------------


def write_weights(path: Path, entries: tuple[Row, ...]) -> Path:
    """The published weight set, written by hand.

    Written here rather than produced by anything in the project, because the whole
    point of these tests is that the file decides the score: a weight set the module
    itself wrote could not be used to argue that the module listens to it.
    """
    path.write_text(
        "".join(
            json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for entry in entries
        ),
        encoding="utf-8",
    )
    return path


def weight_set(**weights: int) -> tuple[Row, ...]:
    """One weight set, with the reasons taken from the committed file so the only
    thing these tests change is the number."""
    reasons = {
        text(row, "signal"): text(row, "rationale") for row in rows(COMMITTED_WEIGHTS)
    }
    return tuple(
        {"signal": signal, "weight": weight, "rationale": reasons[signal]}
        for signal, weight in sorted(weights.items())
    )


def test_a_weight_is_data_so_editing_the_file_moves_the_score_and_no_code_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The claim that the weights live outside the scoring code, shown by changing them.

    Two runs over one Corpus and one weight file edited between them: `domain_frequency`
    at 30 becomes 60, and `domain_lookalike` stays at 20. Out of the 80 points now
    published, a post carrying the frequency Signal alone is 75 and a post carrying the
    lookalike alone is 25 - both exact, so nothing here depends on how the run rounds.
    A post carrying both is 100 either way, because carrying every Signal is carrying
    all of the weight there is; that is the normalisation doing its job rather than a
    detail, and it is why adding a Signal later moves the scores rather than pushing
    the existing ones over the top.

    No code changes between the two runs, which is the whole point. If the weights were
    a constant beside the rules, the second run would print the first run's numbers and
    this would be a test of nothing.
    """
    corpus = write_corpus(tmp_path / "corpus.jsonl", HAND_CORPUS)
    heavier = write_weights(
        tmp_path / "heavier.jsonl",
        weight_set(domain_frequency=60, domain_lookalike=20),
    )

    before_path, _ = run(tmp_path / "before", corpus)
    after_path, after_printed = run(tmp_path / "after", corpus, capsys, weights=heavier)

    before = by_post(before_path)
    after = by_post(after_path)
    assert published_weights() == {"domain_frequency": 30, "domain_lookalike": 20}

    # syn_p_0003 carries only the lookalike Signal, so only that weight is in play.
    assert count(before["syn_p_0003"], "score") == 40
    assert count(after["syn_p_0003"], "score") == 25
    assert count(after["syn_p_0003"], "published_points") == 80

    # syn_p_0006 carries nothing at all, so it is 0 whatever the weights say.
    assert count(after["syn_p_0006"], "score") == 0

    # And the table quotes the file's numbers rather than a constant.
    assert "domain_frequency  60" in after_printed
    assert "2 Signals, 80 points published" in after_printed


def test_each_published_weight_carries_its_own_reason_and_the_output_quotes_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A weight with no reason is an assertion, and the same reason on every row
    describes the build rather than the number.

    The reasons are checked for being distinct rather than merely present, because a
    rationale that is the same sentence on every row is one the generator wrote and no
    reader can disagree with one Signal without disagreeing with all of them. The
    table prints the file's own words, so the reason a weight is what it is sits next
    to the scores it produced.
    """
    published = rows(COMMITTED_WEIGHTS)
    rationales = [text(row, "rationale") for row in published]

    _, printed = run(tmp_path, capsys=capsys)

    assert published, "no weight is published as data"
    assert len(set(rationales)) == len(rationales), (
        f"every row carries the same rationale: {rationales[0]!r}"
    )
    for row, rationale in zip(published, rationales, strict=True):
        assert rationale.strip(), text(row, "signal")
        assert text(row, "signal") in printed
        assert rationale in printed, text(row, "signal")
# --- the weight set has to account for every Signal --------------------------------


def score_with_weights(directory: Path, weights: Path) -> str:
    """Run the command with a weight set somebody wrote by hand, and return whatever
    it said on the way out - including a refusal, which is what most of these want."""
    scores_path = directory / "policy-scores.jsonl"
    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "policy-score",
                "--weights",
                str(weights),
                "--scores",
                str(scores_path),
            ]
        )
    assert not scores_path.exists(), "the run refused and wrote a score file anyway"
    return str(refusal.value)


def test_a_signal_with_no_published_weight_stops_the_run(tmp_path: Path) -> None:
    """Adding an evaluator is not enough: a Signal has to be published as well.

    A Signal nobody has priced is the failure the published-weights claim is made of.
    Scored at zero it appears in the breakdown and no arithmetic adds up to it; dropped,
    the run prints a total that is a subset-sum of a set the reader cannot see. So the
    refusal names both sides - what the file publishes and what the code computes - and
    a reader is told which Signal to go and publish.
    """
    published = write_weights(
        tmp_path / "one-signal.jsonl", weight_set(domain_frequency=30)
    )

    said = score_with_weights(tmp_path, published)

    assert "publishes ['domain_frequency']" in said
    assert "computes ['domain_frequency', 'domain_lookalike']" in said


def test_a_published_weight_naming_no_signal_this_project_computes_stops_the_run(
    tmp_path: Path,
) -> None:
    """The other direction. A row for a Signal that does not exist is a weight nobody
    can check, and it would move the denominator and so every score in the run."""
    published = write_weights(
        tmp_path / "spare.jsonl",
        (
            *weight_set(domain_frequency=30, domain_lookalike=20),
            {"signal": "account_history", "weight": 15, "rationale": "a reason for "
                                                                       "something that "
                                                                       "does not exist"},
        ),
    )

    said = score_with_weights(tmp_path, published)

    assert "names 'account_history'" in said
    assert "the Signals are domain_frequency, domain_lookalike" in said


@pytest.mark.parametrize(
    ("broken", "complaint"),
    (
        pytest.param(
            {"signal": "domain_lookalike", "weight": 0, "rationale": "a reason"},
            "a whole number above zero",
            id="a weight of zero",
        ),
        pytest.param(
            {"signal": "domain_lookalike", "weight": -5, "rationale": "a reason"},
            "a whole number above zero",
            id="a negative weight",
        ),
        pytest.param(
            {"signal": "domain_lookalike", "weight": 2.5, "rationale": "a reason"},
            "a whole number above zero",
            id="a weight that is not a whole number",
        ),
        pytest.param(
            {"signal": "domain_lookalike", "weight": "20", "rationale": "a reason"},
            "a whole number above zero",
            id="a weight that is not a number",
        ),
        pytest.param(
            {"signal": "domain_lookalike", "weight": 20, "rationale": "   "},
            "a weight with no reason is an assertion",
            id="a rationale that is blank",
        ),
    ),
)
def test_a_weight_that_could_not_be_scored_stops_the_run(
    tmp_path: Path, broken: Row, complaint: str
) -> None:
    """Each of these is invisible in the output, which is why each of them stops it.

    A weight of zero would put a Signal in a breakdown that no arithmetic adds up to, and
    would also move the denominator without moving any score; a blank rationale would
    publish a number nothing argues for. Both are mistakes a reader cannot see by looking
    at a score, and a weight file is meant to be edited by somebody who disagrees with a
    number. The refusal names the file and the row, because a weight file is read one row
    at a time.
    """
    published = write_weights(
        tmp_path / "broken.jsonl",
        (
            {"signal": "domain_frequency", "weight": 30, "rationale": "the frequency reason"},
            broken,
        ),
    )

    said = score_with_weights(tmp_path, published)

    assert "broken.jsonl:2" in said
    assert complaint in said


def test_the_same_signal_published_twice_stops_the_run(tmp_path: Path) -> None:
    """A row repeated passes any check on the *names* in the file and then counts twice.

    One name too many and one missing balance out, so a set comparison is satisfied and
    the run goes on with the weight counted twice: every score drops, the denominator
    inflates, and a post carrying every Signal lands well under 100 while the breakdown
    beside it still names one of each. This is the comparison over the list rather than
    the set of names, asked of the failure a set comparison cannot see.
    """
    published = write_weights(
        tmp_path / "twice.jsonl",
        (
            *weight_set(domain_frequency=30, domain_lookalike=20),
            {"signal": "domain_frequency", "weight": 30, "rationale": "the same reason"},
        ),
    )

    said = score_with_weights(tmp_path, published)

    assert "every Signal must be published exactly once" in said


def test_a_weight_row_the_reader_cannot_check_is_refused_rather_than_skipped(
    tmp_path: Path,
) -> None:
    """A row holding a field this project does not publish is a row nobody can argue
    with, so the run stops on it rather than reading the part it recognises."""
    published = write_weights(
        tmp_path / "extra.jsonl",
        (
            {
                "signal": "domain_frequency",
                "weight": 30,
                "rationale": "a reason",
                "weight_set": "v2",
            },
        ),
    )

    said = score_with_weights(tmp_path, published)

    assert "which is not the weight vocabulary" in said


# --- what the run is allowed to read ---------------------------------------------


def test_renaming_every_account_leaves_every_score_unchanged(tmp_path: Path) -> None:
    """No Signal may turn on who an account is called.

    Every account in the Corpus is renamed to a name that appears nowhere in it, by a
    mapping that is one to one, so the number of accounts reaching each registration
    cannot move. Every score has to come out identical: a Signal reading anything about
    an account rather than about what its links reach - how old it is, how much it has
    posted, whether it posts in bursts - would move here, and none of those is in the
    Corpus file to read in the first place (ADR-0007).
    """
    corpus = write_corpus(tmp_path / "corpus.jsonl", HAND_CORPUS)
    accounts = sorted({item.account for item in read_corpus(corpus)})
    renamed = {account: f"opaque-{number:04d}" for number, account in enumerate(accounts)}
    other = write_corpus(
        tmp_path / "renamed.jsonl",
        tuple(replace(item, account=renamed[item.account]) for item in read_corpus(corpus)),
    )

    before, _ = run(tmp_path / "before", corpus)
    after, _ = run(tmp_path / "after", other)

    assert set(renamed.values()).isdisjoint(accounts), "a renamed account kept its old name"

    def scored(path: Path) -> dict[str, tuple[int, int, list[str]]]:
        return {
            text(row, "post_id"): (count(row, "score"), count(row, "points"), signals(row))
            for row in rows(path)
        }

    assert scored(after) == scored(before)


def test_a_corpus_row_carrying_account_history_is_refused_rather_than_ignored(
    tmp_path: Path,
) -> None:
    """"No Signal in the score path reads account history" is structural, not a promise.

    The Corpus vocabulary is the whole of what the pipeline gets to see, and
    `read_corpus` refuses a row holding a field it does not know rather than dropping
    it — the same refusal ADR-0008 exists for, since silently ignoring a field that had
    appeared would be the first way to stop being able to say what the pipeline read.
    So `karma` cannot be scored on: a Corpus carrying it stops the run rather than
    becoming a value nobody has said the score used.
    """
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "post_id": "syn_p_0001",
                "account": "syn_alpha_0001",
                "subreddit": "test",
                "title": "a title",
                "body": "a body",
                "created_at": "2026-01-05T09:00:00Z",
                "links": ["https://vantage-ledger.example/entry"],
                "karma": 41200,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as refusal:
        main(["policy-score", "--corpus", str(corpus), "--scores", str(tmp_path / "s.jsonl")])

    assert "karma" in str(refusal.value)
    assert "not the Corpus vocabulary" in str(refusal.value)


def test_the_run_reads_four_published_files_and_never_the_truth_file(
    tmp_path: Path,
) -> None:
    """The boundary ADR-0008 is about, checked on the run rather than on the code.

    The Corpus file is the whole input about accounts, and the truth file sits next to
    it holding the answer to the question this command answers. Reading either would
    make every number the project reports a demonstration rather than a measurement.

    The other three are in scope deliberately and all three are named by the output: the
    Public Suffix List decides what a registration is, the shared-infrastructure list is
    the filter the frequency Signal has to apply (ADR-0009), and the weight set is the
    published arithmetic this command exists to apply.
    """
    opened = _Opened()
    sys.addaudithook(opened)
    corpus = write_corpus(tmp_path / "corpus.jsonl", HAND_CORPUS)

    opened.recording = True
    try:
        run(tmp_path, corpus)
    finally:
        opened.recording = False

    data_files = {
        Path(path).name for path in opened.record() if Path(path).suffix in {".jsonl", ".dat"}
    }
    assert data_files == {
        "corpus.jsonl",
        "policy-scores.jsonl",
        "public_suffix_list.dat",
        "shared-hosts.jsonl",
        "weights.jsonl",
    }
    assert not [path for path in opened.record() if Path(path).name.startswith("truth")]
    assert not [path for path in opened.record() if Path(path).name.startswith("nuisance")]


def test_the_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same argument as every other command here: the numbers come from committed bytes."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the policy-score command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    run(tmp_path)


# --- the shape of the output ------------------------------------------------------


def test_running_twice_writes_byte_identical_files(tmp_path: Path) -> None:
    """Nothing in the file is assigned an identifier, so the order has to be fixed by
    something else: the Corpus's own order, then the Signal names, then the
    registrations. A run that sorted by a set would print a different file every time
    the Corpus changed and every other commit would be noise."""
    first, _ = run(tmp_path / "first")
    second, _ = run(tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()


def test_the_committed_scores_are_what_the_command_writes(tmp_path: Path) -> None:
    """Held to the command the same way the Corpus is held to its generator."""
    scores_path, _ = run(tmp_path)

    assert scores_path.read_bytes() == COMMITTED_SCORES.read_bytes()


def test_the_command_refuses_to_write_the_corpus_away(tmp_path: Path) -> None:
    """The Corpus is the input, and the measurement is against that exact file."""
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (post("syn_p_0041", "syn_alpha_0001", "https://vantage-ledger.example/a"),),
    )

    with pytest.raises(SystemExit):
        main(["policy-score", "--corpus", str(corpus), "--scores", str(corpus)])

    assert read_corpus(corpus) == (
        post("syn_p_0041", "syn_alpha_0001", "https://vantage-ledger.example/a"),
    )


def test_the_command_refuses_to_write_over_the_weight_set(tmp_path: Path) -> None:
    """The weight set is this run's input and the published arithmetic every later run
    is compared against, so a collision at that path would leave the scores in the
    repository resting on whatever had been written over the weights."""
    published = write_weights(
        tmp_path / "weights.jsonl", weight_set(domain_frequency=30, domain_lookalike=20)
    )
    before = published.read_bytes()

    with pytest.raises(SystemExit):
        main(
            [
                "policy-score",
                "--weights",
                str(published),
                "--scores",
                str(published),
            ]
        )

    assert published.read_bytes() == before


def test_the_run_covers_every_seed_of_the_corpus_without_a_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Corpus is a function of its seed, so a different seed must need no edit here.

    A seed with a different Nuisance Structure draws different accounts onto the shared
    registrations and plants different confusable pairs, so the Signals this run fires
    move and the ones it does not fire stay unfired. What must hold for every seed is
    that each Signal's evidence names a registration the run actually knows about, that
    every Signal a post carries is one the published weight set prices, and that a post
    carrying no Signal scores nothing at all.
    """
    for seed in (DEFAULT_SEED, DEFAULT_SEED + 1):
        corpus_path = tmp_path / f"corpus-{seed}.jsonl"
        assert main(
            [
                "generate-corpus",
                "--seed",
                str(seed),
                "--corpus",
                str(corpus_path),
                "--truth",
                str(tmp_path / f"truth-{seed}.jsonl"),
                "--nuisance",
                str(tmp_path / f"nuisance-{seed}.jsonl"),
                "--shared-infrastructure",
                str(tmp_path / f"shared-{seed}.jsonl"),
            ]
        ) == 0

        scores_path, printed = run(tmp_path / f"out-{seed}", corpus_path, capsys=capsys)
        written = rows(scores_path)
        known = {
            line.split()[0]
            for line in printed.split("\nregistrations  ")[1].split("\n\n")[0].splitlines()[1:]
            if line.strip()
        }

        assert known, f"seed {seed} resolved no registrations at all"
        assert len(written) == len(read_corpus(corpus_path))
        weights = published_weights()
        for row in written:
            carried = records(row, "signals")
            if not carried:
                assert count(row, "points") == 0 and count(row, "score") == 0, text(
                    row, "post_id"
                )
            for hit in carried:
                assert text(hit, "signal") in weights, f"seed {seed}: {text(hit, 'signal')}"
                assert items(hit), text(row, "post_id")
                for item in items(hit):
                    assert domain(item) in known, f"seed {seed}: {domain(item)}"