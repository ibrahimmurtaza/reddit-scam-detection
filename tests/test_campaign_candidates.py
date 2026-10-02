"""Campaign Candidates: the accounts that share a registrable domain, and nothing else.

This is the spine of the project — the first command that cuts a complete path from
the Corpus to output. ADR-0005 puts two accounts in the same Campaign Candidate only
if they share a registrable domain or a Contact Point, and a Contact Point is not
extracted yet, so everything here rests on shared registration and on nothing else.

Seam under test: the `campaign-candidates` command, observed through the file it
writes and the table it prints. Nothing here inspects the code that wrote them.

The grouping rule is deliberately crude, and the corpus the project ships says so:
known-shared infrastructure is not filtered out yet, so a link shortener or a paste
site groups every account that touches it. That is ticket #9, and the output has to
name it rather than let a reader believe the groupings are clean.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_CORPUS_PATH, DEFAULT_SEED, main
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CANDIDATES = REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl"
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_POST_DOMAINS = REPO_ROOT / "data" / "domains" / "post-domains.jsonl"

Row = Mapping[str, object]


def run(
    directory: Path,
    corpus: Path = DEFAULT_CORPUS_PATH,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> tuple[Path, str]:
    candidates_path = directory / "campaign-candidates.jsonl"
    exit_code = main(
        [
            "campaign-candidates",
            "--corpus",
            str(corpus),
            "--candidates",
            str(candidates_path),
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out if capsys is not None else ""
    return candidates_path, printed


def rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def texts(row: Row, field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [str(entry) for entry in value]


def records(row: Row, field: str) -> list[Row]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [entry for entry in value if isinstance(entry, dict)]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} is not text: {value!r}"
    return value


def by_accounts(path: Path) -> dict[tuple[str, ...], Row]:
    return {tuple(texts(row, "accounts")): row for row in rows(path)}


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


def test_accounts_sharing_a_registrable_domain_become_one_candidate(tmp_path: Path) -> None:
    """The hard requirement, read as a question with two answers.

    Two accounts reaching one registration through two different hostnames are one
    candidate — `mirror.vantage-ledger.example` and `vantage-ledger.example` are one
    registration, and a command that stopped at the hostname would report two
    components of one each, which ADR-0005 does not count. A third account on a
    domain of its own is a component of one and is not reported at all.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_9001", "syn_alpha_0001", "https://vantage-ledger.example/entry"),
            post(
                "syn_p_9002",
                "syn_beta_0002",
                "https://mirror.vantage-ledger.example/month-log",
            ),
            post("syn_p_9003", "syn_gamma_0003", "https://plainsaw.example/bench"),
        ),
    )

    candidates_path, _ = run(tmp_path, corpus)
    written = by_accounts(candidates_path)

    assert list(written) == [("syn_alpha_0001", "syn_beta_0002")]
    assert texts(written[("syn_alpha_0001", "syn_beta_0002")], "posts") == [
        "syn_p_9001",
        "syn_p_9002",
    ]


def test_near_identical_text_on_no_shared_domain_groups_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal ADR-0005 is written for, in the exact shape the Corpus plants.

    Two accounts paste the same advert word for word in the same minute and each links
    its own site, and two more paste it with no links at all. Text and timing point the
    same way and point it loudly; nothing says the accounts share anything. Every one
    of them is a component of one, so nothing is reported — and a command that got
    this wrong would merge four unrelated operators on the strength of a template,
    which is the false grouping the recovery figure is measured against. The shipped
    Corpus's own copy of this case is joined anyway, by the shortener all four use,
    which is what makes it the right case to filter next.
    """
    advert = (
        "Remote annotation work, 26 an hour, six hours a day, kit posted out, starts "
        "Monday. No experience needed, they train you over the first two days."
    )
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9011",
                "syn_draycott_5583",
                "https://anvil-labels.example/apply",
                body=advert,
            ),
            post(
                "syn_p_9012",
                "syn_underhill_7420",
                "https://underhilltalent.example/apply",
                body=advert,
            ),
            post("syn_p_9013", "syn_mossgavel_2218", body=advert),
            post("syn_p_9014", "syn_pennyfarthing_8834", body=advert),
        ),
    )

    candidates_path, printed = run(tmp_path, corpus, capsys)

    assert rows(candidates_path) == []
    assert "cc-01" not in printed


def chained_corpus(tmp_path: Path) -> Path:
    """Three accounts, one of them bridging two registrations the others do not share.

    The first and the second share `vantage-ledger.example`; the second and the third
    share `signal-harbor.example`. Nothing links the first and the third, so the only
    way to see that all three belong together is a union-find rather than a rule that
    pairs accounts off per domain — and a candidate that is a chain is the case worth
    looking at, because both its domains have to be printed for the claim to hold.
    """
    return write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9021",
                "syn_alpha_0001",
                "https://vantage-ledger.example/entry",
                created_at="2026-01-05T11:30:00Z",
            ),
            post(
                "syn_p_9022",
                "syn_beta_0002",
                "https://vantage-ledger.example/month-log",
                created_at="2026-01-05T09:00:00Z",
            ),
            post(
                "syn_p_9023",
                "syn_beta_0002",
                "https://signal-harbor.example/intake",
                created_at="2026-01-05T12:45:00Z",
            ),
            post(
                "syn_p_9024",
                "syn_gamma_0003",
                "https://signal-harbor.example/roles/annotation",
                created_at="2026-01-06T08:15:00Z",
            ),
        ),
    )


def test_a_candidate_carries_the_domains_that_join_it_and_the_accounts_on_each(
    tmp_path: Path,
) -> None:
    """The evidence travels with the claim, so the file alone answers "why these"."""
    candidates_path, _ = run(tmp_path, chained_corpus(tmp_path))
    written = rows(candidates_path)

    assert len(written) == 1
    candidate = written[0]
    assert texts(candidate, "accounts") == [
        "syn_alpha_0001",
        "syn_beta_0002",
        "syn_gamma_0003",
    ]
    assert texts(candidate, "posts") == [
        "syn_p_9021",
        "syn_p_9022",
        "syn_p_9023",
        "syn_p_9024",
    ]

    shared = {text(entry, "domain"): entry for entry in records(candidate, "shared_domains")}
    assert sorted(shared) == ["signal-harbor.example", "vantage-ledger.example"]
    assert texts(shared["vantage-ledger.example"], "accounts") == [
        "syn_alpha_0001",
        "syn_beta_0002",
    ]
    assert texts(shared["vantage-ledger.example"], "posts") == [
        "syn_p_9021",
        "syn_p_9022",
    ]
    assert texts(shared["signal-harbor.example"], "accounts") == [
        "syn_beta_0002",
        "syn_gamma_0003",
    ]
    assert texts(shared["signal-harbor.example"], "posts") == [
        "syn_p_9023",
        "syn_p_9024",
    ]

    # The earliest post in the component, which is what `first_seen` claims to be.
    assert text(candidate, "first_seen") == "2026-01-05T09:00:00Z"


def test_the_printed_table_shows_each_candidate_with_the_evidence_for_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The console is the output the ticket asks for, so it has to stand on its own.

    A reader who has never run the project should be able to scroll back and see, for
    one candidate, which accounts it names, which posts those accounts wrote, and
    which registrations put them together — without opening another file. A summary
    index says where to look; the block under each entry is the claim.
    """
    _, printed = run(tmp_path, chained_corpus(tmp_path), capsys)

    assert "Campaign Candidates" in printed
    assert "cc-01" in printed

    heading = next(line for line in printed.splitlines() if line.startswith("candidate "))
    for column in ("accounts", "posts", "first seen", "shared domains"):
        assert column in heading
    index_line = next(line for line in printed.splitlines() if line.startswith("cc-01 "))
    assert index_line.rstrip().endswith("signal-harbor.example, vantage-ledger.example")

    block = printed.split("cc-01  3 accounts, 4 posts, first seen")[1]
    for account in ("syn_alpha_0001", "syn_beta_0002", "syn_gamma_0003"):
        assert account in block
    for post_id in ("syn_p_9021", "syn_p_9022", "syn_p_9023", "syn_p_9024"):
        assert post_id in block
    assert "2026-01-05T09:00:00Z" in block
    assert "title of syn_p_9021" in block
    assert "vantage-ledger.example" in block
    assert "signal-harbor.example" in block

    # The figures the run read, so a candidate can be traced to the bytes behind it.
    assert "data/public-suffix/public_suffix_list.dat" in printed
    assert "3 accounts in 1 candidate, on 2 registrations" in printed


def test_the_output_proposes_each_grouping_and_leaves_the_judgment_to_a_reviewer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The naming is the claim: a proposal, not a verdict.

    A reader who scrolls back has to be able to tell that the accounts listed are the
    system's hypothesis and not a finding, and that deciding whether they belong
    together is a judgment about conduct rather than something this output settled.
    """
    _, printed = run(tmp_path, chained_corpus(tmp_path), capsys=capsys)

    assert "Campaign Candidates" in printed
    assert "Every entry is a proposal" in printed
    assert "for a reviewer. This output does not make it." in printed


# --- the Corpus the project actually ships -------------------------------------


def test_the_planted_campaigns_come_out_as_candidates_on_their_own_registration(
    tmp_path: Path,
) -> None:
    """The spine: both Planted Campaigns are recovered from shared registration alone.

    The alpha Planted Campaign reaches its registration through two hostnames and one
    of them is a mirror, so a command that stopped at the hostname would split it in
    two and report a component of one. This is the check that the whole path from
    Corpus to output works.

    The membership is written out rather than read: these account names are in the
    Corpus file any reader can open, and naming what the right answer is *is* the
    claim. What is not done here is opening `truth.jsonl` to find out, which is what
    makes this a test of the pipeline rather than a restatement of the generator. The
    recovery figure the evaluator will publish is ticket #13.
    """
    candidates_path, _ = run(tmp_path)
    written = by_accounts(candidates_path)

    alpha = written[("syn_harborlight_5517", "syn_pinecrest_9032", "syn_quantproof_2841")]
    assert texts(alpha, "posts") == ["syn_p_0001", "syn_p_0002", "syn_p_0003", "syn_p_0004"]
    assert [
        text(entry, "domain") for entry in records(alpha, "shared_domains")
    ] == ["vantage-ledger.example"]

    beta = written[("syn_clearpathwork_3184", "syn_northwindhire_7736")]
    assert texts(beta, "posts") == ["syn_p_0005", "syn_p_0006", "syn_p_0007"]
    assert [text(entry, "domain") for entry in records(beta, "shared_domains")] == [
        "signal-harbor.example"
    ]


def test_an_account_that_reaches_no_registration_is_in_no_candidate(tmp_path: Path) -> None:
    """The floor of the case, read off the committed resolved links rather than a list.

    An account whose links name no registration registers nothing, so nothing can
    join it. The Corpus plants two: one that links nothing, and satire that recites a
    planted pitch word for word and still links nothing. Any candidate holding either
    of them was reached on text, which ADR-0005 forbids.
    """
    unreachable = {
        text(row, "account") for row in rows(COMMITTED_POST_DOMAINS) if not texts(row, "domains")
    }
    candidates_path, _ = run(tmp_path)
    grouped = {account for row in rows(candidates_path) for account in texts(row, "accounts")}

    assert unreachable, "the Corpus has accounts that reach no registration"
    assert not unreachable & grouped


def test_the_three_account_figures_partition_the_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Grouped, alone, and unreachable account for the whole Corpus, with no gaps.

    Decided per account rather than per post, which is the only way the three add up:
    an account with one post that links a registration and one that links nothing
    reaches a registration and is not beyond grouping, so it belongs to exactly one of
    the three lines. Counted per post it would land in two at once and the figures
    would stop describing every account in the Corpus.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_9051", "syn_alpha_0001", "https://vantage-ledger.example/entry"),
            post("syn_p_9052", "syn_beta_0002", "https://vantage-ledger.example/month-log"),
            # Links nothing at all, so no grouping can ever reach it.
            post("syn_p_9053", "syn_tideline_4471"),
            # One post that links a registration of its own, one that links nothing.
            # It reaches a registration, so it is alone rather than unreachable.
            post("syn_p_9054", "syn_halfway_9054", "https://plainsaw.example/bench"),
            post("syn_p_9055", "syn_halfway_9054", "/r/example/comments/abc"),
        ),
    )

    _, printed = run(tmp_path, corpus, capsys=capsys)

    assert "4 accounts" in printed
    assert "grouped       2 accounts in 1 candidate, on 1 registration" in printed
    assert "alone         1 account with a registration nobody else reaches" in printed
    assert "unreachable   1 account with no registration" in printed


def test_a_registration_one_character_from_a_planted_one_reaches_no_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The near-miss pairs, which reach nothing and so never appear at all.

    `vantage-ledgers.example` is a recruitment firm and `signal-harbour.example` is a
    warehouse employer, each one character from a registration a Planted Campaign
    uses. A rule that matched on names rather than on registrations would put a
    bookkeeping job in the same candidate as a signals desk. Here neither reaches a
    candidate, so neither is printed anywhere in the output.
    """
    _, printed = run(tmp_path, capsys=capsys)

    assert "vantage-ledgers.example" not in printed
    assert "signal-harbour.example" not in printed


def test_the_known_shared_infrastructure_groupings_are_visible_and_admitted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The known-wrong output, pinned while it is known to be wrong.

    The link shortener and the link-in-bio page are registrations like any other
    until ticket #9 filters them, so they group every account that touches them and
    two of today's candidates exist for no better reason. That is the ticket's stated
    expectation, and the output has to name the filter it is missing rather than let a
    reader believe the list is clean — so this test fails when the footer stops saying
    so, and ticket #9 replaces it with the assertions that the junk is gone.
    """
    _, printed = run(tmp_path, capsys=capsys)
    written = rows(tmp_path / "campaign-candidates.jsonl")

    shortener = next(
        entry
        for row in written
        for entry in records(row, "shared_domains")
        if text(entry, "domain") == "hopcut.example"
    )
    assert texts(shortener, "accounts") == [
        "syn_draycott_5583",
        "syn_pellworth_3196",
        "syn_underhill_7420",
        "syn_wrenfield_3308",
    ]

    assert "hopcut.example" in printed
    assert "Known-shared infrastructure is not filtered yet" in printed
    assert "ticket #9" in printed


# --- what the run is allowed to read --------------------------------------------


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


def test_the_run_reads_the_corpus_the_suffix_list_and_nothing_else(tmp_path: Path) -> None:
    """The boundary ADR-0008 is about, checked on the run rather than on the code.

    The Corpus file is the whole input about accounts, and the truth file sits next to
    it holding the answer to the question this command answers. Reading it would make
    every number the project reports a demonstration rather than a measurement, and no
    test on the source can catch that as directly as watching which files the run
    opens.

    The Public Suffix List is named here too, and it is in scope deliberately: it is
    the published rulebook that decides what a registration is, it names no account and
    no post, and the previous command reads the same file for the same reason.
    """
    opened = _Opened()
    sys.addaudithook(opened)
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_9031", "syn_alpha_0001", "https://vantage-ledger.example/entry"),
            post(
                "syn_p_9032",
                "syn_beta_0002",
                "https://vantage-ledger.example/month-log",
            ),
        ),
    )

    opened.recording = True
    try:
        run(tmp_path, corpus)
    finally:
        opened.recording = False

    data_files = {
        Path(path).name for path in opened.record() if Path(path).suffix in {".jsonl", ".dat"}
    }
    assert data_files == {"corpus.jsonl", "public_suffix_list.dat", "campaign-candidates.jsonl"}
    assert not [path for path in opened.record() if Path(path).name.startswith("truth")]


def test_the_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same argument as every other command here: the numbers come from committed bytes."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the campaign-candidates command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    run(tmp_path)


# --- the shape of the output ----------------------------------------------------


def test_running_twice_writes_byte_identical_files(tmp_path: Path) -> None:
    """Identifiers are assigned from the output order, so the order has to be fixed."""
    first, _ = run(tmp_path / "first")
    second, _ = run(tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()


def test_the_committed_candidates_are_what_the_command_writes(tmp_path: Path) -> None:
    """Held to the command the same way the Corpus is held to its generator."""
    candidates_path, _ = run(tmp_path)

    assert candidates_path.read_bytes() == COMMITTED_CANDIDATES.read_bytes()


def test_every_shared_domain_is_one_the_resolved_links_published(tmp_path: Path) -> None:
    """The grouping and the resolved links cannot disagree, because both are checked.

    The command does not read `post-domains.jsonl` — the grouping is a function of the
    Corpus, re-resolving the links through the same code — so the two files are
    produced by separate runs. Reading the committed file here is what makes "the same
    registration" a checked statement rather than a shared assumption.
    """
    resolved = {
        domain
        for row in rows(COMMITTED_POST_DOMAINS)
        for domain in texts(row, "domains")
    }
    candidates_path, _ = run(tmp_path)

    assert resolved, "the resolved links hold no registrations at all"
    for row in rows(candidates_path):
        for entry in records(row, "shared_domains"):
            assert text(entry, "domain") in resolved
            assert len(texts(entry, "posts")) > 1


def test_the_command_refuses_to_write_the_corpus_away(tmp_path: Path) -> None:
    """The Corpus is the input, and the measurement is against that exact file."""
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (post("syn_p_9041", "syn_alpha_0001", "https://vantage-ledger.example/a"),),
    )

    with pytest.raises(SystemExit):
        main(
            [
                "campaign-candidates",
                "--corpus",
                str(corpus),
                "--candidates",
                str(corpus),
            ]
        )

    assert read_corpus(corpus) == (
        post("syn_p_9041", "syn_alpha_0001", "https://vantage-ledger.example/a"),
    )


def test_the_run_covers_every_seed_of_the_corpus_without_a_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Corpus is a function of its seed, so a different seed must need no edit here.

    A seed with a different Nuisance Structure draws different accounts into the
    shortener's component and can plant a spare single-account domain, so the
    assertions are about what must hold for any of them rather than about the counts
    this seed happens to produce.
    """
    corpus_path = tmp_path / "corpus.jsonl"
    for seed in (DEFAULT_SEED, DEFAULT_SEED + 1):
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
        candidates_path, printed = run(tmp_path / f"out-{seed}", corpus_path, capsys=capsys)

        corpus = read_corpus(corpus_path)
        written = rows(candidates_path)
        accounts = {item.account for item in corpus}

        assert {account for row in written for account in texts(row, "accounts")} <= accounts
        assert all(len(texts(row, "accounts")) > 1 for row in written)
        for row in written:
            # Every domain named under a candidate really does join two of its accounts,
            # and every account it joins is one of the candidate's accounts.
            for entry in records(row, "shared_domains"):
                joined = set(texts(entry, "accounts"))
                assert len(joined) > 1
                assert joined <= set(texts(row, "accounts"))

        # The file and the printed index list the same candidates, in the same order.
        indexed = [line.split()[0] for line in printed.splitlines() if line.startswith("cc-")]
        assert indexed[: len(written)] == [text(row, "candidate_id") for row in written]