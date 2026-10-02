"""Campaign Candidates: the accounts that share a registrable domain, and nothing else.

This is the spine of the project — the first command that cuts a complete path from
the Corpus to output. ADR-0005 puts two accounts in the same Campaign Candidate only
if they share a registrable domain or a Contact Point, and a Contact Point is not
extracted yet, so everything here rests on shared registration and on nothing else.

Seam under test: the `campaign-candidates` command, observed through the file it
writes and the table it prints. Nothing here inspects the code that wrote them.

Known-shared infrastructure is withheld before the grouping rather than after it,
from the published list at `data/infrastructure/shared-hosts.jsonl`: a link shortener,
a paste site, or a link-in-bio page joins nothing, because everyone uses one. The
list is data and the query holds no host of its own, so these tests check the output
against a list the reader can edit — including an empty one, which brings back every
grouping the filter took away.
"""

from __future__ import annotations

import ast
import json
import sys
import urllib.request
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_CORPUS_PATH, DEFAULT_SEED, main
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus

REPO_ROOT = Path(__file__).parent.parent
SOURCE = REPO_ROOT / "src" / "reddit_fraud_intelligence"
COMMITTED_CANDIDATES = REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl"
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
COMMITTED_NUISANCE = REPO_ROOT / "data" / "corpus" / "nuisance.jsonl"
COMMITTED_POST_DOMAINS = REPO_ROOT / "data" / "domains" / "post-domains.jsonl"
COMMITTED_SHARED_HOSTS = REPO_ROOT / "data" / "infrastructure" / "shared-hosts.jsonl"

Row = Mapping[str, object]


def run(
    directory: Path,
    corpus: Path = DEFAULT_CORPUS_PATH,
    capsys: pytest.CaptureFixture[str] | None = None,
    shared: Path | None = None,
) -> tuple[Path, str]:
    candidates_path = directory / "campaign-candidates.jsonl"
    argv = [
        "campaign-candidates",
        "--corpus",
        str(corpus),
        "--candidates",
        str(candidates_path),
    ]
    if shared is not None:
        argv += ["--shared-infrastructure", str(shared)]
    exit_code = main(argv)
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


def domains(path: Path) -> set[str]:
    """Every registration any candidate is joined on."""
    return {text(entry, "domain") for row in rows(path) for entry in records(row, "shared_domains")}


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


def test_the_four_account_figures_partition_the_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Grouped, alone, silenced, and unreachable, account for the whole Corpus, with
    no gaps.

    Decided per account rather than per post, which is the only way the four add up:
    an account with one post that links a registration and one that links nothing
    reaches a registration and is not beyond grouping, so it belongs to exactly one of
    the four lines. Counted per post it would land in two at once and the figures
    would stop describing every account in the Corpus.

    `silenced` is its own line rather than folded into `alone`, because the two are
    different claims: an account with a shortener and a site of its own has a
    registration nobody else reaches, and one with nothing but a shortener reaches
    nothing a candidate could ever be built on. Reading them as one number would let
    the filter claim credit for an account it never touched.
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
            # The shortener and nothing else, so the filter is the only reason this
            # account is in no candidate.
            post("syn_p_9056", "syn_shortonly_9056", "https://hopcut.example/q4k"),
        ),
    )

    _, printed = run(tmp_path, corpus, capsys=capsys)

    assert "5 accounts" in printed
    assert "grouped       2 accounts in 1 candidate, on 1 registration" in printed
    assert "alone         1 account with a registration nobody else reaches" in printed
    assert (
        "silenced      1 account reaching nothing but known-shared infrastructure"
        in printed
    )
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


def write_shared_hosts(path: Path, hosts: tuple[Row, ...]) -> Path:
    """The published list, written the way the generator writes it.

    Written by hand rather than generated because the point of these tests is that the
    list decides the grouping: a list the command produced could not be used to argue
    that the command listens to it.
    """
    path.write_text(
        "".join(
            json.dumps(host, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for host in hosts
        ),
        encoding="utf-8",
    )
    return path


def test_accounts_whose_only_shared_host_is_a_known_shared_service_do_not_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The filter applied to the shape the Corpus plants: three accounts pasting one
    advert, joined by the shortener and by nothing else.

    Their own registrations are one account each, so with the shortener withheld every
    one of them is a component of one and nothing is reported. This is the case
    `test_near_identical_text_on_no_shared_domain_groups_nothing` sets up by hand, with
    the one shared registration left in, so the two together say the filter and not
    the missing links are what silences them.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9081",
                "syn_draycott_5583",
                "https://anvil-labels.example/apply",
                "https://hopcut.example/q4k",
            ),
            post(
                "syn_p_9082",
                "syn_underhill_7420",
                "https://underhilltalent.example/apply",
                "https://hopcut.example/q4k",
            ),
            post(
                "syn_p_9083",
                "syn_pellworth_3196",
                "https://pellworthwork.example/apply",
                "https://hopcut.example/q4k",
            ),
        ),
    )

    candidates_path, printed = run(tmp_path, corpus, capsys)

    assert rows(candidates_path) == []
    assert "cc-01" not in printed
    assert "1 of 4 registrations withheld, removing 1 of 1 components" in printed


def test_a_withheld_registration_cannot_bridge_two_accounts_that_share_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Withheld before the grouping rather than after it, so it cannot be a bridge.

    One account reaches the shortener alone, one reaches the shortener and the
    campaign's registration, and one reaches the campaign's registration. Removing the
    component after the grouping would still leave the bridging account attached to
    the first through the very host that was supposed to be ignored; withholding the
    registration first leaves the two accounts that genuinely share one.

    The bridging account is also why the withheld section claims only what is true of
    every run: it reaches the shortener *and* it is in a candidate, so a heading saying
    these registrations grouped nothing would be false here even though it is true of
    the three the Corpus ships.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_9091", "syn_alpha_0001", "https://hopcut.example/q4k"),
            post(
                "syn_p_9092",
                "syn_beta_0002",
                "https://hopcut.example/q4k",
                "https://vantage-ledger.example/entry",
            ),
            post("syn_p_9093", "syn_gamma_0003", "https://vantage-ledger.example/month-log"),
        ),
    )

    candidates_path, printed = run(tmp_path, corpus, capsys=capsys)

    written = by_accounts(candidates_path)
    assert list(written) == [("syn_beta_0002", "syn_gamma_0003")]
    assert [
        text(entry, "domain")
        for entry in records(written[("syn_beta_0002", "syn_gamma_0003")], "shared_domains")
    ] == ["vantage-ledger.example"]

    # Two accounts reach the shortener and only one of them is silenced by it.
    assert "withheld  1 registration, reached by 2 accounts" in printed
    assert "no candidate is joined on one of them" in printed
    assert "silenced      1 account reaching nothing but known-shared" in printed


def test_a_component_the_filter_splits_counts_as_removed_and_its_parts_as_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The removed figure means "no longer a component", which a split is.

    Two pairs of accounts, one pair on a registration and one on another, and one
    account in the first pair reaching the shortener that the second pair's account
    also reaches. Unfiltered that is a single component of four; withheld, it is two
    candidates and a component that no longer exists. A figure that counted only the
    components that vanished entire would report 0 here, which would read as the filter
    having done nothing while it split a group of four in half — which is exactly the
    case where a reader most needs to be told.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9096",
                "syn_alpha_0001",
                "https://hopcut.example/q4k",
                "https://vantage-ledger.example/entry",
            ),
            post("syn_p_9097", "syn_beta_0002", "https://vantage-ledger.example/month-log"),
            post(
                "syn_p_9098",
                "syn_gamma_0003",
                "https://hopcut.example/q4k",
                "https://plainsaw.example/bench",
            ),
            post("syn_p_9099", "syn_delta_0004", "https://plainsaw.example/notes"),
        ),
    )

    candidates_path, printed = run(tmp_path, corpus, capsys=capsys)

    assert sorted(by_accounts(candidates_path)) == [
        ("syn_alpha_0001", "syn_beta_0002"),
        ("syn_delta_0004", "syn_gamma_0003"),
    ]
    assert "1 of 3 registrations withheld, removing 1 of 1 components" in printed
    assert "grouped       4 accounts in 2 candidates, on 2 registrations" in printed


def test_the_hard_negatives_on_shared_infrastructure_are_not_grouped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The decoys the Corpus plants stop collapsing, which is what the ticket asked for.

    A genuine employer whose only link is a link-in-bio page, satire, and two people
    who lost money all reach services everybody uses, and so do three accounts pasting
    one advert. Before the filter they were two components of six and of four accounts
    with nothing else in common; withheld, every one of them is a component of one and
    none is reported. Read from the manifest and the resolved links rather than
    written out here, because a transcribed list of who uses a shortener is a list that
    goes stale at the next seed.
    """
    published = {text(row, "host") for row in rows(COMMITTED_SHARED_HOSTS)}
    hard_negatives = {
        account
        for record in rows(COMMITTED_NUISANCE)
        if text(record, "kind") == "hard_negative"
        for account in texts(record, "accounts")
    }
    touching = {
        text(row, "account")
        for row in rows(COMMITTED_POST_DOMAINS)
        if published & set(texts(row, "domains"))
    }

    candidates_path, printed = run(tmp_path, capsys=capsys)
    written = rows(candidates_path)
    grouped = {account for row in written for account in texts(row, "accounts")}

    assert hard_negatives & touching, "the Corpus plants no Hard Negative on a shared host"
    assert not touching & grouped, (
        f"{sorted(touching & grouped)} reached nothing but a known-shared service"
    )
    assert not published & {
        text(entry, "domain") for row in written for entry in records(row, "shared_domains")
    }
    # Named as withheld rather than left unexplained in the resolved links.
    assert "withheld" in printed


def test_the_filtered_groupings_are_the_planted_campaigns_and_the_shop_that_is_not(
    tmp_path: Path,
) -> None:
    """What is left, read off the output: two Planted Campaigns and the shop decoy.

    Both Planted Campaigns come out on their own registrations, and the three accounts
    of one shop come out on the shop's domain — which is a correct grouping that must
    never be counted as recovery (ADR-0004). Nothing else groups, so the recovery
    figure the evaluator publishes is measured against a list this small that a reader
    can check by hand.
    """
    candidates_path, _ = run(tmp_path)
    written = by_accounts(candidates_path)

    assert list(written) == [
        ("syn_harborlight_5517", "syn_pinecrest_9032", "syn_quantproof_2841"),
        ("syn_rivermill_4417", "syn_rivermill_6620", "syn_rivermill_9085"),
        ("syn_clearpathwork_3184", "syn_northwindhire_7736"),
    ]


def test_the_grouping_query_holds_no_host_list() -> None:
    """The filter is a file, and the query that reads it names no host at all.

    ADR-0009 and ADR-0011 both require the list to be replaceable by a
    domain-reputation feed, and a literal inside the grouping query is not
    replaceable: adding to it would mean editing the code. The check is on string
    literals rather than on the whole module, because prose in a docstring is allowed
    to say what a Registrable Domain looks like — only a host in the code would be a
    list the query is carrying around itself.

    `tests/test_nuisance_structure.py` holds the wider line ADR-0011 states: only the
    module that publishes the hosts may hold one, anywhere in `src/`. This is the same
    rule narrowed to the query that applies the filter, and asked in the terms the
    ticket states it in, so a reader looking for why the filter is data finds it here
    and finds the module-wide version beside it.
    """
    published = {text(row, "host") for row in rows(COMMITTED_SHARED_HOSTS)}
    assert published, "the published list is empty, so this check would pass vacuously"

    tree = ast.parse((SOURCE / "campaigns.py").read_text(encoding="utf-8"))
    prose = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
    }
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    } - {text for text in prose if text is not None}

    assert not published & literals, f"the grouping query names {sorted(published & literals)}"


def test_adding_a_host_to_the_published_list_changes_the_grouping_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The list decides which registrations are shared; the query only reads it.

    Three runs over one Corpus. The committed list keeps the alpha Planted Campaign,
    because its registration is not on the list. Adding `vantage-ledger.example` to a
    copy of the list — a host nobody has to write any code for — takes that candidate
    away. An empty list takes the filter out of the run altogether and every component
    it removed comes back, which is what makes the difference measurable rather than
    merely asserted.
    """
    kept_path, kept_printed = run(tmp_path / "kept", capsys=capsys)
    added = write_shared_hosts(
        tmp_path / "added.jsonl",
        (
            *rows(COMMITTED_SHARED_HOSTS),
            {
                "added": "2026-10-01",
                "host": "vantage-ledger.example",
                "kind": "paste_site",
                "provenance": "Added by this test, to show the list decides.",
            },
        ),
    )
    withheld_path, withheld_printed = run(
        tmp_path / "withheld", capsys=capsys, shared=added
    )
    empty = write_shared_hosts(tmp_path / "empty.jsonl", ())
    empty_path, empty_printed = run(tmp_path / "empty", capsys=capsys, shared=empty)

    alpha = ("syn_harborlight_5517", "syn_pinecrest_9032", "syn_quantproof_2841")
    assert alpha in by_accounts(kept_path)
    assert "3 of 15 registrations withheld, removing 2 of 5 components" in kept_printed

    assert alpha not in by_accounts(withheld_path)
    assert "4 of 15 registrations withheld, removing 3 of 5 components" in withheld_printed

    assert len(rows(empty_path)) == 5
    assert "0 of 15 registrations withheld, removing 0 of 5 components" in empty_printed
    assert "hopcut.example" in domains(empty_path)


def test_the_output_reports_what_the_filter_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The effect of the filter is a figure in the output, not something to infer.

    Withholding a registration is a decision, so the output says how many of the
    Corpus's registrations were withheld, how many components that took away out of how
    many the run would have had, and which registrations they were — with the kind of
    service each one is and the accounts it reached. A reader who finds a registration
    in the resolved links that no candidate names can then see what happened to it.
    """
    _, printed = run(tmp_path, capsys=capsys)

    assert "filtered      3 of 15 registrations withheld, removing 2 of 5 components" in printed

    withheld = printed.split("withheld  ")[1].split("\n\n")[0]
    heading, *lines = withheld.splitlines()
    assert heading.startswith("3 registrations, reached by 10 accounts")
    for host, kind, accounts in (
        ("biopage.example", "link in bio", "4 accounts"),
        ("hopcut.example", "link shortener", "4 accounts"),
        ("pastevault.example", "paste site", "3 accounts"),
    ):
        line = next(line for line in lines if host in line)
        assert kind in line, line
        assert accounts in line, line


def test_the_shared_list_records_where_each_host_came_from_and_when_it_was_added(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every published host says where it came from and when, and the run quotes it.

    The list is the artefact a domain-reputation feed will replace, so it has to carry
    its own provenance: a host nobody can account for is a filter nobody can audit.
    Provenance that is the same sentence on every row describes the build rather than
    the entry, so each row has to say something about its own host — which is also what
    a reader needs in order to disagree with one entry and not with the list.

    The date the run prints is read back out of the file rather than transcribed, so
    the output cannot drift from the data behind it.
    """
    published = rows(COMMITTED_SHARED_HOSTS)
    added = {text(row, "added") for row in published}
    provenance = [text(row, "provenance") for row in published]
    _, printed = run(tmp_path, capsys=capsys)

    assert published, "no shared host is published as data"
    for row, where in zip(published, provenance, strict=True):
        assert where.strip(), text(row, "host")
        assert date.fromisoformat(text(row, "added")).isoformat() == text(row, "added")
    assert len(set(provenance)) == len(provenance), (
        f"every row carries the same provenance: {provenance[0]!r}"
    )

    assert str(SOURCE.relative_to(REPO_ROOT)) not in printed
    assert f"last updated {max(added)}" in printed
    assert "data/infrastructure/shared-hosts.jsonl" in printed


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


def test_the_run_reads_three_published_files_and_nothing_else(tmp_path: Path) -> None:
    """The boundary ADR-0008 is about, checked on the run rather than on the code.

    The Corpus file is the whole input about accounts, and the truth file and the
    Nuisance Structure manifest sit next to it holding the answer to the question this
    command answers. Reading either would make every number the project reports a
    demonstration rather than a measurement, and no test on the source can catch that
    as directly as watching which files the run opens.

    The other two are in scope deliberately: the Public Suffix List is the published
    rulebook that decides what a registration is, and the shared-infrastructure list is
    the filter the run is required to apply (ADR-0011). Neither names an account or a
    post, and both are named by the output so a reader can trace the run to its bytes.
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
    assert data_files == {
        "campaign-candidates.jsonl",
        "corpus.jsonl",
        "public_suffix_list.dat",
        "shared-hosts.jsonl",
    }
    assert not [path for path in opened.record() if Path(path).name.startswith("truth")]
    assert not [path for path in opened.record() if Path(path).name.startswith("nuisance")]


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


def test_the_command_refuses_to_write_over_the_shared_list(tmp_path: Path) -> None:
    """The list is this run's input and the generator's output, so a collision at that
    path would leave every later run filtering on whatever was written over it."""
    listed = write_shared_hosts(
        tmp_path / "shared-hosts.jsonl",
        (
            {
                "added": "2026-10-01",
                "host": "hopcut.example",
                "kind": "link_shortener",
                "provenance": "Written by this test.",
            },
        ),
    )
    before = listed.read_bytes()

    with pytest.raises(SystemExit):
        main(
            [
                "campaign-candidates",
                "--corpus",
                str(COMMITTED_CORPUS),
                "--shared-infrastructure",
                str(listed),
                "--candidates",
                str(listed),
            ]
        )

    assert listed.read_bytes() == before


def test_the_run_refuses_a_shared_host_that_names_no_registration(tmp_path: Path) -> None:
    """A host nobody registered withholds nothing, and the run must not go on believing
    it filtered something.

    `co.uk` is a Public Suffix, so it is not a registration and no post can be joined on
    it. A list naming it is a list with a mistake in it, and a filter that quietly did
    nothing is worse than one that refuses: the output would claim the junk is gone.
    """
    listed = write_shared_hosts(
        tmp_path / "shared-hosts.jsonl",
        (
            {
                "added": "2026-10-01",
                "host": "co.uk",
                "kind": "paste_site",
                "provenance": "Written by this test, wrongly.",
            },
        ),
    )

    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "campaign-candidates",
                "--corpus",
                str(COMMITTED_CORPUS),
                "--shared-infrastructure",
                str(listed),
                "--candidates",
                str(tmp_path / "campaign-candidates.jsonl"),
            ]
        )

    assert "names no registrable domain" in str(refusal.value)
    assert not (tmp_path / "campaign-candidates.jsonl").exists()


def test_the_run_covers_every_seed_of_the_corpus_without_a_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Corpus is a function of its seed, so a different seed must need no edit here.

    A seed with a different Nuisance Structure draws different accounts into the
    shortener's component, and the filter has to survive that: the accounts it silences
    change, and the ones on either side of the shared host do not. So the assertions
    are about what must hold for any of them rather than about the counts this seed
    happens to produce.
    """
    corpus_path = tmp_path / "corpus.jsonl"
    published = {text(row, "host") for row in rows(COMMITTED_SHARED_HOSTS)}
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
        assert not published & domains(candidates_path)
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