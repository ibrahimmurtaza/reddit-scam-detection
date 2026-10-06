"""Campaign Candidates: the accounts that share a registrable domain or a Contact Point.

This is the spine of the project — the first command that cuts a complete path from
the Corpus to output. ADR-0005 puts two accounts in the same Campaign Candidate only
if they share a registrable domain or a Contact Point, and both edges are read here,
from the Corpus alone.

Seam under test: the `campaign-candidates` command, observed through the file it
writes and the table it prints. Nothing here inspects the code that wrote them.

The Contact Points are read by the same `contacts.extract` that `rfi contact-points`
publishes rather than from the file it writes, so this command still opens three
files and no fourth: a grouping that depended on somebody having run the step before it
would not be a function of the Corpus.

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
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_CORPUS_PATH, DEFAULT_SEED, main
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus

REPO_ROOT = Path(__file__).parent.parent
SOURCE = REPO_ROOT / "src" / "reddit_fraud_intelligence"
COMMITTED_CANDIDATES = REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl"
COMMITTED_CONTACTS = REPO_ROOT / "data" / "contacts" / "post-contacts.jsonl"
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
    window: int | None = None,
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
    if window is not None:
        argv += ["--window-hours", str(window)]
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


def integer(row: Row, field: str) -> int:
    value = row[field]
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"{field} is not a whole number: {value!r}"
    )
    return value


def nested(row: Row, field: str) -> Row:
    value = row[field]
    assert isinstance(value, dict), f"{field} is not a row: {value!r}"
    return value


def by_accounts(path: Path) -> dict[tuple[str, ...], Row]:
    return {tuple(texts(row, "accounts")): row for row in rows(path)}


def domains(path: Path) -> set[str]:
    """Every registration any candidate is joined on."""
    return {text(entry, "domain") for row in rows(path) for entry in records(row, "shared_domains")}


def points(path: Path) -> set[str]:
    """Every Contact Point any candidate is joined on."""
    return {
        text(entry, "value")
        for row in rows(path)
        for entry in records(row, "shared_contact_points")
    }


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
    for column in (
        "accounts",
        "posts",
        "first seen",
        "justified by",
        "shared registrations",
        "timing",
    ):
        assert column in heading
    index_line = next(line for line in printed.splitlines() if line.startswith("cc-01 "))
    assert "signal-harbor.example, vantage-ledger.example" in index_line
    assert index_line.rstrip().endswith("2 of 2 corroborated"), index_line

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
    assert "3 accounts in 1 candidate, on 2 registrations and 0 Contact Points" in printed


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
    assert "reviewer. This output does not make it." in printed


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

    Alpha's candidate holds a fourth account, `syn_greyloch_6612`, and that is not a
    mistake this test is written around: the account is a Hard Negative that published
    the desk's Telegram handle while complaining about it, and the handle is one
    Contact Point reached by both of them. `test_the_complaint_that_names_the_handle_is_grouped_with_the_desk`
    takes that case on its own.
    """
    candidates_path, _ = run(tmp_path)
    written = by_accounts(candidates_path)

    alpha = written[
        (
            "syn_greyloch_6612",
            "syn_harborlight_5517",
            "syn_pinecrest_9032",
            "syn_quantproof_2841",
        )
    ]
    assert texts(alpha, "posts") == [
        "syn_p_0001",
        "syn_p_0002",
        "syn_p_0003",
        "syn_p_0004",
        "syn_p_0021",
    ]
    assert [
        text(entry, "domain") for entry in records(alpha, "shared_domains")
    ] == ["vantage-ledger.example"]

    beta = written[("syn_clearpathwork_3184", "syn_northwindhire_7736")]
    assert texts(beta, "posts") == ["syn_p_0005", "syn_p_0006", "syn_p_0007"]
    assert [text(entry, "domain") for entry in records(beta, "shared_domains")] == [
        "signal-harbor.example"
    ]


def test_the_complaint_that_names_the_handle_is_grouped_with_the_desk(tmp_path: Path) -> None:
    """What the second edge costs on this Corpus, named rather than absorbed.

    `syn_greyloch_6612` lost money to the alpha desk and published the desk's Telegram
    handle while writing the complaint down. ADR-0005 says a shared Contact Point is a
    grouping edge, and the handle is one value reached by both accounts, so the two are
    in one candidate — and the Planted Campaign's membership is no longer held alone, so
    the recovery figure falls rather than rising. Nothing here can tell an operator from
    a customer: no rule reads a post to work out which one published the handle, and a
    rule that tried would be the kind of guess ADR-0005 forbids.

    The claim being made here is therefore narrow and checkable: the two accounts are
    in one candidate, and the shared thing that put them there is named.
    """
    written = by_accounts(run(tmp_path)[0])
    complaint = written[
        (
            "syn_greyloch_6612",
            "syn_harborlight_5517",
            "syn_pinecrest_9032",
            "syn_quantproof_2841",
        )
    ]
    shared = records(complaint, "shared_contact_points")

    assert [text(entry, "value") for entry in shared] == ["syn_vantageledger"]
    assert texts(shared[0], "accounts") == [
        "syn_greyloch_6612",
        "syn_harborlight_5517",
        "syn_pinecrest_9032",
        "syn_quantproof_2841",
    ]
    assert "@Syn_VantageLedger" in texts(shared[0], "spellings")


# --- a Contact Point as a grouping edge -----------------------------------------


def contact_corpus(tmp_path: Path) -> Path:
    """A campaign whose registrations move every post and whose reach does not.

    Two accounts, two registrations one character apart, and one Telegram handle
    published in both posts. Nothing links them except the handle, which is the case
    the second edge of ADR-0005 exists for: a desk that pays for a domain per post and
    keeps one contact the whole way through is invisible to a grouping built on
    registrations alone.
    """
    return write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9061",
                "syn_alpha_0001",
                "https://spring-1.example/entry",
                body="questions to @syn_shared_desk",
            ),
            post(
                "syn_p_9062",
                "syn_beta_0002",
                "https://spring-2.example/entry",
                body="questions to @syn_shared_desk",
            ),
            post("syn_p_9063", "syn_gamma_0003", "https://spring-3.example/bench"),
        ),
    )


def test_two_accounts_sharing_a_contact_point_become_one_candidate(tmp_path: Path) -> None:
    """The hard requirement, with no shared registrable domain anywhere in it.

    Neither of the two accounts reaches the other's registration, and each reaches a
    registration of its own, so every one of the three is a component of one under the
    registration rule. The candidate that comes out names the handle, the two accounts
    that published it, and the two posts they published it in, which is the evidence a
    reader needs to disagree with it.
    """
    candidates_path, _ = run(tmp_path, contact_corpus(tmp_path))
    written = by_accounts(candidates_path)

    assert list(written) == [("syn_alpha_0001", "syn_beta_0002")]
    candidate = written[("syn_alpha_0001", "syn_beta_0002")]
    assert records(candidate, "shared_domains") == []
    shared = records(candidate, "shared_contact_points")
    assert [(text(entry, "kind"), text(entry, "value")) for entry in shared] == [
        ("telegram", "syn_shared_desk")
    ]
    assert texts(shared[0], "accounts") == ["syn_alpha_0001", "syn_beta_0002"]
    assert texts(shared[0], "posts") == ["syn_p_9061", "syn_p_9062"]
    assert texts(shared[0], "spellings") == ["@syn_shared_desk"]


def test_accounts_sharing_neither_a_registration_nor_a_contact_point_are_not_grouped(
    tmp_path: Path,
) -> None:
    """The other half of the same rule, because adding an edge can only ever join more.

    Two accounts paste one advert word for word and each publishes an address of its own
    under a domain of its own. Nothing they have in common is infrastructure, so the
    Contact Point edge does not rescue them either — and the third account, which shares
    nothing at all, is not dragged in with them.
    """
    advert = "Remote annotation work, 26 an hour, kit posted out, starts Monday."
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9064",
                "syn_draycott_5583",
                "https://anvil-labels.example/apply",
                body=f"{advert} Write to desk1@anvil-labels.example.",
            ),
            post(
                "syn_p_9065",
                "syn_underhill_7420",
                "https://underhilltalent.example/apply",
                body=f"{advert} Write to desk2@underhilltalent.example.",
            ),
            post("syn_p_9066", "syn_mossgavel_2218", body=advert),
        ),
    )

    candidates_path, printed = run(tmp_path, corpus)

    assert rows(candidates_path) == []
    assert "cc-01" not in printed


def evidence_corpus(tmp_path: Path) -> Path:
    """Three candidates, one per shape of evidence and one for the combination.

    Each pair shares one thing and nothing else, and each pair shares a thing no other
    pair reaches, so a candidate's label is a fact about that pair rather than about the
    Corpus. The accounts are named so the three pairs sort into a known order, which is
    what lets the index be read line by line.
    """
    return write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9067",
                "syn_both_a_0001",
                "https://shared-both.example/entry",
                body="reach @syn_desk_both",
            ),
            post(
                "syn_p_9068",
                "syn_both_b_0002",
                "https://shared-both.example/entry",
                body="reach @syn_desk_both",
            ),
            post("syn_p_9069", "syn_con_a_0003", body="reach @syn_desk_contact"),
            post("syn_p_9070", "syn_con_b_0004", body="reach @syn_desk_contact"),
            post("syn_p_9071", "syn_dom_a_0005", "https://shared-dom.example/entry"),
            post("syn_p_9072", "syn_dom_b_0006", "https://shared-dom.example/entry"),
        ),
    )


def labels(printed: str) -> dict[str, str]:
    """The evidence label each candidate prints, read off the console.

    Read out of the table rather than out of the file, because the label is derived
    from the two evidence lists and the console is where a reader meets it. A candidate
    holding one edge and a candidate holding both are different claims, and a file
    carrying one column for them would let the file and the table disagree about which
    is which.
    """
    found: dict[str, str] = {}
    for line in printed.splitlines():
        if line.startswith("cc-"):
            found[line.split()[0]] = ""
    for index, line in enumerate(printed.splitlines()):
        if line.strip().startswith("justified by"):
            evidence = line.split("justified by", 1)[1].strip()
            found[_candidate_of(printed, index)] = evidence.split(", the weaker")[0]
    return found


def _candidate_of(printed: str, index: int) -> str:
    """The candidate whose block the line at `index` sits in."""
    for line in reversed(printed.splitlines()[:index]):
        if line.startswith("cc-"):
            return line.split()[0]
    raise AssertionError(f"line {index} is not inside a candidate block")


def labels_of_file(path: Path) -> dict[str, str]:
    """The label each published candidate would print, worked out from the file.

    The label is not a column of the file: it is read out of the two evidence lists, so
    the file and the table cannot hold different answers. This reads it back the way a
    reader would, and it is what holds the published file to the published table.
    """
    found: dict[str, str] = {}
    for row in rows(path):
        has_domain = bool(records(row, "shared_domains"))
        has_point = bool(records(row, "shared_contact_points"))
        key = ",".join(texts(row, "accounts"))
        if has_domain and has_point:
            found[key] = "registrations and Contact Points"
        elif has_point:
            found[key] = "Contact Points"
        else:
            found[key] = "registrations"
    return found


def test_a_candidate_reports_which_evidence_justified_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One line per candidate, three answers, and no fourth answer available.

    ADR-0005 permits exactly two edges, so a candidate is justified by a registration,
    by a Contact Point, or by both, and there is no third thing the run could have used.
    Saying which one applies is what lets a reader weigh the candidates against each
    other: a candidate resting on one handle is not the same claim as one resting on a
    registration somebody registered.
    """
    _, printed = run(tmp_path, evidence_corpus(tmp_path), capsys)

    assert labels(printed) == {
        "cc-01": "registrations and Contact Points",
        "cc-02": "Contact Points",
        "cc-03": "registrations",
    }


def test_a_candidate_joined_on_a_contact_point_alone_is_labelled_as_the_weaker_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The label is not decoration: it says the evidence is the weaker of the two.

    Contact Point extraction has the lower recall of the two readings, and this project
    measured that rather than promising it — `rfi contact-points` publishes the
    shortfall row by row against the Labelled Set. A reader is therefore entitled to
    weigh a candidate justified by a handle differently from one justified by a
    registration, and the output has to say so where the candidate is rather than in a
    footer a reader may not reach.
    """
    _, printed = run(tmp_path, evidence_corpus(tmp_path), capsys=capsys)

    contact_block = printed.split("cc-02  2 accounts")[1].split("\n\n")[0]
    assert "justified by  Contact Points" in contact_block
    assert "weaker of the two readings" in contact_block
    assert "rfi contact-points" in contact_block

    # The candidate resting on both does not carry the warning: the registration is
    # there, and the handle is something extra rather than the only thing.
    both_block = printed.split("cc-01  2 accounts")[1].split("\n\n")[0]
    assert "justified by  registrations and Contact Points" in both_block
    assert "weaker of the two readings" not in both_block


def test_an_account_reaching_only_a_withheld_contact_point_is_silenced_not_unreachable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The four figures partition the Corpus, and a withheld Contact Point does not break it.

    An account whose only reach is an address at a withheld registration reaches
    something, and everything it reaches is known-shared, so it belongs to the line that
    counts what the filter silenced. Counting it as `unreachable` as well would put one
    account in two of the four lines, and the four would then describe more accounts than
    the Corpus holds — which is the whole claim of the partition, and a claim about
    addition that a reader checks by adding.

    The case is worth its own Corpus because the shape it needs is narrow and the shipped
    one does not have it: it takes two accounts sharing one address under a withheld
    registration and no registration at all, and no email in the shipped Corpus sits
    under a host the filter withholds.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9091",
                "syn_pastewriter_9091",
                body="the intake is desk@pastevault.example",
            ),
            post(
                "syn_p_9092",
                "syn_pastewriter_9092",
                body="the intake is desk@pastevault.example",
            ),
        ),
    )

    _, printed = run(tmp_path, corpus, capsys=capsys)

    assert "2 accounts" in printed
    assert (
        "grouped       0 accounts in 0 candidates, on 0 registrations and 0 Contact Points"
        in printed
    )
    assert (
        "silenced      2 accounts reaching nothing but known-shared infrastructure"
        in printed
    )
    assert "unreachable   0 accounts reaching nothing this grouping can use" in printed
    assert "alone         0 accounts reaching something nobody else reaches" in printed


def test_a_contact_point_at_a_withheld_registration_joins_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The filter reaches the second edge, because its promise is about both of them.

    A paste site's address is a paste site's address: every account that complains into
    one publishes it, and joining two of them on it would build a candidate out of the
    one service everybody already uses. The registration is withheld whole, and the
    Contact Point under it is withheld with it — and named, for the same reason the
    withheld registrations are named.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9073",
                "syn_alpha_0001",
                "https://vantage-ledger.example/entry",
                body="complaints to desk@hopcut.example",
            ),
            post(
                "syn_p_9074",
                "syn_beta_0002",
                "https://signal-harbor.example/intake",
                body="complaints to desk@hopcut.example",
            ),
        ),
    )

    candidates_path, printed = run(tmp_path, corpus, capsys=capsys)

    assert rows(candidates_path) == []
    withheld = printed.split("withheld Contact Points")[1].split("\n\n")[0]
    assert "desk@hopcut.example" in withheld
    assert "2 accounts" in withheld


def test_the_desks_the_obfuscated_handles_reach_come_out_as_one_candidate(
    tmp_path: Path,
) -> None:
    """The case this ticket is for, read off the shipped Corpus rather than invented.

    `syn-nuisance-obfuscated-copperlantern` is one desk publishing one Telegram handle
    three ways: written out with a full stop between its characters, written out with
    spaces, and with a digit standing in for a letter. Two of the three accounts reach
    no registration at all and the third reaches one that only it reaches, so before
    this edge nothing joined them and the desk was invisible. The candidate names the
    handle and every spelling the Corpus wrote it in, which is what lets a reader see
    that it is one identifier rather than two that happen to fold together.
    """
    candidates_path, _ = run(tmp_path)
    written = by_accounts(candidates_path)
    desk = written[("syn_halberdmoor_8823", "syn_thrushmoot_5524", "syn_wintermarch_4417")]

    assert records(desk, "shared_domains") == []
    shared = records(desk, "shared_contact_points")
    assert [text(entry, "value") for entry in shared] == ["syn_copperlantern"]
    assert texts(shared[0], "accounts") == [
        "syn_halberdmoor_8823",
        "syn_thrushmoot_5524",
        "syn_wintermarch_4417",
    ]
    assert len(texts(shared[0], "spellings")) > 1, "one spelling is not a fold"

    # And every candidate the shipped Corpus produces says which edge it rests on, so
    # the file alone tells a reader which claims are the strong kind.
    assert labels_of_file(candidates_path) == {
        "syn_greyloch_6612,syn_harborlight_5517,syn_pinecrest_9032,syn_quantproof_2841": (
            "registrations and Contact Points"
        ),
        "syn_halberdmoor_8823,syn_thrushmoot_5524,syn_wintermarch_4417": "Contact Points",
        "syn_rivermill_4417,syn_rivermill_6620,syn_rivermill_9085": "registrations",
        "syn_clearpathwork_3184,syn_northwindhire_7736": "registrations and Contact Points",
    }


def test_the_shipped_candidates_are_joinable_on_a_contact_point_or_a_registration(
    tmp_path: Path,
) -> None:
    """Nothing is left over: every candidate names the edge it was joined on.

    Checked against the two committed files the run does not read, so "the same
    Contact Point" and "the same registration" are checked statements rather than a
    shared assumption about what the extractor found.
    """
    published = {
        text(entry, "value")
        for row in rows(COMMITTED_CONTACTS)
        for entry in records(row, "matches")
    }
    candidates_path, _ = run(tmp_path)

    assert published
    for value in points(candidates_path):
        assert value in published


def test_an_account_that_reaches_no_registration_and_no_contact_point_is_in_no_candidate(
    tmp_path: Path,
) -> None:
    """The floor of the case, read off the committed resolved links rather than a list.

    An account whose posts name neither a registration nor a Contact Point publishes
    nothing that could join it. The Corpus plants two: one that links nothing, and
    satire that recites a planted pitch word for word and still publishes nothing. Any
    candidate holding either of them was reached on text, which ADR-0005 forbids.

    Both halves of the floor are named, because one edge on its own is not the floor
    any more: two of the accounts the Corpus leaves with no registration publish a
    Contact Point, and one desk publishes nothing at all. Decided per account rather
    than per post, because an account that publishes a handle in one post and nothing in
    another has published a handle.
    """
    published_by_post = {
        text(row, "post_id"): texts(row, "contact_points")
        for row in rows(COMMITTED_CONTACTS)
    }
    reaching: set[str] = set()
    for row in rows(COMMITTED_POST_DOMAINS):
        if texts(row, "domains") or published_by_post[text(row, "post_id")]:
            reaching.add(text(row, "account"))
    unreachable = {text(row, "account") for row in rows(COMMITTED_POST_DOMAINS)} - reaching
    candidates_path, _ = run(tmp_path)
    grouped = {account for row in rows(candidates_path) for account in texts(row, "accounts")}

    assert unreachable, "the Corpus has accounts that reach nothing at all"
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
    different claims: an account with a shortener and a site of its own has something
    nobody else reaches, and one with nothing but a shortener reaches nothing a
    candidate could ever be built on. Reading them as one number would let the filter
    claim credit for an account it never touched.

    The two handles here reach nothing but each other, so both accounts are grouped
    without either of them ever naming a registration. That is the case that would
    break a partition still counted over registrations: an account can be grouped by
    the Contact Point edge and still reach nothing a domain-based count can see.
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
            # One handle between them and no registration at all, so both are grouped
            # by the second edge.
            post("syn_p_9057", "syn_handle_9057", body="reach @syn_shared_desk"),
            post("syn_p_9058", "syn_handle_9058", body="reach @syn_shared_desk"),
        ),
    )

    _, printed = run(tmp_path, corpus, capsys=capsys)

    assert "7 accounts" in printed
    assert (
        "grouped       4 accounts in 2 candidates, on 1 registration and 1 Contact Point"
        in printed
    )
    assert "alone         1 account reaching something nobody else reaches" in printed
    assert (
        "silenced      1 account reaching nothing but known-shared infrastructure"
        in printed
    )
    assert "unreachable   1 account reaching nothing this grouping can use" in printed


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

    One of them is now grouped, and it is grouped by no shared host: `syn_greyloch_6612`
    published a Planted Campaign's Telegram handle while complaining about the campaign,
    so it is in a candidate on the second edge rather than on the filter's. The filter's
    own claim is the one under test, so the accounts reaching a shared host *and*
    publishing nothing anybody else publishes are the ones checked.
    """
    published = {text(row, "host") for row in rows(COMMITTED_SHARED_HOSTS)}
    hard_negatives = {
        account
        for record in rows(COMMITTED_NUISANCE)
        if text(record, "kind") == "hard_negative"
        for account in texts(record, "accounts")
    }
    publishing = {
        text(row, "account")
        for row in rows(COMMITTED_CONTACTS)
        if texts(row, "contact_points")
    }
    silent = {
        account
        for record in rows(COMMITTED_NUISANCE)
        if text(record, "kind") == "hard_negative"
        for account in texts(record, "accounts")
        if account not in publishing
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
    assert not touching & silent & grouped, (
        f"{sorted(touching & silent & grouped)} reached nothing but a known-shared service"
    )
    assert not published & {
        text(entry, "domain") for row in written for entry in records(row, "shared_domains")
    }
    # Named as withheld rather than left unexplained in the resolved links.
    assert "withheld" in printed


def test_the_filtered_groupings_are_the_two_planted_campaigns_the_shop_and_the_obfuscated_desk(
    tmp_path: Path,
) -> None:
    """What is left, read off the output: two Planted Campaigns and two desks that are not.

    Both Planted Campaigns come out on their own registrations, the three accounts of one
    shop come out on the shop's domain — which is a correct grouping that must never be
    counted as recovery (ADR-0004) — and the desk behind the obfuscated Telegram handles
    comes out on the handle alone. Nothing else groups, so the recovery figure the
    evaluator publishes is measured against a list this small that a reader can check by
    hand.

    The obfuscated desk comes last because it is the one candidate the 24-hour window
    corroborates nothing under: seven posts over three weeks, and one handle held by three
    accounts throughout. The identifier says where a candidate sits in this output and
    nothing else, so moving it to the end of the table moves the identifier with it.
    """
    candidates_path, _ = run(tmp_path)
    written = by_accounts(candidates_path)

    assert list(written) == [
        (
            "syn_greyloch_6612",
            "syn_harborlight_5517",
            "syn_pinecrest_9032",
            "syn_quantproof_2841",
        ),
        ("syn_rivermill_4417", "syn_rivermill_6620", "syn_rivermill_9085"),
        ("syn_clearpathwork_3184", "syn_northwindhire_7736"),
        ("syn_halberdmoor_8823", "syn_thrushmoot_5524", "syn_wintermarch_4417"),
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

    Three runs over one Corpus. The committed list keeps the shop, because its domain
    is not on the list. Adding `rivermill-bikes.example` to a copy of the list — a host
    nobody has to write any code for — takes that candidate away, because it is the one
    candidate in this Corpus resting on a registration alone. An empty list takes the
    filter out of the run altogether and every component it removed comes back, which is
    what makes the difference measurable rather than merely asserted.
    """
    kept_path, kept_printed = run(tmp_path / "kept", capsys=capsys)
    added = write_shared_hosts(
        tmp_path / "added.jsonl",
        (
            *rows(COMMITTED_SHARED_HOSTS),
            {
                "added": "2026-10-01",
                "host": "rivermill-bikes.example",
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

    shop = ("syn_rivermill_4417", "syn_rivermill_6620", "syn_rivermill_9085")
    assert shop in by_accounts(kept_path)
    assert (
        "3 of 16 registrations and 0 of 6 Contact Points withheld, removing 2 of 5 components"
        in kept_printed
    )

    assert shop not in by_accounts(withheld_path)
    assert (
        "4 of 16 registrations and 0 of 6 Contact Points withheld, removing 3 of 5 components"
        in withheld_printed
    )

    assert len(rows(empty_path)) == 5
    assert (
        "0 of 16 registrations and 0 of 6 Contact Points withheld, removing 0 of 5 components"
        in empty_printed
    )
    assert "hopcut.example" in domains(empty_path)


def test_withholding_a_registration_leaves_a_candidate_on_its_contact_point_alone(
    tmp_path: Path,
) -> None:
    """What the second edge is worth, read as the difference it makes under the filter.

    Putting a Planted Campaign's own registration on the known-shared list takes its
    registration away, and a run built on registrations alone loses the campaign with
    it. The campaign here survives, on the one handle its three accounts all publish —
    which is also why the evidence label matters: the candidate that comes out is
    justified by the weaker of the two readings, and the output says so rather than
    leaving the campaign's recovery to look as strong as it was.
    """
    membership = ("syn_harborlight_5517", "syn_pinecrest_9032", "syn_quantproof_2841")
    candidate = tuple(
    sorted((membership[0], "syn_greyloch_6612", *membership[1:]))
)
    added = write_shared_hosts(
        tmp_path / "added.jsonl",
        (
            *rows(COMMITTED_SHARED_HOSTS),
            {
                "added": "2026-10-01",
                "host": "vantage-ledger.example",
                "kind": "paste_site",
                "provenance": "Added by this test, to show the second edge carries it.",
            },
        ),
    )

    candidates_path, _ = run(tmp_path, corpus=DEFAULT_CORPUS_PATH, shared=added)
    written = by_accounts(candidates_path)

    assert candidate in written
    assert records(written[candidate], "shared_domains") == []
    assert [text(entry, "value") for entry in records(written[candidate], "shared_contact_points")] == [
        "syn_vantageledger"
    ]
    assert labels_of_file(candidates_path)[",".join(candidate)] == "Contact Points"


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

    assert "filtered      3 of 16 registrations and 0 of 6 Contact Points withheld, removing 2 of 5 components" in printed

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


# --- temporal proximity as corroboration ----------------------------------------

# 24 hours in seconds, written out because every figure below is checked against it.
DAY = 86_400


def timing_corpus(tmp_path: Path) -> Path:
    """Two candidates whose accounts posted half an hour apart and 135 days apart.

    The times are written out rather than derived, so every gap in the expected output is
    a subtraction a reader can do: 09:30 to 10:00 is thirty minutes, and 5 January to 20
    May is 135 days. One account of the first candidate posts twice, so the gap between
    two accounts and the span across them are two different numbers — 30 minutes and four
    and a half hours — and a run printing one of them for both would be caught here.
    """
    return write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9101",
                "syn_alpha_0001",
                "https://vantage-ledger.example/entry",
                created_at="2026-05-30T09:30:00Z",
            ),
            post(
                "syn_p_9102",
                "syn_alpha_0001",
                "https://vantage-ledger.example/month-log",
                created_at="2026-05-30T14:00:00Z",
            ),
            post(
                "syn_p_9103",
                "syn_beta_0002",
                "https://vantage-ledger.example/intake",
                created_at="2026-05-30T10:00:00Z",
            ),
            post(
                "syn_p_9104",
                "syn_gamma_0003",
                "https://signal-harbor.example/entry",
                created_at="2026-01-05T09:00:00Z",
            ),
            post(
                "syn_p_9105",
                "syn_delta_0004",
                "https://signal-harbor.example/intake",
                created_at="2026-05-20T09:00:00Z",
            ),
        ),
    )


def test_each_candidate_carries_the_timing_beside_everything_that_joins_it(
    tmp_path: Path,
) -> None:
    """Within-component proximity, computed per candidate and reported per candidate.

    Two numbers, and they are not the same number: the gap between the nearest two
    accounts is what the window is measured against, and the span from the earliest post
    to the latest is what says whether a desk was running a campaign or one operator
    having an account. A candidate resting on one registration gets one row of each, and
    the window travels on the row so a reader of the file can check the verdict rather
    than take it.
    """
    candidates_path, _ = run(tmp_path, timing_corpus(tmp_path))
    written = by_accounts(candidates_path)

    near = written[("syn_alpha_0001", "syn_beta_0002")]
    assert records(near, "corroboration") == [
        {
            "kind": "registration",
            "value": "vantage-ledger.example",
            "closest_seconds": 1_800,
            "span_seconds": 16_200,
        }
    ]
    assert near["timing"] == {
        "window_seconds": DAY,
        "pieces": 1,
        "corroborated": 1,
        "closest_seconds": 1_800,
    }

    far = written[("syn_delta_0004", "syn_gamma_0003")]
    assert records(far, "corroboration") == [
        {
            "kind": "registration",
            "value": "signal-harbor.example",
            "closest_seconds": 11_664_000,
            "span_seconds": 11_664_000,
        }
    ]
    assert far["timing"] == {
        "window_seconds": DAY,
        "pieces": 1,
        "corroborated": 0,
        "closest_seconds": 11_664_000,
    }


def test_the_window_is_stated_in_the_output_and_the_gap_beside_every_thing_that_joins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rule is printed where a reader meets it, not left in the code.

    A window nobody can see is a threshold they have to guess at before they can judge
    whether a candidate is corroborated, so the figure says what the window is and the
    block under each candidate says what it made of every registration and Contact Point
    that joined it — the nearest pair of accounts either side of the threshold, in the
    same units a reader can subtract off the `created_at` printed above them.
    """
    _, printed = run(tmp_path, timing_corpus(tmp_path), capsys)

    assert "window        24 hours between two accounts on one piece of evidence" in printed
    assert (
        "timing        1 of 2 candidates and 1 of 2 pieces of evidence corroborated"
        in printed
    )

    near = printed.split("cc-01  2 accounts")[1].split("\n\n")[0]
    assert (
        "timing        1 of 1 piece of evidence within the 24-hour window; "
        "nearest pair 30m apart" in near
    )
    assert "vantage-ledger.example  2 accounts, 3 posts" in near
    assert "4h30m across, inside the 24-hour window" in near

    far = printed.split("cc-02  2 accounts")[1].split("\n\n")[0]
    assert (
        "timing        0 of 1 piece of evidence within the 24-hour window; "
        "nearest pair 135d apart" in far
    )
    assert "135d across, outside the 24-hour window" in far


def test_the_span_beside_a_piece_of_evidence_is_not_the_gap_inside_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two figures over the same accounts, because they answer two questions.

    The nearest pair is 30 minutes apart and the candidate ran for four and a half hours,
    and a run that printed the gap twice would have answered neither: the gap is how close
    any two of the accounts ever came, and the span is what the window is measured against
    and what tells a desk running a campaign apart from one operator having an account for
    years. The span is what the ticket means when it says accounts spread over months may
    be a domain that changed hands.
    """
    candidates_path, _ = run(tmp_path, timing_corpus(tmp_path))
    written = by_accounts(candidates_path)
    piece = records(written[("syn_alpha_0001", "syn_beta_0002")], "corroboration")[0]

    assert piece["closest_seconds"] == 1_800
    assert piece["span_seconds"] == 16_200


def test_every_published_gap_is_the_one_the_corpus_and_the_two_readings_hold(
    tmp_path: Path,
) -> None:
    """Every figure recomputed a second time, off two other commands' files.

    The command does not read `post-domains.jsonl` or `post-contacts.jsonl` — it
    re-resolves the links and re-reads the handles itself, so the grouping and those two
    files come out of separate runs over the same Corpus. Recomputing the gaps here from
    them makes "two accounts sharing this registration" a checked statement rather than a
    shared assumption, and it is done by pairing every account with every other rather than
    by the walk over neighbouring posts the run uses, so the two agree by being the same
    arithmetic twice rather than by sharing a method.
    """
    corpus = {item.post_id: item for item in read_corpus(COMMITTED_CORPUS)}
    reaching_domains: dict[str, set[str]] = {}
    for row in rows(COMMITTED_POST_DOMAINS):
        for domain in texts(row, "domains"):
            reaching_domains.setdefault(domain, set()).add(text(row, "account"))
    reaching_points: dict[str, set[str]] = {}
    for row in rows(COMMITTED_CONTACTS):
        for match in records(row, "matches"):
            reaching_points.setdefault(text(match, "value"), set()).add(text(row, "account"))

    moments: dict[str, list[int]] = {}
    for item in corpus.values():
        moments.setdefault(item.account, []).append(
            int(datetime.fromisoformat(item.created_at).timestamp())
        )

    candidates_path, _ = run(tmp_path)
    written = rows(candidates_path)
    assert written, "the shipped Corpus produces no candidates to check"

    for row in written:
        accounts = set(texts(row, "accounts"))
        for piece in records(row, "corroboration"):
            value = text(piece, "value")
            if text(piece, "kind") == "Contact Point":
                published = reaching_points.get(value)
            else:
                published = reaching_domains.get(value)
            assert published is not None, value

            members = sorted(accounts & published)
            assert len(members) > 1, f"{value} joins one account"
            stamps = sorted(
                (moment, member)
                for member in members
                for moment in moments[member]
            )
            gaps = [
                later[0] - earlier[0]
                for earlier, later in combinations(stamps, 2)
                if earlier[1] != later[1]
            ]
            assert integer(piece, "span_seconds") == max(m[0] for m in stamps) - min(
                m[0] for m in stamps
            ), piece
            assert integer(piece, "closest_seconds") == min(gaps), piece

        summary = nested(row, "timing")
        spans = [integer(piece, "span_seconds") for piece in records(row, "corroboration")]
        assert integer(summary, "pieces") == len(spans)
        assert integer(summary, "corroborated") == sum(
            1 for span in spans if span <= integer(summary, "window_seconds")
        )
        assert integer(summary, "closest_seconds") == min(
            integer(piece, "closest_seconds") for piece in records(row, "corroboration")
        )


def test_the_shipped_corpus_agrees_with_itself_about_which_candidate_the_clock_has_nothing_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the clock has something to say about on the Corpus this project ships.

    Two candidates, both of them the project's hard cases rather than a fixture's easy
    one. `cc-01`'s registration puts three accounts in the same morning and its Telegram
    handle brings in a fourth account two weeks later, which is the very account that
    costs the project a recovery — so the candidate is corroborated by one of its two
    pieces of evidence and the block says which. `cc-04` is the desk behind the
    obfuscated handles: one handle, three accounts, seven posts over three weeks, and a
    24-hour window that calls none of that one operator. It is deprioritised, named below
    the table, and still proposed.

    The counts are checked against the file the run wrote rather than written out here, so
    a figure that stopped agreeing with its own candidates would be caught rather than
    quietly updated.
    """
    candidates_path, printed = run(tmp_path, capsys=capsys)
    written = {text(row, "candidate_id"): row for row in rows(candidates_path)}
    pieces = [piece for row in written.values() for piece in records(row, "corroboration")]
    corroborated = sum(integer(piece, "span_seconds") <= DAY for piece in pieces)
    by_span = {
        (text(row, "candidate_id"), text(piece, "value")): integer(piece, "span_seconds")
        for row in written.values()
        for piece in records(row, "corroboration")
    }

    assert (
        f"timing        3 of 4 candidates and {corroborated} of {len(pieces)} pieces of "
        "evidence corroborated; no candidate removed" in printed
    )
    assert by_span[("cc-01", "vantage-ledger.example")] == 8_100, "2h15m apart"
    assert by_span[("cc-01", "syn_vantageledger")] == 1_242_960, "14d9h16m apart"

    block = printed.split("cc-01  4 accounts")[1].split("\n\n")[0]
    assert "2h15m across, inside the 24-hour window" in block
    assert "14d9h16m across, outside the 24-hour window" in block
    assert "1 of 2 pieces of evidence within the 24-hour window" in block

    below = printed.split("uncorroborated  ")[1].split("\n\n")[0]
    assert "1 of 4 candidates deprioritised" in below
    assert "cc-04  3 accounts, 7 posts, nearest pair of accounts 42m apart" in below
    assert "cc-04" in written, "a candidate the window says nothing about is still proposed"
    assert records(written["cc-04"], "shared_contact_points")


def test_matching_posting_times_and_no_shared_infrastructure_group_nothing_at_any_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rule the ticket is for, at a window wide enough to swallow everything.

    Four accounts, every one of them posting in the same minute, none of them sharing a
    registration or a Contact Point with any other. Timing says they are one operator as
    loudly as timing can say anything, and there is no window at which it is enough: a
    year is a hundred and fifty times the default and the file is still empty. The window
    is a parameter, so a rule that only held at the default would be a rule about a number
    rather than about timing, and this is the check that it is about timing.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9201",
                "syn_draycott_5583",
                "https://anvil-labels.example/apply",
                created_at="2026-01-05T09:00:00Z",
            ),
            post(
                "syn_p_9202",
                "syn_underhill_7420",
                "https://underhilltalent.example/apply",
                created_at="2026-01-05T09:00:00Z",
            ),
            post("syn_p_9203", "syn_mossgavel_2218", created_at="2026-01-05T09:00:00Z"),
            post(
                "syn_p_9204",
                "syn_pellworth_3196",
                "https://pellworthwork.example/apply",
                created_at="2026-01-05T09:00:00Z",
            ),
        ),
    )

    candidates_path, printed = run(tmp_path / "year", corpus, capsys, window=24 * 365)

    assert rows(candidates_path) == []
    assert "cc-01" not in printed
    assert "0 of 0 candidates and 0 of 0 pieces of evidence corroborated" in printed
    assert "uncorroborated  none" in printed


def test_a_candidate_the_window_corroborates_nothing_is_reported_below_the_others(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deprioritising, in the three places a reader can look for it.

    Four accounts on one registration and two on another. Size puts the four first and
    timing puts the two first, so the order in the file, the order in the table and the
    count in the figure all move together. The four are not removed: they are in the file,
    in the table with the gap beside their evidence, and named again under a heading of
    their own with the one figure a reader needs to disagree with the window. A grouping
    the clock had nothing to say about is still a grouping somebody proposed.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9301",
                "syn_alpha_0001",
                "https://vantage-ledger.example/entry",
                created_at="2026-01-05T09:00:00Z",
            ),
            post(
                "syn_p_9302",
                "syn_beta_0002",
                "https://vantage-ledger.example/intake",
                created_at="2026-02-05T09:00:00Z",
            ),
            post(
                "syn_p_9303",
                "syn_gamma_0003",
                "https://vantage-ledger.example/month-log",
                created_at="2026-03-05T09:00:00Z",
            ),
            post(
                "syn_p_9304",
                "syn_delta_0004",
                "https://vantage-ledger.example/desk-update",
                created_at="2026-04-05T09:00:00Z",
            ),
            post(
                "syn_p_9305",
                "syn_epsilon_0005",
                "https://signal-harbor.example/entry",
                created_at="2026-05-30T09:00:00Z",
            ),
            post(
                "syn_p_9306",
                "syn_zeta_0006",
                "https://signal-harbor.example/intake",
                created_at="2026-05-30T09:30:00Z",
            ),
        ),
    )

    candidates_path, printed = run(tmp_path, corpus, capsys)

    indexed = [
        line.split()[0]
        for line in printed.splitlines()
        if line.startswith("cc-") and line.rstrip().endswith("corroborated")
    ]
    written = {text(row, "candidate_id"): texts(row, "accounts") for row in rows(candidates_path)}
    assert indexed == ["cc-01", "cc-02"]
    assert written["cc-01"] == ["syn_epsilon_0005", "syn_zeta_0006"]
    assert written["cc-02"] == [
        "syn_alpha_0001",
        "syn_beta_0002",
        "syn_delta_0004",
        "syn_gamma_0003",
    ]

    assert (
        "timing        1 of 2 candidates and 1 of 2 pieces of evidence corroborated; "
        "no candidate removed" in printed
    )
    assert (
        "uncorroborated  1 of 2 candidates deprioritised: no evidence under them spans less "
        "than 24 hours, and none of them is removed for it" in printed
    )
    deprioritised = printed.split("uncorroborated  ")[1].split("\n\n")[0]
    assert "cc-02  4 accounts, 4 posts, nearest pair of accounts 28d apart" in deprioritised
    # The candidate is still proposed, still labelled with the edge it rests on, and its
    # gap is still printed beside that edge: nothing about it was taken away.
    assert "justified by  registrations\n" in printed
    assert "vantage-ledger.example  4 accounts, 4 posts  90d across, outside the 24-hour window" in printed


def test_the_window_is_a_parameter_and_a_window_of_nothing_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A threshold the caller states, and a threshold that cannot be met is a mistake.

    Two accounts five hours apart on one registration, run twice: at the one-hour default
    change the gap is outside the window and the candidate is reported as uncorroborated,
    and at the 24-hour default it is inside. So the figure, the line under the candidate
    and the heading below the table all move with the argument rather than with the
    Corpus, and a reader who disagrees with the window can set a different one. Zero is
    refused rather than run, because a window nothing can fall inside would report every
    candidate in the Corpus as uncorroborated for a reason that is about the argument.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9401",
                "syn_alpha_0001",
                "https://vantage-ledger.example/entry",
                created_at="2026-05-30T09:00:00Z",
            ),
            post(
                "syn_p_9402",
                "syn_beta_0002",
                "https://vantage-ledger.example/intake",
                created_at="2026-05-30T14:00:00Z",
            ),
        ),
    )

    _, hour = run(tmp_path / "hour", corpus, capsys, window=1)
    _, day = run(tmp_path / "day", corpus, capsys, window=24)

    assert "window        1 hour between two accounts on one piece of evidence" in hour
    assert "0 of 1 candidate and 0 of 1 piece of evidence corroborated" in hour
    assert "5h across, outside the 1-hour window" in hour
    assert "uncorroborated  1 of 1 candidate deprioritised" in hour

    assert "window        24 hours between two accounts on one piece of evidence" in day
    assert "1 of 1 candidate and 1 of 1 piece of evidence corroborated" in day
    assert "5h across, inside the 24-hour window" in day
    assert "uncorroborated  none" in day

    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "campaign-candidates",
                "--corpus",
                str(corpus),
                "--candidates",
                str(tmp_path / "campaign-candidates.jsonl"),
                "--window-hours",
                "0",
            ]
        )

    assert "--window-hours is 0" in str(refusal.value)
    assert not (tmp_path / "campaign-candidates.jsonl").exists()


def test_a_post_stamped_without_an_offset_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A timestamp with no offset cannot be subtracted from one that carries it.

    The Corpus generator writes RFC 3339 in UTC throughout, so this only arises for a
    Corpus Provider that writes a local time — which is exactly when the run has to stop
    rather than raise a `TypeError` from inside a subtraction the reader of this output
    never sees. The refusal names the value, because a provider with a hundred posts
    stamped locally needs to know which field to fix and not that something about gaps
    went wrong.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9501",
                "syn_alpha_0001",
                "https://vantage-ledger.example/entry",
                created_at="2026-05-30T09:00:00",
            ),
            post(
                "syn_p_9502",
                "syn_beta_0002",
                "https://vantage-ledger.example/intake",
                created_at="2026-05-30T09:30:00Z",
            ),
        ),
    )

    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "campaign-candidates",
                "--corpus",
                str(corpus),
                "--candidates",
                str(tmp_path / "campaign-candidates.jsonl"),
            ]
        )

    printed = capsys.readouterr().out
    assert "created_at='2026-05-30T09:00:00' carries no offset" in str(refusal.value)
    assert "2026-05-30T09:00:00" in str(refusal.value)
    assert not (tmp_path / "campaign-candidates.jsonl").exists()
    assert "Traceback" not in printed


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
        indexed = [
            line.split()[0]
            for line in printed.splitlines()
            if line.startswith("cc-") and line.rstrip().endswith("corroborated")
        ]
        assert indexed[: len(written)] == [text(row, "candidate_id") for row in written]
