"""Contact Points: what the posts say to be reached at, per post, and which are shared.

ADR-0005 lets two accounts reach the same Campaign Candidate on a Contact Point as
well as on a registrable domain, and this is the step that finds them. A Contact
Point is off-platform infrastructure an operator controls and a reader can be sent
to, so a handle two accounts publish is a shared registration in everything but name
— and it is the one the Corpus's planted campaigns are only half reachable by, since
a campaign that rotates domains per post is invisible to the domain path and visible
here.

The number matters less than the sharing. One handle in one post says an account
published something; the same handle in two posts by two accounts is what ADR-0005
counts as grounds for a proposal, which is why the shared table is printed separately
from the per-post list rather than folded into it.

Seam under test: the `contact-points` command, observed through the two files it
writes and the text it prints. Nothing here inspects the code that wrote them.

The shipped Corpus holds the ordinary half: two Planted Campaigns naming a channel,
an intake address in one post, and one handle written two different ways. The half
that is easy to get silently wrong — an `@` that names nothing, a run too short or
too long for the thing it claims to be, an address with no host, a link to a channel
rather than to a person — is exercised against a Corpus written here in the test,
through the same command. A candidate that names no Contact Point must still come out
the far side with a reason attached: a candidate quietly dropped is indistinguishable
from a post that named none, and the difference is the whole claim.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_CORPUS_PATH, DEFAULT_SEED, main
from reddit_fraud_intelligence.contacts import Unread
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_CONTACTS = REPO_ROOT / "data" / "contacts" / "post-contacts.jsonl"
COMMITTED_REPORT = REPO_ROOT / "docs" / "contact-points.md"
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"

Row = Mapping[str, object]


def run(
    directory: Path,
    corpus: Path = DEFAULT_CORPUS_PATH,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> tuple[Path, Path, str]:
    contacts_path = directory / "post-contacts.jsonl"
    report_path = directory / "contact-points.md"
    exit_code = main(
        [
            "contact-points",
            "--corpus",
            str(corpus),
            "--contacts",
            str(contacts_path),
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out if capsys is not None else ""
    return contacts_path, report_path, printed


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


def points_of(row: Row) -> list[tuple[str, str]]:
    """Every occurrence as `(kind, canonical value)`, in the order the file holds."""
    return [
        (text(match, "kind"), text(match, "value"))
        for match in records(row, "matches")
    ]


def unread_of(row: Row) -> list[tuple[str, str]]:
    """Every candidate that named no Contact Point, as `(found, reason)`."""
    return [(text(entry, "found_as"), text(entry, "reason")) for entry in records(row, "unread")]


def point_block(printed: str, value: str) -> str:
    """One Contact Point's block from the console output.

    The console prints a heading, then one block per Contact Point — a two-space line
    naming the value and its reach, then a four-space line per occurrence — and then a
    footer of prose. Reading one block rather than the whole output is what lets a test
    ask about one value and not about the others, which is the difference between
    checking that a handle is shared by four accounts and checking that four accounts
    appear somewhere in the run. The block stops at the next value's heading, so the
    next value's accounts and the footer's prose about sharing are not read as this
    one's.
    """
    for paragraph in printed.split("\n\n"):
        lines = paragraph.splitlines()
        if not lines or not lines[0].startswith("points  "):
            continue
        for index, line in enumerate(lines):
            if not line.startswith("  ") or value not in line:
                continue
            block = [line]
            for following in lines[index + 1 :]:
                if following.startswith("  ") and not following.startswith("   "):
                    break
                block.append(following)
            return "\n".join(block)
    raise AssertionError(f"the output has no block for {value!r}")


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
    *,
    title: str = "title",
    body: str = "body",
    links: tuple[str, ...] = (),
    created_at: str = "2026-01-05T09:00:00Z",
) -> CorpusItem:
    return CorpusItem(
        post_id=post_id,
        account=account,
        subreddit="test",
        title=title,
        body=body,
        created_at=created_at,
        links=links,
    )


# --- what a post says, and where -------------------------------------------------


def test_an_address_is_read_out_of_a_title_a_body_and_a_link(tmp_path: Path) -> None:
    """Three of the four ways a post names a Contact Point, all of them plain.

    A title, a body, and a `mailto:` link are the three spellings the Corpus and the
    wild carry, and the last one is the reason this command exists at all for a link:
    `post-domains` reports a `mailto:` link as naming no registration, because it
    names a person rather than a site, and it says so by handing the case to this
    ticket. A reader who finds a `mailto:` in the resolved links has to be able to
    follow it to the address it carried.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9001",
                "syn_inthree_0001",
                title="Intake is open, write to intake@vantage-ledger.example",
                body="Or ask in here first.",
                links=("mailto:desk@vantage-ledger.example",),
            ),
        ),
    )
    contacts_path, report_path, _ = run(tmp_path, corpus)

    written = by_post(contacts_path)["syn_p_9001"]
    assert points_of(written) == [
        ("email", "intake@vantage-ledger.example"),
        ("email", "desk@vantage-ledger.example"),
    ]
    assert [text(match, "field") for match in records(written, "matches")] == [
        "title",
        "link",
    ]

    # The title's address is quoted as the title wrote it, so a reader can find it.
    assert text(records(written, "matches")[0], "found_as") == (
        "intake@vantage-ledger.example"
    )
    assert "mailto:desk@vantage-ledger.example" in report_path.read_text(encoding="utf-8")


def test_a_telegram_handle_is_read_and_a_link_to_one_is_read_too(tmp_path: Path) -> None:
    """A handle in prose, and the same handle behind a `t.me` link.

    The second is not a smaller case. `https://t.me/somebody` is how a handle is
    shared whenever the platform it came from renders links, and an extractor that
    only looked at prose would miss every one of them — so the link is read for the
    handle behind it and the whole link is reported as what was found, which is what
    lets a reader go back to the post and see it there.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9002",
                "syn_inhandle_0002",
                body="Add me on @syn_vantageledger before you pay anything.",
                links=("https://t.me/syn_northwindhire",),
            ),
        ),
    )
    contacts_path, _, _ = run(tmp_path, corpus)

    written = by_post(contacts_path)["syn_p_9002"]
    assert points_of(written) == [
        ("telegram", "syn_vantageledger"),
        ("telegram", "syn_northwindhire"),
    ]
    found = [text(match, "found_as") for match in records(written, "matches")]
    assert found == ["@syn_vantageledger", "https://t.me/syn_northwindhire"]


def test_one_post_that_names_the_same_thing_twice_lists_it_once(tmp_path: Path) -> None:
    """A post is read into a set of Contact Points, not a count of how often it wrote
    one.

    The same handle in the title and the body is one handle, and listing it twice
    would overstate how much shared infrastructure a post reaches — which is the
    figure a grouping step is about to read. Every occurrence is still reported, so
    a reader can see both places it was written.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9003",
                "syn_intwice_0003",
                title="Ask @syn_vantageledger",
                body="Or @syn_vantageledger on Telegram, whichever is easier.",
            ),
        ),
    )
    contacts_path, _, _ = run(tmp_path, corpus)

    written = by_post(contacts_path)["syn_p_9003"]
    assert texts(written, "contact_points") == ["syn_vantageledger"]
    assert len(records(written, "matches")) == 2


# --- what the reader cannot be given a wrong answer about ------------------------


def test_a_candidate_that_names_no_contact_point_is_reported_with_its_reason(
    tmp_path: Path,
) -> None:
    """Every way a candidate fails, each with its own reason, none of them dropped.

    An `@` standing on its own, a run too short for a username and one too long, a run
    carrying a character a username cannot hold, an address with nothing after the
    `@`, an address whose host names no host, an address too long to be one, a Telegram
    link to a channel rather than to a person, and a link that will not parse at all.
    All eight are things a scam post carries in the wild or a broken post does, and all
    eight come out named rather than dropped — the last one most of all, because a
    link the parser refuses is the one case this command cannot even classify, and an
    unclassifiable case is exactly the one that needs a reason attached.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9004",
                "syn_ineight_0004",
                body=(
                    "Write to me at @ or to intake@ or to nobody@localhost or to "
                    "help@vantage-ledger.example, or try @jo or @"
                    + "a" * 40
                    + " or @someone.co.uk."
                ),
                links=(
                    "https://t.me/+k3Qv7xLm",
                    "http://[::1/pay",
                    "mailto:" + "a" * 250 + "@vantage-ledger.example",
                ),
            ),
        ),
    )
    contacts_path, report_path, _ = run(tmp_path, corpus)

    written = by_post(contacts_path)["syn_p_9004"]
    assert unread_of(written) == [
        ("@", "no_name"),
        ("intake@", "no_domain"),
        ("nobody@localhost", "no_domain"),
        ("@jo", "handle_too_short"),
        ("@" + "a" * 40, "handle_too_long"),
        ("@someone.co.uk.", "handle_character"),
        ("https://t.me/+k3Qv7xLm", "invite_link"),
        ("http://[::1/pay", "malformed"),
        ("mailto:" + "a" * 250 + "@vantage-ledger.example", "address_too_long"),
    ]
    assert points_of(written) == [("email", "help@vantage-ledger.example")]

    # Every reason the enum holds is reached above, so a reason added later cannot be
    # published in the report's vocabulary without a case that produces it.
    assert {reason for _, reason in unread_of(written)} == {
        reason.value for reason in Unread
    }

    # The reasons are named in the report as well as carried in the file, so a reader
    # knows the vocabulary on a run where nothing failed.
    report = report_path.read_text(encoding="utf-8")
    for reason in Unread:
        assert f"`{reason.value}`" in report


def test_an_address_that_names_no_host_is_never_repaired_into_one(tmp_path: Path) -> None:
    """`nobody@localhost` is reported, not turned into `nobody@localhost.example`.

    The tempting repair is to append a TLD, or to take the last two labels of
    whatever follows the `@` and call it a host. Both produce an address that looks
    right, and an address that looks right is what a grouping step would join two
    accounts on — so a candidate its own rules cannot read comes back with the reason
    it could not be read.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (post("syn_p_9005", "syn_inhost_0005", body="Try nobody@localhost or x@."),),
    )
    contacts_path, _, _ = run(tmp_path, corpus)

    written = by_post(contacts_path)["syn_p_9005"]
    assert unread_of(written) == [
        ("nobody@localhost", "no_domain"),
        ("x@.", "no_domain"),
    ]
    assert points_of(written) == []


def test_a_trailing_full_stop_is_sentence_punctuation_and_not_part_of_the_handle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The case a reader will check first, because it decides whether two posts
    sharing a handle are seen to share one.

    A handle at the end of a sentence is written `@syn_vantageledger.`, and the full
    stop is the sentence's. Reporting the run as it stands would make it a different
    username from the one in the next post, so the separator is trimmed and the
    spelling as written is reported beside the value it was read as — which is what
    makes the trimming a decision a reader can check rather than a silent one.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9006",
                "syn_instop_0006",
                body="Ask @syn_vantageledger. Then wait.",
            ),
            post(
                "syn_p_9007",
                "syn_instop_0007",
                body="Ask @syn_vantageledger, they answer.",
            ),
        ),
    )
    contacts_path, _, printed = run(tmp_path, corpus, capsys)
    written = by_post(contacts_path)
    values = {
        text(records(written[post_id], "matches")[0], "value")
        for post_id in ("syn_p_9006", "syn_p_9007")
    }

    assert values == {"syn_vantageledger"}
    assert text(records(written["syn_p_9006"], "matches")[0], "found_as") == (
        "@syn_vantageledger."
    )
    assert text(records(written["syn_p_9007"], "matches")[0], "found_as") == (
        "@syn_vantageledger"
    )
    assert "syn_vantageledger" in printed


def test_one_cannot_be_both_read_and_unread(tmp_path: Path) -> None:
    """A candidate is either a Contact Point or the reason it is not one.

    The same string read twice would put one thing in both columns, and a reader
    counting the figures above would add a row that is in neither list.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9008",
                "syn_inboth_0008",
                body="@syn_vantageledger and @jo and intake@vantage-ledger.example",
            ),
        ),
    )
    contacts_path, _, _ = run(tmp_path, corpus)

    written = by_post(contacts_path)["syn_p_9008"]
    read = {text(match, "found_as") for match in records(written, "matches")}
    unread = {found for found, _ in unread_of(written)}

    assert not read & unread
    assert points_of(written) == [
        ("telegram", "syn_vantageledger"),
        ("email", "intake@vantage-ledger.example"),
    ]


# --- sharing, which is the part that makes a Contact Point useful -----------------


def test_one_contact_point_in_two_posts_by_two_accounts_is_reported_as_shared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The claim ADR-0005 turns on, with the case that must not be counted beside it.

    Three accounts, two of which publish one handle and the third of which publishes a
    different one. The two that publish the same handle are the sharing; the third is a
    Contact Point in the Corpus that happens to be in a third post, and folding it in
    would report a reach of three accounts that nobody has evidence for. It is also the
    case the shipped Corpus makes over and over, so the two are planted together rather
    than argued about separately.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9009",
                "syn_inshare_a",
                body="Intake is @syn_vantageledger if you want it.",
                links=("https://vantage-ledger.example/entry",),
            ),
            post(
                "syn_p_9010",
                "syn_inshare_b",
                body="Same desk, @syn_vantageledger.",
                links=("https://mirror.vantage-ledger.example/log",),
                created_at="2026-01-05T11:00:00Z",
            ),
            post(
                "syn_p_9011",
                "syn_inshare_c",
                body="A different operator, @syn_somebodyelse.",
                created_at="2026-01-05T12:00:00Z",
            ),
        ),
    )
    _, _, printed = run(tmp_path, corpus, capsys)

    assert "2 Contact Points read, 1 of them reached by 2 or more accounts" in printed
    block = point_block(printed, "syn_vantageledger")
    assert "telegram" in block
    assert "shared, 2 posts, 2 accounts" in block
    # The account that published it once is in the Corpus and not in the block.
    assert "syn_inshare_a" in block
    assert "syn_inshare_b" in block
    assert "syn_inshare_c" not in block


def test_a_contact_point_seen_twice_by_one_account_is_not_reported_as_shared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One account saying the same thing twice is not two accounts saying it once.

    Sharing is counted over accounts rather than over posts, because the claim being
    made is about accounts. An account that names one handle in four posts has not
    acquired a second reach, and a count of posts would report it as shared and let
    a Campaign Candidate be proposed on it.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_9012", "syn_inonce_a", body="@syn_vantageledger"),
            post(
                "syn_p_9013",
                "syn_inonce_a",
                body="@syn_vantageledger again",
                created_at="2026-01-06T09:00:00Z",
            ),
        ),
    )
    _, _, printed = run(tmp_path, corpus, capsys)

    assert "1 Contact Point read, 0 of them reached by 2 or more accounts" in printed
    block = point_block(printed, "syn_vantageledger")
    assert "shared" not in block
    assert "2 posts, 1 account" in block


def test_a_handle_written_two_ways_is_one_contact_point_and_the_output_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Normalisation is reported, so a reader can see two posts named the same thing.

    `@Syn_VantageLedger` and `@syn_vantageledger` are one username written two ways,
    and Telegram usernames do not distinguish case. Both spellings are printed beside
    the value they were read as, which is the difference between a claim the reader
    can check and one they have to take: without the spellings, a reader cannot tell
    whether two posts were joined because the handle is the same or because the rule
    lowercased it.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_9014", "syn_inalpha", body="Ask @Syn_VantageLedger"),
            post(
                "syn_p_9015",
                "syn_inbeta",
                body="Ask @syn_vantageledger",
                created_at="2026-01-06T09:00:00Z",
            ),
        ),
    )
    _, _, printed = run(tmp_path, corpus, capsys)

    block = point_block(printed, "syn_vantageledger")
    assert "shared, 2 posts, 2 accounts" in block
    assert "@Syn_VantageLedger" in block
    assert "@syn_vantageledger" in block


def test_an_address_is_case_folded_in_the_host_and_not_in_the_local_part(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one normalisation a reader is most likely to get wrong, and the one that
    would invent a grouping edge.

    RFC 5321 makes a domain case-insensitive and says nothing of the sort about the
    part before the `@`, so `desk@VANTAGE-LEDGER.EXAMPLE` and `desk@vantage-ledger.example`
    are one mailbox while `Desk@` and `desk@` may be two. Folding the whole address
    would merge those two on the strength of a rule nobody publishes — and a merged
    mailbox is a shared identifier between two accounts that share nothing, which is
    the one failure this command exists to avoid, arrived at by being helpful.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9016",
                "syn_inaddr_0016",
                body="Desk@vantage-ledger.example or DESK@VANTAGE-LEDGER.EXAMPLE",
            ),
            post(
                "syn_p_9017",
                "syn_inaddr_0017",
                body="desk@vantage-ledger.example",
                created_at="2026-01-06T09:00:00Z",
            ),
        ),
    )
    contacts_path, _, printed = run(tmp_path, corpus, capsys)

    written = by_post(contacts_path)
    # The host is folded, so the two spellings of the same local part are one value.
    assert texts(written["syn_p_9016"], "contact_points") == [
        "DESK@vantage-ledger.example",
        "Desk@vantage-ledger.example",
    ]
    assert texts(written["syn_p_9017"], "contact_points") == ["desk@vantage-ledger.example"]

    # And the local part is not folded, so `Desk@` and `desk@` stay apart and the
    # shared count does not pick up a mailbox that may not exist.
    assert "3 Contact Points read, 0 of them reached by 2 or more accounts" in printed
    for value in ("Desk@vantage-ledger.example", "DESK@vantage-ledger.example"):
        assert "shared" not in point_block(printed, value)


# --- the Corpus the project actually ships ---------------------------------------


def test_the_planted_campaigns_carry_the_handles_their_accounts_share(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The case the Corpus is planted for, read back out of the run's own output.

    Both campaigns publish a channel, from more than one of their own accounts, which
    is the whole of what makes Contact Points worth extracting: a handle shared
    across a campaign's accounts is shared infrastructure that no registrable domain
    has to be rotated for. The membership is not read from `truth.jsonl` — these
    account names are in the Corpus file any reader can open, and naming what the
    right answer is *is* the claim.

    Not every post carries the handle, and that is the point of the third account in
    the list: an extractor that is only ever right about the posts advertising a
    Contact Point has been measured on nothing else.
    """
    _, _, printed = run(tmp_path, capsys=capsys)

    assert "3 Contact Points read, 2 of them reached by 2 or more accounts" in printed

    alpha = point_block(printed, "syn_vantageledger")
    assert "telegram" in alpha
    assert "shared, 4 posts, 4 accounts" in alpha
    for account in (
        "syn_harborlight_5517",
        "syn_pinecrest_9032",
        "syn_quantproof_2841",
        "syn_greyloch_6612",
    ):
        assert account in alpha

    beta = point_block(printed, "syn_northwindhire")
    assert "shared, 2 posts, 2 accounts" in beta
    for account in ("syn_northwindhire_7736", "syn_clearpathwork_3184"):
        assert account in beta

    # The Corpus is planted so that not every post of a campaign carries the handle,
    # so an extractor is never only right about the posts advertising one.
    assert "syn_quantproof_2841" in alpha
    assert "syn_p_0001" not in alpha


def test_the_intake_address_is_in_one_post_and_is_not_reported_as_shared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The floor of the sharing case: a Contact Point only one post names.

    `intake@vantage-ledger.example` is the one address in the Corpus, it is on the
    alpha campaign's own registration, and exactly one post carries it. Reporting it
    as shared would propose a grouping out of one account writing down where to reach
    it, which is the claim ADR-0005 does not make.
    """
    _, _, printed = run(tmp_path, capsys=capsys)

    block = point_block(printed, "intake@vantage-ledger.example")
    assert "email" in block
    assert "shared" not in block
    assert "1 post, 1 account" in block


def test_the_handle_a_victim_quotes_is_shared_with_the_campaign_it_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The most important number in the shared table, and it is a false grouping.

    One of the four accounts that publish `syn_vantageledger` is a Hard Negative: a
    person who lost money and named the channel they were given. That is correct
    reporting and the worst possible grouping input, because the handle is shared
    across the boundary and nothing in the string says which side of it either party
    is on. Whether the accounts belong together is ticket #20's question to ask and a
    reviewer's to answer; this command's job is to refuse to hide the handle in the
    row and to say how many accounts are on each side of it.
    """
    nuisance = REPO_ROOT / "data" / "corpus" / "nuisance.jsonl"
    hard_negatives = {
        account
        for record in rows(nuisance)
        if text(record, "kind") == "hard_negative"
        for account in texts(record, "accounts")
    }
    _, _, printed = run(tmp_path, capsys=capsys)

    alpha = point_block(printed, "syn_vantageledger")
    sharing = {account for account in hard_negatives if account in alpha}

    assert sharing, "no Hard Negative quotes the planted handle, so the case is absent"
    assert "shared, 4 posts, 4 accounts" in alpha
    # Named as sharing, and no grouping built on it: this command does not group.
    assert "nothing here groups on one" in printed


def test_the_corpus_gets_one_row_per_post_including_the_posts_naming_nothing(
    tmp_path: Path,
) -> None:
    """A post that names no Contact Point is a result, not an absence.

    Most of the Corpus does. They are the floor of the case — there is nothing on
    them for a grouping to join by — so a file that listed only the posts that named
    something would hide exactly the posts worth checking, and the "posts carrying
    none" figure would have nothing to be counted against.
    """
    contacts_path, report_path, _ = run(tmp_path)

    written = rows(contacts_path)
    corpus = read_corpus(COMMITTED_CORPUS)

    assert [text(row, "post_id") for row in written] == [item.post_id for item in corpus]
    assert [text(row, "account") for row in written] == [item.account for item in corpus]

    silent = [row for row in written if not records(row, "matches")]
    assert len(silent) > len(written) // 2, "most of the Corpus names nothing at all"
    for row in silent:
        assert texts(row, "contact_points") == []

    report = report_path.read_text(encoding="utf-8")
    assert report.count("### `syn_p_") >= len(corpus)


def test_the_report_states_the_contact_points_for_each_post(tmp_path: Path) -> None:
    """The list a post resolves to, printed, not only the matches under it.

    The ticket asks that the Contact Points for any post be inspectable by eye. A
    table of occurrences answers a different question — what was written where — and
    a post that names one handle twice would print it twice, so the list has to be
    stated in its own right.
    """
    contacts_path, report_path, _ = run(tmp_path)
    report = report_path.read_text(encoding="utf-8")

    section = report.split("### `syn_p_0004`")[1].split("\n### ")[0]
    listed = [line for line in section.splitlines() if line.startswith("Contact points:")]
    assert listed == [
        "Contact points: `intake@vantage-ledger.example`, `syn_vantageledger`"
    ]

    # A post that names nothing says so in the same place, rather than printing a
    # bare heading with nothing under it.
    quiet = report.split("### `syn_p_0008`")[1].split("\n### ")[0]
    assert quiet.startswith(" · `syn_tideline_4471`")
    assert "No Contact Point named" in quiet

    written = by_post(contacts_path)
    assert texts(written["syn_p_0004"], "contact_points") == [
        "intake@vantage-ledger.example",
        "syn_vantageledger",
    ]


def test_the_corpus_itself_holds_a_candidate_that_names_no_contact_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reporting rather than dropping has to be visible in the committed artefacts.

    Every other unread reason is exercised against a Corpus written inside the test,
    which is this repository's usual arrangement, and it is right for the failures that
    are easy to get silently wrong. It is not enough for the one the ticket asks for:
    a reader who opens `docs/contact-points.md` and sees zero candidates cannot tell
    whether reporting works or whether the run never had anything to report.

    So the Corpus plants one — a Hard Negative describing a contact form that asks for
    an address and then names none — and this test holds the committed file to it. A
    seed whose spare material drops that post would fail here, which is the point: the
    reporting is part of the Corpus rather than only of the test.
    """
    contacts_path, report_path, printed = run(tmp_path, capsys=capsys)
    written = rows(contacts_path)
    unread = [(text(row, "post_id"), *unread_of(row)) for row in written if unread_of(row)]
    report = report_path.read_text(encoding="utf-8")

    assert unread, "the Corpus holds no candidate that names no Contact Point"
    assert "1 unread candidate, naming no Contact Point" in printed

    post_id, (found, reason) = unread[0]
    assert found == "intake@"
    # An address with nothing after the `@`: it begins with a local part, so it is read
    # as an address rather than as a handle, and an address has to name a host.
    assert reason == "no_domain"

    # Reported where a reader looks for it: in the post's own section, not only in a
    # column they have to know to go and find.
    section = report.split(f"### `{post_id}`")[1].split("\n### ")[0]
    assert "Candidates that name no Contact Point:" in section
    assert f"`{found}` in the body — **{reason}**" in section


# --- what the output claims about itself -----------------------------------------


def test_the_output_states_the_recall_limit_rather_than_a_coverage_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A number of Contact Points is not a number of Contact Points.

    Nothing here repairs a handle that was written with separators between its
    characters, or reads a screenshot, or guesses at a handle hidden inside an image
    of one — that is ticket #15's work and this build does none of it. A count
    printed without that beside it reads as how many there were, and it is not, so
    the limit is in the command's own output rather than in the ticket that asked
    for it.
    """
    _, report_path, printed = run(tmp_path, capsys=capsys)

    for text_ in (printed, report_path.read_text(encoding="utf-8")):
        assert "recall" in text_.lower()
        assert "lower bound" in text_.lower()
        assert "obfusc" in text_.lower()


def test_the_synthetic_claim_is_measured_and_can_come_out_false(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Measured, not promised, and the measurement can say otherwise.

    A Contact Point is a Synthetic Entity in this build because the Corpus is
    synthetic. The claim is worth as much as its ability to fail, so it is a count and
    not an assertion: an address is one when its host sits under a TLD reserved for
    examples, and a handle when it carries the `syn_` marker every Synthetic Entity in
    this Corpus carries. Telegram reserves no namespace, so the marker is all there is
    for a handle.

    Two of the three Contact Points in this Corpus carry their kind's marker and one
    does not, which is the whole of the test: a run holding a Contact Point outside the
    synthetic namespace has to say so in the figure rather than in a comment, because
    a Corpus Provider is a swap and the same lines over real content would read
    `0 of 3`.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post("syn_p_9018", "syn_insyn_a", body="intake@vantage-ledger.example"),
            post(
                "syn_p_9019",
                "syn_insyn_b",
                body="@syn_vantageledger and desk@vantage-ledger.co.uk",
                created_at="2026-01-06T09:00:00Z",
            ),
        ),
    )
    _, _, printed = run(tmp_path, corpus, capsys)

    figures = {
        line.strip().split(None, 1)[0]: line.strip() for line in printed.splitlines() if line.startswith("  ")
    }
    assert "2 of 3 are Synthetic Entities" in figures["synthetic"]
    assert "1 of 2 addresses under a TLD reserved for examples" in figures["reserved"]
    assert "1 of 1 handles carrying the syn_ marker" in figures["marked"]


def test_the_output_says_every_contact_point_of_this_corpus_is_a_synthetic_entity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shipped Corpus holds nothing outside the synthetic namespace, and says so.

    Read as a claim about the whole run rather than about one value: every address
    sits under a TLD reserved for examples and every handle carries the marker, so the
    figures cannot be read as a partial statement about a subset.
    """
    _, _, printed = run(tmp_path, capsys=capsys)

    figures = {
        line.strip().split(None, 1)[0]: line.strip() for line in printed.splitlines() if line.startswith("  ")
    }
    assert "3 of 3 are Synthetic Entities" in figures["synthetic"]
    assert "1 of 1 addresses under a TLD reserved for examples" in figures["reserved"]
    assert "2 of 2 handles carrying the syn_ marker" in figures["marked"]


def test_the_output_says_nothing_groups_on_a_contact_point_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR-0005 permits a Contact Point as a grouping edge; this build does not use it.

    The shared table is the evidence ticket #20 needs and the reader needs first, and
    a grouping built on it in this ticket would put an unmeasured edge into the
    recovery figure ADR-0004 is measured against. The output has to say which of the
    two it is doing, or a reader will read the shared table as a list of groupings.
    """
    _, _, printed = run(tmp_path, capsys=capsys)

    assert "nothing here groups on one" in printed
    assert "cc-01" not in printed


# --- what the run is allowed to read, and the shape of what it writes ------------


class _Opened:
    """Records every path a run opens, so a claim about what it read can be checked.

    An audit hook rather than a monkeypatch, because the point is to catch a read
    from anywhere at all — including from inside the standard library, which a
    patched `open` would miss. It cannot be uninstalled, so it records only while
    `recording` is set and does nothing for the rest of the session.
    """

    def __init__(self) -> None:
        self.recording = False
        self.paths: list[str] = []

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        if self.recording and event == "open":
            self.paths.append(str(arguments[0]))

    def record(self) -> list[str]:
        return list(self.paths)


def test_the_run_reads_the_corpus_and_nothing_else(tmp_path: Path) -> None:
    """The boundary ADR-0008 is about, checked on the run rather than on the code.

    The Corpus file is the whole input, and the truth file and the Nuisance
    Structure manifest sit next to it holding the answer to the question this command
    answers: which accounts are in one operation. Reading either would make every
    number the project reports a demonstration rather than a measurement. This run
    reads no published list either, because nothing about what a post says to be
    reached at depends on what anybody registered.
    """
    opened = _Opened()
    sys.addaudithook(opened)
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (post("syn_p_9021", "syn_inread_0021", body="@syn_vantageledger"),),
    )

    opened.recording = True
    try:
        run(tmp_path, corpus)
    finally:
        opened.recording = False

    data_files = {
        Path(path).name for path in opened.record() if Path(path).suffix in {".jsonl", ".dat"}
    }
    assert data_files == {"corpus.jsonl", "post-contacts.jsonl"}
    assert not [path for path in opened.record() if Path(path).name.startswith("truth")]
    assert not [path for path in opened.record() if Path(path).name.startswith("nuisance")]


def test_the_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same argument as every other command here: the numbers come from committed bytes."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the contact-points command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    run(tmp_path)


def test_running_twice_writes_byte_identical_files(tmp_path: Path) -> None:
    """The rows are the Corpus's order and the figures are counted, so a fixed Corpus
    has to write fixed bytes."""
    first_contacts, first_report, _ = run(tmp_path / "first")
    second_contacts, second_report, _ = run(tmp_path / "second")

    assert first_contacts.read_bytes() == second_contacts.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()


def test_the_committed_files_are_what_the_command_writes(tmp_path: Path) -> None:
    """Held to the command the same way the Corpus is held to its generator."""
    contacts_path, report_path, _ = run(tmp_path)

    assert contacts_path.read_bytes() == COMMITTED_CONTACTS.read_bytes()
    assert report_path.read_bytes() == COMMITTED_REPORT.read_bytes()


def test_the_report_states_what_was_read_and_how_a_contact_point_is_read(
    tmp_path: Path,
) -> None:
    """Provenance in the report, so a number in it can be traced to the bytes."""
    _, report_path, _ = run(tmp_path)
    report = report_path.read_text(encoding="utf-8")

    assert "data/corpus/corpus.jsonl" in report
    assert "SHA-256" in report
    assert "email" in report
    assert "telegram" in report
    assert "Synthetic Entity" in report


def test_the_command_refuses_to_write_the_corpus_away(tmp_path: Path) -> None:
    """The Corpus is the input, and the measurement is against that exact file."""
    corpus = write_corpus(
        tmp_path / "corpus.jsonl", (post("syn_p_9022", "syn_inx_0022", body="@syn_xedger"),)
    )
    before = corpus.read_bytes()

    with pytest.raises(SystemExit):
        main(
            [
                "contact-points",
                "--corpus",
                str(corpus),
                "--contacts",
                str(tmp_path / "out.jsonl"),
                "--report",
                str(corpus),
            ]
        )

    assert corpus.read_bytes() == before


def test_the_run_covers_every_seed_of_the_corpus_without_a_rerun(tmp_path: Path) -> None:
    """The Corpus is a function of its seed, so a different seed must need no edit here.

    A seed plants different spare Nuisance Structure material, so the posts that name
    a Contact Point change. Every assertion here is about what must hold for any of
    them rather than about the counts this seed happens to produce.
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
        contacts_path, report_path, _ = run(tmp_path / f"out-{seed}", corpus_path)

        corpus = read_corpus(corpus_path)
        written = rows(contacts_path)
        assert len(written) == len(corpus)

        for row in written:
            # Every value reported is a value that appears in the post it is
            # attributed to, and every occurrence names a field the post has.
            item = next(entry for entry in corpus if entry.post_id == text(row, "post_id"))
            haystack = {"title": item.title, "body": item.body}
            for match in records(row, "matches"):
                field = text(match, "field")
                assert field in {"title", "body", "link"}
                found = text(match, "found_as")
                if field == "link":
                    assert found in item.links
                else:
                    assert found in haystack[field]
                assert text(match, "value").lower() in found.lower()

        assert report_path.read_text(encoding="utf-8").count("### `syn_p_") >= len(corpus)