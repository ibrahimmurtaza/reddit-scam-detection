"""The registrable domain of every link in the Corpus, and what a reader makes of it.

This is the first thing the pipeline computes and the input to every grouping
decision after it (ADR-0005), so the file it writes is the credibility boundary
for the rest of the system: if a domain is wrong here, every Campaign Candidate,
every Signal, and every recovery number is wrong in a way nobody can see.

Seam under test: the `post-domains` command, observed through the two files it
writes and the text it prints. Nothing here inspects the code that wrote them.

The Corpus itself is all `https` links to reserved `.example` hosts, so it can
only ever exercise the easy half of the decision. The half that is easy to get
silently wrong — a relative link, a bare public suffix, an address, a link that
will not parse — is exercised against a Corpus written here in the test, through
the same command. A link that resolves to nothing must still appear in the output
with a reason attached: a link that is quietly dropped is indistinguishable from a
post that carried no link at all, and the difference is the whole claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_CORPUS_PATH, DEFAULT_SEED, main
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.domains import LinkDomain, Unresolved

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_DOMAINS = REPO_ROOT / "data" / "domains" / "post-domains.jsonl"
COMMITTED_REPORT = REPO_ROOT / "docs" / "post-domains.md"
COMMITTED_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"

Row = Mapping[str, object]


def run(directory: Path, corpus: Path = DEFAULT_CORPUS_PATH) -> tuple[Path, Path]:
    domains_path = directory / "post-domains.jsonl"
    report_path = directory / "post-domains.md"
    exit_code = main(
        [
            "post-domains",
            "--corpus",
            str(corpus),
            "--domains",
            str(domains_path),
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    return domains_path, report_path


def rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} is not text: {value!r}"
    return value


def maybe_text(row: Row, field: str) -> str | None:
    value = row[field]
    assert value is None or isinstance(value, str), f"{field} is not text: {value!r}"
    return value


def records(row: Row, field: str) -> list[Row]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [entry for entry in value if isinstance(entry, dict)]


def domains_by_post(path: Path) -> dict[str, list[str]]:
    return {text(row, "post_id"): texts(row, "domains") for row in rows(path)}


def links_by_post(path: Path) -> dict[str, list[Row]]:
    return {text(row, "post_id"): records(row, "links") for row in rows(path)}


def texts(row: Row, field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [str(entry) for entry in value]


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


def post(post_id: str, account: str, *links: str) -> CorpusItem:
    return CorpusItem(
        post_id=post_id,
        account=account,
        subreddit="test",
        title="title",
        body="body",
        created_at="2026-01-05T09:00:00Z",
        links=links,
    )


# --- the Corpus the project actually ships -------------------------------------


def test_every_link_in_the_corpus_is_accounted_for(tmp_path: Path) -> None:
    """No link is dropped from the file, and none is missing from the report.

    The count is read off the Corpus rather than written here, so adding a link to
    the generator cannot quietly fall out of this file's accounting.
    """
    domains_path, report_path = run(tmp_path)

    corpus_links = sum(len(item.links) for item in read_corpus(COMMITTED_CORPUS))
    written = sum(len(records(row, "links")) for row in rows(domains_path))
    report = report_path.read_text(encoding="utf-8")

    assert written == corpus_links
    assert report.count("| `http") == corpus_links


def test_the_corpus_gets_one_row_per_post_including_the_posts_with_no_links(tmp_path: Path) -> None:
    """A post that links nothing is a result, not an absence.

    One of the Corpus posts is exactly that, and it is the floor of the case: there
    is no infrastructure to join it on, so any grouping that reaches it reached it
    on text. It has to be in the file for that to be checkable.
    """
    domains_path, _ = run(tmp_path)

    written = rows(domains_path)
    corpus = read_corpus(COMMITTED_CORPUS)

    assert [text(row, "post_id") for row in written] == [item.post_id for item in corpus]
    assert [text(row, "account") for row in written] == [item.account for item in corpus]

    unlinked = [row for row in written if not records(row, "links")]
    assert unlinked, "the Corpus has a post that links nothing; it should be reported"
    for row in unlinked:
        assert texts(row, "domains") == []


def test_the_campaigns_mirror_collapses_onto_the_campaigns_own_domain(tmp_path: Path) -> None:
    """The case the Corpus plants a host to require, read back out of the file.

    `mirror.vantage-ledger.example` and `vantage-ledger.example` are one
    registration. A rule that stops at the hostname reports two domains and splits
    the alpha campaign in half on its own web layout.
    """
    domains_path, _ = run(tmp_path)
    by_post = domains_by_post(domains_path)

    assert by_post["syn_p_0001"] == ["vantage-ledger.example"]
    assert by_post["syn_p_0002"] == ["vantage-ledger.example"]
    assert by_post["syn_p_0003"] == ["vantage-ledger.example"]
    assert by_post["syn_p_0004"] == ["vantage-ledger.example"]


def test_the_near_miss_pairs_are_four_domains_and_not_two(tmp_path: Path) -> None:
    """The other half of the same coin: a name one character off is a different
    registration, and the two bookkeeping and warehouse posts must not join."""
    domains_path, report_path = run(tmp_path)
    by_post = domains_by_post(domains_path)

    assert by_post["syn_p_0025"] == ["vantage-ledgers.example"]
    assert by_post["syn_p_0026"] == ["signal-harbour.example"]

    assert by_post["syn_p_0025"] != by_post["syn_p_0001"]
    assert by_post["syn_p_0026"] != by_post["syn_p_0005"]

    report = report_path.read_text(encoding="utf-8")
    assert "vantage-ledger.example" in report
    assert "vantage-ledgers.example" in report
    assert "signal-harbor.example" in report
    assert "signal-harbour.example" in report


def test_a_post_that_links_the_same_registration_twice_lists_it_once(tmp_path: Path) -> None:
    """The list a reviewer reads is a set of registrations, not a list of URLs.

    A link-in-bio page and the site behind it are two links to one domain, and
    printing the domain twice would overstate how much infrastructure a post
    reaches — which is the number ticket #19 will measure false groupings against.
    No post in the shipped Corpus does this, so it is written here.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9006",
                "syn_twolinks_0006",
                "https://biopage.example/somebody",
                "https://novemberquill.example/catalogue",
                "https://mirror.novemberquill.example/catalogue",
            ),
        ),
    )
    domains_path, _ = run(tmp_path, corpus)

    assert len(links_by_post(domains_path)["syn_p_9006"]) == 3
    assert domains_by_post(domains_path)["syn_p_9006"] == [
        "biopage.example",
        "novemberquill.example",
    ]


# --- the half the shipped corpus cannot reach ----------------------------------


def test_links_with_no_usable_host_are_reported_with_a_reason(tmp_path: Path) -> None:
    """Every way a link can fail to name a registration, each with its own reason.

    A relative link, a scheme that carries no host, a bare public suffix, an
    address, a label too long to be a name, and a link that will not parse. All six
    are links a scam post carries in the wild, and all six have to come out the far
    side named rather than dropped.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9001",
                "syn_eightcases_0001",
                "https://vantage-ledger.example/entry",
                "/r/example/comments/abc",
                "mailto:someone@vantage-ledger.example",
                "https://co.uk/pricing",
                "http://192.0.2.1/pay",
                f"https://{'a' * 300}.example/x",
                "http://[::1/pay",
            ),
        ),
    )
    domains_path, _ = run(tmp_path, corpus)

    links = links_by_post(domains_path)["syn_p_9001"]
    assert [text(link, "link") for link in links] == [
        "https://vantage-ledger.example/entry",
        "/r/example/comments/abc",
        "mailto:someone@vantage-ledger.example",
        "https://co.uk/pricing",
        "http://192.0.2.1/pay",
        f"https://{'a' * 300}.example/x",
        "http://[::1/pay",
    ]
    assert [maybe_text(link, "unresolved") for link in links] == [
        None,
        "relative",
        "no_host",
        "public_suffix",
        "address",
        "invalid_host",
        "malformed",
    ]
    assert maybe_text(links[0], "domain") == "vantage-ledger.example"
    assert maybe_text(links[0], "host") == "vantage-ledger.example"
    assert all(link["domain"] is None for link in links[1:])

    # The post still resolves to the one domain it does have.
    assert domains_by_post(domains_path)["syn_p_9001"] == ["vantage-ledger.example"]


def test_the_scheme_never_decides_whether_a_host_is_read(tmp_path: Path) -> None:
    """Every scheme that carries a host is read the same way, and the choice is stated.

    The alternative — extracting `http` and `https` only — is simpler to describe
    and quietly loses infrastructure: a shortener or a mirror reached over `ftp://`
    is the same registration, and a scam operator picks the scheme. So the rule is
    one rule, about whether there is a host, and it is written into the report so a
    reader is not left guessing which schemes were in scope.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9002",
                "syn_schemes_0002",
                "https://vantage-ledger.example/a",
                "http://vantage-ledger.example/b",
                "ftp://vantage-ledger.example/c",
                "//vantage-ledger.example/d",
            ),
        ),
    )
    domains_path, report_path = run(tmp_path, corpus)

    links = links_by_post(domains_path)["syn_p_9002"]
    assert [text(link, "scheme") for link in links] == ["https", "http", "ftp", ""]
    assert {maybe_text(link, "domain") for link in links} == {"vantage-ledger.example"}
    assert all(link["unresolved"] is None for link in links)

    report = report_path.read_text(encoding="utf-8")
    assert "scheme" in report.lower()
    assert "ftp://" in report


def test_a_link_that_cannot_be_parsed_at_all_does_not_stop_the_run(tmp_path: Path) -> None:
    """One broken link must not cost the reader every other link on the post."""
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9003",
                "syn_broken_0003",
                "http://[::1/pay",
                "https://anvil-labels.example/apply",
            ),
            post("syn_p_9004", "syn_fine_0004", "https://pellworthwork.example/apply"),
        ),
    )
    domains_path, _ = run(tmp_path, corpus)

    assert domains_by_post(domains_path)["syn_p_9004"] == ["pellworthwork.example"]
    assert maybe_text(links_by_post(domains_path)["syn_p_9003"][0], "unresolved") == "malformed"


# --- the shape of the output ---------------------------------------------------


def test_the_report_states_the_registration_list_for_each_post(tmp_path: Path) -> None:
    """The list a post resolves to, printed, not only the per-link table under it.

    The ticket asks that the domain list for any post be inspectable by eye. A
    table with one row per link answers a different question — what each link was —
    and a post that reaches one registration through three of them would print that
    registration three times, so the list has to be stated in its own right.
    """
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        (
            post(
                "syn_p_9007",
                "syn_threelinks_0007",
                "https://novemberquill.example/catalogue",
                "https://mirror.novemberquill.example/catalogue",
                "https://novemberquill.example/sale",
            ),
        ),
    )
    _, report_path = run(tmp_path, corpus)
    report = report_path.read_text(encoding="utf-8")

    section = report.split("### `syn_p_9007`")[1]
    listed = [line for line in section.splitlines() if line.startswith("Registrations:")]
    assert listed == ["Registrations: `novemberquill.example`"]


def test_a_link_cannot_be_both_resolved_and_unresolved() -> None:
    """The two are two readings of one fact, and the type says so.

    A link with neither would be a link the pipeline never looked at, which is the
    one failure this whole module exists to prevent — so it is refused at
    construction rather than discovered downstream.
    """
    with pytest.raises(ValueError, match="has a domain and a reason, or neither"):
        LinkDomain(
            link="https://vantage-ledger.example/a",
            scheme="https",
            host="vantage-ledger.example",
            domain="vantage-ledger.example",
            unresolved=Unresolved.PUBLIC_SUFFIX,
        )

    with pytest.raises(ValueError, match="has a domain and a reason, or neither"):
        LinkDomain(
            link="https://co.uk/",
            scheme="https",
            host="co.uk",
            domain=None,
            unresolved=None,
        )


def test_the_report_is_plain_text_a_reviewer_can_read_by_eye(tmp_path: Path) -> None:
    """Every post named, every link on it named, and the decision spelled out.

    A reviewer who has never run the project should be able to open the report,
    find a post, and see which registrations it links to without running anything.
    """
    _, report_path = run(tmp_path)
    report = report_path.read_text(encoding="utf-8")

    assert "\r" not in report

    for item in read_corpus(COMMITTED_CORPUS):
        assert item.post_id in report
        assert item.account in report
        for link in item.links:
            assert link in report

    # The unresolvable case is rendered where a reader will see it, not hidden in
    # a column they have to know to look at.
    unlinked_section = report.split("syn_p_0008")[1].split("###")[0]
    assert "No links" in unlinked_section


def test_the_report_states_what_was_read_and_how_a_link_is_read(tmp_path: Path) -> None:
    """Provenance in the report, so a number in it can be traced to the bytes."""
    _, report_path = run(tmp_path)
    report = report_path.read_text(encoding="utf-8")

    assert "data/corpus/corpus.jsonl" in report
    assert "data/public-suffix/public_suffix_list.dat" in report
    assert "public_suffix_list" in report
    assert "Multi-part" in report or "multi-part" in report
    assert "co.uk" in report


def test_the_command_needs_no_network_and_reads_only_its_two_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same argument as the CAFC figures: reproducible offline, from bytes.

    Patching `urllib.request.urlopen` to blow up is the check — anything in the
    command that reached for the network would raise rather than quietly succeed.
    """
    import urllib.request

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the post-domains command reached for the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    run(tmp_path)


def test_running_twice_writes_byte_identical_files(tmp_path: Path) -> None:
    first_domains, first_report = run(tmp_path / "first")
    second_domains, second_report = run(tmp_path / "second")

    assert first_domains.read_bytes() == second_domains.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()


def test_the_committed_files_are_what_the_command_writes(tmp_path: Path) -> None:
    """Held to the command the same way the Corpus is held to its generator."""
    domains_path, report_path = run(tmp_path)

    assert domains_path.read_bytes() == COMMITTED_DOMAINS.read_bytes()
    assert report_path.read_bytes() == COMMITTED_REPORT.read_bytes()


def test_the_command_refuses_to_write_the_corpus_away(tmp_path: Path) -> None:
    """The Corpus is the input. A path collision that overwrote it would destroy the
    thing the whole measurement is against, and the generator has the same guard."""
    corpus = write_corpus(tmp_path / "corpus.jsonl", (post("syn_p_9005", "syn_x_0005"),))

    with pytest.raises(SystemExit):
        main(
            [
                "post-domains",
                "--corpus",
                str(corpus),
                "--domains",
                str(tmp_path / "out.jsonl"),
                "--report",
                str(corpus),
            ]
        )

    assert read_corpus(corpus) == (post("syn_p_9005", "syn_x_0005"),)


def test_the_report_covers_every_seed_of_the_corpus_without_a_rerun(tmp_path: Path) -> None:
    """The Corpus is a function of its seed, and this reads one committed seed.

    A seed whose output is shaped differently — different posts, different hosts —
    must not need a change here, so the assertions above are all in terms of counts
    read off the Corpus rather than numbers written into this file.
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
        domains_path, report_path = run(tmp_path / f"out-{seed}", corpus_path)

        corpus = read_corpus(corpus_path)
        written = rows(domains_path)
        assert len(written) == len(corpus)
        assert sum(len(records(row, "links")) for row in written) == sum(
            len(item.links) for item in corpus
        )
        assert all(
            texts(row, "domains")
            for row in written
            if records(row, "links")
        )
        assert report_path.read_text(encoding="utf-8").count("###") >= len(corpus)
