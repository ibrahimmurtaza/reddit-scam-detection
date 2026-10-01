"""The registrable domain of a host, and the multi-part public suffixes that decide it.

The corpus generator plants `mirror.vantage-ledger.example` alongside
`vantage-ledger.example` so that a reader has to reason about the registrable
domain rather than the hostname, and it plants two near-miss pairs one character
apart so that anything matching on names rather than on registration joins a
bookkeeping job to a signals desk (ADR-0005). Both cases are cheap to pass by
accident and this file is where they are made expensive.

Two seams are under test and nothing else is:

- `PublicSuffixes.registrable_domain`, the algorithm, held against the rules the
  project's own corpus needs and against the rule shapes the Public Suffix List
  uses. A hand-written list of multi-part suffixes would pass the corpus cases
  and fail the shapes, so both are here.
- the `fetch-suffix-list` command, observed through the two files it writes.

The naive alternative — split on the first dot and hope — returns `co.uk` for
`example.co.uk`, `github.io` for `attacker.github.io`, and splits the
`vantage-ledger` pair nowhere at all. The first test below is the one that
fails when someone replaces the list with a constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reddit_fraud_intelligence.cli import main
from reddit_fraud_intelligence.suffixes import PublicSuffixes, to_ascii

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_LIST = REPO_ROOT / "data" / "public-suffix" / "public_suffix_list.dat"
COMMITTED_PROVENANCE = REPO_ROOT / "data" / "public-suffix" / "provenance.jsonl"

# The hosts the Corpus actually carries, and what each must resolve to. Read off the
# links in `data/corpus/corpus.jsonl`; a host added to the generator belongs here.
CORPUS_HOSTS = {
    "vantage-ledger.example": "vantage-ledger.example",
    "mirror.vantage-ledger.example": "vantage-ledger.example",
    "vantage-ledgers.example": "vantage-ledgers.example",
    "signal-harbor.example": "signal-harbor.example",
    "signal-harbour.example": "signal-harbour.example",
    "hopcut.example": "hopcut.example",
    "pastevault.example": "pastevault.example",
    "biopage.example": "biopage.example",
    "rivermill-bikes.example": "rivermill-bikes.example",
    "rimcure.example": "rimcure.example",
    "single-run.example": "single-run.example",
    "plainsaw.example": "plainsaw.example",
    "anvil-labels.example": "anvil-labels.example",
    "underhilltalent.example": "underhilltalent.example",
    "pellworthwork.example": "pellworthwork.example",
    "novemberquill.example": "novemberquill.example",
}


def committed() -> PublicSuffixes:
    assert COMMITTED_LIST.exists(), f"the committed Public Suffix List is missing: {COMMITTED_LIST}"
    return PublicSuffixes.read(COMMITTED_LIST)


@pytest.mark.parametrize(
    ("host", "expected"),
    sorted(CORPUS_HOSTS.items()),
)
def test_every_host_the_corpus_carries_resolves(host: str, expected: str) -> None:
    assert committed().registrable_domain(host) == expected


def test_a_multi_part_public_suffix_is_not_itself_a_registrable_domain() -> None:
    """The case the ticket names, and the one a split on the first dot gets wrong.

    `co.uk` is a public suffix: nobody registers it, so it is never the answer.
    `example.co.uk` is what somebody registered.
    """
    suffixes = committed()

    assert suffixes.registrable_domain("example.co.uk") == "example.co.uk"
    assert suffixes.registrable_domain("www.example.co.uk") == "example.co.uk"
    assert suffixes.registrable_domain("shop.example.co.uk") == "example.co.uk"
    assert suffixes.registrable_domain("co.uk") is None
    assert suffixes.registrable_domain("uk") is None
    assert suffixes.registrable_domain("com") is None


def test_a_registrable_domain_is_one_label_more_than_its_public_suffix() -> None:
    """Exactly one label more, so everything deeper is a subdomain of the registration.

    `a.b.example.co.uk` and `example.co.uk` are one registration, and a grouping
    that reads the second as a different domain from the first splits a campaign
    on its own web layout.
    """
    suffixes = committed()

    assert suffixes.registrable_domain("a.b.example.co.uk") == "example.co.uk"
    assert suffixes.registrable_domain("a.b.c.example.co.uk") == "example.co.uk"


def test_other_multi_part_suffixes_from_the_list_agree() -> None:
    """Not just the one suffix the ticket names: the list is the list."""
    suffixes = committed()

    for host, expected in (
        ("example.com.au", "example.com.au"),
        ("www.example.com.au", "example.com.au"),
        ("example.co.jp", "example.co.jp"),
        ("bbc.co.uk", "bbc.co.uk"),
        ("example.gov.uk", "example.gov.uk"),
    ):
        assert suffixes.registrable_domain(host) == expected


def test_the_private_section_of_the_list_is_applied() -> None:
    """Hosting platforms are public suffixes too, and this is where it earns its keep.

    `attacker.github.io` is a registration on GitHub Pages. Reading the ICANN
    section alone would report `github.io`, which is a suffix nobody registered
    and which would collapse every Pages site in the Corpus into one domain.
    """
    suffixes = committed()

    assert suffixes.registrable_domain("attacker.github.io") == "attacker.github.io"
    assert suffixes.registrable_domain("a.b.attacker.github.io") == "attacker.github.io"
    assert suffixes.registrable_domain("github.io") is None
    assert (
        suffixes.registrable_domain("bucket.s3.amazonaws.com")
        == "bucket.s3.amazonaws.com"
    )
    assert suffixes.registrable_domain("blogspot.com") is None


def test_a_wildcard_rule_covers_every_label_under_it() -> None:
    """`*.ck` is in the list: every `something.ck` is a suffix, not a registration.

    `foo.ck` therefore has no registrable domain, while `bar.foo.ck` does. A list
    that stores only whole-label rules gets both of these wrong.
    """
    suffixes = committed()

    assert suffixes.registrable_domain("foo.ck") is None
    assert suffixes.registrable_domain("bar.foo.ck") == "bar.foo.ck"


def test_an_exception_rule_undercuts_the_wildcard_beneath_it() -> None:
    """`!www.ck` is in the list: `www.ck` is registrable even though `*.ck` matches.

    The exception wins over the longer wildcard that also matches, and its public
    suffix is the rule with its leftmost label removed.
    """
    suffixes = committed()

    assert suffixes.registrable_domain("www.ck") == "www.ck"
    assert suffixes.registrable_domain("foo.www.ck") == "www.ck"
    assert suffixes.registrable_domain("city.kawasaki.jp") == "city.kawasaki.jp"
    assert suffixes.registrable_domain("a.city.kawasaki.jp") == "city.kawasaki.jp"


def test_an_unlisted_tld_is_its_own_public_suffix() -> None:
    """The prevailing rule is `*` when nothing matches, and the Corpus relies on it.

    `.example` is reserved by RFC 2606 and has no rules, so the implicit single
    label rule applies: the TLD is the public suffix and the registration is the
    one label in front of it. That is what collapses the campaign's mirror onto the
    campaign's own domain, which is the case the Corpus plants a host to require.

    The limit this leaves, stated rather than hidden: with no rule to say otherwise,
    the registration under an unlisted TLD is always the last two labels. So a host
    three or more labels deep is truncated — `a.b.c.vantage-ledger.example` is
    reported as `vantage-ledger.example`, which is not what a registrar would say,
    and where a listed suffix absorbs the depth instead. Every host the Corpus
    carries is at most one label deep, so no host in it is affected, and a real
    Corpus under a real TLD has a rule for the TLD that says otherwise.
    """
    suffixes = committed()

    assert suffixes.registrable_domain("vantage-ledger.example") == "vantage-ledger.example"
    assert suffixes.registrable_domain("mirror.vantage-ledger.example") == (
        "vantage-ledger.example"
    )
    assert suffixes.registrable_domain("example") is None
    assert suffixes.registrable_domain("a.b.c.vantage-ledger.example") == (
        "vantage-ledger.example"
    )


def test_the_near_miss_pairs_stay_apart() -> None:
    """Two domains one character apart are two registrations, and stay two.

    This is the whole reason the pairs are planted: any rule that reaches past the
    registrable domain, or that compares names rather than registration, joins a
    recruitment firm to a signals desk and a warehouse to an annotation job. The
    mirror is the control for the other direction — the one case where two
    hostnames must collapse to one domain.
    """
    suffixes = committed()

    registrations = {suffixes.registrable_domain(host) for host in CORPUS_HOSTS}
    # Sixteen hosts, fifteen registrations: only the mirror collapses.
    assert len(registrations) == len(CORPUS_HOSTS) - 1
    assert registrations == set(CORPUS_HOSTS.values())

    assert suffixes.registrable_domain("vantage-ledger.example") != suffixes.registrable_domain(
        "vantage-ledgers.example"
    )
    assert suffixes.registrable_domain("signal-harbor.example") != suffixes.registrable_domain(
        "signal-harbour.example"
    )


def test_hosts_are_normalised_before_they_are_looked_up() -> None:
    """Case, a trailing root dot, and an empty host.

    A URL host is not required to be tidy, and none of these differences change
    who registered what, so none of them may change the answer.
    """
    suffixes = committed()

    assert suffixes.registrable_domain("EXAMPLE.CO.UK") == "example.co.uk"
    assert suffixes.registrable_domain("Www.Example.Co.Uk") == "example.co.uk"
    assert suffixes.registrable_domain("example.co.uk.") == "example.co.uk"
    assert suffixes.registrable_domain("VANTAGE-LEDGER.EXAMPLE") == (
        "vantage-ledger.example"
    )
    assert suffixes.registrable_domain("") is None


def test_an_oversized_or_undecodable_host_is_reported_rather_than_guessed() -> None:
    """A host that cannot be normalised has no answer, and `None` is that answer.

    A phishing link does carry a 300-character label, and it must not be reported
    as some truncated domain.
    """
    suffixes = committed()

    assert to_ascii(f"{'a' * 300}.example") is None
    assert suffixes.registrable_domain(f"{'a' * 300}.example") is None
    assert to_ascii("\ud800.example") is None
    assert to_ascii(None) is None


def test_an_internationalised_host_matches_by_its_punycode_form() -> None:
    """The list ships its internationalised rules in Unicode; hosts arrive in either form.

    Both spellings name the same registration, so both must reach the same rule and
    return the same punycode, or a campaign registered under one spelling would be
    split in two by a redirect between the two.
    """
    suffixes = committed()

    assert to_ascii("пример.xn--p1ai") == "xn--e1afmkfd.xn--p1ai"
    assert to_ascii("пример.рф") == "xn--e1afmkfd.xn--p1ai"
    assert suffixes.registrable_domain("shop.пример.рф") == "xn--e1afmkfd.xn--p1ai"
    assert suffixes.registrable_domain("shop.xn--e1afmkfd.xn--p1ai") == "xn--e1afmkfd.xn--p1ai"


def test_the_rule_shapes_the_list_uses_are_all_understood() -> None:
    """Parsed from a literal list, so this holds whatever the committed file does.

    A parser that mishandles one rule shape still returns a plausible answer for
    the corpus, because the corpus needs none of the three shapes. Each shape is
    therefore checked against a list written out here.
    """
    rules = PublicSuffixes.parse(
        "\n".join(
            (
                "// a comment, and a blank line, and a rule with a trailing comment",
                "",
                "com",
                "co.uk",
                "*.ck",
                "!www.ck",
                "*.jp",
                "!city.kawasaki.jp",
                "рф",
            )
        )
    )

    assert rules.registrable_domain("example.com") == "example.com"
    assert rules.registrable_domain("example.co.uk") == "example.co.uk"
    assert rules.registrable_domain("foo.ck") is None
    assert rules.registrable_domain("www.ck") == "www.ck"
    assert rules.registrable_domain("city.kawasaki.jp") == "city.kawasaki.jp"
    assert rules.registrable_domain("a.city.kawasaki.jp") == "city.kawasaki.jp"
    assert rules.registrable_domain("shop.рф") == "shop.xn--p1ai"
    # Nothing in this list matches, so the prevailing rule is `*` and the
    # registration is the last two labels.
    assert rules.registrable_domain("nothing.unlisted") == "nothing.unlisted"
    assert rules.registrable_domain("a.b.nothing.unlisted") == "nothing.unlisted"
    assert rules.plain_rules == 3
    assert rules.wildcard_rules == 2
    assert rules.exception_rules == 2


def test_the_list_is_parsed_from_bytes_that_are_not_guessed_at() -> None:
    """Comments, blank lines, and CRLF are all things a published file arrives with.

    A rule parsed with its comment still attached matches nothing, and a rule
    parsed with a trailing carriage return matches nothing either. Both failures
    are silent, so both are pinned.
    """
    rules = PublicSuffixes.parse("// header\r\n\r\nco.uk // trailing\r\n")

    assert rules.plain_rules == 1
    assert rules.registrable_domain("example.co.uk") == "example.co.uk"


def test_fetching_the_list_writes_the_rules_and_where_they_came_from(tmp_path: Path) -> None:
    """Seam under test: the command, observed through the two files it writes."""
    list_path = tmp_path / "public_suffix_list.dat"
    provenance_path = tmp_path / "provenance.jsonl"

    exit_code = main(
        [
            "fetch-suffix-list",
            "--list",
            str(list_path),
            "--provenance",
            str(provenance_path),
        ]
    )

    assert exit_code == 0
    assert list_path.read_text(encoding="utf-8").startswith("// This Source Code Form")

    records = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["licence"] == "Mozilla Public License 2.0"
    assert record["source_url"].startswith("https://publicsuffix.org/list/")
    assert record["plain_rules"] > 9_000
    assert record["wildcard_rules"] > 0
    assert record["exception_rules"] > 0
    assert record["icann_rules"] > 0
    assert record["private_rules"] > 0
    assert len(record["sha256"]) == 64
    assert record["list_bytes"] == list_path.stat().st_size


def test_fetching_the_list_refuses_to_replace_a_list_that_is_already_there(
    tmp_path: Path,
) -> None:
    """A new release moves every registrable domain, so it must be asked for.

    Same argument as the CAFC cache, and the same guard: a quietly replaced list
    would change what every downstream Signal is computed against without
    anything in the repository recording that it had moved.
    """
    list_path = tmp_path / "public_suffix_list.dat"
    provenance_path = tmp_path / "provenance.jsonl"
    list_path.write_text("com\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "fetch-suffix-list",
                "--list",
                str(list_path),
                "--provenance",
                str(provenance_path),
            ]
        )

    assert list_path.read_text(encoding="utf-8") == "com\n"
    assert not provenance_path.exists()


def test_the_committed_list_and_its_provenance_are_what_the_command_writes() -> None:
    """Held to the published file the same way the Corpus is held to its generator.

    The one thing that is not re-derived here is the rules themselves, which come
    from the network. What is held is that the provenance describes the committed
    bytes, so a reader can check the digest against publicsuffix.org themselves
    and know which release this is.
    """
    assert COMMITTED_LIST.exists()
    assert COMMITTED_PROVENANCE.exists()

    records = [
        json.loads(line)
        for line in COMMITTED_PROVENANCE.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]

    assert record["list_bytes"] == COMMITTED_LIST.stat().st_size
    assert record["sha256"] == _sha256(COMMITTED_LIST.read_bytes())

    suffixes = committed()
    assert suffixes.plain_rules == record["plain_rules"]
    assert suffixes.wildcard_rules == record["wildcard_rules"]
    assert suffixes.exception_rules == record["exception_rules"]


def test_the_committed_list_writes_line_feeds_whatever_the_checkout_did() -> None:
    """Git can check a text file out with CRLF; a rule with a carriage return matches
    nothing, so this file is held to LF by `.gitattributes` and pinned here."""
    assert b"\r\n" not in COMMITTED_LIST.read_bytes()
    assert COMMITTED_LIST.read_bytes().endswith(b"\n")


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
