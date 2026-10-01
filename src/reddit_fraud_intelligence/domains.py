"""What every link in the Corpus registers, reported per post.

The first thing the pipeline computes, and the input to every grouping decision
after it: ADR-0005 puts two accounts in the same Campaign Candidate only if they
share a registrable domain, so a domain that is wrong here is wrong in a way that
cannot be seen from the output of whatever ran on top of it. Which is why the rule
for reading a link is one rule, and why a link that names no registration is
reported rather than dropped.

A link is read for a host and the host is asked for its registrable domain. The
scheme is never consulted, because the scheme is transport and the registration is
not: `https://`, `http://`, `ftp://`, and a protocol-relative `//` all name the
same domain, and a scam operator chooses the scheme as freely as the path.
Extracting two schemes and excluding the rest would be easier to describe and
would quietly lose infrastructure, so the rule is about whether there is a host and
not about which scheme there is.

Every link produces a row. A link that names no registration is not an error and is
not a gap in the output: it comes back with the reason it could not be read, and the
reason is one of a closed set, so the next step can decide what to do about each
kind rather than discovering the kinds as it goes. A post that links nothing gets a
row too, listing no domains — it is the floor of the grouping case, since there is
no infrastructure on it for anything to join it by, and a file that only listed
posts that linked something would hide exactly the posts worth checking.

Both outputs are generated. The JSON Lines file is what the next step reads; the
report is what a reviewer reads, and it is generated from the same rows rather than
written beside them, so the two cannot disagree.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from reddit_fraud_intelligence.corpus import CorpusItem
from reddit_fraud_intelligence.jsonl import JsonObject, write_lines
from reddit_fraud_intelligence.suffixes import PublicSuffixes, is_address, to_ascii


class Unresolved(StrEnum):
    """Why a link names no registration. A closed set, so nothing is merely unexplained.

    Each is a different fault with a different fix, and the pipeline needs to tell
    them apart: a relative link is a broken link, a scheme with no host is a
    Contact Point that ticket #11 has to read, a bare public suffix is a link
    somebody has no domain for, and an address is infrastructure that is not a
    domain at all.
    """

    RELATIVE = "relative"
    NO_HOST = "no_host"
    PUBLIC_SUFFIX = "public_suffix"
    ADDRESS = "address"
    UNDECODABLE_HOST = "undecodable_host"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class LinkDomain:
    """One link and what it names. `domain` is None exactly when `unresolved` is not."""

    link: str
    scheme: str
    host: str | None
    domain: str | None
    unresolved: Unresolved | None


@dataclass(frozen=True, slots=True)
class PostDomains:
    """One post and every link it carries, resolved."""

    post_id: str
    account: str
    links: tuple[LinkDomain, ...]

    @property
    def domains(self) -> tuple[str, ...]:
        """The registrations this post links to, distinct and sorted.

        A set, not a list of URLs: a post that links a link-in-bio page and the
        site behind it touches one registration, and printing it twice would
        overstate how much infrastructure a post reaches.
        """
        return tuple(sorted({link.domain for link in self.links if link.domain is not None}))


@dataclass(frozen=True, slots=True)
class DomainFacts:
    """What reading the two inputs establishes, stated as a claim about bytes."""

    accounts: int
    corpus_path: str
    corpus_sha256: str
    domains: int
    links: int
    posts: int
    resolved_links: int
    rules_sha256: str
    suffix_list_bytes: int
    suffix_list_path: str
    suffix_list_sha256: str
    unresolved_links: int


def resolve_link(link: str, suffixes: PublicSuffixes) -> LinkDomain:
    """Read one link for a host, and the host for a registration.

    Tolerant by construction: a link that will not parse, carries no host, or names
    something that is not a name at all comes back with a reason, because a post
    that carries a broken link is still a post whose other links count.
    """
    try:
        parsed = urlparse(link)
        host = parsed.hostname
    except ValueError:
        return LinkDomain(
            link=link, scheme="", host=None, domain=None, unresolved=Unresolved.MALFORMED
        )

    scheme = parsed.scheme
    if host is None:
        # A scheme with no authority names something other than a page: a mail
        # address, a telephone number, a script. No scheme at all means the link is
        # a path, which is a broken link rather than a different kind of contact.
        return LinkDomain(
            link=link,
            scheme=scheme,
            host=None,
            domain=None,
            unresolved=Unresolved.NO_HOST if scheme else Unresolved.RELATIVE,
        )

    reason = _unreasoned(host, suffixes)
    if reason is not None:
        return LinkDomain(link=link, scheme=scheme, host=host, domain=None, unresolved=reason)

    domain = suffixes.registrable_domain(host)
    return LinkDomain(link=link, scheme=scheme, host=host, domain=domain, unresolved=None)


def _unreasoned(host: str, suffixes: PublicSuffixes) -> Unresolved | None:
    """The reason this host names no registration, or None when it names one.

    Split from `resolve_link` so the two questions stay apart: a host is offered to
    the list only once it has been shown to be readable as a name, because a
    truncated reading of a host is a wrong domain that looks right.
    """
    if is_address(host):
        return Unresolved.ADDRESS
    if to_ascii(host) is None:
        return Unresolved.UNDECODABLE_HOST
    if suffixes.registrable_domain(host) is None:
        return Unresolved.PUBLIC_SUFFIX
    return None


def post_domains(
    items: Iterable[CorpusItem], suffixes: PublicSuffixes
) -> tuple[PostDomains, ...]:
    """Every post, and every link on it, resolved. Posts with no links are included."""
    return tuple(
        PostDomains(
            post_id=item.post_id,
            account=item.account,
            links=tuple(resolve_link(link, suffixes) for link in item.links),
        )
        for item in items
    )


def survey(
    corpus_path: Path, list_path: Path, rows: Sequence[PostDomains], suffixes: PublicSuffixes
) -> DomainFacts:
    """Derive every figure the report prints, from the rows and the two files."""
    corpus_bytes = corpus_path.read_bytes()
    list_bytes = list_path.read_bytes()
    links = [link for row in rows for link in row.links]
    resolved = [link for link in links if link.domain is not None]

    return DomainFacts(
        accounts=len({row.account for row in rows}),
        corpus_path=corpus_path.as_posix(),
        corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
        domains=len({link.domain for link in resolved}),
        links=len(links),
        posts=len(rows),
        resolved_links=len(resolved),
        rules_sha256=hashlib.sha256("\n".join(suffixes.rules()).encode("utf-8")).hexdigest(),
        suffix_list_bytes=len(list_bytes),
        suffix_list_path=list_path.as_posix(),
        suffix_list_sha256=hashlib.sha256(list_bytes).hexdigest(),
        unresolved_links=len(links) - len(resolved),
    )


def write_post_domains(path: Path, rows: Sequence[PostDomains]) -> None:
    def objects() -> Iterator[JsonObject]:
        for row in rows:
            record = asdict(row)
            record["links"] = [
                {**asdict(link), "unresolved": None if link.unresolved is None else link.unresolved.value}
                for link in row.links
            ]
            record["domains"] = list(row.domains)
            yield record

    write_lines(path, objects())


def render_report(facts: DomainFacts, rows: Sequence[PostDomains]) -> str:
    """The reader-facing report, generated from the rows rather than written beside them."""
    registrations = _registration_table(rows)
    posts = "\n".join(_post_section(row) for row in rows)

    return f"""# Registrable domains in the Corpus

Generated by `rfi post-domains` from the Corpus file and the published Public
Suffix List. Do not edit it by hand — a test holds this file to what those two
produce, and re-running the command rewrites it byte for byte.

## What was read

| | |
|---|---|
| Posts | {facts.posts} |
| Accounts | {facts.accounts} |
| Links | {facts.links} |
| Resolved to a registrable domain | {facts.resolved_links} |
| Unresolvable | {facts.unresolved_links} |
| Distinct registrable domains | {facts.domains} |
| Corpus | `{facts.corpus_path}` |
| SHA-256 of the Corpus | `{facts.corpus_sha256}` |
| Public Suffix List | `{facts.suffix_list_path}` ({facts.suffix_list_bytes:,} bytes) |
| SHA-256 of the list | `{facts.suffix_list_sha256}` |
| SHA-256 of the rules, whatever the formatting | `{facts.rules_sha256}` |

## How a link is read

A link is read for a host, and the host is asked for its registrable domain: the
public suffix, plus the one label in front of it. A host is not a domain, so
`mirror.vantage-ledger.example` and `vantage-ledger.example` are one registration
here and not two.

**The scheme is never consulted.** The scheme is transport and the registration is
not, so `https://`, `http://`, `ftp://`, and a protocol-relative `//` all name the
same domain. Extracting `http` and `https` only would be easier to describe and
would quietly lose infrastructure, because an operator picks the scheme as freely
as the path.

**Multi-part public suffixes are resolved from the published list, not from a
constant.** `co.uk` is a public suffix: nobody registers it, so it is never the
answer, and `example.co.uk` is what somebody registered. A rule written by hand
would be wrong the first time a suffix appeared that nobody had thought of, so the
list is published at `{facts.suffix_list_path}` and read from there. Its private
section is applied too: a hosting platform hands out subdomains of a name it owns,
and `attacker.github.io` is a registration rather than a subdomain of a suffix.

Every link produces a row, including a post that links nothing. A link that names
no registration is not dropped: it is reported with the reason, which is one of
{", ".join(f"`{reason.value}`" for reason in Unresolved)}.

## The registrations in the Corpus

Distinct registrations and how much of the Corpus reaches each one. This is a count
of links, not a grouping: deciding which accounts belong together is the next step
(ADR-0005) and nothing here anticipates it.

{registrations}

## Every post, and every link it carries

{posts}
"""


def _registration_table(rows: Sequence[PostDomains]) -> str:
    counts: dict[str, int] = {}
    hosts: dict[str, set[str]] = {}
    for row in rows:
        for link in row.links:
            if link.domain is None:
                continue
            counts[link.domain] = counts.get(link.domain, 0) + 1
            if link.host is not None:
                hosts.setdefault(link.domain, set()).add(link.host)

    if not counts:
        return "The Corpus links nothing that names a registration."
    return "\n".join(
        f"| `{domain}` | {counts[domain]} | {len(hosts[domain])} |"
        for domain in sorted(counts)
    ).join(("| Registrable domain | Links | Hosts |\n| --- | ---: | ---: |\n", "\n"))


def _post_section(row: PostDomains) -> str:
    heading = f"### `{row.post_id}` · `{row.account}`\n"
    if not row.links:
        return f"{heading}\nNo links. Nothing to resolve, and nothing for a grouping to join it by.\n"

    lines = ["| Link | Host | Registrable domain |", "| --- | --- | --- |"]
    for link in row.links:
        host = f"`{link.host}`" if link.host is not None else "—"
        if link.domain is not None:
            resolved = f"**`{link.domain}`**"
        else:
            resolved = f"**unresolvable** — {link.unresolved.value if link.unresolved else 'unresolved'}"
        lines.append(f"| `{link.link}` | {host} | {resolved} |")
    return heading + "\n" + "\n".join(lines) + "\n"
