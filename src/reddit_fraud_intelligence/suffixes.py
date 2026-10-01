"""Registrable domains, and the public suffix list that decides them.

ADR-0005 puts two accounts in the same Campaign Candidate only if they share a
registrable domain, which makes this the first thing the pipeline has to get
right: every grouping decision downstream is read off what this returns. A host is
not a domain. `mirror.vantage-ledger.example` and `vantage-ledger.example` are two
hostnames and one registration, and the Corpus plants both so that a reader has to
conclude that rather than be told it.

The rule for what a registration is comes from the Public Suffix List: the public
suffix is the part of a host nobody can register, and the registrable domain is
that plus the one label in front of it. A hardcoded list of multi-part suffixes
would be shorter and would be wrong the first time a suffix appeared that nobody
had thought of, so the list is published at `data/public-suffix/` and parsed here.
`rfi fetch-suffix-list` is the only command that needs the network to do with it,
and it refuses to replace a list that is already committed for the same reason
`fetch-cafc` does: a new release moves every registrable domain in the Corpus and
would change every Signal computed from one without the repository recording that
it had moved.

The list has three kinds of rule and all three are honoured, because a parser that
mishandles one of them still returns a plausible answer for this Corpus, which
needs none of them: a whole-label rule (`co.uk`), a wildcard (`*.ck`, every label
under it is a suffix), and an exception (`!www.ck`, which undercuts the wildcard
beneath it). Its private section is applied too, so `attacker.github.io` is a
registration and not a subdomain of a suffix nobody registered.

Every answer that cannot be reached is `None` â€” a host that is a public suffix
with no label in front of it, and a host that cannot be read at all. A link whose
host cannot be read is reported as unresolvable by the caller, never guessed at
and never dropped.
"""

from __future__ import annotations

import encodings.idna
import hashlib
import ipaddress
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from reddit_fraud_intelligence.jsonl import JsonObject, write_lines

SOURCE_URL = "https://publicsuffix.org/list/public_suffix_list.dat"
LICENCE = "Mozilla Public License 2.0"
LICENCE_URL = "https://mozilla.org/MPL/2.0/"
USER_AGENT = "reddit-fraud-intelligence/0.1 (registrable domains for Campaign Candidates)"

# publicsuffix.org marks the two halves of the list with comments rather than
# anything the format requires, so they are read from the text. Recorded in the
# provenance because the halves answer different questions: the ICANN section is
# what somebody registered through a registrar, and the private section is the
# hosting platforms that hand out subdomains of names they own. Grouping is
# affected by both, so both are parsed.
ICANN_MARKER = "===BEGIN ICANN DOMAINS==="
PRIVATE_MARKER = "===BEGIN PRIVATE DOMAINS==="

_COMMENT = "//"
_WILDCARD = "*"
_EXCEPTION = "!"
_CHUNK_BYTES = 1 << 20

# RFC 1035. A name longer than this is not a name, and truncating one to fit is how
# two unrelated hosts end up reported as the same domain.
_MAX_HOST_CHARS = 253
_MAX_LABEL_CHARS = 63


@dataclass(frozen=True, slots=True)
class ListFacts:
    """What reading the published list establishes, stated as a claim about bytes."""

    exception_rules: int
    icann_rules: int
    licence: str
    licence_url: str
    list_bytes: int
    plain_rules: int
    private_rules: int
    rules_sha256: str
    sha256: str
    source_url: str
    wildcard_rules: int


@dataclass(frozen=True, slots=True)
class PublicSuffixes:
    """A parsed Public Suffix List, asked for the registrable domain of a host.

    Immutable, and built once and shared. The parse is the expensive part and the
    lookup is a walk over at most a few dozen labels, so the whole list is held in
    three sets and every question about a host is answered from those. The three
    sets are the rules and nothing else; the counts and the digest are derived, so
    a rule cannot be in the list and missing from the figures the provenance
    records.
    """

    plain: frozenset[str]
    wildcards: frozenset[str]
    exceptions: frozenset[str]

    @property
    def plain_rules(self) -> int:
        return len(self.plain)

    @property
    def wildcard_rules(self) -> int:
        return len(self.wildcards)

    @property
    def exception_rules(self) -> int:
        return len(self.exceptions)

    @classmethod
    def parse(cls, text: str) -> PublicSuffixes:
        """Read the list as published. Comments, blank lines, and CRLF are all
        things the file arrives with, and a rule that keeps any of them matches
        nothing â€” silently, which is why the parse is not a `split`."""
        plain: set[str] = set()
        wildcards: set[str] = set()
        exceptions: set[str] = set()

        for raw in text.splitlines():
            rule = raw.split(_COMMENT, 1)[0].strip()
            if not rule:
                continue
            if rule.startswith(_EXCEPTION):
                exceptions.add(cls._canonical(rule[1:]))
            elif rule.startswith(f"{_WILDCARD}."):
                wildcards.add(cls._canonical(rule[2:]))
            else:
                plain.add(cls._canonical(rule))

        return cls(frozenset(plain), frozenset(wildcards), frozenset(exceptions))

    @classmethod
    def read(cls, path: Path) -> PublicSuffixes:
        return cls.parse(path.read_text(encoding="utf-8"))

    def rules(self) -> Iterator[str]:
        """Every rule, as the list spells it, in a stable order.

        A digest over this is a digest over the meaning of the list rather than
        over its formatting, so reformatting the file does not look like a new
        release and a changed rule always does.
        """
        for rules, prefix in (
            (self.plain, ""),
            (self.wildcards, f"{_WILDCARD}."),
            (self.exceptions, _EXCEPTION),
        ):
            for rule in sorted(rules):
                yield f"{prefix}{rule}"

    def rules_digest(self) -> str:
        """SHA-256 of the rules rather than of the file they arrived in.

        Both digests are recorded, and they answer different questions. The one
        over the bytes is what a reader compares against publicsuffix.org. This one
        changes when a rule changes and not when the file is reformatted, so it is
        the one to watch when a grouping result moves.
        """
        return hashlib.sha256("\n".join(self.rules()).encode("utf-8")).hexdigest()

    def registrable_domain(self, host: str | None) -> str | None:
        """The domain somebody registered, or `None` when there is not one.

        `None` covers both a host that is nothing but a public suffix â€” `co.uk`,
        `uk`, `github.io` â€” and a host that cannot be read at all. The two are
        different failures and the caller reports them differently, so nothing here
        distinguishes them: a link with a host that cannot be read has no domain,
        and a link to a bare public suffix has no domain either, and both are
        reported as unresolvable rather than resolved to something plausible.
        """
        ascii_host = to_ascii(host)
        if ascii_host is None:
            return None
        labels = ascii_host.split(".")
        suffix_length = self._public_suffix_length(labels)
        if suffix_length >= len(labels):
            return None
        return ".".join(labels[len(labels) - suffix_length - 1 :])

    def _public_suffix_length(self, labels: list[str]) -> int:
        """How many labels, counting from the right, are the public suffix.

        An exception rule beats every other rule that matches, and its public
        suffix is the rule with its leftmost label removed â€” that is what makes
        `www.ck` a registration under a `*.ck` that would otherwise claim it.
        Otherwise the longest matching rule wins, whichever kind it is. With
        nothing matching, the prevailing rule is `*`: the TLD alone is the suffix.
        """
        for index in range(len(labels)):
            if ".".join(labels[index:]) in self.exceptions:
                return max(len(labels) - index - 1, 1)

        longest = 1
        for index in range(len(labels)):
            candidate = len(labels) - index
            if ".".join(labels[index:]) in self.plain:
                longest = max(longest, candidate)
            elif labels[index + 1 :] and ".".join(labels[index + 1 :]) in self.wildcards:
                longest = max(longest, candidate)
        return longest

    @staticmethod
    def _canonical(rule: str) -> str:
        """A rule in the same form a host will be in, so the two can be compared.

        The list ships its internationalised rules as Unicode while hosts arrive in
        either form, so both sides are put through the same conversion. A rule that
        cannot be converted matches no host, and dropping it loses nothing that
        could have matched.
        """
        ascii_rule = to_ascii(rule)
        return ascii_rule if ascii_rule is not None else rule


def to_ascii(host: str | None) -> str | None:
    """A host as lowercase ASCII, or `None` when it cannot be one.

    Case, a trailing root dot, and the choice between Unicode and punycode are all
    spelling, and none of them changes who registered what, so none of them may
    change the answer.

    A label that is not a name is the other half of the job. `urlparse` accepts a
    space, a `<`, or a percent sign in a host, and `idna.ToASCII` passes ASCII
    straight through after checking only its length, so without this a link that
    was never a URL comes back as a confident registration â€” and a grouping step
    downstream would then have an exact domain to join on, made out of it. A label
    is letters, digits, and hyphens, not starting or ending with a hyphen, and a
    name is at most 253 characters. Everything else returns `None`: an empty host, a
    lone surrogate, an address, and a host that is not a name all have no
    registration, and reporting any of them as a truncated domain would be a wrong
    answer that looks right.
    """
    if not host:
        return None

    trimmed = host.rstrip(".")
    if not trimmed or len(trimmed) > _MAX_HOST_CHARS:
        return None
    if is_address(trimmed):
        return None

    labels = []
    for label in trimmed.split("."):
        try:
            ascii_label = encodings.idna.ToASCII(label).decode("ascii").lower()
        except (UnicodeError, ValueError):
            return None
        if not _is_label(ascii_label):
            return None
        labels.append(ascii_label)
    return ".".join(labels)


def _is_label(label: str) -> bool:
    """Whether a label is one a domain name can be made of. `xn--` labels pass."""
    if not 0 < len(label) <= _MAX_LABEL_CHARS:
        return False
    if label.startswith("-") or label.endswith("-"):
        return False
    return all(
        character.isascii() and (character.isalnum() or character == "-")
        for character in label
    )


def is_address(host: str) -> bool:
    """Whether a host is an IP address, or something shaped like one.

    Public because the caller has to tell an address from a host it cannot read:
    both have no registrable domain, and they are different faults. An address is
    not a name anybody registered, and the list has no rule for one, so without
    this the implicit single-label rule turns `192.0.2.1` into the domain
    `192.0.2` and groups unrelated addresses that share their first three octets.

    A host whose labels are *all* digits is included even when it will not parse as
    an address, because the alternative is worse: `1.2.3.4.5` has no TLD, so the
    implicit rule would take `5` for one and report the domain `4.5`. No real TLD
    is numeric, so the rule costs nothing.
    """
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return all(label.isdigit() for label in host.split("."))
    return True


def download_list(destination: Path) -> int:
    """Cache the list as published. Returns the size in bytes.

    Written as bytes rather than decoded and re-encoded, so the committed file is
    the file publicsuffix.org serves and a digest of it can be checked against
    theirs.
    """
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        while chunk := response.read(_CHUNK_BYTES):
            written += len(chunk)
            handle.write(chunk)
    return written


def digest(path: Path) -> str:
    """SHA-256 of the published bytes, which is what the rules were read from.

    Not of the parsed rules: see `PublicSuffixes.rules_digest` for that one. Both
    are recorded, and a reader comparing against publicsuffix.org wants this.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def survey(path: Path) -> ListFacts:
    """Read the cached list once and derive every figure the project records.

    Computed from the bytes rather than transcribed, so a figure and the list
    cannot drift apart, and re-running over an unchanged file rewrites identical
    bytes.
    """
    payload = path.read_bytes()
    text = payload.decode("utf-8")
    suffixes = PublicSuffixes.parse(text)
    icann, private = _sections(text)

    return ListFacts(
        exception_rules=suffixes.exception_rules,
        icann_rules=icann,
        licence=LICENCE,
        licence_url=LICENCE_URL,
        list_bytes=len(payload),
        plain_rules=suffixes.plain_rules,
        private_rules=private,
        rules_sha256=suffixes.rules_digest(),
        sha256=digest(path),
        source_url=SOURCE_URL,
        wildcard_rules=suffixes.wildcard_rules,
    )


def _sections(text: str) -> tuple[int, int]:
    """How many rules sit in each half of the list. The marker comments are the only
    place the two halves are marked, so they are read from the text."""
    section = None
    counts = {None: 0, ICANN_MARKER: 0, PRIVATE_MARKER: 0}
    for raw in text.splitlines():
        if raw.startswith(_COMMENT):
            if ICANN_MARKER in raw:
                section = ICANN_MARKER
            elif PRIVATE_MARKER in raw:
                section = PRIVATE_MARKER
            continue
        if raw.strip() and section is not None:
            counts[section] += 1
    return counts[ICANN_MARKER], counts[PRIVATE_MARKER]


def write_provenance(path: Path, facts: ListFacts) -> None:
    def objects() -> Iterator[JsonObject]:
        yield asdict(facts)

    write_lines(path, objects())
