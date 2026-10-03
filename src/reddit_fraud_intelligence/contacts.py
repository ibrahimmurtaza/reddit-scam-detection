"""Contact Points: what the posts say to be reached at, per post, and which are shared.

ADR-0005 lets two accounts reach the same Campaign Candidate on a Contact Point as
well as on a Registrable Domain, and this is the step that finds them. A Contact
Point is off-platform infrastructure an operator controls and a reader can be sent
to, so a handle two accounts publish is shared infrastructure in everything but name.
It is also the edge a campaign that rotates domains per post is visible on and
invisible through, which is why it is worth extracting even where the domains already
group a pair of accounts.

Two kinds are read, and nothing else is looked for: an email address and a Telegram
handle. Both are read **literally**, from the post's own title, its own body, and the
links it carries. Nothing is repaired. A handle written with separators between its
characters, an address written as words rather than as an address, and a screenshot
of either are all in the wild and none of them is found here — that is ticket #15's
work, and until it lands every count this command produces is a lower bound, which is
why the limit is printed with the figures rather than left in the ticket that asked
for them. Reading loosely would be worse than reading narrowly here, because a Contact
Point is the one string in the Corpus that can join two accounts on nothing else, and
a value that was guessed is a grouping edge nobody checked.

A candidate that names no Contact Point is reported with which of eight reasons
applied, so nothing is dropped: a candidate quietly dropped is indistinguishable from
a post that named none, and the difference is the whole claim. The eight are each a
different fault with a different fix, which is what lets the next step decide what to
do about each kind rather than discovering the kinds as it goes.

A link is read for a Contact Point for the same reason a link is read for a
registration, and by the same rule: the scheme decides whether there is anything else
to read. A `mailto:` names an address rather than a page, and `post-domains` already
reports it as naming no registration and hands the case here; a link on Telegram's own
host names the username in its first path segment. Anything else names a page, and
the addresses inside a page's query string are not read — that is a link shortener's
business, not a Contact Point anybody published. A Telegram link to a channel rather
than to a person names no username at all, and is reported as such rather than
guessed at.

Two posts naming the same handle is the claim this command exists to make, and it is
reported separately from the per-post list because a Contact Point in one post says an
account published something, while the same one in two posts by two accounts is what
ADR-0005 counts as grounds for a proposal. The count is over accounts rather than over
posts, because the claim is about accounts: an account naming one handle in four posts
has not acquired a second reach.

Normalisation is minimal, and every spelling a value was written in is printed beside
it, so a reader can tell sharing from folding. A Telegram username is case-insensitive
and a full stop at the end of a sentence belongs to the sentence, so both are folded
into the value. An address is not treated the same way: the host is folded because
RFC 5321 says a domain is case-insensitive, and the local part is not, because it says
nothing of the sort. Folding both would merge `Desk@` and `desk@` into one mailbox
that may be two — this module inventing a grouping edge in order to be helpful, which
is the failure it exists to avoid.

Nothing is grouped here. ADR-0005 permits a Contact Point as a grouping edge and this
command builds none: the shared list is the evidence ticket #20 needs, and a grouping
edge that had not been measured would put an unmeasured input into the recovery figure
ADR-0004 is measured against. The output says so, because a shared list read as a list
of groupings is the more damaging of the two misreadings.

Nothing here reads the truth file or the Nuisance Structure manifest. The Corpus is
the whole input — no published list decides what a post says to be reached at — and
the truth file is joined by the evaluator after this has finished (ADR-0008).

Every Contact Point in this build is a Synthetic Entity, because the Corpus is
synthetic, and the output says which figures make that true rather than asserting it:
an address is one when its host sits under a TLD reserved for examples, and a handle
when it carries the `syn_` marker every Synthetic Entity in this Corpus carries. A
Corpus Provider is a swap, so the same line over real content would read `0 of N`,
which is what measuring it buys over promising it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.jsonl import JsonObject, write_lines

_HEADING = "Contact Points"
_SUBHEADING = """\
Every Contact Point a post's own title, its own body, and its links name, read literally \
and without repairing anything. Two kinds are looked for and nothing else. No grouping \
is built on any of it here."""

# The marker every Synthetic Entity in this Corpus carries, per the README. There is no
# reserved namespace for a Telegram username, so a handle is recognisable as synthetic
# only by carrying it, and the figure the output prints is a count of what carries it.
SYNTHETIC_MARKER = "syn_"

# RFC 2606's reserved TLDs. An address under one of them corresponds to no real host,
# which is what makes it a Synthetic Entity rather than somebody's mailbox.
RESERVED_TLDS = frozenset({"example", "invalid", "localhost", "test"})

# Telegram's own hosts, so a link to a profile is read for the username in it rather
# than for a registration nobody registered. A page on any other host is not read.
TELEGRAM_HOSTS = frozenset({"t.me", "telegram.me"})

# The paths on a Telegram host that name a channel or an invite rather than a person.
# They are the two shapes a scam post uses to be joined without naming anyone.
_INVITE_SEGMENTS = frozenset({"joinchat", "share"})

# RFC 5321's limit on an address. An address longer than this is not one, and
# truncating it would produce a plausible address that belongs to nobody.
_MAX_ADDRESS_CHARS = 254

# Telegram's own limits on a username: five to thirty-two characters, starting with a
# letter and holding only letters, digits, and underscores. Read from the service
# rather than invented, because a rule that invented them would read a different set
# of strings and there would be no way for a reader to check.
_MIN_HANDLE_CHARS = 5
_MAX_HANDLE_CHARS = 32

# The run a candidate is read from: a maximal stretch of the characters an address or a
# username can be made of, containing at least one `@`. The part before the `@` allows
# the punctuation an address's local part allows, the part after it does not, because a
# username cannot hold one. Scanning starts at every position, so a candidate in the
# middle of a sentence is found without the rule knowing anything about sentences.
#
# Both halves stop at a space, which is what makes `write to me at @ or ring` two
# candidates rather than one: the run after the `@` is empty, and a candidate whose
# name is empty is reported rather than skipped.
_CANDIDATE = re.compile(r"[A-Za-z0-9_%+.\-]*@[A-Za-z0-9_.\-]*")

# A Telegram username, once the candidate has been classified as one.
_USERNAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")

# A host label, judged the way `suffixes.py` judges one: letters, digits, and hyphens,
# and a hyphen at neither end. Reusing the rule rather than inventing a second one is
# what keeps `intake@vantage-ledger.example` reading as an address and
# `nobody@local_host` not reading as one.
_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\Z")

# Separators that cannot end either kind of identifier and are the ordinary punctuation
# of the sentence a Contact Point was found in. Trimmed rather than refused, because a
# handle at the end of a sentence is written `@someone.` and reporting the run as it
# stands would make it a different username from the same one in the next post — a
# silent wrong answer that splits a shared Contact Point in two. What was written is
# reported beside what it was read as, so the trim is a decision a reader can see.
_TRAILING_SEPARATORS = ".-"

_FIELDS = ("title", "body")


class ContactKind(StrEnum):
    """The two kinds this build reads, and the only two the output may name.

    The glossary's Contact Point also covers a Discord ID and a crypto wallet. Neither
    is read here, and the enum is the contract between the rules below and the report
    above them: a kind the report does not name cannot appear in the file beside it,
    which is what keeps the prose and the data describing the same set.
    """

    EMAIL = "email"
    TELEGRAM = "telegram"


class Unread(StrEnum):
    """Why a candidate names no Contact Point. A closed set, so nothing is merely
    unexplained.

    Each is a different fault with a different fix, and the next step needs to tell
    them apart: text with no `@` names no Contact Point by construction and is not a
    candidate at all, whereas a candidate that cannot be read is a write somebody got
    wrong or a disguise somebody chose, and only the second is worth arguing about.
    """

    ADDRESS_TOO_LONG = "address_too_long"
    HANDLE_CHARACTER = "handle_character"
    HANDLE_TOO_LONG = "handle_too_long"
    HANDLE_TOO_SHORT = "handle_too_short"
    INVITE_LINK = "invite_link"
    MALFORMED = "malformed"
    NO_DOMAIN = "no_domain"
    NO_NAME = "no_name"


@dataclass(frozen=True, slots=True)
class Match:
    """One Contact Point a post names, and the exact text that named it.

    `found_as` and `value` are kept apart because they answer different questions and a
    reader needs both: one is where to look in the post, the other is what two posts
    are compared on. They differ whenever the post wrote the value differently — case,
    or a trailing full stop that belonged to the sentence — and printing only the
    second would make it impossible to tell a shared Contact Point from a case-folded
    coincidence.

    `field` is carried because a handle in the title is not a handle in the fourth
    paragraph, and a breakdown that did not say which would be making one of them.
    """

    kind: ContactKind
    value: str
    found_as: str
    field: str


@dataclass(frozen=True, slots=True)
class UnreadCandidate:
    """One candidate that names no Contact Point, and which of the eight faults it was."""

    found_as: str
    field: str
    reason: Unread


@dataclass(frozen=True, slots=True)
class PostContacts:
    """One post and every Contact Point it names, with what it could not read."""

    post_id: str
    account: str
    matches: tuple[Match, ...]
    unread: tuple[UnreadCandidate, ...]

    @property
    def contact_points(self) -> tuple[str, ...]:
        """The Contact Points this post names, distinct and sorted.

        A set, not a count of how often it wrote one: a post that names a handle in its
        title and its body reaches one Contact Point, and listing it twice would
        overstate how much shared infrastructure the post touches.
        """
        return tuple(sorted({match.value for match in self.matches}))

    @property
    def posted(self) -> bool:
        """Whether this post named a Contact Point at all."""
        return bool(self.matches)


@dataclass(frozen=True, slots=True)
class SharedContact:
    """One Contact Point, and every place in the Corpus that names it.

    Sharing is counted over accounts and not over posts, because the claim being made
    is about accounts: an account that names one handle in four posts has not acquired
    a second reach, and a count of posts would report it as shared and let a proposal
    be made on one account's own repetition.

    `spellings` is every distinct string the Corpus wrote for this value, which is what
    lets a reader tell sharing from case folding.
    """

    kind: ContactKind
    value: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]
    spellings: tuple[str, ...]

    @property
    def shared(self) -> bool:
        return len(self.accounts) > 1


@dataclass(frozen=True, slots=True)
class ContactFacts:
    """What reading the Corpus establishes, stated as a claim about bytes."""

    accounts: int
    addresses: int
    addresses_reserved: int
    corpus_path: str
    corpus_sha256: str
    handles: int
    handles_marked: int
    matches: int
    points: int
    posts: int
    posts_with_points: int
    shared: int
    synthetic: int
    unread: int


@dataclass(frozen=True, slots=True)
class ContactPoints:
    """Everything one run establishes, so the table and the file cannot disagree.

    `points` is keyed by `(kind, value)` rather than by the value alone: an address
    and a handle are different Contact Points that happen to be different strings, and
    a run that merged them would report one reach where there are two.
    """

    facts: ContactFacts
    points: Mapping[tuple[ContactKind, str], SharedContact]
    rows: tuple[PostContacts, ...]

    @property
    def kinds(self) -> tuple[ContactKind, ...]:
        """The kinds this Corpus actually holds, in declaration order."""
        return tuple(kind for kind in ContactKind if any(k == kind for k, _ in self.points))


def _read_point(candidate: str, field: str) -> Match | UnreadCandidate:
    """One candidate, read by the rules of whichever kind it is, or the reason it is
    neither.

    Which kind it is decided by where the `@` stands rather than by asking both: a
    candidate that begins at its `@` is somebody's handle and nothing else, and a
    candidate with something before it is an address. Asking both and taking whichever
    succeeded would make `nobody@localhost` an address and `@nobody` a handle by
    accident of leniency, and would give a reader two rules where there is one.

    Always one or the other, never neither, so a caller cannot drop a candidate by
    forgetting to check: text is either a Contact Point or a fault, and a reader that
    has to remember that is a reader that will forget.
    """
    found_as = candidate
    trimmed = candidate.rstrip(_TRAILING_SEPARATORS)
    if trimmed.startswith("@"):
        return _handle(trimmed[1:], found_as, field)
    return _address(trimmed, found_as, field)


def _address(candidate: str, found_as: str, field: str) -> Match | UnreadCandidate:
    """One candidate read as an address, or the reason it names no host.

    The host has to be a host: at least two labels, none of them empty, each one a
    label. `nobody@localhost` is the case this exists for, and the tempting repairs —
    appending a TLD, or taking the last two labels of whatever follows the `@` — both
    produce an address that looks right, which is the one thing a Contact Point must
    never be, since it can join two accounts on nothing else.

    **The host is case-folded and the local part is not.** RFC 5321 makes the domain
    case-insensitive and says nothing of the sort about the local part, so
    `desk@vantage-ledger.example` and `DESK@VANTAGE-LEDGER.EXAMPLE` are one mailbox
    while `Desk@` and `desk@` may be two. Folding the whole address would merge those
    two on the strength of a rule RFC 5321 does not state, and a merged mailbox is a
    shared identifier between two accounts that share nothing — the one failure this
    module exists to avoid, arrived at by being helpful.
    """
    _, _, host = candidate.partition("@")
    labels = host.split(".")
    if len(labels) < 2 or any(not _LABEL.match(label) for label in labels):
        return UnreadCandidate(found_as, field, Unread.NO_DOMAIN)
    if len(candidate) > _MAX_ADDRESS_CHARS:
        return UnreadCandidate(found_as, field, Unread.ADDRESS_TOO_LONG)
    return Match(
        kind=ContactKind.EMAIL,
        value=f"{candidate.partition('@')[0]}@{host.lower()}",
        found_as=found_as,
        field=field,
    )


def _handle(name: str, found_as: str, field: str) -> Match | UnreadCandidate:
    """One candidate read as a Telegram username, or the reason it is not one.

    Telegram's own limits, read from the service: five to thirty-two characters,
    starting with a letter and holding only letters, digits, and underscores. The
    character rule is the one that carries the false positives, since `@name` is how
    every service writes a handle and this build cannot tell which service it came
    from — so a candidate that fails it is reported rather than dropped, and the
    output states the limit rather than leaving a reader to find it.

    Case is folded here, and folded here only, because Telegram usernames do not
    distinguish it: `@Syn_VantageLedger` and `@syn_vantageledger` are one username. The
    spelling each was written in travels with the value, so a reader can tell that from
    two accounts writing it identically.
    """
    if not name:
        return UnreadCandidate(found_as, field, Unread.NO_NAME)
    if len(name) < _MIN_HANDLE_CHARS:
        return UnreadCandidate(found_as, field, Unread.HANDLE_TOO_SHORT)
    if len(name) > _MAX_HANDLE_CHARS:
        return UnreadCandidate(found_as, field, Unread.HANDLE_TOO_LONG)
    if not _USERNAME.match(name):
        return UnreadCandidate(found_as, field, Unread.HANDLE_CHARACTER)
    return Match(
        kind=ContactKind.TELEGRAM, value=name.lower(), found_as=found_as, field=field
    )


def _from_link(link: str) -> Match | UnreadCandidate | None:
    """One link, read for the Contact Point it names rather than for a page.

    The same rule as every other link in the system: the scheme decides whether there
    is anything here that is not a page. A `mailto:` names an address, which is why
    `post-domains` reports it as naming no registration and hands the case to this
    command; a link on Telegram's own host names the username in its first path
    segment. Any other link names a page, and the addresses inside a page are not read
    — a shortener's query string is the shortener's business, and reading it would put
    Contact Points in the output that nobody published to be reached at.

    `None` means this link names no Contact Point at all, which is most links and is
    not a fault: a post that links a page and a contact has one Contact Point, not two.
    A link that will not parse is a different thing — there is no telling whether it
    named one — and it comes back as `malformed`, the same reason `post-domains` gives
    it, so the one case this command cannot classify is a case it reports.

    The whole link is reported as `found_as` rather than the Contact Point inside it,
    because the link is what a reader goes back to the post and finds.
    """
    try:
        parsed = urlparse(link)
        host = parsed.hostname
    except ValueError:
        return UnreadCandidate(link, "link", Unread.MALFORMED)

    if parsed.scheme == "mailto":
        return _address(parsed.path.strip(), link, "link")
    if host is None or host.lower() not in TELEGRAM_HOSTS:
        return None

    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments or segments[0].startswith("+") or segments[0] in _INVITE_SEGMENTS:
        return UnreadCandidate(link, "link", Unread.INVITE_LINK)
    return _handle(segments[0], link, "link")


def _record(
    read: Match | UnreadCandidate | None,
    matches: list[Match],
    unread: list[UnreadCandidate],
) -> None:
    """Put one read where it belongs, or put nothing anywhere for a link that names no
    Contact Point at all.

    Two collections rather than one, because a reviewer needs to be able to count the
    Contact Points and the faults separately and add the two up: a reader told
    "3 Contact Points" and "0 unread" has been told nothing about how many candidates
    there were in the first place. `None` is the only way nothing is added, and it is
    the only case where that is right — most links name a page, not a Contact Point.
    """
    if isinstance(read, Match):
        matches.append(read)
    elif read is not None:
        unread.append(read)


def _post(item: CorpusItem) -> PostContacts:
    """One post's own text and links, read. Both halves of the post are handed over
    rather than one of them: the text carries no links and the links carry no text, and
    a Contact Point is written in either."""
    matches: list[Match] = []
    unread: list[UnreadCandidate] = []

    for field_name, text_value in ((name, getattr(item, name)) for name in _FIELDS):
        for found in _CANDIDATE.finditer(text_value):
            _record(_read_point(found.group(0), field_name), matches, unread)

    for link in item.links:
        _record(_from_link(link), matches, unread)

    return PostContacts(
        post_id=item.post_id,
        account=item.account,
        matches=tuple(matches),
        unread=tuple(unread),
    )


def extract(items: Iterable[CorpusItem]) -> tuple[PostContacts, ...]:
    """Every post, and every Contact Point on it. Posts that name none are included."""
    return tuple(_post(item) for item in items)


def _reach(
    rows: Iterable[PostContacts],
) -> dict[tuple[ContactKind, str], SharedContact]:
    """What each Contact Point is reached by, counted once each way.

    Accounts and posts are counted per Contact Point rather than per post, because a
    post naming a handle in its title and its body reaches it once and reporting it
    twice would inflate the figure the sharing claim is made from.
    """
    accounts: dict[tuple[ContactKind, str], set[str]] = {}
    posts: dict[tuple[ContactKind, str], set[str]] = {}
    spellings: dict[tuple[ContactKind, str], set[str]] = {}

    for row in rows:
        for match in row.matches:
            key = (match.kind, match.value)
            accounts.setdefault(key, set()).add(row.account)
            posts.setdefault(key, set()).add(row.post_id)
            spellings.setdefault(key, set()).add(match.found_as)

    return {
        key: SharedContact(
            kind=key[0],
            value=key[1],
            accounts=tuple(sorted(accounts[key])),
            posts=tuple(sorted(posts[key])),
            spellings=tuple(sorted(spellings[key])),
        )
        for key in accounts
    }


def survey(
    rows: Sequence[PostContacts],
    points: Mapping[tuple[ContactKind, str], SharedContact],
    corpus_path: Path,
) -> ContactFacts:
    """Derive every figure the report prints, from the rows and the bytes behind them."""
    return ContactFacts(
        accounts=len({row.account for row in rows}),
        addresses=sum(1 for point in points if point[0] is ContactKind.EMAIL),
        addresses_reserved=sum(1 for point in points.values() if _is_reserved(point)),
        corpus_path=corpus_path.as_posix(),
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        handles=sum(1 for point in points if point[0] is ContactKind.TELEGRAM),
        handles_marked=sum(1 for point in points.values() if _is_marked(point)),
        matches=sum(len(row.matches) for row in rows),
        points=len(points),
        posts=len(rows),
        posts_with_points=sum(1 for row in rows if row.posted),
        shared=sum(1 for point in points.values() if point.shared),
        synthetic=sum(1 for point in points.values() if _is_synthetic(point)),
        unread=sum(len(row.unread) for row in rows),
    )


def _is_reserved(point: SharedContact) -> bool:
    """Whether an address sits under a TLD reserved for examples."""
    if point.kind is not ContactKind.EMAIL:
        return False
    _, _, host = point.value.rpartition("@")
    return host.rsplit(".", 1)[-1] in RESERVED_TLDS


def _is_marked(point: SharedContact) -> bool:
    """Whether a handle carries the marker every Synthetic Entity in this Corpus carries."""
    return point.kind is ContactKind.TELEGRAM and point.value.startswith(SYNTHETIC_MARKER)


def _is_synthetic(point: SharedContact) -> bool:
    """Whether a Contact Point corresponds to no real person, by either kind's marker.

    Two rules rather than one, because the two kinds have different namespaces: an
    address is synthetic when its host is under a reserved TLD, and a handle when it
    carries the `syn_` marker, since Telegram reserves no namespace of its own. Counted
    rather than asserted, because a Corpus Provider is a swap and the same line over
    real content has to read `0 of N`.
    """
    return _is_reserved(point) or _is_marked(point)


def contact_points(corpus_path: Path) -> ContactPoints:
    """Read the Corpus, and return every post's Contact Points and what they reach.

    One call, for the same reason the grouping is one call: the per-post rows, the
    shared list, and the figures are three views of one pass over the Corpus, and a
    caller that extracted and then asked for the sharing separately could end up
    printing one run's figures over another's results.

    The Corpus is read rather than generated, because the file is the boundary
    (ADR-0001, ADR-0008). No published list is read: nothing about what a post says to
    be reached at depends on what anybody registered.
    """
    corpus = read_corpus(corpus_path)
    rows = extract(corpus)
    points = _reach(rows)
    return ContactPoints(facts=survey(rows, points, corpus_path), points=points, rows=rows)


def write_post_contacts(path: Path, rows: Sequence[PostContacts]) -> None:
    """Every post's Contact Points, with what was written where and what was not read.

    The file holds the occurrences rather than the distinct values alone, because the
    evidence a reader has to check is the post's own text: a value with nothing beside
    it is a claim, and the same value written two ways is only visibly one value if
    both spellings are in the file. `Unread` and `ContactKind` are `StrEnum`s, so both
    serialise as their own values and need no conversion here.
    """

    def objects() -> Iterator[JsonObject]:
        for row in rows:
            yield {
                "post_id": row.post_id,
                "account": row.account,
                "contact_points": list(row.contact_points),
                "matches": [
                    {
                        "kind": match.kind.value,
                        "value": match.value,
                        "found_as": match.found_as,
                        "field": match.field,
                    }
                    for match in row.matches
                ],
                "unread": [
                    {
                        "found_as": entry.found_as,
                        "field": entry.field,
                        "reason": entry.reason.value,
                    }
                    for entry in row.unread
                ],
            }

    write_lines(path, objects())


def render_table(found: ContactPoints) -> str:
    """The console output: what was read, then every Contact Point and where it came from.

    ASCII only, so it prints the same way on a console that cannot encode anything else
    and the same way when it is redirected, which is what lets it be pasted into an
    issue or diffed between runs. One block per Contact Point, largest reach first,
    because the shared ones are the claim and the reader starts there.
    """
    sections = (
        f"{_HEADING}\n\n{_SUBHEADING}",
        _figures(found),
        _points(found),
        _footer(found),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _figures(found: ContactPoints) -> str:
    """What was read, and what came of it. One figure per line, labelled."""
    facts = found.facts
    kinds = ", ".join(kind.value for kind in found.kinds) or "none"
    entries = (
        ("corpus", f"{facts.corpus_path} ({_count(facts.posts, 'post')} across "
                   f"{_count(facts.accounts, 'account')})"),
        ("sha256", facts.corpus_sha256),
        ("kinds", kinds),
        (
            "read",
            f"{_count(facts.points, 'Contact Point')} from {facts.posts_with_points} of "
            f"{facts.posts} posts, {facts.matches} occurrences: "
            f"{_count(facts.addresses, 'email address')}, "
            f"{facts.handles} telegram {_plural(facts.handles, 'handle', 'handles')}",
        ),
        (
            "unread",
            f"{_count(facts.unread, 'unread candidate')}, naming no Contact Point",
        ),
        ("synthetic", f"{facts.synthetic} of {facts.points} are Synthetic Entities"),
        (
            "reserved",
            f"{facts.addresses_reserved} of {facts.addresses} addresses under a TLD reserved "
            "for examples",
        ),
        (
            "marked",
            f"{facts.handles_marked} of {facts.handles} handles carrying the "
            f"{SYNTHETIC_MARKER} marker",
        ),
    )
    width = max(len(name) for name, _ in entries)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in entries)


def _plural(number: int, one: str, many: str) -> str:
    """A bare noun agreeing with a count already printed beside it.

    Used only where the noun sits after another number, so `_count` cannot reach it:
    "1 email address, 2 telegram handles" reads as two counts, and folding the first
    into the second would make a reader count the addresses again to learn there was
    one."""
    return one if number == 1 else many


def _count(number: int, noun: str) -> str:
    """One count, agreeing with its noun, for the same reason."""
    return f"{number} {noun if number == 1 else f"{noun}s"}"


def _points(found: ContactPoints) -> str:
    """Every Contact Point the Corpus names, and every place in the Corpus that names it.

    Sorted by reach and then by value, so the shared ones come first and a fixed
    Corpus prints a fixed block. Each occurrence is a line, carrying the account, the
    post, and the spelling: the three things a reader needs to go and see for
    themselves that two posts named the same thing.
    """
    ordered = sorted(
        found.points.values(), key=lambda point: (-len(point.accounts), point.kind.value, point.value)
    )
    reached = sum(1 for point in ordered if point.shared)
    heading = (
        f"points  {_count(len(ordered), 'Contact Point')} read, {reached} of them reached by 2 "
        "or more accounts. Sharing is what makes one useful; nothing here groups on one yet."
    )
    if not ordered:
        return f"{heading}\n\nThe Corpus names no Contact Point at all."

    width = max(len(point.kind.value) for point in ordered)
    value_width = max(len(point.value) for point in ordered)
    lines = [heading]
    for point in ordered:
        shared = "shared, " if point.shared else ""
        counts = (
            f"{shared}{_count(len(point.posts), 'post')}, "
            f"{_count(len(point.accounts), 'account')}"
        )
        lines.append(f"  {point.kind.value.ljust(width)}  {point.value.ljust(value_width)}  {counts}")
        for row in found.rows:
            for match in row.matches:
                if (match.kind, match.value) == (point.kind, point.value):
                    lines.append(f"    {row.account}  {row.post_id}  {match.found_as}")
    return "\n".join(lines)


def _footer(found: ContactPoints) -> str:
    """What the run can and cannot claim, stated at the point of use rather than in a
    ticket: the recall limit, the two kinds of handle this build cannot tell apart,
    what it does not group on, and why every figure above is a Synthetic Entity."""
    return f"""\
Every figure above is a lower bound. What is read is every Contact Point a post names
plainly, and the recall of that is unknown and probably poor: a handle written with
separators between its characters, an address written as words rather than as an
address, and a screenshot of either are all in the wild and none of them is found
here. This build does no obfuscation tolerance at all, which is ticket #15's work, and
until it lands nothing above should be read as how many Contact Points the Corpus
holds. Deliberately disguised handles are a case the output cannot yet see, so the
count is a floor and the sharing is a floor.

A handle is read as a Telegram handle whatever service it belongs to. Every service
writes one as `@name` and this build cannot tell which it came from, so a Contact Point
from another service is reported as though it were this one. Every spelling of every
value is printed beneath it, so a reader can see how many places a string came from
before deciding what it is.

Nothing here groups on a Contact Point. ADR-0005 permits one as a grouping edge and
this command builds none: the shared blocks above are the evidence ticket #20 needs,
and a grouping edge that had not been measured would put an unmeasured input into the
recovery figure ADR-0004 is measured against.

Every Contact Point above is a Synthetic Entity because the Corpus is synthetic, and
which figures say so is printed rather than asserted. There is no reserved namespace
for a Telegram username, so a handle is recognisable as synthetic only by carrying the
{SYNTHETIC_MARKER} marker, and a Corpus Provider is a swap: the same lines over real
content would read `0 of {found.facts.points}`, which is what measuring them buys.

Nothing on this path reads the truth file or the Nuisance Structure manifest, and no
published list is read either: what a post says to be reached at does not depend on
what anybody registered (ADR-0008)."""


def render_report(found: ContactPoints) -> str:
    """The reader-facing report, generated from the rows rather than written beside them.

    Markdown rather than the console's tables because this one is meant to be read in
    a browser: it is the artefact a reviewer opens to see what a post said to be
    reached at, and it is per post, which the console deliberately is not.
    """
    facts = found.facts
    return f"""# Contact Points in the Corpus

Generated by `rfi contact-points` from the Corpus file. Do not edit it by hand — a
test holds this file to what that produces, and re-running the command rewrites it
byte for byte.

## What was read

| | |
|---|---|
| Posts | {facts.posts} |
| Accounts | {facts.accounts} |
| Distinct Contact Points | {facts.points} |
| Places a Contact Point was named | {facts.matches} |
| Posts naming at least one | {facts.posts_with_points} |
| Reached by two or more accounts | {facts.shared} |
| Candidates naming no Contact Point | {facts.unread} |
| Synthetic Entities | {facts.synthetic} of {facts.points} |
| Corpus | `{facts.corpus_path}` |
| SHA-256 of the Corpus | `{facts.corpus_sha256}` |

## How a Contact Point is read

A Contact Point is an off-platform identifier somebody can be reached at, and this
build looks for two kinds and nothing else: an email address and a Telegram handle.
Both are read **literally**. Nothing is repaired, and nothing that looks like one of
these and is not one is dropped — it is reported with which of eight faults applied:

{", ".join(f"`{reason.value}`" for reason in Unread)}

Reading loosely would be worse than reading narrowly here, and the reason is specific
to this step: a Contact Point is the one string in the Corpus that can join two
accounts on nothing else, so a value that was guessed becomes a grouping edge nobody
checked. A false positive here is not a wrong severity figure, which is what a Signal
can produce; it is a shared identifier between two accounts that share nothing.

**Case and trailing punctuation are normalised, and the spelling is kept.** A handle is
`@someone`, and Telegram usernames do not distinguish case, so `@Syn_VantageLedger` and
`@syn_vantageledger` are one Contact Point. A full stop at the end of a sentence
belongs to the sentence: a handle written `@syn_vantageledger.` is the same handle, and
reporting the run as it stands would silently split one shared Contact Point into two —
a wrong answer in the direction of losing a share, produced by being pedantic about
punctuation. Both spellings are printed for every value, so a reader can tell sharing
from case folding.

**An address is normalised more carefully, because its two halves are not the same.**
The host is case-folded, because RFC 5321 says a domain is case-insensitive. The part
before the `@` is not, because RFC 5321 says nothing of the sort about it: folding both
would merge `Desk@` and `desk@` into one mailbox that may be two, which is a shared
identifier between two accounts that share nothing — this module inventing a grouping
edge in order to be helpful, which is the failure it exists to avoid.

**A link is read for a Contact Point, on the same rule as any other link.** A
`mailto:` names an address rather than a page — `rfi post-domains` reports it as naming
no registration and hands the case here — and a link on Telegram's own host
({", ".join(f"`{host}`" for host in sorted(TELEGRAM_HOSTS))}) names the username in its
first path segment. Any other link names a page, and an address inside a page is not
read: a shortener's query string is the shortener's business. A Telegram link to a
channel or an invite names no username at all and is reported as `invite_link` rather
than guessed at, and a link that will not parse is reported as `malformed` — the one
case this command cannot classify, and the one that most needs saying so.

**Sharing is counted over accounts, not over posts.** Two accounts naming one Contact
Point is the claim this command exists to make; one account naming it in four posts is
not, because it has not acquired a second reach.

## What this cannot claim

Every figure above is a lower bound, and the output says so at the point of use rather
than only here. Nothing is done about a handle written with separators between its
characters, an address written as words rather than as an address, or a screenshot of
either: this build does no obfuscation tolerance at all. The recall is unmeasured, and
measuring it is ticket #15's work.

A handle is read as a Telegram handle whatever service it belongs to. Every service
writes one as `@name` and this build cannot tell which it came from, so a Contact Point
belonging to another service is reported as though it were this one.

Nothing here groups on a Contact Point. ADR-0005 permits one as a grouping edge and
this command builds none — the shared blocks are the evidence `rfi
campaign-candidates` does not use yet, because a grouping edge that had not been
measured would put an unmeasured input into the recovery figure ADR-0004 is measured
against.

Every Contact Point in this file is a Synthetic Entity, because the Corpus is
synthetic, and which figures say so is printed rather than asserted: an address is one
when its host sits under a TLD reserved for examples (RFC 2606), and a handle when it
carries the `{SYNTHETIC_MARKER}` marker every Synthetic Entity in this Corpus carries,
since Telegram reserves no namespace of its own.

## The Contact Points in the Corpus

{_point_table(found)}

## Every post, and every Contact Point it names

{_post_sections(found)}
"""


def _point_table(found: ContactPoints) -> str:
    """One row per Contact Point: what it is, what reaches it, and how it was written.

    The same information as the console's blocks, in a shape a reader can scan down a
    column. Rows are ordered by reach and then by value, so the shared ones come first.
    Written unpadded, as Markdown renders it anyway and as the registrable-domain
    report does — a table padded to a fixed width carries a column of spaces into a
    committed file for no reader.
    """
    ordered = sorted(
        found.points.values(),
        key=lambda point: (-len(point.accounts), point.kind.value, point.value),
    )
    if not ordered:
        return "The Corpus names no Contact Point at all."

    headings = ("Contact Point", "kind", "posts", "accounts", "shared", "written as")
    body = [
        (
            f"`{point.value}`",
            point.kind.value,
            str(len(point.posts)),
            str(len(point.accounts)),
            "yes" if point.shared else "-",
            ", ".join(f"`{spelling}`" for spelling in point.spellings),
        )
        for point in ordered
    ]
    table = "\n".join(
        [f"| {' | '.join(headings)} |", "| --- | --- | ---: | ---: | :-: | --- |"]
        + [f"| {' | '.join(entry)} |" for entry in body]
    )
    return table + "\n\nSharing is not a grouping. Nothing in this file is used to put two \
accounts together yet."


def _post_sections(found: ContactPoints) -> str:
    """One section per post: what it named, where it was written, and what was not read.

    Per post and in the Corpus's own order, including the posts that named nothing —
    most of them. Those are the floor of the case, and a report that listed only the
    posts with something on them would hide exactly the posts worth checking.
    """
    return "\n".join(_post_section(row) for row in found.rows)


def _post_section(row: PostContacts) -> str:
    """One post's Contact Points, with the spelling that named each and the field it was
    found in, and any candidate that could not be read with its reason.

    The list is printed first and on its own, because that is what a reader opens the
    report to find; the table under it is the reasoning, and the unread rows are the
    cases a reviewer may want to look at rather than take.
    """
    heading = f"### `{row.post_id}` · `{row.account}`\n"
    if not row.matches and not row.unread:
        return (
            f"{heading}\nNo Contact Point named in the title, the body, or the links. "
            "Nothing to share with another post, and nothing for a grouping to join it by.\n"
        )

    lines = [heading, "Contact points: " + ", ".join(f"`{value}`" for value in row.contact_points)]
    if row.matches:
        lines += ["", "| Kind | Contact Point | Written as | Found in |", "| --- | --- | --- | --- |"]
        lines += [
            f"| {match.kind.value} | `{match.value}` | `{match.found_as}` | {match.field} |"
            for match in row.matches
        ]
    if row.unread:
        lines += ["", "Candidates that name no Contact Point:", ""]
        lines += [
            f"- `{entry.found_as}` in the {entry.field} — **{entry.reason}**"
            for entry in row.unread
        ]
    return "\n".join(lines) + "\n"