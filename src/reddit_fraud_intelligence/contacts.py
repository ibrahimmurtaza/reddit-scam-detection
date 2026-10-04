"""Contact Points: what the posts say to be reached at, per post, and which are shared.

ADR-0005 lets two accounts reach the same Campaign Candidate on a Contact Point as
well as on a Registrable Domain, and this is the step that finds them. A Contact
Point is off-platform infrastructure an operator controls and a reader can be sent
to, so a handle two accounts publish is shared infrastructure in everything but name.
It is also the edge a campaign that rotates domains per post is visible on and
invisible through, which is why it is worth extracting even where the domains already
group a pair of accounts.

Two kinds are read, and nothing else is looked for: an email address and a Telegram
handle. Both are read from the post's own title, its own body, and the links it
carries, and both are read with the obfuscation tolerance below: an identifier written
out character by character, with a digit standing in for a letter, or with either of
those in a post's own text rather than in a link, is the Contact Point it is hiding and
is read as it. Every spelling a value was written in travels beside the value, because a
fold a reader cannot see is indistinguishable from case folding.

That tolerance is the one place in this project where a wrong answer invents rather than
misses, and it is stated rather than argued. Dropping a full stop or a hyphen from a
username cannot merge two usernames that both exist — neither character can occur in one
— while folding a digit into the letter it stands in can, and every figure that fold can
move is measured and printed. A Contact Point is the one string in the Corpus that can
join two accounts on nothing else, so a value that was guessed is a grouping edge nobody
checked; reading loosely would be worse than reading narrowly here even where the recall
would be better.

Which is why the run reads a **labelled set** — one row per post, naming every Contact
Point that post publishes and whether it published it plainly, written out, or only as a
picture of one — and prints the recall and the false-positive rate beside the figures
those numbers bound. A count of Contact Points is a count of the ones this reading found,
and the shortfall is named row by row rather than left to be inferred from the count. The
labelled set is read after the reading has finished and is handed to `measure` rather than
to the rules, so a label has no path to a decision about what was read; the Corpus is
still the whole input, no published list decides what a post says to be reached at, and
the truth file is joined by the evaluator after this has finished (ADR-0008, ADR-0019).

A candidate that names no Contact Point is reported with which of nine reasons
applied, so nothing is dropped: a candidate quietly dropped is indistinguishable from
a post that named none, and the difference is the whole claim. The nine are each a
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

Normalisation is published rather than minimal, and every spelling a value was written
in is printed beside it, so a reader can tell sharing from folding. A Telegram username
is case-insensitive, a full stop at the end of a sentence belongs to the sentence, and a
username cannot hold a full stop or a hyphen at all — so all three are folded into the
value. An address is not treated the same way: the host is folded because RFC 5321 says
a domain is case-insensitive, the local part is not because RFC 5321 says nothing of the
sort, and no full stop is dropped from either half because a full stop is a label boundary
in both. Folding both halves of an address would merge `Desk@` and `desk@` into one
mailbox that may be two, and dropping a full stop would merge two hosts that both exist —
this module inventing a grouping edge in order to be helpful, which is the failure it
exists to avoid.

Nothing is grouped here. ADR-0005 permits a Contact Point as a grouping edge and this
command builds none: the shared list is the evidence ticket #20 needs, and a grouping
edge that had not been measured would put an unmeasured input into the recovery figure
ADR-0004 is measured against. The output says so, because a shared list read as a list
of groupings is the more damaging of the two misreadings.

Nothing here reads the truth file or the Nuisance Structure manifest. The Corpus and the
labelled set beside it are the whole input — no published list decides what a post says
to be reached at — and the truth file is joined by the evaluator after this has finished
(ADR-0008).

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
from reddit_fraud_intelligence.jsonl import (
    JsonObject,
    read_object,
    read_rows,
    read_text,
    read_vocabulary,
    refuse_repeated,
    write_lines,
)

_HEADING = "Contact Points"
_SUBHEADING = """\
Every Contact Point a post's own title, its own body, and its links name, read with the
obfuscation tolerance this build does and with nothing beyond it. Two kinds are looked for
and nothing else. No grouping is built on any of it here."""

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
# username can be made of, holding at least one `@`. The part before the `@` allows the
# punctuation an address's local part allows, the part after it does not, because a
# username cannot hold one. Scanning starts at every `@`, so a candidate in the middle
# of a sentence is found without the rule knowing anything about sentences.
#
# Both halves stop at a character neither half can hold, which is what makes
# `write to me at @ or ring` two candidates rather than one: the run after the `@` is
# empty, and a candidate whose name is empty is reported rather than skipped.
_LEFT_RUN = re.compile(r"[A-Za-z0-9_%+.\- ]")
_RIGHT_RUN = re.compile(r"[A-Za-z0-9_.\- ]")

# The same run with the space left out, which is the plain candidate an obfuscated run
# falls back to. Two walks rather than one because the space is the only difference
# between "a post wrote one handle" and "a post wrote one handle out": without it, a
# run written out character by character is three or four unread candidates instead of
# one Contact Point.
_LEFT_PLAIN = re.compile(r"[A-Za-z0-9_%+.\-]")
_RIGHT_PLAIN = re.compile(r"[A-Za-z0-9_.\-]")

# The characters a username is made of, which is Telegram's own rule read in
# `contacts.py` rather than a second opinion: letters, digits, and underscores. A fold
# keeps exactly these, so a handle written with anything else between its characters is
# read as the handle without them.
_HANDLE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)

# The characters a run is cut into before it is judged written out: each maximal stretch
# of characters an identifier is made of, which is one piece whether a separator sits
# between two of them or not. A piece of one character is a character a post has written
# out; a piece of more is a word or a whole identifier, and `_written_out` stops there.
_TOKEN = re.compile(r"[A-Za-z0-9_]+")

# The characters a post stands in for the letter it is hiding, and the letters it stands
# for. Published here rather than kept in prose because this is the part of the reading
# that can invent a Contact Point: a digit is legal in a username, so `syn_vantag3ledger`
# and `syn_vantageledger` are two usernames as far as Telegram is concerned and one
# Contact Point as far as this reader is concerned. Every figure that fold can move is
# measured and printed; see `Recall` and `_points`.
#
# **A username and never a host.** `gr4vy.io` and `gravy.io` are two domains that both
# exist and a post that writes one of them does not say which it meant, so folding a host
# would merge two sites on a guess about the author's keyboard — and `intake@vantage.ex4mple`
# would stop being a real registration and start counting as a Synthetic Entity. The
# handle fold is a guess made visible in the `stray` figure; a host fold would be a guess
# about somebody else's domain, which is the one merge this project must not make on its
# own. `_address` reads a host exactly as it was written.
_SUBSTITUTION = str.maketrans("0134578", "oieastb")

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
_TRAILING_SEPARATORS = ".- "

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
    PROSE = "prose"


class Writing(StrEnum):
    """How a post published a Contact Point, which is what the reading is measured on.

    Three ways and no fourth, because they are the three the measurement has to be able
    to take apart: published plainly, published disguised, and published only as a
    picture of one. The third is not readable by anything in this project, which is why
    it is a value rather than a limitation in prose — it is counted as a miss against
    the reading and printed in the report, where a reader can see that this build is
    blind to it rather than being told so.
    """

    PLAIN = "plain"
    OBFUSCATED = "obfuscated"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class PublishedContact:
    """One Contact Point a post publishes, and the spelling it published it in.

    Declared by the generator beside the post it belongs to, so a label cannot drift
    away from the text it describes: the file this is written to holds one row per post
    and this is what is in that post.
    """

    kind: ContactKind
    value: str
    written_as: str
    written: Writing


@dataclass(frozen=True, slots=True)
class PublishedPost:
    """One post and every Contact Point it publishes.

    An empty `published` is the case that makes the false-positive rate a rate: without
    posts labelled as publishing nothing, the question "how often does this invent a
    Contact Point" has no denominator and answers itself.
    """

    post_id: str
    published: tuple[PublishedContact, ...]


def write_published(path: Path, posts: Iterable[PublishedPost]) -> None:
    """The labelled set, one row per post, sorted so a fixed seed writes fixed bytes.

    Sorted by post rather than written in the plan's order because the Corpus file is
    sorted by time and the two are different projections of the same plan; a reader
    holding both files wants to line them up by identifier.
    """

    def objects() -> Iterator[JsonObject]:
        for post in sorted(posts, key=lambda post: post.post_id):
            yield {
                "post_id": post.post_id,
                "published": [
                    {
                        "kind": contact.kind.value,
                        "value": contact.value,
                        "written_as": contact.written_as,
                        "written": contact.written.value,
                    }
                    for contact in post.published
                ],
            }

    write_lines(path, objects())


def read_published(path: Path) -> tuple[PublishedPost, ...]:
    """The labelled set, read back and checked row by row.

    Every field is parsed rather than passed through, for the reason `read_nuisance`
    gives and because this file decides what the recall figure means: a kind or a way of
    writing the reader cannot account for would be silently dropped from the
    measurement, and a post named on two rows would be counted twice in both directions
    of it. A Contact Point published with no value is refused for the same reason an
    unread candidate is reported — it names nothing that could be found.
    """
    posts = tuple(
        _published_post(path, number, text) for number, text in read_rows(path)
    )
    refuse_repeated(
        path.as_posix(),
        (post.post_id for post in posts),
        "the recall figure would count one post's Contact Points twice",
    )
    return posts


def _published_post(path: Path, number: int, text: str) -> PublishedPost:
    where = f"{path.as_posix()}:{number}"
    record = read_object(where, text)
    if set(record) != {"post_id", "published"}:
        raise ValueError(
            f"{where} holds {sorted(record)}, which is not the labelled vocabulary "
            "['post_id', 'published']"
        )
    value = record["published"]
    if not isinstance(value, list):
        raise ValueError(f"{where} has published={value!r}, which is not a list")
    return PublishedPost(
        post_id=read_text(where, record, "post_id"),
        published=tuple(_published_contact(where, index, entry) for index, entry in enumerate(value)),
    )


def _published_contact(where: str, index: int, entry: object) -> PublishedContact:
    if not isinstance(entry, dict):
        raise ValueError(f"{where} published[{index}] is {entry!r}, which is not a row")
    at = f"{where} published[{index}]"
    vocabulary = {"kind", "value", "written_as", "written"}
    if set(entry) != vocabulary:
        raise ValueError(f"{at} holds {sorted(entry)}, which is not {sorted(vocabulary)}")
    return PublishedContact(
        kind=read_vocabulary(at, "kind", entry["kind"], ContactKind),
        value=read_text(at, entry, "value"),
        written_as=read_text(at, entry, "written_as"),
        written=read_vocabulary(at, "written", entry["written"], Writing),
    )


def _walk_back(text: str, at: int, run: re.Pattern[str]) -> int:
    """Where the run reaching left from `at` begins."""
    start = at
    while start > 0 and run.match(text, start - 1):
        start -= 1
    return start


def _walk_forward(text: str, at: int, run: re.Pattern[str]) -> int:
    """Where the run reaching right from `at` ends."""
    end = at
    while end < len(text) and run.match(text, end):
        end += 1
    return end


def _candidates(text: str) -> Iterator[Candidate]:
    """Every `@` in a post's own text, with the run around it and how to read it.

    One pass, from left to right, and the run is walked twice per `@`: once over the
    characters an identifier is written with, and once over those plus the space. The
    wider run is what a post writing an identifier out one character at a time produces,
    and the narrower one is the candidate that would have been read without any
    obfuscation tolerance. Which of the two is read is `_read_run`'s decision and not
    this one's, so that the rule is stated in one place.

    Scanning resumes past the run rather than past the `@`, because a run holds exactly
    one `@` — the walk stops at the next one — so nothing inside it is a candidate the
    next pass would miss.
    """
    index = 0
    while (at := text.find("@", index)) >= 0:
        start = _walk_back(text, at, _LEFT_RUN)
        end = _walk_forward(text, at + 1, _RIGHT_RUN)
        index = end
        yield _read_run(
            left=text[start:at],
            right=text[at + 1 : end],
            plain_left=text[_walk_back(text, at, _LEFT_PLAIN) : at],
            plain_right=text[at + 1 : _walk_forward(text, at + 1, _RIGHT_PLAIN)],
        )


def _leading_singles(side: str) -> int:
    """How far from the start of a side the pieces of one character run."""
    taken = 0
    for token in _TOKEN.finditer(side):
        if len(token.group()) != 1:
            break
        taken = token.end()
    return taken


def _written_out(side: str, at_the_end: bool) -> str:
    """The identifier one side of a run is written as, or the empty string for none.

    **The identifier is the run of pieces that each hold exactly one character, read
    from the end of the side nearest the `@`** — the leading pieces of a right-hand side
    and the trailing pieces of a left-hand one, because the words a post puts on the far
    side of the `@` are not part of what it names. That is the fingerprint of a post
    writing an identifier out one character at a time, and stopping at the first
    ordinary word is what lets `@ s y n _ v a n t a g e l e d g e r, or ask in here` be
    read as the handle it is rather than as a sentence.

    The cost is stated rather than hidden: a post that spaces out the letters of a word
    beside an `@` produces an identifier nobody published, `email me @ t o n i g h t at
    8` reads as the handle `tonight`, and that is the one false positive this reading
    has that the literal reader did not. It is measured against the labelled set and
    printed beside the figures rather than argued here.

    An empty result means the side holds no written-out identifier and is read plainly,
    which is the right answer for `@someone`, for `Write to me at @`, and for the words
    a post puts on the far side of an `@`.
    """
    if not at_the_end:
        return side[: _leading_singles(side)]
    end = len(side)
    for token in reversed(list(_TOKEN.finditer(side))):
        if len(token.group()) != 1:
            break
        end = token.start()
    return side[end:]


def _read_run(*, left: str, right: str, plain_left: str, plain_right: str) -> Candidate:
    """One run around an `@`, read as far as the writing-out of each side allows.

    **The `@` decides the kind before either side is read.** A run with an identifier
    touching its `@` on the left is an address, and both of its sides can be written
    out; a run with nothing there is somebody's handle, and a handle is written after
    its `@` and never before it. Reading the left side of a handle would turn `on @
    s y n` into the address `on@syn`, which names no host and reports the wrong fault
    for a post that did nothing wrong — and it is the left side being written out that
    says a run is an address at all, since a local part written out has a space before
    the `@` where a plain one has a word.

    `prose` is the run that is neither an identifier nor nothing at all: nothing
    identifier-shaped touches the `@` and words follow it, which is what `email me @
    home about the invoice` is. It is a fault of its own because a reader looking at
    that run needs to know the `@` was never a handle rather than that a handle lost
    its name.
    """
    written_left = _written_out(left, at_the_end=True)
    written_right = _written_out(right, at_the_end=False)
    found_as = f"{written_left or plain_left}@{written_right or plain_right}"

    if written_left or written_right:
        return Candidate(found_as, Writing.OBFUSCATED, prose=False)
    if plain_left or plain_right or not right:
        return Candidate(found_as, Writing.PLAIN, prose=False)
    return Candidate(found_as, Writing.PLAIN, prose=True)


@dataclass(frozen=True, slots=True)
class Candidate:
    """The text around one `@`, and how that run was written.

    `writing` says how much of the run was read as the identifier: plainly, or written
    out one character at a time, which is what lets `_handle` drop the separators a
    username cannot hold. `prose` says the run is ordinary wording around the `@` and
    holds no identifier at all — `email me @ home about the invoice` — which is a
    different fault from a bare `@` and has a different fix.
    """

    found_as: str
    writing: Writing
    prose: bool


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
    labelled_path: str
    labelled_sha256: str
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
    recall: Recall
    rows: tuple[PostContacts, ...]

    @property
    def kinds(self) -> tuple[ContactKind, ...]:
        """The kinds this Corpus actually holds, in declaration order."""
        return tuple(kind for kind in ContactKind if any(k == kind for k, _ in self.points))


def _read_point(candidate: Candidate, field: str) -> Match | UnreadCandidate:
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
    found_as = candidate.found_as
    if candidate.prose:
        return UnreadCandidate(found_as, field, Unread.PROSE)
    trimmed = found_as.rstrip(_TRAILING_SEPARATORS)
    if trimmed.startswith("@"):
        return _handle(trimmed[1:], found_as, field, candidate.writing)
    return _address(trimmed, found_as, field)


def _address(candidate: str, found_as: str, field: str) -> Match | UnreadCandidate:
    """One candidate read as an address, or the reason it names no host.

    The host has to be a host: at least two labels, none of them empty, each one a
    label. `nobody@localhost` is the case this exists for, and the tempting repairs —
    appending a TLD, or taking the last two labels of whatever follows the `@` — both
    produce an address that looks right, which is the one thing a Contact Point must
    never be, since it can join two accounts on nothing else.

    **An address is folded for spacing and for nothing else.** A space cannot occur in an
    address at all, so removing it cannot merge two addresses that both exist; a full stop
    cannot be removed from either half of one, because a full stop is a label boundary in
    both and `vantage.ledger.example` is not `vantage-ledger.example`. It is not folded
    for substitution, which a username is: nothing in a post says whether `desk@gr4vy.io`
    means `gravy.io`, and a guess that joins two accounts over two different domains is
    the failure this module exists to avoid. See `_SUBSTITUTION`.

    **The host is case-folded and the local part is not.** RFC 5321 makes the domain
    case-insensitive and says nothing of the sort about the local part, so
    `desk@vantage-ledger.example` and `DESK@VANTAGE-LEDGER.EXAMPLE` are one mailbox
    while `Desk@` and `desk@` may be two. Folding the whole address would merge those
    two on the strength of a rule RFC 5321 does not state, and a merged mailbox is a
    shared identifier between two accounts that share nothing — the one failure this
    module exists to avoid, arrived at by being helpful.
    """
    spaced = "".join(character for character in candidate if not character.isspace())
    _, _, host = spaced.partition("@")
    labels = host.split(".")
    if len(labels) < 2 or any(not _LABEL.match(label) for label in labels):
        return UnreadCandidate(found_as, field, Unread.NO_DOMAIN)
    if len(spaced) > _MAX_ADDRESS_CHARS:
        return UnreadCandidate(found_as, field, Unread.ADDRESS_TOO_LONG)
    return Match(
        kind=ContactKind.EMAIL,
        value=f"{spaced.partition('@')[0]}@{host.lower()}",
        found_as=found_as,
        field=field,
    )


def _handle(
    name: str, found_as: str, field: str, writing: Writing = Writing.PLAIN
) -> Match | UnreadCandidate:
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

    **Two folds, kept apart because they rest on different evidence.** A substitution
    needs none: a digit is a letter standing in for one whether or not anything else
    about the run says so, which is why `@syn_vantag3ledger` and `@syn_vantageledger`
    are one Contact Point here — and why substitution can merge two usernames that both
    exist, which is why `Recall` counts the folds and names the posts they moved.
    Dropping a separator needs the run to have been written out, because a full stop or
    a hyphen in a username is illegal whatever the intent was: `@someone.co.uk` is a
    mask on an address far more often than it is a disguised handle, so read plainly it
    is refused and named `handle_character` where a written-out run has its separators
    taken out and reaches the username they were hiding.

    A link on Telegram's own host names a username in one path segment with nothing to
    write out, so it arrives here with the plain default.
    """
    canonical = name.translate(_SUBSTITUTION)
    if writing is Writing.OBFUSCATED:
        canonical = "".join(
            character for character in canonical if character in _HANDLE_CHARS
        )
    if not canonical:
        return UnreadCandidate(found_as, field, Unread.NO_NAME)
    if len(canonical) < _MIN_HANDLE_CHARS:
        return UnreadCandidate(found_as, field, Unread.HANDLE_TOO_SHORT)
    if len(canonical) > _MAX_HANDLE_CHARS:
        return UnreadCandidate(found_as, field, Unread.HANDLE_TOO_LONG)
    if not _USERNAME.match(canonical):
        return UnreadCandidate(found_as, field, Unread.HANDLE_CHARACTER)
    return Match(
        kind=ContactKind.TELEGRAM,
        value=canonical.lower(),
        found_as=found_as,
        field=field,
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
        for candidate in _candidates(text_value):
            _record(_read_point(candidate, field_name), matches, unread)

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
    labelled_path: Path,
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
        labelled_path=labelled_path.as_posix(),
        labelled_sha256=hashlib.sha256(labelled_path.read_bytes()).hexdigest(),
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


@dataclass(frozen=True, slots=True)
class Finding:
    """Where a measurement found something, and what it found.

    The two facts every finding carries, and the reason they are a type rather than four
    fields copied into three dataclasses: the measurement is read by a reader checking one
    post at a time, and every finding in it is a post and a value in that post.
    """

    post_id: str
    value: str


@dataclass(frozen=True, slots=True)
class Missed(Finding):
    """A Contact Point the Corpus publishes that this run did not read."""

    written_as: str
    written: Writing


@dataclass(frozen=True, slots=True)
class Invented(Finding):
    """A Contact Point read out of a post the labelled set says publishes none."""

    written_as: str


@dataclass(frozen=True, slots=True)
class Stray(Finding):
    """A Contact Point read out of a post that publishes one, and it is not that one.

    The case a fold has when it is wrong: two Contact Points the Corpus publishes
    separately have been read as one value, so one of them is missing from the output and
    the other has been reported in a post that did not publish it. It is the only way the
    reading can join two accounts that share nothing, which is why each one is printed
    with both values rather than counted.
    """

    published: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Recall:
    """What the reading found against what the Corpus publishes.

    Three numbers and three lists, and the three lists are what make the numbers
    checkable: how much of what was published was found, how much was invented where
    nothing was published, and how many published Contact Points came out under another
    one's name. Each is counted over the labelled set rather than over the run, so none of
    them can be moved by a change to the Corpus that the labels do not move with.

    The figure is a measurement of *this* reading over *this* Corpus, and the report says
    so: a Corpus from a provider has no labelled set beside it, and a reader wanting the
    figure over other content has to label that content themselves.
    """

    posts: int
    clean_posts: int
    published: int
    found: int
    plain: int
    plain_found: int
    obfuscated: int
    obfuscated_found: int
    image: int
    image_found: int
    invented: tuple[Invented, ...]
    missed: tuple[Missed, ...]
    stray: tuple[Stray, ...]

    @property
    def invented_posts(self) -> int:
        """The posts a Contact Point was invented in, which is the rate's numerator.

        Counted over posts rather than over values: one post reading two invented values
        is one post this reading was wrong about, and a rate over posts is the question
        a reader of the Corpus is asking.
        """
        return len({entry.post_id for entry in self.invented})


def measure(
    rows: Sequence[PostContacts], published: Sequence[PublishedPost]
) -> Recall:
    """What this reading found, against what the Corpus says each post publishes.

    Takes the finished reading rather than the Corpus, which is the whole reason this can
    sit in the same command as the reading: the labels reach nothing that decides what is
    read, because the only thing they reach is this function and this function is handed
    the result. `tests/test_contact_points.py` holds that by running the reading over the
    Corpus twice, once with the labelled set beside it and once without, and asserting
    the rows are identical.

    A labelled post the Corpus does not hold, or a Corpus post the labelled set does not
    cover, is refused rather than measured: a recall figure over a subset of the posts
    would read as a figure over the Corpus, and a false-positive rate with the posts that
    publish nothing left out of it has no denominator.
    """
    read = {
        row.post_id: {(match.kind, match.value): match for match in row.matches}
        for row in rows
    }
    labelled = {post.post_id: post for post in published}
    missing = sorted(set(read) - set(labelled))
    if missing:
        raise ValueError(
            f"the labelled set says nothing about {missing}, and a recall figure over "
            "some of the posts is a figure over none of the Corpus"
        )
    extra = sorted(set(labelled) - set(read))
    if extra:
        raise ValueError(
            f"the labelled set names {extra}, which the Corpus does not hold"
        )

    missed: list[Missed] = []
    invented: list[Invented] = []
    stray: list[Stray] = []
    published_count = {writing: 0 for writing in Writing}
    found_count = {writing: 0 for writing in Writing}

    for post_id in sorted(labelled):
        found = read[post_id]
        claims = {contact.value: contact for contact in labelled[post_id].published}
        for key, match in found.items():
            if key[1] in claims:
                continue
            if claims:
                stray.append(
                    Stray(post_id=post_id, value=key[1], published=tuple(sorted(claims)))
                )
            else:
                invented.append(
                    Invented(post_id=post_id, value=key[1], written_as=match.found_as)
                )
        for contact in labelled[post_id].published:
            published_count[contact.written] += 1
            if (contact.kind, contact.value) in found:
                found_count[contact.written] += 1
            else:
                missed.append(
                    Missed(
                        post_id=post_id,
                        value=contact.value,
                        written_as=contact.written_as,
                        written=contact.written,
                    )
                )

    return Recall(
        posts=len(labelled),
        clean_posts=sum(1 for post in published if not post.published),
        published=sum(published_count.values()),
        found=sum(found_count.values()),
        plain=published_count[Writing.PLAIN],
        plain_found=found_count[Writing.PLAIN],
        obfuscated=published_count[Writing.OBFUSCATED],
        obfuscated_found=found_count[Writing.OBFUSCATED],
        image=published_count[Writing.IMAGE],
        image_found=found_count[Writing.IMAGE],
        invented=tuple(invented),
        missed=tuple(missed),
        stray=tuple(stray),
    )


def contact_points(corpus_path: Path, labelled_path: Path) -> ContactPoints:
    """Read the Corpus, measure the reading against the labelled set, and return both.

    One call, for the same reason the grouping is one call: the per-post rows, the shared
    list, the figures, and the measurement are four views of one pass over the Corpus, and
    a caller that extracted and then asked for the sharing separately could end up
    printing one run's figures over another's results.

    The Corpus is read rather than generated, because the file is the boundary
    (ADR-0001, ADR-0008). The labelled set is read afterwards and only to be measured
    against, and no published list is read at all: nothing about what a post says to be
    reached at depends on what anybody registered.
    """
    corpus = read_corpus(corpus_path)
    rows = extract(corpus)
    points = _reach(rows)
    return ContactPoints(
        facts=survey(rows, points, corpus_path, labelled_path),
        points=points,
        recall=measure(rows, read_published(labelled_path)),
        rows=rows,
    )


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
    recall = found.recall
    entries = (
        ("corpus", f"{facts.corpus_path} ({_count(facts.posts, 'post')} across "
                   f"{_count(facts.accounts, 'account')})"),
        ("sha256", facts.corpus_sha256),
        ("kinds", kinds),
        (
            "read",
            f"{_count(facts.points, 'Contact Point')} from {facts.posts_with_points} of "
            f"{facts.posts} posts, {facts.matches} occurrences: "
            f"{facts.addresses} {_plural(facts.addresses, 'email address', 'email addresses')}, "
            f"{facts.handles} telegram {_plural(facts.handles, 'handle', 'handles')}",
        ),
        (
            "unread",
            f"{_count(facts.unread, 'unread candidate')}, naming no Contact Point",
        ),
        ("labelled", _labelled(facts, recall)),
        ("recall", _recall(recall)),
        ("picture", _pictures(recall)),
        ("invented", _invented(recall)),
        ("folded", _folded(recall)),
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


def _labelled(facts: ContactFacts, recall: Recall) -> str:
    """Where the figure underneath comes from, so it can be taken apart by a reader.

    The path the run actually read, from the facts, rather than the one the command
    defaults to: a report naming a file this run did not measure against would send a
    reader to check the wrong rows, and the same bytes would be a different report over a
    different labelled set.
    """
    return (
        f"{facts.labelled_path}: {recall.posts} posts, "
        f"{_count(recall.published, 'Contact Point')} published between them"
    )


def _recall(recall: Recall) -> str:
    """What was found of what was published, split by how it was published.

    The split is the figure rather than a footnote on it: 10 of 12 tells a reader
    nothing, while 9 of 9 written plainly and 3 of 4 written out tells them that the
    tolerance works and something else does not.
    """
    return (
        f"{recall.found} of {recall.published} Contact Points the Corpus publishes were "
        f"found: {recall.plain_found} of {recall.plain} published plainly, "
        f"{recall.obfuscated_found} of {recall.obfuscated} written out"
    )


def _pictures(recall: Recall) -> str:
    """The Contact Points published only as a picture of one.

    Counted rather than left to the reader to infer from a recall figure, because this
    build cannot read an image at all: a reader who was told only that two Contact Points
    were missed would have to guess whether the reading was wrong or whether the
    information was never in the text.
    """
    if not recall.image:
        return "no Contact Point in this Corpus is published only as a picture"
    return (
        f"{recall.image} of the {recall.published} is published only as a picture of one "
        "and cannot be read by anything in this project"
    )


def _invented(recall: Recall) -> str:
    """What was read where the Corpus publishes nothing at all.

    The only figure here that can join two accounts that share nothing, so it is counted
    over the posts that publish nothing — the denominator a false-positive rate needs —
    and every invented value is named in the report beside it.
    """
    if recall.invented_posts:
        return (
            f"{recall.invented_posts} of the {recall.clean_posts} posts that publish none "
            f"had {_plural(recall.invented_posts, 'a Contact Point', 'Contact Points')} "
            "invented in them"
        )
    return (
        f"no Contact Point read in the {_count(recall.clean_posts, 'post')} that "
        f"{_plural(recall.clean_posts, 'publishes', 'publish')} none"
    )


def _folded(recall: Recall) -> str:
    """Where a published Contact Point came out under another one's name.

    The failure the fold can cause, which is not a missed Contact Point but a joined
    pair of them, so it is counted separately from the misses and named in the report.
    """
    if recall.stray:
        return (
            f"{recall.found} of {recall.published} found, but {len(recall.stray)} of them "
            "is a value the post publishing it does not publish"
        )
    return (
        f"every Contact Point found is the one its own post publishes; no fold joined two "
        "of the values the Corpus publishes"
    )


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
    ticket: what the measured recall does and does not cover, the two kinds of handle this
    build cannot tell apart, what it does not group on, and why every figure above is a
    Synthetic Entity."""
    recall = found.recall
    return f"""\
Every figure above is a lower bound, and the recall figure above is what bounds it: the
Contact Points in this Corpus that were not read are named in the report, one per row,
rather than left to the reader to infer from a count. What the tolerance does not reach is
stated rather than argued: a handle published only as a picture of one cannot be read by
anything in this project, an address written as words has no `@` in it for any reader to
find, and both are planted in this Corpus so that the recall figure is a measurement
rather than a claim.

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
what anybody registered (ADR-0008). The labelled set is read, and only after the reading
has finished, to be measured against: it names what each post publishes and cannot reach
a decision about what was read."""


def render_report(found: ContactPoints) -> str:
    """The reader-facing report, generated from the rows rather than written beside them.

    Markdown rather than the console's tables because this one is meant to be read in
    a browser: it is the artefact a reviewer opens to see what a post said to be
    reached at, and it is per post, which the console deliberately is not.
    """
    facts = found.facts
    return f"""# Contact Points in the Corpus

Generated by `rfi contact-points` from the Corpus file and the labelled set beside it.
Do not edit it by hand — a test holds this file to what that produces, and re-running
the command rewrites it byte for byte.

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
| Labelled set | `{facts.labelled_path}` |
| SHA-256 of the labelled set | `{facts.labelled_sha256}` |

{_recall_section(found)}

## How a Contact Point is read

A Contact Point is an off-platform identifier somebody can be reached at, and this
build looks for two kinds and nothing else: an email address and a Telegram handle.
Both are read from a post's own title, its own body, and its links, with the obfuscation
tolerance stated below. Nothing that looks like one of these and is not one is dropped —
it is reported with which of nine faults applied:

{", ".join(f"`{reason.value}`" for reason in Unread)}

Reading loosely would be worse than reading narrowly here, and the reason is specific
to this step: a Contact Point is the one string in the Corpus that can join two
accounts on nothing else, so a value that was guessed becomes a grouping edge nobody
checked. A false positive here is not a wrong severity figure, which is what a Signal
can produce; it is a shared identifier between two accounts that share nothing.

**An identifier written out is read as the identifier underneath it.** A post that writes
`@s.y.n._.v.a.n.t.a.g.e.l.e.d.g.e.r` or `@s y n _ v a n t a g e l e d g e r` publishes the
same Contact Point as one that writes `@syn_vantageledger`, and a run of pieces that each
hold one character is what a post writing an identifier out looks like. The words beside
the identifier are not part of it: the run is read up to the first piece holding more than
one character, so `@ s y n _ v a n t a g e l e d g e r, or ask in here` is the handle and
not the sentence. **A full stop and a hyphen are dropped from a username and never from an
address**, because neither can occur in a Telegram username at all while a full stop is a
label boundary in both halves of an address, and `vantage.ledger.example` is not
`vantage-ledger.example`.

**A digit standing in for a letter is folded, and this is the one fold that can merge two
Contact Points that are both real.** `0` is read as `o`, `1` as `i`, `3` as `e`, `4` as
`a`, `5` as `s`, `7` as `t`, and `8` as `b`, so `@syn_v4ntag3ledger` and
`@syn_vantageledger` are one Contact Point here. Dropping a separator cannot do that: a
run holding a full stop was never a username, so the fold can only reach the one it was
hiding, whereas `syn_vantag3ledger` is a username somebody may have registered and is
counted as one. Every figure the fold can move is measured and printed — the recall
figure above, the `folded` line beside it, and the spellings under every value.

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
edge in order to be helpful, which is the failure it exists to avoid. An address is
folded for spacing and for substitution and for nothing else, so a local part written out
(`i n t a k e @vantage-ledger.example`) is read as `intake@vantage-ledger.example` while
`intake @ vantage-ledger.example` is not read at all.

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

Every figure above is a lower bound, and the measured recall above is what bounds it: the
Contact Points this Corpus publishes that were not read are named one per row in the
section above, so the shortfall can be read rather than inferred from a count. What the
tolerance does not reach is planted in the Corpus so that it is measured rather than
asserted: a handle published only as a picture of one is unreadable by anything in this
project, and an address written as words has no `@` in the post for any reader to find.

The one false positive the tolerance has that reading literally did not is stated rather
than hidden: a post that spaces out the letters of a *word* beside an `@` produces the
same shape as one writing out a handle, and `email me @ t o n i g h t at 8` is read as
the handle `tonight`. Nothing in the text tells the two apart. A Contact Point invented
is worse here than one missed — a miss is the floor the output already declares, an
invented value is a shared identifier ADR-0005 would group two accounts on — so the
reading is kept for the shapes that are common, this case is measured against the
labelled set, and the `invented` figure beside the recall is the number that would show
it happening in a Corpus of other content.

A handle is read as a Telegram handle whatever service it belongs to. Every service
writes one as `@name` and this build cannot tell them apart, so a Contact Point
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


def _recall_section(found: ContactPoints) -> str:
    """What the reading found against what the Corpus publishes, and every miss named.

    A recall figure with no list beside it is a number to take on trust, and this one
    bounds every other figure in the page: the Contact Points below are the ones a count
    of Contact Points is not counting. So each miss is a row carrying the post, the value,
    and how that post published it, and the false positives are named the same way rather
    than counted, because a reader has to be able to see which post to go and look at.
    """
    recall = found.recall
    lines = [
        "## What the reading found, and what it missed",
        "",
        f"Measured against `{found.facts.labelled_path}`, which `rfi generate-corpus` writes",
        "beside the Corpus: one row per post, naming every Contact Point that post",
        "publishes and whether it published it plainly, written out, or only as a",
        "picture of one. The figure is of *this* reading over *this* Corpus — a Corpus",
        "from a provider has no labelled set beside it, and measuring this reading",
        "over other content means labelling that content.",
        "",
        "| How it was published | Published | Found |",
        "| --- | ---: | ---: |",
        f"| Plainly | {recall.plain} | {recall.plain_found} |",
        f"| Written out | {recall.obfuscated} | {recall.obfuscated_found} |",
        f"| As a picture of one | {recall.image} | {recall.image_found} |",
        f"| All | {recall.published} | {recall.found} |",
        "",
        _misses(recall),
        "",
        _invented_rows(recall),
        "",
        _stray_rows(recall),
    ]
    return "\n".join(lines)


def _misses(recall: Recall) -> str:
    """Every Contact Point the Corpus publishes that this run did not read."""
    if not recall.missed:
        return f"""Every one of the {recall.published} Contact Points this Corpus publishes was read.
That is a figure about this Corpus's own contact details and not about contact
details in general: a Corpus that published none would read exactly the same way."""
    rows = "\n".join(
        f"| `{miss.post_id}` | `{miss.value}` | `{miss.written_as}` | {miss.written.value} |"
        for miss in recall.missed
    )
    return f"""{len(recall.missed)} of the {recall.published} Contact Points this Corpus publishes
were not read, and these are all of them:

| Post | Contact Point | Published as | Written |
| --- | --- | --- | --- |
{rows}"""


def _invented_rows(recall: Recall) -> str:
    """Every Contact Point read out of a post that publishes none.

    The figure that can produce a grouping edge nobody checked, so it is named per post
    rather than counted: the rate is the count and the rows are what a reviewer needs to
    decide whether the reading is wrong here or the label is.
    """
    if not recall.invented:
        return f"""No Contact Point was read out of any of the {recall.clean_posts} posts that publish
none, so the false-positive rate of this reading over this Corpus is 0 of
{recall.clean_posts}."""
    rows = "\n".join(
        f"| `{entry.post_id}` | `{entry.value}` | `{entry.written_as}` |"
        for entry in recall.invented
    )
    return f"""A Contact Point was read out of {len(recall.invented)} of the {recall.clean_posts} posts
that publish none, and each one is named so a reviewer can tell a reading that is
wrong here from a label that is:

| Post | Read as | Written as |
| --- | --- | --- |
{rows}"""


def _stray_rows(recall: Recall) -> str:
    """Where a value was read that the post publishing a Contact Point does not publish.

    A fold that has joined two values shows up here and nowhere else, and it is the only
    way this reading can put two accounts together that share nothing.
    """
    if not recall.stray:
        return """Every Contact Point read is the one its own post publishes, so no fold joined two of
the Contact Points this Corpus publishes."""
    rows = "\n".join(
        f"| `{entry.post_id}` | "
        f"{', '.join(f'`{value}`' for value in entry.published)} | `{entry.value}` |"
        for entry in recall.stray
    )
    return f"""{len(recall.stray)} of the values read are not the value the post publishing one
publishes, which is what a fold that has joined two Contact Points looks like from here:

| Post | Publishes | Read as |
| --- | --- | --- |
{rows}"""


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