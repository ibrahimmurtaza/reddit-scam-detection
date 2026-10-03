"""Known-shared infrastructure: the hosts everyone uses, published as data.

A link shortener, a paste site, or a link-in-bio page is shared by thousands of
accounts that have nothing to do with each other. Grouping on one of those hosts
produces a single enormous component containing most of the Corpus, so the filter
that excludes them is held here as data rather than written into the grouping query:
it can be regenerated, tuned, and â€” once an authorised Corpus Provider exists â€”
replaced with an actual reputation feed (ADR-0009).

The hosts are named once, here, and published to `data/infrastructure/`. Other
modules build links from these constants rather than repeating the host, so a test
that greps `src/` for the literal finds one file and no others. The grouping step
does not use those constants: it reads the published file, so adding a host to the
list is a change to a file and not to the query that filters on it (ADR-0011).

Every host sits under the reserved `.example` TLD (RFC 2606), so it corresponds to
no real service and nothing in the output can be mistaken for real data.

To change the list, change `SHARED_HOSTS` and re-run `rfi generate-corpus`. The
published file is derived and is never edited by hand, so a host cannot be in the
list without a host in the Corpus that uses it.

Reading the list resolves each host to its Registrable Domain through the same Public
Suffix List every other step resolves links against, and withholding is whole-domain:
one host on the list withholds the registration it names. A list published at the
granularity it means is therefore a list whose entries are Registrable Domains, and a
host published on somebody else's subdomain withholds that subdomain's registration
with it â€” which is why the host form is kept only because that is what a reputation
feed reports.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import date
from enum import StrEnum
from pathlib import Path

from reddit_fraud_intelligence.jsonl import JsonObject, read_rows, write_lines
from reddit_fraud_intelligence.suffixes import PublicSuffixes


class SharedHostKind(StrEnum):
    """What kind of service a shared host is. The kind is what a reader needs in
    order to judge whether the host belongs on the list at all."""

    LINK_SHORTENER = "link_shortener"
    PASTE_SITE = "paste_site"
    LINK_IN_BIO = "link_in_bio"


@dataclass(frozen=True, slots=True)
class SharedHost:
    """One host shared by unrelated accounts, with where the row came from."""

    host: str
    kind: SharedHostKind
    added: str
    provenance: str


@dataclass(frozen=True, slots=True)
class SharedRegistration:
    """One registration the list withholds, and the published hosts that say so.

    Two rows can resolve to one registration â€” a shortener and its mirror â€” and the
    registration is withheld once because that is what a host on it means.
    """

    registration: str
    hosts: tuple[SharedHost, ...]


@dataclass(frozen=True, slots=True)
class SharedInfrastructure:
    """The published list as the grouping step reads it, and what it withholds.

    `hosts` is the file's own rows. `registrations` is what they withhold: the
    Registrable Domains, so the filter compares like with like and never asks whether
    a shared hostname happens to equal a registration. `last_updated` is the newest
    date any row carries — read out of the data rather than kept beside it, because a
    second date to maintain is a second thing to forget when a host is added, and a
    date that drifts from the rows is worse than no date.
    """

    hosts: tuple[SharedHost, ...]
    registrations: tuple[SharedRegistration, ...]
    last_updated: str
    path: str

    @classmethod
    def read(cls, path: Path, suffixes: PublicSuffixes) -> SharedInfrastructure:
        """The list, its registrations, and the date it was last updated.

        Every row is checked as it is read, and a row that cannot be used stops the
        run. A host that names no registration would withhold nothing at all, and a
        filter that quietly did nothing is the one failure this module exists to
        prevent: the output would report groupings it believes it has filtered.
        """
        hosts = tuple(_host(path, number, text) for number, text in read_rows(path))
        return cls(
            hosts=hosts,
            registrations=_registrations(path, hosts, suffixes),
            last_updated=max((host.added for host in hosts), default=""),
            path=path.as_posix(),
        )

    def withheld(self) -> frozenset[str]:
        """The registrations to keep out of the graph, as the grouping asks for them.

        Asked for rather than held, so the caller cannot hold a set that has drifted
        from the rows, and built once per run rather than on every domain a post links.
        """
        return frozenset(shared.registration for shared in self.registrations)


HOP_CUT = "hopcut.example"
PASTE_VAULT = "pastevault.example"
BIO_PAGE = "biopage.example"

_ADDED = "2026-09-30"
_REPLACED = (
    "A build with an authorised Corpus Provider replaces this with a "
    "domain-reputation feed."
)

SHARED_HOSTS: tuple[SharedHost, ...] = (
    SharedHost(
        host=HOP_CUT,
        kind=SharedHostKind.LINK_SHORTENER,
        added=_ADDED,
        provenance=(
            "Hand-named for the portfolio build: a shortener anybody can create, so it is "
            f"shared by definition. {_REPLACED}"
        ),
    ),
    SharedHost(
        host=PASTE_VAULT,
        kind=SharedHostKind.PASTE_SITE,
        added=_ADDED,
        provenance=(
            "Hand-named for the portfolio build: a paste service that holds text for "
            f"anyone who has the link. {_REPLACED}"
        ),
    ),
    SharedHost(
        host=BIO_PAGE,
        kind=SharedHostKind.LINK_IN_BIO,
        added=_ADDED,
        provenance=(
            "Hand-named for the portfolio build: a link-in-bio page fronts other people's "
            f"sites and hides which one this is. {_REPLACED}"
        ),
    ),
)


def write_shared_infrastructure(path: Path, hosts: Iterable[SharedHost]) -> None:
    """Write the list, sorted by host so a fixed list writes byte-identical bytes.

    The row is the dataclass, so a field cannot be added to `SharedHost` and left out
    of the file, and a field cannot be in the file that the reader does not check.
    """

    def objects() -> Iterable[JsonObject]:
        for host in sorted(hosts, key=lambda host: host.host):
            yield asdict(host)

    write_lines(path, objects())


def _host(path: Path, number: int, line: str) -> SharedHost:
    """One row, checked field by field.

    Checked here rather than trusted because the list is meant to be replaced by a
    feed nobody in this repository controls: a misspelt kind, a date that is not a
    date, and above all a host that names no registration are all mistakes a reader
    could not see in the output, and each of them changes what the filter does.
    """
    try:
        record = json.loads(line)
    except json.JSONDecodeError as refusal:
        raise ValueError(_refusal(path, f"is not JSON: {line!r}", number)) from refusal
    if not isinstance(record, dict):
        raise ValueError(_refusal(path, f"is not a row: {line!r}", number))

    vocabulary = tuple(field.name for field in fields(SharedHost))
    if set(record) != set(vocabulary):
        raise ValueError(
            _refusal(
                path,
                f"holds {sorted(record)}, which is not the shared-host vocabulary "
                f"{sorted(vocabulary)}",
                number,
            )
        )

    row = {name: _field(path, record, name, number) for name in vocabulary}
    try:
        kind = SharedHostKind(row["kind"])
    except ValueError as unknown_kind:
        kinds = ", ".join(kind.value for kind in SharedHostKind)
        raise ValueError(
            _refusal(path, f"names {row['kind']!r}, and the kinds are {kinds}", number)
        ) from unknown_kind
    try:
        added = date.fromisoformat(row["added"]).isoformat()
    except ValueError as not_a_date:
        raise ValueError(
            _refusal(path, f"has added={row['added']!r}, which is not a date", number)
        ) from not_a_date

    return SharedHost(
        host=row["host"], kind=kind, added=added, provenance=row["provenance"]
    )


def _field(path: Path, record: JsonObject, field: str, number: int) -> str:
    """One field of a row, as text. Blank and non-text are refused as much as absent."""
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(_refusal(path, f"has no {field}", number))
    return value


def _registrations(
    path: Path, hosts: Iterable[SharedHost], suffixes: PublicSuffixes
) -> tuple[SharedRegistration, ...]:
    """What the list withholds: the Registrable Domain of every host on it.

    Sorted by registration, and the hosts under one registration sorted by host, so a
    fixed list produces one fixed output however the file happens to be ordered.
    """
    by_registration: dict[str, list[SharedHost]] = {}
    for host in hosts:
        registration = suffixes.registrable_domain(host.host)
        if registration is None:
            raise ValueError(
                _refusal(
                    path,
                    f"names {host.host}, which names no registrable domain, so it would "
                    "withhold nothing",
                )
            )
        by_registration.setdefault(registration, []).append(host)
    return tuple(
        SharedRegistration(
            registration=registration,
            hosts=tuple(sorted(on, key=lambda host: host.host)),
        )
        for registration, on in sorted(by_registration.items())
    )


def _refusal(path: Path, complaint: str, number: int | None = None) -> str:
    """One refusal, naming the file, the row, and what is wrong with it.

    `path:number` because a list with a mistake in it is read one row at a time. The
    number is left off for a complaint about the list as a whole rather than one row.
    """
    where = f"{path.as_posix()}:{number}" if number else path.as_posix()
    return f"{where} {complaint}"

