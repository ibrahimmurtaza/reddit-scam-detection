"""Known-shared infrastructure: the hosts everyone uses, published as data.

A link shortener, a paste site, or a link-in-bio page is shared by thousands of
accounts that have nothing to do with each other. Grouping on one of those hosts
produces a single enormous component containing most of the Corpus, so the filter
that excludes them is held here as data rather than written into the grouping query:
it can be regenerated, tuned, and — once an authorised Corpus Provider exists —
replaced with an actual reputation feed (ADR-0009).

The hosts are named once, here, and published to `data/infrastructure/`. Other
modules build links from these constants rather than repeating the host, so a test
that greps `src/` for the literal finds one file and no others. Every reader
downstream reads the published file instead.

Every host sits under the reserved `.example` TLD (RFC 2606), so it corresponds to
no real service and nothing in the output can be mistaken for real data.

To change the list, change `SHARED_HOSTS` and re-run `rfi generate-corpus`. The
published file is derived and is never edited by hand, so a host cannot be in the
list without a host in the Corpus that uses it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from reddit_fraud_intelligence.jsonl import JsonObject, write_lines


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


HOP_CUT = "hopcut.example"
PASTE_VAULT = "pastevault.example"
BIO_PAGE = "biopage.example"

_PROVENANCE = (
    "Written by hand for the portfolio build. A build with an authorised Corpus "
    "Provider replaces this list with a domain-reputation feed."
)
_ADDED = "2026-09-30"

SHARED_HOSTS: tuple[SharedHost, ...] = (
    SharedHost(
        host=HOP_CUT,
        kind=SharedHostKind.LINK_SHORTENER,
        added=_ADDED,
        provenance=_PROVENANCE,
    ),
    SharedHost(
        host=PASTE_VAULT,
        kind=SharedHostKind.PASTE_SITE,
        added=_ADDED,
        provenance=_PROVENANCE,
    ),
    SharedHost(
        host=BIO_PAGE,
        kind=SharedHostKind.LINK_IN_BIO,
        added=_ADDED,
        provenance=_PROVENANCE,
    ),
)


def write_shared_infrastructure(path: Path, hosts: Iterable[SharedHost]) -> None:
    """Write the list, sorted by host so a fixed list writes byte-identical bytes."""

    def objects() -> Iterable[JsonObject]:
        for host in sorted(hosts, key=lambda host: host.host):
            yield {
                "host": host.host,
                "kind": host.kind,
                "added": host.added,
                "provenance": host.provenance,
            }

    write_lines(path, objects())
