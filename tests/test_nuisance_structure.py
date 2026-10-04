"""The Nuisance Structure is what makes a recovery number mean anything, so these
tests read the Corpus and ask what is in it.

Seam under test: the `generate-corpus` command, observed through the files it
writes. Nothing here inspects the code that wrote them, and the Corpus file is
the only thing a test reads to decide whether material is present — the manifest
is read for what it claims, and every claim in it is checked against the Corpus.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_SEED, main

REPO_ROOT = Path(__file__).parent.parent
SOURCE = REPO_ROOT / "src" / "reddit_fraud_intelligence"

# The two generated files the Corpus generator tests do not already hold. The committed
# Corpus is held by `test_the_committed_corpus_is_what_the_default_seed_produces`.
COMMITTED = (
    REPO_ROOT / "data" / "corpus" / "nuisance.jsonl",
    REPO_ROOT / "data" / "infrastructure" / "shared-hosts.jsonl",
)


class Generated(NamedTuple):
    corpus: Path
    truth: Path
    nuisance: Path
    shared_hosts: Path

# Every kind of Nuisance Structure the ticket asks for. A Corpus that has lost one
# of them is not a Corpus with an easier recovery number, it is a Corpus whose
# recovery number cannot be read, so the list is spelled out here and compared for
# equality rather than checked one at a time.
NUISANCE_KINDS = frozenset(
    {
        "decoy_account_cluster",
        "hard_negative",
        "known_shared_infrastructure",
        "near_miss_domain_pair",
        "obfuscated_contact",
        "single_account_domain",
        "staggered_paraphrase",
    }
)

# The characters of a Hard Negative the glossary names. Each has to be present for
# the false-grouping baseline to be the realistic one the glossary describes.
HARD_NEGATIVE_CHARACTERS = frozenset(
    {"genuine_job_post", "satire", "scam_adjacent_discussion", "scam_complaint"}
)

# The phrases a reader is most likely to key on, transcribed from the two Planted
# Campaigns rather than imported from the generator, so a Hard Negative carrying the
# wrong character cannot quietly agree with itself. Satire and a complaint from a victim
# both quote the pitch back at it; a genuine job post has no reason to.
PITCH_PHRASES = (
    "500 USDT",
    "onboarding call is compulsory",
    "losing months",
    "420 a day",
    "refundable deposit",
    "48 hours",
)

QUOTES_THE_PITCH = frozenset({"satire", "scam_complaint"})


def run_generator(
    directory: Path, seed: int = DEFAULT_SEED, command: str = "generate-corpus"
) -> Generated:
    paths = Generated(
        directory / "corpus.jsonl",
        directory / "truth.jsonl",
        directory / "nuisance.jsonl",
        directory / "shared-hosts.jsonl",
    )
    corpus_path, truth_path, nuisance_path, shared_path = paths
    exit_code = main(
        [
            command,
            "--seed",
            str(seed),
            "--corpus",
            str(corpus_path),
            "--truth",
            str(truth_path),
            "--nuisance",
            str(nuisance_path),
            "--shared-infrastructure",
            str(shared_path),
        ]
    )
    assert exit_code == 0
    return paths


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def text(row: dict[str, object], field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} is not text: {value!r}"
    return value


def texts(row: dict[str, object], field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} is not a list: {value!r}"
    return [str(item) for item in value]


def hosts_of(row: dict[str, object]) -> set[str]:
    hostnames = (urlparse(link).hostname for link in texts(row, "links"))
    return {hostname for hostname in hostnames if hostname is not None}


def accounts_per_host(
    corpus_rows: list[dict[str, object]],
) -> dict[str, set[str]]:
    """Which accounts touch which host, counting subdomains as hosts of their own."""
    touching: dict[str, set[str]] = {}
    for row in corpus_rows:
        for host in hosts_of(row):
            touching.setdefault(host, set()).add(text(row, "account"))
    return touching


def hosts_per_account(corpus_rows: list[dict[str, object]]) -> dict[str, set[str]]:
    touching: dict[str, set[str]] = {}
    for row in corpus_rows:
        for host in hosts_of(row):
            touching.setdefault(text(row, "account"), set()).add(host)
    return touching


def planted(truth_path: Path) -> tuple[set[str], set[str]]:
    accounts, posts = set(), set()
    for membership in rows(truth_path):
        accounts |= set(texts(membership, "accounts"))
        posts |= set(texts(membership, "posts"))
    return accounts, posts


def records_of_kind(
    records: list[dict[str, object]], kind: str
) -> list[dict[str, object]]:
    return [record for record in records if text(record, "kind") == kind]


def words(text_value: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in text_value.lower()).split()


def shingles(text_value: str, width: int = 5) -> set[str]:
    tokens = words(text_value)
    return {" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def overlap(left: str, right: str, width: int = 5) -> float:
    """How much of one body is repeated word for word in another."""
    left_shingles, right_shingles = shingles(left, width), shingles(right, width)
    if not left_shingles or not right_shingles:
        return 0.0
    return len(left_shingles & right_shingles) / len(left_shingles | right_shingles)


def one_edit_apart(left: str, right: str) -> bool:
    """Whether two names differ by a single insertion, deletion, or substitution."""
    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    return any(
        longer[:index] + longer[index + 1 :] == shorter
        for index in range(len(longer))
    )


def test_the_corpus_holds_domains_that_must_not_group(tmp_path: Path) -> None:
    """A domain one account uses is not a Campaign Candidate, and it has to be here
    for that to mean anything: a Corpus where every domain is shared would hand any
    grouping logic a perfect score for free."""
    corpus_path, truth_path, nuisance_path, _ = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    touching = accounts_per_host(corpus_rows)
    posted = {text(row, "post_id"): row for row in corpus_rows}
    planted_accounts = {
        account
        for membership in rows(truth_path)
        for account in texts(membership, "accounts")
    }

    single = records_of_kind(rows(nuisance_path), "single_account_domain")

    assert len(single) >= 2, "the Corpus holds no domain that must not group"
    for record in single:
        assert len(texts(record, "hosts")) == 1
        host = texts(record, "hosts")[0]

        assert len(touching[host]) == 1, f"{host} is shared by {sorted(touching[host])}"
        assert texts(record, "accounts") == sorted(touching[host])

        # One-off links: the post carries the domain and nothing else to group on.
        for post_id in texts(record, "posts"):
            assert hosts_of(posted[post_id]) == {host}

        assert not set(texts(record, "accounts")) & planted_accounts


def test_a_decoy_account_cluster_resembles_a_campaign_without_being_one(
    tmp_path: Path,
) -> None:
    """Accounts running a campaign's playbook without being a campaign. Without them the
    grouping logic is only ever asked to find groups already sitting in one obvious
    component, and the recovery number says nothing about precision."""
    corpus_path, truth_path, nuisance_path, shared_path = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    posted = {text(row, "post_id"): row for row in corpus_rows}
    planted_accounts, planted_posts = planted(truth_path)
    per_account = hosts_per_account(corpus_rows)
    shared = {text(row, "host") for row in rows(shared_path)}

    decoys = records_of_kind(rows(nuisance_path), "decoy_account_cluster")

    assert decoys, "the Corpus holds nothing that resembles a campaign but is not one"
    for record in decoys:
        accounts = texts(record, "accounts")
        post_ids = texts(record, "posts")
        assert len(accounts) >= 3
        assert texts(record, "accounts") == sorted(
            {text(posted[post_id], "account") for post_id in post_ids}
        )
        assert not set(post_ids) & planted_posts
        assert not set(accounts) & planted_accounts

    def template_similar(record: dict[str, object], floor: float) -> None:
        bodies = [text(posted[post_id], "body") for post_id in texts(record, "posts")]
        for left, right in combinations(bodies, 2):
            assert overlap(left, right) >= floor, (
                f"{text(record, 'nuisance_id')} repeats itself less than a cluster has "
                f"to, to judge by {floor}"
            )

    # Two shapes, and both are needed. Held apart by their infrastructure, a cluster is
    # what content similarity and time proximity look like with no domain underneath, and
    # neither is enough to group on (ADR-0005). Joined by a domain, a cluster is a
    # Campaign Candidate the system is right to produce and wrong to count as recovered.
    #
    # The two floors differ on purpose: copy-pasted adverts converge far harder than a
    # business repeating its own opening hours does, and a test that forced them to match
    # would be asking the Corpus to lie about one of them.
    held_apart = [
        record
        for record in decoys
        if all(
            per_account[left] & per_account[right] <= shared
            for left, right in combinations(texts(record, "accounts"), 2)
        )
    ]
    assert held_apart, (
        "no decoy cluster is held apart by its infrastructure, so the Corpus never asks "
        "whether content similarity on its own is enough to group"
    )
    for record in held_apart:
        template_similar(record, 0.3)

    joined_by_domain = [
        record
        for record in decoys
        if any(
            per_account[left] & per_account[right] - shared
            for left, right in combinations(texts(record, "accounts"), 2)
        )
    ]
    assert joined_by_domain, (
        "no decoy cluster shares a registrable domain, so every Campaign Candidate the "
        "grouping produces is a Planted Campaign and recovery has nothing to refuse"
    )
    for record in joined_by_domain:
        template_similar(record, 0.12)


def test_known_shared_infrastructure_is_in_the_corpus_and_published_as_data(
    tmp_path: Path,
) -> None:
    """Link shorteners, paste sites, and link-in-bio services are shared by everyone
    who uses them. The list is published as data so it can be regenerated, tuned, and
    eventually replaced by a reputation feed, rather than living in a query (ADR-0009).
    """
    corpus_path, truth_path, nuisance_path, shared_path = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    posted = {text(row, "post_id"): row for row in corpus_rows}
    planted_accounts, _ = planted(truth_path)
    touching = accounts_per_host(corpus_rows)
    shared_rows = rows(shared_path)
    records = records_of_kind(rows(nuisance_path), "known_shared_infrastructure")

    assert shared_rows, "no known-shared infrastructure is published as data"
    for row in shared_rows:
        host = text(row, "host")
        assert host in touching, f"{host} is published, but nothing in the Corpus uses it"
        assert len(touching[host]) >= 3, (
            f"{host} touches {sorted(touching[host])}: that is one account's domain, "
            "not shared infrastructure"
        )
        # A Planted Campaign that leans on shared infrastructure would be recovered
        # only if the filter were wrong, which makes the recovery number unmeasurable.
        assert not touching[host] & planted_accounts

        assert text(row, "provenance")
        assert text(row, "added")

    # The manifest's claim about each host has to be the Corpus's own truth.
    assert {texts(record, "hosts")[0] for record in records} == {
        text(row, "host") for row in shared_rows
    }
    for record in records:
        host = texts(record, "hosts")[0]
        linking = sorted(
            post_id
            for post_id, row in posted.items()
            if host in hosts_of(row)
        )
        assert texts(record, "posts") == linking
        assert texts(record, "accounts") == sorted(touching[host])

    # The list is data. Only the module that publishes the hosts may hold a host as a
    # literal; everything else builds links from those constants and every reader
    # downstream reads the file. An empty set is the state to aim for, not a failure: it
    # is what this looks like once the list is a reputation feed rather than a constant.
    for row in shared_rows:
        naming = {
            path.relative_to(SOURCE).as_posix()
            for path in SOURCE.rglob("*.py")
            if text(row, "host") in path.read_text(encoding="utf-8")
        }
        assert naming <= {"infrastructure.py"}, f"{text(row, 'host')} is written into {sorted(naming)}"


def test_the_corpus_uses_a_shortener_a_paste_site_and_a_link_in_bio_service(
    tmp_path: Path,
) -> None:
    """The three kinds of service every unrelated account touches. One kind on its own
    is a special case; all three together is what grouping has to survive."""
    _, _, _, shared_path = run_generator(tmp_path)

    assert {text(row, "kind") for row in rows(shared_path)} == {
        "link_shortener",
        "paste_site",
        "link_in_bio",
    }


def test_hard_negatives_sit_beside_the_fraud_boundary_without_being_planted(
    tmp_path: Path,
) -> None:
    """Legitimate content that a keyword reader and a text-similarity step both get
    wrong: a real job post, satire that recites the pitch, a complaint from someone who
    lost money, and a discussion post listing the tell-tale phrases."""
    corpus_path, truth_path, nuisance_path, _ = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    posted = {text(row, "post_id"): row for row in corpus_rows}
    per_account = hosts_per_account(corpus_rows)
    planted_accounts, planted_posts = planted(truth_path)
    campaign_hosts = {
        host for account in planted_accounts for host in per_account[account]
    }

    hard_negatives = records_of_kind(rows(nuisance_path), "hard_negative")
    characters: set[str] = set()
    for record in hard_negatives:
        assert len(texts(record, "characters")) == 1
        character = texts(record, "characters")[0]
        characters.add(character)

        assert len(texts(record, "posts")) == 1
        post_id = texts(record, "posts")[0]
        assert texts(record, "accounts") == [text(posted[post_id], "account")]
        assert not set(texts(record, "posts")) & planted_posts
        assert not set(texts(record, "accounts")) & planted_accounts

        # Beside the boundary: the Hard Negative touches no infrastructure a Planted
        # Campaign touches, so grouping it can only ever come from the text.
        assert not hosts_of(posted[post_id]) & campaign_hosts

        # The character is a claim about the text, so the text has to support it. A
        # genuine job post has no reason to carry the pitch; satire and a complaint from
        # a victim both quote it back, which is exactly what makes them hard.
        quoted = sum(phrase in text(posted[post_id], "body") for phrase in PITCH_PHRASES)
        if character in QUOTES_THE_PITCH:
            assert quoted >= 2, (
                f"a Hard Negative labelled {character} quotes {quoted} of the planted "
                "pitch's phrases, so it is not what it claims to be"
            )
        elif character == "genuine_job_post":
            assert quoted == 0, (
                "a genuine job post carrying the planted pitch's phrases is a Hard "
                "Negative of a different kind"
            )

    assert HARD_NEGATIVE_CHARACTERS <= characters

    # Discussion comes in both kinds — quoting the phrases it warns about, and talking
    # about the problem without using them — and at least one of them must quote.
    discussion = [
        text(posted[texts(record, "posts")[0]], "body")
        for record in hard_negatives
        if texts(record, "characters") == ["scam_adjacent_discussion"]
    ]
    assert any(
        sum(phrase in body for phrase in PITCH_PHRASES) >= 1 for body in discussion
    ), (
        "no piece of scam-adjacent discussion quotes the phrases it warns readers about, "
        "so it is commentary rather than the near-miss it is meant to be"
    )


def test_the_generator_records_the_hard_negative_count_and_its_character(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A count with no character attached cannot be interpreted: a false-grouping rate
    against four Hard Negatives and one against forty are different numbers."""
    _, _, nuisance_path, _ = run_generator(tmp_path)

    printed = capsys.readouterr().out
    hard_negatives = records_of_kind(rows(nuisance_path), "hard_negative")

    assert f"Hard Negatives {len(hard_negatives)}" in printed
    for character in HARD_NEGATIVE_CHARACTERS:
        assert character in printed


def test_a_near_miss_domain_sits_one_edit_from_a_planted_campaign_domain(
    tmp_path: Path,
) -> None:
    """A typo, a plural, a British spelling. Grouping on a name resemblance instead of
    on the registrable domain would merge a recruitment firm with a planted campaign, and
    the only way to find out is to have both of them in the Corpus."""
    corpus_path, truth_path, nuisance_path, _ = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    per_account = hosts_per_account(corpus_rows)
    planted_accounts, planted_posts = planted(truth_path)
    campaign_hosts = {host for account in planted_accounts for host in per_account[account]}

    pairs = records_of_kind(rows(nuisance_path), "near_miss_domain_pair")
    posted = {text(row, "post_id"): row for row in corpus_rows}

    assert pairs, "the Corpus holds no pair of domains that resemble each other"
    for record in pairs:
        hosts = texts(record, "hosts")
        assert len(hosts) == 2, "a near-miss pair is two domains, not a list of them"
        assert one_edit_apart(hosts[0].split(".")[0], hosts[1].split(".")[0]), (
            f"{hosts} do not differ by a single character, so they are not the hard case"
        )

        # Exactly one side belongs to a Planted Campaign. A pair where neither does
        # tests nothing, and a pair where both do is one domain, not two.
        assert sum(host in campaign_hosts for host in hosts) == 1
        near_miss = next(host for host in hosts if host not in campaign_hosts)
        users = {account for account, hosts_ in per_account.items() if near_miss in hosts_}
        assert users, f"{near_miss} is in the manifest but nothing links it"
        assert not users & planted_accounts

        assert texts(record, "accounts") == sorted(
            {text(posted[post_id], "account") for post_id in texts(record, "posts")}
        )
        assert not set(texts(record, "posts")) & planted_posts


def test_a_planted_campaign_is_staggered_paraphrases_of_one_text(tmp_path: Path) -> None:
    """The same offer, reworded, hours apart, across a Planted Campaign's accounts. A
    system that groups on repeated strings recovers this campaign for the wrong reason,
    and one that groups on text alone would recover half the Corpus."""
    corpus_path, truth_path, nuisance_path, _ = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    posted = {text(row, "post_id"): row for row in corpus_rows}
    memberships = rows(truth_path)

    records = records_of_kind(rows(nuisance_path), "staggered_paraphrase")

    assert records, "no Planted Campaign's text is paraphrased across its accounts"
    for record in records:
        post_ids = texts(record, "posts")
        accounts = texts(record, "accounts")
        assert len(accounts) >= 2, "a paraphrase group needs more than one account"

        # The group sits inside exactly one Planted Campaign, and it is that
        # campaign's whole membership rather than a slice of it.
        inside = [m for m in memberships if set(texts(m, "posts")) >= set(post_ids)]
        assert len(inside) == 1
        assert set(texts(inside[0], "accounts")) == set(accounts)

        bodies = [text(posted[post_id], "body") for post_id in post_ids]

        # One offer, different words. Identical bodies would pass a string match, and
        # near-identical ones would pass a similarity match, so neither is allowed.
        assert len(set(bodies)) == len(bodies)
        for left, right in combinations(bodies, 2):
            assert overlap(left, right) < 0.4, (
                "two accounts in the group repeat each other closely enough that "
                "matching the text is enough to find the campaign"
            )
        assert all("500 USDT" in body for body in bodies), (
            "the paraphrases have drifted so far apart they are not the same offer"
        )

        # Staggered: in order, and spread over hours rather than minutes.
        times = [text(posted[post_id], "created_at") for post_id in post_ids]
        assert times == sorted(times)
        assert len(set(times)) == len(times)
        stamps = [datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ") for stamp in times]
        assert stamps[-1] - stamps[0] >= timedelta(hours=2)


def test_no_nuisance_identifier_is_named_in_the_corpus_file(tmp_path: Path) -> None:
    """ADR-0008 holds for the Nuisance Structure too. A generator that labelled its own
    material would make every grouping result unfalsifiable, because a reader could no
    longer tell a hard case from an easy one."""
    corpus_path, _, nuisance_path, _ = run_generator(tmp_path)

    corpus_text = corpus_path.read_text(encoding="utf-8")

    for record in rows(nuisance_path):
        assert text(record, "nuisance_id") not in corpus_text


def test_every_kind_of_nuisance_is_planted_in_the_corpus_whatever_the_seed(
    tmp_path: Path,
) -> None:
    """A Corpus that has quietly lost its decoy clusters or its near-miss pairs still
    produces a recovery figure, and the figure is worse for it. Comparing the set for
    equality means losing one fails here rather than quietly raising the score."""
    for offset in range(4):
        seed = DEFAULT_SEED + offset
        corpus_path, _, nuisance_path, _ = run_generator(tmp_path / str(offset), seed)

        records = rows(nuisance_path)
        in_corpus = {text(row, "post_id") for row in rows(corpus_path)}

        assert {text(record, "kind") for record in records} == NUISANCE_KINDS, (
            f"seed {seed} is missing a kind of Nuisance Structure"
        )
        for record in records:
            assert set(texts(record, "posts")) <= in_corpus, (
                f"{text(record, 'nuisance_id')} claims posts the Corpus does not hold"
            )


def test_the_same_seed_writes_byte_identical_nuisance_files(tmp_path: Path) -> None:
    first = run_generator(tmp_path / "first")
    second = run_generator(tmp_path / "second")

    assert first.nuisance.read_bytes() == second.nuisance.read_bytes()
    assert first.shared_hosts.read_bytes() == second.shared_hosts.read_bytes()


def test_the_seed_varies_the_nuisance_structure_rather_than_only_the_minutes(
    tmp_path: Path,
) -> None:
    """The seed picks what nuisance material to plant. A seed that changed nothing but
    timestamps would give one measurement wearing many seeds' clothes. Four seeds are
    compared rather than two, so the assertion is about the seed choosing something and
    not about one particular pair of draws landing differently."""
    drawn = {
        tuple(
            text(record, "nuisance_id")
            for record in rows(run_generator(tmp_path / str(offset), DEFAULT_SEED + offset).nuisance)
        )
        for offset in range(4)
    }

    assert len(drawn) > 1, "four seeds planted one Nuisance Structure between them"


def test_the_committed_manifest_is_what_the_default_seed_produces(
    tmp_path: Path,
) -> None:
    generated = run_generator(tmp_path)

    for written, committed in (
        (generated.nuisance, COMMITTED[0]),
        (generated.shared_hosts, COMMITTED[1]),
    ):
        assert written.read_bytes() == committed.read_bytes(), committed.name


def test_generation_refuses_to_write_the_manifest_over_the_membership(
    tmp_path: Path,
) -> None:
    """Each projection has a reader that assumes it is the only thing at that path, so a
    collision silently destroys one of them rather than failing."""
    shared = tmp_path / "one.jsonl"

    with pytest.raises(SystemExit):
        main(
            [
                "generate-corpus",
                "--seed",
                str(DEFAULT_SEED),
                "--corpus",
                str(tmp_path / "corpus.jsonl"),
                "--truth",
                str(shared),
                "--nuisance",
                str(shared),
                "--shared-infrastructure",
                str(tmp_path / "shared-hosts.jsonl"),
            ]
        )

    assert not shared.exists()
