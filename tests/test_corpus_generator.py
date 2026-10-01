"""The Corpus file is the credibility boundary, so these tests read the file.

Seam under test: the `generate-corpus` command, observed through the two files it
writes. Nothing here inspects the code that wrote them.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from reddit_fraud_intelligence.cli import DEFAULT_SEED, main

COMMITTED_CORPUS = Path(__file__).parent.parent / "data" / "corpus" / "corpus.jsonl"

CORPUS_FIELDS = frozenset(
    {"post_id", "account", "subreddit", "title", "body", "created_at", "links"}
)

# A membership field, by any name a generator might reach for. Checked against the
# field names on the file rather than its text, so that legitimate post copy can
# never trip it.
MEMBERSHIP_FIELDS = frozenset(
    {
        "campaign",
        "campaign_id",
        "campaigns",
        "ground_truth",
        "is_fraud",
        "label",
        "labels",
        "membership",
        "planted",
        "truth",
    }
)

# A quotation of the planted pitch, so a reviewer can tell from a failure message
# that the content is still readable rather than truncated. Update it when the
# planted copy is rewritten.
PLANTED_PITCH = (
    "What sold me is that they publish the month's result whether it was green or not, "
    "including the two months it lost."
)


def run_generator(directory: Path, seed: int = DEFAULT_SEED) -> tuple[Path, Path]:
    corpus_path = directory / "corpus.jsonl"
    truth_path = directory / "truth.jsonl"
    exit_code = main(
        [
            "generate-corpus",
            "--seed",
            str(seed),
            "--corpus",
            str(corpus_path),
            "--truth",
            str(truth_path),
            "--nuisance",
            str(directory / "nuisance.jsonl"),
            "--shared-infrastructure",
            str(directory / "shared-hosts.jsonl"),
        ]
    )
    assert exit_code == 0
    return corpus_path, truth_path


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


def test_corpus_file_carries_no_membership_fields(tmp_path: Path) -> None:
    corpus_path, _ = run_generator(tmp_path)

    corpus_rows = rows(corpus_path)
    assert corpus_rows, "the generator wrote no Corpus items"

    for row in corpus_rows:
        assert set(row) == CORPUS_FIELDS
        assert not set(row) & MEMBERSHIP_FIELDS


def test_the_corpus_holds_a_few_accounts_and_shared_domains_to_look_at(
    tmp_path: Path,
) -> None:
    """The Corpus is deliberately small enough to read in full, and still shaped like
    something worth grouping: there are fewer accounts than posts, so some accounts
    post more than once, and there are far more hosts than Planted Campaigns. Widen
    these bounds deliberately when the Corpus changes shape, not by accident.
    """
    corpus_path, _ = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)

    hosts = {
        urlparse(link).hostname
        for row in corpus_rows
        for link in texts(row, "links")
    }
    accounts = {text(row, "account") for row in corpus_rows}

    assert 10 < len(accounts) < len(corpus_rows)
    assert 8 < len(hosts) < len(corpus_rows)


def test_same_seed_writes_byte_identical_files(tmp_path: Path) -> None:
    first_corpus, first_truth = run_generator(tmp_path / "first")
    second_corpus, second_truth = run_generator(tmp_path / "second")

    assert first_corpus.read_bytes() == second_corpus.read_bytes()
    assert first_truth.read_bytes() == second_truth.read_bytes()


def test_a_different_seed_writes_a_different_corpus(tmp_path: Path) -> None:
    first_corpus, _ = run_generator(tmp_path / "first")
    second_corpus, _ = run_generator(tmp_path / "second", seed=DEFAULT_SEED + 1)

    assert first_corpus.read_bytes() != second_corpus.read_bytes()


def test_the_committed_corpus_is_what_the_default_seed_produces(tmp_path: Path) -> None:
    corpus_path, _ = run_generator(tmp_path)

    assert corpus_path.read_bytes() == COMMITTED_CORPUS.read_bytes()


def test_generation_refuses_to_write_both_files_to_one_path(tmp_path: Path) -> None:
    shared = tmp_path / "one.jsonl"

    with pytest.raises(SystemExit):
        main(
            [
                "generate-corpus",
                "--seed",
                str(DEFAULT_SEED),
                "--corpus",
                str(shared),
                "--truth",
                str(shared),
                "--nuisance",
                str(tmp_path / "nuisance.jsonl"),
                "--shared-infrastructure",
                str(tmp_path / "shared-hosts.jsonl"),
            ]
        )

    assert not shared.exists()


def test_truth_file_holds_membership_that_matches_the_corpus(tmp_path: Path) -> None:
    corpus_path, truth_path = run_generator(tmp_path)
    corpus_rows = rows(corpus_path)
    memberships = rows(truth_path)

    accounts = {text(row, "account") for row in corpus_rows}
    post_ids = {text(row, "post_id") for row in corpus_rows}

    assert memberships
    for membership in memberships:
        assert set(texts(membership, "accounts")) <= accounts
        assert set(texts(membership, "posts")) <= post_ids
        # A grouping of one is not a grouping: ADR-0005 requires two or more.
        assert len(texts(membership, "accounts")) >= 2
        assert len(texts(membership, "posts")) >= 2


def test_no_planted_campaign_is_named_in_the_corpus_file(tmp_path: Path) -> None:
    corpus_path, truth_path = run_generator(tmp_path)
    corpus_text = corpus_path.read_text(encoding="utf-8")

    for membership in rows(truth_path):
        assert text(membership, "campaign_id") not in corpus_text


def test_the_planted_post_can_be_read_to_judge_its_realism(tmp_path: Path) -> None:
    corpus_path, _ = run_generator(tmp_path)

    corpus_text = corpus_path.read_text(encoding="utf-8")

    assert PLANTED_PITCH in corpus_text


def test_every_synthetic_entity_is_marked_as_synthetic(tmp_path: Path) -> None:
    corpus_path, _ = run_generator(tmp_path)

    for row in rows(corpus_path):
        assert text(row, "post_id").startswith("syn_")
        assert text(row, "account").startswith("syn_")
        for link in texts(row, "links"):
            hostname = urlparse(link).hostname
            assert hostname is not None
            assert hostname.endswith(".example")
