"""The command line — the only interface in v1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from reddit_fraud_intelligence.corpus import write_corpus
from reddit_fraud_intelligence.generator import corpus_items, planted_campaigns
from reddit_fraud_intelligence.truth import write_truth

DEFAULT_SEED = 20260930
DEFAULT_CORPUS_PATH = Path("data/corpus/corpus.jsonl")
DEFAULT_TRUTH_PATH = Path("data/corpus/truth.jsonl")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    match args.command:
        case "generate-corpus":
            return _generate_corpus(
                seed=int(args.seed),
                corpus_path=Path(str(args.corpus)),
                truth_path=Path(str(args.truth)),
            )
        case _:
            parser.error(f"unknown command: {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfi",
        description=(
            "Ranks Reddit content for reviewer attention. It does not detect fraud: it "
            "ranks content for review and shows its reasoning."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate-corpus",
        help="write a synthetic Corpus and its Planted Campaign membership",
        description=(
            "Write two files: a Corpus holding content, accounts, and links, and a truth "
            "file holding Planted Campaign membership at a path of its own. Same seed, same "
            "bytes."
        ),
    )
    generate.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"fixed seed for a reproducible Corpus (default: {DEFAULT_SEED})",
    )
    generate.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"where to write the Corpus (default: {DEFAULT_CORPUS_PATH})",
    )
    generate.add_argument(
        "--truth",
        type=Path,
        default=DEFAULT_TRUTH_PATH,
        help=f"where to write Planted Campaign membership (default: {DEFAULT_TRUTH_PATH})",
    )
    return parser


def _generate_corpus(*, seed: int, corpus_path: Path, truth_path: Path) -> int:
    if corpus_path.resolve() == truth_path.resolve():
        raise SystemExit(
            f"refusing to write the Corpus and the membership to one path: {corpus_path}"
        )

    items = corpus_items(seed)
    campaigns = planted_campaigns(seed)
    write_corpus(corpus_path, items)
    write_truth(truth_path, campaigns)

    accounts = {item.account for item in items}
    print(f"seed           {seed}")
    print(f"corpus         {corpus_path} ({len(items)} posts, {len(accounts)} accounts)")
    print(
        f"truth          {truth_path} "
        f"({len(campaigns)} Planted Campaigns, {sum(len(c.posts) for c in campaigns)} posts)"
    )
    print("the Corpus holds content, accounts, and links; grep it for `campaign` to check")
    return 0
