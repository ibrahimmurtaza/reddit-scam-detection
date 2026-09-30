"""The command line — the only interface in v1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from reddit_fraud_intelligence.cafc import (
    SOURCE_URL,
    download_extract,
    render_report,
    survey,
    write_base_rates,
    write_provenance,
)
from reddit_fraud_intelligence.corpus import write_corpus
from reddit_fraud_intelligence.generator import corpus_items, planted_campaigns
from reddit_fraud_intelligence.truth import write_truth

DEFAULT_SEED = 20260930
DEFAULT_CORPUS_PATH = Path("data/corpus/corpus.jsonl")
DEFAULT_TRUTH_PATH = Path("data/corpus/truth.jsonl")
DEFAULT_EXTRACT_PATH = Path("data/cafc/cafc-extract.csv.gz")
DEFAULT_PROVENANCE_PATH = Path("data/cafc/provenance.jsonl")
DEFAULT_BASE_RATES_PATH = Path("data/cafc/base_rates.jsonl")
DEFAULT_BASE_RATES_REPORT_PATH = Path("docs/cafc-base-rates.md")


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
        case "fetch-cafc":
            return _fetch_cafc(
                extract_path=Path(str(args.extract)),
                force=bool(args.force),
            )
        case "cafc-report":
            return _cafc_report(
                extract_path=Path(str(args.extract)),
                provenance_path=Path(str(args.provenance)),
                base_rates_path=Path(str(args.base_rates)),
                report_path=Path(str(args.report)),
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

    fetch = commands.add_parser(
        "fetch-cafc",
        help="download the CAFC extract and cache it, compressed",
        description=(
            "The one command here that needs the network. CAFC's extract is published "
            "quarterly and grows without bound, so a cached copy is kept and committed: "
            "everything else reads that cache and needs no network at all. Refuses to "
            "replace a cache that is already there, because a new release would move "
            "every base rate and quietly change what the Corpus is compared against."
        ),
    )
    fetch.add_argument(
        "--extract",
        type=Path,
        default=DEFAULT_EXTRACT_PATH,
        help=f"where to cache the extract (default: {DEFAULT_EXTRACT_PATH})",
    )
    fetch.add_argument(
        "--force",
        action="store_true",
        help="replace a cache that is already there, moving every base rate",
    )

    report = commands.add_parser(
        "cafc-report",
        help="read the cached CAFC extract and record what it holds",
        description=(
            "Read the cache, compute the base rate of every thematic category, and write "
            "the provenance, the base rates, and a reader-facing report. Every figure is "
            "computed here from the bytes, so a figure and the extract cannot drift apart. "
            "Reads no network."
        ),
    )
    report.add_argument(
        "--extract",
        type=Path,
        default=DEFAULT_EXTRACT_PATH,
        help=f"the cached extract to read (default: {DEFAULT_EXTRACT_PATH})",
    )
    report.add_argument(
        "--provenance",
        type=Path,
        default=DEFAULT_PROVENANCE_PATH,
        help=(
            "where to record the report count, date range, and licence "
            f"(default: {DEFAULT_PROVENANCE_PATH})"
        ),
    )
    report.add_argument(
        "--base-rates",
        type=Path,
        default=DEFAULT_BASE_RATES_PATH,
        help=f"where to write the base rates (default: {DEFAULT_BASE_RATES_PATH})",
    )
    report.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_BASE_RATES_REPORT_PATH,
        help=f"where to write the report (default: {DEFAULT_BASE_RATES_REPORT_PATH})",
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


def _fetch_cafc(*, extract_path: Path, force: bool) -> int:
    if extract_path.exists() and not force:
        raise SystemExit(
            f"refusing to replace a cache that is already there: {extract_path}. "
            "Pass --force to fetch a new release, and expect every base rate to move."
        )

    written = download_extract(extract_path)
    print(f"extract        {extract_path} ({written:,} bytes uncompressed, cached gzipped)")
    print(f"source         {SOURCE_URL}")
    print("now run `rfi cafc-report` to record what it holds")
    return 0


def _cafc_report(
    *,
    extract_path: Path,
    provenance_path: Path,
    base_rates_path: Path,
    report_path: Path,
) -> int:
    facts, base_rates = survey(extract_path)
    write_provenance(provenance_path, facts)
    write_base_rates(base_rates_path, base_rates)
    _write_text(report_path, render_report(facts, base_rates))

    free_text = "ruled out" if facts.free_text_ruled_out else "NOT ruled out"
    largest = ", ".join(f"{rate.category} {rate.base_rate:.2%}" for rate in base_rates[:3])
    print(f"extract        {extract_path}")
    print(f"reports        {facts.reports:,} ({facts.date_from} to {facts.date_to})")
    print(
        f"categories     {facts.categories}, longest value in any column "
        f"{facts.longest_value_chars} characters"
    )
    print(f"free text      {free_text}")
    print(f"largest        {largest}")
    print(f"provenance     {provenance_path}")
    print(f"base rates     {base_rates_path}")
    print(f"report         {report_path}")
    return 0


def _write_text(path: Path, text: str) -> None:
    """LF regardless of the checkout, so a committed report stays byte-stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
