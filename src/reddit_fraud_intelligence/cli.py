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
from reddit_fraud_intelligence.campaigns import (
    group as group_accounts,
    render_table as render_candidates,
    write_campaign_candidates,
)
from reddit_fraud_intelligence.categories import OTHER, SCAM_CATEGORIES, TOP_LEVEL_COUNT
from reddit_fraud_intelligence.corpus import read_corpus, write_corpus
from reddit_fraud_intelligence.domains import (
    post_domains,
    render_report as render_domains_report,
    survey as survey_domains,
    write_post_domains,
)
from reddit_fraud_intelligence.generator import (
    corpus_items,
    nuisance_records,
    planted_campaigns,
)
from reddit_fraud_intelligence.infrastructure import SHARED_HOSTS, write_shared_infrastructure
from reddit_fraud_intelligence.nuisance import NuisanceKind, write_nuisance
from reddit_fraud_intelligence.projection import (
    place,
    read_base_rates,
    render_report as render_projection_report,
    total,
    unplaced_reports,
    write_mapping,
)
from reddit_fraud_intelligence.signals import (
    render_table as render_policy_score,
    score_corpus,
    write_policy_scores,
)
from reddit_fraud_intelligence.suffixes import (
    SOURCE_URL as SUFFIX_LIST_SOURCE_URL,
    PublicSuffixes,
    download_list,
    survey as survey_suffix_list,
    write_provenance as write_suffix_list_provenance,
)
from reddit_fraud_intelligence.truth import write_truth

DEFAULT_SEED = 20260930
DEFAULT_CORPUS_PATH = Path("data/corpus/corpus.jsonl")
DEFAULT_TRUTH_PATH = Path("data/corpus/truth.jsonl")
DEFAULT_NUISANCE_PATH = Path("data/corpus/nuisance.jsonl")
DEFAULT_SHARED_INFRASTRUCTURE_PATH = Path("data/infrastructure/shared-hosts.jsonl")
DEFAULT_EXTRACT_PATH = Path("data/cafc/cafc-extract.csv.gz")
DEFAULT_PROVENANCE_PATH = Path("data/cafc/provenance.jsonl")
DEFAULT_BASE_RATES_PATH = Path("data/cafc/base_rates.jsonl")
DEFAULT_BASE_RATES_REPORT_PATH = Path("docs/cafc-base-rates.md")
DEFAULT_SUFFIX_LIST_PATH = Path("data/public-suffix/public_suffix_list.dat")
DEFAULT_SUFFIX_PROVENANCE_PATH = Path("data/public-suffix/provenance.jsonl")
DEFAULT_DOMAINS_PATH = Path("data/domains/post-domains.jsonl")
DEFAULT_DOMAINS_REPORT_PATH = Path("docs/post-domains.md")
DEFAULT_CATEGORY_MAPPING_PATH = Path("data/cafc/scam-categories.jsonl")
DEFAULT_CATEGORY_REPORT_PATH = Path("docs/scam-categories.md")
DEFAULT_CANDIDATES_PATH = Path("data/campaigns/campaign-candidates.jsonl")
DEFAULT_WEIGHTS_PATH = Path("data/signals/weights.jsonl")
DEFAULT_SCORES_PATH = Path("data/signals/policy-scores.jsonl")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    match args.command:
        case "generate-corpus":
            return _generate_corpus(
                seed=int(args.seed),
                corpus_path=Path(str(args.corpus)),
                truth_path=Path(str(args.truth)),
                nuisance_path=Path(str(args.nuisance)),
                shared_path=Path(str(args.shared_infrastructure)),
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
        case "fetch-suffix-list":
            return _fetch_suffix_list(
                list_path=Path(str(args.list)),
                provenance_path=Path(str(args.provenance)),
                force=bool(args.force),
            )
        case "post-domains":
            return _post_domains(
                corpus_path=Path(str(args.corpus)),
                list_path=Path(str(args.list)),
                domains_path=Path(str(args.domains)),
                report_path=Path(str(args.report)),
            )
        case "scam-categories":
            return _scam_categories(
                base_rates_path=Path(str(args.base_rates)),
                mapping_path=Path(str(args.mapping)),
                report_path=Path(str(args.report)),
            )
        case "campaign-candidates":
            return _campaign_candidates(
                corpus_path=Path(str(args.corpus)),
                list_path=Path(str(args.list)),
                shared_path=Path(str(args.shared_infrastructure)),
                candidates_path=Path(str(args.candidates)),
            )
        case "policy-score":
            return _policy_score(
                corpus_path=Path(str(args.corpus)),
                list_path=Path(str(args.list)),
                shared_path=Path(str(args.shared_infrastructure)),
                weights_path=Path(str(args.weights)),
                scores_path=Path(str(args.scores)),
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
        help="write a synthetic Corpus, its Planted Campaign membership, and its "
        "Nuisance Structure",
        description=(
            "Write four files: a Corpus holding content, accounts, and links; a truth "
            "file holding Planted Campaign membership; the Nuisance Structure manifest, "
            "naming what else was planted or recorded and what each piece is for; and the "
            "known-shared infrastructure list. Each of the last three is at a path of its "
            "own, so the boundary between what the pipeline sees and what is measured "
            "stays visible. Same seed, same bytes."
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
    generate.add_argument(
        "--nuisance",
        type=Path,
        default=DEFAULT_NUISANCE_PATH,
        help=f"where to write the Nuisance Structure (default: {DEFAULT_NUISANCE_PATH})",
    )
    generate.add_argument(
        "--shared-infrastructure",
        type=Path,
        default=DEFAULT_SHARED_INFRASTRUCTURE_PATH,
        help=(
            "where to write the known-shared infrastructure list "
            f"(default: {DEFAULT_SHARED_INFRASTRUCTURE_PATH})"
        ),
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

    fetch_list = commands.add_parser(
        "fetch-suffix-list",
        help="download the Public Suffix List and cache it, with where it came from",
        description=(
            "The one command that decides what a registrable domain is, which decides "
            "what a Campaign Candidate can be (ADR-0005). The list is published rather "
            "than written by hand, so the rules are the ones in force rather than the "
            "ones somebody remembered, and the two halves of it are both parsed: a "
            "hosting platform hands out subdomains of a name it owns, and reading only "
            "the ICANN half would report every Pages site as one domain. Refuses to "
            "replace a list that is already there, because a new release moves every "
            "registrable domain in the Corpus."
        ),
    )
    fetch_list.add_argument(
        "--list",
        type=Path,
        default=DEFAULT_SUFFIX_LIST_PATH,
        help=f"where to cache the list (default: {DEFAULT_SUFFIX_LIST_PATH})",
    )
    fetch_list.add_argument(
        "--provenance",
        type=Path,
        default=DEFAULT_SUFFIX_PROVENANCE_PATH,
        help=(
            "where to record the source, licence, digests, and rule counts "
            f"(default: {DEFAULT_SUFFIX_PROVENANCE_PATH})"
        ),
    )
    fetch_list.add_argument(
        "--force",
        action="store_true",
        help="replace a list that is already there, moving every registrable domain",
    )

    domains = commands.add_parser(
        "post-domains",
        help="report the registrable domain of every link in the Corpus, per post",
        description=(
            "The first thing the pipeline computes and the input to every grouping "
            "decision after it: ADR-0005 puts two accounts in the same Campaign "
            "Candidate only if they share a registrable domain. A host is not a "
            "domain, so a campaign that mirrors its page is one registration here "
            "and not two. Every link produces a row; a link that names no "
            "registration is reported with the reason rather than dropped. Reads no "
            "network."
        ),
    )
    domains.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"the Corpus to read (default: {DEFAULT_CORPUS_PATH})",
    )
    domains.add_argument(
        "--list",
        type=Path,
        default=DEFAULT_SUFFIX_LIST_PATH,
        help=(
            "the published Public Suffix List to read "
            f"(default: {DEFAULT_SUFFIX_LIST_PATH})"
        ),
    )
    domains.add_argument(
        "--domains",
        type=Path,
        default=DEFAULT_DOMAINS_PATH,
        help=f"where to write the resolved links (default: {DEFAULT_DOMAINS_PATH})",
    )
    domains.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_DOMAINS_REPORT_PATH,
        help=f"where to write the report (default: {DEFAULT_DOMAINS_REPORT_PATH})",
    )

    categories = commands.add_parser(
        "scam-categories",
        help="project CAFC's thematic categories onto the ten Scam Categories",
        description=(
            "The projection ADR-0006 asked for, rendered. The judgement itself — the ten "
            "Scam Categories, the CAFC label each one lands, and a one-line reason for "
            "that landing — lives in `categories.py` and is read, not chosen here; what "
            "this command adds is the base rate each Scam Category lands, computed from "
            "the committed figures rather than written down. It refuses to render if the "
            "base rates hold a category the projection does not account for, because a "
            "page that looks complete and is not is worse than a refusal. Reads no "
            "network."
        ),
    )
    categories.add_argument(
        "--base-rates",
        type=Path,
        default=DEFAULT_BASE_RATES_PATH,
        help=f"the committed base rates to read (default: {DEFAULT_BASE_RATES_PATH})",
    )
    categories.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_CATEGORY_MAPPING_PATH,
        help=f"where to write the mapping (default: {DEFAULT_CATEGORY_MAPPING_PATH})",
    )
    categories.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_CATEGORY_REPORT_PATH,
        help=f"where to write the report (default: {DEFAULT_CATEGORY_REPORT_PATH})",
    )

    candidates = commands.add_parser(
        "campaign-candidates",
        help="group the Corpus into Campaign Candidates by shared registrable domain",
        description=(
            "The whole path from Corpus to output. Two accounts reach the same "
            "Campaign Candidate when a chain of shared registrable domains connects "
            "them, and on nothing else: the same words, the same hour, and the same "
            "playbook are not grounds for a grouping (ADR-0005). Known-shared "
            "infrastructure is filtered out before the grouping rather than after it, "
            "so a shortener, a paste site, or a link-in-bio page joins nothing — and "
            "the list it filters on is published data, not a list in this query, so "
            "adding a host to it changes the result without a line of code changing "
            "(ADR-0009). Every candidate is printed with the accounts, the posts, and "
            "the shared domains that put them together, so a reader can check the "
            "claim rather than take it, and the output reports how many components "
            "the filter removed. Reads no network."
        ),
    )
    candidates.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"the Corpus to read (default: {DEFAULT_CORPUS_PATH})",
    )
    candidates.add_argument(
        "--list",
        type=Path,
        default=DEFAULT_SUFFIX_LIST_PATH,
        help=(
            "the published Public Suffix List to read "
            f"(default: {DEFAULT_SUFFIX_LIST_PATH})"
        ),
    )
    candidates.add_argument(
        "--shared-infrastructure",
        type=Path,
        default=DEFAULT_SHARED_INFRASTRUCTURE_PATH,
        help=(
            "the known-shared infrastructure list to filter on "
            f"(default: {DEFAULT_SHARED_INFRASTRUCTURE_PATH})"
        ),
    )
    candidates.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help=f"where to write the candidates (default: {DEFAULT_CANDIDATES_PATH})",
    )

    scores = commands.add_parser(
        "policy-score",
        help="score every post with the published weights and print the arithmetic",
        description=(
            "The Policy Score, and the Signal-by-Signal arithmetic behind it. The score "
            "is a rules engine: an additive sum over Signals computed from a post's own "
            "text and links and from the registrations those links resolve to, with no "
            "model output anywhere in it and no account history read on the way (ADR-0007). "
            "A reader can recompute every number from what is printed beside it. The "
            "weights are published as data rather than written into the rules, so "
            "changing one is an edit to a file and not to a line of Python, and each one "
            "carries a one-line reason. Every Signal the code computes must be published "
            "in that file, so a Signal cannot be added without publishing what it is "
            "worth. Reads no network."
        ),
    )
    scores.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"the Corpus to read (default: {DEFAULT_CORPUS_PATH})",
    )
    scores.add_argument(
        "--list",
        type=Path,
        default=DEFAULT_SUFFIX_LIST_PATH,
        help=(
            "the published Public Suffix List to read "
            f"(default: {DEFAULT_SUFFIX_LIST_PATH})"
        ),
    )
    scores.add_argument(
        "--shared-infrastructure",
        type=Path,
        default=DEFAULT_SHARED_INFRASTRUCTURE_PATH,
        help=(
            "the known-shared infrastructure list to withhold registrations from "
            f"(default: {DEFAULT_SHARED_INFRASTRUCTURE_PATH})"
        ),
    )
    scores.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"the published weight set to read (default: {DEFAULT_WEIGHTS_PATH})",
    )
    scores.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES_PATH,
        help=f"where to write the scores (default: {DEFAULT_SCORES_PATH})",
    )
    return parser


def _generate_corpus(
    *,
    seed: int,
    corpus_path: Path,
    truth_path: Path,
    nuisance_path: Path,
    shared_path: Path,
) -> int:
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the membership": truth_path,
            "the Nuisance Structure": nuisance_path,
            "the known-shared infrastructure list": shared_path,
        }
    )

    items = corpus_items(seed)
    campaigns = planted_campaigns(seed)
    nuisance = nuisance_records(seed)
    write_corpus(corpus_path, items)
    write_truth(truth_path, campaigns)
    write_nuisance(nuisance_path, nuisance)
    write_shared_infrastructure(shared_path, SHARED_HOSTS)

    accounts = {item.account for item in items}
    hard_negatives = [r for r in nuisance if r.kind is NuisanceKind.HARD_NEGATIVE]
    characters: dict[str, int] = {}
    for record in hard_negatives:
        for character in record.characters:
            characters[character.value] = characters.get(character.value, 0) + 1

    print(f"seed           {seed}")
    print(f"corpus         {corpus_path} ({len(items)} posts, {len(accounts)} accounts)")
    print(
        f"truth          {truth_path} "
        f"({len(campaigns)} Planted Campaigns, {sum(len(c.posts) for c in campaigns)} posts)"
    )
    print(f"nuisance       {nuisance_path} ({len(nuisance)} records)")
    print(f"Hard Negatives {len(hard_negatives)}: {_character_summary(characters)}")
    print(
        f"infrastructure {shared_path} "
        f"({len(SHARED_HOSTS)} hosts: "
        f"{', '.join(sorted({host.kind.value for host in SHARED_HOSTS}))})"
    )
    print("the Corpus holds content, accounts, and links; grep it for `campaign` to check")
    return 0


def _require_present(inputs: dict[str, Path]) -> None:
    """Refuse a missing input by name, rather than failing somewhere inside the run.

    Both commands that read the Corpus and the published list need the same two files,
    and both would otherwise raise a bare `FileNotFoundError` from a different place.
    """
    for what, path in inputs.items():
        if not path.exists():
            raise SystemExit(
                f"{what} is not there: {path}. Run `rfi generate-corpus` and "
                "`rfi fetch-suffix-list` first."
            )


def _refuse_shared_paths(paths: dict[str, Path]) -> None:
    seen: dict[Path, str] = {}
    for what, path in paths.items():
        resolved = path.resolve()
        if resolved in seen:
            raise SystemExit(
                f"refusing to write {what} and {seen[resolved]} to one path: {path}"
            )
        seen[resolved] = what


def _character_summary(characters: dict[str, int]) -> str:
    if not characters:
        return "none"
    return ", ".join(
        f"{character} {count}" for character, count in sorted(characters.items())
    )


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


def _fetch_suffix_list(*, list_path: Path, provenance_path: Path, force: bool) -> int:
    if list_path.exists() and not force:
        raise SystemExit(
            f"refusing to replace a list that is already there: {list_path}. Pass "
            "--force to fetch a new release, and expect every registrable domain in "
            "the Corpus to move."
        )

    written = download_list(list_path)
    facts = survey_suffix_list(list_path)
    write_suffix_list_provenance(provenance_path, facts)

    print(f"list           {list_path} ({written:,} bytes, cached as published)")
    print(f"source         {SUFFIX_LIST_SOURCE_URL}")
    print(
        f"rules          {facts.plain_rules:,} whole-label, {facts.wildcard_rules:,} "
        f"wildcard, {facts.exception_rules:,} exception"
    )
    print(
        f"sections       {facts.icann_rules:,} ICANN, {facts.private_rules:,} private "
        "(both parsed: a hosting platform is a public suffix too)"
    )
    print(f"provenance     {provenance_path}")
    return 0


def _post_domains(
    *, corpus_path: Path, list_path: Path, domains_path: Path, report_path: Path
) -> int:
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the report": report_path,
            "the resolved links": domains_path,
        }
    )
    _require_present({"the Corpus": corpus_path, "the Public Suffix List": list_path})

    suffixes = PublicSuffixes.read(list_path)
    rows = post_domains(read_corpus(corpus_path), suffixes)
    facts = survey_domains(corpus_path, list_path, rows, suffixes)
    write_post_domains(domains_path, rows)
    _write_text(report_path, render_domains_report(facts, rows))

    print(f"corpus         {corpus_path} ({facts.posts} posts, {facts.accounts} accounts)")
    print(
        f"links          {facts.links} ({facts.resolved_links} resolved, "
        f"{facts.unresolved_links} unresolvable)"
    )
    print(f"domains        {facts.domains} distinct registrable domains")
    print(f"suffix list    {list_path} ({facts.suffix_list_bytes:,} bytes)")
    print(f"resolved links {domains_path}")
    print(f"report         {report_path}")
    return 0


def _scam_categories(
    *, base_rates_path: Path, mapping_path: Path, report_path: Path
) -> int:
    _refuse_shared_paths(
        {
            "the mapping": mapping_path,
            "the report": report_path,
        }
    )
    if not base_rates_path.exists():
        raise SystemExit(
            f"the base rates are not there: {base_rates_path}. Run `rfi cafc-report` "
            "over the cached extract first."
        )

    try:
        base_rates = read_base_rates(base_rates_path)
        placed = place(base_rates)
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal

    write_mapping(mapping_path, placed)
    _write_text(report_path, render_projection_report(placed))

    reports = sum(item.reports or 0 for item in placed)
    landed = sum(1 for item in placed if item.reports is not None)
    dropped = sorted(item.category.category for item in placed if item.category.dropped)
    other = total(placed, OTHER.name)

    print(
        f"base rates     {base_rates_path} ({len(base_rates)} categories, "
        f"{reports:,} reports)"
    )
    print(f"projection     {len(placed)} CAFC categories, {landed} in this release")
    print(f"top level      {TOP_LEVEL_COUNT} Scam Categories plus an Other bucket")
    print(f"dropped        {len(dropped)}: {', '.join(dropped)}")
    print(f"unplaced       {unplaced_reports(placed):,} reports, plus {other:,} in Other")
    print(f"mapping        {mapping_path}")
    print(f"report         {report_path}")
    return 0


def _campaign_candidates(
    *,
    corpus_path: Path,
    list_path: Path,
    shared_path: Path,
    candidates_path: Path,
) -> int:
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the candidates": candidates_path,
            "the known-shared infrastructure list": shared_path,
        }
    )
    _require_present(
        {
            "the Corpus": corpus_path,
            "the Public Suffix List": list_path,
            "the known-shared infrastructure list": shared_path,
        }
    )

    try:
        grouping = group_accounts(corpus_path, list_path, shared_path)
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal
    write_campaign_candidates(candidates_path, grouping.candidates)

    print(render_candidates(grouping))
    print(f"candidates     {candidates_path}")
    return 0


def _policy_score(
    *,
    corpus_path: Path,
    list_path: Path,
    shared_path: Path,
    weights_path: Path,
    scores_path: Path,
) -> int:
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the scores": scores_path,
            "the weight set": weights_path,
        }
    )
    _require_present(
        {
            "the Corpus": corpus_path,
            "the Public Suffix List": list_path,
            "the known-shared infrastructure list": shared_path,
            "the weight set": weights_path,
        }
    )

    try:
        scored = score_corpus(corpus_path, list_path, shared_path, weights_path)
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal
    write_policy_scores(scores_path, scored.scores)

    print(render_policy_score(scored))
    print(f"scores         {scores_path}")
    return 0
