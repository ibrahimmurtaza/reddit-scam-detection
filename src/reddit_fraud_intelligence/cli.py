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
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_WINDOW_HOURS,
    group as group_accounts,
    render_table as render_candidates,
    write_campaign_candidates,
)
from reddit_fraud_intelligence.categories import OTHER, SCAM_CATEGORIES, TOP_LEVEL_COUNT
from reddit_fraud_intelligence.composition import (
    compose,
    render_report as render_composition_report,
    render_table as render_composition_table,
    write_composition,
)
from reddit_fraud_intelligence.contacts import (
    contact_points,
    render_report as render_contacts_report,
    render_table as render_contacts_table,
    write_post_contacts,
    write_published,
)
from reddit_fraud_intelligence.corpus import read_corpus, write_corpus
from reddit_fraud_intelligence.domains import (
    post_domains,
    render_report as render_domains_report,
    survey as survey_domains,
    write_post_domains,
)
from reddit_fraud_intelligence.embeddings import DEFAULT_TABLE
from reddit_fraud_intelligence.evaluation import (
    DEFAULT_DEPTHS,
    RECOVERY_PATH,
    recover,
    render_report as render_recovery_report,
    render_table as render_recovery_table,
    write_recovery,
)
from reddit_fraud_intelligence.embeddings import (
    embed_corpus,
    read_content_records,
    render_table as render_embeddings_table,
    write_content_records,
)
from reddit_fraud_intelligence.embeddings import DEFAULT_TABLE
from reddit_fraud_intelligence.generator import (
    corpus_items,
    nuisance_records,
    planted_campaigns,
    published_posts,
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
from reddit_fraud_intelligence.review_queue import build_queue, render_table as render_queue
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
DEFAULT_LABELLED_PATH = Path("data/corpus/labelled-contacts.jsonl")
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
DEFAULT_CONTACTS_PATH = Path("data/contacts/post-contacts.jsonl")
DEFAULT_CONTACTS_REPORT_PATH = Path("docs/contact-points.md")
DEFAULT_CATEGORY_MAPPING_PATH = Path("data/cafc/scam-categories.jsonl")
DEFAULT_CATEGORY_REPORT_PATH = Path("docs/scam-categories.md")
DEFAULT_CANDIDATES_PATH = Path("data/campaigns/campaign-candidates.jsonl")
DEFAULT_RECOVERY_PATH = Path(RECOVERY_PATH)
DEFAULT_RECOVERY_REPORT_PATH = Path("docs/campaign-recovery.md")
DEFAULT_COMPOSITION_PATH = Path("data/corpus/composition.jsonl")
DEFAULT_COMPOSITION_REPORT_PATH = Path("docs/corpus-composition.md")
DEFAULT_WEIGHTS_PATH = Path("data/signals/weights.jsonl")
DEFAULT_SCORES_PATH = Path("data/signals/policy-scores.jsonl")
DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings/content-embeddings.jsonl")
# Aliased rather than repeated: `content-embeddings` writes this table and
# `campaign-candidates` reads it, and two literals for one table name is how the two
# commands stop reading each other's vectors.
DEFAULT_EMBEDDING_TABLE = DEFAULT_TABLE
DEFAULT_REVIEW_DEPTH = 50
DEFAULT_PRECISION_DEPTHS = ",".join(str(depth) for depth in DEFAULT_DEPTHS)


def _parse_depths(value: str) -> tuple[int, ...]:
    """The review depths, comma-separated, each one named in the report."""
    try:
        depths = tuple(int(part.strip()) for part in str(value).split(","))
    except ValueError:
        raise SystemExit(f"--depths is a comma-separated list of whole numbers, is {value!r}")
    if not depths or any(depth < 1 for depth in depths):
        raise SystemExit(f"--depths has to hold whole numbers of one or more, is {value!r}")
    return depths


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    match args.command:
        case "generate-corpus":
            return _generate_corpus(
                seed=int(args.seed),
                corpus_path=Path(str(args.corpus)),
                truth_path=Path(str(args.truth)),
                labelled_path=_labelled_beside(args.labelled, args.corpus),
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
        case "contact-points":
            return _contact_points(
                corpus_path=Path(str(args.corpus)),
                labelled_path=Path(str(args.labelled)),
                contacts_path=Path(str(args.contacts)),
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
                window_hours=int(args.window_hours),
                threshold=float(args.similarity_threshold),
                table=str(args.table),
            )
        case "corpus-composition":
            return _corpus_composition(
                corpus_path=Path(str(args.corpus)),
                base_rates_path=Path(str(args.base_rates)),
                composition_path=Path(str(args.composition)),
                report_path=Path(str(args.report)),
            )
        case "campaign-recovery":
            return _campaign_recovery(
                corpus_path=Path(str(args.corpus)),
                candidates_path=Path(str(args.candidates)),
                truth_path=Path(str(args.truth)),
                nuisance_path=Path(str(args.nuisance)),
                scores_path=Path(str(args.scores)),
                depths=_parse_depths(args.depths),
                recovery_path=Path(str(args.recovery)),
                report_path=Path(str(args.report)),
            )
        case "policy-score":
            return _policy_score(
                corpus_path=Path(str(args.corpus)),
                list_path=Path(str(args.list)),
                shared_path=Path(str(args.shared_infrastructure)),
                weights_path=Path(str(args.weights)),
                scores_path=Path(str(args.scores)),
            )
        case "review-queue":
            return _review_queue(
                scores_path=Path(str(args.scores)),
                candidates_path=Path(str(args.candidates)),
                depth=int(args.depth),
            )
        case "content-embeddings":
            return _content_embeddings(
                corpus_path=Path(str(args.corpus)),
                embeddings_path=Path(str(args.embeddings)),
                table=str(args.table),
                replace=bool(args.replace),
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
            "Write five files: a Corpus holding content, accounts, and links; a truth "
            "file holding Planted Campaign membership; the Labelled Set, one row per post "
            "naming every Contact Point that post publishes; the Nuisance Structure "
            "manifest, naming what else was planted or recorded and what each piece is "
            "for; and the known-shared infrastructure list. Each of the last four is at a "
            "path of its own, so the boundary between what the pipeline sees and what is "
            "measured stays visible. Same seed, same bytes."
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
        "--labelled",
        type=Path,
        default=None,
        help=(
            "where to write the Labelled Set: what every post publishes as a Contact "
            "Point (default: labelled-contacts.jsonl beside the Corpus)"
        ),
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

    contacts = commands.add_parser(
        "contact-points",
        help="report the Contact Points every post names, per post, and which are shared",
        description=(
            "The off-platform half of ADR-0005: two accounts may reach one Campaign "
            "Candidate on a Contact Point as well as on a Registrable Domain, and this "
            "is the step that finds them. Two kinds are read — an email address and a "
            "Telegram handle — from a post's own title, its own body, and its links, "
            "with the obfuscation tolerance a post actually uses: an identifier written "
            "out character by character, and a digit standing in for a letter, are read "
            "as the identifier underneath them. The reading is then measured against "
            "the Labelled Set the generator wrote beside the Corpus, and the recall and "
            "the false-positive rate are printed beside the figures they bound, with "
            "every miss named. Nothing is dropped: a candidate that names no Contact "
            "Point is reported with which of nine faults applied, because a false "
            "positive here is not a wrong severity figure as a Signal can be, it is a "
            "shared identifier between two accounts that share nothing. Sharing is "
            "counted over accounts rather than over posts, every spelling of every value "
            "is printed beside it, and nothing is grouped on any of it here — the shared "
            "list is what `rfi campaign-candidates` groups on, and the recall printed "
            "beside it is what keeps that edge from being an unmeasured input into the "
            "recovery figure. Reads no network."
        ),
    )
    contacts.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"the Corpus to read (default: {DEFAULT_CORPUS_PATH})",
    )
    contacts.add_argument(
        "--labelled",
        type=Path,
        default=DEFAULT_LABELLED_PATH,
        help=(
            "the Labelled Set to measure the reading against: one row per post naming "
            f"what it publishes (default: {DEFAULT_LABELLED_PATH})"
        ),
    )
    contacts.add_argument(
        "--contacts",
        type=Path,
        default=DEFAULT_CONTACTS_PATH,
        help=f"where to write the Contact Points (default: {DEFAULT_CONTACTS_PATH})",
    )
    contacts.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_CONTACTS_REPORT_PATH,
        help=f"where to write the report (default: {DEFAULT_CONTACTS_REPORT_PATH})",
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
            "the filter removed. Temporal proximity and content similarity are read as "
            "corroboration: each orders and deprioritises, neither creates a candidate, "
            "and each states the threshold it judged by (ADR-0025, ADR-0026). The "
            "similarity is computed from the vectors `rfi content-embeddings` stored, "
            "read out of the vector table over the environment's connection, which is "
            "why this command needs the database as well as the three files. Reads no "
            "network."
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
    candidates.add_argument(
        "--window-hours",
        type=int,
        default=DEFAULT_WINDOW_HOURS,
        help=(
            "the window, in hours, within which the accounts sharing one registration or "
            "Contact Point corroborate it; timing corroborates and never groups, so a "
            "window moves a candidate down the list and renumbers it, and nothing else "
            f"(default: {DEFAULT_WINDOW_HOURS})"
        ),
    )
    candidates.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=(
            "the cosine distance, at most, for two posts in one candidate to count as "
            "near-identical; similarity corroborates and never groups, so a threshold "
            "moves a candidate down the list and renumbers it, and nothing else "
            f"(default: {DEFAULT_SIMILARITY_THRESHOLD})"
        ),
    )
    candidates.add_argument(
        "--table",
        type=str,
        default=DEFAULT_EMBEDDING_TABLE,
        help=(
            "the table to read the stored vectors from, which is where another model's "
            f"vectors would live rather than this one's (default: {DEFAULT_EMBEDDING_TABLE})"
        ),
    )

    recovery = commands.add_parser(
        "campaign-recovery",
        help="report how many Planted Campaigns the grouping recovered, as X of N",
        description=(
            "The project's one substantive claim, and the only command permitted to "
            "read the Planted Campaign membership. It is a separate step over a "
            "separate file: it joins the membership the generator wrote against the "
            "candidates `rfi campaign-candidates` published, does no grouping of its "
            "own, and opens neither the Public Suffix List nor the "
            "shared-infrastructure list, so measurement cannot reach inference "
            "(ADR-0018). A candidate counts as a recovery only when it holds a "
            "campaign's whole membership, and a candidate holding part of one is "
            "reported beside the figure rather than counted into it — the accounts "
            "held, missing, and unexpected are all named, so X of N can be taken "
            "apart. Candidates that reach no campaign are printed with the "
            "registrations that join them: a shop's accounts sharing a domain is a "
            "grouping the system is right to produce and is not a miss against N. "
            "The Nuisance Structure the figure was measured against is read and "
            "printed, because a recovery rate with no nuisance baseline beside it is "
            "uninterpretable; the rate at which the grouping is wrong is measured "
            "against the same manifest and reported together with the recovery, and "
            "the Review Queue's precision is measured at several depths beside "
            "them. No person reviewed any of it and no "
            "figure over the whole Corpus is published (ADR-0004). Reads no network."
        ),
    )
    recovery.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"the Corpus the measurement is about (default: {DEFAULT_CORPUS_PATH})",
    )
    recovery.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help=(
            "the Campaign Candidates the grouping published "
            f"(default: {DEFAULT_CANDIDATES_PATH})"
        ),
    )
    recovery.add_argument(
        "--truth",
        type=Path,
        default=DEFAULT_TRUTH_PATH,
        help=f"the Planted Campaign membership to read (default: {DEFAULT_TRUTH_PATH})",
    )
    recovery.add_argument(
        "--nuisance",
        type=Path,
        default=DEFAULT_NUISANCE_PATH,
        help=(
            "the Nuisance Structure the figure was measured against "
            f"(default: {DEFAULT_NUISANCE_PATH})"
        ),
    )
    recovery.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES_PATH,
        help=(
            "the Policy Scores the queue precision is measured over "
            f"(default: {DEFAULT_SCORES_PATH})"
        ),
    )
    recovery.add_argument(
        "--depths",
        type=str,
        default=DEFAULT_PRECISION_DEPTHS,
        help=(
            "comma-separated review depths the queue precision is measured at "
            f"(default: {DEFAULT_PRECISION_DEPTHS})"
        ),
    )
    recovery.add_argument(
        "--recovery",
        type=Path,
        default=DEFAULT_RECOVERY_PATH,
        help=f"where to write the join (default: {DEFAULT_RECOVERY_PATH})",
    )
    recovery.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_RECOVERY_REPORT_PATH,
        help=f"where to write the report (default: {DEFAULT_RECOVERY_REPORT_PATH})",
    )

    scores = commands.add_parser(
        "policy-score",
        help="score every post with the published weights and print the arithmetic",
        description=(
            "The Policy Score, and the Signal-by-Signal arithmetic behind it. The score "
            "is a rules engine: an additive sum over Signals computed from a post's own "
            "text and links, from the registrations those links resolve to, and from "
            "what the posts reaching those registrations say, with no model output "
            "anywhere in it and no account history read on the way (ADR-0007). A reader "
            "can recompute every number from what is printed beside it. The weights are "
            "published as data rather than written into the rules, so changing one is an "
            "edit to a file and not to a line of Python, and each one carries a one-line "
            "reason. Every Signal the code computes must be published in that file, so a "
            "Signal cannot be added without publishing what it is worth. Reads no "
            "network."
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

    queue = commands.add_parser(
        "review-queue",
        help="order the scored posts by Triage Priority and print the top of them",
        description=(
            "The Review Queue: every post in the published scores, ordered by Triage "
            "Priority, truncated to a depth stated here and in the output header. The order "
            "is the Policy Score and nothing else, so the number a reviewer reads at the "
            "top of the queue is the number the scoring command published for that post "
            "(ADR-0003); two posts that tie on it are ordered by the points they earned of "
            "the published total, which is that same arithmetic one step finer. Every entry "
            "carries the Signal-by-Signal breakdown behind its score and names the Campaign "
            "Candidates the post sits in, so an entry can be worked from this output alone. "
            "It reads the scores and the candidates and runs neither step again, which is "
            "what keeps it from reaching the Corpus or the membership (ADR-0018). No figure "
            "is computed over the queue: how good the ordering is needs a reviewer. Reads "
            "no network."
        ),
    )
    queue.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES_PATH,
        help=f"the published Policy Scores to read (default: {DEFAULT_SCORES_PATH})",
    )
    queue.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help=(
            "the published Campaign Candidates to read "
            f"(default: {DEFAULT_CANDIDATES_PATH})"
        ),
    )
    queue.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_REVIEW_DEPTH,
        help=f"how many entries of the queue to print (default: {DEFAULT_REVIEW_DEPTH})",
    )

    composition = commands.add_parser(
        "corpus-composition",
        help="report the Corpus's Scam Category distribution beside CAFC's base rates",
        description=(
            "The comparison ADR-0006 asked for, and the one that turns a generated Corpus "
            "into evidence rather than a demonstration. A post is placed by reading its own "
            "title and body against a published phrase list per Scam Category, the same "
            "way a Content Signal fires, and the share of the Corpus in each class is set "
            "beside the base rate CAFC publishes for that class. The two columns are shares "
            "of two different wholes, so the difference is the figure and the output prints "
            "it for every row. The Other bucket is a row rather than a remainder, because a "
            "bucket read as rounding says nothing about what the generator did not write. "
            "CAFC's extract has no free-text field, so it constrains the Scam Categories "
            "and their priors and cannot validate the reading; the output says so rather "
            "than leaving the reader to work out why a base rate is being used this way. "
            "Reads no network."
        ),
    )
    composition.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"the Corpus to read (default: {DEFAULT_CORPUS_PATH})",
    )
    composition.add_argument(
        "--base-rates",
        type=Path,
        default=DEFAULT_BASE_RATES_PATH,
        help=f"the committed base rates to read (default: {DEFAULT_BASE_RATES_PATH})",
    )
    composition.add_argument(
        "--composition",
        type=Path,
        default=DEFAULT_COMPOSITION_PATH,
        help=f"where to write the placements (default: {DEFAULT_COMPOSITION_PATH})",
    )
    composition.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_COMPOSITION_REPORT_PATH,
        help=f"where to write the report (default: {DEFAULT_COMPOSITION_REPORT_PATH})",
    )

    embeddings = commands.add_parser(
        "content-embeddings",
        help="embed every Content Item's text and links in pgvector, and index it",
        description=(
            "The substrate the content-similarity corroboration is built on: one sentence "
            "embedding per Content Item, in a vector column under a cosine index, so a "
            "similarity query can be asked of it. What is embedded is the post's own title, "
            "its own body and its links, and nothing about its account - the Corpus file is "
            "the boundary this project measures itself against, and a vector built out of "
            "who posted would put Planted Campaign membership back through a side door "
            "(ADR-0008). The model is this project's own and it is published: its name, its "
            "dimensionality and a digest of the recipe travel on every row and in the column, "
            "because a vector's comparability is a property of that vector and a table "
            "holding two models would hold nothing. A vector is computed once and reused - the "
            "SHA-256 of the text it came from is stored beside it, and a rerun over an "
            "unchanged Corpus embeds nothing and prints that it computed nothing. HNSW is "
            "chosen over IVFFlat because it answers without training data and without a "
            "parameter that has to be revisited as the Corpus grows, and the output reports "
            "what the planner actually chose rather than claiming the index is at work over a "
            "table this small. Nothing here groups, corroborates or decides anything "
            "(ADR-0024). Reads no network."
        ),
    )
    embeddings.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"the Corpus to read (default: {DEFAULT_CORPUS_PATH})",
    )
    embeddings.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help=(
            "where to write the record of what is stored: which post, which model, how wide, "
            f"and the digest of its text (default: {DEFAULT_EMBEDDINGS_PATH})"
        ),
    )
    embeddings.add_argument(
        "--table",
        type=str,
        default=DEFAULT_EMBEDDING_TABLE,
        help=(
            "the table to store the vectors in, which is where another model's vectors would "
            f"go rather than this one's (default: {DEFAULT_EMBEDDING_TABLE})"
        ),
    )
    embeddings.add_argument(
        "--replace",
        action="store_true",
        help=(
            "treat the named table as disposable: rows for Content Items this Corpus does "
            "not hold are removed even if the two share no Content Item, which is what a "
            "rerun against a rebuilt Corpus wants and what a mistyped --corpus must not do"
        ),
    )
    return parser


def _labelled_beside(labelled: Path | str | None, corpus: Path | str) -> Path:
    """Where the Labelled Set goes: where it was asked for, or beside the Corpus.

    Beside the Corpus rather than at a path of its own, because a run that writes a
    Corpus somewhere else on purpose — every test, and any reader who does not want the
    committed one — would otherwise reach over and overwrite the committed Labelled Set
    beside it. The set is a projection of the Corpus (ADR-0008), so it belongs in the
    Corpus's own directory rather than at a fixed one of the project's.
    """
    if labelled is not None:
        return Path(str(labelled))
    return Path(str(corpus)).parent / DEFAULT_LABELLED_PATH.name


def _generate_corpus(
    *,
    seed: int,
    corpus_path: Path,
    truth_path: Path,
    labelled_path: Path,
    nuisance_path: Path,
    shared_path: Path,
) -> int:
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the membership": truth_path,
            "the Labelled Set": labelled_path,
            "the Nuisance Structure": nuisance_path,
            "the known-shared infrastructure list": shared_path,
        }
    )

    items = corpus_items(seed)
    campaigns = planted_campaigns(seed)
    nuisance = nuisance_records(seed)
    published = published_posts(seed)
    write_corpus(corpus_path, items)
    write_truth(truth_path, campaigns)
    write_published(labelled_path, published)
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
    print(
        f"labelled       {labelled_path} "
        f"({sum(len(post.published) for post in published)} Contact Points published by "
        f"{len(published)} posts, {sum(1 for post in published if not post.published)} of "
        "them publishing none)"
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


def _require_present(inputs: dict[str, Path], hint: str | None = None) -> None:
    """Refuse a missing input by name, rather than failing somewhere inside the run.

    The commands that read the Corpus and the published list all need the same two
    files, and they would otherwise raise a bare `FileNotFoundError` from a different
    place. The hint names the commands that produce them, and a command that reads no
    published list passes its own rather than sending a reader to fetch a rulebook it
    will never open.
    """
    remedy = hint or "Run `rfi generate-corpus` and `rfi fetch-suffix-list` first."
    for what, path in inputs.items():
        if not path.exists():
            raise SystemExit(f"{what} is not there: {path}. {remedy}")


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


def _contact_points(
    *, corpus_path: Path, labelled_path: Path, contacts_path: Path, report_path: Path
) -> int:
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the report": report_path,
            "the Labelled Set": labelled_path,
            "the Contact Points": contacts_path,
        }
    )
    _require_present(
        {"the Corpus": corpus_path, "the Labelled Set": labelled_path},
        "Run `rfi generate-corpus` first; this command reads no published list.",
    )

    try:
        found = contact_points(corpus_path, labelled_path)
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal
    write_post_contacts(contacts_path, found.rows)
    _write_text(report_path, render_contacts_report(found))

    print(render_contacts_table(found))
    print(f"contact points {contacts_path}")
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
    window_hours: int,
    threshold: float,
    table: str,
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
        grouping = group_accounts(
            corpus_path,
            list_path,
            shared_path,
            window_hours,
            threshold,
            table,
        )
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal
    write_campaign_candidates(candidates_path, grouping.candidates)

    print(render_candidates(grouping))
    print(f"candidates     {candidates_path}")
    return 0


def _corpus_composition(
    *,
    corpus_path: Path,
    base_rates_path: Path,
    composition_path: Path,
    report_path: Path,
) -> int:
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the placements": composition_path,
            "the report": report_path,
        }
    )
    _require_present(
        {"the Corpus": corpus_path},
        "Run `rfi generate-corpus` first; this command reads no published list.",
    )
    if not base_rates_path.exists():
        raise SystemExit(
            f"the base rates are not there: {base_rates_path}. Run `rfi cafc-report` "
            "over the cached extract first."
        )

    try:
        composition = compose(corpus_path, base_rates_path)
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal
    write_composition(composition_path, composition.posts)
    _write_text(report_path, render_composition_report(composition))

    print(render_composition_table(composition))
    print(f"composition    {composition_path}")
    print(f"report         {report_path}")
    return 0


def _campaign_recovery(
    *,
    corpus_path: Path,
    candidates_path: Path,
    truth_path: Path,
    nuisance_path: Path,
    scores_path: Path,
    depths: Sequence[int],
    recovery_path: Path,
    report_path: Path,
) -> int:
    """Join the membership against what the grouping published, and report it.

    The seed is passed rather than read from the Corpus because it is not in the Corpus
    file — there is no field for it, and adding one would put the generator's own
    bookkeeping into the file the pipeline reads (ADR-0008). It decides nothing here: it
    is carried so the report names the seed a reader can regenerate the Corpus with,
    rather than leaving the claim that they can checkable on trust.
    """
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the candidates": candidates_path,
            "the membership": truth_path,
            "the Nuisance Structure": nuisance_path,
            "the Policy Scores": scores_path,
            "the join": recovery_path,
            "the report": report_path,
        }
    )
    _require_present(
        {
            "the Corpus": corpus_path,
            "the Campaign Candidates": candidates_path,
            "the Planted Campaign membership": truth_path,
            "the Nuisance Structure": nuisance_path,
            "the Policy Scores": scores_path,
        },
        "Run `rfi generate-corpus`, then `rfi campaign-candidates`, then "
        "`rfi policy-score` first; this command measures what the grouping published.",
    )

    try:
        measured = recover(
            corpus_path,
            candidates_path,
            truth_path,
            nuisance_path,
            DEFAULT_SEED,
            scores_path,
            depths,
        )
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal
    write_recovery(recovery_path, measured.recoveries)
    _write_text(report_path, render_recovery_report(measured))

    print(render_recovery_table(measured))
    print(f"join           {recovery_path}")
    print(f"report         {report_path}")
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
            "the known-shared infrastructure list": shared_path,
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


def _review_queue(*, scores_path: Path, candidates_path: Path, depth: int) -> int:
    """Print the queue. Nothing is written: this is the first command that produces no
    file, because the queue is a projection of two files another command published and a
    copy of it in a third place would be a third thing to keep in step with the other two.
    """
    _require_present(
        {
            "the Policy Scores": scores_path,
            "the Campaign Candidates": candidates_path,
        },
        "Run `rfi policy-score` and `rfi campaign-candidates` first; this command orders "
        "what those two published.",
    )

    try:
        queue = build_queue(scores_path, candidates_path, depth)
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal

    print(render_queue(queue))
    return 0


def _content_embeddings(
    *, corpus_path: Path, embeddings_path: Path, table: str, replace: bool = False
) -> int:
    """Embed every Content Item, store the vectors, and publish the record of what is stored.

    The record is written and then read back before the run reports success, because a file
    this project publishes and cannot read is a file nothing will ever hold to its bytes: the
    vocabulary check in `read_content_records` is the only thing standing between a future
    field — a Planted Campaign's identifier above all — and a file nobody examines (ADR-0008).

    The vectors are not read back and are not published. They live in the table, which is
    where the ticket asked for them, and what is published is which post each stored vector
    belongs to and under which model.
    """
    _refuse_shared_paths(
        {
            "the Corpus": corpus_path,
            "the record": embeddings_path,
        }
    )
    _require_present(
        {"the Corpus": corpus_path},
        "Run `rfi generate-corpus` first; this command reads no published list.",
    )

    try:
        embedded = embed_corpus(corpus_path, table, replace=replace)
        write_content_records(embeddings_path, embedded.records)
        published = read_content_records(embeddings_path)
    except ValueError as refusal:
        raise SystemExit(refusal) from refusal
    if published != embedded.records:
        raise SystemExit(
            f"{embeddings_path.as_posix()} does not read back as the {len(published)} records "
            "this run stored, so it is not published"
        )

    print(render_embeddings_table(embedded))
    print(f"record         {embeddings_path}")
    print(f"vectors        in the table {table}")
    return 0
