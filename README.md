# Reddit Fraud Intelligence

An analysis system that ranks Reddit content for reviewer attention, links
suspicious content to shared infrastructure, and explains its reasoning. It
proposes Campaign Candidates — groupings of accounts that share a registrable
domain or a Contact Point — and orders content by a Policy Score a reader can
recompute by hand. It makes no claim about what the content is.

The project builds against a Corpus it generates. Nothing here reads real Reddit
content, and no account name, domain, or Contact Point in this repository
corresponds to a real person.

## Where it is

Two things are built. The Corpus generator, which is the credibility boundary the
rest of the system rests on, and the Nuisance Structure it plants around the two
Planted Campaigns. And the CAFC extract: real, analyst-reviewed fraud
reports, cached and committed, with the base rate of every thematic category
computed from it. The pipeline itself — extraction, Signals, Policy Score, Campaign
Candidate analysis, Review Queue, evaluation — is not built yet. Its tickets are
numbered #3 to #28 in the tracker; this README is updated as they land.

## Running it

Python is pinned to 3.13 (3.14 does not yet have prebuilt machine-learning
wheels). [uv](https://docs.astral.sh/uv/) manages the environment.

```
uv sync
uv run rfi generate-corpus
```

That writes four files, each with one reader:

| File | Holds | Read by |
| --- | --- | --- |
| `data/corpus/corpus.jsonl` | Content, accounts, and links. Nothing else. | the pipeline |
| `data/corpus/truth.jsonl` | Planted Campaign membership, written to a different path. | the evaluator |
| `data/corpus/nuisance.jsonl` | The Nuisance Structure: what else was planted or recorded, and what each piece is for. | the evaluator |
| `data/infrastructure/shared-hosts.jsonl` | Known-shared infrastructure: a link shortener, a paste site, a link-in-bio service. | the grouping step, once it is built |

The pipeline receives the Corpus file and nothing else; the truth file is joined
only by the evaluator, after inference has finished (ADR-0008). A reader does
not have to take the code's word for it, and regenerating the Corpus is not part
of the check — the committed file is what you grep:

```
rg -i campaign data/corpus/corpus.jsonl
```

No output. Then open the file and read the planted posts. They are written to be
read, so a reviewer can judge whether the content is realistic without running
anything. Every Synthetic Entity is marked `syn_` and every domain sits under the
reserved `.example` TLD, so nothing in the output can be mistaken for real data.

The generator is deterministic: a fixed seed produces byte-identical files, and
`tests/test_corpus_generator.py` holds the committed Corpus and membership to that
while `tests/test_nuisance_structure.py` holds the manifest and the infrastructure
list. The seed fixes the window the Corpus covers, the minute of each post, and which
spare Nuisance Structure material gets planted. Which Synthetic Entities exist is
fixed, because every test in the repository reads the Corpus by name.

## The Nuisance Structure

Recovering Planted Campaigns is this project's one substantive claim, and a
recovery number measured against a clean sweep is worth nothing. So the Corpus
carries the material that makes a grouping decision hard, and
`data/corpus/nuisance.jsonl` records it:

| Kind | What it is |
| --- | --- |
| `decoy_account_cluster` | Accounts running one campaign's playbook without being one: three accounts pasting the same advert on infrastructure that does not connect them, and three accounts of one small business sharing a domain they are right to group on. |
| `near_miss_domain_pair` | A planted domain and an unrelated business whose name is one character away from it. |
| `staggered_paraphrase` | One planted offer, reworded across a Planted Campaign's accounts. |
| `known_shared_infrastructure` | Each shared host, with the posts and accounts that touch it. |
| `single_account_domain` | A domain one account uses, which must not group. |
| `hard_negative` | Legitimate content near the boundary, recorded with its character: a genuine job post, satire, a complaint from someone who lost money, scam-adjacent discussion. |

Every record carries a note in prose saying what it is there to test, so the file
reads without running anything. Two of the kinds are computed from the Corpus rather
than hand-written, so the manifest cannot end up disagreeing with the Corpus it
describes: who links a link shortener is a fact, not something to transcribe.

The same command prints the Hard Negative count and what kind each one is, because a
false-grouping rate against four Hard Negatives and one against forty are different
numbers.

One limit is worth stating here rather than discovering later. Every registrable
domain in this Corpus that more than one account uses belongs either to a Planted
Campaign or to the shop that is planted there as a decoy, so a correctly filtered
system produces no false groupings against domains and the interesting false
groupings live in the shared infrastructure. Ticket #19 measures the rate; this
Corpus is small enough to read the answer by hand.

## The base rates

`docs/cafc-base-rates.md` is the report: the base rate of every one of the
thematic categories in the Canadian Anti-Fraud Centre's extract, beside the
licence, the attribution, and what the extract cannot be asked to do. Read that
rather than the summary here.

CAFC is the one source of real, analyst-reviewed fraud reports that is freely
downloadable under a licence permitting this use (ADR-0006). The project's word
for the file is *extract*, which is CAFC's own word for it.

```
uv run rfi fetch-cafc    # the only command that needs the network
uv run rfi cafc-report   # reads the cache and records what it holds
```

| File | Holds |
| --- | --- |
| `data/cafc/cafc-extract.csv.gz` | The cached extract, 72 MiB uncompressed, 3.7 MiB as committed. |
| `data/cafc/provenance.jsonl` | Report count, date range, category count, SHA-256, licence. |
| `data/cafc/base_rates.jsonl` | One line per thematic category: reports, and its share. |
| `docs/cafc-base-rates.md` | The report, generated from those figures. |

Every figure is computed by reading the cache, never transcribed, so a figure and
the bytes behind it cannot drift apart — and `fetch-cafc` refuses to replace a
cache that is already there, because a new quarterly release would move every base
rate and silently change what the Corpus is compared against. Nothing after the
fetch needs the network, so the comparison is reproducible offline.

Two limits are stated in the report rather than buried here. The extract has **no
free-text field** — no column of it can hold a sentence — so it constrains the
taxonomy and the priors and cannot validate a text classifier. And CAFC documents
41 thematic categories while this window holds 39; the missing two are absent from
the release, not observed at zero, and the projection in ticket #7 reconciles
against CAFC's annex.

## Tests

```
uv run pytest
uv run mypy
```

## Where things are decided

- `GLOSSARY.md` — the vocabulary, enforced by `tests/test_vocabulary.py` against
  the phrases the project must never utter.
- `docs/adr/` — the decisions. The ones this code implements are 0001 (Corpus
  Provider), 0004 (evaluate by recovering Planted Campaigns), 0005 (Campaign
  Candidates require registrable infrastructure), 0007 (Signals come only from
  observable text and links), 0008 (the Corpus file carries no membership), 0010
  (CAFC figures are computed from the cache), and 0011 (the Nuisance Structure has a
  file of its own, and so does the shared-infrastructure list).
