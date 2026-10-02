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

Three things are built. The Corpus generator, which is the credibility boundary the
rest of the system rests on, and the Nuisance Structure it plants around the two
Planted Campaigns. The CAFC extract: real, analyst-reviewed fraud reports, cached
and committed, with the base rate of every thematic category computed from it. The
projection of those categories down to ten Scam Categories. And the pipeline itself,
as far as grouping goes: the Registrable Domain of every link, then the Campaign
Candidates those registrations produce, with known-shared infrastructure filtered out
as published data. Link Signals, Policy Score, Review Queue, and evaluation are not
built yet. Their tickets are numbered #10 to #28 in the tracker; this README is
updated as they land.

## Running it

Python is pinned to 3.13 (3.14 does not yet have prebuilt machine-learning
wheels). [uv](https://docs.astral.sh/uv/) manages the environment.

```
uv sync
uv run rfi generate-corpus
```

That writes four files; the other three in the table below are written by later
steps, which are named against each row. Every file has one reader:

| File | Holds | Read by |
| --- | --- | --- |
| `data/corpus/corpus.jsonl` | Content, accounts, and links. Nothing else. | the pipeline |
| `data/corpus/truth.jsonl` | Planted Campaign membership, written to a different path. | the evaluator |
| `data/corpus/nuisance.jsonl` | The Nuisance Structure: what else was planted or recorded, and what each piece is for. | the evaluator |
| `data/infrastructure/shared-hosts.jsonl` | Known-shared infrastructure: a link shortener, a paste site, a link-in-bio service. Each row carries where the host came from and when it was added. | `rfi campaign-candidates` |
| `data/public-suffix/public_suffix_list.dat` | The Public Suffix List, as published. | `rfi post-domains`, `rfi campaign-candidates` |
| `data/domains/post-domains.jsonl` | The Registrable Domain of every link, per post. | a reader, and the report |
| `data/campaigns/campaign-candidates.jsonl` | Every Campaign Candidate: its accounts, its posts, and the shared registrations that join them. | a reader, then the corroboration ticket #24 |

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
Campaign or to the shop that is planted there as a decoy, so the interesting false
groupings live entirely in the shared infrastructure — which is why the filter is the
subject of its own section above rather than a detail of the domain report. Ticket
#19 measures the rate; this Corpus is small enough to read the answer by hand.

## Registrable domains

`docs/post-domains.md` is the report: the registrable domain of every link in
the Corpus, per post, and the reasoning for reading a link the way it is read.
Read that rather than the summary here.

ADR-0005 places two accounts in the same Campaign Candidate only if they share a
registrable domain, so this is the input to every grouping decision in the
system. A host is not a domain: the Corpus plants
`mirror.vantage-ledger.example` beside `vantage-ledger.example` so that a
campaign that mirrors its page is one registration here rather than two, and it
plants two near-miss pairs one character apart so that anything matching on names
rather than on registration is caught joining a recruitment firm to a signals
desk.

```
uv run rfi fetch-suffix-list   # needs the network
uv run rfi post-domains        # reads the two files, writes two more
```

| File | Holds |
| --- | --- |
| `data/public-suffix/public_suffix_list.dat` | The Public Suffix List, as published. Committed as text, so a reader can grep the rules rather than trust a summary of them. |
| `data/public-suffix/provenance.jsonl` | Source, licence, digests of the bytes and of the rules, and the rule counts in each section. |
| `data/domains/post-domains.jsonl` | One line per post: every link, the host it names, the Registrable Domain, and the reason if there is none. |
| `docs/post-domains.md` | The report, generated from those rows. |

The rules come from publicsuffix.org rather than from a hand-written list, which
is the option ADR-0012 rules out with reasons; the two multi-part suffixes
everybody remembers would pass against a Corpus that uses neither and then fail
on the first suffix nobody thought of, silently, returning a public suffix's own
last label as though it were somebody's domain. Both halves of the published list
are parsed, so a hosting platform is a Public Suffix too and
`attacker.github.io` is a registration rather than a subdomain. A host three
labels deep under the reserved `.example` TLD is truncated to its last two, which
is stated in the report and affects nothing in this Corpus.

Two things are stated rather than left to be discovered. The scheme of a link is
never consulted — `https`, `http`, `ftp` and a protocol-relative `//` all name the
same registration, and a scheme allowlist would quietly lose infrastructure
an operator chose the scheme for. And every link produces a row: a link that
names no registration — a relative path, a bare `co.uk`, an IP address, a host
with a space in it, one that will not parse — is reported with which of six
reasons applied, never dropped, because a dropped link is indistinguishable from
a post that carried none. `docs/post-domains.md` names all six.

## Campaign Candidates

`rfi campaign-candidates` is the whole path from the Corpus to an output: two
accounts reach the same Campaign Candidate when a chain of shared registrable
domains connects them, and on nothing else. The console output is the report — an
index of the candidates, then for each one the accounts, the posts, and the shared
domains that put them together — and it carries its own evidence with it because a
grouping a reader cannot check is a claim rather than a result.

```
uv run rfi campaign-candidates   # reads the Corpus, the Public Suffix List, and the list
```

| File | Holds |
| --- | --- |
| `data/campaigns/campaign-candidates.jsonl` | One line per candidate: its accounts, its posts, its first-seen date, and each shared registration with the accounts and posts that reach it. |

It recovers both Planted Campaigns from shared registration alone — alpha on
`vantage-ledger.example`, which one of its accounts reaches through a mirror
hostname, and beta on `signal-harbor.example` — and it refuses the case ADR-0005 was
written about: accounts pasting one advert word for word, with nothing shared but
text, produce no candidate at all. `tests/test_campaign_candidates.py` writes that
case out by hand and asserts an empty output; the Corpus's own copy of it is joined
by nothing but the shortener those accounts all use, so the filter rather than the
absence of links is what silences it. The Corpus also plants two near-miss pairs for
the same reason, and neither reaches a candidate, so nothing about the output depends
on matching a name rather than a registration.

## Known-shared infrastructure

Five candidates used to come out of that grouping and two of them were junk: a link
shortener and a link-in-bio page are Registrable Domains like any other, so they
grouped every account that touched them — `hopcut.example` alone reached four
accounts, and with the paste site six, which was `cc-01` in the list. They are
withheld now, before the grouping rather than after it, so a withheld registration
joins nothing and cannot bridge two accounts either.

The list of registrations to withhold is
`data/infrastructure/shared-hosts.jsonl`, and it is read as data rather than written
into the grouping query (ADR-0009, ADR-0011). Each row carries the host, the kind of
service, the date it was added, and where the entry came from, and the output quotes
the path and the date the list was last updated. Every host on it is resolved through
the same Public Suffix List as any other link, so a host withholds the registration
it names; a host that resolves to nothing at all stops the run rather than quietly
filtering nothing.

The consequence is measurable rather than asserted. Every run groups the Corpus twice
— once with the shared registrations withheld and once with nothing withheld, which
is the direct-adjacency baseline ADR-0009 asks for — and the output reports the
difference:

```
  filtered      3 of 15 registrations withheld, removing 2 of 5 components
```

followed by what left the graph, so a registration the resolved links hold and no
candidate names is a decision rather than a gap:

```
withheld  3 registrations, reached by 10 accounts; no candidate is joined on one of them
  biopage.example     link in bio     4 accounts
  hopcut.example      link shortener  4 accounts
  pastevault.example  paste site      3 accounts
```

What is left is the two Planted Campaigns and one shop that shares a domain it is
right to group on — a correct grouping that must never be counted as recovery
(ADR-0004) — and every Hard Negative that reaches a shared host and nothing else is
now in no candidate at all. The four account figures above the index are a partition,
so they add up to the accounts in the Corpus: grouped, alone (a registration nobody
else reaches), silenced (nothing but known-shared infrastructure), and unreachable (no
registration at all). `silenced` is a line of its own because the six accounts the
filter silences are not the ten that touch a shared host — four of those share a
registration with nobody and are simply alone. The cost is stated rather than hidden:
a Planted Campaign that leans on a shared host is lost with the host, so a recall
figure measured this way is a lower bound. `tests/test_campaign_candidates.py` proves
the list is what decides, by adding a host to a copy of it and watching a candidate
disappear, and by running with an empty list and watching all five components come
back.

The figures at the top of the output are a partition of the Corpus's accounts, and two
accounts in the Corpus reach nothing at all: they are therefore beyond any amount of
grouping, which is the floor of the case rather than a defect in it.

Nothing in this path reads the truth file or the Nuisance Structure manifest: the
Corpus, the Public Suffix List, and the shared-infrastructure list are the whole
input, and `tests/test_campaign_candidates.py` checks that by watching which files
the run opens rather than by reading the code that decides what to open.

## The base rates

`docs/cafc-base-rates.md` is the report: the base rate of every one of the
thematic categories in the Canadian Anti-Fraud Centre's extract, beside the
licence, the attribution, and what the extract cannot be asked to do. Read that
rather than the summary here.

CAFC is the one source of real, analyst-reviewed fraud reports that is freely
downloadable under a licence permitting this use (ADR-0006). The project's word
for the file is *extract*, which is CAFC's own word for it.

```
uv run rfi fetch-cafc      # needs the network
uv run rfi cafc-report     # reads the cache and records what it holds
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
fetch needs the network, so the comparison is reproducible offline. Two commands
in this repository reach for the network — `fetch-cafc` and `fetch-suffix-list` —
and both refuse to replace a cache that is already committed; everything else
reads committed bytes and works offline.

Two limits are stated in the report rather than buried here. The extract has **no
free-text field** — no column of it can hold a sentence — so it constrains the
Scam Categories and the priors and cannot validate a text classifier. And CAFC's
annex defines 35 thematic-category headings while this window enumerates 39
values; the two are not the same set, so **41 — the figure ADR-0006 assumed — is
not a number CAFC publishes**. The union of names CAFC has published is 40, and
`docs/scam-categories.md` accounts for all of them.

## The Scam Categories

`docs/scam-categories.md` is the projection ADR-0006 asked for: CAFC's thematic
categories read down to ten Scam Categories, plus an Other bucket for content that
fits none of them. The argument — the ten, the CAFC label each one lands, the Annex E
heading CAFC defines it under, and a one-line reason — is written once in
`src/reddit_fraud_intelligence/categories.py`. The command adds the base rate each
Scam Category lands, computed from the committed figures rather than written down, and
quotes the reasons rather than restating them, so the argument has one home.

```
uv run rfi scam-categories   # reads data/cafc/base_rates.jsonl, writes both outputs
```

| File | Holds |
| --- | --- |
| `data/cafc/scam-categories.jsonl` | Every CAFC category, where it lands, what that class means, and the one-line reason. |
| `docs/scam-categories.md` | The report: what each Scam Category lands, and the argument, generated. |

Read the report rather than this summary. It is written to be disagreed with: each
merge carries a one-line rationale, so a reader who thinks a particular one is
wrong can find it and say why. Two things it is careful not to smooth over: the
size of the Other bucket, which is a standing measure of what the system fails to
represent, and CAFC's three categories that are not kinds of fraud at all, which
are dropped with a reason and counted rather than absorbed into a neighbour.

One thing in that page is weaker than the rest, and the page says so. CAFC's annex
is a PDF, so its headings are read by hand into `categories.py`; the extract is the
enumeration the mapping is guaranteed against, and a new value in a new release
stops the command until the projection accounts for it. Committing the annex, so
that the transcription is checkable against bytes, is not done here.


## Tests

```
uv run pytest
uv run mypy
```

Both run on every push to `main` and every pull request, on Ubuntu with Python
3.13, in `.github/workflows/ci.yml`. The run installs from the committed
`uv.lock` with `uv sync --locked`, so a change to `pyproject.toml` that was
never re-locked fails the run instead of installing something nobody committed.
It ends by checking that the working tree is still clean, which is the check
that would catch a test writing a generated file instead of comparing it —
every generated file here is committed, and every one is held to its bytes.

The suite needs no network: the tests that exercise the two fetching commands
patch `urllib.request.urlopen` to refuse.

## Where things are decided

- `GLOSSARY.md` — the vocabulary, enforced by `tests/test_vocabulary.py` against
  the phrases the project must never utter.
- `docs/adr/` — the decisions. The ones this code implements are 0001 (Corpus
  Provider), 0004 (evaluate by recovering Planted Campaigns), 0005 (Campaign
  Candidates require registrable infrastructure), 0007 (Signals come only from
  observable text and links), 0008 (the Corpus file carries no membership), 0009
  (direct adjacency is the baseline — this command is that baseline, with
  known-shared infrastructure filtered as published data rather than as a list in
  the query, and its corroborated tier is ticket #26), 0010 (CAFC figures are
  computed from the cache), 0011 (the Nuisance Structure has a file of its own, and
  so does the shared-infrastructure list), 0012 (registrable domains are resolved
  from the published Public Suffix List), and 0013 (the Scam Categories are CAFC's
  thematic categories read down to ten, and the 41 in ADR-0006 is corrected).
