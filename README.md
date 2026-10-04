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

Six things are built. The Corpus generator, which is the credibility boundary the
rest of the system rests on, and the Nuisance Structure it plants around the two
Planted Campaigns. The CAFC extract: real, analyst-reviewed fraud reports, cached
and committed, with the base rate of every thematic category computed from it. The
projection of those categories down to ten Scam Categories, and the comparison of
the Corpus's own distribution against those base rates. The pipeline itself: the
Registrable Domain of every link, then the Contact Points every post names, then
the Campaign Candidates those registrations produce, with known-shared infrastructure
filtered out as published data, then the Policy Score those same links and posts add
up to, with the arithmetic printed beside it. And the measurement: how many of the two
Planted Campaigns that grouping recovered, as X of N, joined by a command that runs
after it rather than inside it. The Review Queue those scores are ordered into, at a
stated depth. The false-grouping rate beside that recovery figure, the Confidence, the
figure over the Review Queue, and the corroborated grouping tier are not built yet. Their
tickets are numbered #16 to #28 in the tracker; this README is updated as they land.

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
| `data/corpus/labelled-contacts.jsonl` | The Labelled Set: one row per post, naming every Contact Point that post publishes and whether it published it plainly, written out, or only as a picture of one. | `rfi contact-points`, to measure its own reading |
| `data/corpus/nuisance.jsonl` | The Nuisance Structure: what else was planted or recorded, and what each piece is for. | the evaluator |
| `data/corpus/composition.jsonl` | Every post's Scam Category, with the sentences of its own text that placed it. | a reader, then the category-conflict ticket #18 |
| `data/infrastructure/shared-hosts.jsonl` | Known-shared infrastructure: a link shortener, a paste site, a link-in-bio service. Each row carries where the host came from and when it was added. | `rfi campaign-candidates` |
| `data/public-suffix/public_suffix_list.dat` | The Public Suffix List, as published. | `rfi post-domains`, `rfi campaign-candidates` |
| `data/domains/post-domains.jsonl` | The Registrable Domain of every link, per post. | a reader, and the report |
| `data/contacts/post-contacts.jsonl` | Every Contact Point a post names, per post, with the spelling and the field it was found in. | a reader, then the grouping-edge ticket #20 |
| `data/campaigns/campaign-candidates.jsonl` | Every Campaign Candidate: its accounts, its posts, and the shared registrations that join them. | `rfi campaign-recovery`, `rfi review-queue`, then the corroboration ticket #24 |
| `data/evaluation/recovery.jsonl` | One line per Planted Campaign: its membership, its outcome, and every candidate that reached it, with the accounts held, missing, and unexpected. | a reader, then the false-grouping ticket #19 |
| `data/signals/weights.jsonl` | The published weight of every Signal, with the one-line reason it is that number. | `rfi policy-score` |
| `data/signals/policy-scores.jsonl` | Every post's Policy Score, with the Signal-by-Signal arithmetic behind it. | `rfi review-queue` |

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
| `obfuscated_contact` | One desk publishing one handle 4 ways: written out with a full stop between its characters, written out with spaces, with a digit standing in for a letter, and only as a picture of one. 3 of the 4 are found and the picture is not. Beside them an address carrying a digit in its local part is published plainly and read exactly as written, and a second address written as words is not read at all. A 7th post publishes nothing, and the 1 Contact Point the reading invents in it comes from there — which is what makes the recall beside them a measurement rather than a promise. |
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
subject of its own section above rather than a detail of the domain report. The
recovery report below prints that shop beside the figure, because a recovery rate
readable without knowing what the other candidates are is not readable; ticket #19
measures the rate. This Corpus is small enough to read the answer by hand.

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

## Contact Points

`docs/contact-points.md` is the report: the Contact Point every post names, per post,
and the reasoning for reading one the way it is read. Read that rather than the
summary here.

ADR-0005 lets two accounts reach the same Campaign Candidate on a Contact Point as
well as on a Registrable Domain, and this is the step that finds them. Two kinds are
read — an email address and a Telegram handle — from a post's own title, its own body,
and its links.

```
uv run rfi contact-points   # reads the Corpus and the Labelled Set, writes two files
```

| File | Holds |
| --- | --- |
| `data/contacts/post-contacts.jsonl` | One line per post: its Contact Points, every occurrence with the spelling and the field it was found in, and every candidate that named none. |
| `data/corpus/labelled-contacts.jsonl` | The Labelled Set the reading is measured against, named here because it is an input rather than an output of this command. |
| `docs/contact-points.md` | The report, generated from those rows. |

**An obfuscated username is read as the identifier underneath it, an address's host is read
exactly as written, and the reading is measured (ADR-0019).** A handle written out character by
character — `@s.y.n._.v.a.n.t.a.g.e.l.e.d.g.e.r`, `@s y n _ v a n t a g e l e d g e r` — and one
with a digit standing in for a letter (`@syn_v4ntag3ledger`) are the same Contact Point as the
plain spelling, and every spelling is printed beside the value it was read as. A **username** is
folded for substitution and, where it was written out, for its separators. An **address** is
folded for spacing and for nothing else: no full stop is dropped from either half of one,
because a full stop is a label boundary in both and `vantage.ledger.example` is not
`vantage-ledger.example`, and no digit is folded in either half either — `desk@gr4vy.io` and
`desk@gravy.io` are two domains that both exist, and nothing in the post says which one it meant.
The Corpus carries both cases so the difference is measured rather than argued.

Two things are stated rather than left to be discovered, because both are places where
this project could invent a Contact Point rather than miss one. **Dropping a separator
cannot merge two usernames that both exist** — a full stop or a hyphen cannot occur in a
Telegram username at all — **while folding a digit into its letter can**, so the
substitution table is published and every figure that fold can move is measured and printed.
And **a post that spaces out the letters of a word beside an `@` produces the same shape as
one writing out a handle**: `email me @ t o n i g h t at 8` is read as the handle
`tonight`, which is the one false positive the tolerance has that reading literally did
not. Nothing in the text tells the two apart; a miss is a floor the output already
declares, an invented value is a shared identifier ADR-0005 would group two accounts on,
so the fold is kept, the invented value is named in the report, and the false-positive
rate is printed beside the recall.

```
  labelled   data/corpus/labelled-contacts.jsonl: 34 posts, 13 Contact Points published between them
  recall     11 of 13 Contact Points the Corpus publishes were found: 8 of 8 published plainly, 3 of 4 written out
  picture    1 of the 13 is published only as a picture of one and cannot be read by anything in this project
  invented   1 of the 22 posts that publish none had a Contact Point invented in them
  folded     every Contact Point found is the one its own post publishes; no fold joined two of the values the Corpus publishes
```

That figure is of *this* reading over *this* Corpus, and the report says so. It is measured
against `data/corpus/labelled-contacts.jsonl`, which `rfi generate-corpus` writes beside the
Corpus from the same plan the posts come from: one row per post, naming every Contact Point
that post publishes and whether it published it plainly, written out, or only as a picture of
one. A Corpus from a provider has no Labelled Set beside it, and measuring this reading over
other content means labelling that content. The labels are read after the reading has
finished and reach nothing that decides what was read — `measure()` is handed the finished
reading rather than the Corpus, and a test reads the same Corpus twice, once with labels that
agree and once with labels that do not, and asserts the rows are identical.

The Corpus is planted so that the measurement cannot flatter itself. One desk publishes the
same handle three ways from three accounts — written out with separators, written out with
spaces, and one letter standing in as the digit it looks like — which is the case the
reading exists for: three accounts publishing one reach three different ways is one shared
Contact Point and three that nobody can join. Beside them are a handle published only as a
screenshot, which nothing here can read, and an address written as words, with no `@` in the
post for any reader to find: both misses are named row by row in the report, so a reader can
see that the shortfall is the material rather than a defect. The other half of the figure is
planted too, and it is the uncomfortable half: a post that publishes nothing at all and
spells a word out beside an `@`, which the reading invents the handle `tonight` from. Clean
posts whose shapes the literal reader also refused would have reported a false-positive rate
of nothing, so the shaped case is in the material and the figure has to name it. The two
near-misses (`@ 4200`, `@ 9`) sit beside it.

Sharing is counted over accounts, not over posts. Two accounts naming one Contact
Point is the claim this command exists to make. One account naming it in four posts is
not, because it has not acquired a second reach. The Corpus is planted so both planted
campaigns publish a channel from more than one of their own accounts, and so that not
every post of a campaign carries one — an extractor that is only ever right about the
posts advertising a Contact Point has been measured on nothing else.

The most important number in the output is a false grouping waiting to happen. Four
accounts name `syn_vantageledger`, and one of them is a declared Hard Negative: a person
who lost money and quoted the channel they were given. That is correct reporting and the
worst possible grouping input, which is why nothing groups on it yet. ADR-0005 permits a
Contact Point as a grouping edge and this command builds none — the shared blocks are
the evidence ticket #20 needs, and an unmeasured grouping edge would put an unmeasured
input into the recovery figure ADR-0004 measures against.

Every figure the command prints is a lower bound, and the measured recall above is what
bounds it: the Contact Points this Corpus publishes that were not read are named in the
report, so the shortfall can be read rather than inferred from a count. One more limit is
stated at the point of use: a handle is read as a Telegram handle whatever service it
belongs to, because every service writes one as `@name` and this build cannot tell them
apart.

The Corpus holds one candidate that names no Contact Point — a Hard Negative describing
a contact form that asks for an address and then names none — so the reporting is
visible in the committed report and not only in the tests. It deliberately holds no
Telegram invite link: `t.me` is a real domain, and nothing in the Corpus is anything
but a Synthetic Entity.

Every Contact Point the Corpus publishes is a Synthetic Entity because the Corpus is
synthetic, and which figures say so is printed rather than asserted — an address is one
when its host sits under a TLD reserved for examples, and a handle when it carries the
`syn_` marker every Synthetic Entity in this Corpus carries, since Telegram reserves no
namespace of its own. The one value the reading holds that no post published is not one, and
the synthetic figure reads `5 of 6` for that reason rather than the `6 of 6` a reader would
prefer: it is the invented `tonight` named above, and a figure that could not come out wrong
would be a promise rather than a measurement. A Corpus Provider is a swap, so the same lines
over real content would read `0 of N`, which is what measuring them buys. Nothing on this
path reads the truth file, the Nuisance Structure manifest, or any published list: what a
post says to be reached at does not depend on what anybody registered.

## Campaign Candidates

`rfi campaign-candidates` is the whole path from the Corpus to an output: two
accounts reach the same Campaign Candidate when a chain of shared registrable domains
connects them, and on nothing else. The console output is the report — an
index of the candidates, then for each one the accounts, the posts, and the shared
domains that put them together — and it carries its own evidence with it because a
grouping a reader cannot check is a claim rather than a result. The Contact Points above
are the second thing ADR-0005 permits as an edge, and this command reads neither them
nor the file they are written to.

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
  filtered      3 of 16 registrations withheld, removing 2 of 5 components
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

## The recovery figure

`docs/campaign-recovery.md` is the report: how many of the two Planted Campaigns the
grouping recovered. **2 of 2**, with the shop printed beside them.

```
uv run rfi campaign-candidates   # writes the candidates this measures
uv run rfi campaign-recovery     # reads them, and the membership, and writes two files
```

| File | Holds |
| --- | --- |
| `data/evaluation/recovery.jsonl` | One line per Planted Campaign: its membership, its outcome, and every candidate that reached it, with the accounts held, missing, and unexpected. |
| `docs/campaign-recovery.md` | The report: the figure, the join behind it, the candidates that are not a recovery, the Nuisance Structure it was measured against, and what the number cannot say. |

**It is a separate command, and that is the point.** It reads the candidates
`rfi campaign-candidates` published and does no grouping of its own — it opens neither
the Public Suffix List nor the shared-infrastructure list — and nothing in the grouping
path opens the truth file or the Nuisance Structure manifest. The two steps are two
commands and a committed file between them rather than one process with a boundary
drawn inside it (ADR-0018), and
`tests/test_campaign_recovery.py` checks it by watching which files each of the two
opens. The evaluator also refuses a membership or a candidates file naming an account
the Corpus does not hold, and refuses a candidates file carrying a field it does not
know: a `campaign_id` there would be Planted Campaign membership in the pipeline's
own output, and quietly ignoring a field that has appeared is the first way to stop
being able to say the file does not carry one.

A candidate counts as a recovery only when it holds a campaign's whole membership.
That is stricter than it sounds: two accounts of a three-account campaign grouped on
their own registration is a real grouping and not a recovery, and a candidate holding
a campaign's three accounts and one of its own is an over-grouping rather than half a
recovery. Both are reported as `partial`, with the accounts held, missing, and
unexpected named, and neither is counted into the figure. The three outcomes —
recovered, partial, missed — partition the membership, so a reader can add them up and
get N rather than take the numerator on trust.

**Nobody reviewed any of it.** The figure is computed by this command against
membership the generator planted, and the report says so in its own words, because a
reader who cannot tell a command from a reviewer has been told something false about
how the number came about. The report also names the Nuisance Structure the figure
sits on — 20 records over seven kinds, every kind counted — because a recovery rate over
a Corpus that held nothing else would be a figure about a generator that planted two
campaigns and said so nowhere.

No figure over the whole Corpus is published in either view. Every label in this
Corpus was assigned by the generator that wrote the posts, so such a figure would
measure agreement with the generator rather than anything about fraud (ADR-0004), and
`tests/test_campaign_recovery.py` asserts the string that would name it appears
nowhere in the report.

The figure is reproducible from the seed the report prints, and the test proves that
rather than the report claiming it: it reads the seed out of the page, regenerates the
Corpus, the membership, the manifest, and the shared-infrastructure list at that seed,
re-runs the grouping over them, and joins again for the same figure and the same bytes.

The report states the limits rather than leaving them in the tickets, and one of them
is why the figure is only half the claim. A Planted Campaign leaning on a shared host
is lost with the host, an account that reaches no registration cannot be proposed at
all, and a campaign that rotates its registration per post is invisible by
construction — so this is a lower bound. Recovery on its own can also be produced by
a grouping that merges unrelated accounts, so the rate of false groupings against the
same manifest is not measured yet (ticket #19), and until it arrives a reader should
treat the figure as uninterpretable on its own.

## The Policy Score

**The Policy Score is a rules engine by design.** It is the published weights of the
Signals a post carries, added up and printed. No model output feeds it and none ever
will; the model's contribution to this project is the Confidence, which is measured
separately and never displayed. A reader can recompute every number from what the
output prints beside it, and that is testable rather than promised:
`tests/test_policy_score.py` takes the weights out of the committed file, applies
them to the Signals each post's own links justify, and requires the result to equal
the number the command displayed.

```
uv run rfi policy-score   # reads the Corpus and the three published lists, writes one file
```

| File | Holds |
| --- | --- |
| `data/signals/weights.jsonl` | Every Signal, its weight, and the one-line reason it is that number. Read as data; the scoring rules hold no weight of their own. |
| `data/signals/policy-scores.jsonl` | One line per post: the score, the points earned, the published total, and each Signal with its weight and the evidence it fired on. |

Five Signals reach the score. Two are read from links and three from the post's own text.
Nothing in this path needs anything about an account, and that is the whole of the
difference between a Signal and a thing that is not one: no account's age, karma, posting
rate, or activity change is read, and the Corpus file holds no such field to read — a
Corpus row carrying one is refused rather than ignored, which is what makes the constraint
structural. `tests/test_policy_score.py` checks it two ways: renaming every account in the
Corpus leaves every score identical, and the run is watched to open four published files
and never the truth file.

The two Link Signals are `domain_frequency`, which fires when a post links a
Registrable Domain two or more accounts in the Corpus reach and prints those two figures
beside it — `4 posts by 3 accounts` is the claim, and `data/domains/post-domains.jsonl`
is where a reader counts it — and `domain_lookalike`, which fires when a post links a
registration one edit away from another registration in the Corpus, under the same Public
Suffix, and names the other one. An edit is an insertion, a deletion, a substitution, or
a transposition, because a copy of a name is made by doing exactly one of those to it.

The three Content Signals read the post and nothing else, and they read it literally: a
Signal fires where one of its phrases appears in the post's own title or body.
`guaranteed_return` is the post claiming the outcome cannot fail, `payment_request` is
money asked for before any work is done, and `urgency_language` is a deadline or a claim
that waiting costs the reader the place. The whole of every phrase list is printed with
the weights, because a reviewer holding the post can only check a match if they can see
what the rule was looking for.

Each Content Signal carries the sentence it fired on rather than the phrase alone,
because the sentence is the judgement a reviewer wants to make — it either names the
claim or denies it — and it is the whole of what the rule looked at. One match is
silenced by a negator standing earlier in the same sentence, and the whole of the
negator list is printed with the phrase lists, so a reader can reproduce a match the run
dropped and not only one it kept. What that buys on this Corpus is `syn_p_0018`: a
declared Hard Negative, a genuine job post that spells out that it asks for no deposit,
whose title and body match three of `payment_request`'s phrases and lose all three to
the guard. `syn_p_0026` loses one more; the other posts that disclaim a deposit never
match at all. The guard is scoped to the sentence because `We do not ask for a deposit, a
kit fee, or any money up front` is one request denied three times, and a guard stopping at
the comma would read the second and third as requests.

The guard costs something and the output says so rather than leaving it in the ADR: a
post that asks for money in one sentence and denies asking in the next carries no
`payment_request`. A bare `no` is the one exception, counting only when it stands
directly before the phrase — `no experience needed` is in every job post ever written, so
a `no` that counted anywhere in the sentence would silence the Signal on exactly the posts
it exists for, which in this Corpus means `syn_p_0027`.

That first Signal is not computable from one post, and the project says so rather than
rounding the claim up: it counts how much of the Corpus reaches a registration. What it
can promise is that the counts are printed and checkable, and
`tests/test_policy_score.py` counts them a second time off `post-domains.jsonl` —
published by a different command over the same Corpus — and requires the two counts to
agree. Read the post, read the registration table printed beside the Signal, and the
posts and accounts that put it there are there to be found.

The three things a reader should be suspicious of are stated rather than buried.
A Signal is present or absent, so a post linking three shared registrations carries
`domain_frequency` once, not three times, and a post using four phrases from one Content
Signal's list carries that Signal once, not four times. The score is a subset-sum of the
published weights and nothing else: the points earned as a share of the published total,
to the nearest whole number out of 100 with halves going up. The frequency figures count
other posts; they are structure to go and look at, not a judgement about them. And the
known-shared registrations are out of the scoring path rather than scored at zero —
`hopcut.example` is reached by four accounts here, more than any planted registration, and
handing that to a Signal would reward every post that used a service everybody uses. A
Planted Campaign leaning on a shared host is lost with it, the same lower bound the
grouping reports.

One more thing the score cannot do by itself: it cannot say what to read first. Five
Signals give thirty-two subsets and this Corpus takes eight of them, so five posts tie at
45 and three at 14. That ordering is the Review Queue's job and not this command's, which
is why this index is ordered by score and then by post id - enough to be a stable table,
which is all it claims to be. `rfi review-queue` is where the ties are broken.

The lookalike Signal fires on both members of a pair and names neither as the copy,
because the spelling does not say which imitates which: the Corpus plants a
recruitment firm and a warehouse employer, each one character from a planted
registration, and nothing in the Corpus tells the four apart. The output says so at
the point of use. One case it does not reach: a host carrying a homoglyph comes back
from the Public Suffix List punycoded, so a Cyrillic `а` in `apple.example` makes the
registration `xn--pple-43d.example` and no edit-distance rule over the resolved form
can see what it was imitating.

Nothing in this Corpus promises a guaranteed return, and that is a finding rather than a
gap: the two Planted Campaigns go out of their way to publish their losing months, which
is the whole of their craft. `guaranteed_return` is published anyway, because it is the
strongest claim a post can make in its own words and it is a rule a reader can apply to a
real Corpus. `syn_p_0021`, a post about money already lost, carries no Content Signal at
all: the Signals score the pitch rather than the aftermath.

Adding a Signal is a two-part change and the run refuses to skip either half: the
rule goes in `src/reddit_fraud_intelligence/signals.py` and the weight goes in
`data/signals/weights.jsonl` with its reason. A weight file that leaves a Signal
unpriced, prices one that does not exist, carries a weight of zero, or leaves a
rationale blank stops the run and names the file and the row — otherwise a Signal
could appear in a breakdown that no arithmetic adds up to, which is the failure
ADR-0007 rules out.

## The Review Queue

**The order is the Policy Score, so the two can never disagree.** The queue is every post
in `data/signals/policy-scores.jsonl` sorted by the score the scoring command published for
it, and nothing else is consulted: no account's age or posting rate, no model output, and
the size of a Campaign Candidate moves nothing (ADR-0003, ADR-0007). That is why this is a
separate command rather than an option on the scoring one — a queue is only worth a
reviewer's attention if the number at the top of it is the number in the file beside it.

```
uv run rfi review-queue --depth 50   # reads the scores and the candidates, writes nothing
```

| Argument | Reads | Default |
| --- | --- | --- |
| `--scores` | `data/signals/policy-scores.jsonl`, which `rfi policy-score` wrote | that path |
| `--candidates` | `data/campaigns/campaign-candidates.jsonl`, which `rfi campaign-candidates` wrote | that path |
| `--depth` | how many entries to print | 50 |

It runs neither step again and opens neither the Corpus nor the membership, which is
ADR-0018's boundary used for a second purpose: measurement has to be unable to reach
inference, and so does a ranking. `tests/test_review_queue.py` watches the files a run
opens and requires the two published ones and nothing else. It is also the only command
here that writes no file, because the queue is a projection of two files other commands
publish and a third copy is a third thing to keep in step.

**Ties are broken on the points earned, and the depth is in the header.** The score is
points rounded to a whole number, so two posts can display the same one: 50 points and 49
both come out as 45 of 110 published. The queue orders those by the points, which is that
same arithmetic one step finer, and then by post id, which is stability rather than
judgement. Nothing else could be used, because anything else could put a lower-scoring post
above a higher-scoring one. The post id is last because a queue with nothing else in its
ordering would print a different one every time the Corpus changed.

Stated plainly, because a tie-break that never fires is worth knowing about: with the
published weights every sum is a multiple of five and no two of them round to the same
score, so on this Corpus the points decide nothing and the ordering falls through to the
post id. It matters for a weight set where two different sums do land on one score, and
`tests/test_review_queue.py` has to hand-write a row no run of this weight set could
produce to reach the case at all.

A depth is a parameter rather than a property of the Corpus, so it is stated twice in the
output: in the header sentence, which says how many posts the file holds and how many sit
below the cut, and as a figure of its own. A queue pasted into an issue without its depth
says nothing about what was left out, and the default 50 over this 34-post Corpus has to
say so rather than read as a claim that 50 posts were worth reading. A depth below one is
refused: an empty queue under a header is indistinguishable from a Corpus in which nothing
is worth reading, which is a claim about the Corpus this command has no business making.
Every post is ordered, including the ones carrying no Signal, or the depth could never bind
— how many posts carry a Signal is a property of the Corpus rather than of the depth.

**An entry carries its arithmetic, not a number.** Each entry prints the Signals behind its
score with the weight each was earned at and the evidence that fired it, the total against
the published total, and the Campaign Candidates the post sits in with the registration
each is joined on — an identifier on its own says nothing to somebody who has not opened
the candidates file. What a candidate is named beside an entry for is worth being exact
about: a hypothesis produced by cohesion analysis, never a confirmed claim about anyone's
behaviour (ADR-0005). It does not say the post is fraud, and nobody has reviewed any of it.

Two things are refused rather than smoothed over. A candidates file naming a post the
scores file does not hold stops the run, because a queue that dropped that candidate would
hide two published files disagreeing behind a shorter table. And a scores row whose score
is not the share of its own points that the file publishes beside it, or whose points are
not what its Signals are priced at, or whose rows disagree about the published total, stops
the run too — which is what makes "the order and the number cannot disagree" structural
rather than a promise. `read_policy_scores` also refuses a row carrying a field this
project does not publish, for ADR-0008's reason: the one field that must never appear
there is a Planted Campaign's identifier.

**No figure is computed here.** How many of the entries at a stated depth turned out to be
findings is a question about a reviewer's judgement about each of them, and every label in
this Corpus was assigned by the generator that wrote the posts, so a figure computed over
the queue here would measure agreement with the generator rather than anything about fraud
(ADR-0004). The depth is published because that measurement reads it as an input, not
because publishing the input measures anything. The words that would name such a figure
appear nowhere in the queue's own prose, which `tests/test_review_queue.py` asserts rather
than trusting this paragraph.

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

## The Corpus composition

`docs/corpus-composition.md` is the report ADR-0006 asked for: the Corpus's own
distribution over the ten Scam Categories, with CAFC's published base rate for the
same class beside it and the difference between them on the line. A post is placed
by reading its own title and body against a published phrase list per Scam Category,
the same way a Content Signal fires, so a reader holding the post can put the lists
beside it and arrive at the same placement.

```
uv run rfi corpus-composition   # reads the Corpus and the base rates, writes two files
```

| File | Holds |
| --- | --- |
| `data/corpus/composition.jsonl` | One line per post: its Scam Category, the sentences of its own text that placed it, and any other class the same post matched. |
| `docs/corpus-composition.md` | The report: both distributions, the widest gap, what the Corpus holds nothing of, the Other bucket, and every phrase list. |

**The comparison is the finding, and on this Corpus it is blunt.** Eight of the ten
Scam Categories hold no post at all, and **82.5% of real reports land in classes
this Corpus has nothing to say about**. All ten and the Other bucket, as the report
publishes them:

```
  category                          posts  corpus   CAFC  difference
  Investment and Money Offers         13   38.2%   7.5% +30.7pp
    corpus  ##############################             38.2%
    CAFC    ######                                     7.5%
  Work and Payroll                    12   35.3%   4.1% +31.2pp
    corpus  ############################               35.3%
    CAFC    ###                                        4.1%
  Other                                9   26.5%   3.0% +23.5pp
    corpus  #####################                      26.5%
    CAFC    ##                                         3.0%
  Bills, Invoicing and Collections     0    0.0%   1.1%  -1.1pp
    corpus                                             0.0%
    CAFC    #                                          1.1%
  Extortion                            0    0.0%   9.4%  -9.4pp
    corpus                                             0.0%
    CAFC    #######                                    9.4%
  Identity and Account Takeover        0    0.0%  31.0% -31.0pp
    corpus                                             0.0%
    CAFC    ########################                   31.0%
  Impersonating an Institution         0    0.0%  12.0% -12.0pp
    corpus                                             0.0%
    CAFC    #########                                  12.0%
  Merchandise and Goods                0    0.0%  12.0% -12.0pp
    corpus                                             0.0%
    CAFC    #########                                  12.0%
  Phishing                             0    0.0%  10.5% -10.5pp
    corpus                                             0.0%
    CAFC    ########                                   10.5%
  Prizes, Appeals and Psychics         0    0.0%   1.6%  -1.6pp
    corpus                                             0.0%
    CAFC    #                                          1.6%
  Relationships and Second Contacts    0    0.0%   4.8%  -4.8pp
    corpus                                             0.0%
    CAFC    ####                                       4.8%
```

Every recovery figure this project will publish is bounded by that, which is why the
report prints it rather than leaving it in a ticket. The Other bucket is a row
beside the ten rather than a remainder below them, held against CAFC's own residual
category at 3.0%: CAFC filed that one for a report its analysts could not describe
any other way, which is the position the projection is in when no list matches a post.
The two are a share of posts this project wrote and a share of reports Canada filed,
so they are read against each other rather than equated.

Two limits are stated in the report rather than buried here. **CAFC carries no
free-text field**, so it constrains which Scam Categories exist and what their
priors are and it cannot validate a text classifier — every placement in the file is
unvalidated by construction. And the lists read what a post is talking about, not
whether it is a scam: the Corpus plants genuine job adverts, satire about both
planted pitches, and complaints from people who lost money to one, and all of them
land in the class of the pitch they make. 158 phrases across ten classes is enough to
place the Corpus this project wrote and blunt enough to be wrong often on real
content. `tests/test_corpus_composition.py` asserts the first of those against the
Nuisance Structure manifest, which is the only direction ADR-0011 lets that file be
read in.


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
  known-shared infrastructure filtered as published data rather than as a list in the
  query, and its corroborated tier is ticket #26), 0010 (CAFC figures are
  computed from the cache), 0011 (the Nuisance Structure has a file of its own, and
  so does the shared-infrastructure list), 0012 (registrable domains are resolved
  from the published Public Suffix List), 0013 (the Scam Categories are CAFC's
  thematic categories read down to ten, and the 41 in ADR-0006 is corrected), 0014 (the
  Policy Score is a subset-sum of published weights, and a Signal fires once however many
  registrations fired it), 0015 (Content Signals are phrase lists with a
  sentence-scoped negation guard, and the sentence is the evidence), 0016
  (Contact Points are read literally — which 0019 amends — the ones this build cannot
  read are reported rather than repaired, and nothing groups on one yet), and 0017 (a post's Scam
  Category is read off its own text with published phrase lists, the first match wins,
  and CAFC cannot validate the reading), and 0018 (the evaluator joins the published
  candidates in a command of its own, so measurement cannot reach inference and the
  two steps are a file apart rather than a boundary drawn inside one process), and 0019
  (an obfuscated username is read as the identifier underneath it, an address's host is
  read exactly as written, and the reading is measured against a Labelled Set the
  generator writes beside the Corpus, which is the one place a measurement shares a
  command with the step it measures), and 0020 (the Review Queue is the Policy Score at a
  stated depth, its ties are broken on the points earned, it reads the published scores and
  candidates rather than re-running either step, and it computes no figure).
