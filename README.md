# Reddit Fraud Intelligence

An analysis system that ranks Reddit content for reviewer attention, links
suspicious content to shared infrastructure, and explains its reasoning. It
proposes Campaign Candidates — groupings of accounts that share a registrable
domain or a Contact Point — and orders content by a Policy Score a reader can
recompute by hand. It makes no claim about what the content is.

The project builds against a Corpus it generates. Nothing here reads real Reddit
content, and no account name, domain, or Contact Point in this repository
corresponds to a real person.

Seven things are built. The Corpus generator, which is the credibility boundary the
rest of the system rests on, and the Nuisance Structure it plants around the two
Planted Campaigns. The CAFC extract: real, analyst-reviewed fraud reports, cached
and committed, with the base rate of every thematic category computed from it. The
projection of those categories down to ten Scam Categories, and the comparison of the
Corpus's own distribution against those base rates. The pipeline itself: the
Registrable Domain of every link, then the Contact Points every post names, then
the Campaign Candidates those registrations and Contact Points produce, with
known-shared infrastructure filtered out as published data and with temporal
proximity and content similarity reported beside every candidate as corroboration
that can deprioritise one and never create one, then the Policy Score those
same links and posts add up to, with the arithmetic printed beside it, then the Content
Embeddings every post is stored with under a similarity index. And the
measurement: how many of the two Planted Campaigns that grouping recovered, as X of N,
joined by a command that runs after it rather than inside it, with the method's recall
bound stated as a count beside it. The Review Queue those scores are ordered into, at a
stated depth. The false-grouping rate beside that recovery figure, and the Review
Queue's precision at several depths, are measured by the same evaluator.
And the Confidence: one probability per Content Item, fitted here over seven counts read
off the post, measured out of fold against the base rate and the Policy Score, and
displayed nowhere.
The cohesion system and the graph report are not built yet. Their tickets are
numbered #26 and #28 in the tracker; confidence calibration is #27; this README is
updated as they land.

## Running it

Python is pinned to 3.13 (3.14 does not yet have prebuilt machine-learning
wheels). [uv](https://docs.astral.sh/uv/) manages the environment.

```
uv sync
uv run rfi generate-corpus
```

Two steps need a database: `rfi content-embeddings` stores its vectors in Postgres
through pgvector, and `rfi campaign-candidates` reads them back out of that table to
corroborate each candidate on content similarity — so `pgvector` has to be built and
activated first — see
`docs/pgvector.md`, which holds the build and the activation step. The connection
comes from the environment (`RFI_DATABASE_URL`, or the `PG*` variables libpq
reads), never from an argument.

That writes four files; the others in the table below are written by later
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
| `data/contacts/post-contacts.jsonl` | Every Contact Point a post names, per post, with the spelling and the field it was found in. | `rfi campaign-candidates`, which reads them from the Corpus rather than from here, and a reader |
| `data/campaigns/campaign-candidates.jsonl` | Every Campaign Candidate: its accounts, its posts, the shared registrations and Contact Points that join them, the temporal proximity of each, and the content similarity of every post in it. | `rfi campaign-recovery` and `rfi review-queue`, both of which check the figures row by row without acting on them |
| `data/evaluation/recovery.jsonl` | One line per Planted Campaign: its membership, its outcome, and every candidate that reached it, with the accounts held, missing, and unexpected. | a reader, and the recovery report beside it |
| `data/signals/weights.jsonl` | The published weight of every Signal, with the one-line reason it is that number. | `rfi policy-score` |
| `data/signals/policy-scores.jsonl` | Every post's Policy Score, with the Signal-by-Signal arithmetic behind it. | `rfi review-queue` |
| `data/embeddings/content-embeddings.jsonl` | Which post, which model, how wide, under which recipe, and the digest of the text each stored vector was computed from. No vector, and no account. | the reuse rule, and a reader recomputing the digests |
| `data/model/confidences.jsonl` | Every post's Confidence: the probability, the model, and the digest of the recipe. No Policy Score, no label, and no fold. | nobody yet — the Confidence is displayed nowhere, and the calibration ticket #27 is what will read it |

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
readable without knowing what the other candidates are is not readable. The
false-grouping rate is measured now, and the Corpus is small enough to read the
answer by hand: one false grouping of three, and the shop is it.

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
worst possible grouping input. ADR-0005 permits a Contact Point as a grouping edge and
`rfi campaign-candidates` now groups on one — which merges that Hard Negative into the
alpha Planted Campaign's candidate and costs the project a recovery, as the recovery
section below says. This command still builds no grouping of its own: the shared blocks
are the evidence the grouping reads, and they are measured here first so that the edge is
not an unmeasured input into the recovery figure ADR-0004 is measured against.

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
or shared Contact Points connects them, and on nothing else. The console output is the
report — an index of the candidates, then for each one the accounts, the posts, and the
shared registrations and Contact Points that put them together — and it carries its own
evidence with it because a grouping a reader cannot check is a claim rather than a
result. The Contact Points are read out of the Corpus by the same rule
`rfi contact-points` publishes, and the vectors are read out of the table
`rfi content-embeddings` fills, so the grouping is a function of the Corpus, of the two
lists it names, and of the vectors over the Corpus's own text: it still opens three
files and no fourth.

```
uv run rfi campaign-candidates   # reads the Corpus, the two lists, and the vector table
```

| File | Holds |
| --- | --- |
| `data/campaigns/campaign-candidates.jsonl` | One line per candidate: its accounts, its posts, its first-seen date, each shared registration with the accounts and posts that reach it, each shared Contact Point with the accounts that published it and every spelling the Corpus wrote it in, the temporal proximity of each of those, and the content similarity of every post in the candidate against a stated threshold. |

**The second edge is what makes a desk that pays for a domain per post visible.** Its
registrations move every time the advert is posted and nothing joins the accounts that
share them; one Telegram Contact Point published throughout is the same infrastructure by
any other name. On this Corpus that is `syn-nuisance-obfuscated-copperlantern`: one desk
publishing one Contact Point three ways — written out with a full stop between its
characters, written out with spaces, and with a digit standing in for a letter — where two
of the three accounts reach no registration at all. It comes out as `cc-04`.

**The two edges are not equally good, and every candidate says which one it rests on.**
A registration is somebody's property and this project resolves it against the published
Public Suffix List. A Contact Point is read out of a post by a rule with a measured
recall against the Labelled Set, and a candidate built on a Contact Point alone says so
on its own line rather than leaving the reader to work it out from the section heading:

```
cc-04  3 accounts, 7 posts, first seen 2026-06-06T14:23:00Z
  justified by  Contact Points, the weaker of the two readings; `rfi contact-points` has the measured recall, no two accounts under it posted inside the window
  timing        0 of 1 piece of evidence within the 24-hour window; nearest pair 42m apart
  similarity    2 of 7 posts within the 0.50 cosine-distance threshold; nearest pair 0.40 apart
  shared contact points
    syn_copperlantern  telegram, 3 accounts, 3 posts  22d14h16m across, outside the 24-hour window  written @s y n _ c o p p e r l a n t e r n, @s.y.n._.c.o.p.p.e.r.l.a.n.t.e.r.n, @syn_c0pperlantern
```

The label is worked out from the two evidence lists rather than stored, so the file and
the table cannot hold different opinions about which edge a candidate rests on.

**Timing corroborates a grouping and can never produce one.** Accounts sharing a
registration and posting within a tight window are more likely one operator, so every
candidate carries the temporal proximity of the things it rests on: the span from the
earliest post by any account reaching one to the latest, and the gap between the nearest
two. The window is a figure in the output rather than a rule in the code — 24 hours by
default, `--window-hours` to change it — and a piece of evidence is corroborated when its
span falls inside it:

```
  window        24 hours between two accounts on one piece of evidence
  timing        3 of 4 candidates and 4 of 6 pieces of evidence corroborated; no candidate removed
```

It may deprioritise and nothing else. A candidate the window corroborates nothing under is
printed in full, with the gap beside each of its pieces of evidence, named again below the
table under a heading of its own, and moved to the end of the list — which is why the desk
above is `cc-04` and not one of the three ahead of it, seven posts over three weeks being not
what a 24-hour window calls one operator. **Nothing is removed for want of timing**, because a domain two
accounts reached months apart may be a domain that changed hands and a campaign may simply
be a patient one. What it refuses is the case ADR-0005 was written about:
`tests/test_campaign_candidates.py` runs four accounts posting in the same minute with
nothing shared at a window of a year and requires an empty file, so the rule is about
timing rather than about the default.

**Content similarity corroborates a grouping and can never establish one.** Accounts whose
posts are near-identical are more likely running the same playbook, so every candidate
carries the content similarity of the posts inside it: for each post, the distance to its
nearest post by a *different* account in the same candidate, over the stored vectors. The
threshold is a figure in the output rather than a rule in the code — cosine distance at most
0.50 by default, `--similarity-threshold` to change it — and a candidate is corroborated
only when *every* post in it has a near-twin inside that distance:

```
  threshold     cosine distance at most 0.50 between two posts in one candidate
  similarity    1 of 4 candidates and 5 of 18 pieces of evidence corroborated; no candidate removed
```

It may deprioritise and nothing else, exactly as timing does: a candidate the threshold
corroborates nothing under is printed in full, with its nearest pair beside it, and named
again below the table under a heading of its own. The order carries two keys, the clock's
first and the vectors' second, so a candidate both corroborate is printed ahead of every
candidate the clock corroborates and the vectors do not. **Nothing is removed for want of
similarity**, and the reason it may not establish anything is the reason a threshold is
published beside every figure it judged: a scam template converges across unrelated
operators, so two unrelated desks pasting the same advert are evidence about the template and
not about who runs it. The Corpus's own copy of that case is
`syn-nuisance-decoy-converged`, whose three accounts paste one advert word for word — the
nearest text on this Corpus, at 0.1504 to 0.2091 apart — and reach no candidate at all,
because the only thing they share is the link shortener the filter withholds.

The order on this Corpus is the finding. `cc-01` is the shop: both signals agree, its three
accounts recycling the same updates at 0.45 to 0.48. `cc-02` is the alpha Planted Campaign,
whose four staggered paraphrases of one offer read as *different words* to a model that
reads words and not meaning — 0.59 at the nearest — so the clock corroborates it and the
vectors do not, and it is still proposed and still recovered. `cc-03` is beta, same verdict.
`cc-04` is the desk, which neither signal corroborates.

The other half of the finding is inside `cc-02`, where the two edges disagree: the
registration puts three accounts inside two and a quarter hours and the Telegram handle
brings in a fourth account two weeks later, which is the very account that costs the
project a recovery.

Both Planted Campaigns come out on their own registration — alpha on
`vantage-ledger.example`, which one of its accounts reaches through a mirror hostname,
and beta on `signal-harbor.example` — and both now carry the Contact Point their accounts
share as well. What it refuses is the case ADR-0005 was written about: accounts pasting
one advert word for word, with nothing shared but text, produce no candidate at all.
`tests/test_campaign_candidates.py` writes that case out by hand and asserts an empty
output; the Corpus's own copy of it is joined by nothing but the shortener those accounts
all use, so the filter rather than the absence of links is what silences it. The Corpus
also plants two near-miss pairs for the same reason, and neither reaches a candidate, so
nothing about the output depends on matching a name rather than a registration.

**What the second edge costs is published rather than absorbed.** It also joins
`syn_greyloch_6612` to the alpha Planted Campaign: that account is a Hard Negative that
lost money to the desk and published the desk's Telegram Contact Point while writing the
complaint down. Nothing in this project can tell an operator from a customer, so the
candidate holds the membership and one account from outside it, alpha's outcome becomes
`partial`, and the recovery figure below is 1 of 2 rather than 2 of 2. `cc-02` names the
account that cost it, and the false-grouping rate rises with it. The timing says the same
thing independently: the registration is corroborated and the Contact Point that joined the
complaint is not.

## Known-shared infrastructure

Five candidates used to come out of a grouping built on registrations alone and two of
them were junk: a link shortener and a link-in-bio page are Registrable Domains like any
other, so they grouped every account that touched them — `hopcut.example` alone reached
four accounts, and with the paste site six, which was `cc-01` in the list. They are
withheld now, before the grouping rather than after it, so a withheld registration
joins nothing and cannot bridge two accounts either. An address at one of those
registrations is withheld with it, because everybody who complains into a paste site
publishes that address.

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
  filtered      3 of 16 registrations and 0 of 6 Contact Points withheld, removing 2 of 5 components
```

followed by what left the graph, so a registration the resolved links hold and no
candidate names is a decision rather than a gap:

```
withheld  3 registrations, reached by 10 accounts; no candidate is joined on one of them
  biopage.example     link in bio     4 accounts
  hopcut.example      link shortener  4 accounts
  pastevault.example  paste site      3 accounts
```

What is left is the two Planted Campaigns and two desks that are not campaigns — the
shop, whose three accounts share a domain it is right to group on and which must never
be counted as recovery (ADR-0004), and the desk behind the obfuscated handles — and
every Hard Negative that reaches a shared host and nothing else is now in no candidate
at all. The four account figures above the index are a partition, so they add up to the
accounts in the Corpus: grouped, alone (something nobody else reaches), silenced
(nothing but known-shared infrastructure), and unreachable (nothing at all). The
partition is over both edges rather than over registrations, because an account grouped
by a shared handle has been reached. `silenced` is a line of its own because the six
accounts the filter silences are not the ten that touch a shared host — four of those
share a registration with nobody and are simply alone. The cost is stated rather than
hidden: a Planted Campaign that leans on a shared host is lost with the host, so a
recall figure measured this way is a lower bound. `tests/test_campaign_candidates.py`
proves the list is what decides, by adding a host to a copy of it and watching a
candidate disappear, and by running with an empty list and watching all five components
come back.

The figures at the top of the output are a partition of the Corpus's accounts, and two
accounts in the Corpus reach nothing at all: they are therefore beyond any amount of
grouping, which is the floor of the case rather than a defect in it.

Nothing in this path reads the truth file or the Nuisance Structure manifest: the
Corpus, the Public Suffix List, and the shared-infrastructure list are the whole
input, and `tests/test_campaign_candidates.py` checks that by watching which files
the run opens rather than by reading the code that decides what to open.

## The recovery figure

`docs/campaign-recovery.md` is the report: how many of the two Planted Campaigns the
grouping recovered. **1 of 2**, with the partial named, the shop and the obfuscated-handle
desk printed beside it, the false-grouping rate counted against the same manifest, and
the Review Queue's precision at several depths beside that.

```
uv run rfi campaign-candidates   # writes the candidates this measures
uv run rfi campaign-recovery     # reads them, and the membership, and writes two files
```

| File | Holds |
| --- | --- |
| `data/evaluation/recovery.jsonl` | One line per Planted Campaign: its membership, its outcome, and every candidate that reached it, with the accounts held, missing, and unexpected. |
| `docs/campaign-recovery.md` | The report: the figure, the join behind it, the candidates that are not a recovery, the recall bound, the false-grouping rate, the precision of the Review Queue at several depths, the Nuisance Structure it was measured against, and what the number cannot say. |

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
get N rather than take the numerator on trust. Alpha is `partial` on this Corpus
because `syn_greyloch_6612` is in its candidate, and the report prints that account
beside the figure rather than leaving the reader to find it.

**The recall bound is a count, and it is published on every run.**

```
  bound              0 of 2 Planted Campaigns have nothing inside them reaching a candidate
```

A campaign with nothing inside its own membership that reaches a Campaign Candidate is
beyond this method by construction: no candidate can hold two of its accounts without
also reaching outside the membership, and a candidate that does is an over-grouping
rather than a recovery. The count is measured against the edges the grouping published,
which is why a membership whose only shared registration was withheld is in it — the
evaluator cannot tell that case from a membership that shares nothing, and does not need
to, because both are beyond the method. The report names each one and lists what every
other membership shares that reaches a candidate, so the count is arithmetic a reader can
redo rather than a claim about the ceiling. It reads zero on this Corpus — both campaigns
share something with themselves — and it is printed at zero anyway, because a figure that
appears only when it is bad is a figure a reader cannot tell from a missing one.

The ceiling it states is why `1 of 2` is not a rate of fraud found in the world, and the
report says so in its own words. The ceiling is a limit on the method rather than a
prediction of the run: alpha is inside it and was still missed, because the candidate
holding it also holds an account from outside it.

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
is lost with the host, an account that reaches nothing the grouping can use cannot be
proposed at all, and a campaign that rotates both its registrations and its Contact
Points is invisible to a method resting on those two edges — so this is a lower bound,
with the size of the last case published as a count. Recovery on its own can also be
produced by a grouping that merges unrelated accounts, so the rate of false groupings
against the same manifest is published beside it (ticket #19) — 3 of 4 candidates on
this Corpus — and the Review Queue's precision at several depths is published with them.
Of the two rates on the page — recovery and precision — the lower of the two numbers is
the more trustworthy, deliberately.

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

Six Signals reach the score. Two are read from links, three from the post's own text, and
one reads both at once. Nothing in this path needs anything about an account, and that is
the whole of the difference between a Signal and a thing that is not one: no account's
age, karma, posting rate, or activity change is read, and the Corpus file holds no such
field to read — a Corpus row carrying one is refused rather than ignored, which is what
makes the constraint structural. `tests/test_policy_score.py` checks it two ways: renaming
every account in the Corpus leaves every score identical, and the run is watched to open
four published files and never the truth file.

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

The sixth, `category_conflict`, is the only one that reasons across two entity types, and
it is the Signal this project has no other way to produce. A post classified as a job scam
that links a registration the Corpus otherwise uses for investment pitches is evidence in
its own right: neither the post nor the registration says anything on its own. It fires
where the post is placed in one of the ten Scam Categories, the registration is associated
with a different one, and the association is a **strict majority of the postings reaching
that registration** — out of the two at least that have to be placed before any of them
counts. Both halves are printed with the Signal (`this post is Work and Payroll, and 2 of
3 placed posts reaching it are Investment and Money Offers`) and both are checkable by hand:
the post against the phrase lists printed with the output, the registration against the
tally in the registrations table. `tests/test_policy_score.py` recomputes the tally a
second time from `data/domains/post-domains.jsonl` and `data/corpus/composition.jsonl` —
two other commands' files, over the same Corpus — and requires the two to agree for every
registration in the Corpus.

The association is where most of this Corpus's registrations come out as nothing, and the
output says which and why rather than leaving them unexplained. A registration reached by
one placed post is one account's own site as far as this Corpus shows, and a registration
whose placed postings split evenly has nothing to prefer: `too few placed` and `no
majority` are printed in its row, and neither is a conflict. `Other` is not a class that
can be associated or contradicted, because it is where a post lands when no list matched it
and a post that says nothing disagrees with nothing — so the bucket's posts are counted in
the tally, printed last, and never decide anything. The known-shared registrations are out
of it for the same reason they are out of the frequency Signal: a majority drawn from
everybody's adverts is a majority about the shortener. Their class is still computed and
printed with `withheld yes` beside it, and counted in the figure only where it can be used.

**On this Corpus the Signal fires on no post, and that is a fact about the Corpus rather
than about the Signal.** Four of the sixteen registrations carry an associated Scam
Category — `vantage-ledger.example` with Investment and Money Offers from four postings,
`signal-harbor.example` with Work and Payroll from three, and two withheld ones beside them
— and every posting reaching any of the four agrees with it. The other twelve are one placed
posting wide or split evenly. There is no cross-pitch post here to find: an operator running
two pitches on one registration is material this Corpus does not plant, so the Signal's case
is exercised by `tests/test_policy_score.py` on Corpora written to contain it rather than by
the file this project ships. A Signal that fires nowhere on the data a reviewer can open
would be a claim; this one states which corpus it needs and what the shipped one lacks. The
smallest Corpus it can fire on is three postings: the floor needs two, and two placed
postings can only agree or split.

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
posts and accounts that put it there are there to be found. The conflict Signal is the
same claim one step further on: it counts how much of the Corpus reaching a registration
says something else, and its counts are recomputed the same way.

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

One more thing the score cannot do by itself: it cannot say what to read first. Six
Signals give sixty-four subsets and this Corpus takes eight of them, so five posts tie at
38 and three at 12. That ordering is the Review Queue's job and not this command's, which
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
all: the Signals score the pitch rather than the aftermath. Two Signals are published and
silent on this Corpus — `guaranteed_return` and `category_conflict` — and both say so here
rather than in the tickets, because a rule that never fires on the file a reviewer can
open is a claim until somebody says what it would take.

Adding a Signal is a two-part change and the run refuses to skip either half: the
rule goes in `src/reddit_fraud_intelligence/signals.py` and the weight goes in
`data/signals/weights.jsonl` with its reason. A weight file that leaves a Signal
unpriced, prices one that does not exist, carries a weight of zero, or leaves a
rationale blank stops the run and names the file and the row — otherwise a Signal
could appear in a breakdown that no arithmetic adds up to, which is the failure
ADR-0007 rules out. The sixth Signal needed a third file: the Scam Category lists it
places posts with, which `rfi corpus-composition` already published from the same
function, so they moved to `src/reddit_fraud_intelligence/placement.py` and both
commands print them from there rather than each holding a copy.

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
publish and a third copy is a third thing to keep in step. This command prints no
figure over the queue: precision at several depths over it is published by
`rfi campaign-recovery`, alongside the recovery and the false-grouping rate.

**Ties are broken on the points earned, and the depth is in the header.** The score is
points rounded to a whole number, so two posts can display the same one: 50 points and 49
both come out as 38 of 130 published. The queue orders those by the points, which is that
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

## The Content Embeddings

**This is the substrate ticket.** One sentence embedding per post, in a `vector` column in
Postgres under a cosine index, so a similarity query can be asked of it. It groups nothing,
corroborates nothing and decides nothing — ticket #24 is the first thing to read these
vectors — and its whole acceptance is that the reading is *possible later*.

```
uv run rfi content-embeddings   # reads the Corpus, needs a database, writes one file
```

| File | Holds |
| --- | --- |
| `data/embeddings/content-embeddings.jsonl` | One line per post: `post_id`, `model`, `dimensions`, `recipe`, and the SHA-256 of the text its stored vector was computed from. |

There is no report beside it and that is deliberate: every figure this command produces is in
its console output and quoted here, the reasoning is in ADR-0024, and the file is five fields
per post. What is published is the record and not the vectors — 34 posts of 256 numbers is
about 100 kB of `0.10000000149011612`, which is a file a reader has to take on trust, and the
values are in the database the ticket asked for them to be in. What the record buys is the
reuse rule in the open: a reader recomputes every digest from the Corpus and sees which text
each stored vector belongs to.

**The connection comes from the environment and there is no `--dsn`.** `RFI_DATABASE_URL`, or
the `PGHOST`, `PGPORT`, `PGUSER`, `PGDATABASE` and `PGPASSWORD` variables libpq reads. With
none of them set the command refuses rather than accepting whatever database libpq would have
picked for itself, and an argument is a place a password ends up in a shell history.
`CREATE EXTENSION vector` is per database and is not run here: `docs/pgvector.md` has the
activation step, and a database without it is refused with the statement to run.

**The index is HNSW, chosen over IVFFlat, and the run says what the planner did with it.**
IVFFlat will not answer a single query until it has been given training data and its `lists`
parameter is a guess to be revisited as a corpus grows; HNSW answers from the moment it is
built. At this Corpus's size the index is 56 kB over 34 rows and the planner will not use it,
so the command runs `EXPLAIN` on the query it actually issues and prints what came back
(`56 kB` is this project's own PostgreSQL 18.1 build, so `tests/test_embedding_store.py`
holds it whenever a database is configured, and a different build will print a different
figure):

```
  table     content_embedding (34 rows)
  index     content_embedding_hnsw
  indexed   hnsw on embedding vector_cosine_ops, 56.0 kB over 34 rows
  stored    34 Content Items: 0 computed, 34 reused, 0 removed
  plan      Seq Scan, not the HNSW index; see the footer for why that is the right answer here
```

A run that claimed its index was at work over 34 rows would be claiming something no
`EXPLAIN` agrees with, and the first Corpus to grow past the planner's threshold would
inherit the claim. The footer says it in as many words, and names what would flip the line.
`tests/test_embedding_store.py` reads the access method back out of the catalogue rather than
out of the run's own word for it, and refuses a table carrying an index of another method
under the name this one would use — `CREATE INDEX IF NOT EXISTS` is satisfied by a *name*, so
somebody else's B-tree would otherwise be left in place under every figure printed beside it.

**The memory characteristics differ between the free tier and a large corpus, which is the
reason the type was chosen rather than defaulted.** HNSW builds a multilayer graph, reads it on
every query, and gets roughly `rows × m` edges, so its cost grows with the Corpus — here 56 kB,
and on a constrained instance the largest single object a query touches is what decides how
large a Corpus can be indexed at all. IVFFlat holds `lists` centroids and nothing else, so its
memory does not grow with the table and neither does it answer: pgvector's own guidance is to
create that index only after the table has some data, and it suggests about `rows / 1000`
lists, so at 34 rows the index is one or two partitions holding a handful of vectors each. So
it is not that HNSW is smaller. It is that HNSW's memory grows with the Corpus and IVFFlat's
recall collapses, and a Corpus this project can hold in one table is not yet the size at which
that trade turns. ADR-0024 has the argument, and `halfvec` — which halves the same growing
quantity — is recorded there as the one to revisit at the size where the graph stops fitting.

**A vector is computed once and reused, decided before any embedding happens.** Every row
carries the digest of the text it came from, so a rerun over an unchanged Corpus digests each
post, embeds nothing, and prints `0 computed, 34 reused`. An implementation that embedded
first and skipped the write would satisfy the rule at the writer rather than at the work. One
edited post costs one embedding and not thirty-four, and rows for posts the Corpus no longer
holds are removed rather than left behind to come out of every query as though they were
posts somebody could read.

That removal is the one destructive thing this command does, and which rows it removes is a
function of which file was named on the command line, so it is guarded. A Corpus sharing no
post id at all with what the table holds is refused rather than allowed to empty it: that is
the shape of a mistyped `--corpus`, and it is the shape in which a run that meant to add a
post adds nothing and shares nothing. A Corpus that *edits* posts keeps every post it did not
touch and is never refused — which is why the guard reads post ids and not the digest of the
file, since editing one post rewrites that digest and would refuse every ordinary edit. Pass
`--replace` when the table really is meant to be disposable. The rows are derived, so nothing
is lost that a rerun cannot rebuild, but the Corpus they came from has to still exist.

**The model is this project's own, and it is published.** `hashed-word-ngrams-v1`, 256
dimensions, and the digest of the recipe — three facts a reader needs before a stored vector
means anything, because two models' numbers have no common distance. A feature is each word
and each pair of adjacent words inside one field; a count is damped as `1 + ln(count)`; each
feature lands in one of 256 buckets with its sign taken from the same `blake2b` digest; the
vector is scaled to unit length. `blake2b` rather than Python's built-in `hash()` because that
one is salted per process and a model producing a different Corpus on every run could not have
its output committed and held to its bytes.

The recipe digest is in the file and in the column rather than only in the output, because a
version number is a promise somebody has to keep and the digest is the promise checked:
editing the recipe without bumping `hashed-word-ngrams-v1` stops the run rather than quietly
reusing vectors the new recipe would not have produced. Nothing else would notice — the name
would match, the width would match, and the digests beside the rows would match the text.

**No account is embedded, and that is structural rather than promised.** The Corpus boundary
exists so the pipeline cannot see Planted Campaign membership (ADR-0008), and a vector built
out of who posted would put that back through a side door. `tests/test_content_embeddings.py`
re-reads the whole Corpus with every account renamed, every subreddit swapped and every
timestamp moved, and requires the vectors and the digests to come back identical; the
published record holds no account field for one to go in. The command reads the Corpus, the
database, and nothing else — no membership, no Nuisance Structure, no Public Suffix List —
which `tests/test_embedding_store.py` checks by watching which files the run opens.

**What the model is and is not, in its own words.** It reads shared vocabulary and nothing
else, so the closest pair on this Corpus is `syn_p_0012` and `syn_p_0013` at a distance of
`0.1504` — one advert pasted word for word by three accounts that share nothing, found by
vocabulary alone, with `syn_p_0014` at `0.2091` from the first of them — and it does not read
meaning: two posts making the same pitch in different words come out far apart, which is
exactly what the alpha Planted Campaign's staggered paraphrases do at `0.59`. That distance
is only legible against a baseline, and this command does not publish one, because a baseline
over every pair is quadratic in the database and the figure belongs to the step that uses it.
`rfi campaign-candidates` now measures it per candidate and states its threshold beside every
figure it judged (ADR-0026).

**And the links are part of what is embedded, which is a limit rather than a detail.** Two
posts sharing nothing but a link shortener share the words of that host whichever pitch they
are making, so a similarity over these vectors is partly a similarity of the hosts two posts
link, and ADR-0005's own edge is reached again here through the text. It is not independent
of the edge the grouping already has, which is the concrete reason a similarity may
corroborate a Campaign Candidate and never establish one.
`tests/test_content_embeddings.py` demonstrates it with two posts whose only shared vocabulary
is one host.

## The Confidence

**Every Content Item has a Confidence, and no reviewer ever sees one.** It is the model's
probability that a post exhibits the pattern the model was trained on, fitted here over seven
counts read off the post's own title, body and links. It goes to
`data/model/confidences.jsonl` and to `docs/confidence.md`, and it is displayed nowhere: not in
the Review Queue, not beside a Campaign Candidate, not as a number out of a hundred, and never
added to the Policy Score (ADR-0003, ADR-0027).

```
uv run rfi confidence   # reads the Corpus, the list, the membership and the scores; writes two files
```

| File | Holds |
| --- | --- |
| `data/model/confidences.jsonl` | One line per post: the probability, the model that produced it, and the digest of the recipe. |
| `docs/confidence.md` | The report: the figure out of fold, both baselines, every coefficient, and what the figure is not. |

**The model is published, and its performance is measured rather than assumed.**
`logistic-content-link-features-v1`, fitted by gradient descent at a stated step count, learning
rate and penalty, over seven features a reader can recount by hand from the post in front of
them. Logistic regression rather than a tree ensemble because seven positives cannot support one,
and no wheel rather than scikit-learn for the reason ADR-0024 gives for refusing a downloaded
sentence-transformer: a repository whose claim is that every published figure can be recomputed
should not depend on a 30 MB artefact to produce one. Two runs over the same Corpus produce the
same bytes, so the published file is held to what this code produces.

**Every probability is out of fold, and the folds are the Planted Campaigns.** A post's
Confidence comes from a fit trained without that post and without every other post of its
campaign, and the negatives are held out by their **account** rather than by their post, so two
posts by one voice cannot sit on either side of a fold. A training fit over seven positives would
be perfect by construction and would say nothing, which is why no figure on the page comes from
one.

**Two baselines, and the finding is uncomfortable.** The model's AUC beside the base rate's 0.5000
and the Policy Score's own ordering of the same posts:

```
  predictor                          AUC     log loss  Brier
  logistic-content-link-features-v1  0.7143  0.6048    0.1538
  constant base rate                 0.5000  0.5084    0.1635
  policy score                       1.0000  -         -
```

So on this Corpus the classifier orders the planted posts better than a coin and worse than the
rules engine this project already ships, and it is scored worse than the constant by a proper
scoring rule. `docs/confidence.md` says so in a sentence of its own rather than leaving it to be
subtracted out of a table, because a model that cannot beat the base rate is worse than saying
nothing and a reader who has to work that out has been handed the work. The Policy Score gets no
log loss and no Brier score, and the dash is the point: a 0-100 editorial figure has no
probability reading, and publishing a likelihood from one would be the fused score ADR-0003 rules
out.

**The training labels are the Planted Campaign membership, so this measures recovery of planted
structure and not a rate of fraud found in anything.** The generator wrote the posts and wrote
the labels; nobody reviewed any of it, and nothing here has seen real Reddit content. The report
gives that a section of its own — "What these figures are not" — rather than a footnote, because
it is the part a reader would otherwise skip and skipping it is what makes a number from that page
mean something it does not. **Calibration is not measured here at all** (ADR-0003): what is
published is discrimination and two proper scoring rules, and deciding whether the probability is
calibrated is ticket #27.

**It is displayed nowhere, and that is structural rather than promised.** `cli.py` is the only
module in this repository that imports the Confidence, so `rfi review-queue` and
`rfi campaign-candidates` cannot reach it even by accident.
`tests/test_confidence.py` checks that by walking the import graph, checks it again by rendering
the Review Queue and searching its output for the word, for the file, and for every published
probability in both the form a probability takes and the form a severity takes, and checks that
every figure the queue prints out of a hundred is a published Policy Score and nothing else. A
second test walks the AST of every module and refuses any expression with a confidence-named
thing on one side of an operator and a Policy-Score-named thing on the other — the strongest form
of "never summed", and one a grep would pass over.

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
beside it and arrive at the same placement. The lists and the placing live in
`src/reddit_fraud_intelligence/placement.py`, because `rfi policy-score` places every
post too: two commands that placed one post two ways would leave two published
figures a reader could not compare.

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

The workflow runs a `pgvector/pgvector:0.8.6-pg18` service container, which is
the exact extension version `docs/pgvector.md` records for the local install, so
a run there and a run here are the same extension on the same major version. Its
`PG*` variables are set for the job, so `tests/test_embedding_store.py` runs
rather than skipping. With no `PG*` variable and no `RFI_DATABASE_URL` set those
tests skip, and they say why: a machine with no server still runs the whole
offline suite, `tests/test_content_embeddings.py`, which holds the model and the
published file to their bytes without a database anywhere near it.

The suite needs no network: the tests that exercise the two fetching commands
patch `urllib.request.urlopen` to refuse, and the embedding command reaches the
network for nothing at all.

## Where things are decided

- `GLOSSARY.md` — the vocabulary, enforced by `tests/test_vocabulary.py` against
  the phrases the project must never utter.
- `docs/confidence.md` — the report ADR-0027 asked for: the Confidence, its model, its
  features, its out-of-fold figure beside two baselines, and a section saying what the figure
  is not.
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
  read are reported rather than repaired, and 0023 is what groups on one), and 0017 (a post's Scam
  Category is read off its own text with published phrase lists, the first match wins,
  and CAFC cannot validate the reading), and 0018 (the evaluator joins the published
  candidates in a command of its own, so measurement cannot reach inference and the
  two steps are a file apart rather than a boundary drawn inside one process), and 0019
  (an obfuscated username is read as the identifier underneath it, an address's host is
  read exactly as written, and the reading is measured against a Labelled Set the
  generator writes beside the Corpus, which is the one place a measurement shares a
  command with the step it measures), and 0020 (the Review Queue is the Policy Score at a
  stated depth, its ties are broken on the points earned, it reads the published scores and
  candidates rather than re-running either step, and it computes no figure), and 0021 (a
  post's Scam Category is compared with the category its Registrable Domain is associated
  with, the association is a majority of the postings reaching it, and a registration with
  too few of them is reported as unassociated rather than decided by a tie-break), and
  0022 (recovery, the false-grouping rate and queue precision are measured together by
  one command), and 0023 (a Contact Point joins two accounts on the same footing as a
  shared registration, every candidate is labelled with the edge that justified it, and
  the method's recall bound is published as a count rather than described), and 0024 (a
  Content Embedding is a published lexical model over a post's own text and links and
  nothing about its account, stored in an HNSW-indexed vector column, reused by the
  digest of the text behind it, and reported with the plan the planner actually chose
  rather than the index the project would have liked it to choose), and 0027 (the Confidence is a
  classifier over seven per-post counts, fitted on Planted Campaign membership, measured out of
  fold against a constant and the Policy Score, stored in a file of its own, and displayed
  nowhere).
