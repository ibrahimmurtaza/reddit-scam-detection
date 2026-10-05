# Reddit Fraud Intelligence

An analysis system that ranks Reddit content for reviewer attention, links
suspicious content to shared infrastructure, and explains its reasoning. Built
as a portfolio artifact: the pipeline is exercised end-to-end against a corpus
the project controls, with a real Reddit corpus pursued in parallel under proper
authorisation.

## Language

### Corpus

**Corpus**:
The body of Reddit content the system analyses — posts, comments, accounts, and
the links between them. Independent of where it came from.
_Avoid_: Dataset, data, feed, records

**Content Item**:
One unit of the Corpus as a reviewer meets it: a post's own title, body, and
links. What the Review Queue ranks and what a Policy Score is computed from. In
this build every Content Item is a post, and a Content Item is never an account
or a Campaign Candidate, which are things Content Items sit inside rather than
are.
_Avoid_: Document, submission

**Corpus Provider**:
The component that supplies a Corpus to the pipeline. Swappable, so no analysis
code assumes where content originated.
_Avoid_: Scraper, connector, source, ingester, adapter

**Synthetic Entity**:
A generated account, domain, URL, or handle that corresponds to no real person.
Corpus content in the portfolio build.
_Avoid_: Fake account, mock user, dummy data, test fixture

**Hard Negative**:
Content that is legitimate but sits near the fraud boundary — scam-adjacent
discussion, genuine job posts, satire, people complaining about being scammed.
The cases a detector gets wrong.
_Avoid_: Tricky example, edge case, difficult sample

**Planted Campaign**:
A grouping of Synthetic Entities deliberately written into the Corpus, whose
membership is known by construction. The unit of evaluation, and the thing a
Campaign Candidate is scored against.
_Avoid_: Ground truth, test case, fixture

**Labelled Set**:
A file naming, for every post in the Corpus, the Contact Points that post publishes
and how each was written. What the Contact Points reading is measured against, so a
recall or false-positive figure is a measurement rather than a claim. Written by the
generator beside the Corpus, from what each post declared, never by hand.
_Avoid_: Ground truth, expected output, test fixture

**Nuisance Structure**:
Material that makes a step hard: in the Corpus, accounts and domains that are not
a Planted Campaign but are genuinely easy to confuse with one — shared link
shorteners, decoy account clusters, near-miss domain pairs, single-account domains —
plus a Planted Campaign's own posts when they are staggered paraphrases of one text,
and Contact Points published in disguise, which make extraction hard rather than
grouping. Without it, recovery rate is uninformative and recall is unmeasured.
_Avoid_: Noise, decoys (the project's own phrase is "decoy account cluster"),
distractors

### Links

**Public Suffix**:
The part of a host that nobody can register — `co.uk`, `com`, `github.io`. Taken
from the published Public Suffix List, not from memory.
_Avoid_: effective TLD, eTLD, domain suffix

**Registrable Domain**:
The domain somebody registered: a Public Suffix plus exactly one more label, so
`example.co.uk` and not `co.uk`. The unit a Campaign Candidate is built on. A
hostname is not a Registrable Domain, a Public Suffix is not one, and neither is
an IP address.
_Avoid_: eTLD+1, base domain, second-level domain, registered domain name

### Signals and scores

Three distinct numbers, never interchanged. Conflating them is the fastest way
to lose the credibility of every explanation the system gives.

**Signal**:
A feature computable from the content and links visible to the reviewer that
contributes to a Policy Score. Anything requiring account history is not yet a
Signal.
_Avoid_: Feature, indicator, heuristic, input

**Link Signal**:
A Signal read from a post's links rather than from its text — how much of the
Corpus reaches a Registrable Domain, whether that registration is one edit away from
another. A Link Signal may need the rest of the Corpus's links to compute, so what it
counted is printed beside it and has to be checkable against the resolved links. It
never reads anything about an account.
_Avoid_: Domain Signal, infrastructure signal

**Content Signal**:
A Signal read from a post's own title and body against a published list of phrases —
a claim that an outcome cannot fail, money asked for up front, a deadline put on the
reader. One post is all it needs. The whole of the phrase list is printed with the
scores, and the sentence each match was found in is the evidence, so the decision is
one a reviewer can redo and overturn.
_Avoid_: Text Signal, keyword, keyword list

**Conflict Signal**:
The `category_conflict` Signal, and the only one that reads two entity types at once — a
post and the Registrable Domain it links. It fires where a post's Scam Category
disagrees with the Scam Category that registration is associated with. Both halves are
printed with it and both are checkable by hand: the post against the Scam Category phrase
lists, the registration against the tally of what the postings reaching it were placed
in. Named this rather than "category-conflict Signal" so the prose in the code and the
output can say "the conflict Signal" without drifting from one name.
_Avoid_: Mismatch, anomaly, contradiction, category error (say which two things
disagree, and never treat Other as one of them: a post no list matched makes no
claim, so it disagrees with nothing)

**Associated Scam Category**:
The Scam Category a Registrable Domain is associated with, taken from the Corpus's
own postings: a strict majority of the posts reaching that registration that a list
placed, out of the two at least that have to be placed before any of them counts. A
registration with fewer placed postings than that, or whose placed postings split
evenly, has none and is reported as unassociated rather than decided by a tie-break.
It is a property of the Corpus rather than of the registration, and a withheld
registration keeps it while staying out of the scoring path.
_Avoid_: Domain category, site category, vertical (the projection is onto Scam
Categories, and the figure is about what this Corpus's postings say)

**Review Queue**:
The ranked list of content items a reviewer works through, ordered by Triage
Priority. The headline metric is the fraction of true findings within its
top D entries — precision at a stated depth, never accuracy — published by
`rfi campaign-recovery` at several depths, alongside the recovery and the
false-grouping rate.
_Avoid_: Inbox, dashboard, results

**Review depth**:
How many entries of the Review Queue a run prints. Stated in the output header,
because a queue without its depth says nothing about what was left below it.
_Avoid_: Limit, cap, cutoff, top-k

**Content Embedding**:
One fixed-width numeric vector computed from a Content Item's own title, its own
body and its links, and stored in a `vector` column so a similarity query can be
asked of it. What it reads is text and links and nothing else — no account, no
subreddit, no timestamp — because a vector built out of who posted would put
Planted Campaign membership back into the pipeline through a side door
(ADR-0008). The model that produced it, its width, and a digest of the recipe
travel with it, because two models' vectors have no common distance. It is a
reading of vocabulary, not of meaning, and a distance between two of them is a
statement about shared words.
_Avoid_: Vector store (that is the table, not the reading), fingerprint, vector
hash (the hashing is how features are placed in buckets, and the digest beside a
stored vector is of the text it came from, not of the vector)

**Policy Score**:
The additive 0-100 severity shown to a reviewer, computed from weighted Signals
with weights chosen and published by this project. The weights of the Signals a post
carries are added up and taken as a share of every published weight, so a Signal can
be added later without any existing score exceeding 100. Auditable, opinionated, and
not a probability.
_Avoid_: Risk score, score, scam score

**Confidence**:
The model's calibrated probability that an item exhibits the pattern it was
trained to find. Held internally; never displayed as a severity number.
_Avoid_: Risk score, score, probability of being a scam, likelihood

**Triage Priority**:
The ordering a reviewer uses to work the review queue. Derived from Policy
Score, not a claim about what the content is.
_Avoid_: Detection, flag, alert, priority score

**Detection**:
_Avoid entirely._ This system does not detect fraud; it ranks content for
review. Reserve "scam" for a judgment a reviewer has actually made.

### Judgment and relationships

**Campaign Candidate**:
A proposed grouping of accounts sharing at least one registrable domain or
Contact Point. A hypothesis produced by cohesion analysis, never a confirmed
claim about anyone's behaviour. It is the system's output; a Planted Campaign is
what it is measured against.
_Avoid_: Campaign, ring, operation, network, cluster of scammers

**Contact Point**:
An off-platform contact identifier extracted from content — a Telegram handle,
Discord ID, email address, or crypto wallet. Published in any of several spellings;
what a reader extracts is the identifier underneath the spelling.
_Avoid_: PII, contact info, handle, identifier

**Scam Category**:
A top-level class in the published projection of the Canadian Anti-Fraud Centre's
thematic categories onto ten classes plus an "Other" bucket. Every CAFC category
is placed in one, sits in Other, or is dropped with a stated reason, and every
placement carries a one-line rationale. The size of the "Other" bucket is a
finding, and is measured.
_Avoid_: Scam type, label, taxonomy (CAFC's categories are the source, not this
term)
