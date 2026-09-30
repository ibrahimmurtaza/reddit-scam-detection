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

**Nuisance Structure**:
Material in the Corpus that is not a Planted Campaign but is genuinely easy to
confuse with one — shared link shorteners, decoy account clusters, near-miss
domain pairs, staggered paraphrases. Without it, recovery rate is uninformative.
_Avoid_: Noise, decoys, distractors

### Signals and scores

Three distinct numbers, never interchanged. Conflating them is the fastest way
to lose the credibility of every explanation the system gives.

**Signal**:
A feature computable from the content and links visible to the reviewer that
contributes to a Policy Score. Anything requiring account history is not yet a
Signal.
_Avoid_: Feature, indicator, heuristic, input

**Review Queue**:
The ranked list of content items a reviewer works through, ordered by Triage
Priority. The headline metric is the fraction of true findings within its top D
entries — precision at a stated depth, never accuracy.
_Avoid_: Inbox, dashboard, results

**Policy Score**:
The additive 0-100 severity shown to a reviewer, computed from weighted Signals
with weights chosen and published by this project. Auditable, opinionated, and
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
Discord ID, email address, or crypto wallet.
_Avoid_: PII, contact info, handle, identifier

**Scam Category**:
A top-level class drawn from a documented projection of the Canadian
Anti-Fraud Centre's 41 thematic categories. An "Other" bucket is expected and is
measured, because its size is a finding.
_Avoid_: Scam type, label, taxonomy (the full 41-way source is not this term)
