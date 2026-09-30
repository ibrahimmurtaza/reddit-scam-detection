# Report direct adjacency as a baseline, and scored cohesion as the system

The Campaign Candidate query is built in two tiers and both are reported. The baseline is direct adjacency: union-find over accounts sharing a registrable domain or Contact Point, filtered to components of size two or more. The system adds corroboration — within-component content similarity, temporal clustering, and agreement on Scam Category — to filter those components down. Known-shared infrastructure (link shorteners, paste sites, link-in-bio services) is filtered as data, not as a hardcoded list in the query.

## Considered Options

- **Scored cohesion only.** Rejected: without the baseline there is no evidence that corroboration earns its keep, and a reader cannot tell a filtering decision from a bug.
- **Advisory cohesion** — report the baseline components and annotate them with corroboration scores, letting a reviewer decide. Rejected as the primary output: it pushes the judgement back onto the reader and produces an unranked pile rather than a queue.

## Consequences

Two numbers ship, and the lower one is the more trustworthy. This is deliberate: it shows the corroboration step's contribution as a measured difference rather than an assertion, and it leaves a fallback when a real Planted Campaign is filtered out by the corroboration rules. Keeping the shared-infrastructure filter in data rather than in the query means it can be regenerated, tuned, and — with a real Corpus Provider — replaced with an actual reputation feed.
