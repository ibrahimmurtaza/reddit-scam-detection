# Campaign candidates require registrable infrastructure

Two accounts are placed in the same Campaign Candidate only if they share a registrable domain or a Contact Point. Content similarity and temporal proximity are corroborating evidence, never sufficient on their own. The result is always presented as a candidate grouping, never as a confirmed campaign.

## Considered Options



- **Any shared edge suffices.** Rejected: link shorteners, pastebins, and widely-shared links connect thousands of unrelated accounts, so similarity-based grouping produces enormous numbers of false groupings and cannot be distinguished from noise.
- **Similarity plus time window.** Rejected as insufficient: scam templates converge across unrelated operators, so two independent campaigns running the same playbook look identical on both axes.

## Consequences

Recall is bounded by what shared infrastructure can reveal, and campaigns that rotate domains every post are invisible to this approach by construction — a limitation to state rather than paper over. Because requiring shared infrastructure is what makes the grouping claim defensible, the Contact Point extractor carries disproportionate weight in the result despite likely low recall. The distinction between Campaign Candidate and Planted Campaign is load-bearing: the first is this system's output, the second is the evaluation target.
