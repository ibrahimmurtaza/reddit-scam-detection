# Evaluate by recovering planted campaigns

The system is evaluated on the fraction of Planted Campaigns it recovers at a stated precision and review depth, measured against a Nuisance Structure that generates its own false groupings. Accuracy is not reported, because there is no held-out ground truth to compute it against.

## Considered Options

- **Accuracy on a held-out split.** Rejected: in a synthetic Corpus every label is assigned by the generator, so a held-out split measures generalisation within the generator's own distribution. The number would be high and meaningless.
- **Human review of synthetic posts.** Rejected as an evaluation: readers judging generated content measure their agreement with the generator's intent, not the system's recall.

## Consequences

The claim the project can make is narrow and defensible — "we recover X of N planted campaigns at review depth D, against a nuisance baseline producing Y false groupings" — rather than any statement about real-world detection. This also makes corpus construction a first-class concern: a generator with no Nuisance Structure produces an uninformative number, and the harder the nuisance, the more the recovery rate means. Evaluator code reads the Corpus through a frozen interface that exposes no Planted Campaign membership, so the pipeline cannot shortcut to the answer.
