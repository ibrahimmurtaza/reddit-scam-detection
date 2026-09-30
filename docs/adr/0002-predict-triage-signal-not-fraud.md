# Predict a triage signal, not fraud

The system's model does not predict whether content is a scam, because no content is labelled as one at collection time and no observer can verify it. It predicts a proxy: whether the item matches a pattern associated with confirmed fraud cases. The product claim is therefore about ranking reviewer attention, and the headline metric is precision at a stated review-queue depth, not accuracy.

## Considered Options

- **Unsupervised clustering, human-named.** Rejected as the primary path: it can only surface structures already present in the corpus, and it produces no ranked queue to measure.
- **Supervised on accumulating human review.** Kept as the intended destination, not the starting point: early proxy errors determine what a reviewer ever sees, so it needs a proxy to bootstrap from.

## Consequences

Every feature, metric, and label in the system is about *priority for review* rather than *fraud*. "Scam" becomes a word only a reviewer may apply, and the model's own confidence is never published as a verdict. Precursors for genuinely new scam families are added separately, via clustering, rather than being folded into the scoring model.
