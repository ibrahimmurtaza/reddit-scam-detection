# Signals come only from observable text and links, and the Policy Score is deliberately a rules engine

Every Signal contributing to a Policy Score must be computable from the content and links visible in front of the reviewer, so that a reader can recompute the score by hand. Account-behaviour Signals — account age, karma, posting frequency, activity changes — are excluded from v1 entirely, because in a synthetic corpus they are values the generator assigned rather than values inferred from evidence. The score is an additive sum over published rule weights; no model output feeds it. The model's contribution is the Confidence, which is held separately and never displayed.

## Considered Options

- **Include simulated account metadata.** Rejected: it makes the score look evidence-based while actually reading the answer key. Any reviewer asking "how did you get account age for a synthetic account?" gets the answer "I made it up", which ends the auditability argument that motivated separating the score in the first place.
- **Feed rule hits as features into a learned model.** Rejected: the additive decomposition stops corresponding to anything the model computed, so the published-weights claim becomes false while the explainability looks the same. The model would sit in the display path with no independent claim to make.
- **Drop the additive framing and show model output only.** Rejected: an uncalibrated 0-1 number rendered as 0-100 reads as a severity judgement it has not earned, and is uninterpretable without the model architecture.

## Consequences

The Policy Score is a rules engine and the project says so in the README, which costs nothing: the ML claim lives in the Confidence and in campaign-recovery numbers that are measured independently. Signals become cheap, inspectable, and testable, and adding a Signal is a documented weight change rather than a retraining event. The trade-off is recall — v1 cannot see anything about an account's history, so accounts that behave suspiciously but write innocuous posts are invisible. This is a scoping decision, not a permanent one: the Signals return when an authorised Corpus Provider can supply real account metadata, and the weight file is where that happens.
