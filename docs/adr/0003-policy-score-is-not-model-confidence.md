# The displayed score is a policy artifact, not a model output

The 0-100 number shown in the dashboard is computed additively from weighted interpretable Signals, with the weights chosen and published by this project. Model confidence is a separate quantity, held internally and never rendered as a severity number. The two are stored in separate columns and never summed or displayed as one value.

## Considered Options

- **A single fused score.** Rejected: it mixes a calibrated probability with an editorial judgement about what matters, and the decomposition shown to the user would not correspond to anything the model actually computed. This is the most common way explainable-AI work gets attacked, and it is avoided here by construction.
- **Model confidence alone.** Rejected as a display value: an uncalibrated number between 0 and 1, formatted as 0-100, reads as a severity judgement it has not earned.

## Consequences

The weights become a published, arguable artefact of the project rather than a hidden hyperparameter — which makes them defensible but also means their reasoning must be documented. Calibration of the confidence figure is a separate concern with its own evaluation, and is not implied by the Policy Score being sensible.
