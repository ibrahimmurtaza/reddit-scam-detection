# The corpus file carries no ground-truth labels

The generator writes two files: `corpus.jsonl`, containing only content, accounts, and links, and a separate `truth.jsonl` holding Planted Campaign membership. The pipeline receives the corpus and nothing else, and joins the truth file only in the evaluator, after inference has finished. A `CorpusProvider` trait enforces this in code, but the evidence is the file itself: anyone can grep `corpus.jsonl` for campaign membership and find nothing.

## Considered Options

- **One file with label fields, hidden behind the trait.** Rejected: the boundary would rest on an interface boundary nobody can check. A future reader would have to take the code's word for it, and the instinct to "just move the labels into the corpus for convenience" would eventually win.
- **Trait-only separation, no separate file.** Rejected on the same grounds: architecture is a claim, an absent field is evidence.

## Consequences

The generator must be able to produce a corpus without ever consulting its own membership record, which constrains how decoys and nuisance are planted — it cannot accidentally leak through a field. Anyone can verify the evaluation is honest by reading two files. The cost is that the corpus format and the truth format evolve separately, and the evaluator owns the join, so a schema change has to touch two files. This boundary is the foundation of every number the project reports; if it leaks, campaign-recovery figures are void.
