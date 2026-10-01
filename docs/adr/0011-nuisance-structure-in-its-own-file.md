# Give the Nuisance Structure its own file

The generator writes four files rather than two. The Corpus holds content, accounts, and links; the truth file holds Planted Campaign membership; the Nuisance Structure manifest names everything else that was planted or recorded, and what each piece is for; and the known-shared infrastructure list is published as data. They are separate because they have two readers, and neither may read the other's files: the pipeline opens the Corpus and the infrastructure list, and the evaluator opens the truth file and the manifest.

## Considered Options

- **Nuisance Structure records as extra rows in the truth file.** Rejected: the truth file is what the evaluator joins, and the grouping step has no business reading it. Rows in that file are Planted Campaign membership; rows meaning "this must not group" would make it two vocabularies, and a reader could no longer tell which is which.
- **Known-shared infrastructure as a constant in the grouping code.** Rejected by ADR-0009, and this is where the cost lands: the list has to be replaceable by a domain-reputation feed, and a list written into a module is not. The list is published here so the grouping step reads a file, and a test holds the line that only the module publishing it may hold a host as a literal.
- **A `kind` field on `CorpusItem` so the Corpus says what it planted.** Rejected by ADR-0008. The Corpus is read by the pipeline, and a field naming decoy account clusters or Hard Negatives is an answer key in the file the pipeline is allowed to open.
- **One file for the manifest and the infrastructure list.** Rejected: the manifest is measurement and the infrastructure list is an input to inference. Merging them puts the Hard Negative labels one read away from the grouping path, which is the boundary ADR-0008 exists to keep visible.

## Consequences

The Corpus is still generated without consulting either of the other files, so a leak has to be added deliberately rather than arriving through a field — and the tests that hold that line are cheap. The cost is four paths to keep distinct, two more generated artefacts to commit, and a schema change to three files rather than one. The generator refuses any pair of them sharing a path, because a collision there silently destroys a file some reader depends on. Two of the manifest's kinds are computed from the Corpus rather than declared, so the manifest cannot disagree with the Corpus it describes; the rest are declarations, and the note beside each says what a grouping step must not do with it.
