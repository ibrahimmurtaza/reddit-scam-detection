# Reddit Fraud Intelligence

An analysis system that ranks Reddit content for reviewer attention, links
suspicious content to shared infrastructure, and explains its reasoning. It
proposes Campaign Candidates — groupings of accounts that share a registrable
domain or a Contact Point — and orders content by a Policy Score a reader can
recompute by hand. It makes no claim about what the content is.

The project builds against a Corpus it generates. Nothing here reads real Reddit
content, and no account name, domain, or Contact Point in this repository
corresponds to a real person.

## Where it is

The Corpus generator, which is the credibility boundary the rest of the system
rests on. The pipeline itself — extraction, Signals, Policy Score, Campaign
Candidate analysis, Review Queue, evaluation — is not built yet. Its tickets are
numbered #3 to #28 in the tracker; this README is updated as they land.

## Running it

Python is pinned to 3.13 (3.14 does not yet have prebuilt machine-learning
wheels). [uv](https://docs.astral.sh/uv/) manages the environment.

```
uv sync
uv run rfi generate-corpus
```

That writes two files:

| File | Holds |
| --- | --- |
| `data/corpus/corpus.jsonl` | Content, accounts, and links. Nothing else. |
| `data/corpus/truth.jsonl` | Planted Campaign membership, written to a different path. |

The pipeline receives the Corpus file and nothing else; the truth file is joined
only by the evaluator, after inference has finished (ADR-0008). A reader does
not have to take the code's word for it, and regenerating the Corpus is not part
of the check — the committed file is what you grep:

```
rg -i campaign data/corpus/corpus.jsonl
```

No output. Then open the file and read the planted posts. They are written to be
read, so a reviewer can judge whether the content is realistic without running
anything. Every Synthetic Entity is marked `syn_` and every domain sits under the
reserved `.example` TLD, so nothing in the output can be mistaken for real data.

The generator is deterministic: a fixed seed produces byte-identical files, and
`tests/test_corpus_generator.py` holds the committed Corpus to that. At this
stage the seed fixes the window the Corpus covers and the minute of each post;
it does not yet change which entities are planted.

## Tests

```
uv run pytest
uv run mypy
```

## Where things are decided

- `GLOSSARY.md` — the vocabulary, enforced by `tests/test_vocabulary.py` against
  the phrases the project must never utter.
- `docs/adr/` — the decisions. The ones this code implements are 0001 (Corpus
  Provider), 0005 (Campaign Candidates require registrable infrastructure), 0007
  (Signals come only from observable text and links), and 0008 (the Corpus file
  carries no membership).
