"""The external labelled set: real content, held out, and never trained on.

Everything this project measures so far is measured against a Corpus the project planted
itself (ADR-0004), which bounds every figure it can honestly make. This module supplies the
one thing a synthetic Corpus cannot: labelled text that was written by somebody else, for a
purpose of their own, before this project existed. It is read from committed bytes like
every other input here, so a reader who does not trust the figure can recompute it.

What it is: the job-advertisement split of DIFrauD, a domain-independent deception
benchmark published under the MIT licence, whose job-scams component is the Employment
Scam Aegean corpus — 1,430 real job advertisements, 60 of them labelled fraudulent by
people other than this project. The benchmark's own name for its sources is at the URL in
the provenance record; this file does not reproduce it, because GLOSSARY.md reserves a
word the published titles both contain and the reservation is about this project's claims
rather than a third party's.

What it is not, and this matters more than what it is:

* It is **held out**. Nothing is fitted on it. The Policy Score has no training step at all
  and the Confidence is never fitted here, so the figure is a measurement of a model that
  existed before these bytes were downloaded.
* It is **not Reddit**, and not scam posts. These are job advertisements from a job board
  between 2012 and 2014. They are promotional content, which is the same *kind* of content
  the pipeline reads and the same *kind* of question — is this offering something worth a
  reviewer's attention — but they are a different population. The report says so in the same
  words as every other figure here.
* It **cannot express a campaign**. A job ad has no account history, no shared
  registrable domain with another ad, and no multi-account structure, so the grouping, the
  recovery rate, and the false-grouping rate are not measurable on it and are not measured
  on it. What it can measure is whether the per-post readers fire on real fraud at all.
* It **has no links**. Every row is a body of text and a label, so the three link-based
  Signals are not merely absent from the figures — they are not computable here, and the
  report says which ones and why rather than presenting a score as though the whole rules
  engine had run.

The CAFC extract is the project's other real data and is a different kind of it entirely: it
carries no free text at all (`docs/cafc-base-rates.md`), so it fixes the taxonomy and the
priors and cannot say whether text is read correctly. The two are not alternatives; they are
the two halves of what a real external check can supply, and this is the half CAFC cannot.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from reddit_fraud_intelligence.jsonl import write_lines

# One file, one split. The benchmark publishes train/validation/test, and the test split is
# the one this project reads: a held-out claim made from a split the validation never saw is
# worth more than a larger one the model has effectively been tuned against, and 1.8 MB of
# committed bytes is a price the repository can carry.
SOURCE_URL = (
    "https://huggingface.co/datasets/difraud/difraud/resolve/main/job_scams/test.jsonl"
)
DATASET_URL = "https://huggingface.co/datasets/difraud/difraud"
PAPER_URL = "https://aclanthology.org/2024.lrec-main.468/"
LICENCE = "MIT"
LICENCE_URL = f"{DATASET_URL}/blob/main/LICENSE.txt"
ATTRIBUTION = (
    "DIFrauD, a domain-independent deception benchmark (LREC-COLING 2024), "
    "job_scams/test.jsonl, MIT licence. Its job-scams component is the Employment Scam "
    "Aegean corpus (University of the Aegean; job advertisements published via Workable, "
    "2012-2014), whose fraudulent listings were annotated by Workable staff. Used "
    "unchanged. Both sources' own names contain a word GLOSSARY.md reserves, so they are "
    "given by their URLs in the provenance record rather than written out here."
)
USER_AGENT = "reddit-fraud-intelligence/0.1 (external validation)"
_CHUNK_BYTES = 1 << 20

# The label the benchmark publishes is 1 for deceptive. Named here because the whole report
# is written in the project's own vocabulary, where a fraud label is a label some other
# project attached and never a thing this project concluded (ADR-0002).
DECEITFUL = 1
TRUTHFUL = 0


@dataclass(frozen=True, slots=True)
class Labelled:
    """One row: text somebody else wrote, and the label they gave it."""

    index: int
    text: str
    label: int

    @property
    def deceptive(self) -> bool:
        return self.label == DECEITFUL


@dataclass(frozen=True, slots=True)
class ExternalFacts:
    """What reading the cached bytes establishes, stated as a claim about bytes.

    A digest of the committed file rather than of the response, because the figure is read
    from the file and a reader recomputing it has the file. Reported the way CAFC reports
    its own extract: every number here is countable, and the report prints them beside the
    figure rather than citing them in prose.
    """

    path: Path
    sha256: str
    rows: int
    deceptive: int
    truthful: int
    longest_chars: int


def download(destination: Path) -> int:
    """Cache the split from its publisher, unchanged. Returns the bytes written.

    Written whole rather than streamed through a compressor, because unlike the CAFC
    extract this file is plain text and already small enough to hold: compressing it would
    make the committed bytes a function of the compression settings rather than of the
    data, which is the reason CAFC digests its uncompressed CSV instead.
    """
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return len(payload)


def digest(path: Path) -> str:
    """SHA-256 of the committed file, which is the bytes the figures were read from."""
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read(path: Path) -> tuple[Labelled, ...]:
    """Every row, in the file's own order.

    A row that is not `text` and an integer `label` is refused rather than skipped. The
    benchmark guarantees two fields and nothing else, and a row arriving with a third would
    mean the file is not the file this module's provenance names — which is the one thing a
    held-out set has to get right, since a silently dropped row is a label this project
    never saw and would have counted as agreement.
    """
    rows: list[Labelled] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != {"text", "label"}:
            raise ValueError(
                f"{path}:{number} holds {sorted(record)}, which is not the labelled-set "
                "vocabulary ['label', 'text'] this module's provenance names"
            )
        label = record["label"]
        if label not in (DECEITFUL, TRUTHFUL):
            raise ValueError(
                f"{path}:{number} carries label {label!r}, which is neither "
                f"{DECEITFUL} (deceptive) nor {TRUTHFUL} (truthful); a label this project "
                "cannot interpret is not a label it may measure against"
            )
        rows.append(Labelled(index=len(rows), text=str(record["text"]), label=label))
    if not rows:
        raise ValueError(f"the cached external set holds no rows: {path}")
    return tuple(rows)


def facts(path: Path) -> ExternalFacts:
    """The counts the report prints beside the figure, counted from the file."""
    rows = read(path)
    return ExternalFacts(
        path=path,
        sha256=digest(path),
        rows=len(rows),
        deceptive=sum(1 for row in rows if row.deceptive),
        truthful=sum(1 for row in rows if not row.deceptive),
        longest_chars=max(len(row.text) for row in rows),
    )


def write_provenance(path: Path, facts: ExternalFacts) -> None:
    """The one record of where these bytes came from and under what terms.

    Beside the file rather than in the report, because it is a fact about the bytes and the
    bytes are what everything else is read from.
    """
    write_lines(
        path,
        iter(
            [
                {
                    "attribution": ATTRIBUTION,
                    "dataset_url": DATASET_URL,
                    "deceptive": facts.deceptive,
                    "licence": LICENCE,
                    "licence_url": LICENCE_URL,
                    "longest_text_chars": facts.longest_chars,
                    "paper_url": PAPER_URL,
                    "rows": facts.rows,
                    "sha256": facts.sha256,
                    "source_url": SOURCE_URL,
                    "split": "test",
                    "trained_on": False,
                    "truthful": facts.truthful,
                }
            ]
        ),
    )