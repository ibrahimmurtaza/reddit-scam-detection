"""The glossary is enforced, not merely documented.

Every `_Avoid_` line in `GLOSSARY.md` is a rule, so a repo-wide test checks the
ones that can be checked mechanically. Scanned: the source, this README, and the
committed Corpus. Not scanned: this directory, which is where the phrases live.

The list is deliberately phrases rather than bare words. "Feature", "adapter",
and "input" appear on the `_Avoid_` lines for *Signal* and *Corpus Provider*,
but as guidance for naming the domain concept — banning them from Python source
would forbid ordinary programming English and train the project to ignore its
own glossary. Anything added here must be a phrase the project could never
utter in good faith.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

BANNED_PHRASES = (
    "detection",
    "detector",
    "risk score",
    "scam score",
    "confirmed campaign",
    "ground truth",
    "ground-truth",
    "fake account",
    "mock user",
    "dummy data",
    "test fixture",
    "dataset",
    "scam type",
)


def scanned() -> list[Path]:
    return [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "src").rglob("*.py")),
        REPO_ROOT / "data" / "corpus" / "corpus.jsonl",
    ]


@pytest.mark.parametrize("path", scanned(), ids=lambda path: path.name)
def test_the_glossary_words_the_project_avoids_are_never_used(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    for phrase in BANNED_PHRASES:
        found = re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE)
        assert found is None, f"{path.name} uses {phrase!r}, which GLOSSARY.md avoids"
