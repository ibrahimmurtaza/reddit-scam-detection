"""The glossary is enforced, not merely documented.

Every `_Avoid_` line in `GLOSSARY.md` is a rule, so a repo-wide test checks the
ones that can be checked mechanically. Scanned: the source, this README, the
committed Corpus, the resolved links, the generated CAFC report, the generated
registrable-domain report, the generated Contact Points and their report, the
generated Scam Category projection, the generated
Campaign Candidates, and the generated Policy Scores — the project's own prose
wherever it lands. Not scanned: this
directory, which is where the phrases live; `data/cafc/base_rates.jsonl`, which holds
CAFC's category names and not ours; and `data/public-suffix/public_suffix_list.dat`,
which is publicsuffix.org's text and not the project's.

The list is deliberately phrases rather than bare words. "Feature", "adapter",
and "input" appear on the `_Avoid_` lines for *Signal* and *Corpus Provider*,
but as guidance for naming the domain concept — banning them from Python source
would forbid ordinary programming English and train the project to ignore its
own glossary. Anything added here must be a phrase the project could never
utter in good faith.

Where one phrase is a prefix of another, only the shorter is listed: the match is
`re.escape`d into a word-bounded pattern, so "eTLD" already covers "eTLD+1" and
listing both would make the second a line that can never fail.

URLs are removed before scanning. A citation is not prose: the CAFC extract is
cited by its record URL, which happens to contain the path segment `/dataset/`
because that is what the Open Government Portal calls it. Requiring the project
to mangle a citation to satisfy its own glossary would make the provenance less
checkable, not more.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

URL = re.compile(r"https?://\S+")

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
    "effective TLD",
    "eTLD",
    "base domain",
    "second-level domain",
    "domain suffix",
    "registered domain name",
)


def scanned() -> list[Path]:
    return [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "src").rglob("*.py")),
        REPO_ROOT / "data" / "corpus" / "corpus.jsonl",
        REPO_ROOT / "data" / "corpus" / "nuisance.jsonl",
        REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl",
        REPO_ROOT / "data" / "contacts" / "post-contacts.jsonl",
        REPO_ROOT / "data" / "domains" / "post-domains.jsonl",
        REPO_ROOT / "data" / "infrastructure" / "shared-hosts.jsonl",
        REPO_ROOT / "data" / "signals" / "policy-scores.jsonl",
        REPO_ROOT / "docs" / "cafc-base-rates.md",
        REPO_ROOT / "docs" / "contact-points.md",
        REPO_ROOT / "docs" / "post-domains.md",
        REPO_ROOT / "docs" / "scam-categories.md",
    ]


@pytest.mark.parametrize("path", scanned(), ids=lambda path: path.name)
def test_the_glossary_words_the_project_avoids_are_never_used(path: Path) -> None:
    text = URL.sub("", path.read_text(encoding="utf-8"))

    for phrase in BANNED_PHRASES:
        found = re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE)
        assert found is None, f"{path.name} uses {phrase!r}, which GLOSSARY.md avoids"
