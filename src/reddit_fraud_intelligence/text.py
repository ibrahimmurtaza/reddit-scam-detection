"""The sentences of a post's own text, and where a published phrase sits inside one.

Two commands read a post literally: `policy-score` looking for the phrases its
Content Signals fire on, and `corpus-composition` looking for the phrases that place a
post in a Scam Category. Both quote the sentence they matched in rather than the
phrase, because the sentence either makes the claim or denies it and the phrase cannot
tell you which, and both take the sentence to be a substring of the text it came from so
a reader can find the evidence in the post rather than take the row's word for it.

That is one rule, so it is written once here rather than in both callers. A second
copy of the sentence splitter would be a second definition of what a sentence is, and a
match printed against one of them would not be findable with the other.

The negation guard that sits on top of this in `signals.py` is deliberately not here:
whether a negator earlier in a sentence cancels a Content Signal is a decision about
scoring, and a Scam Category is a statement about what a post is talking about. The
splat guard applies to a claim of certainty and would silence the category a post that
denies asking for a deposit is nonetheless talking about.
"""

from __future__ import annotations

import re
from functools import lru_cache

# A sentence is what ends in `.`, `!`, or `?` followed by a space: the unit the evidence
# is quoted in. Splitting anywhere finer would lose the case where one negator governs a
# list — "we do not ask for a deposit, a kit fee, or any money up front" is one request
# denied three times, not three requests.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> tuple[str, ...]:
    """The sentences of one field, each one a substring of the text it came from.

    That last part is what the auditability claim rests on: the evidence is quoted
    rather than summarised or trimmed, so a reader can hold it against the post they are
    looking at and find it there, character for character.
    """
    return tuple(part for part in _SENTENCE_BREAK.split(text.strip()) if part)


def spans(sentence: str, phrase: str) -> tuple[int, ...]:
    """Where one published phrase appears in one sentence, counted from the start."""
    return tuple(found.start() for found in matcher(phrase).finditer(sentence.lower()))


@lru_cache(maxsize=None)
def matcher(phrase: str) -> re.Pattern[str]:
    """The pattern one published phrase is matched with.

    Case is folded by the caller and whitespace between the words is the only freedom,
    so a phrase cannot fire across a paragraph break. The lookaround stops `48-hour`
    matching inside `148-hour` and `guaranteed` matching inside `unguaranteed`, which a
    reader re-applying the rule by eye would not do.
    """
    body = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.compile(rf"(?<![0-9a-z]){body}(?![0-9a-z])")