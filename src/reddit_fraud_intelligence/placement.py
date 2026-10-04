"""A post's Scam Category, and the phrase lists that place it (ADR-0017).

Two commands read this rule and have to agree about it. `rfi corpus-composition` places
every post so it can set the Corpus's own distribution beside CAFC's published base
rates, and `rfi policy-score` places every post so it can tell whether a post's Scam
Category disagrees with the category the Registrable Domain it links is associated with
(ADR-0021). A post the first command places in Work and Payroll and the second places in
Investment and Money Offers would make two published figures incomparable, and neither
command would have any way of noticing: each reads only its own output. So the lists and
the placing live here and both commands import them, and the two cannot drift unless this
function is wrong — which its own tests hold it to.

The rule itself is ADR-0017's and nothing here widens it. A post is placed by matching its
own title and body against one published list per Scam Category, the lists are tried in
the order `categories.py` declares the ten in, the first to match takes the post, every
other class that matched the same post is recorded beside the one it lost to, and a post
no list matches lands in Other. A list is a rule and not a number, so it is published by
being printed rather than held in a file: a reader who disagrees with one has to be able
to see it and quote it, and a run that printed only the placements would be asking a
reviewer to accept a class they cannot check.

Nothing here reads the truth file or the Nuisance Structure manifest, and nothing reads
the CAFC base rates: the placement is a function of one post's own text and the published
lists, which is what lets the Policy Score place a post without reading a fifth file
(ADR-0018).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from reddit_fraud_intelligence.categories import OTHER, SCAM_CATEGORIES
from reddit_fraud_intelligence.corpus import CorpusItem
from reddit_fraud_intelligence.text import sentences, spans, wrap

# Every phrase that places a post in a Scam Category, and the whole of each list:
# nothing outside these strings places one. Published by being printed rather than held
# in a file, for the reason ADR-0015 gives for the Content Signals' lists - a phrase list
# is a rule and not a number, and a reader who disagrees with one has to be able to see
# it and quote it.
#
# Each list is what its own `covers` line in `categories.py` says, written out as the
# words a post would use to make that pitch. They are deliberately short: eleven lists of
# a dozen phrases each are enough to place the Corpus this project writes and blunt
# enough to be wrong often on a real one, which is the limit both commands report.
CATEGORY_PHRASES: Mapping[str, tuple[str, ...]] = {
    "Identity and Account Takeover": (
        "account takeover",
        "account in my name",
        "cloned sim",
        "compromised account",
        "hijacked account",
        "hijacked my account",
        "identity stolen",
        "logged into my account",
        "my identity was stolen",
        "opened an account in my name",
        "sim swap",
        "sim-swap",
        "stolen identity",
        "took over my account",
        "unauthorised access",
        "unauthorized access",
    ),
    "Merchandise and Goods": (
        "counterfeit",
        "has not arrived",
        "item never",
        "knock-off",
        "knockoff",
        "never arrived",
        "never came",
        "never turned up",
        "not what i ordered",
        "order never",
        "refund never",
        "still not arrived",
        "stopped shipping",
        "tracking has not",
        "tracking hasn't",
    ),
    "Phishing": (
        "click here to",
        "click the link below",
        "confirm your card",
        "confirm your details",
        "give us the code",
        "one time code",
        "one-time code",
        "reset your password",
        "security code we",
        "sign in here",
        "sign in to restore",
        "six digit code",
        "suspended within",
        "unusual activity on your account",
        "update your billing",
        "verify your account",
        "verify your identity",
        "your account will be closed",
    ),
    "Impersonating an Institution": (
        "as your bank",
        "claiming to be from",
        "claiming to be your",
        "customer service for",
        "impersonating",
        "official helpline",
        "official hotline",
        "posing as",
        "pretending to be from",
        "speaking for your bank",
        "support agent for",
        "the bank has asked",
        "your bank has asked",
        "on behalf of your bank",
    ),
    "Investment and Money Offers": (
        "brokerage",
        "cannot withdraw",
        "can't withdraw",
        "copy trading",
        "forex broker",
        "investment opportunity",
        "managed account",
        "minimum ticket",
        "paid desk",
        "paid signals",
        "signal desk",
        "signal group",
        "signals desk",
        "starting capital",
        "trading course",
        "trading desk",
        "trading signals",
        "usdt",
        "withdrawal fee",
        "withdrawal rules",
    ),
    "Relationships and Second Contacts": (
        "asset recovery",
        "fund recovery",
        "get my money back",
        "get your money back",
        "met online",
        "money recovery",
        "recover my money",
        "recover the funds",
        "recover the money",
        "recover your money",
        "recovery agent",
        "recovery firm",
    ),
    "Bills, Invoicing and Collections": (
        "amount owed",
        "cancel your service",
        "collection agency",
        "debt collection",
        "disconnection notice",
        "final notice",
        "outstanding balance",
        "past due",
        "pay the outstanding",
        "service fee of",
        "settle your debt",
        "shut off",
        "you owe",
    ),
    "Work and Payroll": (
        "an hour",
        "apply now",
        "equipment is provided",
        "equipment provided",
        "full-time",
        "hiring",
        "job opening",
        "kit provided",
        "no experience",
        "paid on the friday",
        "paid weekly",
        "part-time",
        "per day",
        "per hour",
        "payout",
        "remote work",
        "role",
        "roles",
        "shift",
        "shifts",
        "they train you",
        "training provided",
        "work from home",
    ),
    "Prizes, Appeals and Psychics": (
        "a small donation",
        "claim your prize",
        "claim your winnings",
        "donate to",
        "lottery",
        "medical expenses",
        "please help",
        "processing fee",
        "psychic reading",
        "send a fee to claim",
        "sick child",
        "tarot reading",
        "winning ticket",
        "you have been selected",
        "you have won",
        "you've won",
    ),
    "Extortion": (
        "blackmail",
        "doxx",
        "doxxed",
        "i will publish",
        "intimate photos",
        "leak your",
        "nude photos",
        "or i publish",
        "pay or",
        "ransom",
        "sextortion",
    ),
}


@dataclass(frozen=True, slots=True)
class Placement:
    """One sentence of a post's own text, and what in it placed the post.

    The sentence rather than the phrase, for the reason ADR-0015 gives: the phrase says
    what the rule matched and the sentence says whether the post really makes the claim.
    """

    field: str
    phrases: tuple[str, ...]
    sentence: str


@dataclass(frozen=True, slots=True)
class Placed:
    """One post, the Scam Category it lands in, and what put it there.

    `evidence` is empty exactly when the post landed in Other, because a post no list
    matched has nothing to point at. `also_matched` names the other classes whose lists
    matched the same post: the tie is broken by the order the projection declares the
    ten in, and a tie the run resolved on its own is a tie the reader has to be able to
    see.
    """

    post_id: str
    account: str
    scam_category: str
    evidence: tuple[Placement, ...]
    also_matched: tuple[str, ...]


def place(item: CorpusItem) -> Placed:
    """One post's Scam Category, and the sentences in it that put it there.

    The first list to match takes the post, in the order the projection declares the ten,
    and every other list that matched is recorded rather than dropped. The tie-break is
    arbitrary in the sense that no rule prefers one pitch to another, and the cost of an
    arbitrary tie-break is a post placed on the declaration order rather than on what it
    says - which is why the other classes that matched go beside it.
    """
    found: dict[str, list[Placement]] = {}
    for scam_category in SCAM_CATEGORIES:
        evidence = _evidence(item, scam_category.name)
        if evidence:
            found[scam_category.name] = evidence

    placed = next(iter(found), OTHER.name)
    return Placed(
        post_id=item.post_id,
        account=item.account,
        scam_category=placed,
        evidence=tuple(found.get(placed, ())),
        also_matched=tuple(name for name in found if name != placed),
    )


def check_placements() -> None:
    """Refuse a Scam Category with no phrase list, or a list belonging to no category.

    The lists live here and the ten live in `categories.py`, so the two can drift apart
    the way a weight file and a Signal enum can. A class nothing can place a post in is
    the case worth catching: its row in the comparison would hold a base rate, no posts,
    and a difference of the full base rate, which reads as a finding rather than as a gap.
    Both commands call this before they place anything, so neither publishes a placement
    the lists cannot account for.
    """
    known = {scam.name for scam in SCAM_CATEGORIES}
    missing = sorted(known - set(CATEGORY_PHRASES))
    unknown = sorted(set(CATEGORY_PHRASES) - known)
    if missing or unknown:
        raise ValueError(
            f"the phrase lists hold {len(CATEGORY_PHRASES)} classes and the projection has "
            f"{len(known)}; no list for {missing or 'none'}; a list for a category the "
            f"projection does not have: {unknown or 'none'}. Every one of the ten needs "
            "a list, and no list may name a category the projection does not have."
        )


def phrase_count() -> int:
    """Every phrase in every list, so the cost of shortness is stated as a number."""
    return sum(len(CATEGORY_PHRASES[scam.name]) for scam in SCAM_CATEGORIES)


def render_phrases(heading: Sequence[str]) -> str:
    """Every phrase that places a post, printed in full, in the order they are tried.

    Published because a phrase list is a rule and lives in the code: without this the
    output says a post landed in Work and Payroll and leaves the strings that decide it
    unsaid, which is the whole of the auditability claim missing. The order is the
    projection's, and the order is the tie-break, so it is printed rather than left
    implicit.

    The heading lines are the caller's, because the two commands that print these lists
    have different reasons for printing them and the sentences that say so are theirs to
    write. The rule and the lists are this function's, so a change to either reaches both
    outputs at once rather than one of them.
    """
    lines = list(heading)
    width = max(len(scam.name) for scam in SCAM_CATEGORIES)
    for scam in SCAM_CATEGORIES:
        phrases = CATEGORY_PHRASES[scam.name]
        lines.append(
            f"  {scam.name.ljust(width)}  {len(phrases)} "
            f"{'phrase' if len(phrases) == 1 else 'phrases'}"
        )
        lines.extend(wrap(", ".join(f'"{phrase}"' for phrase in phrases), indent=width + 4))
    lines.append(
        f"  {OTHER.name.ljust(width)}  no list, which is what a post none of them matches "
        "lands in"
    )
    return "\n".join(lines)


def _evidence(item: CorpusItem, name: str) -> list[Placement]:
    """Every sentence of this post one of this class's phrases appears in.

    Title before body, for the reason the Content Signals read the title first: the title
    is what a reviewer sees first. A post matching nothing here is a post that class's
    list says nothing about, which is not the same as a post that is not of that class.
    """
    phrases = CATEGORY_PHRASES[name]
    found: list[Placement] = []
    for field, text in (("title", item.title), ("body", item.body)):
        for sentence in sentences(text):
            fired = tuple(sorted(phrase for phrase in phrases if spans(sentence, phrase)))
            if fired:
                found.append(Placement(field=field, phrases=fired, sentence=sentence))
    return found
