"""The Nuisance Structure and the Hard Negatives, written into the Corpus.

Recovery of Planted Campaigns is the project's one substantive claim, and the number
is only readable beside what it was measured against (ADR-0004). A Corpus where every
grouped account is a Planted Campaign hands any grouping logic a perfect score for
free, so this module holds the material that makes a grouping decision hard: decoy
account clusters that run the same converging template without sharing infrastructure,
decoy clusters that do share one, near-miss domain pairs, staggered paraphrases of one
text across a Planted Campaign, single-account domains, and the Hard Negatives.

Every account name, domain, and post here is a Synthetic Entity: account names carry
the `syn_` prefix and every domain sits under the reserved `.example` TLD. The notes
below are written to be read. A reviewer who opens `nuisance.jsonl` should be able to
see what each piece of material is for and what a grouping step must not do with it,
without running anything.

A `NuisanceStructure` is what the generator plants; a `NuisanceRecord` is the same
thing after the plan has filled in which posts and accounts carry it. Two of the kinds
are inventories rather than declarations — known-shared infrastructure and the posts
that touch it — because a hand-maintained list of who links a shortener is a list that
is wrong by the next seed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

from reddit_fraud_intelligence.content import SyntheticPost, link_hosts
from reddit_fraud_intelligence.infrastructure import BIO_PAGE, HOP_CUT, PASTE_VAULT
from reddit_fraud_intelligence.jsonl import JsonObject, read_rows, write_lines


class NuisanceKind(StrEnum):
    """What a piece of material is, and therefore what it is there to test."""

    DECOY_ACCOUNT_CLUSTER = "decoy_account_cluster"
    HARD_NEGATIVE = "hard_negative"
    KNOWN_SHARED_INFRASTRUCTURE = "known_shared_infrastructure"
    NEAR_MISS_DOMAIN_PAIR = "near_miss_domain_pair"
    SINGLE_ACCOUNT_DOMAIN = "single_account_domain"
    STAGGERED_PARAPHRASE = "staggered_paraphrase"


class HardNegativeCharacter(StrEnum):
    """The kinds of legitimate content that sit near the fraud boundary. The
    glossary names them; each one is a different thing a grouping step gets wrong."""

    GENUINE_JOB_POST = "genuine_job_post"
    SATIRE = "satire"
    SCAM_ADJACENT_DISCUSSION = "scam_adjacent_discussion"
    SCAM_COMPLAINT = "scam_complaint"


@dataclass(frozen=True, slots=True)
class NuisanceStructure:
    """Material planted for a reason, declared before it is planted.

    `hosts` is what the record is about, which is not always only what its own posts
    link to: a near-miss pair names the Planted Campaign domain it is a near miss to.
    """

    kind: NuisanceKind
    nuisance_id: str
    posts: tuple[SyntheticPost, ...]
    spread_hours: int
    hosts: tuple[str, ...]
    note: str
    characters: tuple[HardNegativeCharacter, ...] = ()

    def record(self, post_ids: tuple[str, ...]) -> NuisanceRecord:
        """This structure as the manifest will write it, once the plan has filled in
        which identifiers the posts got. The projection lives here beside the
        declaration, so a field cannot be added to one and forgotten in the other."""
        return NuisanceRecord(
            kind=self.kind,
            nuisance_id=self.nuisance_id,
            accounts=tuple(sorted({post.account for post in self.posts})),
            posts=post_ids,
            hosts=self.hosts,
            characters=self.characters,
            note=self.note,
        )


@dataclass(frozen=True, slots=True)
class NuisanceRecord:
    """A `NuisanceStructure` with the plan's post and account identifiers filled in."""

    kind: NuisanceKind
    nuisance_id: str
    accounts: tuple[str, ...]
    posts: tuple[str, ...]
    hosts: tuple[str, ...]
    characters: tuple[HardNegativeCharacter, ...]
    note: str


# A domain one account uses, on its own, with nothing to group it to. Two of them, in
# subreddits where a link is unremarkable, because a Corpus that never shows this case
# cannot show that grouping resists it.
_SINGLE_ACCOUNT_DOMAINS = (
    NuisanceStructure(
        kind=NuisanceKind.SINGLE_ACCOUNT_DOMAIN,
        nuisance_id="syn-nuisance-single-plainsaw",
        spread_hours=30 * 24,
        hosts=("plainsaw.example",),
        posts=(
            SyntheticPost(
                account="syn_thistledown_9088",
                subreddit="woodworking",
                title="Built a shooting bench out of two pallets and eight hex bolts",
                body=(
                    "Third one of these I have built and the first one that did not wobble, "
                    "which came down to a cross brace at knee height rather than anything to do "
                    "with the timber. Bolt heads recessed so nothing catches a jacket sleeve. "
                    "Cut list and the joinery drawings are here if they are useful to anyone."
                ),
                links=("https://plainsaw.example/bench",),
            ),
        ),
        note=(
            "plainsaw.example is one account's own site and nobody else links it. It must "
            "not produce a Campaign Candidate, and a grouping step that finds it has found "
            "a component of one, which ADR-0005 does not count."
        ),
    ),
    NuisanceStructure(
        kind=NuisanceKind.SINGLE_ACCOUNT_DOMAIN,
        nuisance_id="syn-nuisance-single-rimcure",
        spread_hours=30 * 24,
        hosts=("rimcure.example",),
        posts=(
            SyntheticPost(
                account="syn_veldtrail_5502",
                subreddit="bicycling",
                title=(
                    "Trued my own rear wheel three times before it held, here is what I "
                    "got wrong"
                ),
                body=(
                    "Third attempt at truing by hand and the first that stayed true for a month, "
                    "so writing it down. The mistake was tensioning the spokes in the order the "
                    "manual gives rather than small turns from the nipple nearest the rim every "
                    "time, which I now think is the entire difference. Dial indicator readings "
                    "before and after are on the page."
                ),
                links=("https://rimcure.example/trueing-notes",),
            ),
        ),
        note=(
            "rimcure.example is a one-off link in a subreddit full of one-off links. "
            "Grouping it is the false grouping this record exists to make measurable."
        ),
    ),
    NuisanceStructure(
        kind=NuisanceKind.SINGLE_ACCOUNT_DOMAIN,
        nuisance_id="syn-nuisance-single-run",
        spread_hours=30 * 24,
        hosts=("single-run.example",),
        posts=(
            SyntheticPost(
                account="syn_marrowgate_6205",
                subreddit="AskCooking",
                title="Sous vide timing for a 3kg ribeye, finally sorted",
                body=(
                    "Six attempts and I have it down to 54 minutes at 57C with a 20 minute "
                    "sear, which gives me medium rare all the way through without the grey "
                    "band near the bone. The fridge rest is doing more work than the bath "
                    "temperature, which is the thing nobody tells you. Timings and my notes "
                    "here."
                ),
                links=("https://single-run.example/ribeye-timing",),
            ),
        ),
        note=(
            "A domain one account uses, in a subreddit where a link is unremarkable. "
            "Nothing else links single-run.example, nothing resembles it, and it must not "
            "produce a Campaign Candidate either."
        ),
    ),
)

# The same advert, pasted from the same place, by three people who have never met. This
# is the case ADR-0005 refuses on purpose: content similarity and time proximity both
# say campaign, and the only thing that says otherwise is that no two of them share a
# registrable domain. All three shorten the same link, so the only infrastructure they
# have in common is known-shared infrastructure.
_DECOY_CLUSTER = NuisanceStructure(
    kind=NuisanceKind.DECOY_ACCOUNT_CLUSTER,
    nuisance_id="syn-nuisance-decoy-converged",
    spread_hours=8,
    hosts=("anvil-labels.example", "underhilltalent.example", "pellworthwork.example"),
    posts=(
        SyntheticPost(
            account="syn_draycott_5583",
            subreddit="RemoteWork",
            title="Remote annotation work, 26 an hour, kit posted out, starts Monday",
            body=(
                "Found this on a Discord and the people in it seem normal, posting here in case "
                "it is useful to anyone here. Twenty-six an hour, six hours a day, five days a "
                "week, paid on the Friday. No experience needed, they train you over the first "
                "two days. The equipment is posted out to you and it is yours to keep. Intake "
                "takes about an hour if your documents are already scanned, and the slot goes to "
                "someone else if you are late. Two places open this week. Short link because the "
                "long one broke for me."
            ),
            links=(
                "https://anvil-labels.example/apply",
                f"https://{HOP_CUT}/q4k",
            ),
        ),
        SyntheticPost(
            account="syn_underhill_7420",
            subreddit="forhire",
            title="Hiring: annotation work, 26 an hour, six hours a day, equipment posted out",
            body=(
                "Copying what I posted in another sub because it fills fast. Twenty-six an hour, "
                "six hours a day, five days a week, paid on the Friday. No experience needed, "
                "they train you over the first two days. The equipment is posted out to you and "
                "it is yours to keep. Intake takes about an hour if your documents are already "
                "scanned, and the slot goes to someone else if you are late. Two places open "
                "this week. Short link, the long one got flagged for me."
            ),
            links=(
                "https://underhilltalent.example/apply",
                f"https://{HOP_CUT}/q4k",
            ),
        ),
        SyntheticPost(
            account="syn_pellworth_3196",
            subreddit="digitalnomad",
            title="Anyone else doing remote annotation at 26 an hour? What is it actually like",
            body=(
                "Third reply in a thread I keep losing, so posting on its own. Twenty-six an "
                "hour, six hours a day, five days a week, paid on the Friday. No experience "
                "needed, they train you over the first two days. The equipment is posted out to "
                "you and it is yours to keep. Intake takes about an hour if your documents are "
                "already scanned, and the slot goes to someone else if you are late. Two places "
                "open this week. Shortened link because the full one will not fit on my phone."
            ),
            links=(
                "https://pellworthwork.example/apply",
                f"https://{HOP_CUT}/q4k",
            ),
        ),
    ),
    note=(
        "Three accounts, one copy-pasted advert, no shared registrable domain between any "
        "two of them and no Contact Point shared either. Their text is near-identical and "
        "their posts are hours apart, which is what a converged template looks like. "
        "Grouping them is the false grouping that makes the recovery figure readable; the "
        "only thing they do share is a link shortener, which is known-shared infrastructure "
        "and on its own cannot form a grouping."
    ),
)

# One small business, three accounts, one domain. This is the other half of the decoy
# problem. Here the grouping is right by ADR-0005 — the accounts do share a registrable
# domain — so the pipeline will produce a Campaign Candidate, and what it must not do is
# count it as a recovered Planted Campaign. A Corpus whose only multi-account domains are
# the planted ones cannot show that the recovery number is not just a count of everything
# that grouped.
_SHARED_DOMAIN_CLUSTER = NuisanceStructure(
    kind=NuisanceKind.DECOY_ACCOUNT_CLUSTER,
    nuisance_id="syn-nuisance-decoy-shop",
    spread_hours=14,
    hosts=("rivermill-bikes.example",),
    posts=(
        SyntheticPost(
            account="syn_rivermill_4417",
            subreddit="bicycling",
            title="Rivermill Bikes: winter services booked out to the 14th",
            body=(
                "Rivermill Bikes again, sorry. Winter services are booked out to the 14th, so "
                "if yours is due, ring ahead rather than turning up and being disappointed. "
                "Forty-five pounds including a new chain if it needs one. We are still at the "
                "same corner by the bridge, and we open at nine on Saturdays. Ring the shop or "
                "use the form on the site and either is fine."
            ),
            links=("https://rivermill-bikes.example/services/winter",),
        ),
        SyntheticPost(
            account="syn_rivermill_6620",
            subreddit="localbikegroup",
            title="Saturday ride from the shop at nine, two routes, all speeds",
            body=(
                "Rivermill Bikes is running the Saturday ride again from the shop at nine. Two "
                "routes, all speeds, and a stop at the top so nobody gets dropped. We are still "
                "at the same corner by the bridge, and we open at nine on Saturdays. If your "
                "wheel has been making a noise, ring the shop or use the form on the site and we "
                "will check it free before you set off."
            ),
            links=("https://rivermill-bikes.example/club-ride",),
        ),
        SyntheticPost(
            account="syn_rivermill_9085",
            subreddit="forhire",
            title="Saturday mechanic wanted at Rivermill Bikes, tools provided",
            body=(
                "Saturday mechanic wanted at Rivermill Bikes. Six hours, fifteen an hour, tools "
                "provided and you keep them at the end. We are still at the same corner by the "
                "bridge, and we open at nine on Saturdays, which is when the work happens. Free "
                "wheel check training on the job. Ring the shop or use the form on the site."
            ),
            links=("https://rivermill-bikes.example/jobs",),
        ),
    ),
    note=(
        "Three accounts of one shop on rivermill-bikes.example, repeating the shop's own "
        "sentences the way a business does. ADR-0005 says they group, so grouping them is "
        "correct — and this is not a Planted Campaign, so counting it as one would report a "
        "recovery that did not happen. It is the false positive that sits inside the answer "
        "rather than beside it."
    ),
)

def _hard_negative(
    character: HardNegativeCharacter, post: SyntheticPost, note: str
) -> NuisanceStructure:
    """One Hard Negative, on its own, recorded with what kind it is.

    Each is a single post by a single legitimate account, and each sits next to the
    boundary on purpose: the job posts ask for no money and say so, the satire recites
    a planted pitch in order to make fun of it, and the two complaint posts and the two
    discussion posts reuse the phrases a reader would key on.
    """
    return NuisanceStructure(
        kind=NuisanceKind.HARD_NEGATIVE,
        nuisance_id=f"syn-nuisance-hard-negative-{character.value}-{post.account}",
        spread_hours=30 * 24,
        hosts=link_hosts(post.links),
        posts=(post,),
        characters=(character,),
        note=note,
    )


# Two real job posts, one of them satirising the planted pitch word for word, two
# complaints from people who lost money, and two pieces of scam-adjacent discussion.
# Between them they cover every character the glossary names, and the second job post
# reaches the boundary only through a link-in-bio host, which is the shape of the case
# a host filter gets wrong in the other direction.
_HARD_NEGATIVES = (
    _hard_negative(
        HardNegativeCharacter.GENUINE_JOB_POST,
        SyntheticPost(
            account="syn_novemberquill_2260",
            subreddit="forhire",
            title=(
                "Hiring: part-time catalogue tagging, 18 an hour, paid weekly, nothing "
                "to buy up front"
            ),
            body=(
                "We are a two-person label maker and we need about twenty hours a week of "
                "catalogue tagging. Eighteen an hour, paid on the Friday, four weeks paid "
                "holiday. You need your own laptop. We do not ask for a deposit, a kit fee, "
                "or any money up front, and if anyone ever does ask you for it, it is not us "
                "and I will say so in here. Two references at the end, both from people we "
                "have worked with before, which is the only way to check anything in this "
                "line of work."
            ),
            links=(
                f"https://{BIO_PAGE}/novemberquill",
                "https://novemberquill.example/catalogue",
            ),
        ),
        note=(
            "A real employer whose post happens to say the things a scam post says, only "
            "meant honestly: no deposit, no kit fee, no compulsory call. It has to be read "
            "rather than keyword-matched, and nothing in it connects to any other account."
        ),
    ),
    _hard_negative(
        HardNegativeCharacter.GENUINE_JOB_POST,
        SyntheticPost(
            account="syn_halverson_8815",
            subreddit="RemoteWork",
            title=(
                "Weekend stock-check cover at the shop on Saturdays, cash at the end of "
                "the day, two slots left"
            ),
            body=(
                "Our neighbour's shop needs cover for the Saturday stock check. Six hours, "
                "cash at the end of the day, and the same two people have done it for two "
                "years. No experience needed, no uniform to buy, nothing to pay for anything "
                "at any point. It is not something to build anything on and I would rather "
                "say that here than let somebody quit a job for it."
            ),
            links=(f"https://{BIO_PAGE}/halverson",),
        ),
        note=(
            "A genuine offer from a business with no website, so its only link is a "
            "link-in-bio page — the same host a scammer would use, for an unrelated "
            "reason. The host must not join the two."
        ),
    ),
    _hard_negative(
        HardNegativeCharacter.SATIRE,
        SyntheticPost(
            account="syn_pennyfarthing_8834",
            subreddit="CryptoCurrency",
            title=(
                "PSA: my brokerage has added a crystal ball to the premium tier, minimum "
                "ticket 500 USDT"
            ),
            body=(
                "I bought the premium tier myself so nobody else has to. It is 500 USDT, the "
                "onboarding call is compulsory and runs six hours, and at the end of it they "
                "hand you a laminated certificate with your own risk profile printed on it. "
                "The monthly report includes the losing months, annotated with a crayon "
                "drawing of where the losing happened. Worst month so far was down six "
                "percent, published the same afternoon, because publishing the losing months "
                "is the entire pitch. Ten out of ten, would not recommend to a friend."
            ),
            links=(f"https://{BIO_PAGE}/pennyfarthing",),
        ),
        note=(
            "Satire that recites the planted pitch almost word for word in order to mock "
            "it. It carries no infrastructure at all, so any grouping that reaches it did "
            "it on text alone, which ADR-0005 forbids as a sufficient reason."
        ),
    ),
    _hard_negative(
        HardNegativeCharacter.SCAM_COMPLAINT,
        SyntheticPost(
            account="syn_greyloch_6612",
            subreddit="personalfinance",
            title=(
                "Lost 1,100 USDT to a desk that publishes the losing months, writing this "
                "while it is still fresh"
            ),
            body=(
                "The pitch was a signals desk running entries off liquidity sweeps. Minimum "
                "ticket 500 USDT, compulsory onboarding call before they show you a trade, "
                "and a report every month whether the month was green or red. The part about "
                "publishing the losing months is what convinced me, and it turns out that "
                "was true: they published them, in detail, three weeks after the account was "
                "gone. Transcript of the last fortnight is on the paste. I have the "
                "statements and I am not putting them up here. The channel they had me "
                "add was @Syn_VantageLedger, which is how I found the rest of it in the "
                "first place."
            ),
            links=(f"https://{PASTE_VAULT}/g7k",),
        ),
        note=(
            "Someone who lost money, quoting the pitch back. This is the Hard Negative a "
            "text-similarity step merges into the campaign it describes, and a reader has "
            "to tell the complaint from the advertisement without being told which is which."
        ),
    ),
    _hard_negative(
        HardNegativeCharacter.SCAM_COMPLAINT,
        SyntheticPost(
            account="syn_teasdale_4405",
            subreddit="CryptoCurrency",
            title=(
                "PSA to whoever is running the 500 USDT minimum desk with the compulsory "
                "onboarding call"
            ),
            body=(
                "Two people in this thread have just been taken, so putting it plainly. The "
                "desk is 500 USDT minimum, the onboarding call is compulsory, and the monthly "
                "write-up including the losing months is the sales pitch, because the losing "
                "months never make it into the summary they send you afterwards. If your first "
                "call is a sales call, that is the whole of it. I would rather warn people "
                "than sit on it."
            ),
            links=(f"https://{BIO_PAGE}/teasdale",),
        ),
        note=(
            "A warning post carrying every distinctive phrase of the planted pitch and no "
            "shared infrastructure whatsoever. Same words as the campaign, the opposite of it."
        ),
    ),
    _hard_negative(
        HardNegativeCharacter.SCAM_ADJACENT_DISCUSSION,
        SyntheticPost(
            account="syn_wrenfield_3308",
            subreddit="CryptoCurrency",
            title="What a signals pitch sounds like when you take it apart phrase by phrase",
            body=(
                "Collecting the phrases that keep turning up in the same shape, because it is "
                "easier to spot than to prove. Minimum ticket 500 USDT. Onboarding call "
                "compulsory before any trade is shown. Publishes the losing months, which is "
                "true, and is the reason people believe the rest of it. Entries sized off "
                "liquidity sweeps. A refundable materials deposit and a 48-hour intake "
                "window. None of those is proof on its own. Four of them in one post is a "
                "pattern, and counting to four does not need a model."
            ),
            links=(f"https://{PASTE_VAULT}/phrase-list", f"https://{HOP_CUT}/t"),
        ),
        note=(
            "Scam-adjacent discussion: a reader's own list of the tell-tale phrases, which "
            "between them is every distinctive phrase in the planted campaigns. It links two "
            "known-shared hosts and must still be one account of its own."
        ),
    ),
    _hard_negative(
        HardNegativeCharacter.SCAM_ADJACENT_DISCUSSION,
        SyntheticPost(
            account="syn_marlowe_9960",
            subreddit="personalfinance",
            title="Checked the broker three people here recommended, here is what I found",
            body=(
                "Not naming them in the title. Their site has no company number on the terms "
                "page, the withdrawal rules change depending on how you phrase the question, "
                "and support answers arrive under a different name each time. Screenshots and "
                "dates are side by side on the paste so anyone can judge them rather than take "
                "my word. The contact form is half-built: it asks for an address and then tells "
                "me to send it to intake@, which is not an address. I am not claiming it is the "
                "same operation as anything else. I am saying I would not have spotted it without "
                "writing it down."
            ),
            links=(f"https://{PASTE_VAULT}/broker-check",),
        ),
        note=(
            "Careful, sourced criticism of a real-looking service. Nothing about it resembles "
            "the planted copy, which is the point: a Hard Negative that only looks hard when "
            "the keywords line up is not a baseline."
        ),
    ),
)

# Two domains that are one character apart, one side of each pair belonging to a
# Planted Campaign. The near-miss side is an unrelated business that happens to have
# taken the name: a recruitment firm and a small employer. Both are legitimate, which
# is what makes a resemblance-based rule expensive rather than merely wrong.
_NEAR_MISS_PAIRS = (
    NuisanceStructure(
        kind=NuisanceKind.NEAR_MISS_DOMAIN_PAIR,
        nuisance_id="syn-nuisance-near-miss-vantage-ledgers",
        spread_hours=30 * 24,
        hosts=("vantage-ledger.example", "vantage-ledgers.example"),
        posts=(
            SyntheticPost(
                account="syn_lanternrow_5548",
                subreddit="forhire",
                title="Vantage Ledgers is hiring two part-time bookkeepers, hybrid, four days",
                body=(
                    "We are a small recruitment firm and we are hiring two part-time "
                    "bookkeepers for a client in the north. Four days a week, hybrid, "
                    "above the going rate for the region, and we will pay for a qualification "
                    "if you do not have one. Two references, one of which will be from "
                    "inside the firm. There is no kit to buy and no fee of any kind at any "
                    "point in the process, which I realise makes this post sound unusual to "
                    "people who have read a few of the other ones."
                ),
                links=("https://vantage-ledgers.example/roles/bookkeeping",),
            ),
        ),
        note=(
            "vantage-ledger.example belongs to the planted crypto campaign; "
            "vantage-ledgers.example belongs to a recruitment firm that took a similar "
            "name. They are two registrable domains one character apart. Any rule that "
            "matches names rather than registrable domains joins a bookkeeping job to a "
            "signals desk, and the Corpus is unmeasurable without both being present."
        ),
    ),
    NuisanceStructure(
        kind=NuisanceKind.NEAR_MISS_DOMAIN_PAIR,
        nuisance_id="syn-nuisance-near-miss-signal-harbour",
        spread_hours=30 * 24,
        hosts=("signal-harbor.example", "signal-harbour.example"),
        posts=(
            SyntheticPost(
                account="syn_ashgrove_3318",
                subreddit="RemoteWork",
                title="Signal Harbour is taking on two warehouse pickers, shifts of four hours",
                body=(
                    "Family firm, two new picker roles, four-hour shifts you can pick from, "
                    "paid weekly, and we will pay for the licence if you do not have one. "
                    "Interview is a fifteen minute phone call and there is no second stage. "
                    "We ask for nothing up front and there is no equipment charge, because "
                    "every week somebody tells me a job like this should have had one."
                ),
                links=("https://signal-harbour.example/jobs/picking",),
            ),
        ),
        note=(
            "signal-harbor.example belongs to the planted annotation job; "
            "signal-harbour.example is a real employer whose name differs by one "
            "character. A British spelling is exactly what a registration that is not "
            "careful produces, so the pair belongs in the Corpus rather than in a list of "
            "puns about it."
        ),
    ),
)

STRUCTURES: tuple[NuisanceStructure, ...] = (
    *_SINGLE_ACCOUNT_DOMAINS,
    _DECOY_CLUSTER,
    _SHARED_DOMAIN_CLUSTER,
    *_HARD_NEGATIVES,
    *_NEAR_MISS_PAIRS,
)

# Spare material: the same kinds, a little stranger. A seed plants some of it and leaves
# the rest, which is what makes two seeds different Nuisance Structures rather than the
# same one at different minutes. Every kind is already represented above, so a seed that
# draws nothing still produces a Corpus with the whole Nuisance Structure in it.
SPARES: tuple[NuisanceStructure, ...] = (
    NuisanceStructure(
        kind=NuisanceKind.SINGLE_ACCOUNT_DOMAIN,
        nuisance_id="syn-nuisance-single-crumbtheory",
        spread_hours=30 * 24,
        hosts=("crumbtheory.example",),
        posts=(
            SyntheticPost(
                account="syn_fenwick_moss_7714",
                subreddit="gardening",
                title="Lost a whole tray of seedlings to a watering schedule, and the fix",
                body=(
                    "Bottom watering, every time, and I still overdid it on the third tray "
                    "because the top of the compost looked dry and it was not. The fix that "
                    "worked was weighing the pot against a pot of the same size filled with "
                    "dry compost, which tells you the moisture without touching the surface. "
                    "Photographs of the before and after are here."
                ),
                links=("https://crumbtheory.example/hydration",),
            ),
        ),
        note=(
            "crumbtheory.example is one account's blog and nobody else links it. A "
            "component of one is not a Campaign Candidate, and the recovery figure is only "
            "meaningful if grouping has to resist this case as well as find the campaigns."
        ),
    ),
    _hard_negative(
        HardNegativeCharacter.GENUINE_JOB_POST,
        SyntheticPost(
            account="syn_ravelston_5590",
            subreddit="forhire",
            title="Two evening maths tutor posts at the college, £22 an hour, termly contract",
            body=(
                "Our evening programme is short two tutors for the spring term. Two hours per "
                "week per student, groups of four or fewer, twenty-two an hour, paid termly "
                "rather than monthly. You will need an enhanced check, we do that and pay for "
                "it, and we will cover the training the college requires. No fee for the "
                "training and nothing to pay for anything. The job ad says the same in fewer "
                "words than I have managed here."
            ),
            links=("https://ravelston-college.example/evening-tutors",),
        ),
        note=(
            "A real employer quoting a fee, a contract, and a qualification requirement — "
            "the vocabulary of the planted job scam with none of the scam's substance. It "
            "shares its own single-account domain and no other account's infrastructure."
        ),
    ),
    _hard_negative(
        HardNegativeCharacter.SATIRE,
        SyntheticPost(
            account="syn_mossgavel_2218",
            subreddit="RemoteWork",
            title=(
                "PSA: my council now offers 420 a day for six hours of training, kit "
                "provided, refundable deposit"
            ),
            body=(
                "The council has got in on remote work. 420 a day for six hours, kit "
                "provided, no experience needed, and a refundable deposit for the equipment "
                "which is refundable in the same sense that a hostage is released. Intake has "
                "to be completed within 48 hours or your place goes to somebody else, which "
                "is the part that convinced me it was real. Genuinely great work if you want "
                "it. They will call you, and you will enjoy the call."
            ),
            links=(),
        ),
        note=(
            "Satire reciting the planted job scam's figures — the same day rate, the same "
            "kit, the same refundable deposit, the same intake deadline — with no links at "
            "all. It is the hardest thing in the Corpus for a text-similarity step and the "
            "easiest for a reader, which is the whole point of a baseline."
        ),
    ),
)


def write_nuisance(path: Path, records: Iterable[NuisanceRecord]) -> None:
    """Write the Nuisance Structure manifest, sorted so a seed writes fixed bytes."""

    def objects() -> Iterable[JsonObject]:
        for record in sorted(records, key=lambda record: (record.kind, record.nuisance_id)):
            yield {
                "kind": record.kind,
                "nuisance_id": record.nuisance_id,
                "accounts": list(record.accounts),
                "posts": list(record.posts),
                "hosts": list(record.hosts),
                "characters": [character.value for character in record.characters],
                "note": record.note,
            }

    write_lines(path, objects())


def read_nuisance(path: Path) -> tuple[NuisanceRecord, ...]:
    """The manifest, read back and checked row by row.

    The evaluator reads this file to say what a recovery figure was measured
    against, so a row it cannot account for would be counted in a baseline nobody
    can check. Every field is therefore parsed rather than passed through, and the
    identifiers are held distinct: two rows claiming one identifier would inflate
    the counts by kind that the recovery report prints beside the figure, and the
    note is required rather than optional because a record nobody can account for
    is a record nobody can read either.
    """
    records = tuple(_record(path, number, text) for number, text in read_rows(path))
    identifiers = [record.nuisance_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        repeated = sorted(name for name in set(identifiers) if identifiers.count(name) > 1)
        raise ValueError(
            f"{path.as_posix()} names {repeated} on two rows each, so the counts beside "
            "the recovery figure would count them twice"
        )
    return records


def _record(path: Path, number: int, text: str) -> NuisanceRecord:
    where = f"{path.as_posix()}:{number}"
    try:
        row = json.loads(text)
    except json.JSONDecodeError as refusal:
        raise ValueError(f"{where} is not JSON: {text!r}") from refusal
    if not isinstance(row, dict):
        raise ValueError(f"{where} is not a row: {text!r}")

    vocabulary = tuple(field.name for field in fields(NuisanceRecord))
    if set(row) != set(vocabulary):
        raise ValueError(
            f"{where} holds {sorted(row)}, which is not the manifest vocabulary "
            f"{sorted(vocabulary)}"
        )
    note = row["note"]
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"{where} has no note, and a record nobody can account for is one")
    return NuisanceRecord(
        kind=_kind(where, row["kind"]),
        nuisance_id=_text(where, row, "nuisance_id"),
        accounts=_names(where, row, "accounts"),
        posts=_names(where, row, "posts"),
        hosts=_names(where, row, "hosts"),
        characters=_characters(where, row["characters"]),
        note=note,
    )


def _kind(where: str, value: object) -> NuisanceKind:
    if not isinstance(value, str):
        raise ValueError(f"{where} names {value!r}, which is not a kind of Nuisance Structure")
    try:
        return NuisanceKind(value)
    except ValueError as unknown:
        kinds = ", ".join(kind.value for kind in NuisanceKind)
        raise ValueError(f"{where} names {value!r}, and the kinds are {kinds}") from unknown


def _characters(where: str, value: object) -> tuple[HardNegativeCharacter, ...]:
    """The character each Hard Negative is recorded as, and nothing else.

    A record of another kind carries none, which the check enforces rather than
    assumes: a decoy labelled as a joke is a record that would be counted into the
    wrong baseline.
    """
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{where} has characters={value!r}, which is not a list of names")
    characters = []
    for entry in value:
        try:
            characters.append(HardNegativeCharacter(entry))
        except ValueError as unknown:
            names = ", ".join(character.value for character in HardNegativeCharacter)
            raise ValueError(f"{where} names {entry!r}, and the characters are {names}") from unknown
    return tuple(characters)


def _names(where: str, row: JsonObject, field_name: str) -> tuple[str, ...]:
    value = row[field_name]
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{where} has {field_name}={value!r}, which is not a list of names")
    return tuple(value)


def _text(where: str, row: JsonObject, field_name: str) -> str:
    value = row[field_name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} has no {field_name}")
    return value
