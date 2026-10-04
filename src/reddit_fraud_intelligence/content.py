"""The script the generator plants.

Every account name, domain, and post here is a Synthetic Entity: no real person,
no real account, and no real site. Account names carry the `syn_` prefix and
every domain sits under the reserved `.example` TLD (RFC 2606), so nothing in
the output can be mistaken for real data.

The text is written to be read. A reviewer who opens `corpus.jsonl` should be
able to judge whether the planted content is realistic without running
anything, and they should be able to see for themselves that two accounts in each
Planted Campaign link to the same registrable domain while nothing in the
file says they belong together.

Every post also declares the Contact Points it publishes, in `published`, which the
generator writes to a labelled set of its own. That is what makes the reading measurable:
`rfi contact-points` can be held to what the Corpus actually publishes rather than to
what a test expected of it, and a post that publishes a handle disguised is labelled as
publishing that handle. It sits here rather than in the file because a label and the
text it describes are one piece of writing; it reaches no pipeline, because `CorpusItem`
has no field for it and the Corpus file is a projection that takes items and nothing
else (ADR-0008).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlparse

from reddit_fraud_intelligence.contacts import ContactKind, PublishedContact, Writing


@dataclass(frozen=True, slots=True)
class SyntheticPost:
    account: str
    subreddit: str
    title: str
    body: str
    links: tuple[str, ...]
    published: tuple[PublishedContact, ...] = field(default_factory=tuple)


def telegram(
    value: str, written_as: str, *, writing: Writing = Writing.PLAIN
) -> PublishedContact:
    """A Telegram handle a post publishes, in the spelling it published it in.

    Two named constructors rather than a classmethod each, so a label in the text below
    reads as what it is — a handle, or an address, published plainly or written out — and
    a reader can see at a glance that the Corpus publishes Contact Points in three
    shapes, which is the range the reading is measured over. The three ways of writing
    are the same enum the labelled set carries, so a fixture and the row it becomes
    cannot drift apart.
    """
    return PublishedContact(
        kind=ContactKind.TELEGRAM,
        value=value,
        written_as=written_as,
        written=writing,
    )


def address(
    value: str, written_as: str, *, writing: Writing = Writing.PLAIN
) -> PublishedContact:
    """An email address a post publishes, in the spelling it published it in."""
    return PublishedContact(
        kind=ContactKind.EMAIL,
        value=value,
        written_as=written_as,
        written=writing,
    )


def picture(
    value: str, written_as: str, *, kind: ContactKind = ContactKind.TELEGRAM
) -> PublishedContact:
    """A Contact Point a post publishes only as a picture of one.

    Nothing in this project reads an image, so this label exists to be counted as a
    miss: the value is named, the link carrying the picture of it is named, and the
    report prints the pair so a reader can see which Contact Points this build is
    structurally blind to rather than inferring it from a count that might be a bug.
    """
    return PublishedContact(
        kind=kind, value=value, written_as=written_as, written=Writing.IMAGE
    )


def link_hosts(links: Iterable[str]) -> tuple[str, ...]:
    """The hosts a post links to, sorted, so two posts that link the same host compare
    equal however they spell the path."""
    hostnames = (urlparse(link).hostname for link in links)
    return tuple(sorted(host for host in hostnames if host is not None))


@dataclass(frozen=True, slots=True)
class PlantedScript:
    """A Planted Campaign's posts, and the cadence it runs on.

    `stagger_minutes` is how long it takes between one post and the next, so the
    campaign's accounts appear in order over hours rather than all at once.
    `paraphrase_note`, when set, says why the posts are recorded as one group rather
    than as unrelated posts.
    """

    campaign_id: str
    accounts: tuple[str, ...]
    posts: tuple[SyntheticPost, ...]
    stagger_minutes: int
    paraphrase_note: str | None = None


# A signals desk selling a paid crypto entry service. Four posts from three
# accounts, all four linking the same registrable domain: vantage-ledger.example.
# One of them goes through a different host of that domain, because real operators
# mirror their page and a future reader should have to reason about the registrable
# domain rather than the hostname.
#
# The four are staggered paraphrases of one offer: the same minimum ticket, the same
# compulsory call, the same published losing months, in four accounts' own words.
# Matching repeated strings finds nothing here, and text similarity is what finds all
# four — which is exactly the evidence ADR-0005 refuses to accept on its own.
#
# Three of the four posts name the desk's Telegram handle and one names an intake
# address, because a campaign that shares infrastructure shares the address people
# are told to reach it at as well. Not every post carries it: the one that does not
# is the point, because an extractor that is only ever right about the posts that
# advertise a Contact Point has not been measured on anything else.
_ALPHA = PlantedScript(
    campaign_id="syn-campaign-alpha",
    accounts=("syn_quantproof_2841", "syn_harborlight_5517", "syn_pinecrest_9032"),
    stagger_minutes=45,
    paraphrase_note=(
        "Four posts, three accounts, one offer, and no sentence written twice. Each one "
        "gives the minimum ticket as 500 USDT, insists the onboarding call is compulsory "
        "before any trade is shown, and reports that the desk publishes its losing months. "
        "The words move; the offer does not."
    ),
    posts=(
        SyntheticPost(
            account="syn_quantproof_2841",
            subreddit="CryptoCurrency",
            title="Eight months of DCA and I was wrong about the drawdowns",
            body=(
                "Ran a numbers-only DCA model against BTC and ETH for most of last year and "
                "every recovery I waited for turned into a deeper drawdown. Since February I "
                "have been taking entries from a signals desk that sizes positions off "
                "liquidity sweeps instead of moving averages, and the account has been flat "
                "since March. What sold me is that they publish the month's result whether it "
                "was green or not, including the two months it lost. Minimum ticket is 500 "
                "USDT and the onboarding call is compulsory, which they say is how they filter "
                "people. Posting because the drawdown thread here keeps asking what people are "
                "actually running. Happy to share the page in the comments, I am not going to "
                "put it in the post."
            ),
            links=("https://vantage-ledger.example/entry",),
        ),
        SyntheticPost(
            account="syn_harborlight_5517",
            subreddit="CryptoMarkets",
            title="Three months on a paid desk, and what it has cost me so far",
            body=(
                "Been paying for trade signals since February and putting my notes here so "
                "that somebody else does not have to pay to find that out. The desk runs "
                "entries off liquidity sweeps and volume confirmation rather than the usual "
                "pile of indicators bolted together, they will not discuss a position with "
                "you until you have been on a call, and nothing goes in below 500 USDT. The "
                "reports come out monthly either way, the months they lost included, "
                "annotated trade by trade. Three months in: one flat month, two green, and "
                "one call I would have preferred to skip. If you have already paid for one of "
                "the trading courses, read this before you pay for another. If you would rather "
                "look at one of their published months before paying anything, their Telegram "
                "is @syn_vantageledger."
            ),
            links=("https://mirror.vantage-ledger.example/month-log",),
            published=(telegram("syn_vantageledger", "@syn_vantageledger"),),
        ),
        SyntheticPost(
            account="syn_pinecrest_9032",
            subreddit="PersonalFinance",
            title="Where the cash I had set aside for a house deposit actually went",
            body=(
                "Not advice, and not a recommendation either. Part of the money I was holding "
                "back for a deposit is now with a paid signals desk that takes its entries "
                "from liquidity sweeps. Nothing goes under 500 USDT, they insist on an "
                "onboarding call before they will show you anything, and what talked me into "
                "trying it is that they write up the bad months as well as the good ones, in "
                "full, in the month they happen. Two months in, so read this as one person's "
                "early result rather than as a conclusion. Happy to answer questions about "
                "what the onboarding involves, the intake form takes a couple of minutes. They "
                "answer read-only questions on @Syn_VantageLedger if you would rather ask than "
                "pay first."
            ),
            links=("https://vantage-ledger.example/entry",),
            published=(telegram("syn_vantageledger", "@Syn_VantageLedger"),),
        ),
        SyntheticPost(
            account="syn_quantproof_2841",
            subreddit="CryptoCurrency",
            title="Desk update: month four, and the first red month",
            body=(
                "Month four on the desk and we took a red month, down six percent on the book, "
                "which they published the same afternoon with the losing trades annotated. Their "
                "whole pitch is that you see the bad months too. Reminder for anyone asking: the "
                "minimum ticket is 500 USDT, the onboarding call is compulsory, and they do not "
                "onboard without it. The full write-up of the month is on their page, drawdown "
                "included. Questions go to intake@vantage-ledger.example, which is the only "
                "address they answer, or @syn_vantageledger if Telegram is easier."
            ),
            links=("https://vantage-ledger.example/month-log",),
            published=(
                address("intake@vantage-ledger.example", "intake@vantage-ledger.example"),
                telegram("syn_vantageledger", "@syn_vantageledger"),
            ),
        ),
    ),
)

# A remote annotation job that pays, minus the deposit and the 48-hour intake.
# Three posts from two accounts, linking signal-harbor.example. Staggered, and not a
# paraphrase group: this campaign repeats one advert because two people are repeating
# it, which is a different case from three accounts rewording the same offer.
#
# Two of the three posts name the intake channel, from both of the campaign's accounts.
# A recruitment advert whose two posters put the same channel in front of a reader is
# the clearest case in the Corpus of a Contact Point that is worth reporting, because
# the handle is the only thing the two posts share that a registrable domain is not.
_BETA = PlantedScript(
    campaign_id="syn-campaign-beta",
    accounts=("syn_northwindhire_7736", "syn_clearpathwork_3184"),
    stagger_minutes=51,
    posts=(
        SyntheticPost(
            account="syn_northwindhire_7736",
            subreddit="forhire",
            title="Hiring: remote annotation, 420 a day, no experience needed, kit provided",
            body=(
                "My partner started this three weeks ago and the pay is real, 420 a day, six "
                "hours, five days a week, fully remote. No experience needed, they train you in "
                "the first two days. Equipment is provided but there is a refundable materials "
                "deposit for the workstation, and you have to finish the intake within 48 hours "
                "or the slot goes to someone else. I am not putting the link in the post because "
                "they took it down last time, message me and I will send it. Their intake "
                "channel is @syn_northwindhire if you would rather not wait for me."
            ),
            links=("https://signal-harbor.example/roles/annotation",),
            published=(
                telegram("syn_northwindhire", "@syn_northwindhire"),
            ),
        ),
        SyntheticPost(
            account="syn_clearpathwork_3184",
            subreddit="RemoteWork",
            title="data annotation role, 420 a day, equipment provided, no experience",
            body=(
                "Second person I know doing this. 420 a day, six hours, kit provided, they train "
                "you from scratch so there is no experience requirement. They do ask for a "
                "refundable deposit for the equipment before they ship it, which I will not "
                "pretend is normal, but it came back within a week for both of us. Intake has to "
                "be completed in 48 hours. Happy to answer questions here rather than in DMs. The "
                "intake form is on their Telegram, @syn_northwindhire, which is the only "
                "place it is."
            ),
            links=("https://signal-harbor.example/intake",),
            published=(
                telegram("syn_northwindhire", "@syn_northwindhire"),
            ),
        ),
        SyntheticPost(
            account="syn_northwindhire_7736",
            subreddit="RemoteWork",
            title="Second week on the annotation role, quick answers",
            body=(
                "Second week, first payout landed. Six hours a day at 420, equipment still not "
                "fully set up but the deposit came back on the second business day. The intake "
                "takes about an hour if you have your documents ready. Two slots open this week "
                "according to the intake page."
            ),
            links=("https://signal-harbor.example/roles/annotation",),
        ),
    ),
)

# An account that must not group with anything, because it links nothing at all. It is
# the floor of the case: there is no infrastructure here to join on, so any grouping
# that reaches it reached it on text alone.
_UNGROUPED = (
    SyntheticPost(
        account="syn_tideline_4471",
        subreddit="personalfinance",
        title="Is six months in a high-yield account still a normal emergency fund in this rate environment?",
        body=(
            "I have the emergency fund in a high-yield savings account and I keep seeing advice to "
            "put it somewhere with more return, which feels like advice for money you can lose. "
            "What are people actually holding, and has anyone moved theirs without regretting it?"
        ),
        links=(),
    ),
)

PLANTED: tuple[PlantedScript, ...] = (_ALPHA, _BETA)
UNGROUPED: tuple[SyntheticPost, ...] = _UNGROUPED
