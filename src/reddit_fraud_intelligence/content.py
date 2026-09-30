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
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyntheticPost:
    account: str
    subreddit: str
    title: str
    body: str
    links: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlantedScript:
    campaign_id: str
    accounts: tuple[str, ...]
    posts: tuple[SyntheticPost, ...]
    spread_hours: int


# A signals desk selling a paid crypto entry service. Four posts from three
# accounts, all four linking the same registrable domain: vantage-ledger.example.
# One of them goes through a different host of that domain, because real operators
# mirror their page and a future reader should have to reason about the registrable
# domain rather than the hostname.
_ALPHA = PlantedScript(
    campaign_id="syn-campaign-alpha",
    accounts=("syn_quantproof_2841", "syn_harborlight_5517", "syn_pinecrest_9032"),
    spread_hours=5,
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
            title="Signals desk since February, month three in",
            body=(
                "Three months on a paid desk now. The pitch is liquidity sweeps and volume "
                "confirmation, not a pile of tools bolted together, and they want you on a call "
                "before they show you a single trade. The ticket starts at 500 USDT and they will "
                "not do smaller. Reports go out monthly whether the month was green or red, which "
                "is the only reason I stayed after the first drawdown. If you have been burned by "
                "the paid trading courses this is worth reading before you spend another month on "
                "one of those."
            ),
            links=("https://mirror.vantage-ledger.example/month-log",),
        ),
        SyntheticPost(
            account="syn_pinecrest_9032",
            subreddit="PersonalFinance",
            title="Where I actually put the money I had earmarked for a house deposit",
            body=(
                "Not advice and not a recommendation, just reporting. I moved part of the cash I "
                "was holding for a house deposit into a signals desk that runs entries off "
                "liquidity sweeps, minimum ticket 500 USDT, compulsory onboarding call before "
                "you see a trade. Three months in, one flat month and two green. The part that "
                "convinced me is that the desk publishes the losing months as well, in full. "
                "Happy to answer questions about how the onboarding works, the intake form is "
                "quick."
            ),
            links=("https://vantage-ledger.example/entry",),
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
                "included."
            ),
            links=("https://vantage-ledger.example/month-log",),
        ),
    ),
)

# A remote annotation job that pays, minus the deposit and the 48-hour intake.
# Three posts from two accounts, linking signal-harbor.example.
_BETA = PlantedScript(
    campaign_id="syn-campaign-beta",
    accounts=("syn_northwindhire_7736", "syn_clearpathwork_3184"),
    spread_hours=3,
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
                "they took it down last time, message me and I will send it."
            ),
            links=("https://signal-harbor.example/roles/annotation",),
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
                "be completed in 48 hours. Happy to answer questions here rather than in DMs."
            ),
            links=("https://signal-harbor.example/intake",),
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

# Accounts that must not group with anything. One posts in a subreddit where a
# link is unremarkable and points at a domain no one else uses.
_UNGROUPED = (
    SyntheticPost(
        account="syn_marrowgate_6205",
        subreddit="AskCooking",
        title="Sous vide timing for a 3kg ribeye, finally sorted",
        body=(
            "Six attempts and I have it down to 54 minutes at 57C with a 20 minute sear, which "
            "gives me medium rare all the way through without the grey band near the bone. The "
            "fridge rest is doing more work than the bath temperature, which is the thing nobody "
            "tells you. Timings and my notes here."
        ),
        links=("https://single-run.example/ribeye-timing",),
    ),
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
