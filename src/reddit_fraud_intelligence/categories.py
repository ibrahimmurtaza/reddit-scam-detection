"""Scam Categories: CAFC's thematic categories projected down to ten.

ADR-0006 chose the Canadian Anti-Fraud Centre because it is the one source of
real, analyst-reviewed fraud reports that is freely downloadable under a licence
permitting this use, and it committed to projecting CAFC's categories rather than
hand-picking a list of its own. This module is that projection: the ten Scam
Categories, the CAFC label each one lands, the Annex E heading CAFC defines it
under, and a one-line reason for that landing. Nothing here is derived from the
extract — the extract supplies the base rates the report adds — so the argument
and the figures cannot be confused for one another, and a reader who disagrees
with a merge is disagreeing with a judgement, not with arithmetic.

CAFC's labels are written once, here, and other modules read these constants rather
than repeating a label, so a test that greps `src/` for one finds this file and
no others. `rfi scam-categories` writes the mapping out as data and renders the
reader-facing page from it. The report quotes the reasons rather than restating
them, so the ten, the placements, and the reasons have one home.

The projection is a judgement and is written to be disagreed with. Three decisions
in it are worth stating up front because they are the ones a reader is most
likely to want to overturn:

- **A CAFC category is not a Scam Category.** CAFC enumerates a pitch a
  reporter can recognise in a form they were given to fill in; this project
  needs a class a pipeline can rank content under. Nothing is renamed for
  tidiness and nothing is invented: every CAFC label keeps CAFC's spelling.

- **A drop is a claim, so it carries a reason.** Three of CAFC's categories are
  not scams at all — two record that a reporter offered no classification, and
  one is a pitch CAFC's annex records as retired. Their reports are counted and
  reported rather than folded into a Scam Category, because a share of real
  reports that the projection cannot place is a finding about the projection.

- **Recovery Pitch is kept visible rather than merged away.** It is the one CAFC
  category that exists because a victim is contacted a second time, and it stands
  inside Relationships and Second Contacts with its own line, so that folding it
  into the con-based category nearest to it is a choice a reader can see.

One thing here is read by hand and is not machine-checked, and the report says so.
`ANNEX_CATEGORIES` is the set of headings CAFC's Annex E defines, transcribed from
a cited PDF; it supplies the *definitions* the rationales argue from, not the
enumeration. The enumeration the project actually consumes is the extract, and the
guarantee that a category added upstream cannot be silently ignored is a guarantee
against that file.
"""

from __future__ import annotations

from dataclasses import dataclass

from reddit_fraud_intelligence.cafc import (
    ATTRIBUTION,
    LICENCE,
    LICENCE_URL,
    RECORD_URL,
)

# CAFC's Annex E, which defines each thematic category in a sentence. Read by
# hand from the PDF rather than parsed, so it is a transcription and is named as
# one; the extract is the enumeration, and this is the vocabulary.
ANNEX_URL = (
    "https://open.canada.ca/data/dataset/6a09c998-cddb-4a22-beff-4dca67ab892f"
    "/resource/ec49223e-d746-4171-a97b-12fcc79d360f/download"
    "/annex-e_open-government-statistics-description-and-associated-definitions.pdf"
)

# The figure ADR-0006 asserted for the number of thematic categories. CAFC does
# not publish it. Named rather than quietly replaced so that a reader arriving
# from that ADR can see what was corrected and check the correction.
ASSERTED_IN_ADR_0006 = 41


@dataclass(frozen=True, slots=True)
class ScamCategory:
    """One top-level class, and the one line that says what it covers.

    `covers` is what a reader checks a candidate landing against, so it states
    the common thread of the CAFC categories underneath it rather than
    restating their names.
    """

    name: str
    covers: str


@dataclass(frozen=True, slots=True)
class CafcCategory:
    """One CAFC thematic category, and where the projection puts it.

    `category` is CAFC's own label, spelled the way CAFC's extract spells it.
    `annex` is the heading CAFC's Annex E defines for that label, which is
    `None` for the two labels the annex leaves undefined — the projection does
    not guess at a definition to fill the gap. Exactly one of `dropped` and
    `scam_category` is set, and `rationale` says which and why, in one line.
    """

    category: str
    annex: str | None
    scam_category: str | None
    dropped: bool
    rationale: str


SCAM_CATEGORIES: tuple[ScamCategory, ...] = (
    ScamCategory(
        name="Identity and Account Takeover",
        covers=(
            "Personal data or a device leaves the victim's control and is then used "
            "against them, which is the same move however it was obtained."
        ),
    ),
    ScamCategory(
        name="Merchandise and Goods",
        covers="A thing that was bought never arrives, or arrives fake.",
    ),
    ScamCategory(
        name="Phishing",
        covers=(
            "A message, call, or link engineered to make the victim hand something "
            "over, from the mass blast to the one sent to a named person."
        ),
    ),
    ScamCategory(
        name="Impersonating an Institution",
        covers=(
            "A bank, a carrier, or a supplier's agent speaking as itself and asking "
            "for money or access."
        ),
    ),
    ScamCategory(
        name="Investment and Money Offers",
        covers=(
            "Money against a promised return, an official-looking fund, or a sum "
            "waiting in another country."
        ),
    ),
    ScamCategory(
        name="Relationships and Second Contacts",
        covers=(
            "The fraudster borrowing a relationship the victim already has, and the "
            "follow-up approach made to a victim who has already lost money once."
        ),
    ),
    ScamCategory(
        name="Bills, Invoicing and Collections",
        covers=(
            "A demand for money for nothing supplied, or for a debt that is not real."
        ),
    ),
    ScamCategory(
        name="Work and Payroll",
        covers="A job that does not exist, and the cheque that arrives with it.",
    ),
    ScamCategory(
        name="Prizes, Appeals and Psychics",
        covers=(
            "Money sent in answer to a claim — a win, a gift of knowledge, a free "
            "stay, a cause — where nothing arrives that was promised."
        ),
    ),
    ScamCategory(
        name="Extortion",
        covers=(
            "Money, property, or a service taken by coercion, with nothing sold and "
            "nothing pretended."
        ),
    ),
)

# Not one of the ten: the bucket for content that fits no category. ADR-0006
# makes its size a standing measure of what the system fails to represent, so it
# is counted and reported rather than treated as a leftover.
OTHER = ScamCategory(
    name="Other",
    covers=(
        "Content that fits none of the ten. Its size is the finding: the share of "
        "real reports it would hold is how much the projection does not represent."
    ),
)

TOP_LEVEL_COUNT = len(SCAM_CATEGORIES)

# The ten and the bucket together: every Scam Category a CAFC label can land in.
# One expression so the renderer and the writer cannot enumerate them differently.
EVERY_CATEGORY: tuple[ScamCategory, ...] = SCAM_CATEGORIES + (OTHER,)

# The CAFC category the report singles out by name: the one that exists because a
# victim is contacted a second time. Named here so the renderer can point at it
# without CAFC's label being written into a second module.
RECOVERY_PITCH = "Recovery Pitch"

CAFC_CATEGORIES: tuple[CafcCategory, ...] = (
    CafcCategory(
        category="Identity Fraud",
        annex="Identity Theft and Fraud",
        scam_category="Identity and Account Takeover",
        dropped=False,
        rationale=(
            "The largest category in the extract, and the outcome every other theft "
            "of personal data is aiming at, so it belongs with the thefts of it."
        ),
    ),
    CafcCategory(
        category="Identity Theft",
        annex="Identity Theft and Fraud",
        scam_category="Identity and Account Takeover",
        dropped=False,
        rationale=(
            "CAFC's annex defines it beside Identity Fraud and the extract carries "
            "only that spelling, so it is placed on the annex's word that they are "
            "one pitch under two names."
        ),
    ),
    CafcCategory(
        category="Personal Info",
        annex="Personal Information",
        scam_category="Identity and Account Takeover",
        dropped=False,
        rationale=(
            "CAFC's own definition is the harvesting half of this category: the "
            "fraudster impersonates a body to take the data, then commits identity "
            "fraud with it."
        ),
    ),
    CafcCategory(
        category="Modem-Hijacking",
        annex="Modem-Hijacking",
        scam_category="Identity and Account Takeover",
        dropped=False,
        rationale=(
            "Remote access to the victim's own machine, which is the same theft of "
            "access as a stolen account and is described by CAFC in those terms."
        ),
    ),
    CafcCategory(
        category="Merchandise",
        annex="Merchandise",
        scam_category="Merchandise and Goods",
        dropped=False,
        rationale=(
            "The fake listing itself, separated from the counterfeits and the fake "
            "sellers that stand behind them, because it is the con the other two "
            "are variations of."
        ),
    ),
    CafcCategory(
        category="Counterfeit Merchandise",
        annex="Counterfeit Merchandise",
        scam_category="Merchandise and Goods",
        dropped=False,
        rationale=(
            "A knockoff sold from a lookalike site, which is a fake listing with a "
            "delivery attached rather than a different kind of con."
        ),
    ),
    CafcCategory(
        category="Vendor Fraud",
        annex="Vendor",
        scam_category="Merchandise and Goods",
        dropped=False,
        rationale=(
            "CAFC's annex calls Vendor a connected form of merchandise fraud, where "
            "the fake seller asks for the real thing to be posted to them."
        ),
    ),
    CafcCategory(
        category="Phishing",
        annex="Phishing",
        scam_category="Phishing",
        dropped=False,
        rationale=(
            "The mass form of the technique the other entries here are the "
            "targeted and the pretext cases of."
        ),
    ),
    CafcCategory(
        category="Spear Phishing",
        annex="Spear Phishing",
        scam_category="Phishing",
        dropped=False,
        rationale=(
            "The same technique sent to a named person rather than to a list, which "
            "changes the targeting and not what the victim is asked to do."
        ),
    ),
    CafcCategory(
        category="Spoofing",
        annex="Spoofing",
        scam_category="Phishing",
        dropped=False,
        rationale=(
            "CAFC's annex states outright that Spoofing is a subsection within spear "
            "phishing, so splitting it out would separate a technique from itself."
        ),
    ),
    CafcCategory(
        category="Survey",
        annex="Survey",
        scam_category="Phishing",
        dropped=False,
        rationale=(
            "A fraudulent survey is a pretext for collecting personal information, "
            "which is the whole of what phishing is in this projection."
        ),
    ),
    CafcCategory(
        category="Bank Investigator",
        annex="Bank Investigator Fraud",
        scam_category="Impersonating an Institution",
        dropped=False,
        rationale=(
            "A caller claiming to be the bank, or to be chasing the bank, which is "
            "borrowed authority rather than a message engineered to harvest data."
        ),
    ),
    CafcCategory(
        category="Service",
        annex="Service",
        scam_category="Impersonating an Institution",
        dropped=False,
        rationale=(
            "An unsolicited service that does not exist, offered by someone speaking "
            "for the body that would supply it."
        ),
    ),
    CafcCategory(
        category="Telecom Fraud",
        annex=None,
        scam_category="Impersonating an Institution",
        dropped=False,
        rationale=(
            "The extract carries the label and CAFC's annex defines none, so it is "
            "placed on the carrier being the institution impersonated and the gap is "
            "named rather than filled."
        ),
    ),
    CafcCategory(
        category="Investments",
        annex="Investment Fraud",
        scam_category="Investment and Money Offers",
        dropped=False,
        rationale="Money against a promised return, which is the plain case the others vary.",
    ),
    CafcCategory(
        category="Foreign Money Offer",
        annex="Foreign Money Offer",
        scam_category="Investment and Money Offers",
        dropped=False,
        rationale=(
            "A sum in another country needing a fee and an accomplice, which CAFC's "
            "own definition classes with inheritance and business-proposal scams."
        ),
    ),
    CafcCategory(
        category="Pyramid",
        annex="Pyramid",
        scam_category="Investment and Money Offers",
        dropped=False,
        rationale=(
            "Recruitment rather than a product is what makes it a pyramid, and the "
            "return it offers is the same one being sold."
        ),
    ),
    CafcCategory(
        category="GRANT",
        annex="Grant and Loan",
        scam_category="Investment and Money Offers",
        dropped=False,
        rationale=(
            "CAFC's annex defines Grant and Loan as one pitch — government-looking "
            "sites attracting people looking for funds — so the two are not split."
        ),
    ),
    CafcCategory(
        category="Loan",
        annex="Grant and Loan",
        scam_category="Investment and Money Offers",
        dropped=False,
        rationale=(
            "CAFC's annex defines Grant and Loan as one pitch, and a loan is money "
            "promised against terms rather than bought."
        ),
    ),
    CafcCategory(
        category="Romance",
        annex="Romance",
        scam_category="Relationships and Second Contacts",
        dropped=False,
        rationale=(
            "Trust built over time before any money is asked for, which is the same "
            "borrow-a-relationship move as the emergency pitch at a different speed."
        ),
    ),
    CafcCategory(
        category="Emergency (Jail, Accident, Hospital, Help)",
        annex="Emergency Fraud",
        scam_category="Relationships and Second Contacts",
        dropped=False,
        rationale=(
            "A person the victim knows asking for money at once, which is the same "
            "borrowed relationship as Romance compressed into a single call."
        ),
    ),
    CafcCategory(
        category=RECOVERY_PITCH,
        annex="Recovery Pitch",
        scam_category="Relationships and Second Contacts",
        dropped=False,
        rationale=(
            "Kept visible rather than merged away: CAFC documents it because a "
            "victim is contacted a second time, and dropping it would lose the one "
            "category that records that the second contact is itself the pattern."
        ),
    ),
    CafcCategory(
        category="False Billing",
        annex="False Billing",
        scam_category="Bills, Invoicing and Collections",
        dropped=False,
        rationale=(
            "An invoice for nothing requested, which is the plain case this class "
            "is named for."
        ),
    ),
    CafcCategory(
        category="Unauthorized Charge",
        annex="Unauthorized Charge",
        scam_category="Bills, Invoicing and Collections",
        dropped=False,
        rationale=(
            "CAFC's definition is a charge with no product or service in exchange, "
            "which is the definition of the category it is placed in."
        ),
    ),
    CafcCategory(
        category="Collection Agency",
        annex="Collection Agency Fraud",
        scam_category="Bills, Invoicing and Collections",
        dropped=False,
        rationale=(
            "A demand for a debt that is not real, which is an invoice for nothing "
            "supplied with a story attached to it."
        ),
    ),
    CafcCategory(
        category="Directory",
        annex="Directory Fraud (Subsection of False Billing Fraud)",
        scam_category="Bills, Invoicing and Collections",
        dropped=False,
        rationale=(
            "CAFC's annex says this is a subsection of false billing, and the con is "
            "an invoice for a listing nobody ordered."
        ),
    ),
    CafcCategory(
        category="Office Supplies",
        annex="Office Supplies",
        scam_category="Bills, Invoicing and Collections",
        dropped=False,
        rationale=(
            "CAFC's annex calls it related to false billing: goods sent unasked and "
            "then invoiced."
        ),
    ),
    CafcCategory(
        category="Credit Card",
        annex=None,
        scam_category="Bills, Invoicing and Collections",
        dropped=False,
        rationale=(
            "The extract carries the label and CAFC's annex defines none, so it is "
            "placed as a charge against a card for nothing supplied and the gap in "
            "the annex is named rather than filled."
        ),
    ),
    CafcCategory(
        category="Job",
        annex="Job",
        scam_category="Work and Payroll",
        dropped=False,
        rationale=(
            "A job that does not exist, kept as a class of its own because the victim "
            "is between posts rather than shopping, investing, or being phished."
        ),
    ),
    CafcCategory(
        category="Fraudulent Cheque",
        annex="Fraudulent Cheque",
        scam_category="Work and Payroll",
        dropped=False,
        rationale=(
            "CAFC's annex notes the Job category is frequently connected to fraudulent "
            "cheque fraud, and the cheque is the second half of that one con."
        ),
    ),
    CafcCategory(
        category="Prize",
        annex="Prize",
        scam_category="Prizes, Appeals and Psychics",
        dropped=False,
        rationale=(
            "A win that costs an advance fee to claim, which is the plain case this "
            "class is named for."
        ),
    ),
    CafcCategory(
        category="Psychics",
        annex="Psychics",
        scam_category="Prizes, Appeals and Psychics",
        dropped=False,
        rationale=(
            "A claimed gift of sight or knowledge sold by mail or online, which is the "
            "prize con without the lottery."
        ),
    ),
    CafcCategory(
        category="Vacation",
        annex="Vacation",
        scam_category="Prizes, Appeals and Psychics",
        dropped=False,
        rationale=(
            "An automated call about a free trip followed by a deposit, which is the "
            "prize con with a holiday as the prize."
        ),
    ),
    CafcCategory(
        category="Timeshare",
        annex="Timeshare",
        scam_category="Prizes, Appeals and Psychics",
        dropped=False,
        rationale=(
            "Free stays offered for a presentation, and the fees that follow, which "
            "is the same gift-for-attendance shape as Vacation."
        ),
    ),
    CafcCategory(
        category="Charity / Donation",
        annex="Charity/Donation Fraud",
        scam_category="Prizes, Appeals and Psychics",
        dropped=False,
        rationale=(
            "The merge a reader should contest first: a donation pitch is money sent "
            "in answer to a claim, which is the prize con inverted, and well under a "
            "tenth of a percent of reports is too small to carry a class of its own."
        ),
    ),
    CafcCategory(
        category="Extortion",
        annex="Extortion",
        scam_category="Extortion",
        dropped=False,
        rationale=(
            "Left as a class of its own because nothing is pretended and nothing is "
            "sold: it is the one CAFC category with no con in it, and merging it "
            "would hide how much of what gets reported is plain coercion."
        ),
    ),
    CafcCategory(
        category="Other",
        annex="Other",
        scam_category=OTHER.name,
        dropped=False,
        rationale=(
            "CAFC files it for a report its own analysts could not describe any other "
            "way, which is exactly the content this projection cannot place either."
        ),
    ),
    CafcCategory(
        category="Unknown",
        annex="Incomplete (Report) / Unknown",
        scam_category=None,
        dropped=True,
        rationale=(
            "Dropped because it records that the reporter offered no classification, "
            "not a kind of fraud; CAFC's annex pairs it with Incomplete for that "
            "reason, and its reports are counted as unplaced rather than absorbed."
        ),
    ),
    CafcCategory(
        category="Incomplete",
        annex="Incomplete (Report) / Unknown",
        scam_category=None,
        dropped=True,
        rationale=(
            "Dropped because CAFC's annex records it as a report filed without enough "
            "detail to classify, which is an intake gap rather than a fraud kind, and "
            "its reports are counted as unplaced rather than absorbed."
        ),
    ),
    CafcCategory(
        category="Health",
        annex="Health",
        scam_category=None,
        dropped=True,
        rationale=(
            "Dropped because CAFC's own annex calls it a now-disused pitch that has "
            "been divided into more appropriate ones, so it describes nothing a "
            "reporter could still have been pitched."
        ),
    ),
)

# The headings CAFC's Annex E defines. The projection names each one from at
# least one entry, which is what makes the count of published names checkable
# rather than a number typed in: 35 headings, plus the two labels the extract
# carries and the annex leaves undefined.
ANNEX_CATEGORIES: frozenset[str] = frozenset(
    {category.annex for category in CAFC_CATEGORIES if category.annex is not None}
)
