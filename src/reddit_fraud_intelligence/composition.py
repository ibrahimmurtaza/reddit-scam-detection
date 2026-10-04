"""The Corpus's own Scam Category distribution, beside CAFC's published base rates.

ADR-0006 asked for the two to be shown together and called the result the thing that
turns a generated Corpus into evidence: a distortion in what this project planted is
invisible until it is set beside the rate at which the same kind of fraud is actually
reported. So the unit of this module is a comparison — one Scam Category, the share of
the Corpus landing in it, CAFC's share of the same class, and the difference between
the two — and every view here prints all four on one line.

A post is placed by reading its own title and body against a published phrase list per
Scam Category, exactly as a Content Signal fires (ADR-0015). That rule is ADR-0017's and
it lives in `placement.py`, because `rfi policy-score` places every post too, and two
commands placing the same post two ways would leave two published figures that could not
be compared with each other. What this module adds is the comparison.

The placement is a claim about what a post is talking about and nothing else. The lists
are small and blunt, they are written to be argued with, and the classifier that will
eventually replace them (ticket #25) is measured by something this command cannot
supply: CAFC's extract has no free-text field, so it constrains which Scam Categories
exist and what their priors are and it cannot check one placement made from a post's
own words. The report says so in those terms rather than leaving the reader to work out
why a base rate is being used the way it is.

Nothing here reads the truth file or the Nuisance Structure manifest. The Corpus file
and the committed base rates are the whole input, and reading the membership would make
this a statement about the generator rather than about the Corpus a reviewer can read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reddit_fraud_intelligence.cafc import (
    ATTRIBUTION,
    LICENCE,
    LICENCE_URL,
    RECORD_URL,
)
from reddit_fraud_intelligence.categories import (
    EVERY_CATEGORY,
    OTHER,
    SCAM_CATEGORIES,
    TOP_LEVEL_COUNT,
    ScamCategory,
)
from reddit_fraud_intelligence.corpus import CorpusItem, read_corpus
from reddit_fraud_intelligence.jsonl import JsonObject, write_lines
from reddit_fraud_intelligence.placement import (
    CATEGORY_PHRASES,
    Placement,
    Placed,
    check_placements,
    phrase_count,
    place,
    render_phrases,
)
from reddit_fraud_intelligence.projection import Placed as CafcPlaced
from reddit_fraud_intelligence.projection import place as land_cafc_categories
from reddit_fraud_intelligence.projection import read_base_rates, total
from reddit_fraud_intelligence.text import wrap

_HEADING = "Corpus composition"
_SUBHEADING = """\
Two distributions, one table: the share of the Corpus landing in each Scam Category, \
and CAFC's published base rate for the same class. The gap is the finding."""

# The heading above the phrase lists, in four lines because the rule has three parts a
# reader has to be told apart: what places a post, that the lists are tried in the order
# printed and the first one to match wins, and that a post none of them matches lands in
# Other rather than in no class at all. `placement.py` prints the lists under it, because
# two commands now publish this rule and a second copy of the lists would be a second
# thing to keep in step with the first.
_PHRASES_HEADING = (
    f"phrases  what places a post in each of the {TOP_LEVEL_COUNT} Scam Categories, "
    "and nothing outside",
    "these strings places one. They are tried in the order below and the first to "
    "match takes the post;",
    f"a post no list matches lands in {OTHER.name}. Every match is recorded beside "
    "the class it would have",
    "placed, and the sentence is quoted whole",
)

# One cell of a bar is this share of the widest bar in the table, so one scale covers
# every bar printed. Wide enough to see a 4% base rate against a 44% one and narrow
# enough to keep the two bars of a row on one line of an eighty-column console.
_BAR_WIDTH = 30

# Where the placements are published, as the command writes them by default. Named here
# because the report points at the file a reader should open beside it, and a report that
# printed the path of this particular run would produce different bytes for the same
# Corpus depending only on the directory it was written into. `cli.py` holds the default
# the command actually uses; this is the name the report gives it, which is the same thing
# `projection.py` does with the mapping beside `docs/scam-categories.md`.
PLACEMENTS_PATH = "data/corpus/composition.jsonl"
@dataclass(frozen=True, slots=True)
class Share:
    """One figure as a share of a whole, held as the two integers it is.

    Every percentage here is `count` out of `of`, and the arithmetic is integer
    arithmetic so a reader can follow it and so a rounding rule cannot depend on the
    interpreter. Two shares of two different wholes sit side by side here without being
    made comparable by anything: the Corpus counts posts and CAFC counts reports, and
    the difference between the two is the finding.
    """

    count: int
    of: int

    @property
    def per_mille(self) -> int:
        """The share in tenths of a percent, rounded half up."""
        return (1000 * self.count + self.of // 2) // self.of

    def percent(self) -> str:
        tenths = self.per_mille
        return f"{tenths // 10}.{tenths % 10}%"

    def bar(self, scale: int) -> str:
        """The share as a bar on a scale fixed by the widest share in the table."""
        return "#" * ((2 * self.per_mille * _BAR_WIDTH + scale) // (2 * scale))


@dataclass(frozen=True, slots=True)
class Row:
    """One Scam Category, the Corpus's share of it, and CAFC's share of it."""

    scam_category: ScamCategory
    corpus: Share
    cafc: Share

    @property
    def name(self) -> str:
        return self.scam_category.name

    @property
    def posts(self) -> int:
        return self.corpus.count

    def difference(self) -> str:
        """The Corpus's share minus CAFC's, in percentage points, signed."""
        tenths = self.corpus.per_mille - self.cafc.per_mille
        points = abs(tenths)
        sign = "-" if tenths < 0 else "+"
        return f"{sign}{points // 10}.{points % 10}pp"


@dataclass(frozen=True, slots=True)
class CompositionFacts:
    """What was read, and what came of it, for the figures the outputs print."""

    accounts: int
    base_rate_categories: int
    base_rates_path: str
    corpus_path: str
    corpus_sha256: str
    posts: int
    reports: int


@dataclass(frozen=True, slots=True)
class Composition:
    """Every post's Scam Category, and CAFC's figures to hold them against."""

    facts: CompositionFacts
    posts: tuple[Placed, ...]
    placed: tuple[CafcPlaced, ...]

    def posts_in(self, name: str) -> int:
        return sum(1 for post in self.posts if post.scam_category == name)

    def comparison(self) -> tuple[Row, ...]:
        """One row per Scam Category and per the bucket, widest Corpus share first.

        Ordered by the Corpus rather than by CAFC so the largest distortion in a
        generated Corpus is the first line of the output, and ties go by name so a run
        over the same Corpus prints the same table.
        """
        reports = self.facts.reports
        rows = [
            Row(
                scam_category=scam_category,
                corpus=Share(count=self.posts_in(scam_category.name), of=self.facts.posts),
                cafc=Share(count=total(self.placed, scam_category.name), of=reports),
            )
            for scam_category in EVERY_CATEGORY
        ]
        return tuple(sorted(rows, key=lambda row: (-row.corpus.per_mille, row.name)))

    def scale(self, rows: Sequence[Row]) -> int:
        """The widest share anywhere in the table, which fixes the scale of every bar."""
        return max(max(row.corpus.per_mille, row.cafc.per_mille) for row in rows)

    def uncovered(self, rows: Sequence[Row]) -> tuple[Row, ...]:
        """The classes this Corpus holds no post of at all.

        Named from one place so the console and the report cannot count them differently:
        two views of the same run disagreeing about how many classes are empty is the
        kind of drift the Other bucket exists to prevent.
        """
        return tuple(row for row in rows if row.posts == 0)


def dropped_reports(placed: Sequence[CafcPlaced]) -> int:
    """The reports landing in no Scam Category because their category was dropped.

    They are in the denominator of every CAFC share rather than folded into a class that
    does not describe them, so the report has to name the count beside the figures or the
    CAFC column reads as a share of the placed reports alone.
    """
    return sum(item.reports or 0 for item in placed if item.category.dropped)


def compose(corpus_path: Path, base_rates_path: Path) -> Composition:
    """Read the Corpus and the committed base rates, and place every post.

    The two refusals are the ones that matter. A base rate the projection does not
    account for stops the run, because a table that quietly omits a class looks like the
    whole of the comparison; and a Scam Category with no phrase list stops it too,
    because a class that no post can be placed in would print a row of zeroes beside a
    base rate and read as coverage rather than as an absent rule.
    """
    check_placements()
    items = read_corpus(corpus_path)
    posts = tuple(place(item) for item in items)
    base_rates = read_base_rates(base_rates_path)
    landed = land_cafc_categories(base_rates)

    return Composition(
        facts=CompositionFacts(
            accounts=len({item.account for item in items}),
            base_rate_categories=len(base_rates),
            base_rates_path=base_rates_path.as_posix(),
            corpus_path=corpus_path.as_posix(),
            corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            posts=len(posts),
            reports=sum(rate.reports for rate in base_rates.values()),
        ),
        posts=posts,
        placed=landed,
    )


def write_composition(path: Path, posts: Sequence[Placed]) -> None:
    """Every post's Scam Category, with the evidence that placed it.

    In the Corpus's own order rather than the table's, so a row here is a post and the
    two files can be read side by side. A post carries the sentences that placed it
    rather than a single phrase, because a phrase is the smallest thing a reviewer would
    want to argue with and the sentence is the largest thing they would find themselves
    reading.
    """

    def objects() -> Iterator[JsonObject]:
        for post in posts:
            yield {
                "post_id": post.post_id,
                "account": post.account,
                "scam_category": post.scam_category,
                "evidence": [
                    {
                        "field": placement.field,
                        "phrases": list(placement.phrases),
                        "sentence": placement.sentence,
                    }
                    for placement in post.evidence
                ],
                "also_matched": list(post.also_matched),
            }

    write_lines(path, objects())


def render_table(composition: Composition) -> str:
    """The console output: what was read, the comparison, the rules, and the limits.

    ASCII only, so it prints the same way on a console that cannot encode anything else
    and the same way when it is redirected, which is what lets it be pasted into an
    issue or diffed between runs.

    Each category takes three lines: the figures, then a bar for the Corpus beside a bar
    for CAFC on one scale. The figures are the claim and the bars are how a reader sees
    it without reading, which is the criterion this command exists for.
    """
    rows = composition.comparison()
    scale = composition.scale(rows)
    sections = (
        f"{_HEADING}\n\n{_SUBHEADING}",
        _figures(composition),
        _table(rows, scale),
        render_phrases(_PHRASES_HEADING),
        _footer(composition, rows),
    )
    return "\n\n".join(section for section in sections if section) + "\n"


def _figures(composition: Composition) -> str:
    """What was read, and what came of it. One figure per line, labelled."""
    facts = composition.facts
    fields_out = (
        ("corpus", f"{facts.corpus_path} ({facts.posts} posts, {facts.accounts} accounts)"),
        ("sha256", facts.corpus_sha256),
        (
            "base rates",
            f"{facts.base_rates_path} ({facts.base_rate_categories} categories, "
            f"{facts.reports:,} reports)",
        ),
        (
            "placed",
            f"{len(composition.posts)} posts, "
            f"{sum(len(post.evidence) for post in composition.posts)} sentences of evidence",
        ),
    )
    width = max(len(name) for name, _ in fields_out)
    return "\n".join(f"  {name.ljust(width)}  {value}" for name, value in fields_out)


def _table(rows: Sequence[Row], scale: int) -> str:
    """One Scam Category at a time: the four figures, then the two bars on one scale."""
    name = max(len(row.name) for row in rows)
    postings = max(len(str(row.posts)) for row in rows)
    corpus = max(len(row.corpus.percent()) for row in rows)
    cafc = max(len(row.cafc.percent()) for row in rows)
    gap = max(len(row.difference()) for row in rows)

    lines = [
        f"  {'category'.ljust(name)}  {'posts'.rjust(postings)}  "
        f"{'corpus'.rjust(corpus)}  {'CAFC'.rjust(cafc)}  {'difference'.rjust(gap)}"
    ]
    for row in rows:
        lines.append(
            f"  {row.name.ljust(name)}  {str(row.posts).rjust(postings)}  "
            f"{row.corpus.percent().rjust(corpus)}  {row.cafc.percent().rjust(cafc)}  "
            f"{row.difference().rjust(gap)}"
        )
        lines.append(f"    corpus  {row.corpus.bar(scale).ljust(_BAR_WIDTH)}  {row.corpus.percent()}")
        lines.append(f"    CAFC    {row.cafc.bar(scale).ljust(_BAR_WIDTH)}  {row.cafc.percent()}")
    return "\n".join(lines)


def _footer(composition: Composition, rows: Sequence[Row]) -> str:
    """What the comparison is, what it is not, and what it cannot be checked against.

    The limits are the ones a reader would otherwise have to guess at: the two columns
    are shares of two different wholes, a placement says what a post is talking about
    and not whether it is a scam, and CAFC cannot validate the reading because it has no
    free-text field to validate it with.
    """
    facts = composition.facts
    widest = _widest(rows)
    uncovered = composition.uncovered(rows)
    dropped = dropped_reports(composition.placed)
    return f"""\
The two columns are shares of two different wholes. The Corpus column counts the \
{facts.posts} posts this project wrote, and the CAFC column counts {facts.reports:,} \
reports Canada filed, over the same window, out of which the projection places \
{facts.reports - dropped:,} and drops the rest. Neither column is normalised to the \
other and no figure here is a rate of anything: read the difference.

The widest gap is {widest.name}, at {widest.corpus.percent()} of the Corpus against \
{widest.cafc.percent()} of real reports, a difference of {widest.difference()}. \
{len(uncovered)} of the ten Scam Categories hold no post of this Corpus at all, and \
CAFC's reports do not arrive evenly spread across the ten: the generator writes the \
pitches it writes and the rest are simply absent.

A placement says what a post is talking about, not that it is a scam. The lists are \
small and blunt, they are printed above so they can be argued with, and a post that is \
satire about a pitch, or complains about having lost money to one, is placed by that \
pitch - the same class a genuine post of that pitch lands in. The Other bucket is where \
a post no list matched lands, and its size is what the generator did not write, what the \
projection does not cover, and what these particular lists cannot place.

CAFC's extract has no free-text field: no column of it can hold a sentence. It \
constrains which Scam Categories exist and what their priors are, and it cannot \
validate a text classifier - there is nothing in it that says what category a piece of \
text belongs to. So nothing here is checked against CAFC beyond the prior beside it, \
and the classifier that eventually replaces these lists is measured by something else."""


def _widest(rows: Sequence[Row]) -> Row:
    """The class where the Corpus and CAFC disagree most, in percentage points."""
    return max(rows, key=lambda row: abs(row.corpus.per_mille - row.cafc.per_mille))


def render_report(composition: Composition) -> str:
    """The reader-facing page, generated from the Corpus and the committed figures.

    Written for a reader who has to see the distortion rather than infer it: the widest
    gap is named with both figures, the classes the Corpus holds nothing of are named
    with the share of real reports that lands in them, and the Other bucket has a
    section of its own rather than being the last row of a table. Every figure in the
    prose is computed from the Corpus and the base rates — nothing about which posts
    landed where is written by hand, because the page is generated and a sentence that
    was true of one Corpus would quietly be untrue of the next. The Corpus is named by
    the path this run read it from, and the placements by the path the command writes by
    default, so two runs over the same Corpus produce the same bytes whichever directory
    they write to.
    """
    rows = composition.comparison()
    scale = composition.scale(rows)
    facts = composition.facts
    name = max(len(row.name) for row in rows)
    widest = _widest(rows)
    other = next(row for row in rows if row.name == OTHER.name)
    uncovered = composition.uncovered(rows)
    unseen = sum(row.cafc.count for row in uncovered)
    dropped = dropped_reports(composition.placed)
    in_other = tuple(
        sorted(post.post_id for post in composition.posts if post.scam_category == OTHER.name)
    )
    table = "\n".join(
        f"| {row.name:<{name}} | {row.posts:>5} | {row.corpus.percent():>8} | "
        f"{row.cafc.percent():>8} | {row.difference():>9} |"
        for row in rows
    )
    chart = "\n".join(_chart_row(row, scale, name) for row in rows)
    places = "\n\n".join(_section(row) for row in rows)

    return f"""# Corpus composition against CAFC base rates

Generated by `rfi corpus-composition` from the committed Corpus and the committed CAFC
base rates. Do not edit it by hand — a test holds this file to what the command
produces, and re-running the command rewrites it byte for byte.

## What this is

Two distributions, one table. The Corpus column is the share of the
{facts.posts} posts in `{facts.corpus_path}` that lands in each Scam Category, and
the CAFC column is the base rate of the same class over the {facts.reports:,} reports in
CAFC's extract. The two are shares of two different wholes — posts this project wrote
against reports Canada filed — so neither is normalised to the other and the level of
either column means little on its own. **The difference is the finding**, and it is
printed for every row rather than left for the reader to work out.

ADR-0006 asked for exactly this, and the reason it matters is that a generated Corpus
distorts silently: the generator writes the pitches it writes, and nothing about the
output would say so unless the output is set against the rate at which each kind of
fraud is actually reported.

## The two distributions

The Other bucket is a row beside the ten rather than a remainder below them, because its
size is a finding about what neither the generator nor the projection covered, and a
remainder is read as rounding.

| Scam Category | Posts | Corpus | CAFC | Difference |
| --- | ---: | ---: | ---: | ---: |
{table}

## The same two distributions as bars

One scale for every bar in the page, so the two bars in a row are comparable with each
other and a bar in one row is comparable with a bar in another:

```
{chart}
```

## The widest gap

**{widest.name} holds {widest.posts} of {facts.posts} posts, {widest.corpus.percent()},
against {widest.cafc.percent()} of real reports — a difference of {widest.difference()}.**
A Corpus that is heavy in a class the world is not heavy in tells you about the Corpus
and nothing about fraud. The direction matters as much as the size: a class the Corpus
holds more of than the world sees is the generator's habit, and a class it holds less of
is a class it never wrote.

## What the Corpus holds nothing of

{len(uncovered)} of the ten Scam Categories hold no post of this Corpus at all:
{", ".join(row.name for row in uncovered)}. CAFC's reports do not arrive evenly spread
across the ten, so this is where the distortion costs the most. **{_percent(unseen, facts.reports)}
of real reports land in classes this Corpus has nothing to say about**, which bounds
every figure measured against it: a recovery rate here is a recovery rate for the pitches
this Corpus was written to hold.

## The Other bucket

**Other holds {other.posts} of {facts.posts} posts, {other.corpus.percent()}, against
CAFC's own residual category at {other.cafc.percent()}.** CAFC filed that one for a report
its analysts could not describe any other way, which is the position this projection is
in when no list matches a post. The two are not the same population — one is a share of
posts this project wrote, the other a share of reports Canada filed — so they are worth
reading against each other and not equated: a large bucket here says the generator and
these lists left something out, and a large bucket there says CAFC's analysts did.

A post lands in Other by carrying no phrase from any of the ten lists, which means the
lists say nothing about it rather than that it is not fraud. The {other.posts} posts are
{", ".join(in_other)}, and every one of them is in `{PLACEMENTS_PATH}` with an empty
`evidence` list, which is the file's way of saying a list placed them nowhere. The bucket
is measured rather than tuned away: a Corpus whose every post is a pitch is a Corpus that
cannot be measured against anything.

## How a post is placed

A post is placed by reading its own title and body against a published phrase list per
Scam Category, the same way a Content Signal fires (ADR-0015). Nothing outside those
strings places a post, the lists are tried in the order below and the first to match
takes the post, and every other class that matched the same post is recorded in
`{PLACEMENTS_PATH}` beside it. The evidence is the sentence the phrase was
found in, quoted whole, so a reader can find it in the post rather than take the row's
word for it.

{places}

A post is placed by no other means than these lists, and the whole of every list is
printed above and in the console output. A phrase list is a rule rather than a number,
so publishing it is the only way a reader gets to see what the rule was looking for; a
list you cannot see is a figure you cannot check.

## What this cannot do

**CAFC's extract has no free-text field.** No column of it can hold a sentence, so it
constrains which Scam Categories exist and what their priors are, and it cannot validate
a text classifier — there is nothing in it that says what category a piece of text
belongs to. Every placement in `{PLACEMENTS_PATH}` is therefore unvalidated:
it is checked against a phrase list and against nothing else.

Three further limits, all of them consequences of reading a post literally:

- **A placement is about what a post is talking about, not about whether it is a scam.**
  A satire post about an investment pitch, a complaint about having lost money to one,
  and the pitch itself all land in the same class, and so do a genuine advert for work
  and a planted one. The lists have no second signal to tell those apart and are not
  trying to.
- **The lists are short, and shortness is a cost.** {phrase_count()} phrases across the
  ten classes is enough to place the Corpus this project wrote and blunt enough to be
  wrong often on real content, where a pitch in a language this build does not read
  places in Other.
- **Nothing about accounts is read.** No age, karma, posting rate, or activity change, and
  the Corpus file holds no such field to read (ADR-0008). The whole input is the Corpus
  and the committed base rates, and `tests/test_corpus_composition.py` watches which
  files a run opens.

## Where the figures come from

- Corpus: `{facts.corpus_path}`, {facts.posts} posts by {facts.accounts} accounts, SHA-256 `{facts.corpus_sha256}`.
- Base rates: `{facts.base_rates_path}`, {facts.base_rate_categories} thematic categories over {facts.reports:,} reports, computed from the cached extract (ADR-0010).
- The projection both columns are read through: `docs/scam-categories.md`, generated by `rfi scam-categories` from those figures. {dropped:,} reports ({_percent(dropped, facts.reports)}) land in no Scam Category because their category was dropped, and they are in the denominator of the CAFC column.

Open Government Portal record: <{RECORD_URL}>

Attribution: {ATTRIBUTION}

Licence: [{LICENCE}]({LICENCE_URL})
"""


def _chart_row(row: Row, scale: int, name: int) -> str:
    """One row of the page's chart: both bars on the one scale, figures at the end.

    Both bars padded to the full width, including the empty ones, so the two figures on
    a row of the chart line up down the page however short either bar is.
    """
    return (
        f"{row.name:<{name}}  {row.corpus.bar(scale).ljust(_BAR_WIDTH)}  "
        f"{row.corpus.percent():>6}  {row.cafc.bar(scale).ljust(_BAR_WIDTH)}  "
        f"{row.cafc.percent():>6}"
    )


def _section(row: Row) -> str:
    """One class: what it covers, how much of the Corpus is in it, and its whole list.

    Printed for all ten and not only for the ones with posts in them, because a class
    holding nothing is the finding the table above is read for and a reader should not
    have to run the command to see the rule that found nothing.

    The bucket gets the opposite treatment, and necessarily: it has no list, because a
    post no list matched is what lands in it. Saying so in the place where the other ten
    are given their phrases is what stops it reading as a class with an unwritten rule.
    """
    heading = (
        f"{row.posts} of the Corpus's posts, {row.corpus.percent()} against a base rate of "
        f"{row.cafc.percent()}, a difference of {row.difference()}. The posts are listed in "
        f"`{PLACEMENTS_PATH}`, each with the sentences that placed it."
    )
    if row.name == OTHER.name:
        listed = (
            "No list places a post here. The bucket is what a post no list matches lands "
            "in, so its size is a finding about what the generator did not write, what the "
            "projection does not cover, and what these lists cannot place."
        )
    else:
        phrases = CATEGORY_PHRASES[row.name]
        listed = (
            f"{len(phrases)} "
            f"{'phrase' if len(phrases) == 1 else 'phrases'}, tried in this order, and "
            "printed in full because a rule a reader cannot see is a figure they cannot "
            "check:\n\n"
            + "\n".join(
                wrap(", ".join(f'"{phrase}"' for phrase in phrases), indent=0, width=90)
            )
        )
    return f"""### {row.name}

{row.scam_category.covers}

{heading}

{listed}"""


def _percent(count: int, of: int) -> str:
    """A share to the tenth of a percent, the same way the tables print one."""
    return Share(count=count, of=of).percent()


