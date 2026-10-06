"""The README copies figures out of the generated reports, and nothing held it to them.

Every generated report is byte-held by the test beside it, so a report cannot drift from
the command that wrote it. The README is hand-written prose that copies figures out of
those reports and out of the committed data, and nothing checked the copy. Three of its
figures were a commit behind — and every commit that touched `data/` touched the README
too, so the README was never neglected. It was refreshed for the ticket that commit closed
and left stale everywhere else, which no test was in a position to see.

So this file holds the copies. Each figure is read out of a committed data file where one
holds it, and out of the generated report where the only place it exists is a percentage
the command worked out, then rebuilt into the shape the README prints it in. When the
Corpus changes and the reports move, this test names the line of the README that is now
wrong instead of leaving the README quietly behind.

two kinds are deliberately not checked. An aggregate the README works out from a report
rather than copies — the share of real reports landing in classes this Corpus has nothing
to say about — is out, because summing the report's rounded percentages gives a different
figure from the report's own arithmetic and a test over it would fail on rounding rather
than on drift. And the console blocks `rfi campaign-candidates` prints, which have no
committed report, are out except where the number they count is published elsewhere: the
README quotes "3 of 16 registrations and 0 of 6 Contact Points withheld" from the
console, and the 16 and the 6 are the distinct registration count
`data/domains/post-domains.jsonl` holds and the distinct Contact Point count
`data/contacts/post-contacts.jsonl` holds, so that line is checked against the data
rather than against a report that would move with it.

The assertions are patterns rather than whole blocks. The README aligns its columns by hand
and a generated report may be reflowed; what has to hold is the figure, not the whitespace
around it. Order is held where the README is mirroring a report that sorts its own table,
because a row a reader finds in one and looks for in the other should be in the same place
in both.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from reddit_fraud_intelligence.categories import ANNEX_CATEGORIES, CAFC_CATEGORIES

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"
BASE_RATES = REPO_ROOT / "data" / "cafc" / "base_rates.jsonl"
CANDIDATES = REPO_ROOT / "data" / "campaigns" / "campaign-candidates.jsonl"
COMPOSITION_REPORT = REPO_ROOT / "docs" / "corpus-composition.md"
CONTACTS = REPO_ROOT / "data" / "contacts" / "post-contacts.jsonl"
CONTACT_POINTS_REPORT = REPO_ROOT / "docs" / "contact-points.md"
CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
DOMAINS = REPO_ROOT / "data" / "domains" / "post-domains.jsonl"
LABELLED_CONTACTS = REPO_ROOT / "data" / "corpus" / "labelled-contacts.jsonl"
NUISANCE = REPO_ROOT / "data" / "corpus" / "nuisance.jsonl"
SHARED_HOSTS = REPO_ROOT / "data" / "infrastructure" / "shared-hosts.jsonl"

Row = Mapping[str, object]

# The README writes some of these figures as prose — "over seven kinds" rather than
# "over 7 kinds" — so a test over them has to read them as prose. A count past ten raises
# rather than returning nothing, which fails the test and asks for this table to be
# extended instead of quietly checking something else.
NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)


# The report states the false-positive figure in a sentence rather than a table — "A
# Contact Point was read out of 1 of the 22 posts that publish none" — so it is read back
# out of that sentence. Group one is how many of the empty posts gained an invented
# Contact Point, group two is how many publish none at all.
INVENTED_IN_REPORT = re.compile(r"Contact Point was read out of (\d+) of the (\d+) posts")


def readme() -> str:
    """The README as one string, to be searched rather than read."""
    return README.read_text(encoding="utf-8")


def text_of(path: Path) -> str:
    """A committed file's text."""
    return path.read_text(encoding="utf-8")


def records_of(path: Path) -> list[Row]:
    """A committed JSON Lines file, as the objects it holds."""
    return [
        json.loads(line)
        for line in text_of(path).splitlines()
        if line.strip()
    ]


def text(row: Row, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), f"{field} should be text, is {value!r}"
    return value


def texts(row: Row, field: str) -> list[str]:
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    return [str(entry) for entry in value]


def nested(row: Row, field: str) -> list[Row]:
    """A field holding a list of objects, as the objects it holds."""
    value = row[field]
    assert isinstance(value, list), f"{field} should be a list, is {value!r}"
    return [entry for entry in value if isinstance(entry, dict)]


def inner(row: Row, field: str) -> Row:
    """A field holding one object, as that object."""
    value = row[field]
    assert isinstance(value, dict), f"{field} should be a row, is {value!r}"
    return value


def integer(row: Row, field: str) -> int:
    value = row[field]
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"{field} should be a whole number, is {value!r}"
    )
    return value


def tables(path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """Every table in a report, keyed by its header, its rows keyed by their first cell.

    A report is generated, so its tables are uniform: a header row, a separator of dashes,
    then the rows. Keying by the header is how a row is told apart from a heading. A key
    is stripped of backticks, because a report names a kind or a category in code
    formatting and the README does not.
    """
    parsed: dict[str, dict[str, tuple[str, ...]]] = {}
    header: str | None = None
    for line in text_of(path).splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if not cells or not cells[0]:
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        if header is None:
            header = cells[0].strip("`")
            parsed[header] = {}
            continue
        parsed[header][cells[0].strip("`")] = cells[1:]
    return parsed


def spelled(count: int) -> str:
    """The count the way the README's sentence spells it rather than prints it."""
    if not 0 <= count < len(NUMBER_WORDS):
        raise ValueError(f"{count} is not in NUMBER_WORDS; extend it before quoting it")
    return NUMBER_WORDS[count]


def position(needle: str) -> int | None:
    """Where this text sits in the README, or `None` when it is not there.

    A README figure sits in an aligned block whose columns are hand-spaced, so a figure is
    located by its parts with `\\s+` between them and the surrounding alignment ignored.
    """
    found_at = re.search(needle, readme(), re.MULTILINE)
    return None if found_at is None else found_at.start()


def found(needle: str) -> bool:
    """Whether the README carries this text, with its spacing left free."""
    return position(needle) is not None


def row_of(category: str) -> str:
    """The pattern matching one row of the README's composition block, however it is spaced."""
    posts, corpus, cafc, difference = tables(COMPOSITION_REPORT)["Scam Category"][category]
    return (
        rf"^\s*{re.escape(category)}\s+{posts}\s+{re.escape(corpus)}\s+{re.escape(cafc)}"
        rf"\s+{re.escape(difference)}\s*$"
    )


def categories_the_readme_mirrors() -> list[str]:
    """Every Scam Category the report's table publishes, in the report's own order.

    All of them, including the eight the Corpus holds no post of: the README's block is a
    mirror of the report's table rather than a selection from it, so a class the Corpus is
    empty of is a row the README still carries, at `0` posts. Deriving the list from the
    report rather than restating it is what makes a new Scam Category a row the README
    cannot quietly leave out.
    """
    return list(tables(COMPOSITION_REPORT)["Scam Category"])


def test_the_registrations_the_readme_reports_withheld_are_the_ones_the_corpus_resolves() -> None:
    """Both edges of the filter, counted against the two committed files behind them.

    The console line the README quotes names the registrations withheld out of the ones
    the Corpus resolves, and the Contact Points withheld out of the ones the Corpus
    publishes. Both totals are derived from committed data rather than transcribed, so a
    README that quoted only one edge — or neither — would fail here.
    """
    withheld = len(records_of(SHARED_HOSTS))
    resolved = {
        domain
        for record in records_of(DOMAINS)
        for domain in texts(record, "domains")
    }
    published = {
        text(match, "value")
        for record in records_of(CONTACTS)
        for match in nested(record, "matches")
    }
    hosts = {str(row["host"]) for row in records_of(SHARED_HOSTS)}
    at_withheld = sum(1 for value in published if value.rpartition("@")[2] in hosts)

    assert found(
        rf"\b{withheld} of {len(resolved)} registrations and {at_withheld} of "
        rf"{len(published)} Contact Points withheld\b"
    ), (
        f"README.md does not report {withheld} of {len(resolved)} registrations and "
        f"{at_withheld} of {len(published)} Contact Points withheld"
    )


def test_the_timing_the_readme_quotes_is_the_timing_the_candidates_file_publishes() -> None:
    """The corroboration figure, counted off the file the grouping wrote.

    The candidates file is the committed record of which pieces of evidence the window
    corroborates and which it does not, so the README's count is derived from it rather
    than transcribed: a window change or a Corpus change moves the file and the README
    would otherwise be a commit behind, which is what this file exists to stop. Both halves
    of the figure are counted, because a README that quoted the candidates without the
    pieces would look the same whether the clock had agreed with every edge or one.
    """
    candidates = records_of(CANDIDATES)
    pieces = [piece for record in candidates for piece in nested(record, "corroboration")]
    corroborated = [
        piece
        for record in candidates
        for piece in nested(record, "corroboration")
        if integer(piece, "span_seconds") <= integer(inner(record, "timing"), "window_seconds")
    ]
    with_something = [
        record
        for record in candidates
        if integer(inner(record, "timing"), "corroborated") > 0
    ]

    assert candidates and pieces, "the committed candidates carry no timing at all"
    assert found(
        rf"\b{len(with_something)} of {len(candidates)} candidates and "
        rf"{len(corroborated)} of {len(pieces)} pieces of evidence corroborated\b"
    ), (
        f"README.md does not quote {len(with_something)} of {len(candidates)} candidates "
        f"and {len(corroborated)} of {len(pieces)} pieces of evidence corroborated"
    )


def test_the_nuisance_baseline_the_readme_quotes_is_the_baseline_the_figure_sits_on() -> None:
    records = records_of(NUISANCE)
    kinds = {text(record, "kind") for record in records}

    assert found(
        rf"\b{len(records)} records over {spelled(len(kinds))} kinds\b"
    ), f"README.md does not quote {len(records)} records over {spelled(len(kinds))} kinds"


def test_the_obfuscation_the_readme_quotes_is_the_obfuscation_the_corpus_carries() -> None:
    """The `obfuscated_contact` cell's counts, derived rather than taken on trust.

    This cell is the README's account of what the obfuscation tolerance is measured over,
    and it is the one place a reader learns which disguises were planted and which were
    read. It was a commit behind once already — it claimed two of five found where three
    of four are — and nothing was in a position to see it, because no other figure in the
    README covers this kind and the generated report prints the misses under Contact Point
    values rather than under a Nuisance Structure kind.

    So the counts are derived here from the two files that hold them: the Labelled Set says
    what each post publishes, and the committed reading says which of those values came
    back. A handle written four ways is then four entries, and the ones found are the ones
    the reading agrees with, which is the only definition of "found" this project uses.
    """
    planted = [
        record for record in records_of(NUISANCE) if text(record, "kind") == "obfuscated_contact"
    ]
    assert len(planted) == 1, f"expected one obfuscated_contact record, found {len(planted)}"
    posts = set(texts(planted[0], "posts"))

    # Each published entry is paired with the post that declares it, because whether it
    # was read is a fact about that post's own row in the committed reading.
    published = [
        (post_id, entry)
        for record in records_of(LABELLED_CONTACTS)
        if (post_id := text(record, "post_id")) in posts
        for entry in nested(record, "published")
    ]
    found_values = {
        text(record, "post_id"): {text(match, "value") for match in nested(record, "matches")}
        for record in records_of(CONTACTS)
    }

    # The handle is the one Contact Point this kind publishes more than once, so the
    # writings are the entries naming it and the ones read are the ones that came back.
    writings = [entry for _, entry in published if text(entry, "value") == "syn_copperlantern"]
    read_handle = [
        entry
        for post_id, entry in published
        if text(entry, "value") == "syn_copperlantern"
        and text(entry, "value") in found_values[post_id]
    ]
    pictures = [entry for entry in writings if text(entry, "written") == "image"]

    assert len(writings) == 4, f"the handle is published {len(writings)} ways, not 4"
    assert len(read_handle) == 3, f"{len(read_handle)} of the 4 writings were read, not 3"
    assert len(pictures) == 1, f"{len(pictures)} of the writings are a picture, not 1"
    assert pictures[0] not in read_handle, (
        "the picture is read, so the README's 'and the picture is not' is wrong"
    )

    # The address carrying a digit is published plainly and read as written; the one
    # written as words has no `@` in the post and is not read at all.
    emails = [(post_id, entry) for post_id, entry in published if text(entry, "kind") == "email"]
    as_written = [(post_id, entry) for post_id, entry in emails if text(entry, "written") == "plain"]
    as_words = [(post_id, entry) for post_id, entry in emails if text(entry, "written") == "obfuscated"]
    assert [text(entry, "value") for _, entry in as_written] == ["1ntake@copper-lantern.example"]
    assert [text(entry, "value") for _, entry in as_words] == ["intake@copper-lantern.example"]
    for post_id, entry in as_written:
        assert text(entry, "value") in found_values[post_id], (
            "the address carrying a digit was not read as written, "
            "so the README's 'read exactly as written' is wrong"
        )
    for post_id, entry in as_words:
        assert text(entry, "value") not in found_values[post_id], (
            "the address written as words was read, so the README's 'is not read at all' is wrong"
        )

    # And the seventh post publishes nothing at all, which is where the invented one is:
    # a post that gained a value it does not declare is an invented one, so the count is
    # the difference between what was read and what was published rather than a restatement.
    declared = {
        text(record, "post_id"): {text(entry, "value") for entry in nested(record, "published")}
        for record in records_of(LABELLED_CONTACTS)
    }
    silent = sorted(post_id for post_id in posts if not declared[post_id])
    invented = sorted(
        post_id for post_id in posts if found_values[post_id] - declared[post_id]
    )

    assert len(posts) == 7, f"the kind covers {len(posts)} posts, not 7"
    assert len(silent) == 1, f"{len(silent)} of the posts publish nothing, not 1"
    assert len(invented) == 1, f"{len(invented)} posts gained an undeclared Contact Point, not 1"
    assert set(invented) <= set(silent), (
        "an invented Contact Point is in a post that declares one: "
        f"{sorted(set(invented) - set(silent))}"
    )

    assert found(
        rf"one handle {len(writings)} ways\b.*\b{len(read_handle)} of the {len(writings)} are found"
    ), f"README.md does not quote {len(read_handle)} of {len(writings)} writings of the handle found"
    assert found(rf"\bA {len(posts)}th post publishes nothing\b"), (
        f"README.md does not quote the {len(posts)}th post publishing nothing"
    )
    assert found(rf"\bthe {len(invented)} Contact Point the reading invents in it\b"), (
        f"README.md does not quote the {len(invented)} invented Contact Point"
    )


@pytest.mark.parametrize("category", categories_the_readme_mirrors())
def test_each_row_of_the_readme_s_composition_block_is_the_reports_own_row(category: str) -> None:
    assert found(row_of(category)), (
        f"README.md does not carry {category} as {tables(COMPOSITION_REPORT)['Scam Category'][category]}"
    )


def test_the_readme_s_composition_block_lists_its_rows_in_the_reports_order() -> None:
    """The README mirrors the report rather than rearranging it.

    The report sorts its table by the Corpus share, widest first, and by name where the
    share ties — which puts the class the Corpus is heaviest in at the top and the eight
    it holds nothing of in alphabetical order below; the README carries every row the
    report publishes, those eight included. The order is held here because a reader who
    finds a row in one and looks for it in the other should find it in the same place,
    and because a block that can be silently reordered is a block that will be.
    """
    offsets = [
        (category, position(row_of(category))) for category in categories_the_readme_mirrors()
    ]
    missing = [category for category, offset in offsets if offset is None]
    assert not missing, f"the README is missing a row the report publishes: {missing}"

    where = [offset for _, offset in offsets if offset is not None]
    assert where == sorted(where), (
        "the README lists its composition rows in a different order from the report: "
        + ", ".join(f"{category} at {offset}" for (category, _), offset in zip(offsets, where, strict=True))
    )


def test_the_recall_the_readme_quotes_is_the_recall_the_contact_point_report_published() -> None:
    published_how = tables(CONTACT_POINTS_REPORT)["How it was published"]
    total = published_how["All"]
    plainly = published_how["Plainly"]
    written_out = published_how["Written out"]
    picture = published_how["As a picture of one"]
    posts = len(records_of(CORPUS))
    labelled = records_of(LABELLED_CONTACTS)
    contacts = sum(len(texts(record, "published")) for record in labelled)
    publishes_none = sum(1 for record in labelled if not texts(record, "published"))

    assert contacts == int(total[0]), "the Labelled Set and the report disagree on the total"
    assert found(
        rf"\b{posts} posts, {contacts} Contact Points published between them\b"
    ), f"README.md does not quote {posts} posts publishing {contacts} Contact Points"
    assert found(
        rf"\b{total[1]} of {total[0]} Contact Points the Corpus publishes were found:"
        rf"\s+{plainly[1]} of {plainly[0]} published plainly,"
        rf"\s+{written_out[1]} of {written_out[0]} written out\b"
    ), "README.md does not quote the published recall breakdown"
    assert found(
        rf"\b{picture[0]} of the {total[0]} is published only as a picture of one\b"
    ), f"README.md does not quote {picture[0]} of {total[0]} as a picture of one"

    # The false-positive half, both halves of it. The denominator alone is not the
    # figure: a report that invented a Contact Point in half the empty posts and one
    # that invented one in a single post of them share a denominator, and the numerator
    # is the one that tells them apart. It is checked here rather than left out with the
    # rounded percentages, because it is a count the report works out exactly.
    invented = INVENTED_IN_REPORT.search(text_of(CONTACT_POINTS_REPORT))
    assert invented is not None, "the report does not publish the invented figure"
    assert int(invented.group(2)) == publishes_none, (
        f"the report counts {invented.group(2)} posts publishing none, "
        f"the Labelled Set {publishes_none}"
    )
    assert found(
        rf"\b{invented.group(1)} of the {invented.group(2)} posts that publish none\b"
    ), (
        f"README.md does not quote {invented.group(1)} of the {invented.group(2)} "
        "posts that publish none"
    )


def test_the_three_category_counts_the_readme_quotes_are_the_committed_ones() -> None:
    annex = len(ANNEX_CATEGORIES)
    enumerated = len(records_of(BASE_RATES))
    union = len(CAFC_CATEGORIES)

    assert found(rf"\b{annex} thematic-category headings\b"), (
        f"README.md does not quote CAFC's annex as defining {annex} headings"
    )
    assert found(rf"\bthis window enumerates {enumerated}\b"), (
        f"README.md does not quote this window as enumerating {enumerated} values"
    )
    assert found(rf"\bCAFC has published is {union}\b"), (
        f"README.md does not quote the union of published names as {union}"
    )
