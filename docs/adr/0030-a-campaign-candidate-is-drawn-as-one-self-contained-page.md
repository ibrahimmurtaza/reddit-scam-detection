# A Campaign Candidate is drawn as one self-contained page, from the published file alone

Ticket #28 asks for the relationship argument to be visible without a toolchain: a
self-contained HTML file with inline SVG, generated from a single CLI run, showing
accounts, posts, registrations and Contact Points. This records the four decisions that
were open in it and are not in the ticket: where the picture comes from, how many files
there are, what a line on it means, and what happens to a page whose candidate has gone.

**One page per candidate, named for it.** Not one page for the whole run, and not a page
for the largest candidate. A graph of the clearest candidate is a demonstration, and this
is an output a reader argues with: somebody who thinks `cc-04` is a false grouping has
to be able to open `cc-04`. The file is `docs/campaign-graph/<candidate id>.html`, so a
picture and the row it was drawn from are found by the same name, and the console lists
one line per candidate with the page beside it. A page this run did not write is removed
from the directory, because a directory that only grows keeps a picture of a candidate
the file no longer holds — which is the one way this output could drift from the run
without a test failing. `rfi content-embeddings` needs `--replace` for the same reason;
here there is nothing to lose, so it is not a flag.

**Drawn from `data/campaigns/campaign-candidates.jsonl` and from nothing else.** The
Corpus is not re-read, the grouping is not re-run, the Public Suffix List is not opened,
the vector table is not queried, and the membership is not looked at. This is ADR-0018's
boundary used for a third purpose: the Review Queue cannot reach inference because it
reads two published files, and a picture cannot either, because a picture is the output
a reader is most likely to believe on sight — a rendering is evidence-shaped in a way a
table is not. It costs one thing, which is that a post node is an identifier rather than
its text; the picture is about which accounts reach which shared things, and the text is
one tab away in the Corpus file the identifier is a key into.

**Four nodes and four lines, and the line that joins accounts is the strongest thing on
the page.** An account, a post, a shared registration and a shared Contact Point are four
shapes, because a colour is the first thing to go on a page printed in black and white.
An account writes a post, a post links a registration, a post publishes a Contact Point,
and an account shares a registration or a Contact Point with another account — four
lines, four strokes. The last is why the candidate exists (ADR-0005) and it is drawn
bowed underneath the other three, because those are the ones that cross the whole
picture: a straight line from an account to a registration would pass behind every post
between them and read as though it joined one. A candidate resting on one Contact Point
and no registration is drawn with no registration on it, which is the shape ADR-0023 says
is a claim of its own and `rfi contact-points` has the measured recall for.

**The page carries the verdict, because a picture without one shows a grouping.** The
cohesion system retains a candidate on two of three named corroborations (ADR-0028), and
a reader looking at four accounts joined to a handle cannot tell that from one the system
removed. So the page names which corroborations held, what each of them read, and
whether the system kept the candidate or removed it, beside the evidence table and under
the picture rather than behind a link — there are no links, because there is nothing to
link to. The Confidence is not displayed here either (ADR-0027, ADR-0003), and no figure
about fraud is named anywhere on the page (ADR-0004).

The layout is arithmetic over the row's own order: accounts in a column, posts in a
column, the shared things in a third, each column as wide as its own longest label and
each column centred against the tallest one. Nothing is measured and nothing is inferred;
the coordinates are the only thing this command decides for itself, and it decides them
the same way for the same row every time, because a repository of pictures that reorder
themselves on an unrelated commit is a repository where every diff is noise.

**What "cannot drift" means here, hop by hop.** The page is drawn from the file the
grouping published, not from the grouping's own memory, so the chain from a Corpus to a
picture is two hops and each one is byte-held by a test: `rfi campaign-candidates` writes
`data/campaigns/campaign-candidates.jsonl` and
`test_the_committed_candidates_are_what_the_command_writes` regenerates that file from the
committed Corpus and fails if a single byte differs; `rfi campaign-graph` writes the pages
and `test_the_committed_graphs_are_the_ones_this_command_writes` regenerates them from the
committed candidates file and fails on the same terms. So a renderer that changed without
the pages being redrawn, or a grouping that changed without the candidates file being
rewritten, each fails a test rather than leaving a picture nobody checks. The first hop
needs a database, because the grouping corroborates on the stored vectors (ADR-0026); on
a machine with none it skips with the reason, and the second hop still holds.

**What the pruning costs.** Removing a page the run did not write is the only destructive
thing this command does, and it is destructive to the directory it is pointed at: handed
`--graphs docs` it would take every `.html` in there. It does so because the alternative
is worse — a directory that only grows keeps a picture of a candidate the candidates file
no longer holds, committed, describing a grouping nothing produces — and because the
command says so in `--help`, in its ADR and on every line it removes. The narrower
guarantee is also asserted independently:
`test_the_committed_graphs_are_the_ones_this_command_writes` fails on a page in the
repository that no run over the candidates file writes, so the drift the pruning prevents is
caught by a test whether or not the pruning happens.

## Considered Options

- **One page for every candidate.** Rejected: it puts four drawings on one page, which is
  a gallery rather than a report, and it makes "open the candidate you think is wrong"
  mean scrolling. The file size argument is the other way — four pages of a few kilobytes
  each is not what makes a page expensive to open.
- **Re-run the grouping and draw its output directly.** Rejected for ADR-0018 and for a
  second reason: it would make the page a function of a Corpus rather than of a published
  file, so a picture could describe a Corpus nobody has a file for. The cost is that the
  picture cannot quote a post's own text, which is the right thing to give up for a page
  whose whole claim is that it can be checked against something.
- **Read the Corpus as well, to put a post's title on its node.** Rejected: a second input
  file makes the graph a join this project would then have to check, and a title is a
  long string on a node 40 pixels across. The identifier is the key into the Corpus.
- **Force a straight line from an account to the thing it shares.** Rejected: it crosses
  the whole picture and passes behind the post column, so the strongest claim on the page
  would be the line most likely to be read as joining something else. The bow costs a
  function computing where a quadratic curve actually reaches, which is worth it.
- **Draw one shape per kind and tell them apart by colour alone.** Rejected: colour is
  lost on a monochrome print and to a reader who cannot separate the hues, and this is a
  picture meant to survive being pasted into an issue.
- **Commit the pages and hold them to their bytes.** This is what happens, because every
  generated file in this repository is committed and held. A picture is the case where it
  matters most: it reads as an artefact of the analysis rather than as a generated file,
  so a renderer that changed without the pages being redrawn would leave the repository
  describing a graph this code does not draw.
- **Draw the picture and nothing else.** Rejected, and it is the choice the ticket's own
  wording argues against: "the relationship argument visible" is an argument, and an
  argument a reader cannot check is a claim. The evidence table under the picture is where
  the timing and the cohesion verdict are taken apart, and the node notes are where the
  figures already beside each identifier in the candidates file are put beside it on the
  page. Both are the same rows, read twice, rather than a second measurement.

## Consequences

`rfi campaign-graph` is a new command over one input file and one output directory, and it
needs no database and no network. `docs/campaign-graph/` is a new committed directory of
four files, held to their bytes by `tests/test_graph_report.py` the way every other
generated artefact here is.

The vocabulary test scans the pages as well as the source, because they are the project's
own prose in the place a reader is most likely to read it. That is why the page says
"a proposed grouping, not a finding" rather than "a candidate, not a confirmed campaign":
the second is a phrase this project's glossary does not use at all, and a page whose label
was worth a glossary exception would be a page asking for an exception.
