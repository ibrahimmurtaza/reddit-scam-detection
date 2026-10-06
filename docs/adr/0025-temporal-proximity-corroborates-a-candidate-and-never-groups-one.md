# Temporal proximity corroborates a candidate and never groups one

Every Campaign Candidate carries the timing of the accounts behind it: for each
registration and Contact Point it rests on, the span from the earliest post by any account
reaching that thing to the latest, and the gap between the nearest two accounts. A piece of
evidence is corroborated when its span falls inside a stated window — 24 hours by default,
`--window-hours` to change it — and the window is named in the output, on every candidate
and on the candidate's own line in the index.

Timing may deprioritise and nothing else. Candidates the window corroborates something
under are printed first; a candidate it corroborates nothing under is printed in full, with
the gap beside each of its pieces of evidence, is named again below the table under its own
heading, and says so on its `justified by` line. Nothing is removed, because a domain two
accounts reached months apart may be a domain that changed hands and a campaign may simply
be a patient one. Timing can never produce a candidate: the union-find reads shared
infrastructure only, and two accounts posting in the same minute with nothing shared are not
grouped at any window (ADR-0005).

The window decides on the span rather than on the closest pair. A component is a claim
about a group, and one coincidence inside it is not the group.

The published figures move, and the movement is the point. On this Corpus three of the four
candidates are corroborated and the desk behind the obfuscated handles — one handle, three
accounts, seven posts over three weeks — is not, so it moves to the end of the table and its
identifier moves with it. The other half of the finding is inside `cc-01`: the registration
puts three accounts inside two and a quarter hours and the Telegram handle brings in a
fourth account two weeks later, which is the very account that costs the project a recovery.
One candidate, one registration, one Contact Point, and the clock agrees with one of them.

## Considered Options

- **Let timing remove a candidate the window corroborates nothing under.** Rejected: it
  would make a threshold a decision, and a threshold is a number a reader can disagree with
  while a decision is one they have to undo. The ticket asks that a miss be diagnosable
  rather than silent, and a candidate deleted for want of timing is a miss with nothing left
  to diagnose. The same objection is why the shared-infrastructure filter publishes what it
  removed (ADR-0009): removing is a thing you have to be able to look at afterwards.
- **Decide on the closest pair of accounts rather than the span.** Rejected, and it is the
  near miss worth recording. The closest pair is the most forgiving reading and it lets a
  single coincidence corroborate a whole component: two of five accounts posting in the same
  minute by accident would be enough to call the other three corroborated. On this Corpus it
  would also report every piece of evidence in the run as corroborated, including the desk
  that kept one handle for three weeks — a Signal that fires on nothing is a claim. The span
  is the stronger reading and it is the one the ticket's own framing asks for, since what it
  contrasts with "posting within a tight window" is accounts "spread over months".
- **Score each candidate once, over all its accounts together, rather than per piece of
  evidence.** Rejected: the two halves of `cc-01` disagree, and collapsing them hides it. A
  component-wide span would put the alpha candidate outside the window on the strength of the
  complaint that joined it, and would throw away the fact that three of its four accounts
  posted within two and a quarter hours.
- **Print the verdict and not the gaps.** Rejected: a corroborated count without the numbers
  behind it is a score, and this project's own position is that a number a reader cannot
  recompute is a claim. Both gaps are published in the file beside the evidence they
  describe, and `tests/test_campaign_candidates.py` recomputes every one of them a second
  time off `post-domains.jsonl` and `post-contacts.jsonl`.
- **Store the verdict as a column of `campaign-candidates.jsonl`.** Rejected for the reason
  ADR-0023 gives for not storing the evidence label: a verdict nothing recomputes is a
  second account of the same fact, and the file and the table printed beside it could then
  disagree about whether a candidate was corroborated. The window is stored and the verdict
  is worked out from it on demand, in both views.
- **Refuse a `corroboration` row the candidate's evidence lists do not name.** Accepted, and
  it is the check that makes the file usable by the two commands that read it: the evaluator
  does no timing of its own, so a gap for a piece of evidence the candidate is not joined on
  would be weighed by a reader against a grouping it does not justify. The same reader
  refuses a row whose count of pieces, nearest pair, span or corroborated count disagrees
  with the gaps printed above it, a window shorter than the grouping itself accepts, and a
  file whose rows were counted at two different windows.

## Consequences

`CampaignCandidate` gains `corroboration` and `timing`, so the candidates file grows two
fields and every consumer of it grows a reader: `rfi campaign-recovery` and
`rfi review-queue` check the timing row by row without acting on it, because neither is
where a candidate's place in the world is judged. Identifiers move: the identifier says
where a candidate sits in the output and nothing else, so a candidate the clock has nothing
to say about moves to the end of the list and its identifier with it. `cc-02` through `cc-04`
in this repository's committed figures are the new order.

The cost is stated rather than hidden. A campaign with accounts that posted months apart is
deprioritised for it, which is a real loss to a reviewer reading the list in order and a
false alarm to one weighing the figures. Both are the price of a threshold anybody can see
and change, and the alternative — a rule the reader cannot check — was not available at any
price.