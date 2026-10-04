# A registration carries the Scam Category its postings agree on, and a post that disagrees with it is a Signal

A Registrable Domain carries an **Associated Scam Category**: the class a strict majority of
the postings reaching it were placed in, out of the two at least that have to be placed
before any of them counts. A `category_conflict` Signal fires where a post placed in one of
the ten links a registration associated with a different one, and it is the only Signal in
the weight set that reads two entity types at once — the post's own text and the text of the
postings that reach its links.

Both halves are printed with the Signal, so both are checkable by hand: what the post says
against the phrase lists printed with the output, and what the other postings say against the
tally printed for that registration in the registrations table. A registration with too few
placed postings, or whose placed postings split evenly, has no associated Scam Category, is
reported as unassociated rather than decided, and produces no conflict. `Other` is neither
associated nor contradicted: it is where a post lands when no list matched it, so it makes no
claim to disagree with.

This record covers the association and the comparison; ADR-0017 covers how a post is placed,
ADR-0015 covers the phrase-rule shape, ADR-0007 covers the score this Signal contributes to,
and ADR-0005 covers why a Registrable Domain is the only infrastructure a Campaign Candidate
may rest on.

## Considered Options

- **Read the association off the registration's own name, or off a domain-reputation feed.** Rejected for now, and the reason is the same one that keeps the shared-infrastructure list in this project honest: a feed is an input this project does not hold, and a Signal that cannot be recomputed from the bytes a reviewer can open breaks the auditability claim ADR-0007 is made of. The association here is a function of the Corpus, which is the whole of what the reviewer has beside them. When an authorised Corpus Provider can supply a reputation signal, the weight file is where that lands, exactly as ADR-0009 describes for the shared-infrastructure list.
- **Take the category of a registration's most recent posting, or of the posting a candidate was built from.** Rejected: both are a function of one post rather than of the Corpus, so the "association" would be whatever that post happened to say and a Signal would fire on a disagreement between two posts that were never in tension. A majority is the only rule here that can be checked by counting the postings a reader can already see.
- **Resolve a split by taking the widest class, or the one the projection declares first.** Rejected, and the second is the sharpest version of it. A 2–2 split between two pitches has no majority, and taking the class that happens to sort first would be a Signal resting on the order the projection declares the ten in — a rule about the table of class names rather than about the evidence. It would also make the Signal fire on whichever side of an even split the reader did not expect, which is the worst way for this weight set to be wrong. Both unassociated cases are printed with the tally beside them so a reader can see the split rather than take the outcome on trust.
- **Let a post's own posting count towards its own registration's association.** Accepted, and it is what makes the printed figure checkable. Excluding it would define the Signal as "this post disagrees with the other posts reaching this registration", which needs its own report sentence and cannot be read off the tally the output prints. With it, the association is a property of the registration and the Signal is a plain comparison: most of what reaches this registration is about something else, and this post says something different.
- **Count a post in Other as a class the registration can be associated with, and fire the Signal on a post placed in one of the ten.** Rejected: Other is the absence of a claim, not a claim. Treating it as one fires the Signal on every post the phrase lists cannot reach — a pitch in a language this build does not read, a link and a title, a complaint nobody wrote a phrase for — and the bucket's size is already a standing measure of what the lists do not cover (ADR-0017). Its counts stay in the tally and print last, because dropping them would hide why a registration three posts reach has nothing to say.
- **Exclude the withheld registrations from the association as well as from the score.** Rejected, for the reason ADR-0009 gives for the graph: a majority drawn from everybody's adverts is a majority about the shortener, so `hopcut.example` must not decide anything. But the association is a fact about the Corpus and the withholding is a decision this run makes about scoring it, so the class is still computed, printed with the withheld column saying `yes` beside it, and counted in the figure only where it can be used. One number per thing, and the table is what reconciles them.
- **Put the phrase lists in a second published file, or read the placements file the composition command wrote.** Rejected on both counts. A second copy of the lists is a second thing to keep in step with the first, and the command already prints them, so a reader has them in front of them either way. Reading `data/corpus/composition.jsonl` would make the score depend on whether somebody had run another command first, which every other command in this repository refuses to do, and it would put a fifth file on the path ADR-0018 draws. The lists and the placing moved to `placement.py` instead, so `rfi policy-score` and `rfi corpus-composition` place every post by the same function and neither can be wrong about it alone.
- **Price the Signal above the ones that rest on a single reading.** Rejected, and the weight file says why in its own row: both halves here are blunt. A phrase list places the post, and a handful of postings decides the registration, and a post naming two pitches is placed on whichever list the projection declares first — so a disagreement can be an artefact of that tie-break. At 20 it sits with the two middle Signals rather than with the frequency Signal the whole grouping rests on.

## Consequences

Every Registrable Domain in the Corpus now carries a Scam Category in the registrations
table, including the ones with none, and the two reasons a registration has none are printed
rather than inferred. `tests/test_policy_score.py` recomputes every one of those tallies from
`data/domains/post-domains.jsonl` and `data/corpus/composition.jsonl` — two other commands'
files over the same Corpus — and requires the run's own table to agree, so the audit for this
Signal is the same shape as the one `domain_frequency` already had.

The smallest Corpus in which the Signal can fire is three postings: the floor needs two, and
two placed postings can only split or agree, so the first registration that can carry a
disagreement is one with two postings of one pitch and a third of another. That is a
consequence of the floor and of self-inclusion together rather than a rule of its own, and it
is stated here because it bounds what the Signal can find on a small Corpus.

The published set is six Signals and 130 points rather than five and 110, so every score in
`data/signals/policy-scores.jsonl` moved. That is the normalisation doing its job rather than a
detail: a Signal added later moves the existing scores rather than pushing them over the top,
which is what ADR-0014 commits to and what makes a weight a published change instead of a code
edit.

**The Signal fires on no post of the committed Corpus, and that is stated rather than buried.**
Four of the sixteen registrations carry an associated Scam Category — `vantage-ledger.example`
and `signal-harbor.example` with two withheld ones beside them — and every posting reaching any
of the four agrees with it. The other twelve are one placed posting wide or split evenly, and
the other two withheld registrations are among those. A Corpus that planted a post running a
second pitch on a registration somebody else runs a first pitch on would exercise it, and the
generator can write one; planting it renumbers every post id after it and moves every published
figure in the repository, which is a change this ticket did not need in order to make the
Signal correct. So the case is exercised by tests over Corpora written to contain it, and the
README says which Corpus the Signal needs and what the shipped one lacks. A rule a reader
cannot see firing is a claim; this one names its own evidence.
