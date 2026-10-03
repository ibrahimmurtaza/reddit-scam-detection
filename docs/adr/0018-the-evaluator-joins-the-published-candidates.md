# The evaluator joins the published candidates, and runs after inference

Recovery of Planted Campaigns is measured by a separate command, `rfi campaign-recovery`,
which reads four files and runs the grouping itself zero times: the Corpus, the truth
file, the Nuisance Structure manifest, and `data/campaigns/campaign-candidates.jsonl`,
which `rfi campaign-candidates` wrote. It never imports, calls, or re-runs the grouping,
and nothing in the grouping path opens the truth file or the manifest. The figure is
therefore a function of bytes a reader can open, and the ordering between the two steps is
a committed file rather than a promise in prose. A candidate counts as a recovery only
when it holds a Planted Campaign's whole membership; anything less is reported beside the
figure rather than folded into it. This record covers how the two steps are separated and
what the join counts; ADR-0004 covers why campaign recovery is the figure, ADR-0008 covers
the Corpus file carrying no membership, and ADR-0009 covers the baseline the grouping is
compared against.

## Considered Options

- **Re-run the grouping inside the evaluation command.** Rejected, and it is the option the rest of this repository takes: every command is a function of the Corpus, and `campaigns.py` and `signals.py` both re-resolve the links rather than reading `post-domains.jsonl`, so a result cannot depend on whether somebody remembered to run the step before it. The convention is right for a command that measures nothing, and wrong here. The point of this command is that it cannot reach inference: a process that both groups and measures is a process that could group and read the membership, and the only symptom would be a recovery figure that had quietly gone up. Reading a published file also means the number describes bytes — the candidates file is committed and a test holds it to what the command writes — rather than a run that exists only inside one process.
- **One command that groups, then measures, then prints.** Rejected for the same reason, and worse: a single command shows no boundary at all. Two commands and a file between them are the boundary, and the audit-hook test that watches which files each of the two opens is the evidence a reader can check for themselves.
- **Count a campaign recovered when a candidate names any of its accounts.** Rejected, and it is the version of the figure that flatters the system. Two accounts of a three-account campaign grouped on their shared registration is a real grouping and is not a recovery; a campaign split across two candidates is not a miss either. Both are reported apart from X of N, with the accounts held, missing, and unexpected named, because a number a reader cannot decompose is a number a reader has to take on trust.
- **Match on posts rather than on accounts.** Rejected: membership is accounts. A campaign's accounts write other posts, and a candidate's posts are every post its accounts wrote, so the two sets differ legitimately and a post comparison would report a mismatch on a campaign recovered exactly.
- **Call a candidate that matches no campaign a miss against N.** Rejected, and the Corpus contains the case that settles it: a shop's three accounts share its domain, ADR-0005 says they group, and the grouping is right to produce them. Counting that as a miss would report correct behaviour as a failure and would put N above the number of planted things. It is printed instead, named with the registration that joins it, because X of N is unreadable without knowing what the other candidates are.
- **Report the figure with nothing about what it was measured against.** Rejected: ADR-0004 makes the Nuisance Structure the other half of the claim. A recovery rate over a clean sweep is a figure about a generator that planted two campaigns and nothing else, so the manifest is read, its counts by kind are printed, and the rate of groupings that are not recoveries is left to the ticket that measures it rather than half-done here.
- **Report a share of the posts called right or wrong.** Rejected, and not a close call. Every label in this Corpus was assigned by the generator that wrote the posts, so such a figure would measure agreement with the generator rather than anything about fraud, and the word itself appears nowhere in the output — a test asserts its absence rather than trusting this paragraph.

## Consequences

Every file in the chain is committed and held to the bytes its own command writes, so the figure is reproducible: `rfi generate-corpus` at the seed the report prints writes the Corpus, the membership, the manifest, and the shared-infrastructure list again byte for byte, and `rfi campaign-candidates` writes the candidates again byte for byte. Nothing in that chain reads a seed at run time. The seed decides what was generated, and the report carries it so a reader can regenerate rather than take the claim on trust.

The evaluator refuses a truth file or a candidates file that names an account or a post the Corpus does not hold, because a membership measured against a Corpus that does not contain it makes X of N meaningless. It also refuses an unknown field in either file, for the reason `read_corpus` gives: a `campaign_id` appearing in the candidates file would be Planted Campaign membership in the pipeline's output, and quietly ignoring a field that has appeared is the first way to stop being able to say the file does not carry one.

The number is a lower bound and says so. A Planted Campaign that leans on a shared host is lost with the host, an account that reaches no registration cannot be proposed at all, and a campaign that rotates its registration from post to post is invisible by construction. Recovery measured this way is a statement about planted structure in a synthetic Corpus, and the report says that in those words rather than letting the figure stand on its own.

The next measurement over the same manifest is the rate at which the grouping is wrong, which is ticket #19. Until it lands, this figure has no companion and the report says where the companion is coming from rather than implying there is none.