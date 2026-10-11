# Why the per-post readers miss real scammer text

Research date: 2026-10-11. Measurement only; no project code was changed to produce it.

## What was measured

800 rows of `label: 1` (scam) from
`huggingface.co/datasets/Purino/scam-ads-detection-dataset`, pulled through the
datasets-server row API rather than the 706 MB parquet, which is mostly images. These are
Meta Ad Library advertisements **written by scammers to recruit victims** — the correct
genre for this project, and the only labelled corpus of it that is downloadable.

The dataset's own negative class was **not used**. Rows 3600–3711 are generated from
templates describing image contents (`⭐ a bottle of omega fatty...`) with sequential
synthetic page IDs and Roboflow filenames. An AUC against them would measure real-scam
against template. So this is an absolute question, not a ranking one:

> On text written by a real scammer, do the published Signals fire at all?

## The result

| | |
|---|---:|
| Rows | 800 |
| Scoring zero | **714 (89.2%)** |
| Scoring above zero | 86 (10.8%) |
| Mean score | 1.72 of 130 |
| Max score | 31 |

| Signal | Firing rate |
|---|---:|
| `guaranteed_return` | 5.5% |
| `urgency_language` | 5.6% |
| `payment_request` | **0.0%** |

`payment_request` fired on **nothing**. Of the eight Scam Categories, 83.5% of rows landed in
`Other`.

## Why: the phrase lists are the corpus

The three measurable Signals are literal phrase matchers. Their entire content is 31 strings:

```
guaranteed_return: assured, can't lose, cannot lose, guaranteed, no risk, no way it fails,
                   no way to lose, risk free, risk-free, safe return, will not lose, won't lose
payment_request:   buy up front, deposit before, deposit for, equipment charge, kit fee,
                   materials deposit, money up front, pay up front, refundable deposit
urgency_language:  48-hour, act now, before it goes, fills fast, fills up fast, last chance,
                   limited time, place goes to, slot goes to, while it lasts, within 48 hours
```

Six of those phrases occur in the committed **synthetic** Corpus and in **zero** rows of the
real job advertisements:

| Phrase | In synthetic corpus | In real job ads |
|---|---:|---:|
| `slot goes to` | 4 posts | 0 |
| `place goes to` | 1 post | 0 |
| `kit fee` | 1 post | 0 |
| `equipment charge` | 1 post | 0 |
| `money up front` | 1 post | 0 |
| `fills fast` | 1 post | 0 |

`slot goes to` is lifted from `syn_p_0012` — the committed decoy job post whose body reads
*"the slot goes to someone else if you are late."* `kit fee` and `equipment charge` come from
the same planted job-scam cluster.

**The phrase lists were written against the project's own corpus, and the corpus was written
to contain them.** The Policy Score is a matcher for 31 hard-coded strings drawn from the
synthetic data it is then evaluated on. That is why it reaches a perfect ordering on a
Corpus planted to exercise it and does not move on content nobody here wrote.

## False positives, too

The rows that *did* fire are largely incidental matches, not scam language:

- *"restful sleep **guaranteed** on the first night"* — a product-quality claim; the Signal
  is named `guaranteed_return` and fired on a sleep app.
- Several rows are **narrative fiction** ("chapter 1 april 2008. the skies over new york…",
  "i once thought i was cinderella…") scoring 12 points on `urgency_language`. Story prose
  contains the word "limited" or "chance" incidentally.

So the list produces both misses on the target genre and hits on text with no scam content at
all. The two failure directions are the same root cause: the phrases are not a model of scam
language, they are a copy of one corpus's.

## What this does and does not establish

**It does establish something serious.** The Policy Score has no capacity to generalise, and
the external validation performed its job by showing it. Any figure measured on the project's
own Corpus — including the 1-of-2 recovery rate and the Policy Score baseline's AUC of 1.0 —
is circular with respect to these phrase lists, because the Corpus was written to contain
them.

**It does not establish that scam language is unlearnable.** The fix is conceptually simple:
author the phrase lists against real text, or replace literal matching with something that
generalises. Both need labelled real text, which §"What remains unobtainable" in
`external-sources.md` shows is not currently obtainable — so this is a blocked fix, not an
absent idea.

**It does not bear on the link Signals.** Those three could not be measured on either
external corpus (no URLs), and they are the half that would carry most of the signal for
campaign grouping. Whether they generalise is still unknown.