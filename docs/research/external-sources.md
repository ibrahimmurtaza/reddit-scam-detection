# External sources: labelled corpora resembling Reddit scam posts

Research date: 2026-10-11. Research-only; no project code was written.

The question: does a public, labelled, legally-obtainable text corpus exist whose text is
**scammer-authored promotional content** — the pitch a recruiter writes — as opposed to
(a) victim reports about scams, or (b) ordinary non-scam content?

The short answer, established below: **no.** Every genuinely labelled corpus found is either
category (b), category (c), synthetic, or has its URLs stripped. The closest match on *text*
has no usable links and a fabricated negative class.

---

## Candidate table

Categories: **(a)** scammer's own promotional content · **(b)** victim reports · **(c)** ordinary/benign.

| # | Name | URL | Category | Size / range | Licence | URLs/domains? | Multi-account? | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Purino scam-ads-detection-dataset | `huggingface.co/datasets/Purino/scam-ads-detection-dataset` | (a) + synthetic (c) | 3,705 rows; Meta Ad Library, ~2025 | MIT (declared) | **No — redacted to `<URL>`** | **Yes — `page_id` + `page_name`** | Closest text; **negatives are fabricated** |
| 2 | Smishing-Dataset-IMC25 | `github.com/reportsmishing/Smishing-Dataset-IMC25` | (a) | 48,735 rows; 2023–2024 | **CC-BY-4.0** | Partial — `<URL>` placeholders, PII-scrubbed | Sender IDs, no campaign graph | Real labels, clean licence |
| 3 | 419-scam email corpus (419scam.org) | via Isacenkova et al., IEEE S&PW 2013 | (a) | 36,761 messages; 2009–2012 | **Not released** | Yes (emails + phones) | Yes (phone/email clustering) | **Unobtainable** |
| 4 | Reddit self-disclosure narratives (arXiv 2606.16874) | `arxiv.org/abs/2606.16874` | **(b)** | 21,304 posts; 2023–2025 | Paper available | Not stated | No | **Not useful — victim reports** |
| 5 | Arctic Shift | `arctic-shift.photon-reddit.com` | (c), no labels | ~2.5B objects to 2026 | Unclear / not a licence | Yes (raw `url` field) | Yes (author + domain) | **Reddit raw text — unlicensed** |
| 6 | PullPush | `pullpush.io` | (c), no labels | Frozen at **2025-05-20** | None | Yes | Yes | Effectively dead |
| 7 | Pushshift | `pushshift.io` | (c), no labels | n/a | **Mods only** | n/a | n/a | **Closed to us** |
| 8 | Attestable scam-detection-corpus | `huggingface.co/datasets/Attestable/scam-detection-corpus` | (a) synthetic | 153 examples | Stated on card | No | No | Too small, hand-made |
| 9 | Meta fraudulent-ads study (Ferreira et al.) | `exa.ai/library/publication/74hv2f8d0v3` | (a) | 3,365 hand-labelled | **Not released** | Yes (ad library) | Yes (page-level) | **Unobtainable, but closest in spirit** |

---

## 1. Purino scam-ads-detection-dataset — closest text, unusable negatives

The only genuinely labelled corpus of scammer-authored promotional text found. Source is the
**Meta Ad Library**; each row is one ad, with `page_id`, `page_name`, `text`, an image,
a `rule_score`, a `rule_hits` explanation, `label_suggested`, and a `label` ClassLabel of
`non_scam`/`scam`. MIT licence, ungated, 706 MB download.

**Text resemblance is excellent — this is category (a), verbatim.** Two rows read (from the
raw dataset, not a summary):

> "congratulations i'm julie leach the michigan powerball powerplay winner, so you have been
> fortuitously picked among our electoral machine on social media lucky winners those getting
> $20,000,000 million dollars this month from my giveaway program…"

> "hey fans! 🎉 from elon musk. we're excited to announce a giveaway just for you! if you're
> interested in entering to win $250,000 and a brand new tesla, message below with done and get yours"

Both hit this project's Signals directly: `guaranteed_return`, `urgency_language`, and a
`payment_request` in the "message below with done" form. Another row is pure urgency with no
product at all — "𝐉𝐮𝐬𝐭 $𝟕𝟗- 𝐓𝐨𝐝𝐚𝐲 𝐎𝐧𝐥𝐲! 𝐉𝐮𝐬𝐭 $𝟕𝟗- 𝐓𝐨𝐝𝐚𝐲 𝐎𝐧𝐥𝐲!" — which is the failure mode `category_conflict` exists to catch.

**It carries real multi-account structure.** `page_id` is present, and the corpus contains
duplicate creative across distinct page IDs — the same "clearnail pro" copy appears under
`page_id` 434190399773378 ("Enseignement-Apprentissage") and 102819379468768 ("Frases de amor").
That is the shape of a campaign: one creative, many pages. This is the **only** candidate found
that supports the Campaign Candidate grouping claim with actual account identifiers.

### Three disqualifying problems

**1. URLs are redacted to a literal `<URL>` token.** This is not a viewer artefact — it is
what the parquet stores. A row reads "get yours now 👉🏻 `<URL>`". The `domain_frequency` and
`domain_lookalike` Signals cannot be evaluated on this data at all, and neither can the
registrable-domain grouping that ADR-0005 requires.

**2. The negative class is fabricated.** Rows 3600–3711 are `label: 0` with
`rule_hits: "No suspicious patterns detected."` and `rule_score: "0"`. They are not scraped
ads. They are generated from templates describing image contents:

> "⭐ a bottle of omega fatty and a bottle of fish oil — top-rated by customers everywhere!
> Don't miss out. Order yours today! ✅"
>
> "📦 a quality product item ecb15e — fast shipping, hassle-free returns. Shop with confidence today! 💛"

Two tells confirm the fabrication: the `page_id`s are sequential and obviously artificial
(100001001, 100002001, 100003001…), and the `image_filename`s are Roboflow export paths
(`new_batch2_banner_901_jpg.rf.e1e67af6b6a4fb0f5ce24c8fd0ecb15e.jpg`) against 224×224 images,
whereas the real rows carry native Meta ad IDs and 1024×1024 images. The image is described in
the text because the text was written *from* the image.

So an AUC computed on this dataset would measure separation of real scams from templates. Any
positive AUC would be an artefact of the generator, not evidence about this project's rules.

**3. Provenance is unstated.** The card is **empty** ("README.md exists but content is empty"),
there are two likes, 60 downloads, and no paper, author affiliation, or collection method. The
MIT tag is asserted by the uploader over Meta's ad content, which the uploader may not hold
copyright in. Treat the licence as unreliable.

## 2. Smishing-Dataset-IMC25 — clean licence, real labels, wrong genre

48,735 rows from the IMC 2025 paper *Fishing for Smishing* (doi 10.1145/3730567.3764431).
**CC-BY-4.0**, the only unambiguously clear licence found in this whole search. Plain CSV on
GitHub, no auth.

Columns: `time, sender_id, telephone_number_type, original_network_name,
original_network_country, text, translation, language, url_shortener, named_entity, scam_type,
lure_principles`.

This is genuinely category (a) — the scammer's own message — with a labelled taxonomy
(`scam_type`: delivery, others, …) and a hand-coded `lure_principles` field that maps closely
onto this project's Signals. From the raw CSV:

> "Correos Express, seu pacote segue retido em aduanas por impago de das taxas (1,99?) pode
> pagalos no seguinte enlace: `<URL>` v4ApD16" — `scam_type: delivery`,
> `lure_principles: authority,time/urgency`

The `time/urgency` and `authority` labels are close cousins of `urgency_language`. And
`original_network_name` / `original_network_country` (KPN Mobile, NLD; Movistar, ESP) is
attacker-side infrastructure metadata, not victim metadata.

Two limits. It is **SMS, not Reddit** — 160 characters, no subreddit, no comment context.
And links are again reduced to `<URL>`, with sender identifiers anonymised, so campaign grouping
is not possible. Useful for phrase-list calibration; not usable as a ranking benchmark.

## 3. 419-scam corpus — the right shape, not obtainable

36,761 messages from 419scam.org, Jan 2009–Aug 2012, with 112,961 email addresses and 11,768
unique phone numbers (Isacenkova et al., *Inside the SCAM Jungle*, IEEE S&PW 2013,
`ieee-security.org/TC/SPW2013/papers/data/5017a143.pdf`). Category (a), with an explicit
campaign-grouping method — the paper clusters messages by shared phone number and email
address to attribute ~1,000 campaigns. That is precisely this project's grouping claim,
on scammer text, with contact points.

**It is not released.** The source is a victim-report aggregator; the paper analysed a private
download; no licence was ever granted and the paper states IP data are absent. A related line
of work (Park et al., UMD) built a 200,000+ user-reported scam email repository and a
Craigslist honeypot collection — also unpublished. **This class of corpus is not obtainable.**

---

## Reddit access, as it stands in October 2026

This is the binding constraint on the whole question, and it has tightened sharply.

**Reddit for Researchers still exists and is open**, per Reddit's own help article
(`support.reddithelp.com/hc/en-us/articles/49381918834964`, updated 2026-10-05). Out of beta
as of 2026-05-28. Eligibility: accredited university, institutional email, detailed proposal,
IRB approval or exemption, institutional sponsor. Access is **per-project, max one year**,
delivered through **BigQuery Analytics Hub**, covering a **5-year lookback refreshed monthly**.
Non-commercial academic use only. Redistribution is prohibited. Individual team members must
each apply. This project has no institutional affiliation, so this route is closed to us.

Reddit's Responsible Builder Policy is explicit that *"Any research that uses Reddit data
collected outside of the RFR Program is in violation of this policy."*

**The Data API is closing.** From Reddit's 2026-08-05 announcement
(`redditinc.com/news/modernizing-reddits-infrastructure-and-moderation-tools`) and the
2026-09-30 update:

- New Data API access requests close **2026-10-31** (i.e. in three weeks).
- From **2027-01-12**, access is removed for apps that have not registered.
- Public API access ends **March 2027** for everything remaining.
- RSS ends 2026-11-13.

Free tier was 100 requests/minute, non-commercial; commercial use always required a direct
contract with Reddit, and the Data API Terms (last revised 2026-07-20) reserve the right to
revoke access without notice.

**The archives have decayed.**

- **Pushshift** — I fetched `pushshift.io` directly. Its terms now require you to certify you
  are *a registered Reddit moderator*, using Pushshift only for community moderation, via an
  OAuth token expiring after 24 hours. **Not available to this project.**
- **PullPush** — volunteer-run, ~30 req/min, frequent unannounced outages, and **ingestion
  stopped 2025-05-20**. It returns HTTP 200 with sparse results, so it fails silently.
- **Arctic Shift** — **genuinely alive**; its status page reports all components available. This
  is the one working Reddit archive. Roughly 2.5B objects back to 2005, monthly `.zst` dumps
  plus a working API. Post fields include `url` and `author`, so domains and multi-account
  structure are present.

The catch: **Arctic Shift carries no licence.** Its README describes collection and
modification in detail (`file_content_explanations.md` — content retrieved via the official
Reddit API, `was_deleted_later` and `removal_type` tracked in `_meta`) but grants no
redistribution rights. Reddit's Public Content Policy and Developer Terms reserve public
content to approved partners. Scraping it to build and commit derived data into this repository
would put unlicensed Reddit content into the repo.

Note also that Arctic Shift's coverage is strongest historically, which is where the 2009–2012
419-scam material lives — but that corpus remains unreleased regardless.

---

## Weak-label sources on Reddit

Considered and rejected:

- **r/scam, r/ScamAds, r/FraudAdvice, r/PersonalFinance** — these are overwhelmingly
  **category (b)**, victim narratives. The arXiv 2606.16874 paper (Zhang, Bottarelli, Hooper,
  Maple) is the rigorous version of this: 21,304 posts from scam subreddits 2023–2025,
  rule-filtered so a post survives only if flairs as a scam *and* comments affirmatively
  confirm it, then rule-labelled across four "rails" and 11 scam types, with 1,800
  LLM-annotated (Claude Sonnet 4.5) and human-validated on a 170-post sample. Well-built and
  **not released** — arXiv page lists no dataset link. Even if it were, it is (b): victims
  describing what happened to them. Rules tuned on it would learn the vocabulary of being
  conned, not of conning. **Flagged not usable.**
- **Moderator removals as a label.** Arctic Shift exposes `was_deleted_later` and
  `removal_type`. Tempting as free supervision, but biased to an unknown degree toward
  anything moderators consider *annoying* rather than fraudulent — rule violations, politics,
  meta-arguments. Not a fraud label, and unlabellable without re-annotation.
- **Known-scammer account lists.** No maintained, licensed public list found.

---

## Recommendation

**Primary: build the negative half yourself, from the target domain itself.** The Purino
corpus supplies authentic scammer text and genuine `page_id` multi-account structure, which
nothing else has. Pair it with **ordinary non-scam promotional text drawn from the same
subreddits the rules engine already reads** (job ads, SaaS launches, referral posts, crypto
project announcements). That produces a real ranking test — category (a) against category (c),
both real, from the same population.

**Secondary: Smishing-Dataset-IMC25 for phrase-list calibration only.** CC-BY-4.0 is
unambiguous, the labels are real, and `lure_principles` is a usable prior over
`urgency_language` and friends. Do not use it as a ranking benchmark — it is SMS, not Reddit,
and its links are stripped.

**What remains unobtainable, stated plainly:**

1. **Any labelled Reddit corpus of scammer-authored promotional posts.** None exists publicly.
   The one academic Reddit scam corpus that is methodologically serious (arXiv 2606.16874) is
   category (b), and is unreleased.
2. **Any labelled corpus with usable domains *and* account grouping.** Purino has accounts but
   no URLs. The 419 corpora have both but are unreleased. No candidate has both.
3. **Reddit data under terms this project can accept.** RFR needs a university sponsor;
   Pushshift is mods-only; Arctic Shift is unlicensed; PullPush is frozen. A commercial archive
   vendor (e.g. THINKPOL) is the only route to licensed bulk Reddit text, and it costs money and
   is outside a research-only pass.
4. **A legitimate external validation set that is not synthetic.** This is the honest
   conclusion: the near-null AUC against the Employment Scam Aegean Dataset was not caused by
   a bad proxy choice. It was caused by there being no good proxy to find. The remaining
   options are to buy licensed Reddit access, to apply for RFR under an institutional sponsor,
   or to accept that external validation here is bounded by what can be built in-repo.

The pattern across every candidate is the same: the labels are obtainable only where the
content is a Facebook or SMS ad that a platform already flagged, and the moment you have a
credible fraud label you lose the URLs, the account identity, or the licence.