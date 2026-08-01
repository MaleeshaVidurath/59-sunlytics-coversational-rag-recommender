# Personalized Ranker — Full Process Documentation

## Overview

The personalized ranker decides **which products to recommend to this particular user**,
and records **why each one was chosen** in plain language backed by real numbers.

Hard filters (colour, product type, price) narrow the catalogue from 41,794 articles to
a few hundred candidates. Every one of those candidates satisfies the user's request
equally well, so the filters cannot decide which 2 to actually show. That final choice is
what this module makes.

Two things happen at once:

- **Selection** — score every candidate, pick the best few for *this* user
- **Justification** — every scoring signal that fires produces a sentence containing a
  real statistic, and those sentences travel with the item to the product card, to the
  "why did you recommend this" answer, and to the hallucination checker

The justification is not written by the LLM. It is built from numbers by template code,
so it cannot be invented or exaggerated.

**Files:**

| File | Role |
|---|---|
| `text_rag/db/article_stats.py` | Offline builder for buying statistics (run once) |
| `text_rag/core/personalized_ranker.py` | The scorer and selector |
| `text_rag/core/evidence_assembler.py` | Calls the ranker, attaches reasons to items |
| `text_rag/core/response_generator.py` | Feeds reasons into the LLM prompts |
| `text_rag/core/hallucination_checker.py` | Verifies the response against the reasons |
| `memory/core/enrichment.py` | Supplies the user's purchase percentages and age |

---

## The Problem This Replaces

Previously, selection worked like this:

1. Qdrant returned ~20 semantically similar articles
2. PostgreSQL returned ~20 articles ordered by `avg_price ASC`
3. Each list was scored separately by `_rank_by_preferences()`
4. The two lists were **concatenated** — all Qdrant results, then PostgreSQL extras
5. The first 2 were shown

This had four concrete defects.

**1. The semantic score was thrown away.** Qdrant returned a `_score` per article that was
never used in ranking. Relevance survived only as list order.

**2. The two sources were never compared.** Because the merge was concatenation, a Qdrant
item scoring 0.0 always beat a PostgreSQL item scoring 2.5. Since Qdrant almost always
returned results, the PostgreSQL branch could never win — it was effectively dead code.

**3. A single preference could dominate everything.** Preference weights were added raw. A
`garment_group_name=Shorts` preference at weight 0.679 outranked every other consideration,
so a skirt that happened to be filed under the "Shorts" garment group would jump to the top
of a red-skirt search regardless of price, popularity, or fit.

**4. The explanation had no connection to the decision.** Scores were computed, used to
sort, then discarded. When the user asked "why did you recommend this", the system
re-derived a story from whichever preference had the highest weight — *even if that
preference had nothing to do with the item shown*. This is what produced answers like
*"You mentioned you're looking for a shorts garment, but since we don't have any shorts in
stock, we suggested trousers"* when the user had said nothing of the kind.

---

## Data Reality — Why the Statistics Need Smoothing

Before designing the buying-stat signals, the dataset was measured:

| Measurement | Value |
|---|---|
| Customers | 250 |
| Transactions | 185,037 |
| Articles | 41,794 |
| **Purchases per article** | median **3**, p90 = 10, max 615 |
| **Unique buyers per article** | median **2**, p90 = 7, max 50 |
| Articles with ≥30 transactions | **1%** |
| Customer ages | 22–70 |

**This is very sparse.** The median article was bought 3 times by 2 people. Ranking on raw
counts would be ranking on noise: an article bought 4 times is not meaningfully more
popular than one bought 3 times, but a naive sort would treat that as a real difference —
and would show the same handful of blockbusters to every single user.

Density at coarser levels was measured to find a fallback:

| Level | Groups | Median transactions | ≥30 txns |
|---|---|---|---|
| article | 41,794 | 3 | 1% |
| **product_type × colour** | 2,067 | 10 | 32% |
| **garment_group** | 21 | 5,794 | 100% |

This produced two design decisions that run through the whole module:

- **Popularity is shrunk** toward the average for its product type (Dirichlet smoothing)
- **Age distributions back off** to a denser level when an article has too few buyers

---

## Stage 1 — Offline Statistics Build

Run once, before the server starts:

```bash
python -m text_rag.db.article_stats --build
```

Reads the three CSVs and writes two PostgreSQL tables. Takes about 30–60 seconds.

### Table `article_stats` — one row per article (41,794 rows)

| Column | Meaning |
|---|---|
| `buy_count` | Total transactions for this article |
| `unique_buyers` | Distinct customers who bought it |
| `repeat_buyers`, `repeat_rate` | Customers who bought it more than once |
| `age_16_25`, `age_26_35`, `age_36_50`, `age_51_plus`, `age_known` | Buyer age histogram |
| `recent_count`, `prior_count`, `trend_share` | Sales in the last 90 days vs the 90 before |
| `popularity_lift` | Shrunk share of its product type, as a multiple of average |
| `popularity_pct` | Percentile rank of that lift within its product type |
| `product_code` | Used to collapse colour variants of the same product |

### Table `group_stats` — backoff aggregates (2,089 rows)

Same age histogram at three levels: `type_colour`, `garment_group`, and `global`.

### How popularity is smoothed

Raw counts are converted into a **lift relative to the product type average**:

```
shrunk_share = (buy_count + α) / (N_type + α × K_type)
popularity_lift = shrunk_share × K_type

where  α = 3.0 pseudo-counts (POPULARITY_ALPHA)
       N_type = total transactions across this product type
       K_type = number of articles in this product type
```

`popularity_lift = 1.0` means "average sales for this product type". `2.0` means twice the
average.

The α pseudo-counts are what kill the noise. With α = 3, an article with 4 purchases and
one with 3 purchases land within about 2% of each other — correctly treated as
indistinguishable. But an article with 615 purchases still rises far above both.

Measured result across the catalogue:

```
popularity_lift  p10 = 0.51   median = 0.81   p90 = 1.67   max = 83.25
```

Good spread, so the signal genuinely discriminates.

### How age support backs off

Only 32% of articles have 5 or more buyers with a known age. For the other 68%, the
article's own histogram is too thin to trust, so the ranker walks down until support is
sufficient:

```
article           (needs ≥5 buyers with known age)
   ↓ not enough
type_colour       (needs ≥20)     e.g. "Trousers|Black" — 8,770 transactions
   ↓ not enough
garment_group     (needs ≥20)     e.g. "Trousers"
   ↓ not enough
global            → treated as NO SIGNAL, because every article would score
                    identically and nothing would be learned
```

---

## Stage 2 — Candidate Retrieval

Unchanged from before, except that the two sources are now **pooled** rather than
concatenated in priority order.

```
Qdrant semantic search    → ~20 candidates, each carrying a `_score`
PostgreSQL filtered query → ~20 candidates, no score
                ↓
        pool, deduplicate by article_id
                ↓
        ~26–40 candidates, all satisfying the hard filters
```

Order in the pool no longer decides anything — every candidate is scored on one common
scale in Stage 3, so a PostgreSQL candidate can now beat a Qdrant one on merit.

If the pool is empty, the existing relaxation fallbacks still apply (drop price filters,
then fall back to product type only).

---

## Stage 3 — Scoring

Every candidate is scored by 13 components plus semantic relevance. Each component returns
a `Component(name, delta, reason)`:

- `delta` — how much it adds to the score
- `reason` — the user-facing sentence, or `None` if the signal is too weak or too generic
  to be worth showing

A component can contribute to the score while staying silent. For example `gender_fit`
adds 0.14 but emits no reason, because "it's womenswear" is not a persuasive justification.

### Group A — User fit (makes results differ per user)

| Component | Weight | Formula | Example reason |
|---|---|---|---|
| `colour_affinity` | 0.30 | shared formula below | "Black is your most-bought colour (34% of your purchases)" |
| `type_affinity` | 0.22 | shared formula | "Trousers is your 3rd most-bought product type (9% of your purchases)" |
| `garment_affinity` | 0.14 | shared formula | "Jersey Basic is your 2nd most-bought category (22% of your purchases)" |
| `pattern_affinity` | 0.12 | shared formula | "Solid is your most-bought pattern (41% of your purchases)" |
| `section_affinity` | 0.10 | shared formula | "From Divided Collection, one of your top sections (12% of purchases)" |
| `price_fit` | 0.20 | inside band → full weight; outside → decays with distance | "£6.98 is inside your usual £4.02-£9.07 spend range" |
| `gender_fit` | 0.14 | index_group matches inferred gender | *(silent)* |
| `session_preference` | 0.26 | `W × min(Σ matched weights, 1.5)` | "Matches what you told me you like: Black" |

The five attribute-affinity components share one formula:

```
delta = weight × (0.45 + 0.55 × share) × (1 − 0.10 × (rank − 1))

  share = that value's percentage of the user's purchase history
  rank  = its position in the user's top list (1 = most bought)
```

So magnitude comes from **how much** of the user's history that value represents, and a
small decay comes from **where** it ranks. A colour at 34% of purchases matters
considerably more than one at 6%, even though both are "top 5".

Note the cap on `session_preference`: matched conversational preferences are summed but
clamped at 1.5 before weighting. This is what stops a single 0.679 preference from
dominating the entire decision the way it did before.

### Group B — Buying statistics (social proof)

| Component | Weight | Formula | Example reason |
|---|---|---|---|
| `popularity` | 0.26 | `W × max(0, (pct − 50) / 50)` | "Popular choice: bought 39 times by 6 different customers, top 1% of Trousers" |
| `age_group_match` | 0.24 | `W × clamp(log(lift) / log(2.5), 0, 1)` | "71% of its buyers are aged 26-35, like you (1.9x the average)" |
| `repeat_rate` | 0.14 | `W × min(rate × 2, 1)` | "40% of its buyers bought it more than once" |
| `recency_trend` | 0.10 | `W × max(0, (share − 0.5) × 2)` | "Trending: 72% of its sales happened in the last 3 months" |

`age_group_match` computes a **lift**, not a raw share:

```
lift = (share of this article's buyers in the user's age bucket)
       ÷ (share of that age bucket across the whole catalogue)
```

Using the base rate as the denominator matters. If 38% of all customers are aged 26–35,
then an article with 40% buyers in that bracket is unremarkable — lift 1.05. One with 71%
is genuinely distinctive — lift 1.9. Without the base-rate division the ranker would just
keep recommending whatever the largest age group buys, to everyone.

### Thresholds for showing a reason

A component can score without speaking. Reasons appear only when they are worth a line on
the card:

| Component | Shows a reason when |
|---|---|
| `popularity` | percentile ≥ 60 **and** ≥ 3 unique buyers |
| `age_group_match` | lift ≥ 1.15 **and** the user's bucket is ≥ 25% of buyers **and** support is not `global` |
| `repeat_rate` | rate ≥ 0.25 **and** ≥ 4 unique buyers |
| `recency_trend` | recent share ≥ 0.65 **and** ≥ 3 recent sales |
| `price_fit` | the price is inside the band (outside the band scores but stays silent) |

The unique-buyer minimums exist because of the sparsity described earlier. "Bought 2 times
by 1 customer" is not evidence of popularity, so it never gets claimed.

### Semantic relevance and penalties

```
semantic_relevance  weight 0.35   = W × (article._score / best _score in this pool)
                                    PostgreSQL candidates get W × 0.5 (neutral midpoint)
dislike_penalty     weight 0.50   = TOTAL cap, not per-hit
```

Semantic relevance sits **outside** the user/stats blend at a fixed weight, so a heavily
personalised item can never outrank one that actually answers the question better.

The dislike penalty needs care, because **dislike lists drift**. Feedback on one item used
to record every attribute of that item as disliked, so after a few rejections the list
could name most of the catalogue. Four guards protect against that:

| Guard | Rule |
|---|---|
| **Requested** | Never penalise what the user asked for this turn. Matched loosely across columns, so `product_type_name='Shirt'` also protects `garment_group_name='Shirts'` |
| **Staple** | Never penalise a value holding ≥ `PROTECTED_HISTORY_PCT` (8%) of the user's purchases. Someone with 37.5% black purchases does not dislike black |
| **Demographic** | Never penalise `index_group_name`. See below |
| **Cap** | Total penalty is capped at 0.50 regardless of how many attributes hit |

The demographic guard exists because of a concrete failure. With `Ladieswear`, `Menswear`,
`Divided` and `Sport` all marked disliked, every adult garment took a penalty while
`Baby/Children` took none — so a 26-year-old asking for shirts received four children's
shirts. A rejection is never evidence that someone dislikes an entire demographic segment.
Positive gender alignment is already handled by `gender_fit`.

### Confidence-adaptive blending

Group A and Group B are normalised separately, then blended by how much purchase history
the user actually has:

```
confidence = min(1.0, total_purchases / 60)

user_fit_multiplier    = 0.30 + 0.70 × confidence
buying_stats_multiplier = 1.00 − 0.55 × confidence
```

| User | Purchases | User fit × | Buying stats × | Effect |
|---|---|---|---|---|
| Cold start | 0 | 0.30 | 1.00 | "Popular with people your age" carries the decision |
| Moderate | 30 | 0.65 | 0.73 | Balanced |
| Established | 740 | 1.00 | 0.45 | Their own history dominates |

A new user has no history to fit, so instead of falling back to an arbitrary catalogue
order, social proof takes over. As history accumulates the balance shifts automatically —
no separate cold-start code path.

---

## Stage 4 — Selection

```
1. Sort by total score, descending
2. Collapse variants: keep only the highest-scoring article per `product_code`
3. If quantity > 2, spread across colours (highest score wins within each colour)
4. Take the top N, preferring items that can explain themselves
```

Step 2 fixes a visible bug: a search for black trousers used to return `554477008` and
`554477010` — the same Victoria TRS product in two variants, with identical descriptions,
which reads as broken. Qdrant payloads do not carry `product_code`, so it is backfilled
from `article_stats` during scoring.

Step 4 handles a subtler one. Several components are deliberately silent — semantic
relevance, gender fit, and a price *outside* the user's band. An item can therefore win on
silent signals alone and arrive at the card with an empty "Why this for you" block, which
reads as broken. So when the next best candidate has no reason to show, a candidate that
does is promoted ahead of it — but only within `EXPLAINABILITY_MARGIN` (0.15), so a
clearly better item is never displaced just for being quiet.

If an item still ends up with no reasons, `reasons()` returns one honest fallback line
rather than an empty block:

> *"One of the closest matches to what you asked for in shirts"*

That is true — it was selected on semantic relevance — and it does not pretend a
personalised signal existed.

### The match badge

Raw scores are unbounded and go negative once penalties apply, so they can never go on a
card directly. `match_percent()` scales each item against the best score in the selected
set and returns `0–100`, or `None` when the whole set scored at or below zero — in which
case the badge is hidden entirely rather than showing something like "−13".

---

## Stage 5 — Where the Reasons Go

Each selected item carries three new fields into the evidence bundle:

```python
item["why"]             # top 3 reason strings, strongest contribution first
item["match_score"]     # raw total score — internal/audit only, never sent to the UI
item["match_percent"]   # 0-100 display figure, or None when nothing matched
item["score_breakdown"] # every component with its delta — full audit trail
```

These flow to **four** destinations.

### 1. The product card (frontend)

`chat.py` passes `why` and `match_percent` through; `App.jsx` renders them under a
"Why this for you" heading. Rendered verbatim — these are template strings, never LLM
output, so there is no hallucination risk in displaying them directly.

`match_score` deliberately stays server-side: it is unbounded and goes negative once
penalties apply, so only the clamped `match_percent` is ever sent.

```
Doris Twill TRS
Black · Trousers · £6.98
─────────────────────────────────────────────
WHY THIS FOR YOU
✓ £6.98 is inside your usual £4.02-£9.07 spend range
✓ Black is your most-bought colour (34% of your purchases)
✓ Popular choice: bought 39 times by 6 different customers, top 1% of Trousers
```

Reasons are **persisted with the recommendation**, not just sent to the live response.
`ItemInContext` carries `why` and `match_percent`, so reopening a past chat from the
sidebar restores the product cards with their justification intact. Before this, the
history endpoint returned only message text — reopening a chat lost the cards entirely.

The round trip is:

```
ranker → evidence items[].why
       → rag_pipeline store_response → recommendations collection (ItemInContext.why)
       → GET /api/sessions/{id}  joins turns to recommendations by recommendation_id
       → frontend selectSession() restores msg.items
```

Persisting them is safe precisely because they are template strings computed from
statistics: they do not go stale the way an LLM-written sentence would.

### 2. The recommendation prompt

Reasons are injected into the catalog_search prompt as the **only** permitted
justification:

```
Option 1: Doris Twill TRS | Type: Trousers | Colour: Black | Price: £6.98 | ...
    WHY THIS WAS SELECTED: £6.98 is inside your usual £4.02-£9.07 spend range
    WHY THIS WAS SELECTED: Black is your most-bought colour (34% of your purchases)
```

with the instruction: *"For the 'why' sentence, paraphrase ONLY the WHY THIS WAS SELECTED
lines for that item. Never invent a different reason."*

### 3. The "why did you recommend this" answer

This is the biggest change. When the user asks for an explanation, the assembler
**re-runs the ranker over exactly those articles** to recover the reasons. The scorer is
deterministic, so this reproduces the original decision rather than inventing a rationale.

Both explanation paths are covered:

- **Named product** ("why did you recommend June") → `ranking_reasons` in the bundle
- **No product named** ("why did you recommend this") → `why` per article

Globally top-weighted preferences are now used **only as a fallback** when no ranking
reason exists, because citing them alongside real reasons invites the model to claim a
match that never happened.

Before and after, on the same input:

> **Before:** "You mentioned you're looking for a shorts garment, but since we don't have
> any shorts in stock, we suggested trousers." — the user never mentioned shorts; the
> model was handed `Shorts (weight 0.679)` because it was the highest-weighted preference,
> and bridged the gap itself.

> **After:** "I recommended the Doris Twill TRS and the Victoria Pull-On TRS because they
> fit within your usual spending range of £4.02-£9.07. Both options are also in your
> most-bought colour, black, which makes up 34% of your purchases. Additionally, the Doris
> Twill TRS is a popular choice, having been bought 39 times by 6 different customers."

The `Shorts` preference is structurally unable to appear now: no trousers article belongs
to the Shorts garment group, so the component never fires and the reason never exists.

### 4. The hallucination checker

Reasons are added as verifiable facts with field `selection_reason`.

This closed a real hole. On the "why" path with no product named, the evidence bundle used
to contain `article: {}` and `confirmed_matches: []`, so the checker found **zero facts**,
fell through to *"No specific product facts available"*, and passed every response
unconditionally. It now checks 12–14 real facts on that same path.

---

## Worked Example

User `user_hist_01c19c0b`, 740 purchases, age 30, spends £4.02–£9.07, 34% of purchases
black. Asks: *"I need black trousor"*.

Candidate `663515005 Doris Twill TRS, Black, £6.98`:

Actual component breakdown from `score_breakdown`:

```
semantic_relevance    +0.1750
colour_affinity       +0.1911   "Black is your most-bought colour (34% of your purchases)"
type_affinity          0.0000   Trousers not in this user's top 5 types
garment_affinity       0.0000   no match
pattern_affinity      +0.0811   "Solid is your most-bought pattern (41% of your purchases)"
section_affinity      +0.0516   "From Divided Collection, one of your top sections (12% of purchases)"
price_fit             +0.2000   "£6.98 is inside your usual £4.02-£9.07 spend range"
gender_fit             0.0000   inferred_gender 'mixed' maps to no index groups
session_preference     0.0000   no conversational preferences matched
popularity            +0.1150   "Popular choice: bought 39 times by 6 different customers, top 1% of Trousers"
age_group_match       +0.0922   "82% of its buyers are aged 26-35, like you (2.2x the average)"
repeat_rate           +0.0630   "50% of its buyers bought it more than once"
recency_trend          0.0000   no recent sales in the trend window
dislike_penalty        0.0000   no disliked values matched
──────────────────────────────────────────────────────────────────────────────
TOTAL                  0.9690   → rank #1
```

Group A is multiplied by 1.00 and Group B by 0.45, since 740 purchases is full confidence.

Six components produced a reason, but only the **top 3 by contribution** are shown on the
card: `price_fit` (0.200), `colour_affinity` (0.191), `popularity` (0.115). The age match
at 0.092 ranks fourth and stays in `score_breakdown` for auditing.

### The same query, a different user

User B: age 58, spends £25–60, grey-heavy, 61 purchases.

```
USER A (30y, budget)                     USER B (58y, premium)
─────────────────────────────────        ─────────────────────────────────
Doris Twill TRS    Black  £6.98          King PU trouser        Black  £29.58
  £6.98 inside your £4.02-£9.07            £29.58 inside your £25.00-£60.00
  Black is your most-bought colour         50% of buyers aged 51+ (2.4x avg)
  Bought 39 times by 6 customers

Victoria Pull-On   Black  £8.64          Forget-me-not cropped  Black  £22.03
  £8.64 inside your £4.02-£9.07            57% of buyers aged 51+ (2.7x avg)
  71% of buyers aged 26-35 (1.9x avg)      Bought 14 times by 7 customers
```

Zero overlap, from an identical query.

---

## Setup and Operation

### One-time build

```bash
python -m text_rag.db.article_stats --build          # create and populate
python -m text_rag.db.article_stats --force          # rebuild after CSV changes
python -m text_rag.db.article_stats --show 554477008 # inspect one article
```

**If the tables are missing**, the ranker logs a warning and disables all buying-stat
signals. Recommendations still work — they fall back to user fit and semantic relevance
only — but lose popularity, age matching, repeat rate and trend. It never crashes.

```
[StatsCache] WARNING: could not load stats (...). Buying-stat signals disabled.
Run: python -m text_rag.db.article_stats --build
```

### Restart required after code changes

Uvicorn imports these modules at startup. Editing the files does **not** affect a running
server unless it was started with `--reload`. After a restart, a `catalog_search` request
logs:

```
[StatsCache] loaded 41794 article_stats, 2089 group_stats
[RANKER] scored 26 candidates, selected 2
  [RANKER] -> 663515005 Doris Twill TRS   score=0.969  semantic_relevance=+0.175 ...
     [WHY] 663515005: £6.98 is inside your usual £4.02-£9.07 spend range
```

If those lines are absent, the old code is still loaded.

### Performance

`article_stats` is loaded once into memory (~12 MB as dicts) on the first request. Scoring
is dict lookups and arithmetic — microseconds for 35 candidates. No extra database queries
and no extra LLM calls per turn.

---

## Failure Modes and Fallbacks

Every path degrades rather than fails. Nothing here can prevent a recommendation from
being returned.

| Situation | Behaviour |
|---|---|
| `article_stats` / `group_stats` tables missing | Warning logged, buying-stat signals disabled. Ranking continues on user fit + semantic relevance |
| User has no purchase history | Confidence 0 → the blend shifts to buying stats ("popular with people your age") |
| An article has no stats row | Its buying-stat components score 0 and stay silent; user-fit signals still apply |
| Too few buyers to judge age | Backs off to a denser group, then gives up entirely rather than using the uninformative global average |
| No component produced a reason | One honest fallback line, never a blank card |
| Every score ≤ 0 | `match_percent` returns `None` and the badge is hidden rather than showing a negative |
| Recommendation stored before reasons existed | Cards still render from the stored item fields, just without the reasons block |
| Items from M1 / M2 | No `why` key → `.get("why", [])` yields an empty list; cards render normally with no reasons block |

---

## Tuning

All constants are named at the top of the two modules.

**`personalized_ranker.py`**

| Constant | Default | Effect |
|---|---|---|
| `W_USER_FIT` | dict | Per-component weights for user-fit signals |
| `W_BUYING_STATS` | dict | Per-component weights for buying statistics |
| `W_SEMANTIC` | 0.35 | How much question-relevance outweighs personalisation |
| `W_DISLIKE_PENALTY` | 0.50 | Demotion per matched disliked value |
| `FULL_CONFIDENCE_PURCHASES` | 60 | Purchases before history is fully trusted |
| `AGE_LIFT_THRESHOLD` | 1.15 | Minimum lift before an age reason is shown |
| `POPULARITY_PCT_FLOOR` | 60.0 | Minimum percentile before a popularity reason is shown |

**`article_stats.py`**

| Constant | Default | Effect |
|---|---|---|
| `POPULARITY_ALPHA` | 3.0 | Higher = more shrinkage toward the type average |
| `MIN_AGE_SUPPORT_ARTICLE` | 5 | Buyers needed before an article's own ages are trusted |
| `MIN_AGE_SUPPORT_GROUP` | 20 | Buyers needed at a backoff level |
| `TREND_WINDOW_DAYS` | 90 | Window for the trending calculation |

Changing anything in `article_stats.py` requires a rebuild with `--force`.

---

## Changes to Existing Files

| File | Change |
|---|---|
| `evidence_assembler.py` | Removed `_rank_by_preferences()` and `_ensure_colour_diversity()`; pooled merge instead of priority concatenation; ranker call; reasons attached to items; explanation paths recompute reasons |
| `response_generator.py` | Catalog prompt carries per-item reasons; explanation prompts rewritten to cite fired reasons; top-weighted preferences demoted to fallback only |
| `hallucination_checker.py` | `articles` (all-items path) and `selection_reason` facts now checked; `_norm_ws` normalises hyphen spacing; response-level price gate accepts £values grounded in evidence |
| `enrichment.py` | `_get_purchase_hints()` returns percentage maps, age and age bucket; EXPLANATION_WHY payload carries purchase hints and preference boosts |
| `customer_profile_loader.py` | `get_purchase_history_hints()` kept in sync; added `age_bucket_for()` |
| `memory/models/schemas.py` | `ItemInContext` gained optional `why` and `match_percent` so reasons survive a reload |
| `rag_pipeline.py` | `store_response` persists `why`/`match_percent`; the cached-recommendation path carries them through too |
| `api/routers/sessions.py` | History endpoint joins turns to the recommendations collection and returns product cards with their reasons |
| `api/routers/chat.py` | Passes `why` and `match_percent` to the frontend |
| `frontend/src/App.jsx` | `ProductCard` renders the "Why this for you" block and the match badge; `selectSession()` restores items on reload |

All schema additions are optional fields with `None` defaults, so documents written before
this change still load. Nothing in the codebase referenced the two removed functions.

---

## Related Fix — Feedback Credit Assignment

The dislike-list drift described above had a root cause in the memory module, fixed in
`enrichment.py::_attribute_feedback()`.

**Before**, feedback on an item recorded **every one of its attributes at full strength**:

```python
item_entities = {
    "colour_group_name":  item_a.colour_group_name,   # Black
    "product_type_name":  item_a.product_type_name,   # Shirt
    "index_group_name":   item_a.index_group_name,    # Ladieswear
    "garment_group_name": item_a.garment_group_name,  # Shirts
}
update_preferences_from_entities(entities=item_entities, sentiment=sentiment_score)
```

Saying *"I don't like this"* about one black shirt therefore recorded four dislikes:
`Black`, `Shirt`, `Ladieswear`, `Shirts`. The user probably disliked the cut. A handful of
rejections and the list covered most of the catalogue. The same bug on the positive side
is why one user's top preference was `garment_group_name=Shorts` at weight 0.679.

**After**, three rules decide which attributes deserve the credit or blame:

| Rule | Reasoning |
|---|---|
| **Contrastive** | An attribute shared by *every* item shown cannot explain why one was singled out. Four shirts shown, one rejected → "Shirt" explains nothing. If only that one was Red, Red is a genuine candidate |
| **Requested** | Never blame an attribute the user explicitly asked for. Matched loosely across columns, as in the ranker |
| **Demographic** | `index_group_name` is never attributable |

Whatever survives then **splits the sentiment** between the remaining attributes, since we
do not know which one actually caused the reaction — no single attribute absorbs the full
strength.

Worked example — four shirts shown, the black one rejected:

```
BEFORE:  Black −0.8, Shirt −0.8, Ladieswear −0.8, Shirts −0.8
AFTER:   Black −0.8
         dropped: index_group_name (demographic)
                  product_type_name (user asked for it)
                  garment_group_name (shared by all 4 items shown)
```

Only the attribute that actually distinguished the rejected item survives.

**Note on existing data:** these rules govern *new* feedback. Profiles polluted before the
fix still contain the old entries — but the ranker's four penalty guards neutralise them
at the point of use, so no migration is required.

### Two incidental fixes

Both were pre-existing false positives in the hallucination checker that the new reasons
exposed:

**Hyphen spacing in product names.** The catalogue contains `Victoria Pull- On TRS` with a
stray space. Every LLM writes `Victoria Pull-On TRS`, which the name gate read as a
swapped product and flagged. `_norm_ws()` now tightens spacing around hyphens on both
sides of the comparison.

**Spend ranges read as prices.** Reason strings quote the user's budget
(`"£4.59 is inside your usual £4.02-£9.07 range"`). The price gate scanned for any £value
and treated `£4.02` as a wrong item price. Response-level checks now allow £values that
appear anywhere in the evidence. The locked-sentence branch stays strict, so genuine
cross-item price swaps are still caught.
