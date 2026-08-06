# What each chart shows — plain explanation

All charts use the same data: **333 turns logged from the live system**, of which
**282** are used for analysis (5 gate-blocked turns and 46 statistical outliers
removed — see `METHODOLOGY.md` §4).

---

## Read these 4 first

If you only have time for a few, these are the ones that carry your argument:

| Chart | Proves |
|---|---|
| **chart11** | Your system beats both baselines (Gap 1 and Gap 2) |
| **chart12** | Your routing is *correct*, not just varied |
| **chart3** | 95.7% routing accuracy, broken down per intent |
| **chart8** | Going faster did **not** make answers worse |

The other 8 charts are supporting evidence.

---

# The 12 charts, one by one

---

## chart1_tier_distribution.png
### Do user inputs spread across the three tiers?

**What you see:** three horizontal bars — one per tier — showing what share of all
turns went to each.

**The numbers:**
- NO — 45 turns (16.0%)
- PARTIAL — 115 turns (40.8%)
- FULL — 122 turns (43.3%)

**What it proves:** **56.7% of turns never do a full catalogue search.** This is the
basic claim of your whole module. If everything went to FULL, your adaptive trigger
would be pointless. It doesn't — traffic genuinely splits three ways.

The "entropy 0.929" at the bottom is one number summarising how even the split is
(1.0 = perfectly even, 0 = everything in one tier). 0.929 is a very even spread.

**If asked "why does this matter?"** → Because the three tiers are only useful if
real users actually land in all of them. This shows they do.

---

## chart2_routing_heatmap.png
### Where does each type of user message get sent?

**What you see:** a grid. Rows = the 8 intents (CHITCHAT, COMPARISON, …).
Columns = the 3 tiers. Darker blue = higher percentage. Each cell shows a % and a count.

**What it proves:** each intent concentrates in **one** tier — the dark square in each
row. That is the trigger behaving predictably, not randomly.

- CHITCHAT → 100% NO
- EXPLANATION_WHY → 100% PARTIAL
- REFINEMENT → 100% FULL
- SELECTION_REFERENCE → 98% PARTIAL

**The one row that looks "wrong" is FEEDBACK (52% NO / 48% FULL).** That split is
**correct and intentional**. Your CSE routes feedback by *sentiment*:
- "I like them" (positive) → NO — nothing to search for
- "I don't like them" (negative) → FULL — go find alternatives

This is documented in your own code at
`context_sufficiency_evaluator.py:213-217`. The chart footnote says this, so nobody
misreads it as an error.

**If asked "why is FEEDBACK split?"** → It's sentiment-driven by design. Both cells
are correct behaviour. This is actually a *stronger* result than a clean 1:1 mapping,
because it shows the trigger reads meaning, not just the label.

---

## chart3_routing_accuracy.png
### Does the trigger send each message to the RIGHT tier?

**What you see:** one bar per intent. **Green = routed as designed. Red = misrouted.**
The % on the right is that intent's accuracy.

**The numbers — overall 95.7% (270 of 282 turns correct):**

| Intent | Correct |
|---|---|
| CHITCHAT | 31/31 — 100% |
| EXPLANATION_WHY | 19/19 — 100% |
| REFINEMENT | 18/18 — 100% |
| SELECTION_REFERENCE | 51/52 — 98% |
| FEEDBACK | 26/27 — 96% |
| COMPARISON | 14/15 — 93% |
| INITIAL_REQUEST | 87/94 — 93% |
| ATTRIBUTE_QUESTION | 24/26 — 92% |

**What it proves:** this is **the single most important result you have.** Your
submitted report proves the classifier is accurate (97.5%) and that tiers have
different latency — but neither of those proves the *routing decisions* are right.
This does.

"Correct" is measured against your own design spec in
`context_sufficiency_evaluator.py:14-17` — not a standard invented for the evaluation.

**If asked "what are the 12 errors?"** → They're all listed individually in
`statistical_report_v2.txt` (lines 44–56). Seven are INITIAL_REQUEST going to PARTIAL
instead of FULL — repeat requests inside one session, where the system reused cached
results. Arguably correct behaviour, counted as errors to stay conservative.

---

## chart4_latency_box_3tier.png
### How long does each tier take?

**What you see:** a box plot. Each box = one tier.
- The **line inside the box** = median (the typical turn)
- The **white diamond** = mean (average)
- The **box** = the middle 50% of turns
- **Dots above** = unusually slow turns
- The `***` brackets on top = the difference is statistically significant

**The numbers (median):**
- NO — 835 ms
- PARTIAL — 2,886 ms
- FULL — 4,739 ms

**What it proves:** the three tiers really do cost different amounts, and the
difference is not luck. Kruskal-Wallis p = 2.4 × 10⁻³⁹ — that's a very, very small
number, meaning the chance of seeing this by accident is essentially zero.

**Why median and not average?** Latency is "right-skewed" — a few very slow turns
(model loading) drag the average up. FULL has median 4,739 ms but mean 5,776 ms. The
median is what a real user actually experiences.

---

## chart5_latency_box_5subtier.png
### Same as chart4, but split into 5 sub-tiers

**What you see:** the same box plot, but PARTIAL and FULL are broken into their
sub-types.

**The numbers (median):**
- NO — 835 ms
- PARTIAL/RECENT — 2,837 ms
- PARTIAL/SESSION — 3,184 ms
- FULL/STANDARD — 4,995 ms
- FULL/EXCLUSIONS — 4,514 ms

**What it proves:** the ordering holds at the finer level too. RECENT and SESSION
are **not** significantly different (p = 0.617) — meaning both memory paths answer
follow-ups equally fast, without a new search. That matches what your report already
says.

**Caution:** PARTIAL/SESSION only has 23 turns — the smallest group. It's your
weakest number. If you collect more data, target this one.

---

## chart6_counterfactual.png
### How much time did the adaptive trigger actually save?

**What you see:** two bars. Grey = what the system *would* have cost with no adaptive
trigger (every turn does full retrieval). Blue = what it actually cost. Green arrow =
the saving.

**The numbers:**
- Non-adaptive baseline: 1,349 s
- Adaptive (actual): 1,101 s
- **Saved: 18.4%**

**Important — read the footnote:** FULL turns save nothing by definition (they'd
behave the same either way). So the 18.4% is *diluted*. On the 160 non-FULL turns,
the saving is **38.6%**.

**Always quote both numbers together.** 18.4% alone understates the mechanism;
38.6% alone overstates how much traffic it touches.

**Honest limitation:** this is an *estimate*. Nobody actually turned the trigger off
and re-ran. Say "estimated", not "measured".

---

## chart7_memory_vs_rag.png
### Where does the time actually go?

**What you see:** stacked bars per sub-tier. **Blue = memory pipeline** (classification,
session lookup). **Orange = RAG retrieval** (the actual search).

**The numbers (median):**
- Memory time across tiers: 376 → 1,149 ms (**3.1× spread**)
- RAG time across tiers: 514 → 3,541 ms (**6.9× spread**)

**What it proves:** the saving comes from **skipping retrieval**, not from anything
else. The memory pipeline costs roughly the same in every tier; retrieval is what
changes.

**Why this chart exists:** it defends against a specific objection — *"maybe FULL is
slower just because the LLM writes longer answers, not because retrieval is slower."*
This shows retrieval varies more than twice as much as the rest of the pipeline.

---

## chart8_quality_by_tier.png
### Did going faster make the answers worse?

**What you see:** one bar per tier showing the % of responses with **no** hallucination
and **no** contradiction.

**The numbers:**
- NO — 100.0% clean (55 turns)
- PARTIAL — 96.3% clean (135 turns)
- FULL — 96.5% clean (143 turns)
- Chi-square p = 0.359 → **no significant difference between tiers**

**What it proves:** the cheap tiers are **not** lower quality. You got faster without
getting worse. This closes the most obvious attack on your whole module.

**Note this chart uses all 333 raw turns, not 282.** That is deliberate. A
hallucination triggers a regeneration, which makes that turn *slower* — so an outlier
filter would delete defective turns more often than clean ones and produce a fake
100% score. Quality must be measured over every turn the system really served.

**If asked "why a different sample size here?"** → Exactly the answer above. It shows
you thought about it rather than missed it.

---

## chart9_sensitivity.png
### Do the results survive different data cleaning?

**What you see:** three groups of three bars. Each group = one data-cleaning policy.
Within each group, the three tiers.

**The numbers:**

| Policy | n | NO | PARTIAL | FULL | Routing acc. |
|---|---|---|---|---|---|
| Raw (nothing removed) | 333 | 885 | 3,139 | 5,107 | 94.0% |
| Minimal (primary) | 282 | 835 | 2,886 | 4,739 | 95.7% |
| Strict (old v1 rules) | 251 | 885 | 2,877 | 4,553 | 96.0% |

**What it proves:** **NO < PARTIAL < FULL in every single case.** Your conclusion does
not depend on how you clean the data.

**Why this chart matters most for defence:** your v1 analysis removed 19% of rows
using rules that deleted *slow* rows from the *fast* tiers — which inflates the very
effect you're proving. An examiner could call that p-hacking. This chart removes the
attack completely: even with **zero** rows removed, the result holds.

---

## chart10_median_iqr_savings.png
### The "% faster" chart (updated version of your old Figure 35)

**What you see:** the same style as your submitted Figure 35, rebuilt with the new
data. Bars = median. Error bars = the middle 50% of turns (IQR).

**The numbers:**
- NO — 835 ms → **82% faster than FULL**
- PARTIAL — 2,886 ms → **39% faster than FULL**
- FULL — 4,739 ms

**What changed from your report:** your Figure 35 used n = 23/41/35 (99 turns). This
uses n = 45/115/122 (**282 turns**) — nearly 3× the evidence, same conclusion.

**Why median instead of mean ± SD:** at FULL the standard deviation is ±3,163 ms. If
you draw mean − SD, the bar dips down *into* PARTIAL's range and it *looks* like the
tiers overlap — when the statistical tests say they clearly don't (p < 10⁻²³). Median
+ IQR shows the true separation. The mean ± SD numbers are still printed in the
footnote if you want to quote them.

---

## chart11_baseline_latency.png ⭐
### Does your system beat the alternatives from the literature?

**What you see:** three bars, plus two green brackets marking your two claimed gaps.

**The numbers:**

| Policy | Represents | Total |
|---|---|---|
| **B1** non-adaptive | COMPASS, RA-Rec, ChatCRS — "always retrieve" | 1,349 s |
| **B2** binary gate | RAGate — retrieve or not, no middle tier | 1,207 s |
| **Ours** 3-way | NO / PARTIAL / FULL | **1,101 s** |

- **GAP 1 — 18.4% faster** than always-retrieve
- **GAP 2 — 8.8% faster** than a binary gate = **925 ms saved per PARTIAL turn**

**What it proves — this is your novelty, with a number on it.** Your Chapter 2 claims
two gaps. This tests both directly.

**B2 is the important bar.** Your §2.1.3 says RAGate's weakness is that a binary
decision "cannot distinguish a full catalogue search from a partial metadata search."
So B2 is built exactly that way: any turn needing context must pay full retrieval.
**The gap between B2 and yours is the value of inventing the PARTIAL tier — and
nothing else.** That is your headline contribution, isolated and measured.

**Why this chart had to be made:** your submitted §7.2.1 has **zero** experimental
baselines. Every other novelty in your report has them (hallucination guard: 5,
contradiction detector: 4, Module 2: ablations for all 5, Module 3: 3). This was the
gap.

**Honest limitation (printed on the chart):** B1 and B2 are **estimated** by
substituting the median FULL retrieval cost — they are not measured A/B runs. Say
"estimated".

---

## chart12_baseline_routing.png ⭐
### Is your routing actually *correct*, or just *different*?

**What you see:** two panels.
**Left** = routing accuracy of four different routers.
**Right** = the same comparison, but only on FEEDBACK turns.

**Left panel numbers:**

| Router | Accuracy |
|---|---|
| Random (stratified) | 37.9% |
| Majority class (always FULL) | 44.7% |
| Intent label only (no CSE) | 95.0% |
| **Ours (DistilBERT + CSE)** | **95.7%** |

**What it proves — this answers the killer objection.** Someone can look at your
latency chart and say: *"A random router would also produce three different latency
distributions. You've shown the tiers cost different amounts, not that your routing
is right."*

This chart is the answer. A random router **matches your latency profile exactly** but
routes correctly only 37.9% of the time. **Latency alone proves nothing — routing
accuracy is what proves the trigger works.**

**Right panel — why it exists:** on the left, "intent only" looks almost as good as
you (95.0% vs 95.7%, only +0.7). That comparison is **unfair to you**, and you should
say so before anyone else does:

> The intent-only row is a **ceiling**, not a real system. It is scored using the
> already-logged label, so it carries none of the classifier's own mistakes. Your row
> is the live end-to-end system, mistakes included.

The real difference shows on FEEDBACK turns:

- Intent only (no CSE): **48.1%**
- Ours (with CSE): **96.3%**

Without the CSE, all **14 negative-feedback turns route to NO**. The user says
*"I don't like them"* and gets **no new items at all**. That is a **broken feature**,
not a 0.7-point accuracy difference.

**If asked "what does the CSE add?"** → Point at the right panel. 48% → 96% on the
turns where it decides the outcome.

---

# Quick answers to likely questions

**"Is 282 turns enough?"**
Yes. 62 sessions, 8 intents, all 5 sub-tiers populated. Your submitted report used 99.

**"Why did you remove 51 rows?"**
5 were gate-blocked turns (refused before retrieval — not routing observations at
all). 46 were statistical outliers using the Tukey fence *computed separately within
each tier*, so it cannot favour one tier. And chart9 shows the conclusion holds even
with nothing removed.

**"Are the baselines real experiments?"**
No — B1 and B2 are estimated from observed per-tier costs. This is stated on the chart
and in `METHODOLOGY.md`. The stronger version would be re-running the same
conversations with the trigger forced to FULL.

**"What's your weakest result?"**
PARTIAL/SESSION with n = 23, and the fact that the baselines are estimated rather than
measured. Both are written down in `METHODOLOGY.md` §7 as known limitations.

---

# File map

| Chart | Section | Question it answers |
|---|---|---|
| chart1 | E1 | Do inputs spread across tiers? |
| chart2 | E2 | Where does each intent go? |
| chart3 | E2 | Is the routing correct? |
| chart4 | E3 | Do tiers differ in speed? |
| chart5 | E3 | Same, at sub-tier level |
| chart6 | E4 | How much time was saved? |
| chart7 | E6 | Where does the time go? |
| chart8 | E5 | Did quality suffer? |
| chart9 | E7 | Do results survive different cleaning? |
| chart10 | E3 | "% faster" summary |
| **chart11** | **E8a** | **Do we beat the literature baselines?** |
| **chart12** | **E8b** | **Is the routing correct, not just different?** |

Full numbers: `statistical_report_v2.txt`
Method and limitations: `METHODOLOGY.md`
Re-run everything:
`venv/Scripts/python.exe m3_implementation/test_result/adaptive_rag_result_v2/adaptive_rag_analysis_v2.py`
