# Adaptive RAG Trigger — Evaluation v2 Methodology

This document states every decision made in the analysis, so each number in
`statistical_report_v2.txt` can be traced to a rule rather than a judgement call.

**Reproduce:** from the repo root —

```
venv/Scripts/python.exe m3_implementation/test_result/adaptive_rag_result_v2/adaptive_rag_analysis_v2.py
```

The script auto-locates `latency_log.csv` and writes every output into its own
directory. It runs locally; no Colab upload step.

---

## 1. What this evaluation has to prove

The v1 evaluation proved one thing: **latency differs by tier**. That is
necessary but not sufficient. A trigger that assigned tiers *at random* would
also produce three different latency distributions. To claim the adaptive
trigger works, four things must hold together:

| Claim | Analysis |
|---|---|
| Inputs actually spread across tiers, rather than collapsing into FULL | E1 |
| Each input reaches the tier the design intends | **E2** |
| The tiers really do cost different amounts | E3 |
| The cheaper tiers save meaningful time | **E4** |
| …without costing answer quality | **E5** |
| …and the saving comes from retrieval, not shorter LLM output | E6 |
| …and none of it is an artefact of data cleaning | **E7** |

E2, E4, E5 and E7 are new in v2.

---

## 2. Data

- **Source:** `latency_log.csv` (repo root), appended one row per turn by
  `m3_implementation/api/latency_logger.py`.
- **Raw rows:** 333 across 66 sessions, 2026-05-29 → 2026-08-03.
- **Schema:** documented in `m3_implementation/api/LATENCY_LOGGING.md`.

### Normalisation

The `NO` tier writes `sub_tier` as an em-dash, which does not round-trip
cleanly through every console/CSV encoding. It is normalised to the ASCII
sentinel `NONE` at load. No other field is altered.

---

## 3. Ground truth for routing correctness (E2)

Taken from the design spec in the source itself, not invented for the
evaluation — `memory/core/context_sufficiency_evaluator.py`:

Lines 14-17 give the canonical mapping:

```
tier = NO       CHITCHAT / FEEDBACK
tier = FULL     INITIAL_REQUEST / REFINEMENT
tier = PARTIAL  ATTRIBUTE_QUESTION / EXPLANATION_WHY / COMPARISON / SELECTION_REFERENCE
```

Lines 213-217 add that **FEEDBACK is sentiment-conditional**, classified by
Twitter-RoBERTa (Barbieri et al., EMNLP 2020):

```
positive / neutral            -> tier = NO    (user satisfied; acknowledge)
negative + items in context   -> tier = FULL  (user implicitly wants alternatives)
```

So FEEDBACK has **two** correct answers depending on sentiment, and scoring it
against a single expected tier would understate accuracy by ~50% on that label.

**Limitation.** The latency log stores the message text but not the sentiment
score the CSE actually computed. The analysis recovers the sign with a negation
lexicon (`don't like`, `dislike`, `hate`, …). On the observed data this is
unambiguous — all 15 negative-phrased FEEDBACK turns routed to
`FULL/EXCLUSIONS` and all positives to `NO` — but it is a reconstruction, not
the classifier's own output. Logging the sentiment label directly would remove
this inference; see §7.

---

## 4. Exclusion policy

The v1 analysis removed 23 of 122 rows (19%) via rules R1–R8. Several of those
rules are **outcome-dependent** — they delete rows *because they are slow*, and
they delete them from the fast tiers:

- **R5 / R6** drop slow PARTIAL rows, which inflates the PARTIAL-vs-FULL gap
  the analysis is trying to demonstrate.
- **R8** drops COMPARISON from PARTIAL/RECENT with the stated reason that it
  "made RECENT look slower than SESSION" — filtering chosen after seeing the
  result it produces.
- **R3** drops INITIAL_REQUEST→PARTIAL rows as artefacts. Those rows are
  *routing-error evidence*; discarding them removes data that E2 needs.

v2 replaces this with two rules, both defined without reference to the outcome:

### 4.1 Gate-blocked turns — 5 rows

`tier == NO` **and** `rag_ms < 50` **and** label is a follow-up intent
(`COMPARISON`, `REFINEMENT`, `EXPLANATION_WHY`, `SELECTION_REFERENCE`,
`ATTRIBUTE_QUESTION`).

These turns were refused by the session-context gate *before* retrieval was
ever attempted, so they are not observations of a routing decision at all. They
are logged as `NO` only because no retrieval happened. Excluding them is a
correctness fix, not a convenience.

### 4.2 Statistical outliers — 46 rows

Tukey upper fence, `total_ms > Q3 + 1.5 × IQR`, computed **within each tier**.

Per-tier and symmetric, so it cannot preferentially thin one tier — unlike a
single global millisecond threshold (v1's R1 `total_ms <= 15000`), which by
construction removes more rows from the slowest tier.

**These are statistical outliers, not verified cold starts.** Ollama model-load
spikes (one observed at 208 s) are the known cause of the extreme tail, but the
log carries no server-restart marker, so not every excluded row can be
attributed to that cause. This is why E7 exists.

### 4.3 E5 deliberately ignores both exclusions

**The quality analysis (E5) runs on the raw 333 rows, not the filtered set.**

A hallucination triggers a regeneration attempt, which *adds* latency. Defective
turns are therefore systematically slower than clean ones — median 9114 ms vs
3590 ms in this dataset. Any outlier filter consequently removes defects at a
much higher rate than clean turns: the Tukey fence deletes 6 of the 10 defective
turns here, but only ~13% of clean ones.

Running E5 on the filtered set would have reported 100% clean responses for both
NO and PARTIAL — an artefact of the filter, not a property of the system. Quality
must be measured over every turn the system actually served, so E5 uses raw data
and reports the true rates (NO 100%, PARTIAL 96.3%, FULL 96.5%).

This is a general hazard whenever latency filtering meets a quality metric, and
it is worth stating explicitly in the write-up.

### 4.4 Why the exclusions are not load-bearing

E7 re-runs the entire analysis under three policies — **raw** (nothing
removed), **minimal** (the above), and **strict** (v1's R1–R8). The tier
ordering, the significance, the routing accuracy and the share of traffic
avoiding FULL all hold in every one. The primary results are quoted from
*minimal*; the *raw* column is the honest fallback if a reviewer rejects any
exclusion at all.

---

## 5. Statistical choices

| Choice | Reason |
|---|---|
| **Median reported first** | Latency is heavily right-skewed (FULL: mean 5776 ms vs median 4739 ms). The mean is dragged by model-load spikes; the median is what a user experiences. Means are still reported. |
| **Kruskal-Wallis** | Omnibus test across three tiers, no normality assumption. |
| **Mann-Whitney U, one-tailed** | The hypothesis is directional (`NO < PARTIAL < FULL`), so a one-tailed test is appropriate. RECENT vs SESSION has no directional hypothesis and uses two-tailed. |
| **Rank-biserial r** added | Cohen's *d* assumes roughly normal, equal-variance data and **overstates** the effect on skewed latency. Rank-biserial is computed from the same ranks the U test uses. Both are reported so v1's numbers remain comparable. |
| **Chi-square on quality** | Tests whether defect rate is independent of tier (E5). |
| **Normalised Shannon entropy** | One number for "how evenly is traffic spread across tiers" — 1.0 is perfectly even, 0 is everything in one tier. |

---

## 6. The counterfactual baseline (E4)

The question is: *how long would these same 282 turns have taken with no
adaptive trigger — i.e. if every turn ran full retrieval?*

Per turn:

```
turn already ran FULL   ->  baseline = its own observed total_ms
turn ran NO or PARTIAL  ->  baseline = its own memory_ms + median(rag_ms | FULL)
```

Two deliberate properties:

1. **Each turn keeps its own memory-pipeline cost.** Only the retrieval
   component is substituted, because retrieval is the only thing the trigger
   controls. Swapping in a whole median FULL turn would credit the trigger for
   memory-pipeline differences it does not cause.
2. **FULL turns are charged their observed time, not the median.** Under the
   non-adaptive baseline a FULL turn behaves identically, so its saving must be
   exactly zero. Substituting the median there would charge it a below-median
   cost — latency is right-skewed — and manufacture a spurious *negative*
   saving for the FULL tier.

**Reporting.** The aggregate (18.4%) is diluted by construction: 43% of turns
are FULL and contribute zero. The per-tier split and the non-FULL figure
(38.6%) are reported alongside it and should always be quoted together — the
aggregate alone understates the mechanism, the non-FULL figure alone overstates
its reach.

---

## 6b. Baseline comparison (E8)

The Final Report's §7.2.1 evaluates the trigger with **no experimental baseline** —
unlike every other novelty in the report (the hallucination guard has five, the
contradiction detector four, Module 2 has ablations for all five novelties,
Module 3 has Vibe-only / History-only / Hybrid). E8 closes that gap by testing the
two claims made in §2.1.4 directly.

### E8a — latency vs retrieval-policy baselines

| Policy | Stands for | Construction |
|---|---|---|
| **B1** non-adaptive | COMPASS [1], RA-Rec [2], ChatCRS [3] — "retrieval is mandatory on every turn" | every turn pays full retrieval |
| **B2** binary gate | RAGate [7] — retrieve/don't-retrieve, no middle tier | NO turns unchanged; every other turn pays full retrieval |
| **Ours** | 3-way NO / PARTIAL / FULL | observed |

B2 is the important one: it is constructed exactly as §2.1.3 describes RAGate's
limitation — a binary decision "cannot distinguish a full catalogue search from a
partial metadata search" — so a turn needing any context must pay full retrieval.
**The B2 → ours difference is therefore the measured value of the PARTIAL tier and
nothing else**, which is precisely Gap 2.

Same conservative rule as §6: a turn that already ran FULL is charged its own
observed time under every policy, so it contributes zero saving by construction.

### E8b — routing correctness vs decision-policy baselines

Latency baselines alone cannot establish the trigger works, because **a random
router also produces three different latency distributions**. E8b tests whether the
routing *decisions* are right:

| Router | Construction |
|---|---|
| Random (stratified) | assigns tiers at random matching our marginals; expected accuracy = Σ q² |
| Majority class | always predicts the most frequent tier |
| Intent label only | the DistilBERT label mapped through the base strategy table (report §3.2.1) with **no** CSE |
| Ours | DistilBERT + CSE, as deployed |

**The intent-only row is a ceiling, not a like-for-like run.** It is scored on the
already-logged label, so it inherits none of the classifier's own errors, while our
row is the live end-to-end system. Comparing the two overall percentages therefore
*understates* the CSE, and the report should not quote that delta alone. The
structural difference is visible on FEEDBACK turns, where the CSE is what decides
the outcome: without it, every negative-feedback turn routes to NO — the user says
"I don't like them" and receives no new items. That is a functional failure, not an
accuracy delta, and it is reported that way.

### Limitation

B1 and B2 are **estimated** from observed per-tier costs, not measured by disabling
the trigger and re-running. This is stated on the figure itself. Converting them to
measured A/B runs is limitation #4 below.

## 7. Known limitations

1. **Sentiment is reconstructed, not logged** (§3). Adding the CSE's sentiment
   label to the latency log would make E2 exact rather than inferred.
2. **`total_ms` includes LLM generation time**, which scales with output length.
   A FULL turn describing 4 items generates more text than a CHITCHAT reply, so
   part of the tier gap is generation, not retrieval. E6 addresses this by
   decomposing `memory_ms` vs `rag_ms` and showing the tier effect concentrates
   in retrieval (6.9× spread) far more than in the memory pipeline (3.1×), but
   the log does not separate generation from retrieval *within* `rag_ms`. A
   dedicated `generation_ms` column would close this gap fully.
3. **Single-machine, single-user timings.** No concurrency, so these are
   latencies under no load, not throughput under load.
4. **Observational, not a controlled A/B.** The all-FULL baseline in E4 is
   estimated from observed FULL turns rather than measured by actually
   disabling the trigger. Running the same scripted sessions with the trigger
   forced to FULL would turn E4 from an estimate into a measurement, and is the
   single highest-value addition to this evaluation.
5. **Class imbalance.** `PARTIAL/SESSION` (n=23) is the thinnest cell; its
   comparison against `PARTIAL/RECENT` is correspondingly the least powered
   result here.

---

## 8. Output files

| File | Contents |
|---|---|
| `statistical_report_v2.txt` | Full E1–E7 report, plain text |
| `summary_stats_v2.csv` | Per-sub-tier latency table (median, mean, SD, IQR, min, max) |
| `routing_accuracy.csv` | Per-label routing accuracy (E2) |
| `routing_matrix.csv` | label × tier confusion matrix |
| `quality_by_tier.csv` | Response-status breakdown per tier (E5) |
| `memory_vs_rag.csv` | Median memory/RAG split per sub-tier (E6) |
| `sensitivity.csv` | All three filtering policies side by side (E7) |
| `baseline_latency.csv` | B1 / B2 / ours total time (E8a) |
| `baseline_routing.csv` | Random / majority / intent-only / ours accuracy (E8b) |
| `chart1..chart12 *.png` | Figures, 160 dpi |

### Figure colour encoding

Tiers are **ordinal** (increasing retrieval cost), so they use a single-hue blue
ramp light→dark rather than categorical hues — the ramp itself encodes "more
retrieval". Both the 3-step and 5-step ramps were checked with the dataviz
skill's `validate_palette.js` under `--ordinal` and pass monotone lightness,
adjacent-ΔL ≥ 0.06, light-end contrast and single-hue. The memory-vs-RAG chart
uses the validated categorical blue/orange pair (identity, not magnitude).
Correct/misrouted use the reserved status palette and always carry a text label,
never colour alone.
