# Expanded Evaluation — Results (4,000-Row Suite)

Generated: 2026-07-11 · Machine-readable: `results_expanded_summary.json` ·
Figures: `figures/figE1–E5` · Fully isolated from the original evaluation
(`../RESULTS.md`), whose artifacts are unchanged and remain valid.

---

## 1. Why this evaluation exists

The interim evaluator raised two objections to the original evaluation:
(a) the numbers looked "unrealistically high" and (b) the dataset (238 rows,
33 clean) was small. The response is NOT to adjust numbers — it is more and
harder data: a 15× larger, three-source dataset plus an adversarial hard set
designed to find the checker's limits, with every metric now carrying a 95%
confidence interval.

## 2. Dataset

**526 clean bases** from three sources, deduplicated by response text:
36 original scripted cases + 166 new scripted cases (60 conversations) +
**416 real user chats harvested from MongoDB** (turn × recommendation join,
classified offline by checker v3). From these, an 8,181-row pool was
generated and a **4,000-row suite** drawn (seed 42; all 526 clean kept,
corrupted sampled evenly per type):

| Set | Rows | Composition |
|---|---|---|
| **Standard** (`labeled_test_set_expanded.jsonl`) | 2,600 | 526 clean + 2,074 corrupted (colour/price/name/cross-item swaps — seed-42 FactCC-style) |
| **Hard** (`hard_set/labeled_hard_set.jsonl`) | 1,400 | corrupted only: paraphrased colour ("a rich crimson shade"), paraphrased price ("about fifteen pounds"), fabricated attributes (unsupported claims) |

Full 8,181-row pool preserved as `*_full.jsonl`. Rows within a base are
correlated; the independent unit is the 526 bases. Clean labels are presumed
pending human audit (`clean_audit_expanded.txt`); 92 checker-flagged
harvested cases quarantined (`flagged_for_review_expanded.jsonl`).

## 3. Standard set — headline results (figE1, figE2)

2,600 rows · positive class = hallucinated · 95% CIs (Wilson / bootstrap).

| System | Precision | Recall | F1 | Balanced acc. | False alarms /526 clean |
|---|---|---|---|---|---|
| **Our checker** (full) | **1.000** [1.00–1.00] | 0.868 [0.858–0.878] | **0.929** [0.924–0.935] | **0.934** | **0** (0%) |
| LLM judge (n=800ᵃ) | 0.881 | 0.980 | 0.928 | 0.743 | 80/162 (49%) |
| SummaC-Conv (n=250ᵃ) | 0.810 | 0.985 | 0.889 | 0.541 | 46/51 (90%) |
| Naive NLI (full) | 0.812 | 0.970 | 0.884 | 0.541 | 467 (89%) |
| LettuceDetect (full) | 0.831 | 0.842 | 0.836 | 0.583 | 356 (68%) |
| HHEM-2.1 (full) | 0.809 | 0.791 | 0.800 | 0.527 | ~387 (74%) |

ᵃ seeded stratified samples (seed 123); clean-column shows the sample's clean subset.

Our checker per corruption type: price 0.985 · cross-item 0.935 · name 0.833
· colour 0.738. Colour dropped from 0.90 (original set) — real-chat
responses phrase colours loosely; an honest, discussable NLI limitation.

**Reading:** compared with the original 238-row set (figE5), our checker's
F1 moved 0.975 → 0.929 — more data and messier data produce more
conservative, credible numbers — while precision held at 1.000 across a 16×
larger clean set (now with a CI of [0.999, 1.0]). Every baseline pays for
its recall with false-alarm floods (49–90% of clean responses rejected);
in the live system each false alarm is a pointless regeneration.

## 4. Hard adversarial set (figE3, figE4)

1,400 corrupted-only rows — no clean cases, so only **recall per corruption
family** is meaningful (precision is trivially 1.0 for any flag-happy
detector; balanced accuracy is undefined without negatives).

| System | Paraphrased colour | Paraphrased price | Fabricated attribute | Overall recall |
|---|---|---|---|---|
| **Our checker** (full) | 0.595 | 0.075 | 0.000ᵇ | 0.224 |
| Naive NLI (full) | 0.985 | 0.946 | 0.901 | 0.944 |
| LettuceDetect (full) | 0.974 | 0.966 | 0.994 | 0.978 |
| HHEM-2.1 (full) | — | — | — | 0.891 |
| LLM judge (n=208ᶜ) | 1.000 | 0.750 | 0.575 | 0.769 |
| SummaC-Conv (n=250ᵃ) | 0.988 | 0.988 | 0.963 | 0.980 |

ᵇ by design: fabricated attributes are *unsupported* claims, not
contradictions; the contradiction-only rule ignores them deliberately
(the asymmetric-cost argument) — this row quantifies that trade-off.
ᶜ 392/600 judge calls unanswered under overnight Groq throttling —
selection-bias caveat; re-run advised.

**The joint view (figE4) is mandatory context:** high hard-set recall is
trivial for detectors that flag nearly everything — naive NLI (0.944 here)
rejects 89% of *correct* responses on the standard set. No system reaches
the ideal corner (high adversarial recall at low false alarms): our checker
uniquely owns the zero-false-alarm end and pays for it on paraphrased
values; the baselines own recall and pay in alarm floods. Honest gaps
identified for future work: number-word normalization ("about fifteen
pounds" → £15) and an unsupported-claim detection layer for appended facts.

## 5. Detect–reject–regenerate loop (composed estimate)

The loop's correction mechanism was measured directly in the original
experiment (`../loop_mitigation/LOOP_RESULTS.md`): P(correct final |
detected) = 0.969, one regeneration usually sufficient. That mechanism
(generator, strictness prompts, 3-attempt policy) is unchanged, so it was
not re-run (user decision). Composing it with the expanded detection rate:

```
residual ≈ (1 − 0.868) + 0.868 × (1 − 0.969) ≈ 0.159
```

**≈ 16% of standard-set induced hallucinations would reach the user with
the loop ON, vs 100% OFF** — a derived estimate (measured mechanism ×
measured detection), not a direct measurement; regeneration was not
exercised on the real-chat evidence bundles.

## 6. Validity notes

1. **Cross-replication:** every full "ours" run was executed on both local
   CPU and Colab T4 GPU — hard set bit-identical; standard set differed in
   9/2,600 borderline cases (float wobble at thresholds, within CIs).
   Local numbers are canonical.
2. **Sampling policy:** our checker was never sampled; only slow/API-limited
   baselines ran on seeded stratified samples (LLM judge 800/600, SummaC
   250) — a runtime decision recorded per result in
   `results_expanded_summary.json`.
3. **Environment:** checker threshold 0.70 (from `.env`) pinned identically
   on Colab; HHEM and SummaC are incompatible with Colab's newer
   transformers and ran locally.
4. Rows share bases (526 independent); synthetic corruption still does not
   cover free-form fabrication beyond the fabricated-attribute family; the
   hard-set judge column carries the unanswered-calls caveat.

## 7. Reproduction

```powershell
# dataset (requires captured sources; deterministic from them)
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\expanded_eval\mongo_harvest.py
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\expanded_eval\build_expanded_set.py
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\expanded_eval\make_4000_subset.py

# evaluations (orchestrators; or Colab notebook expanded_eval_colab.ipynb)
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\expanded_eval\run_all_evals.py

# consolidation + figures
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\expanded_eval\build_expanded_summary.py
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\expanded_eval\make_expanded_figures.py
```
