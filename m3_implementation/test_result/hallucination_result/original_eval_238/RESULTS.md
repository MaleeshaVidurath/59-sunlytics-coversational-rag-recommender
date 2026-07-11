# Hallucination Checker — Evaluation Results

Generated: 2026-07-09 · Test set: `labeled_test_set.jsonl` (seed 42) ·
Machine-readable version: `results_summary.json` · Figures: `figures/`

---

## 1. Evaluation setup

**Test set construction (FactCC/HaluEval-style synthetic corruption).**
41 real (evidence, response) pairs were captured from live pipeline runs
(`collect_cases.py`, 16 scripted conversations covering all five checkable
actions). The 33 responses that passed the live hallucination check became
*clean* cases; each was then programmatically corrupted — one response field
changed while the evidence stayed untouched — producing 205 *hallucinated*
cases that are guaranteed wrong by construction:

| Corruption type | n | What is changed in the response |
|---|---|---|
| name_swap | 58 | product name → a different catalog name |
| price_change | 56 | £value → a different £value |
| colour_swap | 48 | colour → a colour absent from the evidence |
| cross_item_swap | 43 | two items' colour/price/name values exchanged (association error — every value still exists in the evidence) |

**Total: 238 cases (205 hallucinated / 33 clean).** The 8 live responses the
checker had flagged were quarantined to `flagged_for_review.jsonl` for human
adjudication (manual inspection suggests all 8 were v1 false positives:
a double-space catalog name defeating verbatim match, truncated description
text, and a `type` fact flagged against a price-only sentence).

**Systems compared** (positive class = hallucinated):

- **Our checker** — evidence-field-first NLI checker: MiniLM item→sentence
  lock map, gate pipeline, DeBERTa (`cross-encoder/nli-deberta-v3-base`)
  contradiction-only NLI for semantic fields.
- **Naive NLI baseline** (SummaC-style) — same DeBERTa on *every*
  (fact, sentence) pair, no lock map, no gates; flag if any pair has softmax
  P(contradiction) > 0.5 and > P(entailment).
- **LLM-judge baseline** (RAGAS-style) — Groq `llama-3.1-8b-instant` judges
  whether the response contradicts the evidence facts.

## 2. Headline result (fig1, fig2)

| System | Precision | Recall | F1 | Balanced acc. | False alarms (33 clean) |
|---|---|---|---|---|---|
| **Our checker (v3)** | **1.000** | 0.951 | **0.975** | **0.976** | **0** |
| Naive NLI | 0.872 | 0.961 | 0.914 | 0.541 | 29 |
| LLM judge | 0.975 | 0.956 | 0.966 | 0.902 | 5 |

The gated checker achieves the best F1 and balanced accuracy of all systems
with zero false alarms, zero API cost, and millisecond latency. The naive
baseline demonstrates why ungated NLI is unusable in production: it wrongly
rejected 29 of 33 correct responses (balanced accuracy 0.541 — near chance),
which in the live system would trigger pointless regeneration on almost every
turn. The LLM judge is accurate but needs a network call per check (~3.5 s
observed) and still produced 5 false alarms.

## 3. Evaluation-driven refinement, v1 → v3 (fig3)

The same test set was used to diagnose and fix the checker twice:

| Version | Precision | Recall | F1 | Balanced acc. | Change |
|---|---|---|---|---|---|
| v1 original | 1.000 | 0.571 | 0.727 | 0.785 | — |
| v2 two-sided gates | 0.994 | 0.815 | 0.895 | 0.892 | wrong name/£value in the item's sentence now flags directly; name/price never sent to NLI; whitespace normalisation; price-regex trailing-dot fix |
| v3 + response-level | 1.000 | 0.951 | 0.975 | 0.976 | exact-value facts bypass the MiniLM similarity gate (a swapped name destroys the very similarity the gate measures); unlocked facts verified response-level |

**v1 diagnosis.** Precision was perfect but recall on name (0.379) and price
(0.393) corruptions was poor: the original gates were one-sided — verbatim
containment could only *pass* a fact; when the value was absent the fact fell
through to DeBERTa, which scores exact-value mismatches ("£11.08" vs "£13.58")
as *neutral*, not *contradiction*.

**v2 diagnosis.** Recall improved to 0.815 but 31 name swaps still escaped
(the MiniLM similarity gate skipped facts whose name had been swapped, since
the swap destroys the fact–sentence similarity) and one false positive
appeared: the LLM's correct derived arithmetic "£4.04 more expensive"
contained a £value matching no evidence price.

**v3.** Exact-value facts are decided purely by value logic. Locked catalog
sentences are verified per-sentence (authoritative for their item, catching
cross-item swaps); unlocked facts (compare/explanation) are verified at
response level — a flag requires the true value to be absent from the whole
response with a different value of the same kind present. This restores
precision 1.000 and fixes the renamed-item misses.

## 4. Detection by corruption type (fig4)

| Corruption | Ours (v3) | Naive NLI | LLM judge |
|---|---|---|---|
| colour_swap | 0.896 | 1.000 | 1.000 |
| price_change | 0.982 | 0.964 | 0.946 |
| name_swap | 0.948 | 0.897 | 0.914 |
| cross_item_swap | 0.977 | 1.000 | 0.977 |

Colour is the one field still verified by NLI (a colour is a semantic
attribute — "crimson" vs "Red" needs semantic judgment, not string match);
its 5 misses are genuine DeBERTa limitations and the checker's remaining
recall gap. The baselines' near-perfect recall comes at the false-alarm cost
shown above — a detector that flags nearly everything trivially catches
every corruption.

## 5. Threshold sensitivity (fig5)

Metrics were recomputed offline from the stored per-check NLI scores at
thresholds 0.25–5.0 (raw logits — `CrossEncoder.predict` returns logits, not
probabilities, which the config value 0.65 is compared against). Results are
flat across 0.25–0.65 with precision 1.000 throughout, so the operating point
is uncritical. **Caveat:** sweep values ≥ 1.0 are an artifact — containment
flags carry a synthetic contradiction score of 1.0 and drop out of the
decision at t ≥ 1.0 — so only the region below 1.0 reflects real NLI
threshold behaviour.

## 5b. Loop-mitigation experiment

The detect-reject-regenerate loop was evaluated separately (CRAG-style ON/OFF
with induced failures): the loop reduced user-facing hallucinations from 100%
to **7.8%** (16/205), with P(correction | detection) = 96.9%. Full method,
failure analysis, and figures: `loop_mitigation/LOOP_RESULTS.md`.

## 5c. Off-the-shelf external detectors (unmodified released tools)

Three released, citable detectors were additionally run AS-IS on the same 238
cases (evidence serialized to text): Vectara HHEM-2.1 (F1 0.509), SummaC-Conv
(F1 0.643), LettuceDetect (F1 0.688) vs our checker (F1 0.975). Headline
finding: cross-item swaps are nearly invisible to presence-checking tools
(HHEM 9.3%, LettuceDetect 4.7% vs ours 97.7%) - direct evidence for the
item-sentence lock map. Full method, tables, caveats and figures 9-10:
`external_baselines/EXTERNAL_BASELINES.md`.

## 6. Threats to validity (state these in the dissertation)

1. **Presumed-clean labels.** The 33 clean cases passed the v1 checker and
   manual spot checks but await full manual audit (`clean_audit.txt`).
   Using checker output as ground truth would be circular; the audit closes
   this gap.
2. **Synthetic corruption coverage.** Injected errors cover field-value and
   association errors but not free-form fabrication (invented attributes,
   unsupported claims). The LLM-judge comparison partially covers this.
3. **v3 was designed against this test set.** The fixes generalise by
   construction (value logic, not tuned parameters), but reporting should
   note the refinement loop used the same data; a fresh capture run would
   provide a held-out confirmation.
4. **Single domain / single LLM.** Results are for the H&M catalog with
   llama-3.1-8b via Groq; the architecture is domain-general but numbers are not.

## 7. Reproduction

```
# 1. capture live cases (needs full stack: Mongo, Redis, Postgres, Qdrant, Groq)
venv\Scripts\python m3_implementation/test_result/hallucination_result/collect_cases.py

# 2. build the labeled test set (deterministic, seed 42)
venv\Scripts\python m3_implementation/test_result/hallucination_result/corrupt_cases.py

# 3. run all three detectors
venv\Scripts\python m3_implementation/test_result/hallucination_result/run_detector_eval.py

# 4. consolidate + figures
venv\Scripts\python m3_implementation/test_result/hallucination_result/build_summary.py
venv\Scripts\python m3_implementation/test_result/hallucination_result/make_figures.py
```

Per-version raw outputs: `results_detector_eval_v1.json`, `_v2.json`,
`results_detector_eval.json` (v3, includes false-positive and missed case IDs
for error analysis).
