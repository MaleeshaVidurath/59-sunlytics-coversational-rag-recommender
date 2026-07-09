# Hallucination Checker Evaluation — Full Implementation Documentation

This document describes the complete evaluation pipeline built for the
hallucination checker and detect-reject-regenerate loop — every file, every
stage, the methodology it follows, and how to re-run it. Results themselves
live in `RESULTS.md`; this file explains **how they were produced**.

---

## 1. Why this evaluation exists

The research contribution being evaluated is the NLI-based hallucination
checker (`text_rag/core/hallucination_checker.py`) and its retry loop in the
RAG pipeline. A dissertation claim ("the checker detects hallucinations
accurately, better than existing approaches") needs:

1. A **labeled test set** — (evidence, response) pairs where the ground truth
   (correct / hallucinated) is known.
2. **Baselines** — established alternative approaches run on the same data.
3. **Standard metrics** — Precision, Recall, F1, Balanced Accuracy.

No public benchmark exists for hallucination detection in multi-item
conversational fashion recommendation, so the test set is built from this
system's own outputs using **synthetic corruption** — the methodology of
FactCC (Kryscinski et al., 2020: entity/number swap transformations) and
HaluEval (Li et al., 2023: injected hallucinated samples).

---

## 2. Pipeline overview

```
                     ┌──────────────────────────────────────────────┐
 STAGE 1  capture    │ collect_cases.py  (drives the real pipeline) │
                     │   + capture.py hook inside rag_pipeline.py   │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                          captured_cases.jsonl   (41 real cases)
                                        │
                     ┌──────────────────▼───────────────────────────┐
 STAGE 2  corrupt    │ corrupt_cases.py  (FactCC-style corruption)  │
                     └──────────────────┬───────────────────────────┘
                                        ▼
              labeled_test_set.jsonl  (238 cases: 33 clean + 205 hallucinated)
              flagged_for_review.jsonl (8 disputed cases → human adjudication)
              clean_audit.txt          (manual audit sheet for clean labels)
                                        │
                     ┌──────────────────▼───────────────────────────┐
 STAGE 3  evaluate   │ run_detector_eval.py                         │
                     │   detector 1: our checker                    │
                     │   detector 2: naive NLI baseline             │
                     │   detector 3: LLM-judge baseline (Groq)      │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                    results_detector_eval{_v1,_v2,}.json
                                        │
                     ┌──────────────────▼───────────────────────────┐
 STAGE 4  report     │ build_summary.py  → results_summary.json     │
                     │ make_figures.py   → figures/*.png (5 charts) │
                     │ RESULTS.md        → written results chapter  │
                     └──────────────────────────────────────────────┘
```

---

## 3. Stage 1 — Case capture

### 3.1 The capture hook (`capture.py` + edit in `rag_pipeline.py`)

One evaluation case = the exact pair the checker receives:
`checker.check(response_text, evidence)`. These pairs exist only transiently
inside `TextRAGPipeline.process()`, so a capture hook was added inside the
generate→check loop:

```python
# rag_pipeline.py — after every hallucination check
_capture_case(evidence=..., response_text=..., action=..., attempt=...,
              session_id=..., user_message=..., check_result=...)
```

- **Off by default.** The hook is a no-op unless the environment variable
  `EVAL_CAPTURE=1` is set — zero impact on normal operation.
- **Every generation attempt is captured**, so retry-loop episodes appear as
  attempt=1/2/3 rows.
- **Evidence is slimmed** (`slim_evidence()`): only the fields the checker
  verifies (name, colour, price, type, pattern, section, index_group,
  material_description, article_id) are stored; personalisation payloads
  (user_preferences, purchase_hints, style_profile, …) are dropped.
- Output: one JSON line per attempt appended to `captured_cases.jsonl`:

```json
{
  "captured_at": "...", "session_id": "...",
  "user_message": "I want a black dress under £50",
  "action": "catalog_search", "attempt": 1,
  "evidence": { "action": "catalog_search", "items": [ ... ] },
  "response_text": "Option 1: London dress, Black, £11.08, ...",
  "checker": { "passed": true, "n_checked": 6, "n_flagged": 0, ... }
}
```

Manual chat sessions are captured the same way when `EVAL_CAPTURE=1` is set
before starting the API — useful for adding natural conversations to the set.

### 3.2 The driver (`collect_cases.py`)

Runs 16 scripted conversations (36 turns) through the **real full pipeline**
(memory pipeline → DistilBERT → CSE → evidence assembler → Groq LLM →
hallucination checker) exactly as a user would, with the capture hook on.

- Conversations are designed for **coverage**: every checkable action appears
  multiple times — catalog_search (incl. refinements), item_attribute_lookup,
  item_detail_lookup, item_compare, explanation_generate.
- Each conversation runs on a **fresh session** (old sessions cleared from
  MongoDB + Redis first, same pattern as `test_full_pipeline.py`).
- Scripted (rather than manual chatting) for **reproducibility** — the exact
  message list is committed in the file.
- Requires the full stack running (MongoDB, Redis, PostgreSQL, Qdrant, Groq
  key in `.env`) and the repo-root venv.

**Run 2026-07-08 result:** 36 turns, 0 failures, **41 cases** captured
(35 attempt-1, 3 attempt-2, 3 attempt-3 — three live retry episodes).

---

## 4. Stage 2 — Labeled test set construction (`corrupt_cases.py`)

### 4.1 Splitting the captured cases

- 8 cases the live checker **flagged** → quarantined to
  `flagged_for_review.jsonl`. Their ground truth is unknown (the checker may
  be right or wrong); a human must adjudicate. They are excluded from the
  test set to avoid contaminating it either way.
- 33 cases the checker **passed** → become `label: "clean"` test cases
  (presumed correct — see 4.4).

### 4.2 Synthetic corruption (label: "hallucinated")

Each clean case is copied and **one fact in the response text is changed
while the evidence stays untouched** — so the corrupted copy provably
contradicts the evidence. Deterministic (seed 42): re-running regenerates the
identical test set.

| Type | n | Mechanism |
|---|---|---|
| `colour_swap` | 48 | item colour → a colour that appears **nowhere** in the evidence (so the lie can't accidentally be true for another item); substring pairs (Blue/Dark Blue) excluded |
| `price_change` | 56 | £value → value ± £2–7.25, regex-replaced |
| `name_swap` | 58 | product name → a different catalog name from the global pool; prefix-related names excluded |
| `cross_item_swap` | 43 | the colour, price, or name values of two items **exchanged** — every value still exists in the evidence, only the item association is wrong. This is the failure mode the item→sentence lock map targets |

Every corrupted row records its recipe:
`"corruption": {"type": "colour_swap", "original": "Black", "corrupted": "Grey", "item_idx": 0}` —
enabling per-corruption-type recall breakdowns.

**Total: 238 cases = 33 clean + 205 hallucinated.**

### 4.3 Known construction caveats

- When two items share a value (both "Black"), the regex replaces **all**
  occurrences — the response-level label stays valid, but `item_idx` is
  approximate for shared values.
- `cross_item_swap` needs two differing values that both appear in the
  response; colour-only yielded 4 cases, extending to price/name raised it to 43.

### 4.4 Label integrity (avoiding circularity)

Using the checker's own pass/fail as ground truth for clean cases would be
circular. Mitigations:

- Corrupted labels are **certain by construction** (independent of the checker).
- Clean labels are **presumed** and exported to `clean_audit.txt` — a
  human-readable sheet (evidence vs response per case) for manual audit.
- The 8 disputed live flags go to human adjudication, not into the set.

---

## 5. Stage 3 — Detector evaluation (`run_detector_eval.py`)

Loads the 238 cases, hides the labels, asks each detector for a verdict per
case, and scores verdicts against labels. Positive class = hallucinated.

### 5.1 Detector 1 — our checker

`HallucinationChecker().check(response_text, evidence)` → `has_hallucination`.
Raw per-check NLI scores are stored, enabling the offline threshold sweep
(the gate pipeline is threshold-independent, so decisions can be recomputed
at any threshold without re-running models).

### 5.2 Detector 2 — naive NLI baseline (SummaC-style)

The ablation-as-baseline: the same DeBERTa model with **everything else
removed** — no lock map, no gates, no containment. Every evidence fact is
paired with every response sentence; flag if any pair has softmax
P(contradiction) > 0.5 and > P(entailment). Quantifies what the gate
architecture contributes.

### 5.3 Detector 3 — LLM-judge baseline (RAGAS-style)

Groq `llama-3.1-8b-instant` receives the flattened evidence facts and the
response, and answers strict JSON `{"hallucinated": true|false}`
(temperature 0, retry with backoff on 429s). Represents the
"just ask an LLM" family of approaches.

### 5.4 Metrics

TP/FP/FN/TN → Precision, Recall, F1, Specificity, Balanced Accuracy,
Accuracy; plus recall per corruption type, false-positive and missed case IDs
(for error analysis), and the threshold sweep. Flags:
`--skip-llm`, `--skip-naive` (baselines are checker-version-independent),
`--limit N` (smoke test).

---

## 6. Evaluation-driven checker refinement (v1 → v3)

The evaluation was run three times against the same test set, with the
checker improved after each diagnosis. Changes live in
`text_rag/core/hallucination_checker.py`.

### v1 → v2: two-sided exact-value gates

**Diagnosis:** name/price recall ≈ 0.38 — the original Gates 6/7 were
one-sided (verbatim containment could only PASS); absent values fell through
to DeBERTa, which scores exact-value mismatches as *neutral*.
**Fix:** name/price never reach NLI. Value present → pass; a *different*
value of the same kind present → contradiction; no value → skip. Plus two bug
fixes: whitespace normalisation (double-space catalog names defeated verbatim
match) and the price regex trailing-dot bug (`£[\d,.]+` captured `"£11.08."`,
so the verbatim pass almost never fired).

### v2 → v3: similarity-gate bypass + response-level verification

**Diagnosis:** 31 name swaps still escaped — the MiniLM similarity gate
skipped facts whose name had been swapped (the swap destroys the very
similarity being measured). One false positive: correct derived arithmetic
("£4.04 more expensive") contained a £value matching no evidence price.
**Fix:** exact-value facts bypass the similarity gate entirely. Locked
catalog items are verified against their locked sentence (authoritative —
catches cross-item swaps). Unlocked facts (compare/explanation) are verified
at **response level**: flag only when the true value is absent from the whole
response and a different same-kind value is present — this both catches
renamed items and tolerates derived values.

Wrong-name detection (`_find_wrong_name`) checks, in order: other evidence
items' names → the name slot of structured "Option N: <name>," sentences
(only when the sentence carries a £) → the full catalog name list from
`sample_articles.csv` (case-sensitive, ≥5 chars, substring-related names
excluded as ambiguous truncations).

**Outcome:** F1 0.727 → 0.975 with precision restored to 1.000
(full numbers and figures in `RESULTS.md`).

---

## 7. Stage 4 — Reporting

- `build_summary.py` → `results_summary.json`: merges the three per-version
  result files + baselines + metadata into one machine-readable summary.
- `make_figures.py` → `figures/fig1…fig5.png` (200 dpi, colorblind-safe
  validated palette, consistent entity colors): metric comparison, false
  alarms on clean responses, v1→v3 progression, recall by corruption type,
  threshold sensitivity (artifact region ≥1.0 shaded — containment flags
  carry a synthetic score of 1.0).
- `RESULTS.md`: written results chapter — setup, tables, v1→v3 narrative,
  threats to validity, reproduction commands.

---

## 8. File inventory

| File | Role |
|---|---|
| `capture.py` | capture hook (EVAL_CAPTURE=1) + evidence slimming |
| `collect_cases.py` | scripted-conversation driver (Stage 1) |
| `captured_cases.jsonl` | 41 raw (evidence, response, check) cases |
| `captured_cases_full_backup.jsonl` | pre-slimming backup (deletable) |
| `corrupt_cases.py` | test-set builder (Stage 2, seed 42) |
| `labeled_test_set.jsonl` | 238 labeled cases — the exam paper |
| `flagged_for_review.jsonl` | 8 disputed live flags → human adjudication |
| `clean_audit.txt` | manual audit sheet for the 33 clean labels |
| `run_detector_eval.py` | 3-detector evaluation + metrics (Stage 3) |
| `results_detector_eval_v1.json` | v1 checker + both baselines |
| `results_detector_eval_v2.json` | v2 checker |
| `results_detector_eval.json` | v3 checker (current) |
| `build_summary.py` / `results_summary.json` | consolidated summary |
| `make_figures.py` / `figures/` | 5 dissertation figures |
| `RESULTS.md` | written results chapter |
| `eval_run.log` | console log of the v1 full run |
| `hallucination_eval_colab.ipynb` | Colab notebook — reproduces Stages 2–4 with the same scripts |
| `make_colab_bundle.py` / `colab_bundle.zip` | builds the minimal upload bundle for the notebook |

## 9. Reproduction (run from repo root, use the repo venv)

```powershell
# Stage 1 — capture (needs MongoDB, Redis, PostgreSQL, Qdrant, GROQ_API_KEY)
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\collect_cases.py

# Stage 2 — build test set (offline, deterministic)
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\corrupt_cases.py

# Stage 3 — evaluate (local models; --skip-llm to avoid Groq calls)
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\run_detector_eval.py

# Stage 4 — consolidate + figures
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\build_summary.py
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\make_figures.py
```

## 9b. Loop-mitigation experiment (subfolder `loop_mitigation/`)

Evaluates the second half of the contribution: the retry loop itself.
Each corrupted test-set case is replayed as the LLM's attempt-1 output into
the REAL pipeline loop (checker v3 + ResponseGenerator with strictness
escalation, 3-attempt policy). Final shipped outputs are graded by an
independent model-free referee (`referee.py`) against the database truth —
the checker never grades its own output. Files: `run_loop_eval.py`,
`referee.py`, `regrade_shipped.py`, `make_loop_figures.py`,
`results_loop_eval.json`, `LOOP_RESULTS.md`, `figures/`.

## 10. Methodology references

- Kryscinski et al. (2020) — *Evaluating the Factual Consistency of Abstractive
  Text Summarization* (FactCC) — synthetic corruption via entity/number swaps.
- Li et al. (2023) — *HaluEval* — large-scale injected-hallucination benchmark.
- Laban et al. (2022) — *SummaC* — NLI-based inconsistency detection; the
  naive baseline follows its all-pairs design; balanced accuracy metric.
- Manakul et al. (2023) — *SelfCheckGPT* — detection-quality evaluation framing.
- Es et al. (2023) — *RAGAS* — LLM-judge faithfulness; basis of baseline 3.
- Yan et al. (2024) — *Corrective RAG* — system-on/off mitigation evaluation
  (planned loop experiment).
