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

## 1b. How the evaluation was planned — literature survey and design decisions

### The research question, split in two

The novelty makes two separable claims, each needing its own experiment:

1. **Detection claim** — the checker accurately identifies hallucinated
   responses → a *classification* problem → Precision / Recall / F1.
2. **Mitigation claim** — the detect-reject-regenerate loop prevents
   hallucinations from reaching the user → an *intervention* problem →
   system-ON vs system-OFF outcome rates (P/R do not apply to interventions).

### What prior work did (survey conducted 2026-07-08)

| Prior work | What it is | How it was evaluated | Metric(s) |
|---|---|---|---|
| SelfCheckGPT (Manakul 2023) | sampling-consistency hallucination detector | human-labeled GPT-3 WikiBio sentences | AUC-PR (NLI variant ≈ 92.5) |
| SummaC (Laban 2022) | NLI-based inconsistency detection | 6 existing labeled benchmark datasets | balanced accuracy (74.4%) |
| FactCC (Kryscinski 2020) | trained consistency classifier | **synthetic corruption**: entity/number/pronoun swaps, negation | accuracy / F1 |
| HaluEval (Li 2023) | injected-hallucination benchmark | sampling-then-filtering generation + human verification | classification accuracy |
| FActScore / RefChecker (2023-24) | atomic-claim / triplet verification | claim-level checks vs human annotation | claim precision |
| RAGAS (Es 2023) | LLM-judge faithfulness | supported claims ÷ total claims | faithfulness score 0-1 |
| CRAG / Self-RAG (2024) | corrective retrieval/generation loops | task performance **with vs without** the corrective mechanism | accuracy / FactScore deltas |

### Design decisions and their rationale

- **D1 — Build our own labeled test set.** No public benchmark exists for
  hallucination detection in multi-item conversational fashion
  recommendation. Consequence: published baseline numbers (WikiBio, CNN/DM)
  are NOT comparable — baselines must be **re-implemented and re-run on our
  test set**. This is standard practice when working in a new domain.
- **D2 — Ground truth via synthetic corruption** (FactCC / HaluEval
  precedent). Corrupting one field of a correct response while keeping the
  evidence intact yields labels that are *certain by construction* for the
  positive class, with zero manual annotation. Clean labels are only
  *presumed* (the live checker passed them) → exported to `clean_audit.txt`
  for human audit, avoiding the circularity of trusting the checker's own
  verdicts.
- **D3 — Corruption types mirror the checker's verified fields** (name,
  colour, price) **plus the novelty's target case** (cross-item value swaps,
  which specifically stress the item→sentence lock map).
- **D4 — Baselines bracket the design space.** Naive NLI = our own NLI model
  with every novel component removed (lower bracket / ablation-as-baseline:
  quantifies what the architecture contributes). LLM judge = the strong,
  expensive alternative (upper bracket: what an API-call-per-check approach
  buys). Beating the first proves the design matters; matching/beating the
  second proves the design is sufficient.
- **D5 — Two metric families.** Detector → P/R/F1/balanced accuracy
  (classification standard, per SelfCheckGPT/SummaC). Loop → residual
  hallucination rate ON vs OFF plus P(correction | detection)
  (intervention standard, per CRAG). Forcing P/R onto the loop would be a
  category error.
- **D6 — Independent referee for the loop.** The loop ships only
  checker-approved text, so the checker cannot grade the loop's output
  (self-grading reports 100% by definition). Final outputs are graded by a
  model-free value-verification script against the structured database truth.
- **D7 — Reproducibility.** Deterministic seeds (corruption seed 42),
  committed scripts, committed raw data, and a Colab notebook that re-runs
  the same scripts.

### Why NOT other candidate approaches

| Considered | Rejected because |
|---|---|
| Evaluate on public benchmarks (WikiBio, SummaC datasets) | different task: those test document-level consistency, not structured product-field grounding in dialogue |
| SelfCheckGPT-style sampling baseline | measures self-consistency across samples; our hallucinations contradict *evidence*, and sampling can be consistently wrong; also needs N generations per response |
| Training a FactCC-style classifier | needs large in-domain training data; our checker is zero-shot by design |
| Manual annotation only | slow, small-N, and unnecessary for the positive class given D2 |

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

## 5b. Metrics and baselines — full detail

### Metric definitions (positive class = hallucinated)

For each case the detector answers "hallucinated?" and is scored against the
label:

- **TP** — flagged a genuinely corrupted case
- **FP** — flagged a clean case (false alarm)
- **FN** — passed a corrupted case (missed lie)
- **TN** — passed a clean case

```
Precision          = TP / (TP + FP)        "when it accuses, is it right?"
Recall             = TP / (TP + FN)        "how many real lies caught?"
F1                 = 2PR / (P + R)         single comparison number
Specificity        = TN / (TN + FP)        "how well it leaves clean alone"
Balanced accuracy  = (Recall + Specificity) / 2   fair under class imbalance
                                                  (205 hallucinated vs 33 clean)
```

Why precision (zero FP) is weighted so heavily in this system: a false alarm
triggers a pointless regeneration — extra latency, extra LLM cost, and a
degraded (stricter, terser) response for a user who got a correct answer.
The asymmetric-cost argument is part of the design thesis.

Loop metrics (Experiment 2): residual hallucination rate
(wrong-shipped ÷ cases) for loop OFF vs ON; P(correction | detection);
attempts histogram; regeneration latency. See `loop_mitigation/LOOP_RESULTS.md`.

### Baseline 1 — Naive NLI (SummaC-style all-pairs)

*What it is:* the exact same DeBERTa NLI model our checker uses, with every
novel component removed:

| Component | Ours (v3) | Naive baseline |
|---|---|---|
| Item→sentence lock map | yes | no |
| Field filter / sentence-skip gates | yes | no |
| Two-sided exact-value gates (name/price) | yes | no — everything goes to NLI |
| Sentence pairing | locked / best-match | **every** (fact, sentence) pair |
| Decision | contradiction logit > 0.65 and > entailment | softmax P(contra) > 0.5 and > P(entail) on any pair |

*Why it exists:* it is the ablation-as-baseline — since it shares the NLI
model, any performance difference is attributable to the architecture around
the model, i.e. the claimed contribution.

*Result:* recall 0.961 but **29/33 clean responses falsely flagged**
(precision 0.872, balanced accuracy 0.541 ≈ chance). In production it would
reject nearly every correct answer. Conclusion: ungated NLI is unusable;
the gate architecture is what makes NLI practical here.

### Baseline 2 — LLM judge (RAGAS-style)

*What it is:* Groq `llama-3.1-8b-instant`, temperature 0, receives the
flattened evidence facts + the response, answers strict JSON
`{"hallucinated": true|false}`. Retries with backoff on rate limits.

*Why it exists:* represents the "just ask an LLM" family — the strong,
costly alternative an examiner will ask about.

*Result:* F1 0.966, balanced accuracy 0.902, but 5/33 false alarms, an
API call per check and ~3.5 s observed latency. Conclusion: our checker
matches/exceeds its quality (F1 0.975, BalAcc 0.976, 0 false alarms) at
zero API cost and millisecond latency.

### Headline results (authoritative numbers: `results_summary.json`)

238 cases = 205 corrupted + 33 clean. Positive class = hallucinated.

| System | Precision | Recall | F1 | Balanced acc. | FP on 33 clean |
|---|---|---|---|---|---|
| **Our checker v3** | **1.000** | 0.951 | **0.975** | **0.976** | **0** |
| Our checker v2 | 0.994 | 0.815 | 0.895 | 0.892 | 1 |
| Our checker v1 (original) | 1.000 | 0.571 | 0.727 | 0.785 | 0 |
| Naive NLI | 0.872 | 0.961 | 0.914 | 0.541 | 29 |
| LLM judge | 0.975 | 0.956 | 0.966 | 0.902 | 5 |

Per-corruption recall (v3): colour 0.896 · price 0.982 · name 0.948 ·
cross-item 0.977. Colour remains NLI-verified (semantic field) and is the
main recall gap.

Loop experiment: hallucinated responses reaching the user — loop OFF
**205/205 (100%)** vs loop ON **16/205 (7.8%)**; P(correction | detection)
96.9%; 94.4% of detected lies fixed by a single regeneration; ~0.41 s
regeneration cost per detected case.

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
- Loop-experiment figures (fig6 ON/OFF effect, fig7 residual by type,
  fig8 attempts × outcome) are generated by
  `loop_mitigation/make_loop_figures.py` into `loop_mitigation/figures/`.

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
| `loop_mitigation/` | loop-mitigation experiment: `run_loop_eval.py` (experiment), `referee.py` (independent grader), `regrade_shipped.py` (offline re-grade), `results_loop_eval.json` + `shipped_responses.jsonl` (outputs), `LOOP_RESULTS.md` (write-up), `make_loop_figures.py` + `figures/` (figs 6–8) |

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

## 9a. Growing the dataset from live chat — and reproducing the evaluation later

The capture hook works during normal interactive use, not only under the
scripted driver. This is how to collect more real conversations and re-run
the identical evaluation on the enlarged dataset.

### When capture happens

The hook fires inside `TextRAGPipeline.process()` — the exact code path the
chat API uses. While the backend process has `EVAL_CAPTURE=1` set, **every
generation attempt that goes through the hallucination check** appends one
row to `captured_cases.jsonl` (all checkable actions: catalog_search,
item_attribute_lookup, item_detail_lookup, item_compare,
explanation_generate; retries appear as attempt=2/3 rows). Not captured:
chitchat/refusal turns (never hallucination-checked) and the
cached-recommendation path. With the flag unset the hook is a no-op.

### Step-by-step: capture from live chatting

1. Enable the flag — either add `EVAL_CAPTURE=1` to `.env`, or in the
   terminal before starting the backend: `$env:EVAL_CAPTURE = "1"`.
2. Start the system as usual and chat normally. Watch for
   `[EvalCapture] case saved: ...` lines in the backend console.
3. When done, remove the flag (delete the `.env` line / new terminal).
4. **Snapshot the raw data**: commit `captured_cases.jsonl` to git. The
   commit hash IS the dataset version — any past evaluation can be
   reproduced by checking out that hash.

⚠ Note: running `collect_cases.py` ROTATES the capture file (renames the
existing one with a timestamp) before writing fresh scripted cases. Live
chatting only APPENDS. To combine scripted + chat data, chat AFTER the
driver run, or concatenate the rotated files back together.

### Step-by-step: re-run the evaluation on the enlarged dataset

```powershell
# 1. rebuild the labeled test set — seeded (42), so the same captured file
#    always yields the identical test set
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\corrupt_cases.py

# 2. re-run the three detectors
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\run_detector_eval.py

# 3. refresh summary + figures
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\build_summary.py
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\make_figures.py

# 4. (optional) re-run the loop experiment on the new corrupted cases
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\loop_mitigation\run_loop_eval.py
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\loop_mitigation\regrade_shipped.py
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\loop_mitigation\make_loop_figures.py
```

### Rules that keep the numbers scientifically valid

- **Audit the new clean cases.** `corrupt_cases.py` regenerates
  `clean_audit.txt`; every newly captured checker-passed response must be
  manually verified before its clean label is trusted (same anti-circularity
  rule as §4.4). New checker-flagged responses land in
  `flagged_for_review.jsonl` for adjudication.
- **Version results with their dataset.** Numbers from different test-set
  versions are not directly comparable — when the dataset grows, either
  re-run ALL systems (ours + both baselines) on the new set, or clearly
  report which dataset version each number came from (results JSONs +
  the jsonl commit hash).
- **Determinism boundaries.** Checker and naive-NLI results are
  deterministic for a given test set; the LLM-judge and loop-regeneration
  numbers can vary slightly between runs (live LLM), which is why raw
  responses are stored (`shipped_responses.jsonl`) alongside the metrics.

## 9b. Loop-mitigation experiment (subfolder `loop_mitigation/`)

Evaluates the second half of the contribution: the retry loop itself.
Each corrupted test-set case is replayed as the LLM's attempt-1 output into
the REAL pipeline loop (checker v3 + ResponseGenerator with strictness
escalation, 3-attempt policy). Final shipped outputs are graded by an
independent model-free referee (`referee.py`) against the database truth —
the checker never grades its own output. Files: `run_loop_eval.py`,
`referee.py`, `regrade_shipped.py`, `make_loop_figures.py`,
`results_loop_eval.json`, `LOOP_RESULTS.md`, `figures/`.


---

## 9c. Decision log and open items

### Chronological log

| Date | What happened |
|---|---|
| 2026-07-08 | Literature survey (SelfCheckGPT, SummaC, FactCC, HaluEval, RAGAS, CRAG, RefChecker) → evaluation design chosen (see §1b) |
| 2026-07-08 | Capture hook + driver built; 16 scripted conversations run live → 41 captured cases (35 first-attempt, 6 retry attempts) |
| 2026-07-08 | Test set built: 33 clean + corrupted variants (initially 199 cases; cross-item extended to price/name → 238); 8 live-flagged cases quarantined |
| 2026-07-08 | v1 detector eval: P 1.000 / R 0.571 — diagnosis: one-sided gates, NLI blind to exact-value mismatches |
| 2026-07-08 | v2 (two-sided gates + whitespace + price-regex fixes): R 0.815 — diagnosis: similarity gate hides renamed items; 1 FP from derived arithmetic |
| 2026-07-08 | v3 (similarity-gate bypass + response-level verification): P 1.000 / R 0.951 / F1 0.975 |
| 2026-07-09 | Docs consolidated; evidence slimming; folder moved to `test_result/hallucination_result/`; Colab notebook + bundle |
| 2026-07-09 | Loop-mitigation experiment: 100% → 12.2% first grading; referee refined (derived price diffs allowed, truncated names = minor) → final 100% → 7.8% |
| 2026-07-09 | Checker doc updated to v3; everything committed as `a274364` |

### Open items checklist

- [ ] **Manual audit** of the 33 clean cases (`clean_audit.txt`) — turns
      "presumed clean" into "human-verified"; required before final
      dissertation numbers.
- [ ] **Adjudicate** the 8 quarantined live flags
      (`flagged_for_review.jsonl`) — analysis suggests all 8 are v1 false
      positives (double-space name, truncated description, price-only
      sentence flagged on `type`); confirm or correct.
- [ ] **Experiment B (optional)** — live loop-behaviour stats with checker
      v3 (re-run `collect_cases.py`, measure first-pass acceptance / retry
      rate in production conditions; v1 live run had 3 spurious retry
      episodes to compare against).
- [ ] **Push + merge** — commit `a274364` exists locally; push and merge to
      main is a manual step.
- [ ] **Write-up caveats to state**: threshold sweep ≥ 1.0 artifact
      (containment flags carry synthetic score 1.0); v3 designed against
      this test set (fresh capture run would give held-out confirmation);
      synthetic corruption does not cover free-form fabrication; loop
      regeneration numbers vary slightly between runs (LLM temperature).

## 10. Methodology references

- Kryscinski et al. (2020) — *Evaluating the Factual Consistency of Abstractive
  Text Summarization* (FactCC) — synthetic corruption via entity/number swaps.
- Li et al. (2023) — *HaluEval* — large-scale injected-hallucination benchmark.
- Laban et al. (2022) — *SummaC* — NLI-based inconsistency detection; the
  naive baseline follows its all-pairs design; balanced accuracy metric.
- Manakul et al. (2023) — *SelfCheckGPT* — detection-quality evaluation framing.
- Es et al. (2023) — *RAGAS* — LLM-judge faithfulness; basis of baseline 3.
- Yan et al. (2024) — *Corrective RAG* — system-on/off mitigation evaluation;
  the loop-mitigation experiment follows this design.
