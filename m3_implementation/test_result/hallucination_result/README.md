# Hallucination Evaluation — README (Start-to-End Story)

**Read this first.** This is the plain-language master document for everything
in this folder: what was done, in what order, and WHY each decision was made.
Detailed documents are linked at every step. Written 2026-07-11.

**What is being evaluated:** the research novelty of this project — the
NLI-based hallucination checker (`text_rag/core/hallucination_checker.py`)
and its detect→reject→regenerate loop in the RAG pipeline. The dissertation
must show these work, measure how well, compare against alternatives, and
honestly map their limits.

---

## The story, step by step

### Step 1 — Literature research (2026-07-08)
**What:** surveyed how existing work evaluates hallucination detection:
SelfCheckGPT, SummaC, FactCC, HaluEval, FActScore/RefChecker, RAGAS,
CRAG/Self-RAG, and later RAGTruth/HaluBench/RAGBench.
**Why:** an evaluation only counts if it follows accepted methodology.
**Key findings that shaped everything:** (a) detectors are evaluated as
classifiers → Precision/Recall/F1/balanced accuracy on labeled data;
(b) corrective loops are evaluated system-ON vs system-OFF (CRAG);
(c) test sets are commonly built by **synthetic corruption** of correct
outputs (FactCC's entity/number swaps, HaluEval's injections);
(d) NO public benchmark provides labeled (structured catalog evidence,
generated response) pairs — the checker's input format — so we must build
our own and re-run baselines on it (standard practice; precisely worded
survey in `HALLUCINATION_EVALUATION_PROCESS.md` §1b, decision D1).

### Step 2 — Case capture infrastructure
**What:** a capture hook inside `rag_pipeline.py` (active only when env var
`EVAL_CAPTURE=1`) saves every (evidence, response, checker verdict) triple
to `captured_cases.jsonl`; a driver (`collect_cases.py`) runs scripted
conversations through the REAL full pipeline (memory → DistilBERT → CSE →
assembler → Groq LLM → checker).
**Why scripted, not manual chatting:** coverage of all five checkable
actions is guaranteed and the exact message list is committed →
reproducible. **Why a hook:** the (evidence, response) pair exists only
transiently in memory; nothing on disk stored the full evidence bundle.
**First run:** 16 conversations → 41 cases.

### Step 3 — Labeled test set by synthetic corruption
**What:** `corrupt_cases.py` copies each checker-passed response and breaks
ONE fact in the response while leaving evidence untouched (colour swap,
price change, name swap, cross-item value swap) → 238 rows (33 clean + 205
corrupted, seed 42 = fully reproducible).
**Why this is valid:** corrupted labels are *certain by construction*.
**Why cross-item swaps:** they specifically stress the item→sentence lock
map (the architectural novelty): every value exists in the evidence, only
the item association is wrong.
**Anti-circularity rule:** clean labels come from checker-pass and are only
*presumed* — a human audit sheet (`clean_audit.txt`) exists because using
the checker to grade its own test set would be circular. Checker-flagged
live cases were quarantined (`flagged_for_review.jsonl`), not trusted.

### Step 4 — Detector evaluation and the v1→v3 refinement
**What:** `run_detector_eval.py` runs three detectors on the 238 rows:
ours, a naive all-pairs NLI baseline (SummaC-style — our own NLI model with
every novel component removed = ablation-as-baseline), and a RAGAS-style
LLM judge (Groq).
**What happened:** v1 scored P=1.000 but R=0.571. Diagnosis: the name/price
gates were one-sided (could only PASS) and DeBERTa scores exact-value
mismatches ("£11.08" vs "£13.58") as *neutral*. → **v2**: two-sided gates
(different value present = contradiction; exact values never go to NLI) +
two real bug fixes (whitespace-normalised names; price regex swallowed the
trailing period). → still missed renamed items because the MiniLM
similarity gate hid them (a swapped name destroys the very similarity the
gate measures) → **v3**: exact-value facts bypass the similarity gate;
unlocked facts verified at response level.
**Result:** F1 0.727 → 0.975 at P=1.000. The **evaluation-driven
refinement** narrative (measure → diagnose → fix → re-measure on the same
set) is itself dissertation material. Full detail: `RESULTS.md` §3,
figs 1–5 in `figures/`.

### Step 5 — Loop experiment (detect→reject→regenerate)
**What:** each of the 205 corrupted rows replayed as the LLM's "attempt 1"
into the REAL pipeline loop; final shipped outputs graded by an
**independent model-free referee** (`loop_mitigation/referee.py`).
**Why induced failures:** the LLM almost never hallucinates naturally
(~0 in 36 live turns) — no signal without planting failures (fire-alarm
testing with a smoke machine).
**Why an independent referee:** the loop ships only checker-approved text,
so the checker grading the loop = marking its own homework = 100% success
by definition. **Referee refinement:** first pass over-flagged correct
derived arithmetic ("£4.04 cheaper") and truncated names — fixed, re-graded
offline (same honesty loop as the checker itself).
**Result:** hallucinations reaching the user 100% (loop OFF) → **7.8%**
(loop ON); P(correction | detection)=96.9%; 94% fixed by ONE regeneration.
Full detail: `loop_mitigation/LOOP_RESULTS.md`, figs 6–8.

### Step 6 — Off-the-shelf external baselines
**Why:** answering "what is your baseline?" with "my own re-implementation"
sounds weak. So three released, citable tools were run UNMODIFIED on the
same 238 rows: **SummaC-Conv** (the academic classic — also proves our
re-implementation wasn't a strawman), **Vectara HHEM-2.1** (the industry
leaderboard model), **LettuceDetect** (2025, trained on RAGTruth — embodies
"use an existing RAG benchmark").
**Result:** F1 0.643 / 0.509 / 0.688 vs ours 0.975. Headline: cross-item
swaps are nearly invisible to them (9%/5%/53% vs ours 97.7%) — they check
value *presence*, only the lock map checks *association*. Details + tool
selection rationale: `external_baselines/EXTERNAL_BASELINES.md`, figs 9–10.

### Step 7 — Documentation and reproducibility
`HALLUCINATION_EVALUATION_PROCESS.md` = how everything works (pipeline,
decisions D1–D7, metrics, decision log, open items). `RESULTS.md` = the
original results chapter. The checker's own design doc
(`text_rag/core/HALLUCINATION_CHECKER_PROCESS.md`) was updated to v3.
A Colab notebook (`hallucination_eval_colab.ipynb` + bundle) reproduces
Stages 2–4 with the same scripts. Live-chat capture + re-run procedure for
growing the dataset: PROCESS doc §9a. Everything up to here was committed
as `a274364`.

### Step 8 — The evaluator's objections → the expanded evaluation
**The objections:** numbers look "unrealistically high" (P=1.000, F1=0.975)
and the dataset is small (238 rows, 33 clean).
**The decision (important):** numbers are NEVER adjusted downward — that
would be as dishonest as inflating them. The honest fixes: more data,
harder data, and confidence intervals.
**What was built (all inside `expanded_eval/`, original evaluation left
byte-for-byte untouched — deliberate isolation):**
1. **More data, three sources:** 166 new scripted cases (60 conversations;
   needed 25 s/turn pacing for Groq's 6000 tokens/min limit) + **416 real
   user chats harvested from MongoDB** (`mongo_harvest.py` reconstructs
   evidence from the stored recommendation items) + the original 36 →
   **526 clean bases**, deduplicated.
2. **Standard expanded set:** same seed-42 corruption → 8,181-row pool,
   from which a **4,000-row suite** was drawn (user decision: 4,000 is
   enough; all 526 clean kept — they anchor precision and the audit —
   corrupted sampled evenly per type; full pool kept as `*_full.jsonl`).
3. **Hard adversarial set** (1,400 rows): paraphrased colours ("a rich
   crimson shade"), paraphrased prices ("about fifteen pounds"),
   fabricated attributes (unsupported claims the contradiction-only checker
   ignores BY DESIGN — the set quantifies that trade-off).
4. **95% CIs everywhere** (Wilson + bootstrap, in `compute_metrics`).
5. **Compute split:** slow NLI tools on a Colab T4 GPU (notebook
   `expanded_eval_colab.ipynb`); LLM judge locally (API-bound); HHEM and
   SummaC locally (incompatible with Colab's newer transformers). Ours ran
   FULL everywhere and was cross-replicated local-vs-Colab (hard set
   bit-identical; standard 9/2,600 borderline float differences).
6. **Loop NOT re-run** (user decision): the correction mechanism didn't
   change; a composed estimate (measured mechanism × new detection rate ≈
   16% residual) is reported, clearly labeled as derived.

**The expanded outcome:** ours P=1.000 [CI 0.999–1.0] with **zero false
alarms on 526 clean cases**, R=0.868, F1=0.929, balanced accuracy 0.934 —
best of six systems; every baseline pays for recall with 49–90% false-alarm
rates on clean responses. The hard set exposes honest limits (paraphrased
price 0.075, fabricated 0.0-by-design, colour 0.595) — see the joint-view
figure figE4 before quoting any hard-set number: flag-everything detectors
trivially ace a corrupted-only set. Full detail: `expanded_eval/EXPANDED_RESULTS.md`.

---

## Folder map — where to look

| Path | What it is |
|---|---|
| `README.md` | this file — the story |
| `HALLUCINATION_EVALUATION_PROCESS.md` | HOW everything works: pipeline, design decisions D1–D7, literature survey, metrics, decision log, open items |
| `original_eval_238/RESULTS.md` | original 238-row results chapter (v1→v3, baselines, threats to validity) |
| `capture.py` / `collect_cases.py` / `corrupt_cases.py` / `run_detector_eval.py` / `merge_captures.py` | SHARED pipeline tools (root — used by every evaluation): capture → corrupt → evaluate |
| `original_eval_238/` | the original 238-row evaluation: `RESULTS.md`, data (`captured_cases*.jsonl`, `labeled_test_set.jsonl`), results JSONs, figs 1–5, `build_summary.py`/`make_figures.py`, Colab notebook + bundle, run logs |
| `original_eval_238/clean_audit.txt` / `original_eval_238/flagged_for_review.jsonl` | ⚠ human-audit homework (original) |
| `loop_mitigation/` | loop experiment: runner, independent referee, LOOP_RESULTS.md, figs 6–8 |
| `external_baselines/` | unmodified HHEM/SummaC/LettuceDetect runs, EXTERNAL_BASELINES.md, figs 9–10 |
| `expanded_eval/` | the expanded evaluation: mongo_harvest, set builders, orchestrators, EXPANDED_RESULTS.md, results_expanded_summary.json, figs E1–E5, Colab notebook |
| `expanded_eval/clean_audit_expanded.txt` / `flagged_for_review_expanded.jsonl` | ⚠ human-audit homework (expanded: 526 + 92) |

## Headline numbers (memorise these two rows)

| | Original set (238) | Expanded set (2,600) |
|---|---|---|
| Our checker | P 1.000 · R 0.951 · F1 0.975 | P 1.000 [0.999–1.0] · R 0.868 · F1 0.929 · **0 false alarms/526 clean** |
| Best baseline | LLM judge F1 0.966 (5 FP) | LLM judge F1 0.928 (49% FP rate) |

Loop: 100% → 7.8% user-facing hallucinations (measured, original set);
≈16% composed estimate on expanded data.

## Open items checklist

- [ ] Human audit: `original_eval_238/clean_audit.txt` (33) + `expanded_eval/clean_audit_expanded.txt` (526)
- [ ] Adjudicate flagged: 8 (original) + 92 (expanded)
- [ ] Optional: re-run hard-set LLM judge on fresh Groq quota
      (392/600 calls were unanswered — selection-bias caveat)
- [ ] Restore PC sleep setting: `powercfg /change standby-timeout-ac 30`
      (disabled for the overnight runs)
- [ ] Git commit: everything after commit `a274364` is uncommitted
      (doc expansions, external_baselines/, expanded_eval/)
- [ ] Optional future work: RAGTruth data-to-text transfer experiment;
      number-word price normalization; unsupported-claim detection layer

## Cardinal rules that governed everything (do not break when extending)

1. **Never adjust numbers to look plausible** — make the test harder instead.
2. **Never let a system grade its own output** (checker ≠ referee; clean
   labels need human audit).
3. **Seeds everywhere** — same inputs must regenerate identical datasets.
4. **Version results with their dataset** — numbers from different test-set
   versions are not comparable.
5. **New experiments in new folders** — completed results are never modified.
