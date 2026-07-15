# Contradiction Detector — Evaluation Plan (Novelty 3)

Status: PLAN (awaiting approval before implementation)
Created: 2026-07-12

The system under evaluation: **Evidence-Anchored Session-Graph Contradiction
Detector** (`memory/core/contradiction_detector.py`) — persisted NetworkX
session graph holding DB-evidence ground truth per product, Groq claim
extraction from the LLM response, `values_contradict()` string gate, DeBERTa
NLI confirmation gate, and automatic in-place response correction.

---

## 0. The plan in plain language (read this first)

**The question:** the novelty claims *"my detector catches cases where the
LLM says something about a product that contradicts what was established
earlier in the session, and fixes it before the user sees it."* To prove that
with numbers we need: (1) a test set with known ground-truth labels,
(2) baselines to compare against, (3) metrics and figures.

**Step 1 — Build the test set (inject contradictions ourselves).**
Real contradictions are rare and hand-labelling is infeasible, so we use the
standard trick (HaluEval, our own hallucination eval): take *correct*
responses and *deliberately corrupt* them — then we know with certainty which
cases are contradictory, because we made them so.

1. Run ~35 scripted conversations (5–8 turns each) through the real
   pipeline, capturing per turn: evidence bundle, session-graph state, and
   the LLM response. E.g. Turn 1 shows *"SC COLUMBUS blouse — Black —
   £57.14"*; Turn 4 mentions the blouse again, correctly, as Black.
2. Correct responses = **clean cases** (label: no contradiction).
3. Copy a clean case and corrupt one value in the *response text only* —
   Turn 4's "Black" → "Navy". The graph still says Black. That copy is a
   **contradictory case**, and we record which attribute, which product, and
   how many turns back the truth was established (turn distance = 3 here).
4. Five corruption types (colour, price, name, type, cross-item swap) →
   ~350 contradictory + ~150 clean ≈ **500 labeled cases**.
5. Plus ~50 tricky clean cases ("sports bra" when the DB says "Bra") to
   check the detector doesn't cry wolf.

**Step 2 — Run every system on the same 500 cases.** Each system answers one
yes/no question per case: *"does this response contradict the session's
established facts?"* Ours answers via graph + Groq + NLI; the baselines via
history-NLI, utterance-pair NLI, LLM judge, string-match only, and the
external released tools. Since we planted the labels, every answer is
scorable.

**Step 3 — Compute the numbers.** Recall (of the planted contradictions, how
many were caught?), Precision (of everything flagged, how much was real?),
F1 (the headline number), and **recall vs turn distance** — the signature
result: the graph never forgets, text-based baselines degrade as the session
grows.

**Step 4 — The correction experiment (ON/OFF).** Detection alone isn't the
full novelty — the response also gets *fixed*. Run all contradictory cases
with the detector OFF (all reach the user) vs ON, and a referee script checks
each shipped response for the correct value. Result: *"the detector reduced
user-facing contradictions from 100% to X%, and fixed the text correctly Y%
of the time when it detected."* Mirrors the hallucination loop experiment
(100% → 7.8%).

**Step 5 — Figures.** P/R/F1 comparison bars, false-alarm chart, recall by
corruption type, recall-vs-turn-distance line, ON/OFF mitigation chart,
ablations, threshold sweep, latency table — same style as the existing
hallucination figures.

**In one sentence:** we plant known contradictions into real captured
sessions, ask our detector and 5–6 baselines to find them, score everyone
with precision/recall/F1, and separately show the fix-loop turns detections
into corrected responses.

---

## 1. How other researchers evaluate this task (literature survey)

### 1.1 Dialogue contradiction detection benchmarks

| Work | Task framing | Metrics | Key numbers |
|---|---|---|---|
| **DECODE** (Nie et al., ACL 2021, "I like fish, especially dolphins") | Binary classification: does the final utterance contradict the preceding dialogue? 6 test groups incl. human-bot | Precision, Recall, F1, **AUC** | Best PLM ≈ 80.9% on Test-Strict; humans much higher. Introduced the **structured utterance-pair detector** (RoBERTa on utterance pairs) vs unstructured (whole-context) — structured wins |
| **CI-ToD** (Qin et al., EMNLP 2021) | Task-oriented dialogue consistency with **three fine-grained labels**: QI (query inconsistency), HI (history inconsistency), **KBI (knowledge-base inconsistency)** | Per-label F1 + overall accuracy | SOTA 51.3% overall vs human 93.2% — the task is hard. KBI is *exactly* our setting: response vs structured KB |
| **Self-contradictory Hallucinations** (Mündler et al., ICLR 2024) | Detect + **mitigate** self-contradiction pairs in LLM text | Detection F1; mitigation = % contradictions removed while preserving fluency/informativeness | ~17.7% of ChatGPT sentences self-contradict; ~80% detection F1; two-part eval (detect, then fix) is the template for our detect→correct loop |
| **Model-generated contradictory responses** (Sato et al., 2024, arXiv:2403.12500) | Large dataset of *real model-generated* (not human-written) contradictions; shows synthetic/model-generated contradictions are the accepted way to build test sets | P/R/F1 of suppression methods | Confirms our injection-based test-set construction is standard practice |
| **SKG-Eval** (2026, arXiv:2605.16650) | Multi-turn dialogue evaluation via an **incremental semantic knowledge graph**; detects cross-turn contradiction/entity inconsistency with a "geometric contradiction engine" | Correlation with human judgment; contradiction certificates | Closest published relative of our session-graph idea — strong citation for motivating the architecture; note it *evaluates* dialogues offline, ours *intervenes* online |
| **HalluDial** (2024, arXiv:2406.07070) | Dialogue-level hallucination evaluation benchmark (information-seeking) | Detection acc/F1, localization, rationale quality | Useful for framing "dialogue-level" vs "sentence-level" checking |
| **RefChecker** (Amazon, 2024) / FacTool / FActScore | **Extract-then-verify**: decompose response into claim triplets, verify each against reference | Claim-level P/R/F1, agreement with human labels | Validates our Groq-extraction stage: claim-triplet granularity beats response/sentence granularity by 6.8–26.1 pts. Our (article_id, attribute, value) tuples are exactly claim triplets |

### 1.2 What this survey implies for our evaluation

1. **Binary detection P/R/F1 (+AUC) on a labeled set** is the universal core —
   same as our hallucination chapter. Positive class = contradictory.
2. **Fine-grained labels by source/attribute** (CI-ToD's QI/HI/KBI) → we report
   per-attribute (colour/price/name/type) and per-action results.
3. **Detection and mitigation are evaluated separately** (Mündler) → one
   experiment for the detector, one ON/OFF experiment for the correction loop
   (we already did exactly this for hallucination: `loop_mitigation/`).
4. **Synthetic injection of contradictions is the accepted construction
   method** (DECODE used crowdworkers *writing* contradictions; 2403.12500 and
   HaluEval generate them; our hallucination eval already used programmatic
   corruption). Cross-turn injection = corrupt a later turn's mention of a
   product introduced earlier.
5. **Structured beats unstructured is a publishable claim** (Nie et al. showed
   it for utterance pairs) → our headline comparison is *evidence-anchored
   graph (structured)* vs *NLI-over-dialogue-history (unstructured)* vs
   *LLM judge*.
6. **No existing CRS system does online cross-turn correction** — CRS
   evaluation surveys (2025) measure item/content hallucination but none
   intervene mid-session. That is the gap our novelty fills, and the eval
   should demonstrate it with a turn-distance figure no baseline can match.

---

## 2. Evaluation design

Two experiments, mirroring the hallucination chapter's structure
(detection benchmark → mitigation experiment), plus ablations.

### Experiment A — Detection benchmark

**A1. Test-set construction (session-level, injection-based).**

1. Run ~30–40 scripted multi-turn conversations (5–8 turns each) through the
   live pipeline with the eval-capture hook, capturing per turn:
   evidence bundle, session graph state (nodes before update), response text,
   turn_id, session_id. (~200–280 turn-level cases.)
2. Turns that pass the live checks become **clean cases**.
3. **Inject contradictions** into clean responses — the response mentions a
   product whose ground truth lives in the graph from an *earlier* turn:

   | Corruption type | Example | Tests |
   |---|---|---|
   | `colour_drift` | Turn 1: "SC COLUMBUS blouse — Black"; corrupt Turn 4 mention to "Navy" | core cross-turn drift |
   | `price_drift` | £57.14 → £52.99 in a later mention | numeric comparison + tolerance |
   | `name_drift` | product renamed to a different catalog name | extraction keying by article_id |
   | `cross_item_swap` | two products' attributes exchanged (both values exist in session) | association errors — the blind spot of presence-checkers (replicating our external-baselines finding) |
   | `type_drift` | "Blouse" → "Cardigan" (a real different type, not a subtype) | NLI semantic judgment |

   Injection is stratified by **turn distance** d = (corrupted turn − turn the
   product was introduced): d ∈ {0, 1, 2, 3+} so we can plot recall vs
   distance.
4. **Hard negatives (benign variation set)** — clean cases that *look* like
   contradictions but are not; these stress the NLI gate:
   - subtype paraphrase: "sports bra" vs DB "Bra"
   - name variant: "Utah blouse" vs DB "Utah"
   - price restated with rounding: "£24.76" vs "about £25" *(if present)*
   - reordered/partial attribute mentions
5. Target size: **~500 cases** (≈350 injected contradictory + ≈150 clean incl.
   ~50 hard negatives), seeded and reproducible — same scale and style as the
   original 238-case hallucination set, expandable later like the 4000-row
   suite if time allows.

**A2. Systems compared** (positive class = contradictory):

| System | Type | What it is |
|---|---|---|
| **Ours** (full detector) | proposed | graph + Groq extraction + string gate + NLI confirmation |
| **Dialogue-history NLI** (DECODE/SummaC-style, *unstructured*) | reimplemented baseline | DeBERTa NLI over (concatenated prior bot statements about the session, current response); flag if any pair P(contradiction) > 0.5 |
| **Utterance-pair NLI** (Nie et al. *structured* baseline) | reimplemented baseline | DeBERTa NLI on each (earlier product sentence, current product sentence) pair |
| **LLM judge** (Mündler/RAGAS-style) | reimplemented baseline | Groq llama-3.1-8b-instant given session history + current response: "does this contradict anything previously established?" |
| **String-match only** (ablation-as-baseline) | ablation | our pipeline with NLI gate removed (`values_contradict()` decides alone) |
| **SummaC-Conv / HHEM-2.1** on serialized session history | external released tools | reuse the already-installed external-baselines harness; context = serialized session facts, hypothesis = current response |

**A3. Metrics** — computed with the same `compute_metrics` functions as the
hallucination eval (imported, not duplicated):

- Precision, Recall, F1, balanced accuracy, AUC (from raw scores)
- Recall by corruption type × system
- **Recall vs turn distance × system** ← the signature cross-turn result
- False-alarm count on clean + hard-negative cases (NLI-gate value)
- Per-attribute F1 (colour / price / name / product_type)

### Experiment B — Correction (mitigation) experiment

ON/OFF design, identical in spirit to `loop_mitigation/`:

1. Take all injected-contradiction cases; run the full
   `check_and_resolve()` pipeline **ON** vs **OFF (detector disabled)**.
2. A programmatic referee (string check of the final response against the
   evidence value — the injection makes ground truth known by construction)
   scores every *shipped* response.
3. Metrics:
   - **User-facing contradiction rate**: OFF (=100% by construction) vs ON
   - **P(correct fix | detected)** — did `_fix_response_text()` actually
     produce a consistent response (and not mangle the text)?
   - **Collateral damage rate** — corrections that changed text they should
     not have (e.g. value string appears twice in the response)
   - Optional fluency spot-check: 50-case human/LLM-judge rating that
     corrected responses remain grammatical (Mündler's
     informativeness/fluency-preservation criterion)

### Experiment C — Ablations & sensitivity

| Ablation | Question it answers |
|---|---|
| − NLI gate (string only) | how many false alarms does NLI prevent? (expected: hard negatives all fire) |
| − Groq extraction (NLI direct on response vs graph facts) | is structured extraction necessary, or would sentence-level NLI suffice? |
| − session graph (compare only against *current-turn* evidence) | isolates the cross-turn contribution — expect recall at distance d≥1 to collapse |
| NLI threshold sweep 0.1–0.9 | operating-point sensitivity (mirror of fig5) |

Plus **system-cost table**: added latency per turn (Groq extraction ~x ms,
NLI ~y ms, Mongo graph I/O ~z ms), graph size growth per session, API cost —
vs LLM-judge baseline latency.

### Figures (thesis-ready, numbered continuing the existing convention)

1. **fig-C1** — grouped bar: P / R / F1 / balanced acc. per system
2. **fig-C2** — false alarms on clean + hard-negative cases per system
3. **fig-C3** — recall by corruption type × system (grouped bars; expect
   cross_item_swap to separate ours from presence-checkers again)
4. **fig-C4** — **recall vs turn distance (line plot)** — graph-based stays
   flat; history-NLI and LLM judge degrade as context grows
5. **fig-C5** — ON/OFF user-facing contradiction rate + P(fix | detect)
   (mitigation waterfall)
6. **fig-C6** — ablation bars (full vs −NLI vs −extraction vs −graph)
7. **fig-C7** — NLI threshold sweep
8. **fig-C8** — latency/cost comparison per check

### Thesis narrative the numbers should support

1. Cross-turn contradictions are real and uncaught by single-turn checkers
   (motivation, fig-C4 OFF/−graph collapse).
2. Structured evidence-anchored detection beats unstructured NLI-over-history
   — extends Nie et al.'s structured-beats-unstructured finding to
   evidence-grounded CRS (fig-C1/C2).
3. Association errors (cross-item) require id-keyed claim extraction —
   replicates the hallucination chapter's external-baselines phenomenon in
   the cross-turn setting (fig-C3).
4. Detection converts to user-visible benefit via automatic correction
   (fig-C5) at negligible latency/cost (fig-C8).

---

## 3. Realistic workload estimate

| Step | Effort | Notes |
|---|---|---|
| Capture hook for contradiction cases (graph state + evidence + response) | 0.5 day | extend existing `capture.py` pattern |
| Scripted conversations run (~35 sessions) | 0.5 day | reuse driver scripts from hallucination eval |
| Injection script (`corrupt_sessions.py`) | 1 day | port `corrupt_cases.py` logic, add turn-distance stratification + hard negatives |
| Baselines (history-NLI, utterance-pair NLI, LLM judge, string-only) | 1 day | DeBERTa + Groq already in project |
| External tools on serialized history | 0.5 day | harness already exists |
| ON/OFF correction experiment + referee | 0.5 day | port `loop_mitigation/referee.py` |
| Ablations + threshold sweep | 0.5 day | offline from stored scores |
| Figures + RESULTS.md | 0.5 day | port `make_figures.py` |
| **Total** | **~5 days** | all local/free except Groq calls (~1–2k requests) |

## 4. Key references to cite

- Nie et al., ACL 2021 — DECODE, arXiv:2012.13391
- Qin et al., EMNLP 2021 — CI-ToD, arXiv:2109.11292
- Mündler et al., ICLR 2024 — Self-contradictory hallucinations, arXiv:2305.15852
- Welleck et al., ACL 2019 — Dialogue NLI
- Sato et al., 2024 — Model-generated contradictory responses, arXiv:2403.12500
- Hu et al., 2024 — RefChecker, arXiv:2405.14486 (claim-triplet granularity)
- Laban et al., TACL 2022 — SummaC (already cited in hallucination chapter)
- SKG-Eval, 2026 — arXiv:2605.16650 (graph-based multi-turn evaluation; closest architectural relative)
- HalluDial, 2024 — arXiv:2406.07070 (dialogue-level hallucination benchmark)
