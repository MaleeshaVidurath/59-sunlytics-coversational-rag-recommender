# Contradiction Detector — Evaluation Results

Generated: 2026-07-12 · Novelty 3 (Evidence-Anchored Session-Graph Contradiction
Detector) · Machine-readable: `results_contra_eval.json`,
`results_correction_eval.json` · Figures: `figures/figC1–figC5`

---

## 1. Evaluation setup

**System under evaluation.** The Evidence-Anchored Session-Graph Contradiction
Detector (`memory/core/contradiction_detector.py`): a persisted NetworkX session
graph holds the authoritative DB-evidence value of every product the assistant
has discussed in the session; Groq claim extraction pulls the values the LLM
actually wrote in each response as `(article_id, attribute, value)` triplets;
`values_contradict()` flags mismatches against the graph; a DeBERTa NLI gate
(`cross-encoder/nli-deberta-v3-base`) confirms each mismatch is a genuine
semantic contradiction; confirmed contradictions are rewritten in the response
before the user sees them.

**Test-set construction (synthetic cross-turn corruption).** Following the
accepted practice for contradiction/hallucination benchmarks — DECODE
(Nie et al., ACL 2021) crowd-authored contradictions; HaluEval and Sato et al.
(2024) generated them; our own hallucination chapter corrupted fields — we
inject contradictions into real captured sessions:

1. 37 scripted multi-turn conversations (5–7 turns) were run through the **live
   production pipeline**. A capture hook recorded, per checked turn, the evidence
   ground truth, the session-graph state before the turn, the LLM response, and
   the turn ordinal. **191 turns ran with 0 failures; 188 factual turns were
   captured.**
2. Each captured clean response became a **clean** case.
3. Each clean response was programmatically corrupted — one attribute value of
   one product changed in the response while the ground truth was left
   untouched — producing **contradiction** cases (wrong by construction), each
   tagged with a **turn distance** *d* = (corrupted turn − turn the product
   entered the graph).
4. Benign subtype paraphrases ("Dress" → "maxi dress") were injected as
   **hard-negative** cases (not contradictions) to stress the NLI gate.

**Full labeled set: 1346 cases** — 188 clean, 1013 contradiction, 145 hard
negative. Corruption counts: name_drift 240, price_drift 228, colour_drift 215,
type_drift 190, cross_item_swap 140, subtype_paraphrase 145. Contradiction
turn-distance spread: d=0 455, d=1 171, d=2 115, d≥3 272.

**Headline comparison** was run on a **stratified 599-case sample** (seed 123,
proportions preserved across label × corruption-type × distance): 450
contradiction + 149 negative (84 clean + 65 hard negative).

**Systems compared** (positive class = contradiction):

| System | Description |
|---|---|
| **Ours** | session graph + claim extraction + `values_contradict()` + DeBERTa NLI confirmation |
| **String-only (−NLI)** | ablation: our pipeline with the NLI gate removed (string comparison decides alone) |
| **History-NLI (unstructured)** | DECODE/SummaC-style: DeBERTa NLI over every (session fact, response sentence) pair |
| **Utterance-pair NLI (structured)** | Nie et al. (2021): NLI on (fact about product X, sentence mentioning X) pairs |
| **LLM judge** | Groq judges whether the response contradicts the established session facts |

Metrics use the **same `compute_metrics` implementation as the hallucination
chapter** (imported, not duplicated): Precision, Recall, F1, balanced accuracy,
with Wilson and bootstrap 95% CIs.

> **Methodology note — eval extraction model.** In production the claim
> extractor runs on Groq `llama-3.1-8b-instant`. That model's free-tier ceiling
> of **6000 tokens/min** cannot sustain 599 sequential extractions: >70% of
> calls return empty (rate-limited), which would understate recall as an API
> artifact rather than a detector property (a first run at that pace produced a
> spurious recall of 0.16). For the evaluation only, extraction was therefore
> run on **Llama 4 Scout** (`meta-llama/llama-4-scout-17b-16e-instruct`,
> 30 000 TPM, separate daily bucket) using the **identical prompt and JSON
> parsing** — only the model id differs. This is a deliberate, documented
> eval-only deviation; the detection logic (graph, gates, NLI, correction) is
> unchanged from production.

---

## 2. Headline result (figC1, figC2)

Detection accuracy on the 599-case sample (positive = contradiction):

| System | Precision | Recall | F1 | Balanced acc. | False alarms (clean+hard-neg) |
|---|---|---|---|---|---|
| **Ours (graph + NLI)** | **0.983** | 0.773 | 0.866 | **0.867** | **6** (0 clean, 6 hard-neg) |
| String-only (−NLI) | 0.844 | 0.856 | 0.850 | 0.690 | 71 (22 clean, 49 hard-neg) |
| History-NLI (unstructured) | 0.764 | 0.958 | 0.850 | 0.533 | 133 (74 clean, 59 hard-neg) |
| Utterance-pair NLI (structured) | 0.772 | 0.909 | 0.835 | 0.548 | 121 (66 clean, 55 hard-neg) |
| LLM judge | 0.957 | 0.850 | 0.900 | 0.867 | 15 (5 clean, 10 hard-neg)¹ |

¹ LLM judge answered 523/599 cases (76 unparseable verdicts dropped); its
false-alarm totals are over the answered clean (72) and hard-neg (57) subsets.

**95% confidence intervals (bootstrap, 2000 resamples)** for the two headline
metrics — F1 and balanced accuracy:

| System | F1 [95% CI] | Balanced acc. [95% CI] |
|---|---|---|
| **Ours** | 0.866 [0.840, 0.889] | 0.867 [0.840, 0.891] |
| String-only | 0.850 [0.825, 0.874] | 0.690 [0.647, 0.731] |
| History-NLI | 0.850 [0.826, 0.875] | 0.533 [0.508, 0.561] |
| Utterance-pair NLI | 0.835 [0.808, 0.861] | 0.548 [0.517, 0.583] |
| LLM judge | 0.900 [0.877, 0.923] | 0.867 [0.834, 0.898] |

The CIs make the statistical picture precise: **Ours and the LLM judge are
indistinguishable on balanced accuracy** (intervals overlap almost entirely),
and **both cleanly separate from the three NLI-based systems** on balanced
accuracy (baseline intervals top out at 0.58, ours starts at 0.84 — no overlap).
On F1 the LLM judge's interval [0.877, 0.923] sits just above ours [0.840,
0.889], a small but real edge from its higher recall.

**Reading the table.** Our detector has the **highest precision (0.983) of any
system and zero false alarms on the 84 clean responses** — it never "corrects" a
correct answer. Its balanced accuracy (0.867) is the best of all systems, tied
with the LLM judge. The two NLI-over-text baselines reach high recall
(0.91–0.96) but are **unusable in production**: they wrongly flag 66–74 of 84
clean responses, collapsing balanced accuracy to ~0.53–0.55 (near chance) — in
the live system that would trigger a pointless correction on almost every turn.

**Honest positioning vs the LLM judge.** The LLM judge is a strong baseline: it
edges our detector on F1 (0.900 vs 0.866) via higher recall at comparable
precision, and matches it on balanced accuracy. Our detector's distinguishing
value is therefore not a raw-accuracy win over the judge but: (i) the **highest
precision with zero clean false alarms**, (ii) an **auditable structured trace**
(the session graph + `contradiction_log`), and (iii) **automatic correction**
(§5) — none of which the judge provides.

---

## 3. The NLI-gate ablation — where precision comes from (figC1, figC2)

Comparing **Ours** against **String-only (−NLI)** isolates the DeBERTa gate:

| | Recall | False alarms | Balanced acc. |
|---|---|---|---|
| String-only (string comparison alone) | 0.856 | 71 | 0.690 |
| **+ NLI confirmation (Ours)** | 0.773 | **6** | **0.867** |

Removing the NLI gate raises recall modestly (0.773 → 0.856) but **multiplies
false alarms 6 → 71** — it fires on every benign subtype paraphrase
("sports bra" vs "Bra") and surface variant. The NLI gate trades ~8 points of
recall for a **12× reduction in false alarms** and a +0.18 jump in balanced
accuracy. This is the core design justification for the two-gate architecture.

---

## 4. Detection by corruption type (figC3)

Recall on contradiction cases, per corruption type:

| Corruption | Ours | String-only | History-NLI | Uttr-pair NLI | LLM judge |
|---|---|---|---|---|---|
| colour_drift | 0.969 | 0.969 | 0.969 | 0.969 | 0.988 |
| price_drift | 0.970 | 0.990 | 0.980 | 0.921 | 0.893 |
| name_drift | 0.528 | 0.689 | 0.915 | 0.849 | 0.646 |
| type_drift | 0.741 | 0.847 | 0.965 | 0.882 | 0.897 |
| cross_item_swap | 0.613 | 0.758 | 0.968 | 0.936 | 0.870 |

Our detector is strongest on the two fields it verifies most directly —
**colour (0.97) and price (0.97)** — matching the design docs. It is weaker on
**name_drift (0.53)**: the NLI gate conservatively rejects many name
substitutions (a swapped catalog name still reads as a plausible product name,
so NLI often scores it neutral rather than contradiction), which is the price of
the low false-alarm rate. **cross_item_swap (0.61)** is the hardest: both values
still exist in the session, so only the association is wrong — the NLI premise
(built from the correct node) does not always separate a swapped-but-valid value.

---

## 5. Correction experiment — ON vs OFF (figC5)

Detection is half the novelty; the detector also rewrites the response. Scored
by an **independent, model-free referee** (grades against the corruption ground
truth; shares no logic with the detector), from the same single extraction pass:

| Metric | Value |
|---|---|
| User-facing contradiction rate, detector **OFF** | 1.000 (450/450, by construction) |
| User-facing contradiction rate, detector **ON** | **0.322** (145/450) |
| Detection rate | 0.773 (348/450) |
| P(correct fix \| detected) | **0.876** (305/348) |
| Collateral-damage rate (negatives wrongly altered) | 0.040 (6/149) |

The detector **reduces user-facing contradictions from 100% to 32%**, and when
it detects a contradiction it produces a correct, consistent response **87.6% of
the time**, altering only 4% of responses it should have left alone.

Residual contradictions after correction, by type: colour 0.04, price 0.03,
type 0.29, name 0.48, **cross_item_swap 1.00**. The residual is dominated by
cross-item swaps, which the current `_fix_response_text` (a wrong→correct string
replacement) **cannot repair**: when two valid values are transposed, replacing
one re-introduces a clash rather than restoring the association. Fixing swaps
would require a position-aware rewrite — a clear, honest direction for future
work.

---

## 6. On the turn-distance analysis (figC4)

Recall by turn distance (how many turns back the product's truth was
established):

| System | d=0 | d=1 | d=2 | d≥3 |
|---|---|---|---|---|
| Ours | 0.887 | 0.618 | 0.840 | 0.653 |
| History-NLI | 0.995 | 0.868 | 0.980 | 0.942 |
| LLM judge | 0.864 | 0.768 | 0.864 | 0.876 |

Our detector maintains substantial recall at every distance — a product
introduced three-plus turns earlier is still checked (0.65), because the graph
persists its DB value across the whole session. **Caveat (stated for honesty):**
in this test set the corrupted response is the *current* turn's output, and the
contradicted product is therefore also present in the current turn's evidence.
The baselines were fed the same serialized session facts, so they do *not* decay
with distance here. This harness thus demonstrates that the graph **retains**
product truth over long sessions, but it does **not** isolate a case only the
graph can catch (a response referencing a product absent from the current
retrieval). Constructing that stricter cross-turn probe — and the corresponding
`−graph` ablation — is the natural next experiment.

---

## 7. Threats to validity

- **Synthetic contradictions.** Injected corruptions may not perfectly mirror
  natural LLM drift; mitigated by drawing from the LLM's own field vocabulary
  and including benign hard negatives.
- **Clean-label assumption.** "Clean" cases shipped through the live checker; a
  manual audit sheet (`clean_audit.txt`) accompanies the set.
- **Eval extraction model.** Extraction ran on Llama 4 Scout, not production's
  llama-3.1-8b, for rate-limit reasons (§1 note) — same prompt/parsing.
- **Distance harness.** Does not yet isolate the pure cross-turn case (§6).
- **Sample.** Headline numbers are on a stratified 599-case sample; the full
  1346-case set is retained for robustness re-runs.

## 8. Analyses run vs deferred

For an honest record of scope, against the 8-figure plan in
`EVALUATION_PLAN.md`:

**Run and reported here (figC1–figC5):**
- Detection accuracy vs 4 baselines (P/R/F1/balanced acc. + 95% CIs) — §2
- NLI-gate ablation (Ours vs String-only) — §3
- Recall by corruption type — §4
- Correction ON/OFF experiment with independent referee — §5
- Recall by turn distance — §6

**Planned but NOT run (deferred):**
- **`−graph` and `−extraction` ablations** — isolating the session graph's and
  the Groq-extraction stage's individual contributions. The `−graph` ablation is
  the most important gap (see §6): it would compare against current-turn evidence
  only, and needs a stricter test set where the contradicted product is absent
  from the current retrieval.
- **NLI threshold sweep** — `run_contra_eval.py` does not store per-check NLI
  scores, so an offline sweep is not possible from the current outputs; it would
  need a re-run that logs raw logits.
- **Latency / cost table** — per-turn added latency (Groq extraction + NLI +
  graph I/O) and API cost vs the LLM-judge baseline were not measured.
- **External released tools** (SummaC-Conv, HHEM) on serialized session history —
  the harness exists in the hallucination chapter but was not applied here.

None of these change the reported findings; they are additional evidence that
would strengthen the chapter if time/quota allow.

---

## 9. Reproduction

From `m3_implementation/` with the stack running (MongoDB, Redis, PostgreSQL,
Qdrant) and `GROQ_API_KEY` set:

```bash
# Step 1 — capture sessions through the live pipeline (writes captured_sessions.jsonl)
python test_result/contradiction_result/collect_sessions.py

# Step 2 — build the labeled test set (writes labeled_test_set.jsonl, 1346 cases)
python test_result/contradiction_result/corrupt_sessions.py

# Step 3 — detection eval. NLI baselines + LLM judge on llama-3.1-8b;
#          ours/string extraction on Scout to avoid the 6000-TPM ceiling.
OURS_EVAL_DELAY=2.0 \
python test_result/contradiction_result/run_contra_eval.py --sample 600

# (if ours/string were run separately) merge them into the main results file
python test_result/contradiction_result/merge_ours_results.py

# Correction experiment — scored offline from the detection pass's sidecar
python test_result/contradiction_result/run_correction_eval.py \
    --offline test_result/contradiction_result/ours_case_detail.jsonl

# Figures
python test_result/contradiction_result/make_contra_figures.py
```

Determinism: the stratified sample uses seed 123; the corruption RNG uses
seed 42 — the same input files always yield the same test set and sample.

---

## 10. References

DECODE (Nie et al., ACL 2021) · CI-ToD (Qin et al., EMNLP 2021) ·
Self-contradictory hallucinations (Mündler et al., ICLR 2024) ·
RefChecker (Hu et al., 2024) · SummaC (Laban et al., TACL 2022) ·
SKG-Eval (2026) · HalluDial (2024). Full list in `EVALUATION_PLAN.md`.
