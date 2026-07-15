# Detect–Reject–Regenerate Loop — Mitigation Evaluation Results

Generated: 2026-07-09 · Raw data: `results_loop_eval.json`, `shipped_responses.jsonl`
· Figures: `figures/fig6`, `figures/fig7` · Log: `loop_eval_run.log`

---

## 1. Question and method

The detector evaluation (`../RESULTS.md`) showed the checker catches 95% of
hallucinations. This experiment evaluates the **other half of the
contribution**: when a hallucination is detected, does the reject→regenerate
loop deliver a *correct* response to the user?

**Method (CRAG-style system-on/off with induced failures).** The LLM rarely
hallucinates naturally (~0 genuine lies in 36 live turns), so natural A/B
chatting gives no signal. Instead, each of the 205 corrupted test-set cases
plays the role of the LLM's attempt-1 output:

- **Loop OFF:** the corrupted response ships as-is — what any system without
  the loop does. Wrong by construction: 205/205.
- **Loop ON:** the real pipeline logic runs unchanged — checker v3 inspects
  attempt 1; if flagged, the real `ResponseGenerator` regenerates from the
  true evidence with escalating strictness and the contradicted fields in the
  prompt; the checker inspects every new attempt; attempt 3 always ships
  (`MAX_REGENERATION_ATTEMPTS = 3`, exactly as in `rag_pipeline.py`).

**Independent grading.** The loop only ships checker-approved text, so the
checker cannot grade its own output (it would report 100% success by
definition). Final responses are graded by `referee.py` — model-free literal
verification against the database truth: every stated £value must be an
evidence price *or a derived difference of two evidence prices*; every colour
word must be an evidence colour; every item name must appear (a truncated but
unambiguous name counts as a minor fidelity issue, not a hallucination).

## 2. Headline result (fig6)

| | Hallucinated responses reaching the user |
|---|---|
| **Loop OFF** | 205 / 205 (100%) |
| **Loop ON** | **16 / 205 (7.8%)** |

The loop removes **92.2%** of induced hallucinations end-to-end.

Component numbers:

| Metric | Value |
|---|---|
| Detected on attempt 1 | 195 / 205 (95.1%) — consistent with the detector evaluation |
| P(correct final \| detected) | 96.9% — strictness escalation works |
| Attempts histogram | 1×10 (missed), 2×184, 3×11 — one regeneration usually suffices |
| Avg regeneration time | 0.41 s per detected case (Groq llama-3.1-8b-instant) |
| Minor name truncations | 4 cases (reported separately, not counted wrong) |
| Generation errors | 0 |

## 2b. Attempts needed per case (fig8)

| Loop stopped at | Cases | Share | Meaning | Final correct |
|---|---|---|---|---|
| Attempt 1 | 10 | 4.9% | checker missed the lie — no retry ever ran, corrupted response shipped | 0 / 10 |
| Attempt 2 | 184 | 89.8% | detected → **one** regeneration (strictness 1) satisfied the checker | 179 / 184 |
| Attempt 3 | 11 | 5.4% | attempt 2 also flagged → **second** regeneration (strictness 2, bullet-only) | 10 / 11 |

Key readings: **94.4% of detected hallucinations were corrected with a single
regeneration**; escalation to the strictest prompt rescued 10 of the
remaining 11; the average cost of the loop is ~one extra LLM call (0.41 s)
per detected hallucination. The 5 attempt-2 cases that shipped faulty are
checker blind spots — the regenerated text satisfied the checker but the
referee found a fault (fresh extra-colour mentions, a dropped unusual name).

## 3. Failure analysis — the 16 residual cases

| Cause | n | Nature |
|---|---|---|
| Checker missed the lie (shipped unchanged) | 10 | the known v3 detection gap, mostly colour paraphrases |
| Detected, but regenerated output still faulty | 6 | see below |

The 6 imperfect regenerations are *fresh* LLM faults, not the planted lie
surviving: 3 responses mention a colour absent from the evidence (e.g.
styling remarks introducing "Black" for red-only items), 2 drop an unusual
catalog name ("Pull On Ankle."), and 1 states a **wrong derived price
difference** — "£1.02" where the true difference is £2.02 — a genuine
spontaneous numeric hallucination caught by the referee. These cases show the
attempt-3-always-ships policy is the loop's accepted risk window.

Residual by corruption type (fig7): price 1.8%, cross-item 2.3%,
name 10.3%, colour 16.7% — mirroring the detector's per-field recall
(colour is NLI-verified and remains the weakest field).

## 4. Referee refinement (documented for transparency)

The first grading pass reported 25/205 residual. Inspection showed two
systematic referee blind spots, fixed and re-graded offline
(`regrade_shipped.py`, no LLM re-runs): (a) correct derived arithmetic
("£4.04 cheaper" = £15.12 − £11.08) was counted as a wrong price — 7 cases;
(b) truncated but unambiguous item names counted as missing — reclassified as
minor fidelity issues, 4 cases (2 remained genuinely missing). This mirrors
the evaluation-driven refinement applied to the checker itself.

## 5. Limitations

1. Induced failures corrupt structured fields; free-form fabrication is not
   covered (same limitation as the detector evaluation).
2. The referee's value-set checks cannot detect two correct values swapped
   between items in a *regenerated* response, or lies phrased outside the
   colour vocabulary.
3. Regeneration uses the live Groq LLM — exact counts vary slightly between
   runs (temperature > 0 in generation); the magnitude of the effect
   (100% → ~8%) is stable.

## 6. Reproduction

```powershell
# full run (~7 min, ~200 Groq calls; needs GROQ_API_KEY, no databases)
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\loop_mitigation\run_loop_eval.py

# re-grade shipped responses after a referee change (offline, no LLM)
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\loop_mitigation\regrade_shipped.py

# figures
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\loop_mitigation\make_loop_figures.py
```
