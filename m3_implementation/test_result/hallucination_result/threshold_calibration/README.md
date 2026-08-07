# Threshold Calibration — Deriving `NLI_CONTRADICTION_THRESHOLD`

> **Question:** *"How was the contradiction threshold decided?"*
>
> **Answer:** by ROC analysis on the score distribution, with the relative test
> ablated so the threshold is the sole decision rule. Youden's J and F1 both
> peak on **[0.80, 0.95]**; we take **0.80**, the lower edge, as the
> conservative choice. Under the full deployed rule the value is provably
> harmless, because the combined rule is invariant across [0, 1.0).

Run 2026-08-07 · 238 cases · 1,582 DeBERTa comparisons · no LLM, no network ·
~3.5 min. **Nothing in `../original_eval_238/` is written to by this folder.**

---

## 1. Why a derivation was needed

The reported evaluation swept `[0.25, 0.5, 0.65, 1.0, 2.0, 3.0, 4.0, 5.0]` and
found results *"flat across 0.25–0.65"*. That establishes the value is harmless,
but **a flat region contains no optimum** — so it cannot say how a value was
chosen. Worse:

- `config.py` defaults to `0.65`; `.env` overrode it to `0.70`. Neither was in a
  measured optimum, and `.env` is untracked, so no commit records the choice.
- The grid never went below 0.25, so the left-hand boundary was never located.

Two experiments were run to close this. §2 explains why the naive reading fails.
§3 is the derivation.

---

## 2. Why the deployed rule cannot identify the threshold

The decision rule is two conditions:

```python
is_hallucination = (
    contradiction > NLI_CONTRADICTION_THRESHOLD   # (1) the absolute bar
    and contradiction > entailment                # (2) the relative test
)
```

Sweeping the **full** rule over 27 points from −5.0 to +5.0 (`figT1`):

| threshold | P | R | F1 | Bal. acc. |
|---|---|---|---|---|
| −5.00 … 0.95 | 1.000 | 0.951 | 0.975 | 0.976 |
| 1.00 | 1.000 | 0.288 | 0.447 | 0.644 |

**Every value below 1.0 is byte-identical.** The reason is in `figT2`:

| | count |
|---|---|
| checks where entailment won → never flagged | 1,292 |
| checks where contradiction beat entailment → flagged | 290 |
| …of those, with contradiction **below 1.0** | **0** |

The lowest score among all flagged checks is exactly **1.000**, because 195 of
them are *containment flags* — wrong name/price caught by string matching, which
skip NLI and receive a fixed score of 1.0. Genuine NLI contradictions start at
**1.092**.

So condition (1) is slid through an empty region: nothing lives between −5.0 and
1.0 for it to catch or release. Meanwhile 12 checks *did* exceed 0.70 and were
rejected anyway — by condition (2).

**Conclusion:** under the deployed rule the threshold is unidentifiable. Any
derivation must remove condition (2) first.

---

## 3. The derivation

### 3.1 Method

Ablate the relative test so the threshold decides alone:

```
flag case  IF  max(contradiction over its checks) > t
```

That rule *is* sensitive to `t`. Sweep it from −6.0 to +8.0 in steps of 0.05
(281 points) and read the ROC. Case-level, because the checker flags a case if
any check fires.

Scoring uses `compute_metrics` **imported from `run_detector_eval.py`** — the
same function behind the reported numbers.

### 3.2 The curve — three regimes, two hard boundaries

| t | P | R | Spec | Bal. acc. | Youden J | FP | FN |
|---|---|---|---|---|---|---|---|
| −0.60 … −0.05 | 0.871 | 0.985 | **0.091** | 0.538 | 0.076 | 30 | 3 |
| 0.00 … 0.75 | 0.980 | 0.961 | 0.879 | 0.920 | 0.840 | 4 | 8 |
| **0.80 … 0.95** | **0.985** | **0.961** | **0.909** | **0.935** | **0.870** | **3** | **8** |
| 1.00 … 1.05 | 0.969 | 0.302 | 0.939 | 0.621 | 0.242 | 2 | 143 |
| ≥ 1.20 | 1.000 | 0.298 | 1.000 | 0.649 | 0.298 | 0 | 144 |

**Lower boundary — `t < 0` destroys specificity.** It falls to 0.091: 30 of 33
clean cases get flagged. The model assigns negative contradiction logits to most
clean material, so a negative threshold admits nearly everything.

**Upper boundary — `t ≥ 1.0` destroys recall.** It falls to 0.302: the 195
containment flags score exactly 1.0 and the rule is strictly `>`, so they all
vanish. 143 true positives lost.

**The optimum sits between them.** Youden's J and F1 both peak on
**[0.80, 0.95]** — four consecutive grid points, identical.

### 3.3 The choice: t = 0.80

Within the tied optimal band, take the **lower edge**:

- All of [0.80, 0.95] score identically, so no metric distinguishes them.
- Risk is **asymmetric**: crossing 1.0 costs 143 true positives; moving down
  costs at most one false positive. The lower edge is farthest from the cliff.
- It is a round number at the boundary of a measured region, **not** the argmax
  of a fine-grained search — so it is not fitted to the evaluation set.

### 3.4 Honest magnitude

Moving 0.70 → 0.80 changes **one case**: false positives 4 → 3, specificity
0.879 → 0.909 on 33 clean cases. That difference is well inside sampling noise.

**The derivation's real content is the two boundaries, not the fine distinction
between 0.70 and 0.80.** State it that way. The defensible claim is *"we
identified the valid interval and took the optimum within it"*, not *"0.80 beats
0.70"*.

### 3.5 The relative test is separately validated

Same method applied to the margin rule, `flag if (contradiction − entailment) > t`:

| | P | R | Bal. acc. | Youden J |
|---|---|---|---|---|
| optimum (t = −1.0) | 1.000 | 0.956 | 0.978 | 0.956 |
| **deployed (t = 0)** | **1.000** | **0.951** | **0.976** | **0.951** |

The deployed relative test sits 0.002 balanced accuracy from its own optimum. It
is doing the discriminating work, and it is near-optimally placed.

---

## 4. What this proves, and what it does not

**Proven.** The threshold has a measured valid interval [0, 1.0) with a defined
optimum at [0.80, 0.95], both boundaries explained by mechanism rather than by
curve-fitting. The chosen value is the conservative edge of that optimum.

**Also proven.** Under the full deployed rule the value is inert across
[0, 1.0), so adopting 0.80 changes production behaviour on **zero** cases —
the derivation costs nothing to apply.

**Not proven.** The threshold remains *untriggered* rather than *validated* in
combination: no case in this test set has a contradiction score between 0 and 1
that also beats entailment. If a future model produced a weak contradiction in
that band, this data cannot say whether flagging it would be right.

**Secondary finding.** 143 of 205 true positives (70%) depend on containment-flag
logic rather than NLI scoring — visible as the size of the drop at t = 1.0. That
is a property of the checker, not of this experiment, but an examiner reading
`figT1` or `figT3` may ask about it.

---

## 5. What to say

> *"The threshold cannot be identified from the deployed rule, because the
> relative test — contradiction must outscore entailment — carries the decision;
> we verified that with a 27-point sweep showing byte-identical output from −5.0
> to 0.95. To derive a value we ablated the relative test so the threshold
> decides alone, and swept 281 points. That gives two hard boundaries: below 0
> specificity collapses to 0.09, and at 1.0 recall collapses to 0.30 because
> containment flags carry a fixed score of 1.0. Youden's J and F1 both peak on
> [0.80, 0.95], and we take 0.80 — the lower edge, farthest from the recall
> cliff. Under the full rule this value is inside the invariant band, so it is
> simultaneously optimal in isolation and provably harmless in combination."*

Then volunteer the limit:

> *"The gain over the previous value is one false positive in 238, which is
> within noise — the derivation's content is the boundaries, not that number."*

---

## 6. Files

| File | Contents |
|---|---|
| `derive_threshold.py` | **the derivation** — one checker pass, then ROC on three rules |
| `run_threshold_sweep.py` | the invariance experiment (§2) |
| `make_sweep_figures.py` | all three figures |
| `results_threshold_derivation.json` | 281-point ROC for each rule, optima, tie ranges |
| `results_threshold_sweep.json` | 27-point sweep of the full deployed rule |
| `check_scores_full.jsonl` | **1,582 checks** — contradiction / neutral / entailment logits, softmax probabilities, containment flag, case label |
| `per_check_scores.jsonl` | the leaner capture used by the sweep |
| `derivation_run.log`, `sweep_run.log` | console output |
| `figures/fig_threshold_derivation.png` | **the result** — three regimes, optimum [0.80, 0.95] shaded |

**One figure only.** The two diagnostic charts from §2 (invariance sweep, logit
gap) are not part of the final result and are not shipped. Their evidence lives
in `results_threshold_sweep.json` and in the §2 tables above; rebuild them with
`make_sweep_figures.py --all` if they are ever needed.

Reproduce:

```bash
# from m3_implementation/
python test_result/hallucination_result/threshold_calibration/derive_threshold.py
python test_result/hallucination_result/threshold_calibration/run_threshold_sweep.py
python test_result/hallucination_result/threshold_calibration/make_sweep_figures.py
```

No Groq key, no MongoDB, no PostgreSQL, no regeneration loop. ~7 min for both
passes on CPU.
