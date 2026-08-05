# Cross-Turn Consistency (v2) — Evaluation

> **Full explanation of the method** — how the test data is created, what the
> baselines are and how they work, how scoring is done — is in
> [`memory/core/CONTRADICTION_DETECTOR_PROCESS.md` §14](../../memory/core/CONTRADICTION_DETECTOR_PROCESS.md).
> This file is the results summary.

> **v1's results are not touched by anything in this folder.**
> `../contradiction_result/` holds the numbers reported in the final report
> (Tables 15/16, Figures 43/44). This folder only *reads* its labelled test set.

---

## 1. Why a second evaluation exists

The detector was rebuilt after the reported evaluation. The rebuild was not a
tuning pass — it changed what the component is measuring:

| | v1 (reported) | v2 (current code) |
|---|---|---|
| Compared against | the **current turn's evidence** | the **prior assertion / live catalogue** |
| Same-turn mismatch | counted as a contradiction | **deferred** to the hallucination guard |
| Extraction | one Groq LLM call per turn | deterministic, **no LLM** |
| Correction | whole-response string replace | sentence-scoped |
| Catalogue revisions | not detected | detected, superseded, annotated |

The old comparison reduced to *response vs current evidence* — which is the
hallucination guard's job. v2 stops reporting those and hands them back, so on
this test set it necessarily **reports** far fewer contradictions. That is the
designed behaviour, not a regression, and the two quantities are measured
separately so the difference is visible rather than hidden:

- **`v2 detected`** — every mismatch identified, including deferred ones.
  Answers *"did the rebuild cost us detection power?"*
- **`v2 reported`** — only verdicts that change what the user sees.
  Answers *"how much of v1's recall was double-counting?"*

---

## 2. Headline result — like-for-like sample

Same labelled test set (1,346 cases), same stratified sampler and seed (v1's
own `_stratified_sample`, seed 123), same metric code.

> **Exact counts:** v1 scored **599** cases (450 contradictions); this v2 run
> scored **598** (449). The one-case difference comes from rounding in the
> per-stratum allocation, not from a different sampling procedure. Both are
> stratified samples of the same population drawn the same way; the gap is
> immaterial to the comparison, but it is stated rather than glossed over.

| System | Precision | Recall | F1 | Bal. acc. |
|---|---|---|---|---|
| **v2 detected** | 0.976 | **0.915** | **0.945** | **0.924** |
| v1 ours (graph+NLI) | 0.983 | 0.773 | 0.866 | 0.867 |
| v1 llm_judge | 0.957 | 0.850 | 0.900 | 0.867 |
| v1 string_only | 0.844 | 0.856 | 0.850 | 0.690 |
| v1 history_nli | 0.764 | 0.958 | 0.850 | 0.533 |
| v1 uttr_pair_nli | 0.772 | 0.909 | 0.835 | 0.548 |

> **`v2 reported` is not in this table on purpose.** Every row here answers
> *"did the system find the mismatch?"*. Reporting answers a different question —
> *"did it rewrite the text?"* — which on this test set is 0.9% by design. Placed
> beside detection bars it reads as failure rather than as the deliberate
> hand-off it is. It is shown on its own terms in §2.1 and figV2, and both
> metrics remain in full in `results_v2_eval.json`.

### 2.1 Ownership of what v2 detected

| | Count |
|---|---|
| detected | **411** / 449 contradictions |
| deferred to the hallucination guard | **407** (99.0% of detections) |
| reported by v2 itself | 4 |

**Two findings.**

1. **Detection improved.** +14.2 pts recall and +7.9 pts F1 over v1, for −0.7 pts
   precision — and with **no LLM call at all**. v1 needed one paced Groq request
   per case and a non-production model id to survive the free tier.

2. **99.0% of what v2 detects, it deliberately does not report** (407 of 411).
   Those are same-turn mismatches the hallucination guard already owns. That
   figure is the quantitative form of the criticism that prompted the rebuild:
   v1's recall was counting a single failure twice, once in each component.

Recall by turn distance (v2 detected): `d=0: 0.92 · d=1: 0.83 · d=2: 0.94 · d≥3: 0.96`
— detection does not decay with distance.

---

## 3. Files

| File | Contents |
|---|---|
| `run_v2_eval.py` | the runner (detection + correction in one pass) |
| `make_v2_figures.py` | the figure script |
| `results_v2_eval.json` | full test set, 1,346 cases |
| `v2_case_detail.jsonl` | per-case verdicts, kinds and corrected text |
| `sample599/` | the like-for-like run against v1's evaluated slice |
| `figures/` | five figures, built from `sample599/` |

### Figures

| File | Shows |
|---|---|
| `figV1_detection_v2_vs_v1.png` | P/R/F1/BalAcc — v2 beside v1 and all four baselines |
| `figV2_ownership_split.png` | the 99% deferred share — the duplication, in one bar |
| `figV3_recall_by_distance.png` | recall vs how many turns back the value was set |
| `figV4_recall_by_corruption.png` | recall per corruption type |
| `figV5_false_alarms.png` | false alarms on clean and hard-negative cases |

Every bar on a shared axis comes from the same sample. Palette and chrome match
the v1 and hallucination chapters exactly, so the figures sit together without a
visual seam.

Reproduce:

```bash
# like-for-like with v1 (the table above)
python test_result/contradiction_result_v2/run_v2_eval.py \
       --sample 599 --out-dir test_result/contradiction_result_v2/sample599

# full test set
python test_result/contradiction_result_v2/run_v2_eval.py

# figures (reads sample599/)
python test_result/contradiction_result_v2/make_v2_figures.py

# diagnose a single case
python test_result/contradiction_result_v2/run_v2_eval.py --debug-case ccase_0008
```

No Groq key, no MongoDB, no PostgreSQL. Runs in a couple of minutes.

---

## 4. Honest caveats

**The baselines were not re-run.** They are independent of our detector and
unchanged, so their rows are carried forward verbatim from
`../contradiction_result/results_contra_eval.json`. Re-running them could only
perturb numbers already printed in the report.

**The correction experiment is not comparable to v1's Table 16.** v2 rewrites
only what it reports (0.9% of cases here), because same-turn errors are fixed by
the guard's own regenerate loop instead. `user_facing_contradiction_rate_on`
therefore stays near 1.0 on *this* test set — it measures a path v2 delegates.
Measuring v2's correction properly needs cases where evidence is silent or the
catalogue has moved, which this test set does not contain (see §5).

**This test set cannot exercise the new capability.** Every case corrupts the
response while leaving evidence and database correct. Catalogue-revision
detection — the capability with no baseline in the literature review — scores
nothing here because nothing ever changes in the catalogue. It has been
demonstrated live but is not yet quantified.

**No live database read.** Offline, `truth_is_live` is False throughout, so the
anchor falls to the prior assertion or session context, never to `live_evidence`.
That matches behaviour when re-verification is unavailable and keeps scoring
deterministic.

---

## 5. What would complete the picture

Two corruption types added to `../contradiction_result/corrupt_sessions.py`:

- **`stale_carry`** — evidence silent on the attribute this turn; the response
  repeats a value that has since changed. Exercises the prior-assertion anchor.
- **`db_revision`** — the catalogue value changes between turns. Exercises
  revision detection, superseding, and the retroactive notice.

Both are constructible from the existing capture records, which already carry
`product_refs` and `graph_before`.

---

## 6. A production bug this evaluation found

The first full run scored badly for reasons that had nothing to do with the
detector. `_load_vocabularies()` in `assertion_extractor.py` was resolving the
articles CSV from a path assembled out of module `__file__`s; under this import
chain it accumulated enough `..` segments to fail to open. The loader caught the
error, printed a quiet note, and returned **empty** vocabularies — silently
disabling colour and product-type extraction while everything appeared to run
normally.

Fixed by normalising the path, falling back to a search rooted at the module,
and making the failure a loud warning instead of a footnote. The same loader
backs the hallucination guard's name gate, so the fix protects both.

Worth stating plainly: the evaluation earned its keep by finding a silent
degradation that no test had caught.
