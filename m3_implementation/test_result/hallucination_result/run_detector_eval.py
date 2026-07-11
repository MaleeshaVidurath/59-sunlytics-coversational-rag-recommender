# m3_implementation/test_result/hallucination_result/run_detector_eval.py
#
# Step 3 of the hallucination checker evaluation — detection accuracy vs baselines.
#
# Runs three detectors over labeled_test_set.jsonl (label = clean|hallucinated)
# and reports Precision / Recall / F1 / Balanced Accuracy for each:
#
#   1. ours        — the full HallucinationChecker (lock map + 9 gates + NLI)
#   2. naive_nli   — SummaC-style baseline: DeBERTa on EVERY (fact, sentence)
#                    pair, no lock map, no gates, no containment checks.
#                    Flag if any pair has softmax P(contradiction) > 0.5 and
#                    contradiction > entailment.
#   3. llm_judge   — RAGAS-style baseline: Groq LLM judges whether the response
#                    contradicts the evidence facts. (--skip-llm to disable)
#
# Extra outputs:
#   - Recall broken down by corruption type (colour/price/name/cross-item)
#   - False positives on the clean cases
#   - Threshold sensitivity sweep for the checker (recomputed offline from the
#     stored NLI scores — no model re-runs needed). NOTE: CrossEncoder.predict
#     returns raw LOGITS, so the sweep covers logit-scale thresholds.
#
# No databases or LLM generation needed — only the local NLI/embedding models
# (plus Groq API for the llm_judge baseline).
#
# Run:  python test_result/hallucination_result/run_detector_eval.py [--skip-llm] [--limit N]

import argparse
import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

_DIR      = os.path.dirname(os.path.abspath(__file__))
TEST_SET  = os.path.join(_DIR, "original_eval_238", "labeled_test_set.jsonl")
RESULTS   = os.path.join(_DIR, "original_eval_238", "results_detector_eval.json")

NAIVE_CONTRA_PROB = 0.5   # softmax threshold for the naive baseline
SWEEP_THRESHOLDS  = [0.25, 0.5, 0.65, 1.0, 2.0, 3.0, 4.0, 5.0]  # raw logits


# ── Metrics ──────────────────────────────────────────────────────────────────

def _wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a proportion k/n. [0,0] when n == 0."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom  = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _bootstrap_cis(y_true, y_pred, n_boot: int = 2000, seed: int = 0):
    """Bootstrap 95% CIs for F1 and balanced accuracy (case resampling)."""
    import numpy as np
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    n = len(yt)
    if n == 0:
        return [0.0, 0.0], [0.0, 0.0]
    rng = np.random.default_rng(seed)
    f1s, bals = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t, p = yt[idx], yp[idx]
        tp = int(np.sum(t & p));  fp = int(np.sum(~t & p))
        fn = int(np.sum(t & ~p)); tn = int(np.sum(~t & ~p))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        bals.append((rec + spec) / 2)
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    blo, bhi = np.percentile(bals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)], \
           [round(float(blo), 4), round(float(bhi), 4)]


def compute_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Positive class = hallucinated. Includes 95% CIs: Wilson intervals for
    proportion metrics, bootstrap (2000 seeded resamples) for F1 and balanced
    accuracy — so extreme values on small samples are always presented with
    their statistical uncertainty."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1_ci, bal_ci = _bootstrap_cis(y_true, y_pred)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1":                round(f1, 4),
        "specificity":       round(specificity, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "accuracy":          round((tp + tn) / len(y_true), 4) if y_true else 0.0,
        "precision_ci95":    _wilson_ci(tp, tp + fp),
        "recall_ci95":       _wilson_ci(tp, tp + fn),
        "specificity_ci95":  _wilson_ci(tn, tn + fp),
        "f1_ci95":           f1_ci,
        "balanced_accuracy_ci95": bal_ci,
    }


def recall_by_corruption(cases: list[dict], preds: list[bool]) -> dict:
    """Detection rate per corruption type (hallucinated cases only)."""
    buckets: dict[str, list[bool]] = {}
    for case, pred in zip(cases, preds):
        if case["label"] != "hallucinated":
            continue
        ctype = case["corruption"]["type"]
        buckets.setdefault(ctype, []).append(pred)
    return {
        ctype: {"detected": sum(v), "total": len(v),
                "recall": round(sum(v) / len(v), 4)}
        for ctype, v in sorted(buckets.items())
    }


# ── Detector 1: our checker ──────────────────────────────────────────────────

def eval_ours(cases: list[dict]) -> tuple[list[bool], list[dict]]:
    from text_rag.core.hallucination_checker import HallucinationChecker
    checker = HallucinationChecker()

    preds, details = [], []
    t0 = time.time()
    for i, case in enumerate(cases):
        # The checker prints verbose debug matrices — silence them for the eval
        with contextlib.redirect_stdout(io.StringIO()):
            result = checker.check(case["response_text"], case["evidence"])
        preds.append(result["has_hallucination"])
        details.append({
            "case_id": case["case_id"],
            "pred":    result["has_hallucination"],
            "contradicted_fields": result["contradicted_fields"],
            # keep raw NLI scores per check for the offline threshold sweep
            "checks": [
                {"field": c["fact_field"],
                 "contradiction": c["nli_scores"]["contradiction"],
                 "entailment":    c["nli_scores"]["entailment"]}
                for c in result["all_checks"]
            ],
        })
        if (i + 1) % 25 == 0:
            print(f"  [ours] {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    return preds, details


# ── Detector 2: naive NLI baseline (no gates, no lock map) ───────────────────

def eval_naive_nli(cases: list[dict]) -> list[bool]:
    import numpy as np
    from text_rag.core.hallucination_checker import (
        _split_sentences, _flatten_evidence, _get_nli_model
    )
    model = _get_nli_model()

    def softmax(row):
        e = np.exp(np.array(row) - np.max(row))
        return e / e.sum()

    preds = []
    t0 = time.time()
    for i, case in enumerate(cases):
        sentences = _split_sentences(case["response_text"])
        facts     = [f["text"] for f in _flatten_evidence(case["evidence"])]
        if not sentences or not facts:
            preds.append(False)
            continue
        pairs  = [(fact, sent) for fact in facts for sent in sentences]
        with contextlib.redirect_stdout(io.StringIO()):
            scores = model.predict(pairs)
        flagged = False
        for row in scores:
            p = softmax(row)           # [contradiction, neutral, entailment]
            if p[0] > NAIVE_CONTRA_PROB and p[0] > p[2]:
                flagged = True
                break
        preds.append(flagged)
        if (i + 1) % 25 == 0:
            print(f"  [naive_nli] {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    return preds


# ── Detector 3: LLM-judge baseline (RAGAS-style, via Groq) ───────────────────

_JUDGE_PROMPT = """You are a strict factual-consistency judge for a fashion \
recommendation assistant.

EVIDENCE FACTS (ground truth from the product database):
{facts}

ASSISTANT RESPONSE to check:
{response}

Does the response contradict any evidence fact (wrong product name, wrong \
colour, wrong price, wrong attribute, or attributes attached to the wrong \
product)? Extra detail that is merely absent from the evidence is NOT a \
contradiction.

Answer with ONLY this JSON, nothing else:
{{"hallucinated": true or false}}"""


def eval_llm_judge(cases: list[dict]) -> list[bool | None]:
    import httpx
    from text_rag.core.hallucination_checker import _flatten_evidence
    from text_rag.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

    if not GROQ_API_KEY:
        print("  [llm_judge] GROQ_API_KEY not set — skipping baseline")
        return [None] * len(cases)

    preds: list[bool | None] = []
    t0 = time.time()
    with httpx.Client(timeout=30) as client:
        for i, case in enumerate(cases):
            facts = "\n".join(
                f"- {f['text']}" for f in _flatten_evidence(case["evidence"])
            )
            prompt = _JUDGE_PROMPT.format(
                facts=facts, response=case["response_text"]
            )
            verdict = None
            for attempt in range(4):
                try:
                    r = client.post(
                        f"{GROQ_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json={
                            "model": GROQ_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                            "max_tokens": 30,
                        },
                    )
                    if r.status_code == 429:
                        wait = min(2 ** attempt * 2, 20)
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    text = r.json()["choices"][0]["message"]["content"]
                    m = text.lower()
                    if '"hallucinated": true' in m or "'hallucinated': true" in m:
                        verdict = True
                    elif '"hallucinated": false' in m or "'hallucinated': false" in m:
                        verdict = False
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"  [llm_judge] case {case['case_id']} failed: {e}")
                    time.sleep(2)
            preds.append(verdict)
            time.sleep(0.4)  # stay under Groq free-tier rate limits
            if (i + 1) % 25 == 0:
                print(f"  [llm_judge] {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    return preds


# ── Threshold sweep (offline, from stored NLI scores) ────────────────────────

def threshold_sweep(cases: list[dict], details: list[dict]) -> list[dict]:
    """Recomputes the checker's decision at different contradiction thresholds
    using the per-check raw logits saved during eval_ours(). The gate pipeline
    (which checks get run at all) does not depend on the threshold, so this is
    exact — no model re-runs needed."""
    y_true = [c["label"] == "hallucinated" for c in cases]
    rows = []
    for t in SWEEP_THRESHOLDS:
        y_pred = [
            any(ch["contradiction"] > t and ch["contradiction"] > ch["entailment"]
                for ch in d["checks"])
            for d in details
        ]
        m = compute_metrics(y_true, y_pred)
        rows.append({"threshold": t, **{k: m[k] for k in
                     ("precision", "recall", "f1", "balanced_accuracy")}})
    return rows


# ── Report helpers ───────────────────────────────────────────────────────────

def print_metrics_table(named_metrics: dict[str, dict]):
    header = f"{'system':<12} {'P':>7} {'R':>7} {'F1':>7} {'BalAcc':>7} " \
             f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    print(header)
    print("-" * len(header))
    for name, m in named_metrics.items():
        print(f"{name:<12} {m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} {m['balanced_accuracy']:>7.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip the Groq LLM-judge baseline")
    ap.add_argument("--skip-naive", action="store_true",
                    help="skip the naive NLI baseline (unchanged between checker versions)")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N cases (smoke test)")
    ap.add_argument("--test-set", default=TEST_SET,
                    help="path to a labeled test set jsonl (default: standard set)")
    ap.add_argument("--out", default=RESULTS,
                    help="path for the results json (default: standard results file)")
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified random sample of N cases (seed 123) — for "
                         "slow/API-limited detectors on large sets")
    args = ap.parse_args()

    with open(args.test_set, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if args.sample and args.sample < len(cases):
        import random as _random
        rng = _random.Random(123)
        clean = [c for c in cases if c["label"] == "clean"]
        hall  = [c for c in cases if c["label"] == "hallucinated"]
        k_clean = max(1, round(args.sample * len(clean) / len(cases))) if clean else 0
        k_hall  = args.sample - k_clean
        cases = (rng.sample(clean, min(k_clean, len(clean)))
                 + rng.sample(hall, min(k_hall, len(hall))))
        print(f"Stratified sample: {len(cases)} cases (seed 123)")

    y_true = [c["label"] == "hallucinated" for c in cases]
    print(f"Loaded {len(cases)} cases "
          f"({sum(y_true)} hallucinated / {len(y_true)-sum(y_true)} clean)\n")

    results: dict = {"n_cases": len(cases)}
    named: dict[str, dict] = {}

    print("[1/3] Our checker (lock map + gates + NLI)...")
    ours_preds, ours_details = eval_ours(cases)
    named["ours"] = compute_metrics(y_true, ours_preds)
    results["ours"] = {
        "metrics": named["ours"],
        "recall_by_corruption": recall_by_corruption(cases, ours_preds),
        "false_positive_case_ids": [
            c["case_id"] for c, t, p in zip(cases, y_true, ours_preds)
            if not t and p
        ],
        "missed_case_ids": [
            c["case_id"] for c, t, p in zip(cases, y_true, ours_preds)
            if t and not p
        ],
    }
    results["threshold_sweep"] = threshold_sweep(cases, ours_details)

    if not args.skip_naive:
        print("\n[2/3] Naive NLI baseline (all pairs, no gates)...")
        naive_preds = eval_naive_nli(cases)
        named["naive_nli"] = compute_metrics(y_true, naive_preds)
        results["naive_nli"] = {
            "metrics": named["naive_nli"],
            "recall_by_corruption": recall_by_corruption(cases, naive_preds),
        }
    else:
        print("\n[2/3] Naive NLI baseline skipped (--skip-naive)")

    if not args.skip_llm:
        print("\n[3/3] LLM-judge baseline (Groq)...")
        judge_preds = eval_llm_judge(cases)
        answered = [(t, p) for t, p in zip(y_true, judge_preds) if p is not None]
        if answered:
            jt, jp = zip(*answered)
            named["llm_judge"] = compute_metrics(list(jt), list(jp))
            results["llm_judge"] = {
                "metrics": named["llm_judge"],
                "n_answered": len(answered),
                "n_unanswered": len(cases) - len(answered),
                "recall_by_corruption": recall_by_corruption(
                    [c for c, p in zip(cases, judge_preds) if p is not None],
                    [p for p in judge_preds if p is not None],
                ),
            }
    else:
        print("\n[3/3] LLM-judge baseline skipped (--skip-llm)")

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DETECTION ACCURACY (positive class = hallucinated)")
    print("=" * 60)
    print_metrics_table(named)

    print("\nRECALL BY CORRUPTION TYPE")
    for system in ("ours", "naive_nli", "llm_judge"):
        if system not in results:
            continue
        print(f"  {system}:")
        for ctype, r in results[system]["recall_by_corruption"].items():
            print(f"    {ctype:<18} {r['detected']:>3}/{r['total']:<3}"
                  f"  recall={r['recall']:.3f}")

    print("\nTHRESHOLD SWEEP (ours, raw NLI logits)")
    print(f"  {'thresh':>7} {'P':>7} {'R':>7} {'F1':>7} {'BalAcc':>7}")
    for row in results["threshold_sweep"]:
        print(f"  {row['threshold']:>7.2f} {row['precision']:>7.3f} "
              f"{row['recall']:>7.3f} {row['f1']:>7.3f} "
              f"{row['balanced_accuracy']:>7.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
