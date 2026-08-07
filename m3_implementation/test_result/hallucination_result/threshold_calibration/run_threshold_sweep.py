# m3_implementation/test_result/hallucination_result/threshold_calibration/run_threshold_sweep.py
#
# WHY THIS EXISTS
#   The reported evaluation swept NLI_CONTRADICTION_THRESHOLD over
#   [0.25, 0.5, 0.65, 1.0, 2.0, 3.0, 4.0, 5.0] and reported that results are
#   "flat across 0.25-0.65". Two problems with using that to justify the
#   deployed value:
#
#     1. The deployed value is 0.70 (.env), which is NOT in that grid. It sits
#        in an unmeasured gap between the last verified point (0.65) and the
#        collapse (1.0).
#     2. The grid never goes below 0.25, so the LEFT edge of the plateau was
#        never located either. "Precision 1.000 throughout" is only evidence of
#        headroom on one side.
#
#   This script locates BOTH edges of the plateau on a dense grid spanning
#   negative logits, so the operating point can be defended as sitting in the
#   middle of a measured flat region rather than at an arbitrary point.
#
# WHY NEGATIVE THRESHOLDS
#   CrossEncoder.predict() returns raw LOGITS (roughly -6 .. +7), not
#   probabilities. A threshold of 0 means "any positive lean toward
#   contradiction". Negative thresholds progressively admit weaker and weaker
#   evidence, so they are exactly where precision must eventually break. If it
#   does not break even at -5, that is itself a finding: the decision is being
#   carried entirely by the relative test (contradiction > entailment).
#
# WHAT IT DOES
#   1. Runs the real HallucinationChecker once over the labelled test set.
#      This is the only expensive step (DeBERTa + MiniLM, no network).
#   2. PERSISTS the per-check raw logits to per_check_scores.jsonl — the gap
#      that made this re-run necessary in the first place. Any future sweep is
#      then pure arithmetic.
#   3. Recomputes the decision at every threshold in the dense grid, using the
#      SAME scoring code as the reported evaluation (imported, not copied).
#   4. Reports the plateau edges and where the deployed value falls.
#
#   The gate pipeline (which checks run at all) does not depend on the
#   threshold, so recomputation is exact — identical to re-running the checker
#   at each value.
#
# ISOLATION
#   Writes only inside this folder. The reported artifacts in
#   ../original_eval_238/ are read-only here and are never touched.
#
# Run:  python test_result/hallucination_result/threshold_calibration/run_threshold_sweep.py
#       (add --limit N for a smoke test)

import argparse
import json
import os
import sys

_DIR   = os.path.dirname(os.path.abspath(__file__))
_PAR   = os.path.dirname(_DIR)                      # hallucination_result/
sys.path.insert(0, os.path.join(_DIR, '..', '..', '..'))   # m3_implementation/
sys.path.insert(0, _PAR)                                   # for run_detector_eval

from dotenv import load_dotenv
load_dotenv()

# Same scoring and same checker driver as the reported evaluation — imported so
# there is no possibility of a divergent reimplementation.
from run_detector_eval import compute_metrics, eval_ours, recall_by_corruption

TEST_SET = os.path.join(_PAR, "original_eval_238", "labeled_test_set.jsonl")
OUT_JSON = os.path.join(_DIR, "results_threshold_sweep.json")
OUT_SCORES = os.path.join(_DIR, "per_check_scores.jsonl")

# Dense grid spanning the full logit range the model actually produces.
# Fine-grained through the region where the deployed and default values sit.
GRID = [
    -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
    0.0, 0.25, 0.4, 0.5, 0.6, 0.65, 0.70, 0.75, 0.8, 0.9, 0.95,
    1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0,
]

DEPLOYED = 0.70    # .env
DEFAULT  = 0.65    # config.py fallback — the value that was evaluated/reported


def decide(details: list[dict], t: float) -> list[bool]:
    """The checker's own decision rule, recomputed at threshold t.

        is_hallucination = contradiction > t AND contradiction > entailment

    Exactly the expression in hallucination_checker.py, applied to the stored
    per-check logits.
    """
    return [
        any(ch["contradiction"] > t and ch["contradiction"] > ch["entailment"]
            for ch in d["checks"])
        for d in details
    ]


def plateau(rows: list[dict], key: str = "balanced_accuracy") -> dict:
    """The widest run of consecutive thresholds sharing the best value of `key`."""
    best = max(r[key] for r in rows)
    run, runs = [], []
    for r in rows:
        if abs(r[key] - best) < 1e-9:
            run.append(r["threshold"])
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)
    widest = max(runs, key=len) if runs else []
    return {"metric": key, "value": round(best, 4),
            "from": widest[0] if widest else None,
            "to": widest[-1] if widest else None,
            "n_thresholds": len(widest)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N cases")
    ap.add_argument("--test-set", default=TEST_SET)
    args = ap.parse_args()

    with open(args.test_set, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[:args.limit]

    y_true = [c["label"] == "hallucinated" for c in cases]
    print(f"Loaded {len(cases)} cases "
          f"({sum(y_true)} hallucinated / {len(y_true) - sum(y_true)} clean)\n")

    # ── the one expensive pass ────────────────────────────────────────────────
    print("Running the checker (DeBERTa + MiniLM, no network)...")
    _, details = eval_ours(cases)

    # Persist the raw logits so no future sweep needs the models again.
    with open(OUT_SCORES, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")
    n_checks = sum(len(d["checks"]) for d in details)
    print(f"\nStored {n_checks} per-check logits -> {os.path.basename(OUT_SCORES)}")

    # ── the sweep ─────────────────────────────────────────────────────────────
    rows = []
    for t in GRID:
        y_pred = decide(details, t)
        m = compute_metrics(y_true, y_pred)
        rows.append({
            "threshold": t,
            **{k: m[k] for k in ("precision", "recall", "f1",
                                 "specificity", "balanced_accuracy")},
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        })

    print(f"\n{'thresh':>8} {'P':>7} {'R':>7} {'F1':>7} {'Spec':>7} {'BalAcc':>7} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("-" * 74)
    for r in rows:
        mark = ""
        if abs(r["threshold"] - DEPLOYED) < 1e-9:
            mark = "  <- DEPLOYED (.env)"
        elif abs(r["threshold"] - DEFAULT) < 1e-9:
            mark = "  <- config default / reported"
        print(f"{r['threshold']:>8.2f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['specificity']:>7.3f} "
              f"{r['balanced_accuracy']:>7.3f} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['tn']:>4}{mark}")

    # ── where do the two candidate values sit? ────────────────────────────────
    by_t = {r["threshold"]: r for r in rows}
    pl_bal = plateau(rows, "balanced_accuracy")
    pl_f1  = plateau(rows, "f1")

    print(f"\nWidest plateau (balanced accuracy = {pl_bal['value']}): "
          f"{pl_bal['from']} .. {pl_bal['to']}  ({pl_bal['n_thresholds']} grid points)")
    print(f"Widest plateau (F1 = {pl_f1['value']}): "
          f"{pl_f1['from']} .. {pl_f1['to']}")

    same = (by_t[DEPLOYED]["balanced_accuracy"] == by_t[DEFAULT]["balanced_accuracy"]
            and by_t[DEPLOYED]["f1"] == by_t[DEFAULT]["f1"])
    print(f"\n0.70 (deployed) vs 0.65 (reported): "
          f"{'IDENTICAL on every metric' if same else 'THEY DIFFER'}")
    for t in (DEFAULT, DEPLOYED):
        r = by_t[t]
        print(f"   t={t:<5} P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} BalAcc={r['balanced_accuracy']:.4f}")

    # ── does precision ever break on the left? ───────────────────────────────
    broke = [r["threshold"] for r in rows if r["precision"] < 1.0]
    print(f"\nThresholds where precision < 1.000: "
          f"{broke if broke else 'NONE across the entire grid'}")

    payload = {
        "meta": {
            "purpose": "Locate both edges of the operating plateau for "
                       "NLI_CONTRADICTION_THRESHOLD, including negative logits.",
            "test_set": os.path.basename(args.test_set),
            "n_cases": len(cases),
            "n_hallucinated": sum(y_true),
            "n_clean": len(y_true) - sum(y_true),
            "scale": "raw cross-encoder logits (CrossEncoder.predict), NOT probabilities",
            "decision_rule": "contradiction > threshold AND contradiction > entailment",
            "deployed_value": DEPLOYED,
            "config_default_value": DEFAULT,
            "scoring_code": "imported from run_detector_eval.compute_metrics",
            "note": "Thresholds >= 1.0 remain partly an artifact: containment "
                    "flags carry a synthetic contradiction score of 1.0 and drop "
                    "out of the decision at t >= 1.0.",
        },
        "grid": GRID,
        "sweep": rows,
        "plateau_balanced_accuracy": pl_bal,
        "plateau_f1": pl_f1,
        "deployed_equals_reported": bool(same),
        "precision_breaks_at": broke,
        "recall_by_corruption_at_deployed": recall_by_corruption(
            cases, decide(details, DEPLOYED)),
        "recall_by_corruption_at_default": recall_by_corruption(
            cases, decide(details, DEFAULT)),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten -> {OUT_JSON}")


if __name__ == "__main__":
    main()
