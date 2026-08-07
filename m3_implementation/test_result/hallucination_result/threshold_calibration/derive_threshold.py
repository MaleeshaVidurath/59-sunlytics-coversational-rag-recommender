# m3_implementation/test_result/hallucination_result/threshold_calibration/derive_threshold.py
#
# DERIVING the contradiction threshold from the score distribution, rather than
# inheriting it.
#
# WHY THE SWEEP WAS NOT ENOUGH
#   run_threshold_sweep.py showed the deployed rule is INVARIANT to the
#   threshold across [-5, 1.0): no check that beats entailment scores below
#   1.0, so the threshold slides through empty space. That proves the value is
#   harmless, but it cannot say how the value was chosen — there is no optimum
#   inside a flat region.
#
# THE FIX — score the threshold as if it were the ONLY rule
#   The deployed decision is:
#       flag  IF  contradiction > t  AND  contradiction > entailment
#   The second clause is what separates the classes, which is exactly why the
#   first is inert. So we also evaluate the ablated rule:
#       flag  IF  contradiction > t                      (threshold alone)
#   That rule is NOT invariant — it has a real ROC and a real optimum. Choosing
#   t there is a genuine derivation, and comparing the two curves measures what
#   the relative test contributes.
#
# WHAT IS STORED
#   Every DeBERTa comparison the checker makes: contradiction, neutral and
#   entailment logits, plus the softmax probabilities, tagged with the case
#   label and which field was corrupted. One pass, no regeneration loop, no
#   LLM calls, no network.
#
# ISOLATION
#   Writes only inside this folder.
#
# Run:  python test_result/hallucination_result/threshold_calibration/derive_threshold.py
#       [--test-set PATH] [--limit N]

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
_PAR = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(_DIR, '..', '..', '..'))
sys.path.insert(0, _PAR)

from dotenv import load_dotenv
load_dotenv()

from run_detector_eval import compute_metrics

TEST_SET  = os.path.join(_PAR, "original_eval_238", "labeled_test_set.jsonl")
OUT_JSON  = os.path.join(_DIR, "results_threshold_derivation.json")
OUT_CHECKS = os.path.join(_DIR, "check_scores_full.jsonl")

CONTAINMENT_SCORE = 1.0   # synthetic score the checker assigns to containment flags


def softmax3(c: float, n: float, e: float) -> tuple:
    m = max(c, n, e)
    ec, en, ee = math.exp(c - m), math.exp(n - m), math.exp(e - m)
    s = ec + en + ee
    return ec / s, en / s, ee / s


# ── Stage 1: one pass, capture everything ────────────────────────────────────

def capture(cases: list[dict]) -> list[dict]:
    from text_rag.core.hallucination_checker import HallucinationChecker
    checker = HallucinationChecker()

    out, t0 = [], time.time()
    for i, case in enumerate(cases):
        with contextlib.redirect_stdout(io.StringIO()):
            res = checker.check(case["response_text"], case["evidence"])

        checks = []
        for c in res["all_checks"]:
            s = c["nli_scores"]
            con, neu, ent = (float(s["contradiction"]), float(s.get("neutral", 0.0)),
                             float(s["entailment"]))
            pc, pn, pe = softmax3(con, neu, ent)
            checks.append({
                "field": c["fact_field"],
                "contradiction": con, "neutral": neu, "entailment": ent,
                "p_contradiction": round(pc, 6), "p_neutral": round(pn, 6),
                "p_entailment": round(pe, 6),
                "is_containment": abs(con - CONTAINMENT_SCORE) < 1e-9 and abs(ent) < 1e-9,
            })

        out.append({
            "case_id": case["case_id"],
            "label": case["label"],
            "corruption_type": (case.get("corruption") or {}).get("type"),
            "deployed_pred": res["has_hallucination"],
            "checks": checks,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    return out


# ── Stage 2: case-level scores ───────────────────────────────────────────────

def case_scores(rec: dict, exclude_containment: bool = False) -> dict:
    """A case is flagged if ANY check fires, so the case-level score is the max."""
    chs = [c for c in rec["checks"]
           if not (exclude_containment and c["is_containment"])]
    if not chs:
        return {"max_contra": None, "max_margin": None, "max_p_contra": None}
    return {
        "max_contra":   max(c["contradiction"] for c in chs),
        "max_margin":   max(c["contradiction"] - c["entailment"] for c in chs),
        "max_p_contra": max(c["p_contradiction"] for c in chs),
    }


def roc(y_true: list[bool], scores: list[float], grid: list[float]) -> list[dict]:
    rows = []
    for t in grid:
        y_pred = [s is not None and s > t for s in scores]
        m = compute_metrics(y_true, y_pred)
        rows.append({
            "threshold": round(t, 4),
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "specificity": m["specificity"],
            "balanced_accuracy": m["balanced_accuracy"],
            "youden_j": round(m["recall"] + m["specificity"] - 1, 4),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        })
    return rows


def best(rows: list[dict], key: str) -> dict:
    """Best row by `key`; ties broken toward the LOWER threshold (conservative:
    recall matters more than a marginal specificity gain here)."""
    top = max(r[key] for r in rows)
    tied = [r for r in rows if abs(r[key] - top) < 1e-9]
    return {"value": top, "at": tied[0]["threshold"],
            "tied_range": [tied[0]["threshold"], tied[-1]["threshold"]],
            "n_tied": len(tied), "row": tied[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=TEST_SET)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.test_set, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[:args.limit]

    y_true = [c["label"] == "hallucinated" for c in cases]
    print(f"{len(cases)} cases ({sum(y_true)} hallucinated / "
          f"{len(y_true)-sum(y_true)} clean)\n")
    print("Pass 1 — running the checker (DeBERTa + MiniLM, no LLM, no network)")
    recs = capture(cases)

    with open(OUT_CHECKS, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    n_checks = sum(len(r["checks"]) for r in recs)
    n_cont   = sum(1 for r in recs for c in r["checks"] if c["is_containment"])
    print(f"\nStored {n_checks} checks ({n_cont} containment-flag, "
          f"{n_checks-n_cont} genuine NLI) -> {os.path.basename(OUT_CHECKS)}")

    # ── the distribution, split by ground truth ──────────────────────────────
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION — genuine NLI checks only (containment excluded)")
    print("=" * 70)
    nli = [(r["label"], c) for r in recs for c in r["checks"]
           if not c["is_containment"]]
    for lab in ("hallucinated", "clean"):
        vals = [c["contradiction"] for l, c in nli if l == lab]
        marg = [c["contradiction"] - c["entailment"] for l, c in nli if l == lab]
        if not vals:
            continue
        vals_s = sorted(vals)
        q = lambda p: vals_s[min(len(vals_s) - 1, int(p * len(vals_s)))]
        print(f"\n  {lab}  (n={len(vals)})")
        print(f"    contradiction logit: min {min(vals):7.3f}  p25 {q(.25):7.3f}  "
              f"median {q(.50):7.3f}  p75 {q(.75):7.3f}  max {max(vals):7.3f}")
        print(f"    margin (con-ent)   : min {min(marg):7.3f}  "
              f"median {sorted(marg)[len(marg)//2]:7.3f}  max {max(marg):7.3f}")

    # ── ROC on the threshold ALONE ───────────────────────────────────────────
    grid = [round(-6 + 0.05 * i, 2) for i in range(int(14 / 0.05) + 1)]

    sc_all  = [case_scores(r)["max_contra"] for r in recs]
    sc_nli  = [case_scores(r, exclude_containment=True)["max_contra"] for r in recs]
    sc_marg = [case_scores(r)["max_margin"] for r in recs]

    roc_all  = roc(y_true, sc_all, grid)
    roc_nli  = roc(y_true, sc_nli, grid)
    roc_marg = roc(y_true, sc_marg, grid)

    print("\n" + "=" * 70)
    print("RULE A — threshold ALONE:  flag if max(contradiction) > t")
    print("         (the relative test removed, so t actually decides)")
    print("=" * 70)
    b_all = best(roc_all, "youden_j")
    b_f1  = best(roc_all, "f1")
    r = b_all["row"]
    print(f"  optimal by Youden's J : t = {b_all['at']}  (J={b_all['value']:.4f})")
    print(f"     ties across        : {b_all['tied_range']}  ({b_all['n_tied']} grid points)")
    print(f"     P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  "
          f"BalAcc={r['balanced_accuracy']:.4f}")
    print(f"  optimal by F1         : t = {b_f1['at']}  (F1={b_f1['value']:.4f})")

    print("\n" + "=" * 70)
    print("RULE A' — same, but genuine NLI checks only (containment flags removed)")
    print("=" * 70)
    b_nli = best(roc_nli, "youden_j")
    r = b_nli["row"]
    print(f"  optimal by Youden's J : t = {b_nli['at']}  (J={b_nli['value']:.4f})")
    print(f"     ties across        : {b_nli['tied_range']}")
    print(f"     P={r['precision']:.4f}  R={r['recall']:.4f}  "
          f"BalAcc={r['balanced_accuracy']:.4f}")

    print("\n" + "=" * 70)
    print("RULE B — margin rule:  flag if max(contradiction - entailment) > t")
    print("         (t=0 is exactly the deployed relative test)")
    print("=" * 70)
    b_marg = best(roc_marg, "youden_j")
    r = b_marg["row"]
    at0 = next(x for x in roc_marg if abs(x["threshold"]) < 1e-9)
    print(f"  optimal by Youden's J : t = {b_marg['at']}  (J={b_marg['value']:.4f})")
    print(f"     ties across        : {b_marg['tied_range']}")
    print(f"     P={r['precision']:.4f}  R={r['recall']:.4f}  "
          f"BalAcc={r['balanced_accuracy']:.4f}")
    print(f"  at t=0 (deployed)     : P={at0['precision']:.4f}  R={at0['recall']:.4f}  "
          f"BalAcc={at0['balanced_accuracy']:.4f}  J={at0['youden_j']:.4f}")

    payload = {
        "meta": {
            "purpose": "Derive the contradiction threshold from the score "
                       "distribution by scoring it as the sole decision rule.",
            "test_set": os.path.basename(args.test_set),
            "n_cases": len(cases),
            "n_hallucinated": sum(y_true),
            "n_clean": len(y_true) - sum(y_true),
            "n_checks": n_checks,
            "n_containment_flags": n_cont,
            "scale": "raw cross-encoder logits; softmax probabilities also stored",
            "grid": "-6.00 to 8.00 step 0.05",
        },
        "rule_a_threshold_alone": {"roc": roc_all, "best_youden": b_all, "best_f1": b_f1},
        "rule_a_nli_only":        {"roc": roc_nli, "best_youden": b_nli},
        "rule_b_margin":          {"roc": roc_marg, "best_youden": b_marg,
                                   "at_zero": at0},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten -> {OUT_JSON}")


if __name__ == "__main__":
    main()
