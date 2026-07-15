# m3_implementation/test_result/contradiction_result/run_correction_eval.py
#
# Experiment B of the contradiction detector evaluation — the ON/OFF correction
# (mitigation) experiment (see EVALUATION_PLAN.md, section 2, Experiment B).
#
# Detection alone is not the whole novelty: the detector also REWRITES the
# response before the user sees it. This experiment measures that benefit, in
# the two-part style of Mündler et al. (ICLR 2024) and our own loop-mitigation
# experiment (hallucination chapter).
#
# DESIGN (per contradiction case, ground truth known by construction):
#   OFF : the corrupted response ships unchanged -> user sees the contradiction
#         (100% user-facing contradiction rate by construction).
#   ON  : run the detector's full detect->fix path (reconstructed graph + Groq
#         extraction + values_contradict + NLI confirm + _fix_response_text),
#         then an INDEPENDENT referee grades the shipped (corrected) response.
#
# REFEREE (model-free, shares no logic with the detector — it grades using the
# corruption ground truth, so the detector cannot mark its own homework):
#   single-field drift (colour/price/name/type):
#       fixed  <=> correct value present AND wrong value absent in final text
#   cross_item_swap:
#       fixed  <=> final text == the original clean response (swap undone)
#
# COLLATERAL DAMAGE is measured on the clean + hard_negative cases: run the same
# ON path and check the detector did NOT alter a response it should have left
# alone (a false correction is as harmful as a missed one).
#
# METRICS:
#   user_facing_contradiction_rate : OFF (=1.0) vs ON (residual)
#   detection_rate                 : fraction of contradictions detected
#   p_fix_given_detect             : of detected, fraction correctly rewritten
#   collateral_damage_rate         : fraction of negatives wrongly altered
#
# Needs Groq (claim extraction) + local DeBERTa NLI. No MongoDB.
#
# Run:  python test_result/contradiction_result/run_correction_eval.py
#         [--limit N] [--test-set PATH] [--out PATH]

import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import networkx as nx

_DIR      = os.path.dirname(os.path.abspath(__file__))
TEST_SET  = os.path.join(_DIR, "labeled_test_set.jsonl")
RESULTS   = os.path.join(_DIR, "results_correction_eval.json")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip().lower()


# ── Independent referee (grades using corruption ground truth) ────────────────

def referee_fixed(case: dict, final_text: str) -> bool:
    """True if the shipped (corrected) response is now consistent with truth."""
    corr = case["corruption"]
    ctype = corr["type"]
    if ctype == "cross_item_swap":
        # The only correct state is the original clean response (swap undone).
        return _norm(final_text) == _norm(case.get("source_response", ""))
    correct_val = str(corr["original"])
    wrong_val   = str(corr["corrupted"])
    ft = _norm(final_text)
    correct_present = _norm(correct_val) in ft
    wrong_absent    = _norm(wrong_val) not in ft
    return correct_present and wrong_absent


# ── Detector correction path (offline replication) ───────────────────────────

def _build_graph(graph_before: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for aid, attrs in (graph_before or {}).items():
        g.add_node(str(aid), **{k: v for k, v in attrs.items() if v is not None})
    return g


async def _extract_with_retry(response_text: str, refs: list) -> dict:
    """Groq claim extraction with SINGLE-call pacing (see the twin helper in
    run_contra_eval.py). One call + at most one gentle retry after a full
    6000-TPM-bucket refill wait — no retry storm that would cascade into 429s."""
    from memory.core.contradiction_detector import _extract_claims_groq
    base_delay = float(os.getenv("OURS_EVAL_DELAY", "8.0"))

    with contextlib.redirect_stdout(io.StringIO()):
        ext = await _extract_claims_groq(response_text, refs)
    if not ext:
        await asyncio.sleep(base_delay)
        with contextlib.redirect_stdout(io.StringIO()):
            ext = await _extract_claims_groq(response_text, refs)
    await asyncio.sleep(base_delay)
    return ext


async def detect_and_fix(case: dict) -> tuple[bool, str]:
    """Runs the detector's detect->fix path on one case.
    Returns (detected, final_response_text)."""
    from memory.core.contradiction_detector import (
        _update_graph_nodes, values_contradict,
        _confirm_with_nli, _fix_response_text, _CHECKABLE_FIELDS,
    )
    refs = case.get("product_refs", []) or []
    graph = _build_graph(case.get("graph_before", {}))
    _update_graph_nodes(graph, refs, case.get("turn_id", "t"),
                        case.get("session_id", "s"))

    response = case["response_text"]
    extracted = await _extract_with_retry(response, refs)

    corrected = response
    detected = False
    for aid, fields in (extracted or {}).items():
        if not graph.has_node(aid):
            continue
        node = graph.nodes[aid]
        for attr in _CHECKABLE_FIELDS:
            ex = fields.get(attr, "")
            if not ex or not values_contradict(node.get(attr, ""), ex):
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                is_c, _ = _confirm_with_nli(node, ex, attr)
            if not is_c:
                continue
            detected = True
            corrected = _fix_response_text(corrected, ex, node.get(attr, ""))
    return detected, corrected


# ── Main ─────────────────────────────────────────────────────────────────────

async def run(cases: list[dict]) -> dict:
    contra = [c for c in cases if c["label"] == "contradiction"]
    negs   = [c for c in cases if c["label"] in ("clean", "hard_negative")]
    print(f"Contradiction cases: {len(contra)}  Negative cases: {len(negs)}")

    # ── ON path over contradiction cases ─────────────────────────────────────
    n_detected = 0
    n_fixed    = 0
    off_wrong  = 0     # referee-verified user-facing contradictions, OFF
    on_wrong   = 0     # residual user-facing contradictions, ON
    residual_by_type: dict[str, list[int]] = {}
    t0 = time.time()

    for i, case in enumerate(contra):
        # OFF: corrupted response ships unchanged
        off_ok = referee_fixed(case, case["response_text"])
        if not off_ok:
            off_wrong += 1  # expected for ~all (contradiction present by construction)

        # ON: detect + fix, then referee grades the shipped text
        detected, final_text = await detect_and_fix(case)
        on_ok = referee_fixed(case, final_text)
        if detected:
            n_detected += 1
            if on_ok:
                n_fixed += 1
        if not on_ok:
            on_wrong += 1
        residual_by_type.setdefault(case["corruption"]["type"], []).append(0 if on_ok else 1)

        if (i + 1) % 25 == 0:
            print(f"  [ON] {i+1}/{len(contra)}  detected={n_detected} "
                  f"fixed={n_fixed}  ({time.time()-t0:.0f}s)")

    # ── Collateral damage over negative cases ────────────────────────────────
    collateral = 0
    for i, case in enumerate(negs):
        _, final_text = await detect_and_fix(case)
        if _norm(final_text) != _norm(case["response_text"]):
            collateral += 1
        if (i + 1) % 25 == 0:
            print(f"  [collateral] {i+1}/{len(negs)}  altered={collateral}")

    return _build_results(contra, negs, n_detected, n_fixed, off_wrong,
                          on_wrong, collateral, residual_by_type)


def _build_results(contra, negs, n_detected, n_fixed, off_wrong, on_wrong,
                   collateral, residual_by_type) -> dict:
    n_contra = len(contra)
    return {
        "n_contradiction": n_contra,
        "n_negative": len(negs),
        "off": {
            "user_facing_contradiction_rate": round(off_wrong / n_contra, 4) if n_contra else 0.0,
            "user_facing_contradictions": off_wrong,
        },
        "on": {
            "user_facing_contradiction_rate": round(on_wrong / n_contra, 4) if n_contra else 0.0,
            "user_facing_contradictions": on_wrong,
            "detection_rate": round(n_detected / n_contra, 4) if n_contra else 0.0,
            "n_detected": n_detected,
            "p_fix_given_detect": round(n_fixed / n_detected, 4) if n_detected else 0.0,
            "n_fixed": n_fixed,
        },
        "collateral_damage_rate": round(collateral / len(negs), 4) if negs else 0.0,
        "n_collateral": collateral,
        "residual_by_corruption": {
            t: {"residual": sum(v), "total": len(v),
                "residual_rate": round(sum(v) / len(v), 4)}
            for t, v in sorted(residual_by_type.items())
        },
    }


def run_offline(cases: list[dict], detail_by_id: dict) -> dict:
    """Scores the correction experiment OFFLINE from the detection pass's
    per-case detail sidecar (ours_case_detail.jsonl) — the corrected text and
    detection flag were produced during the single Groq detection pass, so no
    further API calls are needed. Only cases present in the sidecar are scored."""
    contra = [c for c in cases
              if c["label"] == "contradiction" and c["case_id"] in detail_by_id]
    negs   = [c for c in cases
              if c["label"] in ("clean", "hard_negative") and c["case_id"] in detail_by_id]
    print(f"Offline scoring — contradiction: {len(contra)}  negative: {len(negs)}")

    n_detected = n_fixed = off_wrong = on_wrong = collateral = 0
    residual_by_type: dict[str, list[int]] = {}

    for case in contra:
        d = detail_by_id[case["case_id"]]
        if not referee_fixed(case, case["response_text"]):
            off_wrong += 1
        on_ok = referee_fixed(case, d["corrected_text"])
        if d.get("ours_pred"):
            n_detected += 1
            if on_ok:
                n_fixed += 1
        if not on_ok:
            on_wrong += 1
        residual_by_type.setdefault(case["corruption"]["type"], []).append(0 if on_ok else 1)

    for case in negs:
        d = detail_by_id[case["case_id"]]
        if _norm(d["corrected_text"]) != _norm(case["response_text"]):
            collateral += 1

    return _build_results(contra, negs, n_detected, n_fixed, off_wrong,
                          on_wrong, collateral, residual_by_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test-set", default=TEST_SET)
    ap.add_argument("--offline", default="",
                    help="path to ours_case_detail.jsonl — score offline from "
                         "the detection pass (no Groq calls)")
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    with open(args.test_set, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[:args.limit]

    if args.offline:
        with open(args.offline, encoding="utf-8") as f:
            detail_by_id = {d["case_id"]: d
                            for d in (json.loads(l) for l in f if l.strip())}
        results = run_offline(cases, detail_by_id)
    else:
        results = asyncio.run(run(cases))

    print("\n" + "=" * 60)
    print("CORRECTION EXPERIMENT (ON vs OFF)")
    print("=" * 60)
    print("User-facing contradiction rate:")
    print(f"  OFF : {results['off']['user_facing_contradiction_rate']:.3f} "
          f"({results['off']['user_facing_contradictions']}/{results['n_contradiction']})")
    print(f"  ON  : {results['on']['user_facing_contradiction_rate']:.3f} "
          f"({results['on']['user_facing_contradictions']}/{results['n_contradiction']})")
    print(f"Detection rate     : {results['on']['detection_rate']:.3f} "
          f"({results['on']['n_detected']}/{results['n_contradiction']})")
    print(f"P(fix | detect)    : {results['on']['p_fix_given_detect']:.3f} "
          f"({results['on']['n_fixed']}/{results['on']['n_detected']})")
    print(f"Collateral damage  : {results['collateral_damage_rate']:.3f} "
          f"({results['n_collateral']}/{results['n_negative']})")
    print("\nResidual contradiction by type (ON):")
    for t, r in results["residual_by_corruption"].items():
        print(f"  {t:<20} {r['residual']:>3}/{r['total']:<3} residual_rate={r['residual_rate']:.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
