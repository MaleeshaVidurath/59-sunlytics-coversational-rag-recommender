# m3_implementation/test_result/contradiction_result_v2/run_v2_eval.py
#
# Evaluation of the v2 cross-turn consistency layer (assertion ledger).
#
# WHY THIS LIVES IN A SEPARATE FOLDER
#   The v1 results in ../contradiction_result/ are the ones reported in the
#   final report (Tables 15/16, Figures 43/44). Nothing here writes to that
#   folder. The labeled test set is READ from it, unchanged, so both versions
#   are scored on identical data with identical metric code.
#
# WHAT CHANGED BETWEEN THE VERSIONS
#   v1 compared the response against the CURRENT turn's evidence. Every case in
#   the test set corrupts one attribute while leaving that evidence correct, so
#   v1 reported all of them as contradictions.
#
#   v2 treats that same situation as the hallucination guard's territory: the
#   mismatch is still identified, but it is recorded as "deferred" rather than
#   reported, because counting it here double-reports a single failure.
#
#   So v2 is expected to REPORT far fewer contradictions on this test set. That
#   is the designed behaviour, not a regression, and this script measures both
#   quantities separately so the distinction is visible:
#
#     v2_reported  — verdicts that change what the user sees (drift | stale)
#     v2_detected  — every mismatch identified, including deferred ones
#
#   v2_detected vs v1's recall answers "did we lose detection power?".
#   v2_reported vs v1's recall answers "how much of v1's recall was duplication?".
#
# NO LLM CALLS
#   v2 extracts deterministically, so this runs offline in minutes with no Groq
#   quota and no rate-limit artefacts. v1's runner needed a paced Groq call per
#   case and a non-production model id to survive the free tier.
#
# BASELINES
#   Not re-run. They are independent of our detector and unchanged, so their
#   numbers are carried forward verbatim from v1's results file for the
#   comparison table.
#
# Run:  python test_result/contradiction_result_v2/run_v2_eval.py
#         [--limit N] [--test-set PATH] [--out-dir PATH]

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

# Same statistics and table printer both earlier chapters used.
from test_result.hallucination_result.run_detector_eval import (
    compute_metrics, print_metrics_table,
)
# Same breakdown helpers as v1 — imported, not reimplemented, so the two
# versions cannot drift apart in how they slice the results.
from test_result.contradiction_result.run_contra_eval import (
    recall_by_corruption, recall_by_distance, false_alarms, _stratified_sample,
)

from memory.core.assertion_ledger import AssertionLedger, TRACKED_ATTRIBUTES
from memory.core.contradiction_detector import (
    _attributable_refs, _classify_assertion,
)
from text_rag.core.assertion_extractor import extract_assertions, replace_in_scope

_DIR     = os.path.dirname(os.path.abspath(__file__))
_V1_DIR  = os.path.join(_DIR, "..", "contradiction_result")
TEST_SET = os.path.join(_V1_DIR, "labeled_test_set.jsonl")
V1_RESULTS = os.path.join(_V1_DIR, "results_contra_eval.json")

# Verdicts that change the text the user sees.
_REPORTED_KINDS = {"drift", "stale"}
# Verdicts that mean "a mismatch was identified", whoever ends up owning it.
_DETECTED_KINDS = {"drift", "stale", "defer"}


# ── Rebuilding one turn's state offline ──────────────────────────────────────

def _seed_ledger(case: dict) -> AssertionLedger:
    """
    Rebuilds the session ledger as it stood BEFORE this turn.

    graph_before holds the products established on earlier turns. Each of their
    values is replayed as both an evidence record and a response assertion, so
    prior_assertion() can anchor exactly as it would live. An empty
    graph_before (turn_distance 0) simply yields a ledger with no history,
    which is the correct representation of a first mention.
    """
    ledger = AssertionLedger()
    prior_nodes = case.get("graph_before") or {}

    if prior_nodes:
        ledger.begin_turn(f"{case.get('turn_id', 't')}__prior", action="seed")
        for aid, node in prior_nodes.items():
            name = node.get("name", "")
            for attribute in TRACKED_ATTRIBUTES:
                value = node.get(attribute)
                if not value:
                    continue
                ledger.apply_evidence(str(aid), attribute, value, name_hint=name)
                ledger.record_assertion(str(aid), attribute, value,
                                        status="active", name_hint=name)

    ledger.begin_turn(case.get("turn_id") or "t", action=case.get("action", ""))
    return ledger


def evaluate_case(case: dict) -> dict:
    """
    Runs v2's reconciliation on one case using the production functions.

    Offline, so there is no live PostgreSQL read: truth_is_live is False
    throughout and the anchor falls to the prior assertion or to session
    context, never to live_evidence. That matches how the detector behaves
    when re-verification is unavailable, and it keeps the scoring deterministic.
    """
    ledger   = _seed_ledger(case)
    response = case["response_text"]
    refs     = [
        {
            "article_id":   str(r.get("article_id", "")),
            "name":         r.get("name", "") or "",
            "colour":       r.get("colour", "") or "",
            "price":        r.get("price", "") or "",
            "product_type": r.get("product_type", "") or "",
        }
        for r in (case.get("product_refs") or [])
        if r.get("article_id")
    ]

    evidence_ids = {r["article_id"] for r in refs}
    truth, guarded = {}, set()
    for ref in refs:
        aid = ref["article_id"]
        for attribute in TRACKED_ATTRIBUTES:
            value = ref.get(attribute)
            if value:
                truth[(aid, attribute)] = value
                guarded.add((aid, attribute))
            ledger.apply_evidence(aid, attribute, value, name_hint=ref["name"])

    # Same candidate widening as production: this turn's evidence first, then
    # everything the session already knows.
    seen = set(evidence_ids)
    candidates = list(refs)
    for known in ledger.known_products():
        if known["article_id"] not in seen:
            seen.add(known["article_id"])
            candidates.append(known)

    attributable = _attributable_refs(candidates, evidence_ids, set(), response)
    described = [
        {
            "article_id":   r["article_id"],
            "name":         truth.get((r["article_id"], "name"), r.get("name", "")),
            "colour":       truth.get((r["article_id"], "colour"), r.get("colour", "")),
            "price":        truth.get((r["article_id"], "price"), r.get("price", "")),
            "product_type": truth.get((r["article_id"], "product_type"),
                                      r.get("product_type", "")),
        }
        for r in attributable
    ]

    extraction = extract_assertions(response, described, verbose=False)
    sentences  = extraction["sentences"]
    names      = {d["article_id"]: d["name"] for d in described}

    corrected, kinds, anchors = response, [], []
    for a in extraction["assertions"]:
        aid, attribute = a["article_id"], a["attribute"]
        verdict = _classify_assertion(
            stated        = a["value"],
            slot_truth    = truth.get((aid, attribute), ""),
            is_guarded    = (aid, attribute) in guarded,
            revision      = None,          # the test set never moves the DB
            prior         = ledger.prior_assertion(aid, attribute),
            attribute     = attribute,
            product_name  = names.get(aid, ""),
            truth_is_live = False,
        )
        kinds.append(verdict["kind"])
        if verdict["anchor_source"]:
            anchors.append(verdict["anchor_source"])
        if verdict["correct_to"]:
            corrected = replace_in_scope(
                corrected, sentences, a["sentence_idx"],
                a["value"], verdict["correct_to"],
            )

    return {
        "case_id":   case.get("case_id"),
        "label":     case.get("label"),
        "reported":  any(k in _REPORTED_KINDS for k in kinds),
        "detected":  any(k in _DETECTED_KINDS for k in kinds),
        "deferred":  sum(1 for k in kinds if k == "defer"),
        "kinds":     kinds,
        "anchors":   anchors,
        "corrected_text": corrected,
    }


# ── Correction referee (identical rule to v1's, no shared logic with detector) ─

def _fixed(case: dict, final_text: str) -> bool:
    """
    Grades the shipped response against the corruption ground truth.

      single-field drift : correct value present AND wrong value absent
      cross_item_swap    : final text equals the original clean response
    """
    corruption = case.get("corruption") or {}
    original   = corruption.get("original")
    corrupted  = corruption.get("corrupted")

    if corruption.get("type") == "cross_item_swap" or not (original and corrupted):
        return final_text.strip() == (case.get("source_response") or "").strip()

    low = final_text.lower()
    return original.lower() in low and corrupted.lower() not in low


# ── Main ─────────────────────────────────────────────────────────────────────

def _carry_forward_baselines(named: dict) -> dict:
    """
    Adds v1's baseline rows to the comparison table, read verbatim from its
    results file. They are unaffected by the rewrite — re-running them would
    only risk perturbing numbers that are already in the report.
    """
    try:
        with open(V1_RESULTS, encoding="utf-8") as f:
            v1 = json.load(f)
    except Exception as e:
        print(f"  (v1 results unavailable — comparison omitted: {e})")
        return named

    for system in ("ours", "string_only", "history_nli", "uttr_pair_nli", "llm_judge"):
        block = v1.get(system) or {}
        metrics = block.get("metrics")
        if metrics:
            label = "v1 ours (graph+NLI)" if system == "ours" else f"v1 {system}"
            named[label] = metrics
    return named


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified sample of N cases using v1's sampler and "
                         "seed — use 599 to reproduce v1's evaluated slice")
    ap.add_argument("--test-set", default=TEST_SET)
    ap.add_argument("--out-dir", default=_DIR)
    ap.add_argument("--debug-case", default="",
                    help="print the extraction and verdicts for one case_id "
                         "and exit — for diagnosing a disagreement")
    args = ap.parse_args()

    if args.debug_case:
        import text_rag.core.assertion_extractor as _ax
        vocab = _ax._load_vocabularies()
        print(f"module   : {_ax.__file__}")
        print(f"vocab    : names={len(vocab['names'])} "
              f"colours={len(vocab['colours'])} types={len(vocab['types'])}")
        print(f"'Jacket' recognised as a type: {'Jacket' in vocab['types']}")
        with open(args.test_set, encoding="utf-8") as f:
            for line in f:
                case = json.loads(line)
                if case.get("case_id") != args.debug_case:
                    continue
                out = evaluate_case(case)
                print(f"case     : {case['case_id']}  "
                      f"corruption={case['corruption']}")
                print(f"kinds    : {out['kinds']}")
                print(f"detected : {out['detected']}  reported: {out['reported']}")
                return
        print(f"case {args.debug_case} not found")
        return

    with open(args.test_set, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if args.sample and args.sample < len(cases):
        # v1's own sampler, same seed — the ONLY way the comparison row is
        # like-for-like. v1 was scored on a 599-case stratified sample because
        # its per-case Groq call could not cover the full set in one day's
        # token budget; v2 needs no LLM and can run everything, but a number
        # from 1,346 cases must not be placed beside one from 599.
        cases = _stratified_sample(cases, args.sample)
        print(f"Stratified sample: {len(cases)} cases (v1's sampler, seed 123)")

    y_true = [c["label"] == "contradiction" for c in cases]
    n_pos  = sum(y_true)
    print(f"Loaded {len(cases)} cases "
          f"({n_pos} contradiction / {len(cases) - n_pos} negative)\n")

    print("[v2] reconciling (deterministic — no LLM calls)...")
    t0, outcomes = time.time(), []
    for i, case in enumerate(cases):
        with contextlib.redirect_stdout(io.StringIO()):   # silence detector logs
            outcomes.append(evaluate_case(case))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.0f}s\n")

    preds_reported = [o["reported"] for o in outcomes]
    preds_detected = [o["detected"] for o in outcomes]

    results = {
        "n_cases":         len(cases),
        "n_contradiction": n_pos,
        "note": ("v2_reported = verdicts that change the shipped response. "
                 "v2_detected = every mismatch identified, including those "
                 "deferred to the hallucination guard."),
    }
    named = {}

    # Both quantities are recorded in full. Only "detected" enters the printed
    # comparison, because it is the only one answering the same question as
    # every other row there — "did the system find the mismatch?". "reported"
    # answers a different question — "did it rewrite the text?" — and a
    # near-zero bar for it on a detection axis reads as failure rather than as
    # the deliberate hand-off it is. That is told properly by the ownership
    # split below and by figV2.
    for name, preds in (("v2 reported", preds_reported),
                        ("v2 detected", preds_detected)):
        block = {
            "metrics":              compute_metrics(y_true, preds),
            "recall_by_corruption": recall_by_corruption(cases, preds),
            "recall_by_distance":   recall_by_distance(cases, preds),
            "false_alarms":         false_alarms(cases, preds),
        }
        results[name.replace(" ", "_")] = block
        if name == "v2 detected":
            named[name] = block["metrics"]

    # ── Ownership split — the quantitative form of the anti-duplication claim ─
    contra_idx  = [i for i, t in enumerate(y_true) if t]
    n_detected  = sum(outcomes[i]["detected"] for i in contra_idx)
    n_deferred  = sum(1 for i in contra_idx
                      if outcomes[i]["detected"] and not outcomes[i]["reported"])
    results["ownership"] = {
        "contradiction_cases":     len(contra_idx),
        "detected":                n_detected,
        "deferred_to_guard":       n_deferred,
        "reported_by_v2":          n_detected - n_deferred,
        "share_deferred":          round(n_deferred / n_detected, 4) if n_detected else 0.0,
    }

    # ── Correction experiment (same referee rule as v1's Experiment B) ────────
    fixed = sum(1 for i in contra_idx if _fixed(cases[i], outcomes[i]["corrected_text"]))
    reported = sum(outcomes[i]["reported"] for i in contra_idx)
    damaged = sum(
        1 for i, o in enumerate(outcomes)
        if not y_true[i] and o["corrected_text"] != cases[i]["response_text"]
    )
    n_neg = len(cases) - n_pos
    results["correction"] = {
        "user_facing_contradiction_rate_off": 1.0,
        "user_facing_contradiction_rate_on":  round(1 - fixed / len(contra_idx), 4)
                                              if contra_idx else 0.0,
        "reported_rate":        round(reported / len(contra_idx), 4) if contra_idx else 0.0,
        "p_fix_given_reported": round(fixed / reported, 4) if reported else 0.0,
        "collateral_damage_rate": round(damaged / n_neg, 4) if n_neg else 0.0,
    }

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 62)
    print("DETECTION ACCURACY (positive class = contradiction)")
    print("=" * 62)
    print_metrics_table(_carry_forward_baselines(dict(named)))

    own = results["ownership"]
    print("\nOWNERSHIP OF DETECTED CONTRADICTIONS")
    print(f"  detected by v2            {own['detected']}/{own['contradiction_cases']}")
    print(f"  deferred to the guard     {own['deferred_to_guard']} "
          f"({own['share_deferred']*100:.1f}% of detections)")
    print(f"  reported by v2 itself     {own['reported_by_v2']}")
    print("  → the deferred share is the duplication v1's recall was counting.")
    print("  (the 'v2 reported' metrics are kept in results_v2_eval.json; they "
          "are\n   left out of the table above because they measure rewriting, "
          "not detection)")

    print("\nRECALL BY TURN DISTANCE")
    for system in named:
        key = system.replace(" ", "_")
        row = "  ".join(
            f"d={k}:{v['recall']:.2f}({v['detected']}/{v['total']})"
            for k, v in results[key]["recall_by_distance"].items()
        )
        print(f"  {system:<14} {row}")

    print("\nRECALL BY CORRUPTION TYPE")
    for system in named:
        key = system.replace(" ", "_")
        print(f"  {system}:")
        for ctype, r in results[key]["recall_by_corruption"].items():
            print(f"    {ctype:<20} {r['detected']:>4}/{r['total']:<4} "
                  f"recall={r['recall']:.3f}")

    print("\nFALSE ALARMS (negative class)")
    for system in named:
        fa = results[system.replace(' ', '_')]["false_alarms"]
        print(f"  {system:<14} clean={fa['clean']['fp']}/{fa['clean']['total']}  "
              f"hard_neg={fa['hard_negative']['fp']}/{fa['hard_negative']['total']}")

    corr = results["correction"]
    print("\nCORRECTION EXPERIMENT (contradiction cases)")
    print(f"  user-facing rate OFF      {corr['user_facing_contradiction_rate_off']:.3f}")
    print(f"  user-facing rate ON       {corr['user_facing_contradiction_rate_on']:.3f}")
    print(f"  reported rate             {corr['reported_rate']:.3f}")
    print(f"  P(fix | reported)         {corr['p_fix_given_reported']:.3f}")
    print(f"  collateral damage         {corr['collateral_damage_rate']:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "results_v2_eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    out_detail = os.path.join(args.out_dir, "v2_case_detail.jsonl")
    with open(out_detail, "w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"\nWrote {os.path.basename(out_json)} and {os.path.basename(out_detail)}")
    print(f"v1 results in ../contradiction_result/ were not modified.")


if __name__ == "__main__":
    main()
