# m3_implementation/test_result/hallucination_result/loop_mitigation/regrade_shipped.py
#
# Re-grades the already-shipped responses with the current referee — no LLM
# calls, no loop re-run. Used after a referee refinement (e.g. allowing
# derived price differences, classifying truncated names as minor).
# Rewrites results_loop_eval.json and shipped_responses.jsonl in place.
#
# Run:  python test_result/hallucination_result/loop_mitigation/regrade_shipped.py

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from test_result.hallucination_result.loop_mitigation.referee import grade

_DIR     = os.path.dirname(os.path.abspath(__file__))
TEST_SET = os.path.join(_DIR, "..", "original_eval_238", "labeled_test_set.jsonl")
RESULTS  = os.path.join(_DIR, "results_loop_eval.json")
SHIPPED  = os.path.join(_DIR, "shipped_responses.jsonl")


def main():
    with open(TEST_SET, encoding="utf-8") as f:
        evidence_by_id = {c["case_id"]: c["evidence"]
                          for c in map(json.loads, f) if c["label"] == "hallucinated"}
    with open(RESULTS, encoding="utf-8") as f:
        results = json.load(f)
    with open(SHIPPED, encoding="utf-8") as f:
        shipped = {r["case_id"]: r for r in map(json.loads, f)}

    records = results["records"]
    for rec in records:
        s = shipped[rec["case_id"]]
        if rec.get("shipped_is_corrupted"):
            rec["shipped_correct"] = False
            rec["problems"] = ["planted corrupted response shipped unchanged"]
            rec["minor_issues"] = []
        else:
            correct, problems, minor = grade(
                evidence_by_id[rec["case_id"]], s["shipped_response"]
            )
            rec["shipped_correct"] = correct
            rec["problems"] = problems
            rec["minor_issues"] = minor
        s["shipped_correct"] = rec["shipped_correct"]
        s["problems"] = rec["problems"]
        s["minor_issues"] = rec["minor_issues"]

    # ── Recompute metrics ────────────────────────────────────────────────────
    n = len(records)
    detected  = [r for r in records if r["detected"]]
    corrected = [r for r in detected if r["shipped_correct"]]
    wrong_on  = [r for r in records if not r["shipped_correct"]]
    with_minor = [r for r in records if r.get("minor_issues")]

    def rate(x, y):
        return round(len(x) / len(y), 4) if y else 0.0

    by_type = {}
    for ctype in sorted({r["corruption_type"] for r in records}):
        subset = [r for r in records if r["corruption_type"] == ctype]
        wrong  = [r for r in subset if not r["shipped_correct"]]
        by_type[ctype] = {
            "n": len(subset),
            "detected": sum(1 for r in subset if r["detected"]),
            "still_wrong_with_loop": len(wrong),
            "residual_rate": rate(wrong, subset),
        }

    results["loop_on"].update({
        "wrong_shipped": len(wrong_on),
        "residual_hallucination_rate": rate(wrong_on, records),
        "corrected_after_detection": len(corrected),
        "p_correction_given_detection": rate(corrected, detected),
        "cases_with_minor_issues": len(with_minor),
        "minor_issue_note": "truncated-but-unambiguous item names — fidelity "
                            "blemish, reported separately, not hallucination",
    })
    results["by_corruption_type"] = by_type
    results["wrong_with_loop_case_ids"] = [r["case_id"] for r in wrong_on]
    results["attempts_histogram"] = dict(sorted(
        Counter(r["attempts"] for r in records).items()))

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(SHIPPED, "w", encoding="utf-8") as f:
        for r in shipped.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("REGRADED LOOP MITIGATION RESULT")
    print("=" * 50)
    print(f"cases:                        {n}")
    print(f"LOOP OFF wrong shipped:       {n}/{n} (100%)")
    print(f"LOOP ON  wrong shipped:       {len(wrong_on)}/{n} "
          f"({100 * rate(wrong_on, records):.1f}%)")
    print(f"detected on attempt 1:        {len(detected)}/{n}")
    print(f"P(correct final | detected):  {100 * rate(corrected, detected):.1f}%")
    print(f"minor name truncations:       {len(with_minor)} cases (not counted wrong)")
    print("\nBy corruption type (still wrong with loop ON):")
    for ctype, d in by_type.items():
        print(f"  {ctype:<18} {d['still_wrong_with_loop']}/{d['n']}")
    print("\nWrong case breakdown:")
    missed = [r for r in wrong_on if not r["detected"]]
    failed = [r for r in wrong_on if r["detected"]]
    print(f"  checker missed (lie shipped unchanged): {len(missed)}")
    print(f"  detected but final still wrong:         {len(failed)}")
    for r in failed:
        print(f"    {r['case_id']} [{r['corruption_type']}] {r['problems'][:1]}")


if __name__ == "__main__":
    main()
