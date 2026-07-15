# m3_implementation/test_result/hallucination_result/loop_mitigation/run_loop_eval.py
#
# Loop-mitigation experiment (CRAG-style system-on/off comparison).
#
# QUESTION: when a hallucinated response is detected, does the
# detect-reject-regenerate loop actually deliver a correct response to the
# user — and how many bad responses reach the user WITH the loop vs WITHOUT?
#
# METHOD (induced failure): each of the 205 corrupted cases from
# ../labeled_test_set.jsonl plays the role of the LLM's attempt-1 output.
#
#   LOOP OFF arm: the corrupted response ships as-is → wrong by construction
#                 (this is what a system without the loop does).
#   LOOP ON arm:  the REAL pipeline logic runs unchanged —
#                 attempt 1 (corrupted) → HallucinationChecker
#                   flagged → ResponseGenerator.generate(strictness=1,
#                             contradicted_fields) → checker again
#                   flagged → strictness=2 → attempt 3 always ships
#                 (mirrors rag_pipeline.py MAX_REGENERATION_ATTEMPTS=3)
#
# GRADING: the independent referee (referee.py) grades every shipped response
# against the database truth. The checker is NOT the grader — it decides what
# ships (system under test); the referee decides what was actually correct
# (measurement). A shipped response identical to the corrupted input is wrong
# by construction (the planted lie reached the user).
#
# Needs: GROQ_API_KEY in .env (regenerations) + local NLI models. No databases.
#
# Run:  python test_result/hallucination_result/loop_mitigation/run_loop_eval.py [--limit N]

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dotenv import load_dotenv
load_dotenv()

_DIR     = os.path.dirname(os.path.abspath(__file__))
TEST_SET = os.path.join(_DIR, "..", "original_eval_238", "labeled_test_set.jsonl")
RESULTS  = os.path.join(_DIR, "results_loop_eval.json")

MAX_ATTEMPTS = 3   # mirrors MAX_REGENERATION_ATTEMPTS in text_rag/config.py


async def run_loop_on_case(case, checker, generator):
    """Runs the real detect-reject-regenerate logic with the corrupted
    response as attempt 1. Returns a per-case record."""
    from test_result.hallucination_result.loop_mitigation.referee import grade

    evidence  = case["evidence"]
    corrupted = case["response_text"]

    record = {
        "case_id":         case["case_id"],
        "corruption_type": case["corruption"]["type"],
        "action":          case["action"],
        "detected":        False,
        "attempts":        1,
        "regenerations":   0,
        "gen_seconds":     0.0,
        "shipped_is_corrupted": False,
    }

    response = corrupted
    contradicted = []
    shipped = corrupted

    for attempt in range(1, MAX_ATTEMPTS + 1):
        record["attempts"] = attempt

        with contextlib.redirect_stdout(io.StringIO()):
            check = checker.check(response, evidence)

        if attempt == 1:
            record["detected"] = check["has_hallucination"]

        if not check["has_hallucination"]:
            shipped = response          # checker approved → ships
            break

        contradicted = check.get("contradicted_fields", contradicted)
        shipped = response              # attempt 3 ships even if flagged

        if attempt == MAX_ATTEMPTS:
            break

        # regenerate — REAL generator, same call as rag_pipeline.py
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                response = await generator.generate(
                    evidence=evidence,
                    strictness=attempt,          # attempt 1 failed → strictness 1, then 2
                    contradicted_fields=contradicted,
                )
        except Exception as e:
            record["gen_error"] = str(e)[:200]
            break
        record["gen_seconds"] += time.time() - t0
        record["regenerations"] += 1
        if not response:
            record["gen_error"] = "empty generation"
            break
        await asyncio.sleep(0.3)        # stay under Groq free-tier rate limits

    record["shipped_is_corrupted"] = (shipped == corrupted)
    if record["shipped_is_corrupted"]:
        # the planted lie reached the user — wrong by construction
        record["shipped_correct"] = False
        record["problems"] = ["planted corrupted response shipped unchanged"]
        record["minor_issues"] = []
    else:
        correct, problems, minor = grade(evidence, shipped)
        record["shipped_correct"] = correct
        record["problems"] = problems
        record["minor_issues"] = minor
    record["shipped_response"] = shipped
    return record


async def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N cases (smoke test)")
    ap.add_argument("--test-set", default=TEST_SET,
                    help="path to a labeled test set jsonl")
    ap.add_argument("--out", default=RESULTS, help="path for the results json")
    ap.add_argument("--sample", type=int, default=0,
                    help="seeded random sample of N corrupted cases (seed 123)")
    args = ap.parse_args()
    RESULTS = args.out

    with open(args.test_set, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]
    cases = [c for c in cases if c["label"] == "hallucinated"]
    if args.limit:
        cases = cases[:args.limit]
    if args.sample and args.sample < len(cases):
        import random as _random
        cases = _random.Random(123).sample(cases, args.sample)
        print(f"Seeded sample: {len(cases)} corrupted cases (seed 123)")
    print(f"Loop-mitigation experiment: {len(cases)} induced-hallucination cases")

    from text_rag.core.hallucination_checker import HallucinationChecker
    from text_rag.core.response_generator import ResponseGenerator
    checker   = HallucinationChecker()
    generator = ResponseGenerator()

    records = []
    t0 = time.time()
    for i, case in enumerate(cases):
        rec = await run_loop_on_case(case, checker, generator)
        records.append(rec)
        if (i + 1) % 10 == 0:
            n_ok = sum(1 for r in records if r["shipped_correct"])
            print(f"  {i+1}/{len(cases)}  correct_so_far={n_ok}  ({time.time()-t0:.0f}s)")

    # ── Metrics ──────────────────────────────────────────────────────────────
    n = len(records)
    detected  = [r for r in records if r["detected"]]
    corrected = [r for r in detected if r["shipped_correct"]]
    wrong_on  = [r for r in records if not r["shipped_correct"]]
    gen_errors = [r for r in records if "gen_error" in r]

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

    attempts_hist = dict(sorted(Counter(r["attempts"] for r in records).items()))
    avg_gen_s = (sum(r["gen_seconds"] for r in records) / len(detected)) if detected else 0.0

    results = {
        "n_cases": n,
        "loop_off": {
            "wrong_shipped": n,
            "residual_hallucination_rate": 1.0,
            "note": "without the loop every induced-bad response ships by construction",
        },
        "loop_on": {
            "wrong_shipped": len(wrong_on),
            "residual_hallucination_rate": rate(wrong_on, records),
            "detected_attempt1": len(detected),
            "detection_rate": rate(detected, records),
            "corrected_after_detection": len(corrected),
            "p_correction_given_detection": rate(corrected, detected),
            "attempts_histogram": attempts_hist,
            "avg_regeneration_seconds_per_detected_case": round(avg_gen_s, 2),
            "generation_errors": len(gen_errors),
        },
        "by_corruption_type": by_type,
        "wrong_with_loop_case_ids": [r["case_id"] for r in wrong_on],
        "records": [
            {k: v for k, v in r.items() if k != "shipped_response"} for r in records
        ],
    }

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # shipped responses for error analysis, separate file (large)
    shipped_path = RESULTS.replace(".json", "_shipped.jsonl") \
        if RESULTS != os.path.join(_DIR, "results_loop_eval.json") \
        else os.path.join(_DIR, "shipped_responses.jsonl")
    with open(shipped_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"case_id": r["case_id"],
                                "shipped_correct": r["shipped_correct"],
                                "problems": r["problems"],
                                "shipped_response": r["shipped_response"]},
                               ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("LOOP MITIGATION RESULT (induced failures)")
    print("=" * 60)
    print(f"cases:                         {n}")
    print(f"LOOP OFF  wrong shipped:       {n}/{n}  (100%)")
    print(f"LOOP ON   wrong shipped:       {len(wrong_on)}/{n}  "
          f"({100 * rate(wrong_on, records):.1f}%)")
    print(f"detected on attempt 1:         {len(detected)}/{n}  "
          f"({100 * rate(detected, records):.1f}%)")
    print(f"P(correct final | detected):   {100 * rate(corrected, detected):.1f}%")
    print(f"attempts histogram:            {attempts_hist}")
    print(f"avg regeneration time:         {avg_gen_s:.2f}s per detected case")
    print(f"generation errors:             {len(gen_errors)}")
    print("\nBy corruption type (residual wrong with loop ON):")
    for ctype, d in by_type.items():
        print(f"  {ctype:<18} {d['still_wrong_with_loop']}/{d['n']}  "
              f"(detected {d['detected']})")
    print(f"\nResults: {os.path.basename(RESULTS)} + shipped_responses.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
