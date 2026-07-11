# m3_implementation/test_result/hallucination_result/expanded_eval/build_expanded_set.py
#
# Builds the EXPANDED evaluation dataset — fully isolated from the original
# evaluation. Nothing outside expanded_eval/ is modified, except restoring
# the original 41-case captured_cases.jsonl to its pre-run state.
#
# SOURCES MERGED (deduplicated by normalized response text):
#   1. original 41 scripted cases   (captured_cases_20260710_202447.jsonl —
#      the file the driver rotated away; restored back to captured_cases.jsonl)
#   2. new 60-conversation driver run (current captured_cases.jsonl after the
#      run; copied to driver_run2_cases.jsonl here)
#   3. real chat history from MongoDB (mongo_cases.jsonl, see mongo_harvest.py)
#   EXCLUDED: captured_cases_20260710_203548.jsonl — an aborted partial run
#   whose conversations were fully re-executed by the complete run.
#
# OUTPUTS (all inside expanded_eval/):
#   captured_expanded.jsonl          merged raw cases
#   labeled_test_set_expanded.jsonl  standard corruption set (reuses
#                                    corrupt_cases.py functions, seed 42)
#   flagged_for_review_expanded.jsonl / clean_audit_expanded.txt
#   hard_set/labeled_hard_set.jsonl  adversarial set (make_hard_cases, seed 77)
#
# Run:  python test_result/hallucination_result/expanded_eval/build_expanded_set.py

import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.normpath(os.path.join(_DIR, ".."))

ORIGINAL_41   = os.path.join(PARENT, "original_eval_238", "captured_cases_20260710_202447.jsonl")
CURRENT       = os.path.join(PARENT, "captured_cases.jsonl")
DRIVER_RUN2   = os.path.join(_DIR, "driver_run2_cases.jsonl")
MONGO_CASES   = os.path.join(_DIR, "mongo_cases.jsonl")
MERGED        = os.path.join(_DIR, "captured_expanded.jsonl")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    # ── 1. Secure the driver-run output, restore the original 41 ────────────
    if not os.path.exists(DRIVER_RUN2):
        shutil.copy2(CURRENT, DRIVER_RUN2)
        print(f"driver run secured → {os.path.basename(DRIVER_RUN2)}")
        shutil.copy2(ORIGINAL_41, CURRENT)
        print("original 41-case captured_cases.jsonl restored")

    # ── 2. Merge with dedupe (captured sources take priority over mongo) ────
    sources = [("original", _load(ORIGINAL_41)),
               ("driver_run2", _load(DRIVER_RUN2)),
               ("mongodb", _load(MONGO_CASES))]
    seen, merged = set(), []
    per_source = {}
    for name, rows in sources:
        kept = 0
        for r in rows:
            key = _norm(r.get("response_text", ""))[:400]
            if not key or key in seen:
                continue
            seen.add(key)
            r.setdefault("source", name)
            merged.append(r)
            kept += 1
        per_source[name] = (len(rows), kept)

    with open(MERGED, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    n_pass = sum(1 for r in merged if r["checker"].get("passed"))
    print("\nMerged sources (rows → kept after dedupe):")
    for name, (total, kept) in per_source.items():
        print(f"  {name:<12} {total:>4} → {kept}")
    print(f"captured_expanded.jsonl: {len(merged)} cases "
          f"({n_pass} checker-passed / {len(merged)-n_pass} flagged)")

    # ── 3. Standard corruption set (reuse corrupt_cases.py, redirected) ─────
    from test_result.hallucination_result import corrupt_cases as cc
    cc.CAPTURED    = MERGED
    cc.TEST_SET    = os.path.join(_DIR, "labeled_test_set_expanded.jsonl")
    cc.FLAGGED_OUT = os.path.join(_DIR, "flagged_for_review_expanded.jsonl")
    cc.AUDIT_OUT   = os.path.join(_DIR, "clean_audit_expanded.txt")
    print("\n── standard corruption set (seed 42) ──")
    cc.build_test_set()

    # ── 4. Hard set (reuse make_hard_cases, redirected) ─────────────────────
    from test_result.hallucination_result.expanded_eval.hard_set import make_hard_cases as hc
    hc.CAPTURED = MERGED
    hc.HARD_SET = os.path.join(_DIR, "hard_set", "labeled_hard_set.jsonl")
    print("\n── hard corruption set (seed 77) ──")
    hc.main()


if __name__ == "__main__":
    main()
