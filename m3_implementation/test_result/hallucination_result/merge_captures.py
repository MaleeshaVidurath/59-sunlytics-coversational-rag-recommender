# m3_implementation/test_result/hallucination_result/merge_captures.py
#
# Merges rotated capture files back into captured_cases.jsonl.
#
# collect_cases.py ROTATES the existing capture file (renames it with a
# timestamp) before each scripted run, so after a new run the previous
# cases live in captured_cases_<timestamp>.jsonl. This script recombines
# every rotated file + the current file into one captured_cases.jsonl,
# in chronological order, de-duplicated (captured_at + session_id + attempt).
#
# Excluded: captured_cases_full_backup.jsonl (the pre-slimming backup).
#
# Run:  python test_result/hallucination_result/merge_captures.py

import glob
import json
import os

_DIR    = os.path.dirname(os.path.abspath(__file__))
CURRENT = os.path.join(_DIR, "captured_cases.jsonl")


def main():
    rotated = sorted(
        p for p in glob.glob(os.path.join(_DIR, "captured_cases_*.jsonl"))
        if not p.endswith("_full_backup.jsonl")
    )
    sources = rotated + ([CURRENT] if os.path.exists(CURRENT) else [])
    print("Merging, in order:")
    for p in sources:
        print(f"  {os.path.basename(p)}")

    seen, merged = set(), []
    for path in sources:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (rec.get("captured_at"), rec.get("session_id"),
                       rec.get("attempt"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(rec)

    merged.sort(key=lambda r: r.get("captured_at", ""))
    with open(CURRENT, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    n_passed = sum(1 for r in merged if r["checker"].get("passed"))
    print(f"\nMerged {len(merged)} unique cases → captured_cases.jsonl "
          f"({n_passed} checker-passed / {len(merged) - n_passed} flagged)")


if __name__ == "__main__":
    main()
