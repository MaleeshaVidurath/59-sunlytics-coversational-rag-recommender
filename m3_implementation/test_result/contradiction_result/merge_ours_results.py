# m3_implementation/test_result/contradiction_result/merge_ours_results.py
#
# Merges the rate-limit-corrected `ours` + `string_only` blocks (from the
# --skip-nli --skip-llm re-run, results_ours_fixed.json) into the main results
# file (results_contra_eval.json), whose history_nli / uttr_pair_nli / llm_judge
# blocks are valid (they do not use Groq claim extraction, so were never
# affected by the 429 rate-limiting). The stratified sample is deterministic
# (seed 123), so both runs scored the identical case set.
#
# Run:  python test_result/contradiction_result/merge_ours_results.py

import json
import os

_DIR  = os.path.dirname(os.path.abspath(__file__))
MAIN  = os.path.join(_DIR, "results_contra_eval.json")
FIXED = os.path.join(_DIR, "results_ours_fixed.json")


def main():
    with open(MAIN, encoding="utf-8") as f:
        main_res = json.load(f)
    with open(FIXED, encoding="utf-8") as f:
        fixed = json.load(f)

    if fixed.get("n_cases") != main_res.get("n_cases"):
        print(f"WARNING: sample size mismatch "
              f"(main={main_res.get('n_cases')} fixed={fixed.get('n_cases')}) "
              f"— seeds may differ; merge aborted.")
        return

    for system in ("ours", "string_only"):
        if system in fixed:
            main_res[system] = fixed[system]
            m = fixed[system]["metrics"]
            print(f"merged {system}: P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} BalAcc={m['balanced_accuracy']:.3f}")

    with open(MAIN, "w", encoding="utf-8") as f:
        json.dump(main_res, f, indent=2, ensure_ascii=False)
    print(f"Merged into {os.path.basename(MAIN)}")


if __name__ == "__main__":
    main()
