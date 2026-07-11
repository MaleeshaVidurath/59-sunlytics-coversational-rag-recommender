# m3_implementation/test_result/hallucination_result/expanded_eval/run_remaining_evals.py
#
# Runs only the stages Colab could NOT cover (division of labour documented
# in EXPANDED_RESULTS.md):
#   - LLM judge — needs the local Groq key; GPU irrelevant for API calls
#   - HHEM      — crashed on Colab's newer transformers
#                 ('all_tied_weights_keys'); works on local 4.57.6
#   - SummaC    — crashed on Colab (batch_encode_plus removed); local shim OK,
#                 CPU-slow → seeded 250-case stratified samples
# Colab covered FULL runs of: ours, naive NLI, LettuceDetect (both sets).
#
# Run:  python test_result/hallucination_result/expanded_eval/run_remaining_evals.py

import os
import subprocess
import sys
import time

_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.normpath(os.path.join(_DIR, ".."))
PY     = sys.executable
DETECT = os.path.join(PARENT, "run_detector_eval.py")
EXTERN = os.path.join(PARENT, "external_baselines", "run_external_baselines.py")
STD    = os.path.join(_DIR, "labeled_test_set_expanded.jsonl")
HARD   = os.path.join(_DIR, "hard_set", "labeled_hard_set.jsonl")

STAGES = [
    ("R1 LLM judge sample 800 — standard",
     [PY, "-u", DETECT, "--test-set", STD, "--skip-naive", "--sample", "800",
      "--out", os.path.join(_DIR, "results_standard_judge800.json")]),
    ("R2 LLM judge sample 600 — hard",
     [PY, "-u", DETECT, "--test-set", HARD, "--skip-naive", "--sample", "600",
      "--out", os.path.join(_DIR, "results_hard_judge600.json")]),
    ("R3 HHEM FULL — standard",
     [PY, "-u", EXTERN, "--test-set", STD, "--tools", "hhem",
      "--out", os.path.join(_DIR, "results_external_standard_hhem.json")]),
    ("R4 HHEM FULL — hard",
     [PY, "-u", EXTERN, "--test-set", HARD, "--tools", "hhem",
      "--out", os.path.join(_DIR, "results_external_hard_hhem.json")]),
    ("R5 SummaC sample 250 — standard",
     [PY, "-u", EXTERN, "--test-set", STD, "--tools", "summac", "--sample", "250",
      "--out", os.path.join(_DIR, "results_external_standard_summac.json")]),
    ("R6 SummaC sample 250 — hard",
     [PY, "-u", EXTERN, "--test-set", HARD, "--tools", "summac", "--sample", "250",
      "--out", os.path.join(_DIR, "results_external_hard_summac.json")]),
]


def main():
    t_all = time.time()
    for name, cmd in STAGES:
        print(f"\nSTAGE {name} (started {time.strftime('%H:%M:%S')})", flush=True)
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, check=False).returncode
        except Exception as e:
            print(f"STAGE {name} CRASHED: {e}", flush=True)
            continue
        status = "DONE" if rc == 0 else f"FAILED rc={rc}"
        print(f"STAGE {name} {status} in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"\nALL REMAINING STAGES FINISHED in {(time.time()-t_all)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
