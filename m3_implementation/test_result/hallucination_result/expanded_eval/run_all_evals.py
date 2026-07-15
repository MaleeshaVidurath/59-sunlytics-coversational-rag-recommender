# m3_implementation/test_result/hallucination_result/expanded_eval/run_all_evals.py
#
# Sequential orchestrator for the expanded evaluation. Runs every detector on
# the expanded standard set (5218 rows) and the hard set (2963 rows), most
# important results first, one stage at a time (CPU-bound models must not
# compete for cores). Total wall time ≈ 12–16 h on CPU — designed to run
# detached overnight; each stage's results file lands as soon as it is done.
#
# SAMPLING POLICY (documented for the write-up):
#   - OUR CHECKER runs on the FULL sets — headline numbers, no sampling.
#   - Slow / API-limited baselines run on large seeded stratified samples
#     (seed 123): naive NLI 2000/1500, LLM judge 800/600 (Groq TPM limits),
#     SummaC 250 (≈28 s/case on CPU). At these sizes the 95% CIs are within
#     a couple of points — sampling is a runtime decision, not a shortcut,
#     and every results file records exactly what it ran on.
#
# Run:  python test_result/hallucination_result/expanded_eval/run_all_evals.py

import os
import subprocess
import sys
import time

_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT  = os.path.normpath(os.path.join(_DIR, ".."))
PY      = sys.executable
DETECT  = os.path.join(PARENT, "run_detector_eval.py")
EXTERN  = os.path.join(PARENT, "external_baselines", "run_external_baselines.py")
STD     = os.path.join(_DIR, "labeled_test_set_expanded.jsonl")
HARD    = os.path.join(_DIR, "hard_set", "labeled_hard_set.jsonl")

STAGES = [
    ("S1  ours FULL — expanded standard (5218)",
     [PY, "-u", DETECT, "--test-set", STD, "--skip-naive", "--skip-llm",
      "--out", os.path.join(_DIR, "results_standard_ours.json")]),
    ("S2  ours FULL — hard set (2963)",
     [PY, "-u", DETECT, "--test-set", HARD, "--skip-naive", "--skip-llm",
      "--out", os.path.join(_DIR, "results_hard_ours.json")]),
    ("S3  naive NLI FULL — standard (2600)",
     [PY, "-u", DETECT, "--test-set", STD, "--skip-llm",
      "--out", os.path.join(_DIR, "results_standard_naive.json")]),
    ("S4  naive NLI FULL — hard (1400)",
     [PY, "-u", DETECT, "--test-set", HARD, "--skip-llm",
      "--out", os.path.join(_DIR, "results_hard_naive.json")]),
    ("S5  LLM judge sample 800 — standard",
     [PY, "-u", DETECT, "--test-set", STD, "--skip-naive", "--sample", "800",
      "--out", os.path.join(_DIR, "results_standard_judge800.json")]),
    ("S6  LLM judge sample 600 — hard",
     [PY, "-u", DETECT, "--test-set", HARD, "--skip-naive", "--sample", "600",
      "--out", os.path.join(_DIR, "results_hard_judge600.json")]),
    ("S7  HHEM + LettuceDetect FULL — standard",
     [PY, "-u", EXTERN, "--test-set", STD, "--tools", "hhem,lettuce",
      "--out", os.path.join(_DIR, "results_external_standard.json")]),
    ("S8  HHEM + LettuceDetect FULL — hard",
     [PY, "-u", EXTERN, "--test-set", HARD, "--tools", "hhem,lettuce",
      "--out", os.path.join(_DIR, "results_external_hard.json")]),
    ("S9  SummaC sample 250 — standard",
     [PY, "-u", EXTERN, "--test-set", STD, "--tools", "summac", "--sample", "250",
      "--out", os.path.join(_DIR, "results_external_standard_summac.json")]),
    ("S10 SummaC sample 250 — hard",
     [PY, "-u", EXTERN, "--test-set", HARD, "--tools", "summac", "--sample", "250",
      "--out", os.path.join(_DIR, "results_external_hard_summac.json")]),
]


def main():
    t_all = time.time()
    for name, cmd in STAGES:
        print(f"\n{'█'*70}\nSTAGE {name}  (started {time.strftime('%H:%M:%S')})\n{'█'*70}",
              flush=True)
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, check=False).returncode
        except Exception as e:
            print(f"STAGE {name} CRASHED: {e}", flush=True)
            continue
        status = "DONE" if rc == 0 else f"FAILED rc={rc}"
        print(f"STAGE {name} {status} in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"\nALL STAGES FINISHED in {(time.time()-t_all)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
