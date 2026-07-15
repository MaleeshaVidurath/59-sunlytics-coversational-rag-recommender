# m3_implementation/test_result/hallucination_result/expanded_eval/build_expanded_summary.py
#
# Consolidates every expanded-evaluation results file (local + Colab) into
# one machine-readable summary with provenance. Re-run any time new results
# land; missing files are recorded as "pending".
#
# Run:  python test_result/hallucination_result/expanded_eval/build_expanded_summary.py

import json
import os
from datetime import date

_DIR = os.path.dirname(os.path.abspath(__file__))

# (set, system) → (file, block key, provenance note)
SOURCES = {
    ("standard", "ours"):      ("results_standard_ours.json", "ours",
                                "local CPU, FULL 2600 (canonical; Colab GPU replication within CI)"),
    ("standard", "naive_nli"): ("colab_results_standard_ours_naive.json", "naive_nli",
                                "Colab T4, FULL 2600"),
    ("standard", "llm_judge"): ("results_standard_judge800.json", "llm_judge",
                                "local, stratified sample 800 (seed 123, Groq rate limits)"),
    ("standard", "lettuce"):   ("colab_results_external_standard.json", "lettuce",
                                "Colab T4, FULL 2600"),
    ("standard", "hhem"):      ("results_external_standard_hhem.json", "hhem",
                                "local CPU, FULL 2600 (HHEM incompatible with Colab transformers)"),
    ("standard", "summac"):    ("results_external_standard_summac.json", "summac",
                                "local CPU, stratified sample 250 (seed 123)"),
    ("hard", "ours"):          ("results_hard_ours.json", "ours",
                                "local CPU, FULL 1400 (Colab replication bit-identical)"),
    ("hard", "naive_nli"):     ("colab_results_hard_ours_naive.json", "naive_nli",
                                "Colab T4, FULL 1400"),
    ("hard", "llm_judge"):     ("results_hard_judge600.json", "llm_judge",
                                "local, sample 600 — CAVEAT: 392/600 calls unanswered "
                                "(overnight Groq throttling); selection-bias risk, re-run advised"),
    ("hard", "lettuce"):       ("colab_results_external_hard.json", "lettuce",
                                "Colab T4, FULL 1400"),
    ("hard", "hhem"):          ("results_external_hard_hhem.json", "hhem",
                                "local CPU, FULL 1400"),
    ("hard", "summac"):        ("results_external_hard_summac.json", "summac",
                                "local CPU, stratified sample 250 (seed 123)"),
}


def main():
    summary = {
        "meta": {
            "generated": str(date.today()),
            "suite": "expanded evaluation — 4,000-row suite drawn from an "
                     "8,181-row pool (all 526 clean kept, corrupted sampled "
                     "evenly per type, seed 42)",
            "standard_set": "labeled_test_set_expanded.jsonl — 2,600 rows "
                            "(526 clean + 2,074 corrupted)",
            "hard_set": "hard_set/labeled_hard_set.jsonl — 1,400 corrupted-only "
                        "rows (paraphrase_colour/price, fabricated_attribute)",
            "bases": "526 clean bases: 36 original scripted + 166 new scripted "
                     "(60 conversations) + 416 real MongoDB user chats, "
                     "deduplicated by response text",
            "checker_version": "v3, NLI_CONTRADICTION_THRESHOLD=0.70 (.env)",
            "loop_experiment": "NOT re-run (user decision): mechanism results "
                               "reused from the original 205-case experiment "
                               "(P(correction|detection)=0.969); composed "
                               "end-to-end estimate in EXPANDED_RESULTS.md",
            "notes": [
                "hard set has no clean rows: precision/balanced-accuracy are "
                "not meaningful there — report recall per corruption family",
                "fabricated_attribute rows are unsupported claims, not "
                "contradictions — the checker ignores them BY DESIGN",
                "clean labels presumed pending human audit "
                "(clean_audit_expanded.txt, 526 rows)",
            ],
        },
        "results": {},
    }

    for (setname, system), (fname, key, prov) in sorted(SOURCES.items()):
        path = os.path.join(_DIR, fname)
        entry = {"provenance": prov, "file": fname}
        blk = {}
        if os.path.exists(path) and os.path.getsize(path) > 10:
            with open(path, encoding="utf-8") as f:
                blk = json.load(f).get(key, {})
        if "metrics" in blk:
            entry["metrics"] = blk["metrics"]
            entry["recall_by_corruption"] = blk.get("recall_by_corruption", {})
            if "n_answered" in blk:
                entry["n_answered"] = blk["n_answered"]
                entry["n_unanswered"] = blk["n_unanswered"]
        else:
            entry["status"] = "pending" if not blk else f"failed: {json.dumps(blk)[:120]}"
        summary["results"].setdefault(setname, {})[system] = entry

    out = os.path.join(_DIR, "results_expanded_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Written {os.path.basename(out)}")
    for setname in ("standard", "hard"):
        print(f"\n{setname}:")
        for system, e in summary["results"][setname].items():
            if "metrics" in e:
                m = e["metrics"]
                print(f"  {system:<10} P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f} BalAcc={m['balanced_accuracy']:.3f}")
            else:
                print(f"  {system:<10} {e.get('status')}")


if __name__ == "__main__":
    main()
