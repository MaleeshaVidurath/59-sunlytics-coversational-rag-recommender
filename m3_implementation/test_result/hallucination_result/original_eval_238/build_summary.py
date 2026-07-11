# m3_implementation/test_result/hallucination_result/build_summary.py
#
# Consolidates the per-version evaluation outputs into one machine-readable
# summary (results_summary.json). Re-run after any evaluation to refresh.
#
#   v1 = original checker            (results_detector_eval_v1.json — also
#        contains the naive-NLI and LLM-judge baselines, which are
#        checker-version-independent)
#   v2 = two-sided name/price gates  (results_detector_eval_v2.json)
#   v3 = final: similarity-gate bypass + response-level verification
#        (results_detector_eval.json)
#
# Run:  python test_result/hallucination_result/build_summary.py

import json
import os
from datetime import date

_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "v1": "results_detector_eval_v1.json",
    "v2": "results_detector_eval_v2.json",
    "v3": "results_detector_eval.json",
}


def load(name):
    with open(os.path.join(_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    v1 = load(SOURCES["v1"])
    v2 = load(SOURCES["v2"])
    v3 = load(SOURCES["v3"])

    summary = {
        "meta": {
            "generated":     str(date.today()),
            "test_set":      "labeled_test_set.jsonl",
            "n_cases":       v3["n_cases"],
            "n_clean":       33,
            "n_hallucinated": v3["n_cases"] - 33,
            "corruption_seed": 42,
            "corruption_method": "FactCC-style synthetic field corruption "
                                 "(colour_swap, price_change, name_swap, cross_item_swap)",
            "nli_model":       "cross-encoder/nli-deberta-v3-base",
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_judge_model": "llama-3.1-8b-instant (Groq)",
            "positive_class":  "hallucinated",
            "notes": [
                "Clean labels are presumed-correct pending manual audit (clean_audit.txt).",
                "Threshold sweep values > 0.65 are partially an artifact: containment "
                "flags carry a synthetic contradiction score of 1.0, so only the "
                "<= 0.65 region reflects real NLI threshold behaviour.",
                "8 checker-flagged originals from live runs are excluded from the "
                "test set and await human adjudication (flagged_for_review.jsonl).",
            ],
        },
        "checker_versions": {
            "v1": {
                "description": "Original checker: one-sided verbatim containment for "
                               "name/price (pass-only), NLI fallback for all fields",
                "metrics":              v1["ours"]["metrics"],
                "recall_by_corruption": v1["ours"]["recall_by_corruption"],
            },
            "v2": {
                "description": "Two-sided containment gates: wrong name/price value in "
                               "the sentence flags directly; name/price never reach NLI; "
                               "whitespace normalisation; price-regex trailing-dot fix",
                "metrics":              v2["ours"]["metrics"],
                "recall_by_corruption": v2["ours"]["recall_by_corruption"],
            },
            "v3": {
                "description": "Final: exact-value facts bypass the MiniLM similarity "
                               "gate; locked catalog sentences verified per-sentence; "
                               "unlocked facts verified response-level (fixes renamed-item "
                               "misses and the derived-arithmetic false positive)",
                "metrics":              v3["ours"]["metrics"],
                "recall_by_corruption": v3["ours"]["recall_by_corruption"],
                "false_positive_case_ids": v3["ours"]["false_positive_case_ids"],
                "missed_case_ids":         v3["ours"]["missed_case_ids"],
            },
        },
        "baselines": {
            "naive_nli": {
                "description": "SummaC-style: DeBERTa on every (fact, sentence) pair, "
                               "no lock map, no gates; flag if any pair has softmax "
                               "P(contradiction) > 0.5 and > P(entailment)",
                "metrics":              v1["naive_nli"]["metrics"],
                "recall_by_corruption": v1["naive_nli"]["recall_by_corruption"],
            },
            "llm_judge": {
                "description": "RAGAS-style LLM judge: Groq llama-3.1-8b-instant asked "
                               "whether the response contradicts the evidence facts",
                "metrics":              v1["llm_judge"]["metrics"],
                "recall_by_corruption": v1["llm_judge"]["recall_by_corruption"],
            },
        },
        "threshold_sweep_v3": v3["threshold_sweep"],
    }

    out = os.path.join(_DIR, "results_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Written {os.path.basename(out)}")
    print(f"  versions: {list(summary['checker_versions'])}")
    print(f"  baselines: {list(summary['baselines'])}")
    v3m = summary["checker_versions"]["v3"]["metrics"]
    print(f"  headline (v3): P={v3m['precision']} R={v3m['recall']} "
          f"F1={v3m['f1']} BalAcc={v3m['balanced_accuracy']}")


if __name__ == "__main__":
    main()
