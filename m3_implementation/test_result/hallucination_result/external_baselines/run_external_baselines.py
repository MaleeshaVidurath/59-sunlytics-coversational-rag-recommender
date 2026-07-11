# m3_implementation/test_result/hallucination_result/external_baselines/run_external_baselines.py
#
# Off-the-shelf baseline experiment — runs UNMODIFIED, third-party, citable
# hallucination detectors on the same 238-case labeled test set used for the
# main evaluation. Complements (does not replace) the re-implemented
# baselines in ../run_detector_eval.py.
#
# WHY: "we compare against the actual released tools" is a stronger claim
# than "we re-implemented the methods". These tools consume free-text
# contexts, so each case's structured evidence is serialized to plain text
# ("Item 1: London dress, colour Black, price £11.08. ...") — the tools
# themselves are used exactly as published.
#
# TOOLS (each optional — skipped with a warning if not installed):
#   hhem      Vectara HHEM-2.1-open — the model behind the public LLM
#             hallucination leaderboard. transformers, trust_remote_code.
#             Consistency score 0..1; < 0.5 → hallucinated.
#   summac    SummaC-Conv (Laban et al., TACL 2022), official package,
#             released weights. Score < 0.5 → hallucinated.
#   lettuce   LettuceDetect (2025) — RAG hallucination detector trained on
#             RAGTruth (ModernBERT). Any predicted hallucination span →
#             hallucinated.
#
# Thresholds are the tools' conventional defaults (0.5); raw scores are
# stored in the results JSON so threshold sensitivity can be analysed later.
#
# Run:  python test_result/hallucination_result/external_baselines/run_external_baselines.py
#       [--tools hhem,summac,lettuce] [--limit N]

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

_DIR     = os.path.dirname(os.path.abspath(__file__))
TEST_SET = os.path.join(_DIR, "..", "original_eval_238", "labeled_test_set.jsonl")
RESULTS  = os.path.join(_DIR, "results_external_baselines.json")

from test_result.hallucination_result.run_detector_eval import (
    compute_metrics, recall_by_corruption,
)


# ── Evidence serialization (structured bundle → plain text context) ─────────

def _items_of(evidence: dict) -> list[dict]:
    action = evidence.get("action", "")
    if action == "catalog_search":
        return evidence.get("items", []) or []
    if action == "item_compare":
        return [x for x in (evidence.get("item_a"), evidence.get("item_b")) if x]
    article = evidence.get("article")
    return [article] if article else []


def serialize_evidence(evidence: dict) -> str:
    """Flattens the evidence bundle into a plain-text context the external
    tools can consume. Field order and wording are fixed for determinism."""
    lines = []
    for i, it in enumerate(_items_of(evidence), 1):
        parts = [f"Item {i}: {it.get('name', 'unknown')}"]
        for label, key in (("type", "type"), ("colour", "colour"),
                           ("price", "price"), ("pattern", "pattern"),
                           ("section", "section"),
                           ("category", "index_group")):
            if it.get(key):
                parts.append(f"{label} {it[key]}")
        lines.append(", ".join(parts) + ".")
        if it.get("material_description"):
            lines.append(f"Item {i} description: {it['material_description']}")
    for key in ("extracted_facts", "comparison_facts"):
        for k, v in (evidence.get(key) or {}).items():
            lines.append(f"{k}: {v}.")
    return "\n".join(lines)


# ── Tool runners — each returns (preds: list[bool], scores: list[float|None]) ──

def run_hhem(cases, contexts):
    from transformers import AutoModelForSequenceClassification
    print("[hhem] loading vectara/hallucination_evaluation_model ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", trust_remote_code=True
    )
    preds, scores = [], []
    t0 = time.time()
    BATCH = 8
    for start in range(0, len(cases), BATCH):
        batch = [(contexts[i], cases[i]["response_text"])
                 for i in range(start, min(start + BATCH, len(cases)))]
        out = model.predict(batch)   # consistency scores, 1.0 = fully grounded
        for s in out.tolist():
            scores.append(float(s))
            preds.append(s < 0.5)    # low consistency → hallucinated
        if (start + BATCH) % 40 < BATCH:
            print(f"  [hhem] {min(start+BATCH, len(cases))}/{len(cases)} "
                  f"({time.time()-t0:.0f}s)")
    return preds, scores


def run_summac(cases, contexts):
    from summac.model_summac import SummaCConv
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[summac] loading SummaC-Conv (vitc) on {device} ...")
    # trained conv weights are not bundled in the pip package — use the copy
    # downloaded from the official repo (github.com/tingofurro/summac)
    start_file = os.path.join(_DIR, "summac_conv_vitc_sent_perc_e.bin")
    model = SummaCConv(models=["vitc"], bins="percentile",
                       granularity="sentence", nli_labels="e",
                       device=device, start_file=start_file, agg="mean")

    # Compatibility shim: summac 0.0.4 passes the legacy kwarg
    # truncation_strategy alongside truncation=True, which modern
    # transformers rejects. Wrap the tokenizer to drop the legacy kwarg —
    # semantics are identical (truncation=True already applies).
    for imager in model.imagers:
        imager.load_nli()
        _orig = imager.tokenizer.batch_encode_plus
        imager.tokenizer.batch_encode_plus = (
            lambda *a, _o=_orig, **kw: _o(
                *a, **{k: v for k, v in kw.items() if k != "truncation_strategy"}
            )
        )
    preds, scores = [], []
    t0 = time.time()
    BATCH = 8
    for start in range(0, len(cases), BATCH):
        docs = contexts[start:start + BATCH]
        resp = [c["response_text"] for c in cases[start:start + BATCH]]
        out = model.score(docs, resp)["scores"]
        for s in out:
            scores.append(float(s))
            preds.append(s < 0.5)    # low consistency → hallucinated
        if (start + BATCH) % 40 < BATCH:
            print(f"  [summac] {min(start+BATCH, len(cases))}/{len(cases)} "
                  f"({time.time()-t0:.0f}s)")
    return preds, scores


def run_lettuce(cases, contexts):
    from lettucedetect.models.inference import HallucinationDetector
    print("[lettuce] loading LettuceDetect (ModernBERT) ...")
    detector = HallucinationDetector(
        method="transformer",
        model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1",
    )
    preds, scores = [], []
    t0 = time.time()
    for i, case in enumerate(cases):
        spans = detector.predict(
            context=[contexts[i]],
            question=case.get("user_message", ""),
            answer=case["response_text"],
            output_format="spans",
        )
        preds.append(len(spans) > 0)   # any hallucinated span → hallucinated
        scores.append(max((s.get("confidence", 1.0) for s in spans), default=0.0))
        if (i + 1) % 40 == 0:
            print(f"  [lettuce] {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    return preds, scores


TOOLS = {
    "hhem":    ("Vectara HHEM-2.1-open", run_hhem),
    "summac":  ("SummaC-Conv (official)", run_summac),
    "lettuce": ("LettuceDetect (RAGTruth-trained)", run_lettuce),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default="hhem,summac,lettuce",
                    help="comma-separated subset of: hhem,summac,lettuce")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test-set", default=TEST_SET,
                    help="path to a labeled test set jsonl")
    ap.add_argument("--out", default=RESULTS, help="path for the results json")
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified random sample of N cases (seeded) — for "
                         "slow tools on large sets")
    args = ap.parse_args()
    wanted = [t.strip() for t in args.tools.split(",") if t.strip()]

    with open(args.test_set, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if args.sample and args.sample < len(cases):
        # stratified by label: keep the clean/hallucinated ratio, seeded
        import random as _random
        rng = _random.Random(123)
        clean = [c for c in cases if c["label"] == "clean"]
        hall  = [c for c in cases if c["label"] == "hallucinated"]
        k_clean = max(1, round(args.sample * len(clean) / len(cases))) if clean else 0
        k_hall  = args.sample - k_clean
        cases = (rng.sample(clean, min(k_clean, len(clean)))
                 + rng.sample(hall, min(k_hall, len(hall))))
        print(f"Stratified sample: {len(cases)} cases (seed 123)")
    y_true = [c["label"] == "hallucinated" for c in cases]
    contexts = [serialize_evidence(c["evidence"]) for c in cases]
    print(f"External baselines on {len(cases)} cases "
          f"({sum(y_true)} hallucinated / {len(y_true)-sum(y_true)} clean)")

    results = {"n_cases": len(cases),
               "context_serialization": "see serialize_evidence()",
               "threshold_note": "conventional defaults (0.5); raw scores stored"}

    for tool in wanted:
        label, runner = TOOLS[tool]
        print(f"\n=== {label} ===")
        try:
            preds, scores = runner(cases, contexts)
        except ImportError as e:
            print(f"  SKIPPED — not installed: {e}")
            results[tool] = {"skipped": str(e)}
            continue
        except Exception as e:
            print(f"  FAILED: {e}")
            results[tool] = {"error": str(e)[:300]}
            continue
        m = compute_metrics(y_true, preds)
        results[tool] = {
            "name": label,
            "metrics": m,
            "recall_by_corruption": recall_by_corruption(cases, preds),
            "scores": scores,
        }
        print(f"  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"BalAcc={m['balanced_accuracy']:.3f} "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 62)
    print("OFF-THE-SHELF DETECTORS vs OUR CHECKER (same 238 cases)")
    print("=" * 62)
    print(f"{'system':<28} {'P':>7} {'R':>7} {'F1':>7} {'BalAcc':>7}")
    print("-" * 62)
    print(f"{'ours (v3, from main eval)':<28} {1.000:>7.3f} {0.951:>7.3f} "
          f"{0.975:>7.3f} {0.976:>7.3f}")
    for tool in wanted:
        r = results.get(tool, {})
        if "metrics" in r:
            m = r["metrics"]
            print(f"{r['name']:<28} {m['precision']:>7.3f} {m['recall']:>7.3f} "
                  f"{m['f1']:>7.3f} {m['balanced_accuracy']:>7.3f}")
    print(f"\nResults written to {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
