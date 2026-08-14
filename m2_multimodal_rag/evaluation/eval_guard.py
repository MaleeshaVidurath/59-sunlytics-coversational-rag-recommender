"""
N4 evaluation — Multi-Stage Hallucination Guard (detection ablation).

Pipeline modelled on m3_implementation/test_result/hallucination_result/
(capture → corrupt → labeled set → per-config detection metrics), adapted
to M2's product-explanation guard with a visual corruption class that only
Layer 3 (CLIPScore + ViLT) can catch.

Stages:
  capture  — generate raw (guard-off) explanations for N sampled articles
             that have cached images                          → guard_cases.jsonl
  corrupt  — derive labeled clean + hallucinated variants:
               colour_swap   (wrong colour claim)             — text-detectable
               type_swap     (wrong garment type claim)       — text-detectable
               fabric_claim  (invented fabric/care claim)     — text-detectable
               visual_claim  (pattern/appearance contradicting
                              the image, metadata-silent)     — visual-only
                                                              → guard_labeled.jsonl
  run      — detection-only pass per config (l1 / l12 / full):
             a case is "flagged" if any enabled layer fails it.
             Metrics: P/R/F1 (positive = hallucinated), per-corruption
             recall, first-catching-layer attribution, per-case latency.

Usage (from repo root; capture+run need GROQ_API_KEY, run loads all models):
    python -m m2_multimodal_rag.evaluation.eval_guard --stage capture --n 40
    python -m m2_multimodal_rag.evaluation.eval_guard --stage corrupt
    python -m m2_multimodal_rag.evaluation.eval_guard --stage run

Outputs: evaluation/results/guard_results.csv (+ guard_case_results.csv)
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import pandas as pd

SEED = 42
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
CASES_PATH = EVAL_DIR / "guard_cases.jsonl"
LABELED_PATH = EVAL_DIR / "guard_labeled.jsonl"

COLOURS = ["Black", "White", "Red", "Dark Blue", "Grey", "Pink", "Beige", "Green"]
TYPES = ["Dress", "Sweater", "T-shirt", "Trousers", "Skirt", "Jacket", "Blouse", "Shorts"]
FABRICS = ["pure silk", "genuine leather", "100% cashmere", "heavy denim", "linen"]
VISUAL_CLAIMS = [
    "featuring a bold striped pattern",
    "decorated with a large floral print",
    "with prominent polka dots across the front",
    "showing a bright checked pattern",
]

META_FIELDS = ["article_id", "prod_name", "product_type_name", "colour_group_name",
               "graphical_appearance_name", "department_name", "detail_desc"]


# ────────────────────────────────────────────────────────────────────
# Stage 1 — capture raw explanations
# ────────────────────────────────────────────────────────────────────

def stage_capture(n: int):
    from shared.data_loader import data_loader
    from m2_multimodal_rag.llm_generator import llm_generator
    if not llm_generator.is_available:
        raise SystemExit("GROQ_API_KEY required for capture.")

    articles = data_loader.load_articles()
    cache_dir = data_loader.image_cache_dir
    # Prefer articles whose image is already cached (Layer 3 needs images)
    cached = []
    for sub in cache_dir.iterdir() if cache_dir.exists() else []:
        if sub.is_dir():
            cached += [p.stem for p in sub.glob("*.jpg")]
    random.seed(SEED)
    random.shuffle(cached)
    print(f"Articles with cached images: {len(cached)}")

    cases = []
    for aid in cached:
        row = articles[articles["article_id"] == int(aid)]
        if row.empty:
            continue
        meta = row.iloc[0].to_dict()
        expl = llm_generator.generate(aid, meta, product_knowledge="",
                                      image_path=str(cache_dir / aid[:3] / f"{aid}.jpg"))
        if not expl:
            continue
        cases.append({**{f: str(meta.get(f, "")) for f in META_FIELDS},
                      "article_id": aid, "explanation": expl})
        print(f"  [{len(cases)}/{n}] {aid}: {expl[:60]}...")
        if len(cases) >= n:
            break

    with open(CASES_PATH, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"Saved {len(cases)} cases → {CASES_PATH}")


# ────────────────────────────────────────────────────────────────────
# Stage 2 — corrupt into a labeled set
# ────────────────────────────────────────────────────────────────────

def _swap_word(text: str, current: str, pool: list) -> str | None:
    """Replaces `current` (case-insensitive) with a different pool word."""
    low = text.lower()
    cur = current.lower().strip()
    if not cur or cur not in low:
        return None
    wrong = random.choice([p for p in pool if p.lower() != cur])
    i = low.index(cur)
    return text[:i] + wrong + text[i + len(cur):]


def stage_corrupt():
    random.seed(SEED)
    cases = [json.loads(l) for l in open(CASES_PATH, encoding="utf-8")]
    labeled = []
    for c in cases:
        labeled.append({**c, "label": "clean", "corruption": "none"})

        col = _swap_word(c["explanation"], c["colour_group_name"], COLOURS)
        if col:
            labeled.append({**c, "explanation": col, "label": "hallucinated",
                            "corruption": "colour_swap"})

        typ = _swap_word(c["explanation"], c["product_type_name"], TYPES)
        if typ:
            labeled.append({**c, "explanation": typ, "label": "hallucinated",
                            "corruption": "type_swap"})

        fab = c["explanation"].rstrip(". ") + f". It is crafted from {random.choice(FABRICS)}."
        labeled.append({**c, "explanation": fab, "label": "hallucinated",
                        "corruption": "fabric_claim"})

        # Visual-only corruption: valid for solid items — the claim contradicts
        # the image but no metadata field names the pattern in the sentence.
        if str(c.get("graphical_appearance_name", "")).lower() == "solid":
            vis = c["explanation"].rstrip(". ") + f", {random.choice(VISUAL_CLAIMS)}."
            labeled.append({**c, "explanation": vis, "label": "hallucinated",
                            "corruption": "visual_claim"})

    with open(LABELED_PATH, "w", encoding="utf-8") as f:
        for c in labeled:
            f.write(json.dumps(c) + "\n")
    counts = pd.Series([c["corruption"] for c in labeled]).value_counts().to_dict()
    print(f"Saved {len(labeled)} labeled cases → {LABELED_PATH}\n  {counts}")


# ────────────────────────────────────────────────────────────────────
# Stage 3 — detection-only evaluation per config
# ────────────────────────────────────────────────────────────────────

def stage_run():
    from shared.data_loader import data_loader
    from m2_multimodal_rag.llm_generator import llm_generator
    from m2_multimodal_rag.hallucination_guard.layer_1_knowledge_self_reflection import knowledge_reflector
    from m2_multimodal_rag.hallucination_guard.layer_2_cove_verification import cove_verifier
    from m2_multimodal_rag.hallucination_guard.layer_3_vlm_visual_verification import blip_verifier
    from m2_multimodal_rag.hallucination_guard.clip_faithfulness_scorer import clip_faithfulness_scorer

    cases = [json.loads(l) for l in open(LABELED_PATH, encoding="utf-8")]
    print(f"Labeled cases: {len(cases)}")

    # Incremental resume: only VALIDLY scored cases are skipped. A case that
    # was previously api_degraded (hit a real API failure mid-case) is NOT
    # considered done — it stays eligible for retry on the next invocation,
    # otherwise a quota outage would permanently exclude that case forever
    # instead of just delaying its scoring.
    progress_path = EVAL_DIR / "guard_run_progress.jsonl"
    done = {}
    n_pending_retry = 0
    if progress_path.exists():
        for l in open(progress_path, encoding="utf-8"):
            r = json.loads(l)
            key = (r["article_id"], r["corruption"], r["label"])
            if r.get("api_degraded"):
                n_pending_retry += 1
                continue
            done[key] = r
        print(f"Resuming: {len(done)} cases validly scored, "
              f"{n_pending_retry} degraded cases eligible for retry.")
    progress_f = open(progress_path, "a", encoding="utf-8")

    # Product knowledge depends only on the article — cache it per article
    # (~4 case variants share it), cutting L1 generation calls 4x.
    pk_cache: dict = {}

    # Every guard layer has a "pass on failure" fallback (self_evaluate →
    # True + a non-empty message; CoVe.verify → True, [], 1.0) so a quota
    # outage is INDISTINGUISHABLE from a genuine pass by looking at output
    # alone — this is what silently contaminated the first two runs. The
    # only reliable signal is llm_generator.fail_count, incremented on every
    # real API failure (None/exception/empty response).
    consecutive_degraded = 0
    n_degraded = 0
    for i, c in enumerate(cases):
        key = (c["article_id"], c["corruption"], c["label"])
        if key in done:
            continue
        meta = {f: c[f] for f in META_FIELDS}
        expl = c["explanation"]
        aid = c["article_id"]
        img = data_loader.image_cache_dir / aid[:3] / f"{aid}.jpg"

        fails_before = llm_generator.fail_count
        t0 = time.perf_counter()
        if aid not in pk_cache:
            pk_cache[aid] = knowledge_reflector.generate_product_knowledge(meta)
        pk = pk_cache[aid]
        l1_pass, _ = knowledge_reflector.self_evaluate(expl, meta, product_knowledge=pk)
        t1 = time.perf_counter()
        l2_pass, _, _ = cove_verifier.verify(expl, meta)
        t2 = time.perf_counter()
        l3_pass = True
        if img.exists():
            _, clip_pass, _ = clip_faithfulness_scorer.score(str(img), expl)
            vilt_pass, _ = blip_verifier.verify(str(img), expl)
            l3_pass = clip_pass and vilt_pass
        t3 = time.perf_counter()

        api_degraded = llm_generator.fail_count > fails_before
        if api_degraded:
            n_degraded += 1
            consecutive_degraded += 1
        else:
            consecutive_degraded = 0

        row = {
            "article_id": aid, "label": c["label"], "corruption": c["corruption"],
            "l1_flag": int(not l1_pass), "l2_flag": int(not l2_pass),
            "l3_flag": int(not l3_pass), "api_degraded": int(api_degraded),
            "l1_ms": round((t1 - t0) * 1000), "l2_ms": round((t2 - t1) * 1000),
            "l3_ms": round((t3 - t2) * 1000),
        }
        done[key] = row
        progress_f.write(json.dumps(row) + "\n")
        progress_f.flush()
        print(f"  [{i+1}/{len(cases)}] {c['corruption']:<13} "
              f"L1={'F' if not l1_pass else '.'} L2={'F' if not l2_pass else '.'} "
              f"L3={'F' if not l3_pass else '.'}"
              f"{'  [API-DEGRADED — quota]' if api_degraded else ''}")

        if consecutive_degraded >= 3:
            progress_f.close()
            raise SystemExit(
                f"\nABORT at case {i+1}: 3 consecutive cases hit real API "
                f"failures (llm_generator.fail_count confirms quota "
                f"exhaustion, not inferred). {len(done)} cases checkpointed "
                f"({n_degraded} degraded, excluded automatically). Rerun "
                f"this stage after the daily quota resets to continue.")

    progress_f.close()
    per_case = list(done.values())
    n_valid_degraded = sum(1 for r in per_case if r.get("api_degraded"))
    if n_valid_degraded:
        print(f"\n{n_valid_degraded}/{len(per_case)} cases were API-degraded "
              f"(quota failures mid-case) — excluded from metrics below.")
    per_case = [r for r in per_case if not r.get("api_degraded")]
    df = pd.DataFrame(per_case)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "guard_case_results.csv", index=False)

    configs = {"l1": ["l1_flag"], "l12": ["l1_flag", "l2_flag"],
               "full": ["l1_flag", "l2_flag", "l3_flag"]}
    rows = []
    print("\n" + "=" * 74)
    print(f"{'Config':<8}{'Precision':>11}{'Recall':>9}{'F1':>8}"
          f"{'FalseAlarms':>13}{'VisualRecall':>14}")
    print("-" * 74)
    for name, flags in configs.items():
        pred = df[flags].max(axis=1)
        truth = (df["label"] == "hallucinated").astype(int)
        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        vis = df[df["corruption"] == "visual_claim"]
        vis_recall = float(vis[flags].max(axis=1).mean()) if len(vis) else float("nan")
        rows.append({"config": name, "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(f1, 3), "false_alarms": fp,
                     "visual_recall": round(vis_recall, 3), "n": len(df)})
        print(f"{name:<8}{p:>11.3f}{r:>9.3f}{f1:>8.3f}{fp:>13}{vis_recall:>14.3f}")
    print("=" * 74)

    # First-catching-layer attribution among hallucinated cases
    hall = df[df["label"] == "hallucinated"]
    first = {"l1": int((hall["l1_flag"] == 1).sum()),
             "l2_only": int(((hall["l1_flag"] == 0) & (hall["l2_flag"] == 1)).sum()),
             "l3_only": int(((hall["l1_flag"] == 0) & (hall["l2_flag"] == 0)
                             & (hall["l3_flag"] == 1)).sum()),
             "missed": int(((hall[["l1_flag", "l2_flag", "l3_flag"]].max(axis=1)) == 0).sum())}
    print(f"First caught by: {first}")
    print(f"Mean added latency: L1 {df['l1_ms'].mean():.0f}ms | "
          f"L2 {df['l2_ms'].mean():.0f}ms | L3 {df['l3_ms'].mean():.0f}ms")

    # Per-corruption recall (full config)
    print("\nPer-corruption recall (full guard):")
    for corr, grp in hall.groupby("corruption"):
        rec = grp[["l1_flag", "l2_flag", "l3_flag"]].max(axis=1).mean()
        print(f"  {corr:<14} {rec:.3f}  (n={len(grp)})")

    with open(RESULTS_DIR / "guard_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {RESULTS_DIR / 'guard_results.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["capture", "corrupt", "run"], required=True)
    ap.add_argument("--n", type=int, default=40, help="articles to capture")
    args = ap.parse_args()
    random.seed(SEED)
    if args.stage == "capture":
        stage_capture(args.n)
    elif args.stage == "corrupt":
        stage_corrupt()
    else:
        stage_run()


if __name__ == "__main__":
    main()
