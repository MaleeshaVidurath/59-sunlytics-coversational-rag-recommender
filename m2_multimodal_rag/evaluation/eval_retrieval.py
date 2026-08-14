"""
N1 evaluation — Multi-Vector CLIP Ensemble + Domain Fine-Tuning.

Extends finetune_clip/evaluate_retrieval.py (whose helpers are reused) with:
  - a third config: LLM 3-variant query expansion + RRF fusion (the ensemble)
  - MRR in addition to Recall@1/5/10
  - paired bootstrap 95% CIs on the config deltas
  - CSV output for report figures (make_figures.py)

Ground truth: known-item retrieval on the fine-tuning validation split
(same seed/fraction as training — genuinely held-out articles).

Configs:
  A  Stock CLIP + old index          (single vector)      ← pre-novelty baseline
  B  Fine-tuned CLIP + new index     (single vector)      ← fine-tuning only
  C  Fine-tuned + expansion + RRF    (multi-vector)       ← full novelty 1

Usage (from repo root; CPU fine):
    python -m m2_multimodal_rag.evaluation.eval_retrieval                # A vs B, full val set
    python -m m2_multimodal_rag.evaluation.eval_retrieval --ensemble    # + C (needs GROQ_API_KEY)
    python -m m2_multimodal_rag.evaluation.eval_retrieval --n 150       # cap query count

Outputs:
    evaluation/results/retrieval_results.csv     (config-level metrics)
    evaluation/results/retrieval_ranks.csv       (per-query ranks, for CIs/figures)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

from m2_multimodal_rag.finetune_clip.evaluate_retrieval import (
    build_text, load_encoder, encode_texts,
    NEW_INDEX, NEW_MAPPING, OLD_INDEX, OLD_MAPPING,
    VAL_FRACTION, SEED,
)
from shared.config import ARTICLES_PATH

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANK_CAP = 100          # ranks beyond this count as "not found" (reciprocal rank 0)
RRF_K = 60              # standard RRF constant (matches faiss_index.search_multi)
BOOTSTRAP_N = 1000


# ────────────────────────────────────────────────────────────────────
# Rank computation
# ────────────────────────────────────────────────────────────────────

def ranks_single(index, id_list, query_vectors, true_ids) -> list:
    """Rank (1-based) of the true article per query; None if outside RANK_CAP."""
    _, neighbor_rows = index.search(query_vectors, RANK_CAP)
    out = []
    for row_ids, true_id in zip(neighbor_rows, true_ids):
        retrieved = [id_list[r] for r in row_ids if r != -1]
        out.append(retrieved.index(true_id) + 1 if true_id in retrieved else None)
    return out


def ranks_ensemble(index, id_list, model, tokenizer, texts, true_ids) -> list:
    """
    Full novelty-1 pipeline per query: LLM expansion (Groq) → CLIP-encode all
    variants → per-variant FAISS search → RRF fusion → rank of true article.
    Mirrors faiss_index.FAISSDatabase.search_multi.
    """
    from m2_multimodal_rag.llm_generator import llm_generator
    if not llm_generator.is_available:
        raise SystemExit("GROQ_API_KEY not configured — cannot run the ensemble config.")

    out = []
    for qi, (text, true_id) in enumerate(zip(texts, true_ids)):
        variants = llm_generator.expand_query(text)
        vecs = encode_texts(model, tokenizer, variants)
        rrf: dict = {}
        for vec in vecs:
            _, rows = index.search(vec[None, :], RANK_CAP)
            for rank, r in enumerate(rows[0]):
                if r == -1:
                    continue
                aid = id_list[r]
                rrf[aid] = rrf.get(aid, 0.0) + 1.0 / (RRF_K + rank + 1)
        fused = sorted(rrf, key=rrf.get, reverse=True)
        out.append(fused.index(true_id) + 1 if true_id in fused[:RANK_CAP] else None)
        if (qi + 1) % 10 == 0:
            print(f"    ensemble query {qi + 1}/{len(texts)}")
    return out


# ────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────

def metrics_from_ranks(ranks: list) -> dict:
    n = len(ranks)
    rec = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 5, 10)}
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    return {"R@1": rec[1], "R@5": rec[5], "R@10": rec[10], "MRR": mrr, "n": n}


def bootstrap_delta_ci(ranks_a: list, ranks_b: list, k: int = 10,
                       n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> tuple:
    """Paired bootstrap 95% CI for Recall@k(B) − Recall@k(A)."""
    rng = np.random.default_rng(seed)
    hits_a = np.array([1 if r is not None and r <= k else 0 for r in ranks_a])
    hits_b = np.array([1 if r is not None and r <= k else 0 for r in ranks_b])
    n = len(hits_a)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(hits_b[idx].mean() - hits_a[idx].mean())
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", action="store_true",
                    help="also run config C (LLM expansion + RRF; needs GROQ_API_KEY)")
    ap.add_argument("--n", type=int, default=None,
                    help="cap number of validation queries (default: all; ensemble default 150)")
    args = ap.parse_args()

    # ── Validation split (identical logic to finetune_clip/evaluate_retrieval) ──
    articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
    articles["article_id"] = articles["article_id"].str.zfill(10)
    new_mapping = pd.read_csv(NEW_MAPPING, dtype={"article_id": str})
    new_ids = new_mapping["article_id"].str.zfill(10).tolist()
    articles = articles[articles["article_id"].isin(set(new_ids))].reset_index(drop=True)
    articles = articles.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = articles.iloc[:int(len(articles) * VAL_FRACTION)].reset_index(drop=True)

    old_mapping = pd.read_csv(OLD_MAPPING, dtype={"article_id": str})
    old_ids = old_mapping["article_id"].str.zfill(10).tolist()
    val_df = val_df[val_df["article_id"].isin(set(old_ids))].reset_index(drop=True)

    n_queries = args.n or len(val_df)
    val_df = val_df.iloc[:n_queries].reset_index(drop=True)
    texts = [build_text(row) for _, row in val_df.iterrows()]
    true_ids = val_df["article_id"].tolist()
    print(f"Queries: {len(texts)} held-out validation articles\n")

    all_ranks: dict = {}

    # Config A — stock CLIP + old index
    print("=== A: Stock CLIP + old index (single vector) ===")
    model, tok = load_encoder(finetuned=False)
    qv = encode_texts(model, tok, texts)
    index_old = faiss.read_index(str(OLD_INDEX))
    all_ranks["A_stock_single"] = ranks_single(index_old, old_ids, qv, true_ids)
    del model

    # Config B — fine-tuned CLIP + new index
    print("\n=== B: Fine-tuned CLIP + new index (single vector) ===")
    model, tok = load_encoder(finetuned=True)
    qv = encode_texts(model, tok, texts)
    index_new = faiss.read_index(str(NEW_INDEX))
    all_ranks["B_finetuned_single"] = ranks_single(index_new, new_ids, qv, true_ids)

    # Config C — fine-tuned + LLM expansion + RRF (full novelty)
    if args.ensemble:
        n_ens = min(args.n or 150, len(texts))
        print(f"\n=== C: Fine-tuned + LLM expansion + RRF (n={n_ens}) ===")
        all_ranks["C_finetuned_ensemble"] = ranks_ensemble(
            index_new, new_ids, model, tok, texts[:n_ens], true_ids[:n_ens]
        )
    del model

    # ── Report ────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\n" + "=" * 68)
    print(f"{'Configuration':<28}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'n':>6}")
    print("-" * 68)
    for name, ranks in all_ranks.items():
        m = metrics_from_ranks(ranks)
        rows.append({"config": name, **{k: round(v, 4) for k, v in m.items()}})
        print(f"{name:<28}{m['R@1']*100:>7.1f}%{m['R@5']*100:>7.1f}%"
              f"{m['R@10']*100:>7.1f}%{m['MRR']:>8.3f}{m['n']:>6}")
    print("=" * 68)

    # Paired deltas with bootstrap CIs (on the common query subset)
    def _delta(a: str, b: str, label: str):
        n = min(len(all_ranks[a]), len(all_ranks[b]))
        ra, rb = all_ranks[a][:n], all_ranks[b][:n]
        d = (metrics_from_ranks(rb)["R@10"] - metrics_from_ranks(ra)["R@10"])
        lo, hi = bootstrap_delta_ci(ra, rb, k=10)
        sig = "significant" if lo > 0 or hi < 0 else "not significant"
        print(f"Δ R@10 {label}: {d*100:+.1f} pp  (95% CI [{lo*100:+.1f}, {hi*100:+.1f}] — {sig})")
        rows.append({"config": f"delta_{label}", "R@10": round(d, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": n})

    _delta("A_stock_single", "B_finetuned_single", "finetuning(B-A)")
    if "C_finetuned_ensemble" in all_ranks:
        _delta("B_finetuned_single", "C_finetuned_ensemble", "ensemble(C-B)")

    with open(RESULTS_DIR / "retrieval_results.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(RESULTS_DIR / "retrieval_ranks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_idx", "true_id"] + list(all_ranks.keys()))
        for i, tid in enumerate(true_ids):
            w.writerow([i, tid] + [all_ranks[c][i] if i < len(all_ranks[c]) else ""
                                   for c in all_ranks])

    print(f"\nSaved: {RESULTS_DIR / 'retrieval_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'retrieval_ranks.csv'}")


if __name__ == "__main__":
    main()
