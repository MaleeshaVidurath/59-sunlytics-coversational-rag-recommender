"""
N3 evaluation — Thompson Sampling Diversity Bandit + MMR.

Fixes the preliminary attempt's broken metric (colour-dissimilarity ILD was
always 0 because hard filters force same-colour pools): diversity is now
measured in CLIP vector space (1 − mean pairwise cosine of selected items).

Simulation (deterministic seed, no LLM/server needed):
  - 20 template queries → FAISS top-30 candidate pool each
  - user rejects the top r items (r = 0..10 cumulative rejections)
  - selection re-runs on the remaining pool with:
      fixed λ ∈ {0.5, 0.7, 0.9}   (baseline: MMR without the bandit)
      adaptive λ ~ Thompson(α,β)  (novelty; 10 samples per point for mean±std)
  - metrics per point: intra-list diversity (ILD), mean relevance (faiss
    score) of the selected items, and the λ actually used

Usage (from repo root):
    python -m m2_multimodal_rag.evaluation.eval_diversity

Outputs: evaluation/results/diversity_results.csv  (one row per config × r)
"""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from m2_multimodal_rag.clip_embeddings import clip_encoder
from m2_multimodal_rag.faiss_index import faiss_db
from m2_multimodal_rag.diversity_bandit import diversity_bandit

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
POOL_K = 30
SELECT_K = 4
REJECTION_LEVELS = list(range(0, 11))
ADAPTIVE_SAMPLES = 10

QUERIES = [
    f"{colour} {item} for {occasion}"
    for colour in ("black", "white", "red", "dark blue", "grey")
    for item, occasion in (
        ("dress", "a party"), ("top", "everyday wear"),
        ("sweater", "winter"), ("trousers", "the office"),
    )
]  # 20 queries


def ild(vectors: list) -> float:
    """Intra-list diversity: 1 − mean pairwise cosine similarity."""
    vecs = [np.asarray(v).ravel() for v in vectors if v is not None]
    if len(vecs) < 2:
        return 0.0
    sims = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            for a, b in combinations(vecs, 2)]
    return 1.0 - float(np.mean(sims))


def evaluate_point(pool: list, r: int, lam: float) -> tuple:
    """Reject top-r of the pool, MMR-select with λ, return (ILD, mean relevance)."""
    remaining = pool[r:]
    if len(remaining) < SELECT_K:
        return None, None
    qv = remaining[0]["_query_vector"]
    picked = faiss_db.mmr_select(
        candidates=[{k: v for k, v in c.items() if not k.startswith("_")}
                    for c in remaining],
        query_vector=qv, top_k=SELECT_K, lambda_param=lam,
    )
    vecs = [faiss_db.get_item_vector(c["article_id"]) for c in picked]
    rel = float(np.mean([c["final_score"] for c in picked]))
    return ild(vecs), rel


def main():
    np.random.seed(SEED)
    if not faiss_db.database_ready:
        raise SystemExit("FAISS index not loaded — cannot evaluate (DUMMY mode).")

    # Build candidate pools
    pools = []
    for q in QUERIES:
        vec = clip_encoder.encode_text(q)
        if vec is None:
            continue
        results = faiss_db.search(vec, top_k=POOL_K)
        pool = [{"article_id": aid, "final_score": float(score), "_query_vector": vec}
                for aid, score in results]
        if len(pool) >= POOL_K:
            pools.append(pool)
    print(f"Candidate pools built: {len(pools)} queries × {POOL_K} items\n")

    rows = []
    configs = [("fixed_0.5", 0.5), ("fixed_0.7", 0.7), ("fixed_0.9", 0.9), ("adaptive", None)]

    for cfg_name, fixed_lam in configs:
        for r in REJECTION_LEVELS:
            ilds, rels, lams = [], [], []
            for pool in pools:
                if fixed_lam is not None:
                    d, rel = evaluate_point(pool, r, fixed_lam)
                    if d is not None:
                        ilds.append(d); rels.append(rel); lams.append(fixed_lam)
                else:
                    for _ in range(ADAPTIVE_SAMPLES):
                        lam = diversity_bandit.sample_lambda(exclude_count=r, retained_count=0)
                        d, rel = evaluate_point(pool, r, lam)
                        if d is not None:
                            ilds.append(d); rels.append(rel); lams.append(lam)
            rows.append({
                "config": cfg_name, "rejections": r,
                "mean_lambda": round(float(np.mean(lams)), 4),
                "ild_mean": round(float(np.mean(ilds)), 4),
                "ild_std": round(float(np.std(ilds)), 4),
                "relevance_mean": round(float(np.mean(rels)), 4),
                "n_points": len(ilds),
            })
        last = rows[-1]
        first = rows[-len(REJECTION_LEVELS)]
        print(f"{cfg_name:<12} ILD r=0: {first['ild_mean']:.4f}  →  r=10: {last['ild_mean']:.4f}"
              f"   (λ r=0: {first['mean_lambda']:.2f} → r=10: {last['mean_lambda']:.2f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "diversity_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")

    # Headline: does the adaptive bandit increase diversity after rejections
    # while matching fixed-λ relevance when there are none?
    adaptive_0 = next(r for r in rows if r["config"] == "adaptive" and r["rejections"] == 0)
    adaptive_10 = next(r for r in rows if r["config"] == "adaptive" and r["rejections"] == 10)
    fixed7_0 = next(r for r in rows if r["config"] == "fixed_0.7" and r["rejections"] == 0)
    fixed7_10 = next(r for r in rows if r["config"] == "fixed_0.7" and r["rejections"] == 10)
    print("\nHeadline:")
    print(f"  fixed λ=0.7 : ILD {fixed7_0['ild_mean']:.4f} → {fixed7_10['ild_mean']:.4f} "
          f"(no adaptation to rejections)")
    print(f"  adaptive    : ILD {adaptive_0['ild_mean']:.4f} → {adaptive_10['ild_mean']:.4f} "
          f"(λ {adaptive_0['mean_lambda']:.2f} → {adaptive_10['mean_lambda']:.2f})")


if __name__ == "__main__":
    main()
