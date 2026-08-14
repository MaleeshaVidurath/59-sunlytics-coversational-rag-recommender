"""
N2 evaluation — Two-Tower NCF with Content Features.

Upgrades the preliminary evaluate_ncf.py (n=20 users, Hit@5 only, 49 negatives):
  - all sampled users (~250), leave-last-out per user (chronological)
  - 100 random negatives per case (standard NCF protocol, He et al. 2017)
  - Hit@5, Hit@10, NDCG@10
  - three configs: popularity-only, rule-based (Phase-2 fallback), NCF
  - cold-start slice: held-out items with zero purchases in the training data
  - paired bootstrap 95% CI on NCF − rule-based Hit@10

Usage (from repo root):
    python -m m2_multimodal_rag.evaluation.eval_cf
    python -m m2_multimodal_rag.evaluation.eval_cf --users 100 --negatives 100

Outputs: evaluation/results/cf_results.csv + per-case cf_cases.csv
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd

from shared.data_loader import data_loader
from m2_multimodal_rag.collaborative_filtering.cf_scorer import cf_scorer

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "collaborative_filtering" / "models"
TX_PATH = Path(__file__).resolve().parent.parent.parent / "shared" / "main_data_set" / "sample_transactions.csv"


def ndcg_at_k(ranked: list, ground_truth: str, k: int) -> float:
    if ground_truth in ranked[:k]:
        return 1.0 / np.log2(ranked.index(ground_truth) + 2)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=None, help="cap number of users (default all)")
    ap.add_argument("--negatives", type=int, default=100)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    articles_df = data_loader.load_articles()
    tx = pd.read_csv(TX_PATH, dtype={"article_id": str, "customer_id": str})
    tx["article_id"] = tx["article_id"].str.zfill(10)
    cf_scorer.load()
    if not cf_scorer._loaded:
        raise SystemExit("NCF artifacts not found in collaborative_filtering/models/ — cannot evaluate.")

    # Fast metadata lookup (the preliminary script's per-call DataFrame filter
    # was the main slowness — build a dict once instead)
    meta = {}
    for _, row in articles_df.iterrows():
        meta[str(row["article_id"]).zfill(10)] = {
            "colour_group_name": str(row.get("colour_group_name", "")),
            "product_type_name": str(row.get("product_type_name", "")),
            "price": row.get("price"),
        }
    all_ids = list(meta.keys())

    # Popularity baseline from the trained artifacts
    pop_ids = np.load(MODELS_DIR / "article_ids.npy", allow_pickle=True)
    pop_vals = np.load(MODELS_DIR / "popularity.npy")
    popularity = {str(a).zfill(10): float(p) for a, p in zip(pop_ids, pop_vals)}

    # ── Build test cases: leave-last-out per user ─────────────────────
    users = tx["customer_id"].unique().tolist()
    random.shuffle(users)
    if args.users:
        users = users[: args.users]

    # Global training purchase counts (for the cold-start slice): every
    # transaction except each user's held-out last purchase.
    held_out = {}
    for cid, group in tx.groupby("customer_id"):
        g = group.sort_values("t_dat")
        if len(g) >= 2:
            held_out[cid] = str(g.iloc[-1]["article_id"])
    train_counts = tx["article_id"].value_counts().to_dict()
    for cid, aid in held_out.items():
        train_counts[aid] = train_counts.get(aid, 0) - 1

    configs = ["popularity", "rule_based", "ncf"]
    cases = []

    for ui, cid in enumerate(users):
        if cid not in held_out:
            continue
        g = tx[tx["customer_id"] == cid].sort_values("t_dat")
        ground_truth = held_out[cid]
        train_tx = g.iloc[:-1]

        # purchase_hints exactly as the live Phase-2 fallback consumes them
        colours = [meta[a]["colour_group_name"] for a in train_tx["article_id"] if a in meta]
        types = [meta[a]["product_type_name"] for a in train_tx["article_id"] if a in meta]
        top_colours = pd.Series(colours).value_counts().head(5).index.tolist() if colours else []
        top_types = pd.Series(types).value_counts().head(5).index.tolist() if types else []
        purchase_hints = {"top_colours": top_colours, "top_product_types": top_types}

        negatives = []
        while len(negatives) < args.negatives:
            aid = random.choice(all_ids)
            if aid != ground_truth and aid not in negatives:
                negatives.append(aid)
        candidates = [ground_truth] + negatives
        random.shuffle(candidates)

        def rule_score(aid: str) -> float:
            m = meta.get(aid)
            if not m:
                return 0.0
            s = 0.0
            if m["colour_group_name"] in top_colours:
                s += 0.12 * (1 - top_colours.index(m["colour_group_name"]) / max(len(top_colours), 1))
            if m["product_type_name"] in top_types:
                s += 0.08
            return s

        scores = {
            "popularity": {aid: popularity.get(aid, 0.0) for aid in candidates},
            "rule_based": {aid: rule_score(aid) for aid in candidates},
            "ncf": {aid: cf_scorer.score(aid, purchase_hints, articles_df) for aid in candidates},
        }

        case = {"user": cid[:12], "ground_truth": ground_truth,
                "cold_start": train_counts.get(ground_truth, 0) <= 0}
        for cfg in configs:
            ranked = sorted(candidates, key=lambda a: scores[cfg][a], reverse=True)
            case[f"{cfg}_hit5"] = int(ground_truth in ranked[:5])
            case[f"{cfg}_hit10"] = int(ground_truth in ranked[:10])
            case[f"{cfg}_ndcg10"] = round(ndcg_at_k(ranked, ground_truth, 10), 4)
        cases.append(case)
        if (ui + 1) % 25 == 0:
            print(f"  evaluated {ui + 1}/{len(users)} users")

    if not cases:
        raise SystemExit("No evaluable users (need >=2 purchases each).")

    # ── Report ────────────────────────────────────────────────────────
    df = pd.DataFrame(cases)
    cold = df[df["cold_start"]]
    print("\n" + "=" * 70)
    print(f"{'Config':<14}{'Hit@5':>9}{'Hit@10':>9}{'NDCG@10':>10}"
          f"{'Cold Hit@10':>13}{'n':>6}{'n_cold':>8}")
    print("-" * 70)
    rows = []
    for cfg in configs:
        r = {"config": cfg,
             "hit5": df[f"{cfg}_hit5"].mean(),
             "hit10": df[f"{cfg}_hit10"].mean(),
             "ndcg10": df[f"{cfg}_ndcg10"].mean(),
             "cold_hit10": cold[f"{cfg}_hit10"].mean() if len(cold) else float("nan"),
             "n": len(df), "n_cold": len(cold)}
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print(f"{cfg:<14}{r['hit5']*100:>8.1f}%{r['hit10']*100:>8.1f}%{r['ndcg10']:>10.3f}"
              f"{r['cold_hit10']*100:>12.1f}%{r['n']:>6}{r['n_cold']:>8}")
    print("=" * 70)

    # Paired bootstrap CI: NCF − rule-based on Hit@10
    rng = np.random.default_rng(SEED)
    a = df["rule_based_hit10"].to_numpy()
    b = df["ncf_hit10"].to_numpy()
    deltas = [(b[idx].mean() - a[idx].mean())
              for idx in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    sig = "significant" if lo > 0 or hi < 0 else "not significant"
    print(f"Δ Hit@10 NCF − rule-based: {(b.mean()-a.mean())*100:+.1f} pp "
          f"(95% CI [{lo*100:+.1f}, {hi*100:+.1f}] — {sig})")
    rows.append({"config": "delta_ncf_vs_rule", "hit10": round(float(b.mean()-a.mean()), 4),
                 "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "n": len(a)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cf_results.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    df.to_csv(RESULTS_DIR / "cf_cases.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'cf_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'cf_cases.csv'}")


if __name__ == "__main__":
    main()
