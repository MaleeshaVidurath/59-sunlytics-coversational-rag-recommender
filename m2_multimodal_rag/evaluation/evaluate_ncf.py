"""
Two-Tower NCF Evaluation
Compares NCF scoring vs Rule-Based baseline using Hit@5.

Run from project root:
    python Accuracy/evaluate_ncf.py
"""
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.data_loader import data_loader
from m2_multimodal_rag.collaborative_filtering.cf_scorer import cf_scorer

random.seed(42)
np.random.seed(42)

SEP  = "=" * 65
THIN = "-" * 65

# ── Load data ────────────────────────────────────────────────────────────────
articles_df = data_loader.load_articles()
transactions = pd.read_csv(
    Path(__file__).parent.parent / "shared" / "main_data_set" / "sample_transactions.csv",
    dtype={"article_id": str, "customer_id": str}
)

# ── Load NCF model ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Two-Tower NCF — Model Info")
print(SEP)
cf_scorer.load()

embeddings = np.load(
    Path(__file__).parent.parent /
    "m2_multimodal_rag/collaborative_filtering/models/item_embeddings.npy"
)
print(f"  Architecture     : Two-Tower NCF (He et al. 2017 — extended)")
print(f"  Training data    : 185,037 H&M purchase transactions")
print(f"  Items embedded   : {embeddings.shape[0]:,}")
print(f"  Embedding dim    : {embeddings.shape[1]}-D per item")
print(f"  Loss function    : BPR (Bayesian Personalised Ranking)")
print(f"  Content features : colour, type, department, index group,")
print(f"                     garment group, graphical appearance (6 total)")
print(f"  Cold-start       : YES — new items scored via content features")

# ── Build test set: 20 users, last purchase as ground truth ──────────────────
print(f"\n{SEP}")
print("  Hit@5 Evaluation  |  NCF vs Rule-Based Baseline")
print(f"  Metric: ground truth item appears in top-5 recommendations")
print(SEP)

customers   = transactions["customer_id"].unique().tolist()
test_users  = random.sample(customers, min(20, len(customers)))

rule_hits, ncf_hits = [], []

print(f"\n  {'User':<12} {'Rule-Based':>12} {'NCF':>6} {'Result'}")
print(f"  {THIN}")

for cid in test_users:
    user_tx = transactions[transactions["customer_id"] == cid].sort_values("t_dat")
    if len(user_tx) < 2:
        continue

    ground_truth = str(user_tx.iloc[-1]["article_id"]).zfill(10)
    train_tx     = user_tx.iloc[:-1].copy()

    art_lookup = articles_df[["article_id", "colour_group_name", "product_type_name"]].copy()
    art_lookup["article_id"] = art_lookup["article_id"].astype(str)
    train_tx["article_id"]   = train_tx["article_id"].astype(str)

    top_colours = (
        train_tx.merge(art_lookup[["article_id", "colour_group_name"]],
                       on="article_id", how="left")
        ["colour_group_name"].value_counts().head(3).index.tolist()
    )
    top_types = (
        train_tx.merge(art_lookup[["article_id", "product_type_name"]],
                       on="article_id", how="left")
        ["product_type_name"].value_counts().head(3).index.tolist()
    )
    purchase_hints = {"top_colours": top_colours, "top_product_types": top_types}

    # 50 candidates: ground truth + 49 random negatives
    negatives  = [
        str(aid).zfill(10)
        for aid in random.sample(articles_df["article_id"].tolist(), 60)
        if str(aid).zfill(10) != ground_truth
    ][:49]
    candidates = [ground_truth] + negatives
    random.shuffle(candidates)

    # Rule-based: flat +0.08 for colour match, +0.08 for type match
    def rule_score(aid):
        row = articles_df[articles_df["article_id"] == int(aid)]
        if row.empty:
            return 0.0
        m = row.iloc[0]
        score = 0.0
        if str(m.get("colour_group_name", "")) in top_colours:
            score += 0.08
        if str(m.get("product_type_name", "")) in top_types:
            score += 0.08
        return score

    rule_ranked = sorted(candidates, key=rule_score, reverse=True)[:5]
    ncf_ranked  = sorted(
        candidates,
        key=lambda aid: cf_scorer.score(aid, purchase_hints, articles_df),
        reverse=True
    )[:5]

    r_hit = 1 if ground_truth in rule_ranked else 0
    n_hit = 1 if ground_truth in ncf_ranked  else 0
    rule_hits.append(r_hit)
    ncf_hits.append(n_hit)

    result = "BOTH" if r_hit and n_hit else ("NCF only" if n_hit else ("Rule only" if r_hit else "neither"))
    print(f"  {cid[:12]:<12} {'HIT' if r_hit else 'miss':>12} {'HIT' if n_hit else 'miss':>6}   {result}")

# ── Summary ──────────────────────────────────────────────────────────────────
rule_rate = float(np.mean(rule_hits))
ncf_rate  = float(np.mean(ncf_hits))
gain      = ncf_rate - rule_rate

print(f"\n{SEP}")
print("  SUMMARY")
print(SEP)
print(f"  Rule-Based baseline  Hit@5 : {rule_rate:.0%}  ({sum(rule_hits)}/{len(rule_hits)} users)")
print(f"  Two-Tower NCF        Hit@5 : {ncf_rate:.0%}  ({sum(ncf_hits)}/{len(ncf_hits)} users)")
print(f"  Improvement                : {gain:+.0%}  ({ncf_rate/rule_rate:.1f}x improvement)")
print(SEP)
print("  NCF learns personalised item affinity from purchase history.")
print("  Rule-based only matches colour/type — no learning, no personalisation.")
print(f"{SEP}\n")
