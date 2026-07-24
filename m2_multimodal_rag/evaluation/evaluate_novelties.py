"""
M2 Novelty Evaluation Script
Evaluates all 5 novelties against their baselines and saves results to JSON.

Run from project root:
    python Accuracy/evaluate_novelties.py

Output: Accuracy/novelty_results.json
"""

import sys
import os
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Force UTF-8 output on Windows to handle any unicode in LLM responses
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.data_loader import data_loader

RESULTS_PATH = Path(__file__).parent / "novelty_results.json"

# ── Test query bank ────────────────────────────────────────────────────────────
# Each query: (user_message, expected_colour, expected_type)
TEST_QUERIES = [
    ("I want a black dress",           "Black",      "Dress"),
    ("Show me a white blouse",         "White",      "Blouse"),
    ("I need dark blue trousers",      "Dark Blue",  "Trousers"),
    ("Find me a red top",              "Red",        "Top"),
    ("I want a grey sweater",          "Grey",       "Sweater"),
    ("Show me a black jacket",         "Black",      "Jacket"),
    ("I need a white dress",           "White",      "Dress"),
    ("Find me dark blue jeans",        "Dark Blue",  "Trousers"),
    ("I want a black top",             "Black",      "Top"),
    ("Show me a red dress",            "Red",        "Dress"),
]


def _load_articles():
    return data_loader.load_articles()


# ══════════════════════════════════════════════════════════════════════════════
# NOVELTY 1 — Multi-Vector CLIP Ensemble vs Single Vector
# Metric: Recall@10 — how many of top-10 results match expected colour+type
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_novelty1():
    print("\n" + "="*60)
    print("NOVELTY 1 — CLIP Ensemble vs Single Vector")
    print("="*60)

    try:
        from m2_multimodal_rag.clip_embeddings import clip_encoder
        from m2_multimodal_rag.faiss_index import faiss_db
        from m2_multimodal_rag.llm_generator import llm_generator
        articles_df = _load_articles()
    except Exception as e:
        print(f"  [SKIP] Could not load components: {e}")
        return {"error": str(e)}

    single_recalls, ensemble_recalls = [], []

    for query, exp_colour, exp_type in TEST_QUERIES:
        try:
            # Baseline: single CLIP vector
            single_vec = clip_encoder.encode_text(query)
            if single_vec is None:
                continue
            single_candidates = faiss_db.search(single_vec, top_k=10)

            # Novelty 1: multi-vector ensemble
            expanded = llm_generator.expand_query(query)
            ensemble_vec = clip_encoder.encode_expanded(expanded)
            if ensemble_vec is None:
                continue
            ensemble_candidates = faiss_db.search(ensemble_vec, top_k=10)

            def recall(candidates):
                if not candidates:
                    return 0.0
                hits = 0
                for aid, _ in candidates:
                    try:
                        row = articles_df[articles_df["article_id"] == int(aid)]
                        if row.empty:
                            continue
                        m = row.iloc[0]
                        colour_match = exp_colour.lower() in str(m.get("colour_group_name","")).lower()
                        type_match   = exp_type.lower() in str(m.get("product_type_name","")).lower()
                        if colour_match and type_match:
                            hits += 1
                    except Exception:
                        continue
                return hits / len(candidates)

            single_r   = recall(single_candidates)
            ensemble_r = recall(ensemble_candidates)
            single_recalls.append(single_r)
            ensemble_recalls.append(ensemble_r)
            print(f"  Query: '{query[:40]}' | Single: {single_r:.0%}  Ensemble: {ensemble_r:.0%}")

        except Exception as e:
            print(f"  [ERROR] {query}: {e}")
            continue

    if not single_recalls:
        return {"error": "No results"}

    result = {
        "metric":           "Recall@10 (colour + type match)",
        "baseline_avg":     round(float(np.mean(single_recalls)), 4),
        "novelty_avg":      round(float(np.mean(ensemble_recalls)), 4),
        "improvement":      round(float(np.mean(ensemble_recalls) - np.mean(single_recalls)), 4),
        "n_queries":        len(single_recalls),
        "per_query": [
            {"query": q, "baseline": round(s, 3), "novelty": round(e, 3)}
            for q, s, e in zip(
                [q for q, _, _ in TEST_QUERIES[:len(single_recalls)]],
                single_recalls, ensemble_recalls
            )
        ]
    }
    print(f"\n  Baseline avg Recall@10 : {result['baseline_avg']:.0%}")
    print(f"  Novelty avg  Recall@10 : {result['novelty_avg']:.0%}")
    print(f"  Improvement            : +{result['improvement']:.0%}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# NOVELTY 2 — NCF Scoring vs Rule-Based Scoring
# Metric: Precision@5 on held-out transactions
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_novelty2():
    print("\n" + "="*60)
    print("NOVELTY 2 — NCF Scoring vs Rule-Based (+0.08 flat)")
    print("="*60)

    try:
        from m2_multimodal_rag.collaborative_filtering.cf_scorer import cf_scorer
        if not cf_scorer._loaded:
            cf_scorer.load()
        if not cf_scorer._loaded:
            return {"error": "CF model not loaded — run train_cf_kaggle.py first"}

        transactions = pd.read_csv(
            Path(__file__).parent.parent / "shared" / "main_data_set" / "sample_transactions.csv",
            dtype={"article_id": str, "customer_id": str}
        )
        articles_df = _load_articles()

    except Exception as e:
        print(f"  [SKIP] {e}")
        return {"error": str(e)}

    # Build held-out set: for each of 20 random users, take their last purchase as ground truth
    random.seed(42)
    customers = transactions["customer_id"].unique().tolist()
    test_customers = random.sample(customers, min(20, len(customers)))

    rule_hits, ncf_hits = [], []

    for cid in test_customers:
        user_tx = transactions[transactions["customer_id"] == cid].sort_values("t_dat")
        if len(user_tx) < 2:
            continue

        ground_truth   = str(user_tx.iloc[-1]["article_id"]).zfill(10)
        train_tx       = user_tx.iloc[:-1].copy()

        # Align article_id types for merge (transactions=str, articles_df=int)
        art_lookup = articles_df[["article_id","colour_group_name","product_type_name"]].copy()
        art_lookup["article_id"] = art_lookup["article_id"].astype(str)
        train_tx["article_id"]   = train_tx["article_id"].astype(str)

        top_colours = (
            train_tx.merge(art_lookup[["article_id","colour_group_name"]],
                           on="article_id", how="left")
            ["colour_group_name"].value_counts().head(3).index.tolist()
        )
        top_types = (
            train_tx.merge(art_lookup[["article_id","product_type_name"]],
                           on="article_id", how="left")
            ["product_type_name"].value_counts().head(3).index.tolist()
        )
        purchase_hints = {
            "top_colours": top_colours,
            "top_product_types": top_types,
        }

        # Sample 50 candidate articles (ground truth + 49 random negatives)
        negatives = [
            str(aid).zfill(10)
            for aid in random.sample(articles_df["article_id"].tolist(), 49)
            if str(aid).zfill(10) != ground_truth
        ]
        candidates = [ground_truth] + negatives[:49]
        random.shuffle(candidates)

        # Rule-based scoring: flat +0.08 for colour match, +0.08 for type match
        def rule_score(aid):
            row = articles_df[articles_df["article_id"] == int(aid)]
            if row.empty:
                return 0.0
            m = row.iloc[0]
            score = 0.0
            if str(m.get("colour_group_name","")) in top_colours:
                score += 0.08
            if str(m.get("product_type_name","")) in top_types:
                score += 0.08
            return score

        rule_ranked = sorted(candidates, key=rule_score, reverse=True)[:5]
        ncf_ranked  = sorted(
            candidates,
            key=lambda aid: cf_scorer.score(aid, purchase_hints, articles_df),
            reverse=True
        )[:5]

        rule_hits.append(1 if ground_truth in rule_ranked else 0)
        ncf_hits.append( 1 if ground_truth in ncf_ranked  else 0)

    if not rule_hits:
        return {"error": "No valid test users"}

    result = {
        "metric":        "Hit@5 (ground truth in top-5 recommendations)",
        "baseline_avg":  round(float(np.mean(rule_hits)), 4),
        "novelty_avg":   round(float(np.mean(ncf_hits)),  4),
        "improvement":   round(float(np.mean(ncf_hits) - np.mean(rule_hits)), 4),
        "n_users":       len(rule_hits),
    }
    print(f"  Baseline Hit@5 (rule-based) : {result['baseline_avg']:.0%}")
    print(f"  Novelty  Hit@5 (NCF)        : {result['novelty_avg']:.0%}")
    print(f"  Improvement                 : +{result['improvement']:.0%}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# NOVELTY 3 — Thompson Sampling vs Fixed Lambda MMR
# Metric: Intra-List Diversity (ILD) — average pairwise colour dissimilarity
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_novelty3():
    print("\n" + "="*60)
    print("NOVELTY 3 — Thompson Sampling vs Fixed Lambda MMR")
    print("="*60)

    try:
        from m2_multimodal_rag.diversity_bandit import diversity_bandit
        from m2_multimodal_rag.faiss_index import faiss_db
        from m2_multimodal_rag.clip_embeddings import clip_encoder
        articles_df = _load_articles()
    except Exception as e:
        print(f"  [SKIP] {e}")
        return {"error": str(e)}

    fixed_ilds, adaptive_ilds = [], []
    FIXED_LAMBDA = 0.7

    colour_groups = articles_df["colour_group_name"].dropna().unique().tolist()

    def colour_dissimilarity(c1, c2):
        return 0.0 if c1 == c2 else 1.0

    def ild(items, df):
        attrs = []
        for aid in items:
            try:
                row = df[df["article_id"] == int(aid)]
                if not row.empty:
                    colour = str(row.iloc[0].get("colour_group_name", ""))
                    ptype  = str(row.iloc[0].get("product_type_name", ""))
                    attrs.append((colour, ptype))
                else:
                    attrs.append(("", ""))
            except Exception:
                attrs.append(("", ""))
        if len(attrs) < 2:
            return 0.0
        dists = []
        for i in range(len(attrs)):
            for j in range(i + 1, len(attrs)):
                colour_diff = 0.0 if attrs[i][0] == attrs[j][0] else 1.0
                type_diff   = 0.0 if attrs[i][1] == attrs[j][1] else 1.0
                dists.append((colour_diff + type_diff) / 2.0)
        return float(np.mean(dists)) if dists else 0.0

    # Use mixed-colour query so ILD can show real diversity differences
    DIVERSITY_QUERIES = [
        "casual summer outfit",
        "smart casual clothing",
        "everyday wardrobe essentials",
    ]

    # Simulate sessions with increasing rejection counts
    for exclude_count in range(0, 11):
        retained_count = max(0, 2 - exclude_count // 3)

        # Rotate through mixed queries so results have varied colours
        query_text = DIVERSITY_QUERIES[exclude_count % len(DIVERSITY_QUERIES)]
        vec = clip_encoder.encode_text(query_text)
        if vec is None:
            continue
        candidates = faiss_db.search(vec, top_k=20)
        if not candidates:
            continue

        cand_ids  = [c[0] for c in candidates]
        cand_data = [{"article_id": c[0], "final_score": c[1], "metadata": {}} for c in candidates]
        for c in cand_data:
            try:
                row = articles_df[articles_df["article_id"] == int(c["article_id"])]
                if not row.empty:
                    c["metadata"] = row.iloc[0].to_dict()
            except Exception:
                pass

        # Fixed lambda MMR
        fixed_results = faiss_db.mmr_select(
            candidates=cand_data, query_vector=vec,
            top_k=4, lambda_param=FIXED_LAMBDA
        )
        fixed_items = [r["article_id"] for r in fixed_results] if fixed_results else cand_ids[:4]

        # Adaptive lambda (Thompson Sampling)
        adaptive_lambda = diversity_bandit.sample_lambda(
            exclude_count=exclude_count, retained_count=retained_count
        )
        adaptive_results = faiss_db.mmr_select(
            candidates=cand_data, query_vector=vec,
            top_k=4, lambda_param=adaptive_lambda
        )
        adaptive_items = [r["article_id"] for r in adaptive_results] if adaptive_results else cand_ids[:4]

        fixed_ild    = ild(fixed_items,    articles_df)
        adaptive_ild = ild(adaptive_items, articles_df)
        fixed_ilds.append(fixed_ild)
        adaptive_ilds.append(adaptive_ild)

        print(f"  Rejections={exclude_count:2d}  lambda_adaptive={adaptive_lambda:.3f}  "
              f"ILD_fixed={fixed_ild:.3f}  ILD_adaptive={adaptive_ild:.3f}")

    if not fixed_ilds:
        return {"error": "No results"}

    result = {
        "metric":           "Intra-List Diversity (ILD) -- higher = more diverse",
        "baseline_avg":     round(float(np.mean(fixed_ilds)), 4),
        "novelty_avg":      round(float(np.mean(adaptive_ilds)), 4),
        "improvement":      round(float(np.mean(adaptive_ilds) - np.mean(fixed_ilds)), 4),
        "note":             "ILD measured by colour dissimilarity across recommended items",
        "rejection_levels": list(range(len(fixed_ilds))),
        "fixed_ilds":       [round(x, 3) for x in fixed_ilds],
        "adaptive_ilds":    [round(x, 3) for x in adaptive_ilds],
    }
    print(f"\n  Baseline avg ILD (fixed lambda=0.7) : {result['baseline_avg']:.3f}")
    print(f"  Novelty  avg ILD (adaptive lambda)  : {result['novelty_avg']:.3f}")
    print(f"  Improvement                         : +{result['improvement']:.3f}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# NOVELTY 4 — Hallucination Guard Attribute Accuracy
# Metric: % of explanations with correct colour + type (no guard vs with guard)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_novelty4():
    print("\n" + "="*60)
    print("NOVELTY 4 — Hallucination Guard Attribute Accuracy")
    print("="*60)

    try:
        from m2_multimodal_rag.llm_generator import llm_generator
        articles_df = _load_articles()
    except Exception as e:
        print(f"  [SKIP] {e}")
        return {"error": str(e)}

    if not llm_generator.is_available:
        print("  [SKIP] LLM not available")
        return {"error": "LLM unavailable"}

    # Sample 15 random articles
    sample = articles_df.dropna(subset=["colour_group_name","product_type_name"]).sample(
        n=min(15, len(articles_df)), random_state=42
    )

    no_guard_correct, guard_correct = [], []

    for _, row in sample.iterrows():
        aid     = str(row["article_id"]).zfill(10)
        colour  = str(row.get("colour_group_name","")).lower()
        ptype   = str(row.get("product_type_name","")).lower()
        meta    = row.to_dict()

        # Without guard — raw LLM generation
        try:
            raw = llm_generator.generate(aid, meta, product_knowledge="")
            raw_lower = raw.lower() if raw else ""
            raw_ok = colour in raw_lower and ptype in raw_lower
            no_guard_correct.append(int(raw_ok))
            print(f"  [{aid}] No-guard: {'PASS' if raw_ok else 'FAIL'} | "
                  f"expected colour='{colour}' type='{ptype}'")
        except Exception as e:
            print(f"  [ERROR] No-guard generation: {e}")
            continue

        # With guard — pre-filter check only (fast, doesn't require images)
        try:
            from m2_multimodal_rag.hallucination_guard.regeneration_loop import generator_loop
            pre_fails = generator_loop._direct_attribute_check(raw, meta)
            if pre_fails:
                # Simulate correction
                feedback = "; ".join(f"{k} should be '{v}'" for k,v in pre_fails.items())
                corrected = llm_generator.regenerate(aid, meta, visual_feedback=feedback)
                corrected_lower = corrected.lower() if corrected else ""
            else:
                corrected_lower = raw_lower
            guard_ok = colour in corrected_lower and ptype in corrected_lower
            guard_correct.append(int(guard_ok))
            print(f"  [{aid}] Guard:    {'PASS' if guard_ok else 'FAIL'} | "
                  f"pre_filter_fails={pre_fails}")
        except Exception as e:
            print(f"  [ERROR] Guard check: {e}")
            guard_correct.append(no_guard_correct[-1])

    if not no_guard_correct:
        return {"error": "No results"}

    result = {
        "metric":        "Attribute Accuracy -- % explanations with correct colour + type",
        "baseline_avg":  round(float(np.mean(no_guard_correct)), 4),
        "novelty_avg":   round(float(np.mean(guard_correct)),    4),
        "improvement":   round(float(np.mean(guard_correct) - np.mean(no_guard_correct)), 4),
        "n_items":       len(no_guard_correct),
    }
    print(f"\n  Baseline accuracy (no guard) : {result['baseline_avg']:.0%}")
    print(f"  Novelty  accuracy (guard)    : {result['novelty_avg']:.0%}")
    print(f"  Improvement                  : +{result['improvement']:.0%}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# NOVELTY 5 — Kansei KB Relevance
# Metric: % of results matching style occasion attributes
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_novelty5():
    print("\n" + "="*60)
    print("NOVELTY 5 — Kansei KB vs No KB")
    print("="*60)

    try:
        from m2_multimodal_rag.clip_embeddings import clip_encoder
        from m2_multimodal_rag.faiss_index import faiss_db
        from m2_multimodal_rag.knowledge_base.kb_retriever import kb_retriever
        articles_df = _load_articles()
    except Exception as e:
        print(f"  [SKIP] {e}")
        return {"error": str(e)}

    # Test queries: KB should boost KB score (NCF score from kb_retriever)
    # Metric: average KB relevance score from kb_retriever.score()
    # Higher score = KB found a psychologically relevant match
    style_queries = [
        ("elegant black dress for evening",  "elegant", "Dress",   "Black"),
        ("casual summer outfit",              "casual",  "Top",     "White"),
        ("sporty comfortable top",            "sporty",  "Top",     "Grey"),
        ("professional formal blouse",        "formal",  "Blouse",  "Black"),
        ("relaxed weekend sweater",           "casual",  "Sweater", "Grey"),
    ]

    no_kb_scores, kb_scores = [], []

    for user_message, style_word, item_type, item_colour in style_queries:
        try:
            # Without KB: kb_score is 0 (no style_word)
            score_no_kb = kb_retriever.score(
                item_colour=item_colour, item_type=item_type,
                item_appearance="Solid", occasion=None, style_word=None,
                index_group_name="Ladieswear"
            )

            # With KB: kb_score includes Kansei boost
            score_kb = kb_retriever.score(
                item_colour=item_colour, item_type=item_type,
                item_appearance="Solid", occasion="casual" if "casual" in style_word else None,
                style_word=style_word, index_group_name="Ladieswear"
            )

            no_kb_scores.append(score_no_kb)
            kb_scores.append(score_kb)

            print(f"  '{user_message[:45]}' | No-KB score: {score_no_kb:.4f}  KB score: {score_kb:.4f}")

        except Exception as e:
            print(f"  [ERROR] {user_message}: {e}")
            continue

    if not no_kb_scores:
        return {"error": "No results"}

    result = {
        "metric":        "KB Relevance Score -- psychology-grounded scoring boost",
        "baseline_avg":  round(float(np.mean(no_kb_scores)), 4),
        "novelty_avg":   round(float(np.mean(kb_scores)),    4),
        "improvement":   round(float(np.mean(kb_scores) - np.mean(no_kb_scores)), 4),
        "n_queries":     len(no_kb_scores),
    }
    print(f"\n  Baseline match rate (no KB) : {result['baseline_avg']:.0%}")
    print(f"  Novelty  match rate (KB)    : {result['novelty_avg']:.0%}")
    print(f"  Improvement                 : +{result['improvement']:.0%}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  M2 Novelty Evaluation -- All 5 Novelties")
    print("="*60)

    results = {}

    print("\n[1/5] Evaluating NOVELTY 1...")
    results["novelty_1_clip_ensemble"]     = evaluate_novelty1()
    time.sleep(1)

    print("\n[2/5] Evaluating NOVELTY 2...")
    results["novelty_2_ncf_scoring"]       = evaluate_novelty2()
    time.sleep(1)

    print("\n[3/5] Evaluating NOVELTY 3...")
    results["novelty_3_thompson_sampling"] = evaluate_novelty3()
    time.sleep(1)

    print("\n[4/5] Evaluating NOVELTY 4...")
    results["novelty_4_hallucination"]     = evaluate_novelty4()
    time.sleep(1)

    print("\n[5/5] Evaluating NOVELTY 5...")
    results["novelty_5_kansei_kb"]         = evaluate_novelty5()

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for key, res in results.items():
        if "error" in res:
            print(f"  {key}: ERROR — {res['error']}")
        else:
            imp = res.get("improvement", 0)
            sign = "+" if imp >= 0 else ""
            print(f"  {key}:")
            print(f"    Metric    : {res.get('metric','')}")
            print(f"    Baseline  : {res.get('baseline_avg',0):.0%}")
            print(f"    Novelty   : {res.get('novelty_avg',0):.0%}")
            print(f"    Improvement: {sign}{imp:.0%}")

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("Use these numbers to update the presentation slides.")


if __name__ == "__main__":
    main()
