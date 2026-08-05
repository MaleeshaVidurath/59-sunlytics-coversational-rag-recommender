import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import your graph and search engine
from build_graph import construct_knowledge_graph
from graph_search import run_catalog_search

def precision_at_k(recommended, relevant, k=5):
    rec_k = [r['article_id'] for r in recommended[:k]]
    true_positive = len(set(rec_k).intersection(set(relevant)))
    return true_positive / k

def ndcg_at_k(recommended, relevant, k=5):
    rec_k = [r['article_id'] for r in recommended[:k]]
    dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in relevant])
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / idcg if idcg > 0 else 0.0

def main():
    print("1. Loading Graph and Test Data...")
    G = construct_knowledge_graph() 
    
    with open('test_dataset.json', 'r') as f:
        test_cases = json.load(f)

    results = {
        "Baseline A (Vibe Only)": {"precision": [], "ndcg": []},
        "Baseline B (History Only)": {"precision": [], "ndcg": []},
        "Model C (Hybrid Scoring)": {"precision": [], "ndcg": []}
    }

    print(f"2. Running Ablation Study on {len(test_cases)} users (This may take a minute)...")
    
    for case in tqdm(test_cases):
        cid = case['customer_id']
        msg = case['user_message']
        ground_truth = case['ground_truth_articles']
        
        # We pass quantity: 5 so the system returns top 5 items for evaluation
        payload = {"quantity": 5}
        
        # Baseline A: Semantic Vibe Only (Turn history off)
        res_vibe = run_catalog_search(G, payload, {}, [], msg, cid, vibe_weight=1.0, history_weight=0.0)
        results["Baseline A (Vibe Only)"]["precision"].append(precision_at_k(res_vibe['data'], ground_truth))
        results["Baseline A (Vibe Only)"]["ndcg"].append(ndcg_at_k(res_vibe['data'], ground_truth))
        
        # Baseline B: Structural History Only (Turn vibe off)
        res_hist = run_catalog_search(G, payload, {}, [], msg, cid, vibe_weight=0.0, history_weight=1.0)
        results["Baseline B (History Only)"]["precision"].append(precision_at_k(res_hist['data'], ground_truth))
        results["Baseline B (History Only)"]["ndcg"].append(ndcg_at_k(res_hist['data'], ground_truth))
        
        # Model C: Your Hybrid Novelty (Both turned on)
        res_hybrid = run_catalog_search(G, payload, {}, [], msg, cid, vibe_weight=5.0, history_weight=3.0)
        results["Model C (Hybrid Scoring)"]["precision"].append(precision_at_k(res_hybrid['data'], ground_truth))
        results["Model C (Hybrid Scoring)"]["ndcg"].append(ndcg_at_k(res_hybrid['data'], ground_truth))

    print("\n3. Calculating Final Metrics...")
    final_data = []
    for model_name, metrics in results.items():
        final_data.append({
            "Model": model_name,
            "Precision@5": np.mean(metrics["precision"]),
            "NDCG@5": np.mean(metrics["ndcg"])
        })

    df = pd.DataFrame(final_data)
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(df.to_markdown(index=False))

    print("\n4. Drawing Graph for Presentation...")
    sns.set_theme(style="whitegrid", context="talk")
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted, palette="viridis")
    plt.title("Ablation Study: Impact of Hybrid Scoring", pad=15)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontsize=11)
        
    plt.tight_layout()
    plt.savefig("hybrid_evaluation_results.png", dpi=300)
    print("✅ Graph successfully saved as 'hybrid_evaluation_results.png'!")

if __name__ == "__main__":
    main()