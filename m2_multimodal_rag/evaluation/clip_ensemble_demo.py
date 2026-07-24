"""
CLIP Ensemble — Terminal Output Demo
Run from project root:
    python Accuracy/clip_ensemble_demo.py
"""
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m2_multimodal_rag.llm_generator import llm_generator
from m2_multimodal_rag.clip_embeddings import clip_encoder
from m2_multimodal_rag.faiss_index import faiss_db
from shared.data_loader import data_loader

DEMO_QUERIES = [
    "elegant black dress for a formal evening party",
    "casual white blouse for summer weekend",
    "comfortable dark blue trousers for travel",
]

articles_df = data_loader.load_articles()
articles_df["article_id"] = pd.to_numeric(articles_df["article_id"], errors="coerce").astype("Int64")

def lookup(aid: str) -> str:
    try:
        row = articles_df[articles_df["article_id"] == int(str(aid).lstrip("0") or "0")]
        if row.empty:
            return "Unknown"
        m = row.iloc[0]
        return f"{m['prod_name']}  |  {m['colour_group_name']}  |  {m['product_type_name']}"
    except Exception:
        return "Unknown"

print("\n" + "=" * 70)
print("  NOVELTY 1 — Multi-Vector CLIP Ensemble  |  Live Demo")
print("=" * 70)

for query in DEMO_QUERIES:
    print(f"\n  Query : '{query}'")
    print("  " + "-" * 67)

    # Step 1: LLM expands query into 3 semantic variants
    expanded = llm_generator.expand_query(query)

    # Step 2: Encode all variants + average into one enriched vector
    ensemble_vec = clip_encoder.encode_expanded(expanded)

    # Step 3: FAISS retrieval — show product names so results are meaningful
    if ensemble_vec is not None:
        results = faiss_db.search(ensemble_vec, top_k=3)
        print("  Retrieved items:")
        for rank, (aid, score) in enumerate(results, 1):
            print(f"    [{rank}] {lookup(aid)}")

print("\n" + "=" * 70 + "\n")
