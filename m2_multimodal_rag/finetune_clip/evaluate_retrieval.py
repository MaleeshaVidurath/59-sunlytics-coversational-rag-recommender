"""
Before/after retrieval evaluation: stock CLIP + old FAISS index vs
fine-tuned CLIP + rebuilt FAISS index.

Metric: text-to-image Recall@K on the SAME held-out validation articles used
during fine-tuning (recreated with the same seed). For each validation
article, its metadata text is encoded into a query vector and searched
against the index; a "hit" means the article's own image vector appears in
the top-K results. Higher = the encoder places matching text and images
closer together.

Runs locally (CPU is fine — text encoding only, no images needed).

    python -m m2_multimodal_rag.finetune_clip.evaluate_retrieval
"""

import numpy as np
import pandas as pd
import torch
import faiss
import open_clip
from pathlib import Path

from shared.config import ARTICLES_PATH

VECTOR_DB_DIR = Path(__file__).resolve().parent.parent / "vector_db"
NEW_INDEX = VECTOR_DB_DIR / "m2_clip_faiss.bin"
NEW_MAPPING = VECTOR_DB_DIR / "m2_faiss_mapping.csv"
OLD_INDEX = VECTOR_DB_DIR / "m2_clip_faiss_old.bin"
OLD_MAPPING = VECTOR_DB_DIR / "m2_faiss_mapping_old.csv"

FINETUNED_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "clip_finetuned_hm_best.pt"

VAL_FRACTION = 0.05   # must match finetune_clip_kaggle.py
SEED = 42             # must match finetune_clip_kaggle.py
TOP_K = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_text(row) -> str:
    """Identical to finetune_clip_kaggle.py — the text used as retrieval query."""
    parts = [
        row.get("prod_name", ""),
        row.get("colour_group_name", ""),
        row.get("product_type_name", ""),
        row.get("department_name", ""),
        row.get("graphical_appearance_name", ""),
        str(row.get("detail_desc", ""))[:200],
    ]
    parts = [str(p).strip() for p in parts if p and str(p).lower() != "nan"]
    return " ".join(parts)


def load_encoder(finetuned: bool):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    if finetuned:
        state_dict = torch.load(FINETUNED_WEIGHTS, map_location="cpu")
        model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        tokens = tokenizer(texts[i:i + BATCH_SIZE]).to(DEVICE)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy().astype("float32"))
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"    encoded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} texts")
    return np.vstack(vectors)


def recall_at_k(index, id_list: list, query_vectors: np.ndarray,
                true_ids: list, ks=(1, 5, 10)) -> dict:
    _, neighbor_rows = index.search(query_vectors, max(ks))
    hits = {k: 0 for k in ks}
    for row_ids, true_id in zip(neighbor_rows, true_ids):
        retrieved = [id_list[r] for r in row_ids if r != -1]
        for k in ks:
            if true_id in retrieved[:k]:
                hits[k] += 1
    n = len(true_ids)
    return {k: hits[k] / n for k in ks}


def main():
    print(f"Device: {DEVICE}")
    articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
    articles["article_id"] = articles["article_id"].str.zfill(10)
    print(f"Total articles: {len(articles):,}")

    # Recreate the exact validation split used on Kaggle: filter to articles
    # that have images (= the new index's contents, which preserved CSV order),
    # then the same seeded shuffle and 5% head.
    new_mapping = pd.read_csv(NEW_MAPPING, dtype={"article_id": str})
    indexed_ids = set(new_mapping["article_id"].str.zfill(10))
    articles = articles[articles["article_id"].isin(indexed_ids)].reset_index(drop=True)

    articles = articles.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = int(len(articles) * VAL_FRACTION)
    val_df = articles.iloc[:n_val].reset_index(drop=True)
    print(f"Validation articles (held out during fine-tuning): {len(val_df):,}")

    # Old index may cover a slightly different article set — evaluate only on
    # articles present in BOTH indexes so the comparison is apples-to-apples.
    old_mapping = pd.read_csv(OLD_MAPPING, dtype={"article_id": str})
    old_ids = set(old_mapping["article_id"].str.zfill(10))
    val_df = val_df[val_df["article_id"].isin(old_ids)].reset_index(drop=True)
    print(f"Validation articles present in both indexes: {len(val_df):,}\n")

    texts = [build_text(row) for _, row in val_df.iterrows()]
    true_ids = val_df["article_id"].tolist()

    results = {}
    configs = [
        ("Stock CLIP + old index", False, OLD_INDEX,
         old_mapping["article_id"].str.zfill(10).tolist()),
        ("Fine-tuned CLIP + new index", True, NEW_INDEX,
         new_mapping["article_id"].str.zfill(10).tolist()),
    ]

    for name, finetuned, index_path, id_list in configs:
        print(f"=== {name} ===")
        print("  Loading encoder...")
        model, tokenizer = load_encoder(finetuned)
        print("  Encoding validation texts...")
        query_vectors = encode_texts(model, tokenizer, texts)
        index = faiss.read_index(str(index_path))
        print(f"  Index: {index.ntotal:,} vectors. Searching...")
        results[name] = recall_at_k(index, id_list, query_vectors, true_ids)
        del model
        print()

    print("=" * 56)
    print(f"{'Configuration':<32}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    print("-" * 56)
    for name, recalls in results.items():
        print(f"{name:<32}"
              f"{recalls[1]*100:>7.1f}%{recalls[5]*100:>7.1f}%{recalls[10]*100:>7.1f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
