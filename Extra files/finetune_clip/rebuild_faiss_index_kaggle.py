"""
Rebuilds the M2 FAISS index using the FINE-TUNED CLIP checkpoint.

The existing m2_clip_faiss.bin was built with stock laion2b weights. After
fine-tuning, query vectors (new weights) and item vectors (old weights) live
in different embedding spaces — retrieval quality degrades instead of
improving. This script re-encodes every article image with the fine-tuned
model so both sides match again.

RUN THIS ON KAGGLE (same setup as finetune_clip_kaggle.py).

── Kaggle notebook setup ──────────────────────────────────────────────────
1. New Notebook → Settings → Accelerator → GPU (T4 x2 or P100)
2. Add Data (3 inputs):
     - Competition: "h-and-m-personalized-fashion-recommendations"
     - Your dataset: "hm-sample-articles" (sample_articles.csv)
     - Your dataset: "hm-clip-finetuned"  ← upload clip_finetuned_hm_best.pt
       from your PC as a new private Kaggle Dataset with this name first
       (kaggle.com → Create → New Dataset → drag the .pt file in)
3. Install dependencies:
     !pip install -q open_clip_torch faiss-cpu
4. Paste this whole file into a cell and run (~15-25 min on GPU).
─────────────────────────────────────────────────────────────────────────

Output (saved to /kaggle/working/):
    m2_clip_faiss.bin      — rebuilt FAISS index (fine-tuned vectors)
    m2_faiss_mapping.csv   — row → article_id mapping

Download both and place them in m2_multimodal_rag/vector_db/
(replacing the old files — keep backups if you want a before/after demo).
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
import faiss

# ── Paths (Kaggle) ────────────────────────────────────────────────────────────
IMAGES_DIR      = "/kaggle/input/competitions/h-and-m-personalized-fashion-recommendations/images"
ARTICLES_CSV    = "/kaggle/input/datasets/maleeshavidurath/hm-sample-articles/sample_articles.csv"
CHECKPOINT_PATH = "/kaggle/input/datasets/maleeshavidurath/hm-clip-finetuned/clip_finetuned_hm_best.pt"
INDEX_OUT       = "/kaggle/working/m2_clip_faiss.bin"
MAPPING_OUT     = "/kaggle/working/m2_faiss_mapping.csv"

BATCH_SIZE  = 256
NUM_WORKERS = 2
EMBED_DIM   = 512
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HMImageDataset(Dataset):
    """Yields (preprocessed image tensor, row index) for every article with an image."""
    def __init__(self, article_ids: list, preprocess):
        self.article_ids = article_ids
        self.preprocess = preprocess

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        aid = self.article_ids[idx]
        image_path = os.path.join(IMAGES_DIR, aid[:3], f"{aid}.jpg")
        with Image.open(image_path) as img:
            image = self.preprocess(img.convert("RGB"))
        return image, idx


def filter_existing_images(article_ids: list) -> list:
    """Keeps only articles whose image file exists (folder-level listing — fast on FUSE)."""
    print("Indexing available image files...")
    existing = set()
    for prefix in os.listdir(IMAGES_DIR):
        prefix_dir = os.path.join(IMAGES_DIR, prefix)
        if os.path.isdir(prefix_dir):
            existing.update(os.listdir(prefix_dir))

    kept = [aid for aid in article_ids if f"{aid}.jpg" in existing]
    print(f"Skipping {len(article_ids) - len(kept):,} articles with missing image files.")
    return kept


def load_finetuned_model():
    print(f"Device: {DEVICE}")
    print("Loading CLIP ViT-B-32 architecture...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    print(f"Loading fine-tuned weights from {CHECKPOINT_PATH}...")
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess


def main():
    print(f"Loading articles from {ARTICLES_CSV}...")
    articles = pd.read_csv(ARTICLES_CSV, dtype={"article_id": str})
    article_ids = articles["article_id"].astype(str).str.zfill(10).tolist()
    print(f"Total articles: {len(article_ids):,}")

    article_ids = filter_existing_images(article_ids)
    print(f"Articles to encode: {len(article_ids):,}")

    model, preprocess = load_finetuned_model()

    dataset = HMImageDataset(article_ids, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)

    # Encode all images → normalized 512-D vectors (row i == article_ids[i])
    all_vectors = np.zeros((len(article_ids), EMBED_DIM), dtype="float32")
    start = time.time()

    with torch.no_grad():
        for batch_num, (images, indices) in enumerate(loader, 1):
            images = images.to(DEVICE)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            all_vectors[indices.numpy()] = features.cpu().numpy().astype("float32")

            if batch_num % 20 == 0 or batch_num == len(loader):
                done = batch_num * BATCH_SIZE
                rate = done / (time.time() - start)
                print(f"  Encoded batch {batch_num}/{len(loader)} "
                      f"({min(done, len(article_ids)):,} images, {rate:.0f} img/s)")

    # Build inner-product index (vectors are normalized → IP == cosine similarity)
    print("\nBuilding FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(all_vectors)
    faiss.write_index(index, INDEX_OUT)
    print(f"Saved index with {index.ntotal:,} vectors -> {INDEX_OUT}")

    # Row-order mapping so faiss_index.py can translate row -> article_id
    pd.DataFrame({"article_id": article_ids}).to_csv(MAPPING_OUT, index=False)
    print(f"Saved mapping -> {MAPPING_OUT}")

    print("\nDone. Download both files from the Output tab and place them in")
    print("m2_multimodal_rag/vector_db/ (replacing the old index + mapping).")


if __name__ == "__main__":
    main()
