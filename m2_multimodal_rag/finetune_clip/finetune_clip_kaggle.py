"""
Fine-tunes CLIP (ViT-B-32, laion2b_s34b_b79k) on the H&M sample dataset
(41,794 articles) using the SAME contrastive objective CLIP was originally
trained with.

RUN THIS ON KAGGLE (free GPU, dataset already mounted — no download needed).

── Kaggle notebook setup ──────────────────────────────────────────────────
1. New Notebook → Settings → Accelerator → GPU T4 x2 (or P100)
2. Add Data:
     - Competition dataset: "h-and-m-personalized-fashion-recommendations"
       (mounts at /kaggle/input/h-and-m-personalized-fashion-recommendations/)
     - Your own dataset: upload `sample_articles.csv` as a private Kaggle
       Dataset (e.g. named "hm-sample-articles")
       (mounts at /kaggle/input/hm-sample-articles/sample_articles.csv)
3. Install dependency:
     !pip install -q open_clip_torch
4. Paste this whole file into a cell and run.
─────────────────────────────────────────────────────────────────────────

Output (saved to /kaggle/working/):
    clip_finetuned_hm.pt   — fine-tuned model state_dict (~600MB)
                              Download this from the notebook's Output tab.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip

# ── Paths (Kaggle) ────────────────────────────────────────────────────────────
IMAGES_DIR   = "/kaggle/input/competitions/h-and-m-personalized-fashion-recommendations/images"
ARTICLES_CSV = "/kaggle/input/datasets/maleeshavidurath/hm-sample-articles/sample_articles.csv"
OUTPUT_PATH      = "/kaggle/working/clip_finetuned_hm.pt"
BEST_OUTPUT_PATH = "/kaggle/working/clip_finetuned_hm_best.pt"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE      = 128
EPOCHS          = 4
LR              = 5e-6          # low LR — fine-tuning, not training from scratch
WEIGHT_DECAY    = 0.01
NUM_WORKERS     = 2
UNFREEZE_BLOCKS = 2              # fine-tune only the last N transformer blocks of each tower
VAL_FRACTION    = 0.05
EARLY_STOP_PATIENCE = 2           # stop if val_loss doesn't improve for this many epochs
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Dataset ─────────────────────────────────────────────────────────────────

def build_text(row) -> str:
    """Same field combination used by the M2 cross-encoder — keeps text style consistent."""
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


class HMImageTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess, tokenizer):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article_id = str(row["article_id"]).zfill(10)
        prefix = article_id[:3]
        image_path = os.path.join(IMAGES_DIR, prefix, f"{article_id}.jpg")

        with Image.open(image_path) as img:
            image = self.preprocess(img.convert("RGB"))

        text = build_text(row)
        text_tokens = self.tokenizer([text])[0]   # (context_length,)

        return image, text_tokens


# ── 2. Build model + freeze most layers ────────────────────────────────────────

def setup_model():
    print(f"Device: {DEVICE}")
    print("Loading CLIP ViT-B-32 (laion2b_s34b_b79k)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(DEVICE)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N visual transformer blocks + final projection
    visual_blocks = model.visual.transformer.resblocks
    for block in visual_blocks[-UNFREEZE_BLOCKS:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.visual.ln_post.parameters():
        param.requires_grad = True
    if model.visual.proj is not None:
        model.visual.proj.requires_grad = True

    # Unfreeze last N text transformer blocks + final projection
    text_blocks = model.transformer.resblocks
    for block in text_blocks[-UNFREEZE_BLOCKS:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.ln_final.parameters():
        param.requires_grad = True
    model.text_projection.requires_grad = True

    # Logit scale (temperature) is small and important — keep trainable
    model.logit_scale.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model, preprocess, tokenizer


# ── 3. CLIP contrastive loss ────────────────────────────────────────────────────

def clip_contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


# ── 4. Train / validate ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for images, texts in dataloader:
        images, texts = images.to(DEVICE), texts.to(DEVICE)
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        loss = clip_contrastive_loss(image_features, text_features, model.logit_scale.exp())
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def filter_existing_images(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows whose image file is missing from the dataset (a small
    number of H&M articles have no corresponding image).

    Lists each prefix folder once (~100 folders) instead of calling
    os.path.exists() per article — the latter is extremely slow on
    Kaggle's FUSE-mounted competition dataset (41k+ individual stat calls).
    """
    print("Indexing available image files...")
    existing = set()
    for prefix in os.listdir(IMAGES_DIR):
        prefix_dir = os.path.join(IMAGES_DIR, prefix)
        if os.path.isdir(prefix_dir):
            existing.update(os.listdir(prefix_dir))

    def has_image(article_id):
        return f"{str(article_id).zfill(10)}.jpg" in existing

    mask = df["article_id"].apply(has_image)
    missing = int((~mask).sum())
    if missing:
        print(f"Skipping {missing:,} articles with missing image files.")
    return df[mask].reset_index(drop=True)


def main():
    print(f"Loading articles from {ARTICLES_CSV}...")
    articles = pd.read_csv(ARTICLES_CSV, dtype={"article_id": str})
    print(f"Total articles: {len(articles):,}")

    articles = filter_existing_images(articles)
    print(f"Articles with images: {len(articles):,}")

    # Train / val split
    articles = articles.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = int(len(articles) * VAL_FRACTION)
    val_df = articles.iloc[:n_val]
    train_df = articles.iloc[n_val:]
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    model, preprocess, tokenizer = setup_model()

    train_ds = HMImageTextDataset(train_df, preprocess, tokenizer)
    val_ds = HMImageTextDataset(val_df, preprocess, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, drop_last=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    print("\nBaseline (pre-fine-tune) validation loss:")
    val_loss = evaluate(model, val_loader)
    print(f"  val_loss = {val_loss:.4f}\n")

    best_val_loss = val_loss
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for step, (images, texts) in enumerate(train_loader, 1):
            images, texts = images.to(DEVICE), texts.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                loss = clip_contrastive_loss(image_features, text_features, model.logit_scale.exp())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Clamp logit_scale as in original CLIP training (prevents instability)
            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"  Epoch {epoch} | step {step}/{len(train_loader)} "
                      f"| loss={running_loss/step:.4f}")

        val_loss = evaluate(model, val_loader)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{EPOCHS} done in {elapsed/60:.1f} min | "
              f"train_loss={running_loss/len(train_loader):.4f} | val_loss={val_loss:.4f}")

        # Save checkpoint after every epoch (in case Kaggle session times out)
        torch.save(model.state_dict(), OUTPUT_PATH)
        print(f"  Saved checkpoint -> {OUTPUT_PATH}")

        # Track the best checkpoint by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_OUTPUT_PATH)
            print(f"  New best val_loss ({val_loss:.4f}) -> saved {BEST_OUTPUT_PATH}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{EARLY_STOP_PATIENCE})")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping — val_loss hasn't improved for "
                  f"{EARLY_STOP_PATIENCE} epochs. Best val_loss = {best_val_loss:.4f}")
            break

    print(f"\nDone. Best val_loss = {best_val_loss:.4f}")
    print(f"Download {BEST_OUTPUT_PATH} (best checkpoint) from the Output tab.")
    print(f"({OUTPUT_PATH} contains the last epoch's weights, for reference.)")


if __name__ == "__main__":
    main()
