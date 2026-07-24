"""
Neural Collaborative Filtering (NCF) Training Script.
Architecture: Two-Tower NCF with Item Content Features (He et al., NeurIPS 2017)

Item Tower:
  item_id embedding (64-D) + 6 content feature embeddings (colour, type,
  department, index_group, garment_group, graphical_appearance)
  → Linear projection → 64-D item representation

User Tower:
  user_id embedding → 64-D user representation

Score = dot_product(user_repr, item_repr)
Loss  = BPR (Bayesian Personalised Ranking — Rendle et al., 2009)

Key advantage over ALS:
  - Non-linear interactions via MLP (captures complex patterns)
  - Item content features enable cold-start for unseen articles
  - Learns jointly from purchase signals AND article metadata

Run from project root:
    python -m m2_multimodal_rag.collaborative_filtering.train_cf_kaggle

Output → m2_multimodal_rag/collaborative_filtering/models/
    item_embeddings.npy   (41794, 64) — content-aware item representations
    article_ids.npy       (41794,)    — corresponding article IDs
    popularity.npy        (41794,)    — normalised purchase frequency
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR         = Path(__file__).resolve().parent.parent.parent
DATA_DIR         = BASE_DIR / "shared" / "main_data_set"
TRANSACTIONS_CSV = DATA_DIR / "sample_transactions.csv"
ARTICLES_CSV     = DATA_DIR / "sample_articles.csv"
MODEL_DIR        = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────

EMBED_DIM    = 64
EPOCHS       = 30
BATCH_SIZE   = 512
LR           = 1e-3
WEIGHT_DECAY = 1e-5
NEG_SAMPLES  = 4      # negatives per positive in BPR
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Load data ───────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
print("Loading data...")

transactions = pd.read_csv(TRANSACTIONS_CSV, dtype={"article_id": str, "customer_id": str})
articles     = pd.read_csv(ARTICLES_CSV,     dtype={"article_id": str})

print(f"Transactions : {len(transactions):,}")
print(f"Articles     : {len(articles):,}")


# ── 2. Encode categorical features ────────────────────────────────────────────

print("\nEncoding features...")

# Canonical article list (all 41,794 — ensures cold-start coverage)
article_id_list = articles["article_id"].values
article_id_to_idx = {aid: idx for idx, aid in enumerate(article_id_list)}
n_articles = len(article_id_list)

# User encoding (only from transactions)
customer_ids      = transactions["customer_id"].unique()
customer_to_idx   = {cid: idx for idx, cid in enumerate(customer_ids)}
n_users           = len(customer_ids)

# Item content feature encoders (fit on articles.csv — all items)
def make_encoder(series):
    cats = series.fillna("Unknown").unique()
    return {v: i for i, v in enumerate(cats)}, len(cats)

colour_enc,   n_colours   = make_encoder(articles["colour_group_name"])
type_enc,     n_types     = make_encoder(articles["product_type_name"])
dept_enc,     n_depts     = make_encoder(articles["department_name"])
index_enc,    n_indexes   = make_encoder(articles["index_group_name"])
garment_enc,  n_garments  = make_encoder(articles["garment_group_name"])
appear_enc,   n_appears   = make_encoder(articles["graphical_appearance_name"])

print(f"Users     : {n_users}")
print(f"Articles  : {n_articles}")
print(f"Colours   : {n_colours}  |  Types: {n_types}  |  Depts: {n_depts}")
print(f"Indexes   : {n_indexes}  |  Garments: {n_garments}  |  Appearances: {n_appears}")

# Pre-build article content feature tensors (shape: n_articles × 1 each)
def build_feature_tensor(articles_df, encoder, col):
    return torch.tensor(
        [encoder.get(str(v), 0) for v in articles_df[col].fillna("Unknown")],
        dtype=torch.long,
    )

art_colour  = build_feature_tensor(articles, colour_enc,  "colour_group_name")
art_type    = build_feature_tensor(articles, type_enc,    "product_type_name")
art_dept    = build_feature_tensor(articles, dept_enc,    "department_name")
art_index   = build_feature_tensor(articles, index_enc,   "index_group_name")
art_garment = build_feature_tensor(articles, garment_enc, "garment_group_name")
art_appear  = build_feature_tensor(articles, appear_enc,  "graphical_appearance_name")


# ── 3. Build training pairs ────────────────────────────────────────────────────

print("\nBuilding training pairs...")

# Map transactions to encoded indices, drop unknowns
tx = transactions.copy()
tx["user_idx"] = tx["customer_id"].map(customer_to_idx)
tx["item_idx"] = tx["article_id"].map(article_id_to_idx)
tx = tx.dropna(subset=["user_idx", "item_idx"])
tx["user_idx"] = tx["user_idx"].astype(int)
tx["item_idx"] = tx["item_idx"].astype(int)

# Set of items each user purchased (for negative sampling)
user_purchased = tx.groupby("user_idx")["item_idx"].apply(set).to_dict()

positive_pairs = list(zip(tx["user_idx"].values, tx["item_idx"].values))
print(f"Positive training pairs : {len(positive_pairs):,}")


# ── 4. BPR Dataset ────────────────────────────────────────────────────────────

class BPRDataset(Dataset):
    """
    Bayesian Personalised Ranking dataset.
    Each sample = (user, positive_item, negative_item).
    Negative item is sampled uniformly from items the user did NOT purchase.
    """
    def __init__(self, pairs, user_purchased, n_items, n_neg=NEG_SAMPLES):
        self.pairs          = pairs
        self.user_purchased = user_purchased
        self.n_items        = n_items
        self.n_neg          = n_neg

    def __len__(self):
        return len(self.pairs) * self.n_neg

    def __getitem__(self, idx):
        user_idx, pos_idx = self.pairs[idx // self.n_neg]
        purchased = self.user_purchased.get(user_idx, set())
        # Sample a true negative
        while True:
            neg_idx = np.random.randint(0, self.n_items)
            if neg_idx not in purchased:
                break
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_idx,  dtype=torch.long),
            torch.tensor(neg_idx,  dtype=torch.long),
        )


# ── 5. Model definition ───────────────────────────────────────────────────────

class ItemTower(nn.Module):
    """
    Encodes an item using its ID embedding fused with 6 content feature
    embeddings, projected to a shared EMBED_DIM space.
    """
    def __init__(self):
        super().__init__()
        self.item_embed    = nn.Embedding(n_articles, EMBED_DIM)
        self.colour_embed  = nn.Embedding(n_colours,  16)
        self.type_embed    = nn.Embedding(n_types,    16)
        self.dept_embed    = nn.Embedding(n_depts,    16)
        self.index_embed   = nn.Embedding(n_indexes,   8)
        self.garment_embed = nn.Embedding(n_garments,  8)
        self.appear_embed  = nn.Embedding(n_appears,   8)

        in_dim = EMBED_DIM + 16 + 16 + 16 + 8 + 8 + 8   # 136
        self.proj = nn.Sequential(
            nn.Linear(in_dim, EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

    def forward(self, item_idx):
        x = torch.cat([
            self.item_embed(item_idx),
            self.colour_embed(art_colour[item_idx].to(DEVICE)),
            self.type_embed(art_type[item_idx].to(DEVICE)),
            self.dept_embed(art_dept[item_idx].to(DEVICE)),
            self.index_embed(art_index[item_idx].to(DEVICE)),
            self.garment_embed(art_garment[item_idx].to(DEVICE)),
            self.appear_embed(art_appear[item_idx].to(DEVICE)),
        ], dim=-1)
        return self.proj(x)


class NeuralCF(nn.Module):
    """
    Two-Tower NCF: user tower (ID embedding) × item tower (ID + content).
    Score = dot_product(user_repr, item_repr).
    """
    def __init__(self):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, EMBED_DIM)
        self.item_tower = ItemTower()

    def forward(self, user_idx, pos_idx, neg_idx):
        user_repr = self.user_embed(user_idx)                   # (B, 64)
        pos_repr  = self.item_tower(pos_idx)                    # (B, 64)
        neg_repr  = self.item_tower(neg_idx)                    # (B, 64)

        pos_score = (user_repr * pos_repr).sum(dim=-1)          # (B,)
        neg_score = (user_repr * neg_repr).sum(dim=-1)          # (B,)

        # BPR loss: maximise score gap between positive and negative
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        return loss


# ── 6. Train ──────────────────────────────────────────────────────────────────

print("\nTraining Neural CF...")
print(f"  EMBED_DIM={EMBED_DIM}  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  NEG={NEG_SAMPLES}\n")

dataset    = BPRDataset(positive_pairs, user_purchased, n_articles)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model     = NeuralCF().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for user_idx, pos_idx, neg_idx in dataloader:
        user_idx = user_idx.to(DEVICE)
        pos_idx  = pos_idx.to(DEVICE)
        neg_idx  = neg_idx.to(DEVICE)

        optimizer.zero_grad()
        loss = model(user_idx, pos_idx, neg_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.5f}")


# ── 7. Extract item embeddings ────────────────────────────────────────────────

print("\nExtracting item embeddings for all articles...")

model.eval()
all_item_indices = torch.arange(n_articles, dtype=torch.long).to(DEVICE)

item_embeddings_list = []
with torch.no_grad():
    for batch_start in range(0, n_articles, 512):
        batch = all_item_indices[batch_start:batch_start + 512]
        emb   = model.item_tower(batch)
        item_embeddings_list.append(emb.cpu().numpy())

item_embeddings = np.vstack(item_embeddings_list).astype(np.float32)
print(f"Item embeddings shape : {item_embeddings.shape}")


# ── 8. Popularity scores ──────────────────────────────────────────────────────

print("Computing popularity scores...")

purchase_counts = transactions["article_id"].value_counts()
max_count       = purchase_counts.max()
popularity      = np.array(
    [purchase_counts.get(aid, 0) / max_count for aid in article_id_list],
    dtype=np.float32,
)
print(f"Most purchased item : {int(max_count)} times")


# ── 9. Save artifacts ─────────────────────────────────────────────────────────

np.save(MODEL_DIR / "item_embeddings.npy", item_embeddings)
np.save(MODEL_DIR / "article_ids.npy",     article_id_list)
np.save(MODEL_DIR / "popularity.npy",      popularity)

print(f"\nSaved to {MODEL_DIR}/")
print(f"  item_embeddings.npy  shape : {item_embeddings.shape}")
print(f"  article_ids.npy      count : {len(article_id_list):,}")
print(f"  popularity.npy       count : {len(popularity):,}")
print("\nM2 system will load these automatically on next startup.")
