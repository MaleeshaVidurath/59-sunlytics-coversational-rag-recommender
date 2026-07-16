"""
CF Scorer — local inference module for Neural Collaborative Filtering.

Loads artifacts produced by train_cf_kaggle.py (NCF two-tower model) and
provides item-level scores for catalog_search Phase 2.

If model files are missing, all scores return 0.0 and the system falls
back to rule-based purchase history scoring automatically.
"""

import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

_USER_VECTOR_SAMPLE_CAP = 500


class CFScorer:
    """
    Scores candidate articles using content-aware item embeddings from a
    Neural Collaborative Filtering model (He et al., NeurIPS 2017) trained
    on 185,037 H&M purchase signals with BPR loss.

    Item embeddings are 64-D representations that jointly encode:
      - Collaborative signal (co-purchase patterns from transactions)
      - Content features (colour, type, department, index group,
        garment group, graphical appearance from articles.csv)

    This enables cold-start scoring for articles with no purchase history
    — a key advantage over pure collaborative filtering (ALS).

    Scoring has two components:
      1. Preference similarity — cosine similarity between the candidate's
         NCF embedding and a proxy user vector built from purchase_hints
      2. Popularity — normalised purchase frequency of the candidate item
    """

    def __init__(self):
        self.item_embeddings = None    # np.ndarray shape: (N_articles, 64)
        self.article_id_map  = {}      # str article_id → row index
        self.popularity      = None    # np.ndarray shape: (N_articles,)
        self._loaded         = False

    def load(self) -> bool:
        """
        Loads NCF model artifacts from disk.
        Safe to call even if files are missing — returns False and disables CF.
        """
        embed_path      = MODEL_DIR / "item_embeddings.npy"
        article_id_path = MODEL_DIR / "article_ids.npy"
        popularity_path = MODEL_DIR / "popularity.npy"

        missing = [p.name for p in [embed_path, article_id_path, popularity_path] if not p.exists()]
        if missing:
            print(f"[CF] Model files not found: {missing}")
            print("[CF] CF scoring disabled. Run: "
                  "python -m m2_multimodal_rag.collaborative_filtering.train_cf_kaggle")
            return False

        print("[CF] Loading Neural CF model...")
        self.item_embeddings = np.load(embed_path, allow_pickle=True)
        article_ids          = np.load(article_id_path, allow_pickle=True)
        self.popularity      = np.load(popularity_path, allow_pickle=True)
        self.article_id_map  = {str(aid): idx for idx, aid in enumerate(article_ids)}
        self._loaded         = True

        print(f"[CF] Ready. "
              f"Items: {len(self.article_id_map):,}  "
              f"Embed dim: {self.item_embeddings.shape[1]}  "
              f"Model: NCF two-tower (BPR)")
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _item_vector(self, article_id: str) -> np.ndarray | None:
        idx = self.article_id_map.get(str(article_id).zfill(10))
        if idx is None:
            return None
        return self.item_embeddings[idx]

    def _build_user_vector(self, purchase_hints: dict, articles_df) -> np.ndarray | None:
        """
        Constructs a proxy user preference vector by averaging NCF item
        embeddings of articles matching the user's top colours and types.

        Since NCF embeddings encode content features, this proxy vector
        captures both collaborative and content-based user preferences —
        without needing explicit purchase article_ids from m3.

        Academic basis: item-based CF proxy (Sarwar et al., 2001).
        """
        top_colours = purchase_hints.get("top_colours") or []
        top_types   = purchase_hints.get("top_product_types") or []

        if not top_colours and not top_types:
            return None

        mask = (
            articles_df["colour_group_name"].isin(top_colours) |
            articles_df["product_type_name"].isin(top_types)
        )
        matching_ids = articles_df[mask]["article_id"].astype(str).values

        vectors = []
        for aid in matching_ids[:_USER_VECTOR_SAMPLE_CAP]:
            idx = self.article_id_map.get(aid.zfill(10))
            if idx is not None:
                vectors.append(self.item_embeddings[idx])

        if not vectors:
            return None

        user_vec = np.mean(vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(user_vec)
        return user_vec / norm if norm > 0 else user_vec

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, article_id: str, purchase_hints: dict, articles_df) -> float:
        """
        Returns a NCF score in [0.0, ~0.4] for a candidate article.

        Score = (cosine_similarity × 0.3) + (popularity × 0.1)

        Returns 0.0 if model not loaded (triggers rule-based fallback).
        Cold-start items (no purchases) still score via content embeddings.
        """
        if not self._loaded:
            return 0.0

        item_vec = self._item_vector(article_id)
        if item_vec is None:
            return 0.0

        idx              = self.article_id_map[str(article_id).zfill(10)]
        popularity_score = float(self.popularity[idx]) * 0.1

        user_vec = self._build_user_vector(purchase_hints, articles_df)
        if user_vec is None:
            return popularity_score

        item_norm = np.linalg.norm(item_vec)
        if item_norm == 0:
            return popularity_score

        similarity = float(np.dot(user_vec, item_vec / item_norm))
        similarity = max(0.0, similarity)

        return round(similarity * 0.3 + popularity_score, 4)


# Global singleton — loaded once at startup via cf_scorer.load()
cf_scorer = CFScorer()
