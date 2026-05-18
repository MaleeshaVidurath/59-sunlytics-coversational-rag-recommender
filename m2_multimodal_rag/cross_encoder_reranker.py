"""
M2 Cross-Encoder Neural Re-ranker.

Implements Stage 2 of a two-stage retrieval architecture:

  Stage 1 — Bi-encoder  (CLIP):         encodes query and items separately
                                          → fast approximate search over 105k products
  Stage 2 — Cross-encoder (MiniLM-BERT): encodes (query, item) jointly via cross-attention
                                          → precise neural relevance scoring on top-N candidates

The cross-encoder reads both the query and item description in a single forward pass,
allowing full attention between every query token and every item token. This makes it
significantly more accurate than the bi-encoder but too slow to run over the full catalog,
hence the two-stage design.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Architecture : 6-layer MiniLM (distilled BERT), 22M parameters, ~90MB
  - Training data: MS MARCO (12.8M real query-passage relevance pairs from Bing)
  - Output       : raw relevance logit — higher means more relevant
  - Inference    : CPU-friendly (~50ms per 20 pairs on modern CPU)

Paper: RAG-VisualRec — two-stage retrieval with neural re-ranking for recommendation.
"""

from sentence_transformers import CrossEncoder


class NeuralCrossEncoderReranker:

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        print(f"M2 CrossEncoder: Loading neural re-ranker ({self.MODEL_NAME})...")
        try:
            self.model = CrossEncoder(self.MODEL_NAME, max_length=512)
            self._ready = True
            print("M2 CrossEncoder: [SUCCESS] Neural re-ranker ready.")
        except Exception as e:
            self._ready = False
            print(f"M2 CrossEncoder: [WARNING] Failed to load: {e}. Neural re-ranking will be skipped.")

    def rerank(self, query: str, candidates: list, top_k: int = 20) -> list:
        """
        Scores each (query, item_description) pair through the cross-encoder transformer.

        Each pair is concatenated as a single sequence fed through MiniLM:
            [CLS] query [SEP] item_name colour product_type description [SEP]

        The model's output logit reflects how relevant the item is to the query
        via full cross-attention — far more expressive than cosine similarity alone.

        Args:
            query      : The base search text (enriched with soft constraints).
            candidates : Filtered candidates from Phase 2, each a dict with 'metadata'.
            top_k      : How many candidates to score (rest are passed through unchanged).

        Returns:
            candidates re-sorted by 'cross_encoder_score' (descending).
            Each candidate in the scored pool gets a new 'cross_encoder_score' key.
        """
        if not self._ready or not candidates:
            return candidates

        pool = candidates[:top_k]

        def _safe(val, max_len=None) -> str:
            """Converts a metadata value to string, treating NaN floats as empty."""
            s = "" if isinstance(val, float) else str(val or "").strip()
            return s[:max_len] if max_len else s

        # Build (query, item_text) input pairs for the cross-encoder
        pairs = []
        for c in pool:
            m = c.get("metadata", {})
            item_text = " ".join(filter(None, [
                _safe(m.get("prod_name")),
                _safe(m.get("colour_group_name")),
                _safe(m.get("product_type_name")),
                _safe(m.get("department_name")),
                _safe(m.get("graphical_appearance_name")),
                _safe(m.get("detail_desc"), max_len=200),
            ]))
            pairs.append([query, item_text])

        # Single batched forward pass through the cross-encoder transformer
        raw_scores = self.model.predict(pairs)

        # Attach neural score to each candidate
        for i, candidate in enumerate(pool):
            candidate["cross_encoder_score"] = float(raw_scores[i])

        # Sort the scored pool by neural relevance (descending)
        reranked_pool = sorted(pool, key=lambda x: x["cross_encoder_score"], reverse=True)

        # Append any candidates beyond top_k unchanged (no cross-encoder score)
        reranked = reranked_pool + candidates[top_k:]

        top_score = reranked_pool[0]["cross_encoder_score"] if reranked_pool else 0.0
        print(f"   [CrossEncoder] Scored {len(pool)} pairs via neural cross-attention. "
              f"Top relevance logit: {top_score:.4f}")
        return reranked


# Singleton — model is loaded once at startup and reused across all requests
cross_encoder_reranker = NeuralCrossEncoderReranker()
