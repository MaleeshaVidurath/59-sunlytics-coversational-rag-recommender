"""
Verification for Filtered FAISS (hard filters at index level, pool 50 → 15).

Checks, without loading CLIP or any LLM:
    1. _hard_filter_allowed_ids matches the old per-candidate filter semantics.
    2. Every result from a selector-filtered search passes the hard filters.
    3. Filtered pool is <= 15; unfiltered baseline needs post-filtering.
    4. Latency comparison: filtered top-15 vs unfiltered top-50 + Python filter.

Usage (from repo root):
    python m2_multimodal_rag/testing_scripts/test_filtered_faiss.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m2_multimodal_rag.faiss_index import faiss_db                      # noqa: E402
from m2_multimodal_rag.m2_handlers import (                             # noqa: E402
    _hard_filter_allowed_ids,
    _load_articles_priced,
)

FILTERS     = {"colour_group_name": "Black", "product_type_name": "Dress"}
EXCLUDE_IDS = []


def _random_query_vectors(n: int = 3, dim: int = 512) -> list:
    """Random unit vectors stand in for CLIP embeddings — the selector logic
    is independent of what the query vector encodes."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return [v for v in vecs]


def _passes_hard_filters(metadata: dict, filters: dict) -> bool:
    """Reference implementation — the old Phase 2 per-candidate check."""
    for key, value in filters.items():
        if key in ("price_max", "price_min"):
            price = metadata.get("price")
            if price is None or price != price:
                continue
            if key == "price_max" and price > value:
                return False
            if key == "price_min" and price < value:
                return False
        elif str(metadata.get(key, "")).strip().lower() != str(value).strip().lower():
            return False
    return True


def main() -> None:
    if not faiss_db.database_ready:
        print("[SKIP] FAISS index not present — download vector_db files first.")
        sys.exit(1)

    articles_df = _load_articles_priced()
    meta_by_id = {
        str(row["article_id"]).zfill(10): row
        for row in articles_df.to_dict("records")
    }

    # ------------------------------------------------------------------
    # 1. Vectorised allowed-ids vs reference per-item semantics
    # ------------------------------------------------------------------
    allowed = set(_hard_filter_allowed_ids(articles_df, FILTERS, EXCLUDE_IDS))
    reference = {
        aid for aid, meta in meta_by_id.items()
        if _passes_hard_filters(meta, FILTERS)
    }
    assert allowed == reference, (
        f"allowed-ids mismatch: {len(allowed)} vectorised vs {len(reference)} reference"
    )
    print(f"[PASS] _hard_filter_allowed_ids matches reference semantics "
          f"({len(allowed):,} articles pass {FILTERS})")

    # ------------------------------------------------------------------
    # 2 + 3. Filtered search returns only filter-valid items, pool <= 15
    # ------------------------------------------------------------------
    query_vectors = _random_query_vectors()
    selector = faiss_db.build_id_selector(list(allowed))
    assert selector is not None, "build_id_selector returned None despite allowed ids"

    t0 = time.perf_counter()
    filtered_candidates = faiss_db.search_multi(query_vectors, top_k=15, selector=selector)
    t_filtered = time.perf_counter() - t0

    assert filtered_candidates, "filtered search returned no candidates"
    assert len(filtered_candidates) <= 15, f"pool too large: {len(filtered_candidates)}"
    bad = [aid for aid, _ in filtered_candidates if aid not in allowed]
    assert not bad, f"filter-violating candidates leaked through: {bad}"
    print(f"[PASS] Filtered search: {len(filtered_candidates)} candidates, "
          f"all pass hard filters, pool <= 15")

    # ------------------------------------------------------------------
    # 4. Latency + survivor comparison vs old unfiltered path
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    unfiltered = faiss_db.search_multi(query_vectors, top_k=50)
    survivors = [aid for aid, _ in unfiltered if aid in allowed]
    t_unfiltered = time.perf_counter() - t0

    print(f"[INFO] Old path: 50 retrieved → {len(survivors)} survive hard filters "
          f"({t_unfiltered*1000:.1f} ms)")
    print(f"[INFO] New path: {len(filtered_candidates)} retrieved, all valid "
          f"({t_filtered*1000:.1f} ms)")
    print("[DONE] Filtered FAISS verified.")


if __name__ == "__main__":
    main()
