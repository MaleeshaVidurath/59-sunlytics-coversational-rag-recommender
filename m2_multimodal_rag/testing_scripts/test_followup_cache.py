"""
Verification for follow-up handling via M3 session memory
(resolve conversational follow-ups without full re-retrieval).

Checks, without any LLM calls:
    1. Session pool cache: put/get round-trip, TTL expiry, LRU size cap.
    2. Fast path survivors: refinement filters (e.g. lowered price_max) are
       re-applied to the cached pool with exclusions honoured.
    3. Fast path refusal: too-few survivors (contradictory refinement)
       returns None so the handler falls back to full retrieval.
    4. End-to-end wiring: handle_catalog_search takes the fast path on a
       simulated REFINEMENT (verified by counting FAISS search_multi calls),
       full pipeline phases 3-6 stubbed out to avoid network calls.

Usage (from repo root):
    PYTHONIOENCODING=utf-8 python m2_multimodal_rag/testing_scripts/test_followup_cache.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import m2_multimodal_rag.m2_handlers as mh                              # noqa: E402

SESSION = "sess_test_followup"


def test_cache_roundtrip_ttl_cap():
    mh._session_pool_cache.clear()

    pool = [("0000000001", 0.9), ("0000000002", 0.8)]
    mh._session_pool_put(SESSION, pool)
    assert mh._session_pool_get(SESSION) == pool, "put/get round-trip failed"

    # TTL expiry
    mh._session_pool_cache[SESSION]["ts"] = time.time() - mh._SESSION_POOL_TTL_S - 1
    assert mh._session_pool_get(SESSION) is None, "expired pool was returned"
    assert SESSION not in mh._session_pool_cache, "expired entry not evicted"

    # LRU size cap
    for i in range(mh._SESSION_POOL_MAX + 5):
        mh._session_pool_put(f"sess_{i}", pool)
    assert len(mh._session_pool_cache) == mh._SESSION_POOL_MAX, "size cap not enforced"
    assert "sess_0" not in mh._session_pool_cache, "oldest session not evicted"

    mh._session_pool_cache.clear()
    print("[PASS] Cache round-trip, TTL expiry and LRU cap all behave correctly")


def test_fast_path_survivors(articles_df):
    mh._session_pool_cache.clear()

    # Build a realistic cached pool: top Black Dresses with prices
    base_filters = {"colour_group_name": "Black", "product_type_name": "Dress"}
    allowed = mh._hard_filter_allowed_ids(articles_df, base_filters, [])
    priced = articles_df[
        articles_df["article_id"].astype(str).str.zfill(10).isin(allowed[:15])
    ][["article_id", "price"]].dropna()
    pool = [(str(a).zfill(10), 0.9 - i * 0.01)
            for i, a in enumerate(priced["article_id"])]
    assert len(pool) >= 5, "need at least 5 priced Black Dresses for this test"
    mh._session_pool_put(SESSION, pool)

    # Refinement: "cheaper ones" → price_max at the pool's median price,
    # excluding the two items already shown
    prices = sorted(priced["price"])
    price_cap = prices[len(prices) // 2]
    shown = [pool[0][0], pool[1][0]]
    refined_filters = {**base_filters, "price_max": price_cap}

    survivors = mh._followup_pool_from_cache(
        SESSION, articles_df, refined_filters, shown, min_needed=2)
    assert survivors is not None, "fast path refused a resolvable refinement"
    assert all(aid not in shown for aid, _ in survivors), "shown items not excluded"

    meta = articles_df.set_index(
        articles_df["article_id"].astype(str).str.zfill(10))
    for aid, _ in survivors:
        p = meta.loc[aid, "price"]
        assert p != p or p <= price_cap, f"{aid} violates price_max={price_cap}"

    pool_order = [aid for aid, _ in pool]
    surv_order = [aid for aid, _ in survivors]
    assert surv_order == [a for a in pool_order if a in set(surv_order)], \
        "relevance order not preserved"
    print(f"[PASS] Refinement fast path: {len(pool)} cached → {len(survivors)} "
          f"survivors, price/exclusion/order all correct")

    # Contradictory refinement (new colour) → too few survivors → None
    contradictory = {**base_filters, "colour_group_name": "Red"}
    result = mh._followup_pool_from_cache(
        SESSION, articles_df, contradictory, [], min_needed=3)
    assert result is None, "fast path should refuse a colour flip against a Black pool"
    print("[PASS] Contradictory refinement correctly falls back to full retrieval")

    # No session cache → None
    assert mh._followup_pool_from_cache(
        "sess_unknown", articles_df, base_filters, [], min_needed=2) is None
    print("[PASS] Unknown session correctly falls back to full retrieval")
    mh._session_pool_cache.clear()


def test_handler_takes_fast_path(articles_df):
    """Runs handle_catalog_search twice with phases 3-6 stubbed, counting
    FAISS searches: turn 1 (initial) must search, turn 2 (refinement) must not."""
    mh._session_pool_cache.clear()

    search_calls = {"n": 0}
    real_search_multi = mh.faiss_db.search_multi

    def counting_search_multi(*args, **kwargs):
        search_calls["n"] += 1
        return real_search_multi(*args, **kwargs)

    # Stub out everything with network cost / heavy models
    stubs = {
        "expand_query":       mh.llm_generator.expand_query,
        "rerank_candidates":  mh.llm_generator.rerank_candidates,
        "cross_rerank":       mh.cross_encoder_reranker.rerank,
        "generator_loop":     mh.generator_loop.generate_faithful_explanation,
        "get_image":          mh.data_loader.get_image,
    }
    mh.faiss_db.search_multi           = counting_search_multi
    mh.llm_generator.expand_query      = lambda q: [q]
    mh.llm_generator.rerank_candidates = lambda **kw: kw["candidates"]
    mh.cross_encoder_reranker.rerank   = lambda **kw: kw["candidates"][:kw.get("top_k", 20)]
    mh.generator_loop.generate_faithful_explanation = (
        lambda **kw: ("stub explanation", []))
    mh.data_loader.get_image           = lambda aid: None

    try:
        base_input = {
            "action": "catalog_search",
            "retrieval_strategy": "FULL",
            "user_message": "show me black dresses",
            "exclude_ids": [],
            "payload": {
                "filters": {"colour_group_name": "Black",
                            "product_type_name": "Dress"},
                "num_items": 2,
            },
        }

        # Turn 1 — INITIAL_REQUEST: full retrieval, pool gets cached
        r1 = mh.handle_catalog_search(
            base_input, memory_context={"session_id": SESSION})
        calls_turn1 = search_calls["n"]
        assert calls_turn1 >= 1, "initial request should hit FAISS"
        assert mh._session_pool_get(SESSION), "pool was not cached after turn 1"
        shown = [it["article_id"] for it in r1.get("items", [])]

        # Turn 2 — REFINEMENT ("cheaper"): must resolve from cache, zero FAISS
        refine_input = dict(base_input)
        refine_input["user_message"] = "show me cheaper ones"
        refine_input["exclude_ids"] = shown
        refine_input["payload"] = dict(base_input["payload"])
        r2 = mh.handle_catalog_search(
            refine_input,
            memory_context={"session_id": SESSION,
                            "new_changes": {"price_max": 999.0},
                            "previous_constraints": base_input["payload"]["filters"]})
        assert search_calls["n"] == calls_turn1, (
            f"refinement re-ran FAISS ({search_calls['n'] - calls_turn1} extra calls)")
        assert r2.get("success"), f"refinement turn failed: {r2.get('error')}"
        assert all(it["article_id"] not in shown for it in r2.get("items", [])), \
            "refinement re-recommended already-shown items"
        print(f"[PASS] Handler fast path: turn 1 used FAISS ({calls_turn1} searches), "
              f"turn 2 resolved from session cache with 0 searches")
    finally:
        mh.faiss_db.search_multi           = real_search_multi
        mh.llm_generator.expand_query      = stubs["expand_query"]
        mh.llm_generator.rerank_candidates = stubs["rerank_candidates"]
        mh.cross_encoder_reranker.rerank   = stubs["cross_rerank"]
        mh.generator_loop.generate_faithful_explanation = stubs["generator_loop"]
        mh.data_loader.get_image           = stubs["get_image"]
        mh._session_pool_cache.clear()


def main() -> None:
    articles_df = mh._load_articles_priced()
    test_cache_roundtrip_ttl_cap()
    test_fast_path_survivors(articles_df)
    if mh.faiss_db.database_ready:
        test_handler_takes_fast_path(articles_df)
    else:
        print("[SKIP] FAISS index not present — handler end-to-end check skipped.")
    print("[DONE] Follow-up session-memory fast path verified.")


if __name__ == "__main__":
    main()
