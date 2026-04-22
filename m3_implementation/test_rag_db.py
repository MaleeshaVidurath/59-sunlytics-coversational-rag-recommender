# test_rag_db.py
# Run from m3_implementation folder: python test_rag_db.py

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_rag.db.postgres_client import (
    get_article_by_id, search_articles_filtered,
    create_schema, close_pool
)
from text_rag.db.qdrant_client import semantic_search


async def test():
    print("=" * 55)
    print("TEXT RAG DATABASE TEST")
    print("=" * 55)

    await create_schema()

    # Test 1: PostgreSQL single article lookup (valid article_id)
    print("\nTest 1: PostgreSQL single article lookup")
    art = await get_article_by_id("108775015")
    if art:
        print(f"  OK: {art['prod_name']} "
              f"({art['colour_group_name']}) "
              f"£{art['avg_price']}")
    else:
        print("  FAIL: article not found")

    # Test 2: PostgreSQL filtered search
    print("\nTest 2: PostgreSQL filtered search")
    results = await search_articles_filtered(
        filters={
            "colour_group_name": "Black",
            "product_type_name": "Dress",
            "price_max": 50.0
        },
        limit=3
    )
    print(f"  OK: {len(results)} results")
    for r in results:
        print(f"    {r['prod_name']} "
              f"({r['colour_group_name']}) "
              f"£{r['avg_price']}")

    # Test 3: Qdrant semantic search with filter
    print("\nTest 3: Qdrant semantic search with filter")
    hits = semantic_search(
        query="black casual dress for summer",
        filters={"colour_group_name": "Black"},
        top_k=3
    )
    print(f"  OK: {len(hits)} results")
    for h in hits:
        print(f"    [{h['_score']:.3f}] {h['prod_name']} "
              f"({h['colour_group_name']}) "
              f"£{h.get('avg_price', 'N/A')}")

    # Test 4: Qdrant natural language search no filter
    print("\nTest 4: Qdrant semantic search - natural language")
    hits2 = semantic_search(
        query="something floral and casual for a summer party",
        top_k=3
    )
    print(f"  OK: {len(hits2)} results")
    for h in hits2:
        print(f"    [{h['_score']:.3f}] {h['prod_name']} "
              f"({h['colour_group_name']}, {h['product_type_name']}) "
              f"£{h.get('avg_price', 'N/A')}")

    await close_pool()
    print("\n" + "=" * 55)
    print("All database tests complete.")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(test())
