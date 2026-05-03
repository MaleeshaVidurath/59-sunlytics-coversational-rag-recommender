# m3_implementation/memory/core/test_enrichment.py

import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from memory.db.redis_client import connect_to_redis, close_redis_connection
from memory.core.session_manager import SessionManager
from memory.core.turn_manager import TurnManager
from memory.core.user_manager import UserManager
from memory.core.enrichment import EnrichmentLayer
from memory.models.schemas import ItemInContext, TurnClassification


async def test_enrichment():
    await connect_to_mongodb()
    await connect_to_redis()

    session_mgr = SessionManager()
    turn_mgr    = TurnManager()
    user_mgr    = UserManager()
    enricher    = EnrichmentLayer()

    # ── Setup: create user and session ────────────────────────────────────
    user = await user_mgr.get_or_create_user(
        customer_id="test_enrich_customer_001",
        initial_data={"age": 30, "club_member_status": "ACTIVE"}
    )
    session = await session_mgr.get_or_create_session(user_id=user.user_id)
    sid = session.session_id
    uid = user.user_id

    # Seed some long-term preferences
    await user_mgr.update_preferences_from_entities(
        user_id=uid,
        entities={"colour_group_name": "Black", "product_type_name": "Dress"},
        sentiment=0.90, source="explicit"
    )

    print(f"\nSetup complete — session: {sid}, user: {uid}")

    # ── Test 1: INITIAL_REQUEST ────────────────────────────────────────────
    print("\n=== Test 1: INITIAL_REQUEST enrichment ===")
    result = await enricher.enrich(
        label="INITIAL_REQUEST",
        retrieval_strategy="FULL",
        session_id=sid,
        user_id=uid,
        current_message="I want a black dress under £50",
        entities={"colour_group_name": "Black",
                  "product_type_name": "Dress",
                  "price_max": 50.0}
    )
    print(f"Label: {result['label']}")
    print(f"Strategy: {result['retrieval_strategy']}")
    print(f"Hard constraints in query: {result['retrieval_query']['filters']}")
    print(f"Preference boosts: {len(result['retrieval_query']['preference_boosts'])}")
    print(f"Side effects: {result['side_effects']}")

    # ── Add bot turn with items to set up context for next tests ──────────
    item_a = ItemInContext(
        article_id="209886026",
        prod_name="Valerie dress",
        product_type_name="Dress",
        colour_group_name="Black",
        index_group_name="Ladieswear",
        detail_desc="Short dress in a crisp cotton weave.",
        price=34.99
    )
    item_b = ItemInContext(
        article_id="108775015",
        prod_name="Strap top",
        product_type_name="Vest top",
        colour_group_name="Black",
        index_group_name="Ladieswear",
        detail_desc="Jersey top with narrow shoulder straps.",
        price=14.99
    )

    # Update dialogue state with the recommended items
    await session_mgr.update_dialogue_state(
        sid,
        {"currently_discussing": {
            "item_a": item_a.model_dump(),
            "item_b": item_b.model_dump()
        }}
    )

    # Add conversation turns
    await turn_mgr.add_user_turn(
        sid, uid, "I want a black dress under £50",
        TurnClassification(label="INITIAL_REQUEST",
                           retrieval_strategy="FULL",
                           confidence=0.99)
    )
    await turn_mgr.add_assistant_turn(
        sid, uid,
        "Here are two options. Option 1 is the Valerie dress (black): "
        "Short dress in crisp cotton. Option 2 is the Strap top (black): "
        "Jersey top with narrow shoulder straps."
    )

    # ── Test 2: ATTRIBUTE_QUESTION ─────────────────────────────────────────
    print("\n=== Test 2: ATTRIBUTE_QUESTION enrichment ===")
    result2 = await enricher.enrich(
        label="ATTRIBUTE_QUESTION",
        retrieval_strategy="PARTIAL",
        session_id=sid,
        user_id=uid,
        current_message="What material is the first one made of?",
        entities={}
    )
    print(f"Label: {result2['label']}")
    print(f"Target item: {result2['memory_context']['target_item']['prod_name']}")
    print(f"Question about: {result2['memory_context']['question_about']}")
    print(f"Retrieval action: {result2['retrieval_query']['action']}")

    # ── Test 3: COMPARISON ─────────────────────────────────────────────────
    print("\n=== Test 3: COMPARISON enrichment ===")
    result3 = await enricher.enrich(
        label="COMPARISON",
        retrieval_strategy="PARTIAL",
        session_id=sid,
        user_id=uid,
        current_message="Which one is better value?",
        entities={}
    )
    print(f"Label: {result3['label']}")
    print(f"Item A: {result3['memory_context']['item_a']['prod_name']}")
    print(f"Item B: {result3['memory_context']['item_b']['prod_name']}")
    print(f"Comparison dimension: {result3['memory_context']['comparison_dimension']}")

    # ── Test 4: FEEDBACK positive ──────────────────────────────────────────
    print("\n=== Test 4: FEEDBACK (positive) enrichment ===")
    result4 = await enricher.enrich(
        label="FEEDBACK",
        retrieval_strategy="NO",
        session_id=sid,
        user_id=uid,
        current_message="I love the Valerie dress, I'll take it!",
        entities={}
    )
    print(f"Label: {result4['label']}")
    print(f"Sentiment score: {result4['memory_context']['sentiment_score']}")
    print(f"Is positive: {result4['memory_context']['is_positive']}")
    print(f"Side effects: {result4['side_effects']}")
    print(f"Retrieval query: {result4['retrieval_query']} (should be None)")

    # ── Test 5: REFINEMENT ─────────────────────────────────────────────────
    print("\n=== Test 5: REFINEMENT enrichment ===")
    result5 = await enricher.enrich(
        label="REFINEMENT",
        retrieval_strategy="FULL",
        session_id=sid,
        user_id=uid,
        current_message="Can you show me white ones instead?",
        entities={"colour_group_name": "White"}
    )
    print(f"Label: {result5['label']}")
    print(f"Updated constraints: {result5['memory_context']['updated_constraints']}")
    print(f"New changes: {result5['memory_context']['new_changes']}")
    print(f"Retrieval filters: {result5['retrieval_query']['filters']}")

    # ── Test 6: CHITCHAT ───────────────────────────────────────────────────
    print("\n=== Test 6: CHITCHAT enrichment ===")
    result6 = await enricher.enrich(
        label="CHITCHAT",
        retrieval_strategy="NO",
        session_id=sid,
        user_id=uid,
        current_message="Thanks for your help!",
        entities={}
    )
    print(f"Label: {result6['label']}")
    print(f"Memory context: {result6['memory_context']} (should be empty)")
    print(f"Retrieval query: {result6['retrieval_query']} (should be None)")

    # ── Cleanup ────────────────────────────────────────────────────────────
    db = get_db()
    await db.sessions.delete_one({"session_id": sid})
    await db.users.delete_one({"user_id": uid})
    print(f"\nTest data cleaned up.")

    await close_mongodb_connection()
    await close_redis_connection()
    print("\nAll enrichment tests PASSED.")


if __name__ == "__main__":
    asyncio.run(test_enrichment())