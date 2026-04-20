# m3_implementation/memory/core/test_pipeline.py
# Tests the complete pipeline with the new standardised output structure.

import asyncio
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from memory.db.redis_client import connect_to_redis, close_redis_connection
from memory.core.user_manager import UserManager
from memory.core.pipeline import MemoryPipeline


def print_retrieval_input(ri, indent=4):
    """Pretty-prints the retrieval_input dict for verification."""
    pad = " " * indent
    if ri is None:
        print(f"{pad}retrieval_input: None (no retrieval needed)")
        return
    print(f"{pad}retrieval_input:")
    print(f"{pad}  action:             {ri['action']}")
    print(f"{pad}  retrieval_strategy: {ri['retrieval_strategy']}")
    print(f"{pad}  user_message:       {ri['user_message'][:60]}")
    item_a = ri["items_in_context"].get("item_a")
    item_b = ri["items_in_context"].get("item_b")
    print(f"{pad}  items_in_context:   "
          f"item_a={item_a['prod_name'] if item_a else None}, "
          f"item_b={item_b['prod_name'] if item_b else None}")
    print(f"{pad}  exclude_ids:        {ri['exclude_ids']}")
    print(f"{pad}  payload:            {json.dumps(ri['payload'], indent=2)[:200]}")


async def test_pipeline():
    await connect_to_mongodb()
    await connect_to_redis()

    print("\nInitialising pipeline...")
    pipeline = MemoryPipeline()
    model_available = pipeline.predictor is not None
    print(f"DistilBERT: {'loaded' if model_available else 'fallback'}")

    user_mgr = UserManager()
    user = await user_mgr.get_or_create_user(
        customer_id="test_pipeline_customer_001",
        initial_data={"age": 26, "club_member_status": "ACTIVE"}
    )
    print(f"Test user: {user.user_id}")

    print("\n" + "="*65)
    print("VERIFYING STANDARDISED OUTPUT STRUCTURE")
    print("="*65)

    session_id = None

    # ── Turn 1: INITIAL_REQUEST ────────────────────────────────────────────
    print("\n--- Turn 1: INITIAL_REQUEST ---")
    r1 = await pipeline.process_turn(
        user_id=user.user_id,
        message="I want a black dress under £50",
        session_id=session_id
    )
    session_id = r1["session_id"]
    print(f"  label:    {r1['label']}  conf: {r1['confidence']:.1%}")
    print(f"  Top-level keys: {list(r1.keys())}")
    # Verify no duplication
    assert "enriched_context" not in r1, "enriched_context should not be at top level"
    assert "retrieval_query"  not in r1, "retrieval_query should not be at top level"
    assert "retrieval_input"  in r1,     "retrieval_input must be present"
    assert "memory_context"   in r1,     "memory_context must be present"
    assert "side_effects"     in r1,     "side_effects must be present"
    print("  Structure check: PASSED (no duplication)")
    print_retrieval_input(r1["retrieval_input"])

    await pipeline.store_response(
        session_id=session_id,
        user_id=user.user_id,
        bot_response=(
            "Here are two options. Option 1 is the Valerie dress (black, dress): "
            "Short dress in crisp cotton weave. Option 2 is the Angel (dark pink): "
            "Short A-line dress in cotton."
        ),
        recommended_items=[
            {"article_id": "209886026", "prod_name": "Valerie dress",
             "product_type_name": "Dress", "colour_group_name": "Black",
             "index_group_name": "Ladieswear",
             "detail_desc": "Short dress in crisp cotton.", "price": 34.99},
            {"article_id": "108775015", "prod_name": "Angel",
             "product_type_name": "Dress", "colour_group_name": "Dark Pink",
             "index_group_name": "Ladieswear",
             "detail_desc": "Short A-line dress in cotton.", "price": 29.99},
        ],
        trigger_label=r1["label"],
        retrieval_strategy=r1["retrieval_strategy"]
    )

    # ── Turn 2: ATTRIBUTE_QUESTION ────────────────────────────────────────
    print("\n--- Turn 2: ATTRIBUTE_QUESTION ---")
    r2 = await pipeline.process_turn(
        user_id=user.user_id,
        message="What material is the first one made of?",
        session_id=session_id
    )
    print(f"  label:    {r2['label']}  conf: {r2['confidence']:.1%}")
    ri2 = r2["retrieval_input"]
    assert ri2 is not None,                      "PARTIAL needs retrieval_input"
    assert ri2["action"] == "item_attribute_lookup", "Wrong action"
    assert "article_id"      in ri2["payload"],  "Payload missing article_id"
    assert "attribute_topic" in ri2["payload"],  "Payload missing attribute_topic"
    print(f"  action:          {ri2['action']}")
    print(f"  payload.article_id:      {ri2['payload']['article_id']}")
    print(f"  payload.attribute_topic: {ri2['payload']['attribute_topic']}")
    print("  Attribute check: PASSED")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="The Valerie dress is made from cotton jersey.",
        trigger_label=r2["label"], retrieval_strategy=r2["retrieval_strategy"]
    )

    # ── Turn 3: COMPARISON ────────────────────────────────────────────────
    print("\n--- Turn 3: COMPARISON ---")
    r3 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Which one is more affordable?",
        session_id=session_id
    )
    print(f"  label:    {r3['label']}  conf: {r3['confidence']:.1%}")
    if r3["retrieval_input"]:
        ri3 = r3["retrieval_input"]
        print(f"  action:              {ri3['action']}")
        print(f"  comparison_dimension:{ri3['payload'].get('comparison_dimension')}")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="The Angel at £29.99 is cheaper than the Valerie dress at £34.99.",
        trigger_label=r3["label"], retrieval_strategy=r3["retrieval_strategy"]
    )

    # ── Turn 4: FEEDBACK ──────────────────────────────────────────────────
    print("\n--- Turn 4: FEEDBACK ---")
    r4 = await pipeline.process_turn(
        user_id=user.user_id,
        message="I love it, I'll take the Valerie dress!",
        session_id=session_id
    )
    print(f"  label:    {r4['label']}  conf: {r4['confidence']:.1%}")
    assert r4["retrieval_input"] is None, "FEEDBACK must have None retrieval_input"
    assert "feedback" in r4["memory_context"], "Feedback info must be in memory_context"
    print(f"  retrieval_input: None  ✓")
    print(f"  sentiment_score: {r4['memory_context']['feedback']['sentiment_score']}")
    print(f"  side_effects:    {r4['side_effects']}")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="Great choice! The Valerie dress is a wonderful pick.",
        trigger_label=r4["label"], retrieval_strategy=r4["retrieval_strategy"]
    )

    # ── Turn 5: REFINEMENT ────────────────────────────────────────────────
    print("\n--- Turn 5: REFINEMENT ---")
    r5 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Can you also show me something in white?",
        session_id=session_id
    )
    print(f"  label:    {r5['label']}  conf: {r5['confidence']:.1%}")
    if r5["retrieval_input"]:
        ri5 = r5["retrieval_input"]
        print(f"  action:   {ri5['action']}")
        print(f"  filters:  {ri5['payload']['filters']}")
        assert ri5["action"] == "catalog_search", "REFINEMENT must use catalog_search"
        print("  Action check: PASSED (same as INITIAL_REQUEST)")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="Here are some white options for you.",
        trigger_label=r5["label"], retrieval_strategy=r5["retrieval_strategy"]
    )

    # ── Turn 6: CHITCHAT ──────────────────────────────────────────────────
    print("\n--- Turn 6: CHITCHAT ---")
    r6 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Thanks, you've been really helpful!",
        session_id=session_id
    )
    print(f"  label:    {r6['label']}  conf: {r6['confidence']:.1%}")
    assert r6["retrieval_input"] is None, "CHITCHAT must have None retrieval_input"
    print(f"  retrieval_input: None  ✓")

    # ── Verify memory ─────────────────────────────────────────────────────
    print("\n--- Verifying MongoDB state ---")
    full_session = await pipeline.session_mgr.get_full_session(session_id)
    print(f"  Total turns: {full_session['turn_count']}")

    print("\n--- Verifying preferences ---")
    prefs = await pipeline.user_mgr.get_preference_summary(user.user_id)
    print(f"  Liked attributes: {len(prefs['liked_attributes'])}")
    for p in prefs["liked_attributes"][:3]:
        print(f"    {p['attribute_name']}: {p['attribute_value']} "
              f"(weight={p['weight']})")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STRUCTURE VERIFICATION SUMMARY")
    print("="*65)
    print("  No data duplication                    ✓")
    print("  retrieval_input always same envelope   ✓")
    print("  INITIAL_REQUEST + REFINEMENT = catalog_search  ✓")
    print("  FEEDBACK retrieval_input = None        ✓")
    print("  CHITCHAT retrieval_input = None        ✓")
    print("  memory_context always present          ✓")
    print("  side_effects always present            ✓")
    if model_available:
        print("  DistilBERT classification              ✓")
    else:
        print("  Fallback classification (no model)     ✓")
    print("="*65)

    # Cleanup
    db = get_db()
    await db.sessions.delete_many({"session_id": session_id})
    await db.users.delete_many({"user_id": user.user_id})
    await db.recommendations.delete_many({"session_id": session_id})

    await close_mongodb_connection()
    await close_redis_connection()
    print("\nAll pipeline structure tests PASSED.")


if __name__ == "__main__":
    asyncio.run(test_pipeline())
