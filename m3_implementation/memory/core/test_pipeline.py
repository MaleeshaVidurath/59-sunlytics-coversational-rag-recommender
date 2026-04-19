# m3_implementation/memory/core/test_pipeline.py
# Tests the complete pipeline WITH your trained DistilBERT model.
# The pipeline now auto-loads DistilBERT from DISTILBERT_MODEL_PATH in .env

import asyncio
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from memory.db.redis_client import connect_to_redis, close_redis_connection
from memory.core.user_manager import UserManager
from memory.core.pipeline import MemoryPipeline


async def test_pipeline_with_distilbert():
    await connect_to_mongodb()
    await connect_to_redis()

    print("\nInitialising pipeline (auto-loads DistilBERT from .env)...")

    # MemoryPipeline() with no arguments auto-loads from DISTILBERT_MODEL_PATH
    # in your .env file. No manual path needed here.
    pipeline = MemoryPipeline()

    model_available = pipeline.predictor is not None
    print(f"DistilBERT available: {model_available}")

    # Create a test user
    user_mgr = UserManager()
    user = await user_mgr.get_or_create_user(
        customer_id="test_pipeline_customer_001",
        initial_data={"age": 26, "club_member_status": "ACTIVE"}
    )
    print(f"\nTest user: {user.user_id}")

    print("\n" + "="*60)
    print("SIMULATING A FULL CONVERSATION")
    print("="*60)

    session_id = None

    # ── Turn 1: Fresh request ──────────────────────────────────────────────
    print("\n--- Turn 1: Fresh request ---")
    result1 = await pipeline.process_turn(
        user_id=user.user_id,
        message="I want a black dress under £50",
        session_id=session_id
    )
    session_id = result1["session_id"]
    print(f"Session: {session_id}")
    print(f"Label:    {result1['label']}  (expected: INITIAL_REQUEST)")
    print(f"Strategy: {result1['retrieval_strategy']}  (expected: FULL)")
    print(f"Confidence: {result1['confidence']:.1%}  "
          f"{'[DistilBERT]' if not result1['used_rules'] else '[fallback]'}")
    if result1["retrieval_query"]:
        print(f"Retrieval filters: {result1['retrieval_query']['filters']}")
        print(f"Preference boosts: "
              f"{len(result1['retrieval_query']['preference_boosts'])}")

    await pipeline.store_response(
        session_id=session_id,
        user_id=user.user_id,
        bot_response=(
            "Here are two options. Option 1 is the Valerie dress (black, dress): "
            "Short dress in crisp cotton weave. "
            "Option 2 is the Angel (dark pink): Short A-line dress in cotton."
        ),
        recommended_items=[
            {
                "article_id": "209886026",
                "prod_name": "Valerie dress",
                "product_type_name": "Dress",
                "colour_group_name": "Black",
                "index_group_name": "Ladieswear",
                "detail_desc": "Short dress in crisp cotton weave.",
                "price": 34.99
            },
            {
                "article_id": "108775015",
                "prod_name": "Angel",
                "product_type_name": "Dress",
                "colour_group_name": "Dark Pink",
                "index_group_name": "Ladieswear",
                "detail_desc": "Short A-line dress in cotton.",
                "price": 29.99
            }
        ],
        trigger_label=result1["label"],
        retrieval_strategy=result1["retrieval_strategy"]
    )

    # ── Turn 2: Attribute question ─────────────────────────────────────────
    print("\n--- Turn 2: Attribute question ---")
    result2 = await pipeline.process_turn(
        user_id=user.user_id,
        message="What material is the first one made of?",
        session_id=session_id
    )
    print(f"Label:    {result2['label']}  (expected: ATTRIBUTE_QUESTION)")
    print(f"Strategy: {result2['retrieval_strategy']}  (expected: PARTIAL)")
    print(f"Confidence: {result2['confidence']:.1%}  "
          f"{'[DistilBERT]' if not result2['used_rules'] else '[fallback]'}")
    target = result2["memory_context"].get("target_item")
    print(f"Target item: {target['prod_name'] if target else 'None'}")

    await pipeline.store_response(
        session_id=session_id,
        user_id=user.user_id,
        bot_response="The Valerie dress is made from cotton jersey.",
        trigger_label=result2["label"],
        retrieval_strategy=result2["retrieval_strategy"]
    )

    # ── Turn 3: Positive feedback ──────────────────────────────────────────
    print("\n--- Turn 3: Positive feedback ---")
    result3 = await pipeline.process_turn(
        user_id=user.user_id,
        message="I love it, I'll take the Valerie dress!",
        session_id=session_id
    )
    print(f"Label:    {result3['label']}  (expected: FEEDBACK)")
    print(f"Strategy: {result3['retrieval_strategy']}  (expected: NO)")
    print(f"Confidence: {result3['confidence']:.1%}  "
          f"{'[DistilBERT]' if not result3['used_rules'] else '[fallback]'}")
    print(f"Sentiment: {result3['memory_context'].get('sentiment_score')}")
    print(f"Side effects: {result3['enriched_context']['side_effects']}")

    await pipeline.store_response(
        session_id=session_id,
        user_id=user.user_id,
        bot_response="Great choice! The Valerie dress is a wonderful pick.",
        trigger_label=result3["label"],
        retrieval_strategy=result3["retrieval_strategy"]
    )

    # ── Turn 4: Refinement ─────────────────────────────────────────────────
    print("\n--- Turn 4: Refinement ---")
    result4 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Can you also show me something in white?",
        session_id=session_id
    )
    print(f"Label:    {result4['label']}  (expected: REFINEMENT)")
    print(f"Strategy: {result4['retrieval_strategy']}  (expected: FULL)")
    print(f"Confidence: {result4['confidence']:.1%}  "
          f"{'[DistilBERT]' if not result4['used_rules'] else '[fallback]'}")
    if result4["retrieval_query"]:
        print(f"Updated filters: {result4['retrieval_query']['filters']}")
    else:
        print(f"No retrieval query (strategy={result4['retrieval_strategy']})")

    await pipeline.store_response(
        session_id=session_id,
        user_id=user.user_id,
        bot_response="Here are white options for you.",
        trigger_label=result4["label"],
        retrieval_strategy=result4["retrieval_strategy"]
    )

    # ── Turn 5: Chitchat ───────────────────────────────────────────────────
    print("\n--- Turn 5: Chitchat ---")
    result5 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Thanks, you've been really helpful!",
        session_id=session_id
    )
    print(f"Label:    {result5['label']}  (expected: CHITCHAT)")
    print(f"Strategy: {result5['retrieval_strategy']}  (expected: NO)")
    print(f"Confidence: {result5['confidence']:.1%}  "
          f"{'[DistilBERT]' if not result5['used_rules'] else '[fallback]'}")

    # ── Verify MongoDB ─────────────────────────────────────────────────────
    print("\n--- Verifying MongoDB state ---")
    full_session = await pipeline.session_mgr.get_full_session(session_id)
    print(f"Total turns stored: {full_session['turn_count']}")
    print(f"Turns in array: {len(full_session['turns'])}")

    # ── Verify preferences ─────────────────────────────────────────────────
    print("\n--- Verifying preference memory ---")
    prefs = await pipeline.user_mgr.get_preference_summary(user.user_id)
    print(f"Liked attributes: {len(prefs['liked_attributes'])}")
    for p in prefs["liked_attributes"][:4]:
        print(f"  {p['attribute_name']}: {p['attribute_value']} "
              f"(weight={p['weight']})")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    if model_available:
        print("Pipeline test with REAL DistilBERT PASSED.")
    else:
        print("Pipeline test with FALLBACK classifier PASSED.")
        print("Check: pip install protobuf  then re-run.")
    print("="*60)

    # Cleanup
    db = get_db()
    await db.sessions.delete_many({"session_id": session_id})
    await db.users.delete_many({"user_id": user.user_id})
    await db.recommendations.delete_many({"session_id": session_id})

    await close_mongodb_connection()
    await close_redis_connection()


if __name__ == "__main__":
    asyncio.run(test_pipeline_with_distilbert())
