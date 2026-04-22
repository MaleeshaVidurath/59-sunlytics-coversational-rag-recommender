# test_full_pipeline.py
# Full end-to-end test: memory pipeline + text RAG pipeline
# Run from m3_implementation folder: python test_full_pipeline.py

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from memory.db.redis_client import connect_to_redis, close_redis_connection
from memory.core.pipeline import MemoryPipeline
from text_rag.db.postgres_client import create_schema, close_pool
from text_rag.db.qdrant_client import get_qdrant
from text_rag.core.rag_pipeline import TextRAGPipeline


def print_turn(turn_num, user_msg, result):
    print(f"\n{'─'*60}")
    print(f"Turn {turn_num}")
    print(f"USER:  {user_msg}")
    print(f"LABEL: {result.get('label')} "
          f"({result.get('retrieval_strategy')}) "
          f"conf: {result.get('confidence', 0):.1%}")

    ri = result.get("retrieval_input")
    if ri:
        print(f"ACTION: {ri['action']}")
        payload = ri.get("payload", {})
        if payload.get("filters"):
            print(f"FILTERS: {payload['filters']}")
        if payload.get("soft_constraints"):
            print(f"SOFT:    {payload['soft_constraints']}")

    rag = result.get("rag_result", {})
    response  = rag.get("response_text", "")
    hall_flag = rag.get("hallucination_flag", False)
    attempts  = rag.get("attempt_count", 1)
    items     = rag.get("items_recommended", [])

    if items:
        print(f"ITEMS:  {[i.get('name','?') for i in items]}")

    print(f"\nBOT:   {response}")

    if hall_flag:
        print(f"  ⚠  HALLUCINATION FLAG (after {attempts} attempts)")
        for s in rag.get("flagged_sentences", [])[:2]:
            print(f"     Flagged: {s.get('sentence','')[:80]}")
    else:
        print(f"  ✓  Hallucination check passed (attempt {attempts})")


async def run_conversation():
    print("=" * 60)
    print("FULL END-TO-END PIPELINE TEST")
    print("Memory Pipeline + Text RAG + Hallucination Check")
    print("=" * 60)

    await connect_to_mongodb()
    await connect_to_redis()
    await create_schema()
    get_qdrant()

    memory_pipeline = MemoryPipeline()
    rag_pipeline    = TextRAGPipeline()

    customer_id = "be1981ab818cf4ef6765b2ecaea7a2cbf14ccd6e8a7ee985513d9e8e53c6d91b"

    from memory.core.user_manager import UserManager
    user_mgr = UserManager()
    user     = await user_mgr.get_or_create_user(customer_id=customer_id)
    user_id  = user.user_id
    print(f"\nCustomer: {customer_id[:20]}...")
    print(f"User ID:  {user_id}")

    # ── IMPORTANT: Force a completely fresh session ────────────────────────
    # Always start with session_id=None so a NEW session is created.
    # Clear BOTH MongoDB sessions AND Redis cache to avoid stale DistilBERT context.
    db = get_db()

    # FULL CLEANUP: Delete ALL sessions for this user from both MongoDB and Redis
    # Must DELETE (not just expire) because get_or_create_session finds sessions
    # by user_id regardless of status, causing stale context for DistilBERT.
    from memory.db.redis_client import get_redis
    redis = await get_redis()

    existing_sessions = await db.sessions.find(
        {"user_id": user_id}
    ).to_list(length=100)

    for sess in existing_sessions:
        sid = sess.get("session_id", "")
        if sid:
            await redis.delete(f"session:{sid}:turns")
            await redis.delete(f"session:{sid}:state")
            await redis.delete(f"session:{sid}")

    # CRITICAL: Also delete the user-level active session pointer in Redis
    # This is the key that get_or_create_session checks FIRST
    # Format: "user:{user_id}:active_session" (from session_manager._active_session_key)
    await redis.delete(f"user:{user_id}:active_session")

    # Delete from MongoDB entirely
    deleted = await db.sessions.delete_many({"user_id": user_id})
    print(f"Deleted {deleted.deleted_count} old sessions. "
          f"Redis fully cleared (including active_session pointer). Starting fresh.")

    # Force a guaranteed fresh session by passing a unique ID
    # This bypasses get_or_create_session lookup entirely
    import uuid as _uuid
    session_id = f"test_{_uuid.uuid4().hex[:8]}"
    print(f"Forced fresh session ID: {session_id}")

    turns = [
        ("1", "I want a black dress under £50"),
        ("2", "What material is the first one made of?"),
        ("3", "Which one is cheaper?"),
        ("4", "I love it, I will take the first one!"),
        ("5", "Can you show me something in white instead?"),
        ("6", "Thanks, that is really helpful!"),
    ]

    results     = []
    new_session = None

    for turn_num, message in turns:
        pipeline_output = await memory_pipeline.process_turn(
            user_id=user_id,
            message=message,
            session_id=session_id,
            customer_id=customer_id,
        )
        # After first turn, use the created session consistently
        session_id  = pipeline_output["session_id"]
        new_session = session_id

        rag_result = await rag_pipeline.process(
            pipeline_output=pipeline_output,
            memory_pipeline=memory_pipeline,
            store_response=True,
        )
        pipeline_output["rag_result"] = rag_result
        print_turn(turn_num, message, pipeline_output)
        results.append(pipeline_output)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Session ID: {new_session}")
    print(f"Turns processed: {len(results)}")

    labels_seen = [(r['label'], r['retrieval_strategy']) for r in results]
    print(f"Labels: {labels_seen}")

    hallucinated = [r for r in results
                    if r.get("rag_result", {}).get("hallucination_flag")]
    print(f"Hallucination flags: {len(hallucinated)}/{len(results)}")

    session = await db.sessions.find_one({"session_id": new_session})
    if session:
        print(f"Turns in MongoDB: {session.get('turn_count', 0)}")

    prefs = await user_mgr.get_preference_summary(user_id)
    print(f"Preferences stored: {len(prefs.get('liked_attributes', []))}")

    # Clean up test session
    if new_session:
        await db.sessions.delete_many({"session_id": new_session})
    print("\nTest session cleaned up.")

    await close_mongodb_connection()
    await close_redis_connection()
    await close_pool()
    print("All connections closed.")
    print("End-to-end test complete.")


if __name__ == "__main__":
    asyncio.run(run_conversation())
