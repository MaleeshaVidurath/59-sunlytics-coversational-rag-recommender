# m3_implementation/test_result/hallucination_result/collect_cases.py
#
# Evaluation dataset driver — Step 1 of the hallucination checker evaluation.
#
# Drives the full pipeline (memory → CSE → evidence assembler → LLM →
# hallucination checker) with a scripted set of conversations, exactly as if
# a real user had typed them. The capture hook in rag_pipeline.py
# (EVAL_CAPTURE=1) saves every (evidence, response, check) pair to
# captured_cases.jsonl — the raw material for the labeled test set.
#
# The conversations are designed for COVERAGE: every checkable action
# (catalog_search, item_attribute_lookup, item_detail_lookup, item_compare,
# explanation_generate) appears multiple times, including the multi-item
# catalog case that the item→sentence lock map targets.
#
# Requires the full stack running: MongoDB, Redis, PostgreSQL, Qdrant,
# and the LLM provider (Groq or Ollama) configured in .env.
#
# Run from m3_implementation folder:
#   python test_result/hallucination_result/collect_cases.py

import asyncio
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Enable the capture hook BEFORE the pipeline runs
os.environ["EVAL_CAPTURE"] = "1"

from dotenv import load_dotenv
load_dotenv()

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from memory.db.redis_client import connect_to_redis, close_redis_connection, get_redis
from memory.core.pipeline import MemoryPipeline
from memory.core.user_manager import UserManager
from text_rag.db.postgres_client import create_schema, close_pool
from text_rag.db.qdrant_client import get_qdrant
from text_rag.core.rag_pipeline import TextRAGPipeline
from test_result.hallucination_result.capture import capture_path

# Same customer as test_full_pipeline.py — known-good profile
EVAL_CUSTOMER_ID = "be1981ab818cf4ef6765b2ecaea7a2cbf14ccd6e8a7ee985513d9e8e53c6d91b"

# Pause between turns so the run stays under Groq's free-tier 6000
# tokens/minute limit (each turn spends several Groq calls: guard judge,
# generation, retries). Override with EVAL_TURN_DELAY.
TURN_DELAY_SECONDS = float(os.getenv("EVAL_TURN_DELAY", "25"))

# ── Scripted conversations ──────────────────────────────────────────────────
# Each inner list is one session, run in order on a fresh session_id.
# Comments show the action each turn is expected to trigger.
CONVERSATIONS = [
    # catalog_search + attribute + compare + explanation
    ["I want a black dress under £50",              # catalog_search
     "What material is the first one made of?",     # item_attribute_lookup
     "Which one is cheaper?",                       # item_compare
     "Why did you recommend the first one?"],       # explanation_generate

    # catalog_search + compare
    ["Show me red summer dresses",                  # catalog_search
     "Compare option 1 and option 2"],              # item_compare

    # catalog_search + refinement (second catalog_search)
    ["I'm looking for white t-shirts for men",      # catalog_search
     "Something cheaper instead please"],           # catalog_search (refinement)

    # catalog_search + attribute
    ["I need a black bra under £20",                # catalog_search
     "What material is the first one?"],            # item_attribute_lookup

    # catalog_search + attribute
    ["Show me blue jeans",                          # catalog_search
     "Do they have pockets?"],                      # item_attribute_lookup

    # catalog_search + refinement + explanation
    ["I want a warm winter jacket",                 # catalog_search
     "Show me something in black instead",          # catalog_search (refinement)
     "Why did you pick the first one?"],            # explanation_generate

    # catalog_search + compare
    ["Casual shirts for the office please",         # catalog_search
     "Which one is better for a job interview?"],   # item_compare

    # catalog_search + detail lookup
    ["Sneakers under £40",                          # catalog_search
     "Tell me more about the second one"],          # item_detail_lookup

    # catalog_search + compare
    ["I'd like a pink skirt",                       # catalog_search
     "Compare the two options"],                    # item_compare

    # catalog_search + attribute
    ["Show me some hoodies",                        # catalog_search
     "What colour is the second one?"],             # item_attribute_lookup

    # catalog_search + explanation
    ["I need an elegant dress for a wedding",       # catalog_search
     "Why is this one a good choice for me?"],      # explanation_generate

    # catalog_search + refinement
    ["Sporty leggings for the gym",                 # catalog_search
     "Do you have cheaper ones instead?"],          # catalog_search (refinement)

    # catalog_search + attribute
    ["Beige trousers please",                       # catalog_search
     "What fabric are they made of?"],              # item_attribute_lookup

    # catalog_search + detail lookup
    ["I'm looking for socks",                       # catalog_search
     "Tell me more about the first one"],           # item_detail_lookup

    # catalog_search + compare + feedback
    ["Show me denim jackets",                       # catalog_search
     "Which of these two is better quality?",       # item_compare
     "I'll take the second one, thanks!"],          # FEEDBACK (not captured)

    # catalog_search + refinement
    ["I want a green blouse",                       # catalog_search
     "Can you show it in white instead?"],          # catalog_search (refinement)

    # ── Extension batch (2026-07-10): 44 further conversations for the
    # expanded dataset (target: 1000+ test rows). Same coverage goals,
    # wider product / colour / price / occasion variety.
    ["Show me black vest tops", "What fabric is the first one?"],
    ["I need beige sandals for the beach", "Which one is cheaper?",
     "Why did you choose the first one?"],
    ["Looking for a brown handbag", "Tell me more about the first one"],
    ["I want a warm scarf for winter", "Why did you recommend the first one?"],
    ["Show me grey cardigans under £30", "Do they have buttons?"],
    ["I need a dark blue blazer for a job interview", "Compare the two options"],
    ["Black leggings for yoga please", "What material are they made of?"],
    ["Show me white socks", "Something cheaper instead"],
    ["I'm looking for brown boots", "Tell me more about the second one"],
    ["I need blue jeans under £25", "Which one is better for casual wear?",
     "Show me black ones instead"],
    ["Show me a yellow summer skirt", "Why is this a good choice for me?"],
    ["I want a red hoodie", "What pattern does the first one have?"],
    ["Grey sweaters for men please", "Compare option 1 and option 2",
     "What material is the first one?"],
    ["I need an orange t-shirt", "Show me something in purple instead"],
    ["Show me black formal trousers", "What is the price of the second one?"],
    ["I'm looking for a pink party dress", "Which of these is more elegant?"],
    ["White shirts for the office", "What fabric is the second one?"],
    ["I want a long winter coat under £60", "Tell me more about the first one",
     "Which one is more affordable?"],
    ["Show me dark blue shorts", "Why did you pick the second one?"],
    ["I need a light blue blouse", "What pattern does it have?"],
    ["Beige trousers for a smart casual look", "Which one is better quality?"],
    ["Show me green t-shirts under £15", "Something in dark green instead"],
    ["I need black tights", "What material is the first one?"],
    ["Show me red dresses for a wedding", "Compare the two",
     "Tell me more about the second one"],
    ["I'm looking for a white bra", "Which one is more comfortable?"],
    ["Grey hoodies under £20", "Tell me more about the first one"],
    ["I need brown trousers", "What is the price difference between them?"],
    ["Show me purple tops", "Why would the first one suit me?"],
    ["I want black sneakers for running", "What are they made of?"],
    ["Light pink cardigan please", "Show me a darker colour instead"],
    ["I need a denim skirt", "Tell me more about the second one"],
    ["Show me men's black jackets", "Which one is warmer?",
     "Show me something cheaper"],
    ["I want a yellow jacket for rainy days", "Is the first one waterproof?"],
    ["I'm looking for a patterned blouse", "What colour is the second one?"],
    ["Show me children's t-shirts", "Something cheaper please"],
    ["I need white trainers under £30", "Compare option 1 and option 2"],
    ["Show me a black leather bag", "Tell me more about the first one"],
    ["I want a soft sweater to wear at home", "What fabric is it made of?"],
    ["Show me winter hats", "Which one is cheaper?"],
    ["I need blue shorts for the beach", "Why the first one?"],
    ["I'm looking for an elegant black blazer", "Compare the two options"],
    ["Show me striped t-shirts", "What colour is the first one?"],
    ["I need warm leggings for winter", "Which one is warmer?"],
    ["Show me dark red cardigans", "Something under £25 instead"],
]


async def _clear_user_sessions(db, redis, user_id: str) -> int:
    """Deletes all sessions for the eval user from MongoDB and Redis so every
    run starts from clean context (same pattern as test_full_pipeline.py)."""
    existing = await db.sessions.find({"user_id": user_id}).to_list(length=200)
    for sess in existing:
        sid = sess.get("session_id", "")
        if sid:
            await redis.delete(f"session:{sid}:turns")
            await redis.delete(f"session:{sid}:state")
            await redis.delete(f"session:{sid}")
    await redis.delete(f"user:{user_id}:active_session")
    deleted = await db.sessions.delete_many({"user_id": user_id})
    return deleted.deleted_count


def _rotate_capture_file() -> None:
    """Renames an existing capture file so each run produces a fresh dataset."""
    path = capture_path()
    if os.path.exists(path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.replace(".jsonl", f"_{stamp}.jsonl")
        os.rename(path, backup)
        print(f"Existing capture file rotated to: {os.path.basename(backup)}")


async def run():
    print("=" * 60)
    print("EVALUATION CASE COLLECTION")
    print(f"Conversations: {len(CONVERSATIONS)}  "
          f"Turns: {sum(len(c) for c in CONVERSATIONS)}")
    print("=" * 60)

    _rotate_capture_file()

    await connect_to_mongodb()
    await connect_to_redis()
    await create_schema()
    get_qdrant()

    memory_pipeline = MemoryPipeline()
    rag_pipeline    = TextRAGPipeline()

    user_mgr = UserManager()
    user     = await user_mgr.get_or_create_user(customer_id=EVAL_CUSTOMER_ID)
    user_id  = user.user_id
    print(f"Eval user: {user_id}")

    db    = get_db()
    redis = await get_redis()
    deleted = await _clear_user_sessions(db, redis, user_id)
    print(f"Cleared {deleted} old sessions.\n")

    turns_run     = 0
    turns_failed  = 0
    eval_sessions = []

    for conv_idx, conversation in enumerate(CONVERSATIONS, start=1):
        session_id = f"eval_{uuid.uuid4().hex[:8]}"
        eval_sessions.append(session_id)
        print(f"\n{'─'*60}")
        print(f"Conversation {conv_idx}/{len(CONVERSATIONS)}  session={session_id}")

        for turn_idx, message in enumerate(conversation, start=1):
            print(f"\n[{conv_idx}.{turn_idx}] USER: {message}")
            try:
                pipeline_output = await memory_pipeline.process_turn(
                    user_id=user_id,
                    message=message,
                    session_id=session_id,
                    customer_id=EVAL_CUSTOMER_ID,
                )
                session_id = pipeline_output["session_id"]

                rag_result = await rag_pipeline.process(
                    pipeline_output=pipeline_output,
                    memory_pipeline=memory_pipeline,
                    store_response=True,
                )
                turns_run += 1
                print(f"[{conv_idx}.{turn_idx}] label={pipeline_output.get('label')} "
                      f"action={rag_result.get('action')} "
                      f"attempts={rag_result.get('attempt_count')} "
                      f"hall_flag={rag_result.get('hallucination_flag')}")
                print(f"[{conv_idx}.{turn_idx}] BOT: {rag_result.get('response_text','')[:120]}")
            except Exception as e:
                turns_failed += 1
                print(f"[{conv_idx}.{turn_idx}] TURN FAILED (continuing): {e}")

            # Rate-limit pacing (see TURN_DELAY_SECONDS)
            if TURN_DELAY_SECONDS > 0:
                await asyncio.sleep(TURN_DELAY_SECONDS)

        # Reset session context between conversations so the next one starts clean
        await redis.delete(f"session:{session_id}:turns")
        await redis.delete(f"session:{session_id}:state")
        await redis.delete(f"session:{session_id}")
        await redis.delete(f"user:{user_id}:active_session")

    # ── Summary ─────────────────────────────────────────────────────────────
    path = capture_path()
    n_captured = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            n_captured = sum(1 for line in f if line.strip())

    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Turns run:       {turns_run}  (failed: {turns_failed})")
    print(f"Cases captured:  {n_captured}")
    print(f"Capture file:    {path}")
    print("Next step: corrupt_cases.py builds the labeled test set from this file.")

    # Keep eval sessions in MongoDB for traceability; clean-up is manual:
    #   db.sessions.delete_many({"session_id": {"$regex": "^eval_"}})

    await close_mongodb_connection()
    await close_redis_connection()
    await close_pool()
    print("All connections closed.")


if __name__ == "__main__":
    asyncio.run(run())
