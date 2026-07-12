# m3_implementation/test_result/contradiction_result/collect_sessions.py
#
# Session collection driver — Step 1 of the contradiction detector evaluation
# (see EVALUATION_PLAN.md, section 2, Experiment A1).
#
# Drives the full pipeline (memory → CSE → evidence assembler → LLM →
# hallucination checker → CONTRADICTION DETECTOR) with scripted multi-turn
# conversations, exactly as if a real user had typed them. The capture hook
# inside contradiction_detector.py (CONTRA_EVAL_CAPTURE=1) saves one record
# per checked turn to captured_sessions.jsonl — including the session-graph
# state before the turn, which is what makes cross-turn distance measurable.
#
# DESIGN DIFFERENCE vs the hallucination driver (collect_cases.py):
# conversations here are LONGER (5–7 turns) and keep returning to products
# introduced in turn 1, so the same product is re-mentioned at ordinal
# distance 1, 2, 3, 4+ from where its ground truth entered the graph.
# Several sessions also run a mid-session refinement (a second catalog
# search) so the graph accumulates products from more than one turn.
#
# Requires the full stack running: MongoDB, Redis, PostgreSQL, Qdrant,
# and the LLM provider (Groq or Ollama) configured in .env.
#
# Run from the m3_implementation folder:
#   python test_result/contradiction_result/collect_sessions.py

import asyncio
import os
import sys
import uuid
from datetime import datetime

# Force UTF-8 stdout/stderr so box-drawing characters in progress logs do not
# crash on a Windows cp1252 console (PowerShell default codepage).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Enable the contradiction capture hook BEFORE the pipeline modules import
os.environ["CONTRA_EVAL_CAPTURE"] = "1"

from dotenv import load_dotenv
load_dotenv()

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from memory.db.redis_client import connect_to_redis, close_redis_connection, get_redis
from memory.core.pipeline import MemoryPipeline
from memory.core.user_manager import UserManager
from text_rag.db.postgres_client import create_schema, close_pool
from text_rag.db.qdrant_client import get_qdrant
from text_rag.core.rag_pipeline import TextRAGPipeline
from test_result.contradiction_result.capture import capture_path

# Same known-good customer profile as the hallucination evaluation
EVAL_CUSTOMER_ID = "be1981ab818cf4ef6765b2ecaea7a2cbf14ccd6e8a7ee985513d9e8e53c6d91b"

# Pause between turns to stay under Groq's free-tier tokens/minute limit
# (each turn spends several Groq calls: guard judge, generation, claim
# extraction). Override with EVAL_TURN_DELAY.
TURN_DELAY_SECONDS = float(os.getenv("EVAL_TURN_DELAY", "25"))

# Limit the number of conversations for smoke testing (0 = run all).
MAX_CONVERSATIONS = int(os.getenv("EVAL_MAX_CONVERSATIONS", "0"))

# ── Scripted conversations ──────────────────────────────────────────────────
# Each inner list is one session, run in order on a fresh session_id.
# Turn 1 always triggers catalog_search (products enter the graph);
# follow-up turns re-mention those products at growing turn distance.
# Phrasings are taken from patterns proven to route correctly in the
# hallucination-eval driver.
CONVERSATIONS = [
    # ── Long single-search sessions: distance grows every turn ──────────
    ["I want a black dress under £50",
     "What material is the first one made of?",
     "Which one is cheaper?",
     "Tell me more about the second one",
     "Why did you recommend the first one?",
     "What colour is the first one?"],

    ["Show me red summer dresses",
     "Compare option 1 and option 2",
     "Tell me more about the first one",
     "What is the price of the second one?",
     "Why did you pick the first one?"],

    ["I need a black bra under £20",
     "What material is the first one?",
     "Tell me more about the second one",
     "Which one is cheaper?",
     "Why did you recommend the first one?",
     "What colour is the second one?"],

    ["Show me blue jeans",
     "Do they have pockets?",
     "Tell me more about the first one",
     "What is the price of the first one?",
     "Compare option 1 and option 2",
     "Why did you pick the second one?"],

    ["Casual shirts for the office please",
     "Which one is better for a job interview?",
     "What material is the second one?",
     "Tell me more about the first one",
     "What is the price of the second one?"],

    ["Sneakers under £40",
     "Tell me more about the second one",
     "What material is the first one?",
     "Which one is cheaper?",
     "Why did you recommend the second one?"],

    ["I'd like a pink skirt",
     "Compare the two options",
     "What colour is the first one?",
     "Tell me more about the second one",
     "What is the price of the first one?"],

    ["Show me some hoodies",
     "What colour is the second one?",
     "Tell me more about the first one",
     "Compare option 1 and option 2",
     "Why did you recommend the first one?",
     "What is the price of the second one?"],

    ["I need an elegant dress for a wedding",
     "Why is this one a good choice for me?",
     "What material is the first one?",
     "Tell me more about the second one",
     "Which one is cheaper?"],

    ["Beige trousers please",
     "What fabric are they made of?",
     "Compare option 1 and option 2",
     "Tell me more about the first one",
     "What colour is the second one?"],

    ["I'm looking for socks",
     "Tell me more about the first one",
     "What is the price of the second one?",
     "Which one is cheaper?",
     "Why did you recommend the first one?"],

    ["Show me denim jackets",
     "Which of these two is better quality?",
     "What colour is the first one?",
     "Tell me more about the second one",
     "What is the price of the first one?"],

    ["Show me black vest tops",
     "What fabric is the first one?",
     "Tell me more about the second one",
     "Compare the two options",
     "Why did you pick the first one?"],

    ["I need beige sandals for the beach",
     "Which one is cheaper?",
     "Why did you choose the first one?",
     "Tell me more about the second one",
     "What colour is the first one?"],

    ["Looking for a brown handbag",
     "Tell me more about the first one",
     "What is the price of the second one?",
     "Compare option 1 and option 2",
     "Why did you recommend the second one?"],

    ["Show me grey cardigans under £30",
     "Do they have buttons?",
     "Tell me more about the first one",
     "What colour is the second one?",
     "Which one is cheaper?"],

    ["I need a dark blue blazer for a job interview",
     "Compare the two options",
     "What material is the first one?",
     "Tell me more about the second one",
     "Why did you pick the first one?"],

    ["Black leggings for yoga please",
     "What material are they made of?",
     "Tell me more about the first one",
     "What is the price of the second one?",
     "Which one is better for the gym?"],

    ["I'm looking for brown boots",
     "Tell me more about the second one",
     "What is the price of the first one?",
     "Compare option 1 and option 2",
     "Why did you recommend the first one?"],

    ["Show me a yellow summer skirt",
     "Why is this a good choice for me?",
     "What material is the first one?",
     "Tell me more about the second one",
     "What colour is the first one?"],

    ["I want a red hoodie",
     "What pattern does the first one have?",
     "Tell me more about the second one",
     "Which one is cheaper?",
     "What colour is the second one?"],

    ["Grey sweaters for men please",
     "Compare option 1 and option 2",
     "What material is the first one?",
     "Tell me more about the second one",
     "Why did you pick the second one?",
     "What is the price of the first one?"],

    ["Show me black formal trousers",
     "What is the price of the second one?",
     "Tell me more about the first one",
     "Compare the two options",
     "What colour is the first one?"],

    ["I'm looking for a pink party dress",
     "Which of these is more elegant?",
     "What material is the second one?",
     "Tell me more about the first one",
     "Why did you recommend the first one?"],

    ["White shirts for the office",
     "What fabric is the second one?",
     "Tell me more about the first one",
     "Which one is cheaper?",
     "Why did you pick the first one?"],

    ["I want a long winter coat under £60",
     "Tell me more about the first one",
     "Which one is more affordable?",
     "What material is the second one?",
     "Why did you recommend the first one?"],

    ["I need a light blue blouse",
     "What pattern does it have?",
     "Tell me more about the first one",
     "What is the price of the second one?",
     "Compare option 1 and option 2"],

    ["Show me red dresses for a wedding",
     "Compare the two",
     "Tell me more about the second one",
     "What material is the first one?",
     "Why did you pick the second one?",
     "What colour is the first one?"],

    ["Grey hoodies under £20",
     "Tell me more about the first one",
     "What is the price of the second one?",
     "Which one is warmer?",
     "Why did you recommend the first one?"],

    ["I want black sneakers for running",
     "What are they made of?",
     "Tell me more about the second one",
     "Which one is cheaper?",
     "What colour is the first one?"],

    # ── Refinement sessions: second catalog_search mid-session grows the
    #    graph, then follow-ups return to the (new) result set ────────────
    ["I'm looking for white t-shirts for men",
     "Something cheaper instead please",
     "What material is the first one?",
     "Tell me more about the second one",
     "Compare option 1 and option 2"],

    ["I want a warm winter jacket",
     "Show me something in black instead",
     "Why did you pick the first one?",
     "Tell me more about the second one",
     "What is the price of the first one?"],

    ["Sporty leggings for the gym",
     "Do you have cheaper ones instead?",
     "What material is the first one?",
     "Tell me more about the second one",
     "Which one is cheaper?"],

    ["I want a green blouse",
     "Can you show it in white instead?",
     "What colour is the first one?",
     "Tell me more about the second one",
     "What is the price of the first one?"],

    ["I need blue jeans under £25",
     "Which one is better for casual wear?",
     "Show me black ones instead",
     "Tell me more about the first one",
     "What is the price of the second one?"],

    ["Show me men's black jackets",
     "Which one is warmer?",
     "Show me something cheaper",
     "What material is the first one?",
     "Tell me more about the second one"],

    ["Light pink cardigan please",
     "Show me a darker colour instead",
     "Tell me more about the first one",
     "What colour is the second one?",
     "Which one is cheaper?"],
]


async def _clear_user_sessions(db, redis, user_id: str) -> int:
    """Deletes all sessions for the eval user from MongoDB and Redis so every
    run starts from clean context (same pattern as the hallucination driver)."""
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


async def _clear_eval_session_graphs(db) -> int:
    """Removes session graphs from previous eval runs (ceval_* session ids)
    so the contradiction detector starts every session with an empty graph."""
    result = await db.session_graphs.delete_many(
        {"session_id": {"$regex": "^ceval_"}}
    )
    return result.deleted_count


def _rotate_capture_file() -> None:
    """Renames an existing capture file so each run produces a fresh dataset."""
    path = capture_path()
    if os.path.exists(path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.replace(".jsonl", f"_{stamp}.jsonl")
        os.rename(path, backup)
        print(f"Existing capture file rotated to: {os.path.basename(backup)}")


async def run():
    conversations = CONVERSATIONS
    if MAX_CONVERSATIONS > 0:
        conversations = CONVERSATIONS[:MAX_CONVERSATIONS]

    print("=" * 60)
    print("CONTRADICTION EVAL — SESSION COLLECTION")
    print(f"Conversations: {len(conversations)}  "
          f"Turns: {sum(len(c) for c in conversations)}")
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
    graphs_deleted = await _clear_eval_session_graphs(db)
    print(f"Cleared {deleted} old sessions, {graphs_deleted} old eval session graphs.\n")

    turns_run     = 0
    turns_failed  = 0
    eval_sessions = []

    for conv_idx, conversation in enumerate(conversations, start=1):
        session_id = f"ceval_{uuid.uuid4().hex[:8]}"
        eval_sessions.append(session_id)
        print(f"\n{'─'*60}")
        print(f"Conversation {conv_idx}/{len(conversations)}  session={session_id}")

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
                      f"contra_found={rag_result.get('contradiction_found')} "
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
    print("Next step: corrupt_sessions.py builds the labeled test set from this file.")

    # Eval session graphs stay in MongoDB for traceability; clean-up:
    #   db.session_graphs.delete_many({"session_id": {"$regex": "^ceval_"}})

    await close_mongodb_connection()
    await close_redis_connection()
    await close_pool()
    print("All connections closed.")


if __name__ == "__main__":
    asyncio.run(run())
