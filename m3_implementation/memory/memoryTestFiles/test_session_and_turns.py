# m3_implementation/memory/core/test_session_and_turns.py
# Tests the session manager and turn manager working together.

import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection
from memory.db.redis_client import connect_to_redis, close_redis_connection
from memory.core.session_manager import SessionManager
from memory.core.turn_manager import TurnManager
from memory.models.schemas import TurnClassification


async def test_full_flow():
    # Connect to databases
    await connect_to_mongodb()
    await connect_to_redis()

    session_mgr = SessionManager()
    turn_mgr    = TurnManager()

    print("\n=== Test 1: Create new session ===")
    session = await session_mgr.get_or_create_session(user_id="user_test_001")
    print(f"Session created: {session.session_id}")
    print(f"Status: {session.status}")

    print("\n=== Test 2: Add user turn ===")
    user_turn = await turn_mgr.add_user_turn(
        session_id=session.session_id,
        user_id="user_test_001",
        content="I want a black dress under £50",
        classification=TurnClassification(
            label="INITIAL_REQUEST",
            retrieval_strategy="FULL",
            confidence=0.993
        ),
        entities={"colour_group_name": "Black",
                  "product_type_name": "Dress",
                  "price_max": 50.0}
    )
    print(f"User turn added: {user_turn.turn_id}")
    print(f"Classification: {user_turn.classification.label}")

    print("\n=== Test 3: Add assistant turn ===")
    bot_turn = await turn_mgr.add_assistant_turn(
        session_id=session.session_id,
        user_id="user_test_001",
        content="Here are two options. Option 1 is the Valerie dress (black): "
                "Short dress in crisp cotton. Option 2 is the Angel (dark pink): "
                "Short A-line dress in cotton.",
        recommendation_id="rec_test_001"
    )
    print(f"Bot turn added: {bot_turn.turn_id}")

    print("\n=== Test 4: Get recent turns for DistilBERT ===")
    classifier_input = await turn_mgr.get_classifier_input(
        session_id=session.session_id,
        current_message="What material is the first one?"
    )
    print(f"Classifier input:\n  {classifier_input}")

    print("\n=== Test 5: Get turns as history (for predict.py) ===")
    history = await turn_mgr.get_turns_as_history(session.session_id)
    print("History for predict.py:")
    for turn in history:
        print(f"  {turn['role'].upper()}: {turn['content'][:60]}...")

    print("\n=== Test 6: Update dialogue state ===")
    await session_mgr.update_dialogue_state(
        session_id=session.session_id,
        updates={
            "hard_constraints": {
                "colour_group_name": "Black",
                "product_type_name": "Dress",
                "price_max": 50.0
            }
        }
    )
    state = await session_mgr.get_dialogue_state(session.session_id)
    print(f"Updated state hard_constraints: {state.hard_constraints}")

    print("\n=== Test 7: Resume existing session ===")
    resumed = await session_mgr.get_or_create_session(
        user_id="user_test_001",
        session_id=session.session_id
    )
    print(f"Resumed session: {resumed.session_id}")
    print(f"Same session? {resumed.session_id == session.session_id}")

    print("\n=== Test 8: Add another turn and check count ===")
    await turn_mgr.add_user_turn(
        session_id=session.session_id,
        user_id="user_test_001",
        content="What material is the first one?",
        classification=TurnClassification(
            label="ATTRIBUTE_QUESTION",
            retrieval_strategy="PARTIAL",
            confidence=0.961
        )
    )
    full_session = await session_mgr.get_full_session(session.session_id)
    print(f"Total turns in MongoDB: {full_session.get('turn_count')}")
    print(f"Turns array length: {len(full_session.get('turns', []))}")

    # Cleanup test data
    from memory.db.mongo import get_db
    db = get_db()
    await db.sessions.delete_one({"session_id": session.session_id})
    print(f"\nTest session cleaned up from MongoDB.")

    await close_mongodb_connection()
    await close_redis_connection()
    print("\nAll session and turn tests PASSED.")


if __name__ == "__main__":
    asyncio.run(test_full_flow())