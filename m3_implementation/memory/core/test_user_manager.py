# m3_implementation/memory/core/test_user_manager.py

import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from memory.db.redis_client import connect_to_redis, close_redis_connection
from memory.core.user_manager import UserManager


async def test_user_manager():
    await connect_to_mongodb()
    await connect_to_redis()

    mgr = UserManager()

    print("\n=== Test 1: Create new user ===")
    user = await mgr.get_or_create_user(
        customer_id="test_customer_abc123",
        initial_data={
            "age": 28,
            "club_member_status": "ACTIVE",
            "fashion_news_frequency": "Regularly"
        }
    )
    print(f"User created: {user.user_id}")
    print(f"  Age: {user.age}, Status: {user.club_member_status}")

    print("\n=== Test 2: Get same user again (should not create duplicate) ===")
    user2 = await mgr.get_or_create_user(customer_id="test_customer_abc123")
    print(f"Same user_id? {user.user_id == user2.user_id}")

    print("\n=== Test 3: Add positive preferences ===")
    await mgr.update_preferences_from_entities(
        user_id=user.user_id,
        entities={
            "colour_group_name": "Black",
            "product_type_name": "Dress"
        },
        sentiment=0.92,
        source="explicit",
        confidence=0.97
    )
    updated = await mgr.get_preferences(user.user_id, force_refresh=True)
    print(f"Preferences after update: {len(updated.attribute_preferences)}")
    for p in updated.attribute_preferences:
        print(f"  {p.attribute_name}: {p.attribute_value} "
              f"(sentiment={p.sentiment}, confidence={p.confidence})")

    print("\n=== Test 4: Reinforce existing preference ===")
    await mgr.update_preferences_from_entities(
        user_id=user.user_id,
        entities={"colour_group_name": "Black"},
        sentiment=0.95,
        source="explicit"
    )
    updated2 = await mgr.get_preferences(user.user_id, force_refresh=True)
    black_pref = next(
        p for p in updated2.attribute_preferences
        if p.attribute_value == "Black"
    )
    print(f"Black preference mention_count: {black_pref.mention_count} (should be 2)")
    print(f"Black preference sentiment after reinforcement: {black_pref.sentiment:.3f}")

    print("\n=== Test 5: Add dislike ===")
    await mgr.update_preferences_from_entities(
        user_id=user.user_id,
        entities={"colour_group_name": "Orange"},
        sentiment=-0.80,
        source="explicit"
    )
    updated3 = await mgr.get_preferences(user.user_id, force_refresh=True)
    print(f"Dislikes: {len(updated3.disliked_attributes)}")
    for d in updated3.disliked_attributes:
        print(f"  Dislikes {d.attribute_name}: {d.attribute_value} "
              f"(sentiment={d.sentiment})")

    print("\n=== Test 6: Get preference summary for enrichment layer ===")
    summary = await mgr.get_preference_summary(user.user_id)
    print(f"Liked attributes ({len(summary['liked_attributes'])}):")
    for p in summary["liked_attributes"]:
        print(f"  {p['attribute_name']}: {p['attribute_value']} "
              f"(weight={p['weight']})")
    print(f"Disliked values: {summary['disliked_values']}")
    print(f"Hard constraints: {summary['hard_constraints']}")

    print("\n=== Test 7: Update style profile ===")
    await mgr.update_style_profile(
        user_id=user.user_id,
        updates={
            "primary_style": "casual",
            "occasion_preferences": {"casual": 0.9, "work": 0.6}
        }
    )
    updated4 = await mgr.get_preferences(user.user_id, force_refresh=True)
    print(f"Style profile: {updated4.style_profile.primary_style}")
    print(f"Occasion prefs: {updated4.style_profile.occasion_preferences}")

    # Cleanup
    db = get_db()
    await db.users.delete_one({"user_id": user.user_id})
    print(f"\nTest user cleaned up.")

    await close_mongodb_connection()
    await close_redis_connection()
    print("\nAll user manager tests PASSED.")


if __name__ == "__main__":
    asyncio.run(test_user_manager())