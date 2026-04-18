# m3_implementation/memory/test_connections.py
# Run this file to verify MongoDB and Redis connections work.
# Delete it or keep it for debugging — it does not affect anything else.

import asyncio
import sys
import os

# Add the project root to Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from memory.db.mongo import connect_to_mongodb, get_db, close_mongodb_connection
from memory.db.redis_client import connect_to_redis, get_redis, close_redis_connection


async def test_mongodb():
    print("\n--- Testing MongoDB ---")
    await connect_to_mongodb()
    db = get_db()

    # Try writing and reading a test document
    result = await db.test_collection.insert_one({"test": "hello", "value": 42})
    print(f"Inserted document with id: {result.inserted_id}")

    doc = await db.test_collection.find_one({"test": "hello"})
    print(f"Retrieved document: {doc}")

    # Clean up the test document
    await db.test_collection.delete_one({"test": "hello"})
    print("Test document cleaned up.")

    await close_mongodb_connection()
    print("MongoDB test PASSED.")


async def test_redis():
    print("\n--- Testing Redis ---")
    await connect_to_redis()
    redis = get_redis()

    # Try writing and reading a key with a 10-second expiry
    await redis.set("test_key", "hello_redis", ex=10)
    value = await redis.get("test_key")
    print(f"Retrieved from Redis: {value}")

    # Test list operations (what we use for conversation turns)
    await redis.rpush("test_list", "turn_1", "turn_2", "turn_3")
    last_two = await redis.lrange("test_list", -2, -1)
    print(f"Last 2 items from list: {last_two}")
    await redis.delete("test_list")

    await close_redis_connection()
    print("Redis test PASSED.")


async def main():
    print("Testing database connections...")
    try:
        await test_mongodb()
    except Exception as e:
        print(f"MongoDB test FAILED: {e}")

    try:
        await test_redis()
    except Exception as e:
        print(f"Redis test FAILED: {e}")

    print("\nAll tests complete.")


if __name__ == "__main__":
    asyncio.run(main())