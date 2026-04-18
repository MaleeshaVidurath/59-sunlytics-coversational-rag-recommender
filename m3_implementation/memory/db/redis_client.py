# m3_implementation/memory/db/redis_client.py
#
# This file manages the Redis connection.
# Redis is used for fast in-memory storage of active session data.
# We use the async redis client (redis.asyncio) to match FastAPI's async style.

import os
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

# Module-level Redis connection pool — shared across all requests
_redis: aioredis.Redis | None = None


async def connect_to_redis():
    """
    Creates the Redis connection pool.
    A connection pool reuses connections rather than creating new ones
    for every operation — much more efficient.
    """
    global _redis

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    db = int(os.getenv("REDIS_DB", 0))

    print(f"Connecting to Redis at {host}:{port}...")
    _redis = aioredis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=True,  # Automatically decode bytes to strings
        max_connections=20      # Connection pool size
    )

    # Test the connection with a PING
    await _redis.ping()
    print("Connected to Redis successfully.")


async def close_redis_connection():
    """Call this when the FastAPI app shuts down."""
    global _redis
    if _redis:
        await _redis.aclose()
        print("Redis connection closed.")


def get_redis() -> aioredis.Redis:
    """
    Returns the Redis client instance.
    
    Usage example:
        from memory.db.redis_client import get_redis
        redis = get_redis()
        await redis.set("key", "value", ex=1800)  # expires in 30 minutes
    """
    if _redis is None:
        raise RuntimeError(
            "Redis not connected. "
            "Make sure connect_to_redis() was called at app startup."
        )
    return _redis