# m3_implementation/memory/db/mongo.py
#
# This file manages the MongoDB connection for the entire memory module.
# We use Motor (async MongoDB driver) because FastAPI is async.
# The connection is created once when the app starts and reused for all requests
# — creating a new connection for every request would be extremely slow.

import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# These module-level variables hold the connection.
# They start as None and are set when connect_to_mongodb() is called.
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongodb():
    """
    Call this once when your FastAPI app starts up.
    It creates the MongoDB connection and stores it in the module-level variables.
    """
    global _client, _db

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "sunlytics_crs")

    print(f"Connecting to MongoDB at {mongodb_url}...")
    _client = AsyncIOMotorClient(mongodb_url)
    _db = _client[db_name]

    # Create the collections and their indexes on first run.
    # Indexes are crucial for performance — without them, every query
    # does a full collection scan which gets slow as data grows.
    await _create_indexes()
    print(f"Connected to MongoDB database: {db_name}")


async def _create_indexes():
    """
    Creates all the indexes needed for fast queries.
    MongoDB creates these only if they don't already exist,
    so it is safe to call this every time the app starts.
    """
    db = get_db()

    # users collection — look up users by their customer_id
    await db.users.create_index("customer_id", unique=True)

    # sessions collection — look up sessions by session_id (unique)
    # and find all sessions for a user ordered by time
    await db.sessions.create_index("session_id", unique=True)
    await db.sessions.create_index([("user_id", 1), ("started_at", -1)])
    await db.sessions.create_index("status")

    # recommendations collection — find recs by session or user
    await db.recommendations.create_index("session_id")
    await db.recommendations.create_index([("user_id", 1), ("created_at", -1)])

    # explanations collection — find explanations by session and by recommendation
    await db.explanations.create_index("session_id")
    await db.explanations.create_index("recommendation_id")
    await db.explanations.create_index([("user_id", 1), ("created_at", -1)])

    # contradiction_log collection — find contradictions by session
    await db.contradiction_log.create_index("session_id")
    await db.contradiction_log.create_index([("user_id", 1), ("detected_at", -1)])

    # session_model_locks — one document per session, stores which model was selected.
    # Standalone collection — never touches sessions/users/recommendations.
    # {session_id, selected_model, user_id, locked_at}
    await db.session_model_locks.create_index("session_id", unique=True)
    await db.session_model_locks.create_index("user_id")

    # ── M2 member collections (full set, same structure as M3) ───────────────
    await db.m2_session_graphs.create_index("session_id", unique=True)

    await db.m2_contradiction_log.create_index("session_id")
    await db.m2_contradiction_log.create_index([("user_id", 1), ("detected_at", -1)])

    await db.m2_recommendations.create_index("session_id")
    await db.m2_recommendations.create_index([("user_id", 1), ("created_at", -1)])

    await db.m2_explanations.create_index("session_id")
    await db.m2_explanations.create_index("recommendation_id")
    await db.m2_explanations.create_index([("user_id", 1), ("created_at", -1)])

    # m2_turns — standalone (M3 turns are embedded in sessions; M2 gets its own collection)
    await db.m2_turns.create_index("turn_id", unique=True)
    await db.m2_turns.create_index("session_id")
    await db.m2_turns.create_index([("user_id", 1), ("created_at", -1)])

    # ── M1 member collections (full set, same structure as M3) ───────────────
    await db.m1_session_graphs.create_index("session_id", unique=True)

    await db.m1_contradiction_log.create_index("session_id")
    await db.m1_contradiction_log.create_index([("user_id", 1), ("detected_at", -1)])

    await db.m1_recommendations.create_index("session_id")
    await db.m1_recommendations.create_index([("user_id", 1), ("created_at", -1)])

    await db.m1_explanations.create_index("session_id")
    await db.m1_explanations.create_index("recommendation_id")
    await db.m1_explanations.create_index([("user_id", 1), ("created_at", -1)])

    # m1_turns — standalone (same reasoning as m2_turns)
    await db.m1_turns.create_index("turn_id", unique=True)
    await db.m1_turns.create_index("session_id")
    await db.m1_turns.create_index([("user_id", 1), ("created_at", -1)])

    print("MongoDB indexes created/verified.")


async def close_mongodb_connection():
    """Call this when the FastAPI app shuts down to cleanly close the connection."""
    global _client
    if _client:
        _client.close()
        print("MongoDB connection closed.")


# ── Multi-member collection name helpers ─────────────────────────────────────
# M3 is the backbone and uses the default collection names.
# M1 and M2 each have their own prefixed session_graphs and contradiction_log
# so their cross-turn memory never mixes with M3's or each other's.
# All other collections (users, sessions, recommendations, etc.) remain shared
# under M3 because M3 is the backbone that owns the session lifecycle.

_MEMBER_COLLECTION_MAP: dict[str, dict[str, str]] = {
    "m3": {
        "session_graphs":    "session_graphs",
        "contradiction_log": "contradiction_log",
        "recommendations":   "recommendations",
    },
    "m2": {
        "session_graphs":    "m2_session_graphs",
        "contradiction_log": "m2_contradiction_log",
        "recommendations":   "m2_recommendations",
    },
    "m1": {
        "session_graphs":    "m1_session_graphs",
        "contradiction_log": "m1_contradiction_log",
        "recommendations":   "m1_recommendations",
    },
}


def get_collection_name(base: str, model: str = "m3") -> str:
    """
    Returns the MongoDB collection name for a given base name and model member.

    Args:
        base:  "session_graphs" or "contradiction_log"
        model: "m3" (default), "m2", or "m1"

    Examples:
        get_collection_name("session_graphs", "m3") → "session_graphs"
        get_collection_name("session_graphs", "m2") → "m2_session_graphs"
        get_collection_name("contradiction_log", "m1") → "m1_contradiction_log"
    """
    member_map = _MEMBER_COLLECTION_MAP.get(model, _MEMBER_COLLECTION_MAP["m3"])
    return member_map.get(base, base)


def get_db() -> AsyncIOMotorDatabase:
    """
    Returns the database instance.
    Every other file imports and calls this function to get the database.
    
    Usage example:
        from memory.db.mongo import get_db
        db = get_db()
        await db.sessions.find_one({"session_id": "sess_abc"})
    """
    if _db is None:
        raise RuntimeError(
            "MongoDB not connected. "
            "Make sure connect_to_mongodb() was called at app startup."
        )
    return _db