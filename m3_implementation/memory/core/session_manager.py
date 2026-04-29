# m3_implementation/memory/core/session_manager.py
#
# Manages the lifecycle of conversation sessions.
#
# RESPONSIBILITIES:
#   - Create a new session when a user starts chatting
#   - Resume an existing active session when a user returns mid-conversation
#   - Expire sessions that have been idle too long
#   - Load sessions from MongoDB into Redis cache on resume
#   - Save session state back to MongoDB on every update
#
# SESSION LIFECYCLE:
#   new → active → (completed | expired | abandoned)
#
# STORAGE STRATEGY:
#   Redis holds the "hot" active session for fast access during a conversation.
#   MongoDB holds the permanent record of all sessions.
#   When a session is active, BOTH are kept in sync on every turn.
#   When a session expires, Redis data is gone but MongoDB keeps everything.

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from memory.db.mongo import get_db
from memory.db.redis_client import get_redis
from memory.models.schemas import (
    SessionDocument, DialogueState, UserDocument,
    now_utc, new_id
)


# ── Redis key helpers ─────────────────────────────────────────────────────────
# Centralising key names here means if you ever change the format,
# you change it in one place and every function automatically uses the new name.

def _session_state_key(session_id: str) -> str:
    """Redis Hash key for session state (dialogue_state, status, user_id)."""
    return f"session:{session_id}:state"

def _session_turns_key(session_id: str) -> str:
    """Redis List key for recent conversation turns."""
    return f"session:{session_id}:turns"

def _user_prefs_key(user_id: str) -> str:
    """Redis String key for cached user preferences."""
    return f"user:{user_id}:preferences"

def _active_session_key(user_id: str) -> str:
    """Redis String key mapping user_id → their current active session_id."""
    return f"user:{user_id}:active_session"


# ── Session timeout ───────────────────────────────────────────────────────────
def _get_timeout_seconds() -> int:
    minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", 30))
    return minutes * 60


# ── Main SessionManager class ─────────────────────────────────────────────────

class SessionManager:
    """
    Manages session creation, resumption, and expiry.

    Usage in your FastAPI endpoint:
        manager = SessionManager()
        session = await manager.get_or_create_session(user_id, session_id)
    """

    async def get_or_create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> SessionDocument:
        """
        The main entry point called at the start of every chat request.

        Logic:
        1. If session_id given → try to resume that specific session
        2. If no session_id → check if user has an active session in Redis
        3. If no active session found → create a brand new one

        Args:
            user_id:    The user's ID (from your users collection)
            session_id: Optional — the session to resume (sent by frontend)

        Returns:
            A SessionDocument ready to use
        """
        redis = get_redis()

        # Case 1: Specific session_id was provided — try to resume it
        if session_id:
            session = await self._resume_session(session_id, user_id)
            if session:
                return session
            # If resume failed (expired/not found), fall through to create new

        # Case 2: Check if this user has an active session cached in Redis
        cached_session_id = await redis.get(_active_session_key(user_id))
        if cached_session_id:
            session = await self._resume_session(cached_session_id, user_id)
            if session:
                return session

        # Case 3: No active session found — create a fresh one
        return await self._create_new_session(user_id)

    async def _create_new_session(self, user_id: str) -> SessionDocument:
        print(f"[SESSION] ─── _create_new_session for user_id={user_id[:20]}")
        """
        Creates a brand new session in both MongoDB and Redis.
        Called when there is no active session for the user.
        """
        db = get_db()
        redis = get_redis()
        timeout = _get_timeout_seconds()

        # Build the session document using our Pydantic schema
        session = SessionDocument(user_id=user_id)

        # ── Write to MongoDB first (permanent record) ──────────────────────
        await db.sessions.insert_one(
            session.model_dump(mode="json")  # Convert to dict for MongoDB
        )

        # ── Write to Redis (fast access cache) ────────────────────────────
        # We store three separate Redis keys for this session:

        # 1. Session state as a Hash (field-level access is fast)
        await redis.hset(
            _session_state_key(session.session_id),
            mapping={
                "user_id": user_id,
                "status": "active",
                "started_at": session.started_at.isoformat(),
                "last_activity_at": session.last_activity_at.isoformat(),
                # Dialogue state stored as JSON string within the hash
                "dialogue_state": json.dumps(
                    session.dialogue_state.model_dump(mode="json")
                )
            }
        )
        await redis.expire(_session_state_key(session.session_id), timeout)

        # 2. Turns list (starts empty — turns get appended as conversation goes)
        # We just set the TTL on a non-existent key by setting a placeholder
        # and immediately deleting it, then we rely on the first rpush to create it
        # Actually we just set the mapping above and let turns be added naturally

        # 3. Map user_id → active session_id so we can find it next time
        await redis.set(
            _active_session_key(user_id),
            session.session_id,
            ex=timeout
        )

        print(f"New session created: {session.session_id} for user {user_id}")
        return session

    async def _resume_session(
        self,
        session_id: str,
        user_id: str
    ) -> Optional[SessionDocument]:
        """
        Attempts to resume an existing session.
        Checks Redis first (fast path), falls back to MongoDB (cold path).
        Returns None if the session is expired, completed, or not found.
        """
        redis = get_redis()
        timeout = _get_timeout_seconds()

        # ── Fast path: check Redis ─────────────────────────────────────────
        state_raw = await redis.hgetall(_session_state_key(session_id))

        if state_raw:
            # Session is in Redis — it is active and warm
            # Just refresh the TTL (reset the 30-minute timeout)
            await redis.expire(_session_state_key(session_id), timeout)
            await redis.expire(_session_turns_key(session_id), timeout)
            await redis.set(_active_session_key(user_id), session_id, ex=timeout)

            # Rebuild the SessionDocument from Redis state
            dialogue_state = DialogueState.model_validate(
                json.loads(state_raw.get("dialogue_state", "{}"))
            )

            # Get turns from Redis
            turns_raw = await redis.lrange(_session_turns_key(session_id), 0, -1)
            turns = []
            # We store turns as JSON in the list
            # (Full turn reconstruction would need MongoDB for older turns)

            # Return a lightweight session object with current state
            # For full turn history, use get_full_session() which hits MongoDB
            session = SessionDocument(
                session_id=session_id,
                user_id=state_raw.get("user_id", user_id),
                status=state_raw.get("status", "active"),
                dialogue_state=dialogue_state,
                turns=[]   # Turns loaded separately when needed
            )
            return session

        # ── Cold path: check MongoDB ───────────────────────────────────────
        db = get_db()
        doc = await db.sessions.find_one({"session_id": session_id})

        if not doc:
            return None  # Session does not exist

        # Check if session is still valid (not expired/completed)
        if doc.get("status") not in ("active",):
            return None

        # Check timeout — if last activity was too long ago, expire it
        last_activity = doc.get("last_activity_at")
        if last_activity:
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity)
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=timezone.utc)

            idle_seconds = (now_utc() - last_activity).total_seconds()
            if idle_seconds > _get_timeout_seconds():
                # Session timed out — mark it as expired in MongoDB
                await db.sessions.update_one(
                    {"session_id": session_id},
                    {"$set": {"status": "expired", "ended_at": now_utc()}}
                )
                return None

        # Session is valid — warm up Redis from MongoDB
        await self._warm_redis_from_mongodb(doc)

        # Rebuild SessionDocument from MongoDB document
        session = SessionDocument.model_validate(doc)
        return session

    async def _warm_redis_from_mongodb(self, doc: dict):
        """
        Loads session data from a MongoDB document back into Redis.
        Called when a session was cold (not in Redis) but still valid in MongoDB.
        This happens when:
          - Redis restarted and lost its data
          - User returns after exactly the timeout period
        """
        redis = get_redis()
        timeout = _get_timeout_seconds()
        session_id = doc["session_id"]
        user_id = doc["user_id"]

        # Restore session state to Redis
        dialogue_state = doc.get("dialogue_state", {})
        await redis.hset(
            _session_state_key(session_id),
            mapping={
                "user_id": user_id,
                "status": doc.get("status", "active"),
                "started_at": str(doc.get("started_at", "")),
                "last_activity_at": str(doc.get("last_activity_at", "")),
                "dialogue_state": json.dumps(dialogue_state)
            }
        )
        await redis.expire(_session_state_key(session_id), timeout)

        # Restore the last 10 turns to Redis list
        turns = doc.get("turns", [])
        recent_turns = turns[-10:] if len(turns) > 10 else turns
        if recent_turns:
            # Delete existing list first (in case of partial data)
            await redis.delete(_session_turns_key(session_id))
            pipe = redis.pipeline()
            for turn in recent_turns:
                pipe.rpush(_session_turns_key(session_id), json.dumps(turn))
            pipe.expire(_session_turns_key(session_id), timeout)
            await pipe.execute()

        # Update the user → session mapping
        await redis.set(_active_session_key(user_id), session_id, ex=timeout)
        print(f"Session {session_id} warmed up in Redis from MongoDB.")

    async def update_dialogue_state(
        self,
        session_id: str,
        updates: dict
    ):
        """
        Updates the dialogue state in both Redis and MongoDB.
        Call this when constraints or currently_discussing items change.

        Args:
            session_id: The session to update
            updates:    Dict of fields to update in the dialogue_state
                        e.g. {"hard_constraints": {"product_type_name": "Dress"}}
        """
        db = get_db()
        redis = get_redis()
        timeout = _get_timeout_seconds()

        # ── Update Redis ───────────────────────────────────────────────────
        # Get current state from Redis
        state_raw = await redis.hgetall(_session_state_key(session_id))
        current_state = json.loads(state_raw.get("dialogue_state", "{}"))

        # Merge updates into current state
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(current_state.get(key), dict):
                # Deep merge for nested dicts (e.g. hard_constraints)
                current_state[key] = {**current_state.get(key, {}), **value}
            else:
                current_state[key] = value

        # Write back to Redis
        await redis.hset(
            _session_state_key(session_id),
            "dialogue_state", json.dumps(current_state)
        )
        await redis.hset(
            _session_state_key(session_id),
            "last_activity_at", now_utc().isoformat()
        )
        await redis.expire(_session_state_key(session_id), timeout)

        # ── Update MongoDB ─────────────────────────────────────────────────
        # Build MongoDB update using dot notation for nested fields
        mongo_updates = {"last_activity_at": now_utc()}
        for key, value in updates.items():
            mongo_updates[f"dialogue_state.{key}"] = value

        await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": mongo_updates}
        )

    async def get_dialogue_state(self, session_id: str) -> DialogueState:
        """
        Gets the current dialogue state for a session.
        Always reads from Redis (fast path).
        """
        redis = get_redis()
        state_raw = await redis.hgetall(_session_state_key(session_id))

        if state_raw and "dialogue_state" in state_raw:
            return DialogueState.model_validate(
                json.loads(state_raw["dialogue_state"])
            )

        # Fallback: load from MongoDB
        db = get_db()
        doc = await db.sessions.find_one(
            {"session_id": session_id},
            {"dialogue_state": 1}
        )
        if doc and "dialogue_state" in doc:
            return DialogueState.model_validate(doc["dialogue_state"])

        return DialogueState()  # Return empty state as fallback

    async def complete_session(self, session_id: str):
        """
        Marks a session as completed (user explicitly ended the conversation).
        Cleans up Redis and updates MongoDB.
        """
        db = get_db()
        redis = get_redis()

        # Update MongoDB
        await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "completed", "ended_at": now_utc()}}
        )

        # Clean up Redis (the data is safely in MongoDB)
        await redis.delete(_session_state_key(session_id))
        await redis.delete(_session_turns_key(session_id))
        print(f"Session {session_id} completed and cleaned from Redis.")

    async def get_full_session(self, session_id: str) -> Optional[dict]:
        """
        Gets the complete session document from MongoDB including all turns.
        Use this when you need the full conversation history,
        not just the recent turns.
        """
        db = get_db()
        return await db.sessions.find_one({"session_id": session_id})

    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> list[dict]:
        """
        Gets the most recent sessions for a user from MongoDB.
        Useful for showing conversation history or for cross-session
        preference analysis.
        """
        db = get_db()
        cursor = db.sessions.find(
            {"user_id": user_id},
            {"turns": 0}      # Exclude turns for efficiency (can be large)
        ).sort("started_at", -1).limit(limit)

        return await cursor.to_list(length=limit)