# m3_implementation/memory/core/turn_manager.py
#
# Manages conversation turns within a session.
#
# RESPONSIBILITIES:
#   - Append new turns (user or assistant) to Redis and MongoDB
#   - Retrieve the last N turns for DistilBERT classification input
#   - Format turns into the exact string format DistilBERT was trained on
#   - Keep Redis and MongoDB in sync on every turn

import json
from datetime import datetime, timezone

from memory.db.mongo import get_db
from memory.db.redis_client import get_redis
from memory.models.schemas import (
    ConversationTurn, TurnClassification, now_utc
)


# How many turns to keep in the Redis list.
# We keep the last 10 in Redis for fast access.
# All turns are in MongoDB regardless.
REDIS_TURNS_BUFFER = 10

# How many turns to feed to DistilBERT for classification.
# Research shows 2-3 prior exchanges (up to 6 turns) is optimal.
CLASSIFIER_CONTEXT_TURNS = 3


def _session_turns_key(session_id: str) -> str:
    return f"session:{session_id}:turns"

def _session_state_key(session_id: str) -> str:
    return f"session:{session_id}:state"


class TurnManager:
    """
    Handles adding and retrieving conversation turns.

    Usage:
        manager = TurnManager()

        # Add a user turn
        turn = await manager.add_user_turn(session_id, user_id, "I want a black dress")

        # Get context for DistilBERT
        context_string = await manager.get_classifier_input(session_id, "Show cheaper ones")
    """

    async def add_user_turn(
        self,
        session_id: str,
        user_id: str,
        content: str,
        classification: TurnClassification | None = None,
        entities: dict | None = None
    ) -> ConversationTurn:
        """
        Adds a user message turn to the session.
        Called after DistilBERT has classified the message
        (classification and entities are available at this point).

        Returns the ConversationTurn object with its generated turn_id.
        """
        db = get_db()
        redis = get_redis()

        # Get current turn count from Redis to assign the right turn_number
        state_raw = await redis.hgetall(_session_state_key(session_id))
        # Turn count is tracked in MongoDB — get it from there
        doc = await db.sessions.find_one(
            {"session_id": session_id},
            {"turn_count": 1}
        )
        turn_number = (doc.get("turn_count", 0) if doc else 0) + 1

        turn = ConversationTurn(
            turn_number=turn_number,
            role="user",
            content=content,
            timestamp=now_utc(),
            classification=classification,
            entities=entities or {}
        )

        await self._persist_turn(session_id, turn)
        return turn

    async def add_assistant_turn(
        self,
        session_id: str,
        user_id: str,
        content: str,
        recommendation_id: str | None = None
    ) -> ConversationTurn:
        """
        Adds an assistant (bot) message turn to the session.
        Called after the system generates its response.
        """
        db = get_db()

        doc = await db.sessions.find_one(
            {"session_id": session_id},
            {"turn_count": 1}
        )
        turn_number = (doc.get("turn_count", 0) if doc else 0) + 1

        turn = ConversationTurn(
            turn_number=turn_number,
            role="assistant",
            content=content,
            timestamp=now_utc(),
            recommendation_id=recommendation_id
        )

        await self._persist_turn(session_id, turn)
        return turn

    async def _persist_turn(self, session_id: str, turn: ConversationTurn):
        """
        Writes a turn to both Redis (fast buffer) and MongoDB (permanent).
        This is called by both add_user_turn and add_assistant_turn.

        Redis: Appends to a list, keeps only the last REDIS_TURNS_BUFFER turns.
        MongoDB: Pushes to the embedded turns array, increments turn_count.
        """
        db = get_db()
        redis = get_redis()

        # Convert turn to a plain dict for storage
        turn_dict = turn.model_dump(mode="json")

        # ── Redis: append to turns list ────────────────────────────────────
        pipe = redis.pipeline()
        # rpush appends to the right end of the list
        pipe.rpush(_session_turns_key(session_id), json.dumps(turn_dict))
        # Keep only the last REDIS_TURNS_BUFFER turns to control memory usage
        # ltrim keeps elements from index 0 to -(1), meaning we trim from the left
        pipe.ltrim(_session_turns_key(session_id), -REDIS_TURNS_BUFFER, -1)
        # Reset TTL since there was activity
        timeout = int(30 * 60)  # 30 minutes in seconds
        pipe.expire(_session_turns_key(session_id), timeout)
        await pipe.execute()

        # ── MongoDB: push to embedded array and increment count ────────────
        await db.sessions.update_one(
            {"session_id": session_id},
            {
                "$push": {"turns": turn_dict},
                "$inc": {"turn_count": 1},
                "$set": {"last_activity_at": now_utc()}
            }
        )

    async def get_recent_turns(
        self,
        session_id: str,
        n: int = CLASSIFIER_CONTEXT_TURNS
    ) -> list[dict]:
        """
        Gets the last N turns from the session.

        This is called right before DistilBERT classification to provide
        conversation context. Always tries Redis first (sub-millisecond),
        falls back to MongoDB if Redis is cold.

        Args:
            session_id: The session to query
            n:          How many recent turns to retrieve (default 3)

        Returns:
            List of turn dicts, oldest first (chronological order)
        """
        redis = get_redis()

        # ── Fast path: Redis ───────────────────────────────────────────────
        # lrange with negative indices: -n means "last n items"
        cached = await redis.lrange(_session_turns_key(session_id), -n, -1)
        if cached:
            return [json.loads(t) for t in cached]

        # ── Cold path: MongoDB ─────────────────────────────────────────────
        db = get_db()
        doc = await db.sessions.find_one(
            {"session_id": session_id},
            {"turns": {"$slice": -n}}  # MongoDB $slice gets last N elements
        )
        if doc and "turns" in doc:
            return doc["turns"]

        return []

    async def get_classifier_input(
        self,
        session_id: str,
        current_message: str
    ) -> str:
        """
        Formats recent turns + current message into the exact string format
        that your DistilBERT model was trained on.

        Format: "USER: turn1 [SEP] BOT: turn2 [SEP] CURRENT: message"

        This is critical — the format must exactly match what was used
        during training, otherwise DistilBERT will perform poorly.

        Args:
            session_id:       The active session
            current_message:  The user's latest message (not yet stored)

        Returns:
            A formatted string ready to pass to the DistilBERT predictor
        """
        # Get the last CLASSIFIER_CONTEXT_TURNS turns (excludes current message)
        recent_turns = await self.get_recent_turns(
            session_id,
            n=CLASSIFIER_CONTEXT_TURNS * 2  # *2 because each exchange = 2 turns
        )

        parts = []
        for turn in recent_turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                parts.append(f"USER: {content}")
            else:
                parts.append(f"BOT: {content}")

        # Add the current message at the end
        parts.append(f"CURRENT: {current_message}")

        return " [SEP] ".join(parts)

    async def get_turns_as_history(
        self,
        session_id: str,
        n: int = CLASSIFIER_CONTEXT_TURNS
    ) -> list[dict]:
        """
        Returns recent turns formatted as the history list that
        predict.py's Predictor.predict() expects.

        Format: [{"role": "user" | "bot", "content": "..."}]

        Note: predict.py uses "bot" but DistilBERT training used "assistant"
        for the role — this method converts to "bot" for predict.py compatibility.
        """
        recent_turns = await self.get_recent_turns(session_id, n=n * 2)

        history = []
        for turn in recent_turns:
            role = turn.get("role", "user")
            # Convert "assistant" → "bot" to match predict.py's expected format
            if role == "assistant":
                role = "bot"
            history.append({
                "role": role,
                "content": turn.get("content", "")
            })

        return history