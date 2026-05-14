# m3_implementation/api/routers/sessions.py
from fastapi import APIRouter, HTTPException, Query
from memory.db.mongo import get_db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions(user_id: str = Query(...)):
    """
    Returns all sessions for a user, newest first.
    Reads from both the sessions collection AND the turns collection
    because turns are stored separately by TurnManager.
    """
    db   = get_db()

    # Get all session documents for this user
    docs = await db.sessions.find(
        {"user_id": user_id}
    ).sort([("last_activity_at", -1), ("started_at", -1)]).to_list(length=100)

    sessions = []
    for doc in docs:
        session_id = doc.get("session_id", "")
        if not session_id:
            continue

        # Try embedded turns first (old format)
        turns = doc.get("turns", [])
        turn_count = doc.get("turn_count", 0)

        # If no embedded turns, query the turns collection (new format)
        if not turns:
            turn_docs = await db.turns.find(
                {"session_id": session_id}
            ).sort("created_at", 1).to_list(length=50)
            turns = [{"role": t.get("role",""), "content": t.get("content","")}
                     for t in turn_docs]
            turn_count = len(turn_docs)

        # Title = first user message
        first_user = next(
            (t.get("content","") for t in turns if t.get("role") == "user"),
            "New conversation"
        )
        title = first_user[:45] + "..." if len(first_user) > 45 else first_user

        # Use best available timestamp
        last_active = (doc.get("last_activity_at") or
                       doc.get("updated_at") or
                       doc.get("started_at") or "")

        sessions.append({
            "session_id":       session_id,
            "title":            title,
            "last_activity_at": str(last_active),
            "started_at":       str(doc.get("started_at", "")),
            "turn_count":       turn_count,
            "message_count":    len(turns),
            "status":           doc.get("status", "active"),
        })
    return {"sessions": sessions}


@router.post("/new")
async def create_new_session(user_id: str = Query(...)):
    """
    Explicitly starts a new session by clearing the Redis active session pointer.
    Call this when user clicks "New Chat" so the next message creates a fresh session.
    The memory pipeline uses "user:{user_id}:active_session" in Redis to track
    the current active session. Deleting this key forces a new session on next message.
    """
    try:
        from memory.db.redis_client import get_redis
        redis = await get_redis()
        # This is the exact key used by session_manager.py
        deleted = await redis.delete(f"user:{user_id}:active_session")
        print(f"[Sessions] Cleared active session pointer for {user_id} "
              f"(key existed: {deleted > 0})")
    except Exception as e:
        print(f"[Sessions] Redis clear error (non-fatal): {e}")

    return {"status": "ok", "message": "New session will be created on next message"}


@router.get("/{session_id}")
async def get_session_history(session_id: str, user_id: str = Query(...)):
    """
    Returns full message history for a session.
    Used when switching to a previous chat in the sidebar.
    """
    db  = get_db()
    doc = await db.sessions.find_one(
        {"session_id": session_id, "user_id": user_id}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    # Try embedded turns first, then separate turns collection
    turns = doc.get("turns", [])
    if not turns:
        turn_docs = await db.turns.find(
            {"session_id": session_id}
        ).sort("created_at", 1).to_list(length=200)
        turns = turn_docs

    messages = []
    for t in turns:
        classification = t.get("classification") or {}
        messages.append({
            "turn_id":           t.get("turn_id", ""),
            "role":              t.get("role", ""),
            "content":           t.get("content", ""),
            "timestamp":         str(t.get("timestamp", t.get("created_at", ""))),
            "label":             classification.get("label", "") if classification else "",
            "recommendation_id": t.get("recommendation_id", None),
        })

    return {
        "session_id":  session_id,
        "messages":    messages,
        "turn_count":  doc.get("turn_count", 0),
        "started_at":  str(doc.get("started_at", "")),
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str, user_id: str = Query(...)):
    """
    Deletes a session and ALL related data:
      - sessions collection
      - explanations collection (claim history)
      - contradiction_log collection
      - Redis cache for this session
    """
    db = get_db()

    # Verify session belongs to this user
    doc = await db.sessions.find_one(
        {"session_id": session_id, "user_id": user_id}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── Collect session outcome RL signal BEFORE deleting ─────────────────
    # Must run before delete — reads the full session from MongoDB.
    try:
        from memory.core.rl_signal_collector import get_rl_collector
        import asyncio as _aio
        _rl = get_rl_collector()
        _aio.ensure_future(_rl.collect_session_outcome(
            session_id=session_id,
            user_id=user_id,
            db=db,
        ))
    except Exception as _rl_err:
        print(f"[Sessions] RL session outcome warning (non-fatal): {_rl_err}")

    # Delete from all MongoDB collections
    await db.sessions.delete_one({"session_id": session_id})
    await db.explanations.delete_many({"session_id": session_id})
    await db.contradiction_log.delete_many({"session_id": session_id})

    # Clear Redis cache
    try:
        from memory.db.redis_client import get_redis
        redis = await get_redis()
        await redis.delete(f"session:{session_id}:turns")
        await redis.delete(f"session:{session_id}:state")
        await redis.delete(f"session:{session_id}")
    except Exception as e:
        print(f"[Sessions] Redis cleanup error (non-fatal): {e}")

    return {
        "deleted":     True,
        "session_id":  session_id,
        "message":     "Session and all related data deleted successfully."
    }

