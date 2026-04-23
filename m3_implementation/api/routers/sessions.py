# m3_implementation/api/routers/sessions.py
from fastapi import APIRouter, HTTPException, Query
from memory.db.mongo import get_db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions(user_id: str = Query(...)):
    """
    Returns all sessions for a user, newest first.
    Each session entry has: session_id, title, last_activity, message_count.
    """
    db   = get_db()
    docs = await db.sessions.find(
        {"user_id": user_id},
        {"session_id": 1, "turns": 1, "started_at": 1,
         "last_activity_at": 1, "turn_count": 1, "status": 1}
    ).sort("last_activity_at", -1).to_list(length=100)

    sessions = []
    for doc in docs:
        turns = doc.get("turns", [])
        # Title = first user message
        first_user = next(
            (t["content"] for t in turns if t.get("role") == "user"), 
            "New conversation"
        )
        title = first_user[:45] + "..." if len(first_user) > 45 else first_user
        
        sessions.append({
            "session_id":       doc["session_id"],
            "title":            title,
            "last_activity_at": doc.get("last_activity_at", ""),
            "started_at":       doc.get("started_at", ""),
            "turn_count":       doc.get("turn_count", 0),
            "message_count":    len(turns),
            "status":           doc.get("status", "active"),
        })
    return {"sessions": sessions}


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

    turns = doc.get("turns", [])
    messages = []
    for t in turns:
        messages.append({
            "turn_id":    t.get("turn_id", ""),
            "role":       t.get("role", ""),
            "content":    t.get("content", ""),
            "timestamp":  str(t.get("timestamp", "")),
            "label":      t.get("classification", {}).get("label", "") if t.get("classification") else "",
        })

    return {
        "session_id":  session_id,
        "messages":    messages,
        "turn_count":  doc.get("turn_count", 0),
        "started_at":  str(doc.get("started_at", "")),
    }
