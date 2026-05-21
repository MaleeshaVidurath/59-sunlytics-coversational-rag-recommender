# =============================================================================
# rl_routes.py
# =============================================================================
# FastAPI routes for Real User RL Signal Collection.
#
# ADD THESE ROUTES TO YOUR EXISTING main.py / api/main.py
#
# Routes added:
#   POST /api/rl/feedback          ← explicit 👍/👎 from frontend
#   GET  /api/rl/stats             ← signal collection statistics
#   POST /api/rl/export            ← export experiences to JSONL for training
#   POST /api/rl/session-complete  ← session outcome signal
# =============================================================================

from fastapi import APIRouter, HTTPException
from memory.db.mongo import get_db
from memory.core.rl_signal_collector import (
    RLSignalCollector,
    ExplicitFeedbackRequest,
    SessionCompleteRequest,
    get_rl_collector,
    LABEL_NAME_TO_ID,
)

router = APIRouter(prefix="/api/rl", tags=["rl"])
collector = get_rl_collector()


@router.post("/feedback")
async def submit_feedback(request: ExplicitFeedbackRequest):
    """
    Explicit user feedback endpoint — called when user clicks 👍 or 👎.

    Uses recommendation_id to look up the classifier_input, turn_id,
    predicted_label and confidence from MongoDB so the RL experience
    has the full [SEP]-joined input_text DistilBERT actually classified.

    Body:
        {
            "session_id":        "sess_abc123",
            "user_id":           "user_def456",
            "recommendation_id": "rec_xyz789",
            "rating":            "up" | "down",
            "article_ids":       ["861848001", "663679001"]
        }
    """
    try:
        db = get_db()

        # ── Look up the recommendation document ───────────────────────────
        # recommendation_id → user_turn_id (patched by chat.py)
        # user_turn_id → turn document with classifier_input
        rec_doc = await db.recommendations.find_one(
            {"recommendation_id": request.recommendation_id}
        )

        classifier_input    = ""
        predicted_label     = 0
        label_name          = ""
        confidence          = 0.0
        predicted_strategy  = "FULL"
        # Prefer turn_id from recommendation doc; fall back to direct turn_id param
        turn_id             = request.turn_id or ""

        if rec_doc:
            # chat.py patches user_turn_id onto recommendation after each message
            # This is the USER turn (not bot turn) that DistilBERT classified
            turn_id = rec_doc.get("user_turn_id", "") or rec_doc.get("turn_id", "")

            # Look up the user turn document
            turn_doc = None
            if turn_id:
                turn_doc = await db.turns.find_one({"turn_id": turn_id})

            if not turn_doc and turn_id:
                # Fallback: search embedded in session document
                session_doc = await db.sessions.find_one(
                    {"session_id": request.session_id,
                     "turns.turn_id": turn_id},
                    {"turns.$": 1}
                )
                if session_doc and session_doc.get("turns"):
                    turn_doc = session_doc["turns"][0]

            if turn_doc:
                cls                = turn_doc.get("classification") or {}
                label_name         = cls.get("label", "")
                predicted_label    = LABEL_NAME_TO_ID.get(label_name, 0)
                confidence         = cls.get("confidence", 0.0)
                predicted_strategy = cls.get("retrieval_strategy", "FULL")

                # Use classifier_input stored by chat.py on the turn document
                classifier_input = turn_doc.get("classifier_input", "")

                # Fallback: rebuild from session history if not yet stored
                if not classifier_input:
                    all_turns = await db.turns.find(
                        {"session_id": request.session_id}
                    ).sort("created_at", 1).to_list(length=50)

                    if not all_turns:
                        session_full = await db.sessions.find_one(
                            {"session_id": request.session_id}
                        )
                        all_turns = session_full.get("turns", []) if session_full else []

                    all_turn_ids = [t.get("turn_id") for t in all_turns]
                    if turn_id in all_turn_ids:
                        pos           = all_turn_ids.index(turn_id)
                        context_turns = all_turns[max(0, pos - 6):pos]
                        current_turn  = all_turns[pos]
                        parts = []
                        for t in context_turns:
                            role    = t.get("role", "user").upper()
                            content = t.get("content", "")
                            if content:
                                parts.append(f"{role}: {content}")
                        parts.append(f"CURRENT: {current_turn.get('content', '')}")
                        classifier_input = " [SEP] ".join(parts)

        exp = await collector.collect_explicit_feedback(
            request=request,
            db=db,
            classifier_input=classifier_input,
            predicted_label=predicted_label,
            label_name=label_name,
            confidence=confidence,
            predicted_strategy=predicted_strategy,
            turn_id=turn_id,
        )
        return {
            "status":        "recorded",
            "experience_id": exp.experience_id,
            "reward":        exp.total_reward,
            "message":       "Feedback recorded. Thank you!",
        }
    except Exception as e:
        import traceback
        print(f"[RL-Feedback] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/stats")
async def get_rl_stats():
    """
    Returns RL signal collection statistics.
    Use this to monitor how many real experiences have been collected
    and whether enough data exists for RL training.
    """
    try:
        db    = get_db()
        stats = await collector.get_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session-complete")
async def session_complete(request: SessionCompleteRequest):
    """
    Called when a conversation session ends (user closes chat or explicitly ends).
    Computes and stores session-level outcome rewards for all relevant turns.

    Body:
        {
            "session_id": "sess_abc123",
            "user_id":    "user_def456"
        }
    """
    try:
        db   = get_db()
        exps = await collector.collect_session_outcome(
            session_id=request.session_id,
            user_id=request.user_id,
            db=db,
        )
        return {
            "status":            "recorded",
            "experiences_added": len(exps),
            "message":           f"Session outcome recorded ({len(exps)} experiences)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_for_training(output_path: str = "rl_buffer_real.jsonl"):
    """
    Export all collected real experiences to a JSONL file.
    This file is directly usable by rl_offline_train.py as input.

    Usage:
        curl -X POST "http://localhost:8000/api/rl/export"
        → creates rl_buffer_real.jsonl in the project root
        → then run: python rl_offline_train.py --buffer rl_buffer_real.jsonl
    """
    try:
        db    = get_db()
        count = await collector.export_for_training(db=db, output_path=output_path)
        return {
            "status":         "exported",
            "experiences":    count,
            "output_path":    output_path,
            "next_step":      f"python rl_offline_train.py --buffer {output_path}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
