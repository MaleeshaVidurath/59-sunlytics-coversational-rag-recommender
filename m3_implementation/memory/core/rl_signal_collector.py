# =============================================================================
# rl_signal_collector.py
# =============================================================================
# Real User RL Signal Collector for the DistilBERT CRS Classifier.
#
# DESIGN PHILOSOPHY:
#   This module collects REAL user feedback signals from actual conversations
#   to use as reward signals for RL fine-tuning — NOT synthetic data.
#
# SIGNAL SOURCES (3 types, all from real user behaviour):
#
#   Signal 1 — Explicit Feedback (thumbs up/down on bot responses)
#     User clicks 👍 or 👎 on a recommendation in the frontend.
#     Maps directly to: recommendation_id + article_ids + rating (+1 / -1)
#     Reward: +1.0 (thumbs up) or -1.0 (thumbs down)
#     Schema already has: RecommendationDocument.outcome field (pending/accepted/rejected)
#
#   Signal 2 — Implicit Behaviour (what the user does NEXT after a recommendation)
#     Captured automatically from the conversation flow:
#     - Next turn = SELECTION_REFERENCE → user engaged → +0.5
#     - Next turn = ATTRIBUTE_QUESTION  → user interested → +0.3
#     - Next turn = REFINEMENT          → recommendation missed → -0.2
#     - Next turn = FEEDBACK (negative) → recommendation failed → -0.8
#     - Session ends after recommendation → neutral 0.0
#     This is the IMPLICIT signal — no user action needed, auto-collected.
#
#   Signal 3 — Conversation Outcome (session-level signal)
#     When a session completes (user stops or explicitly ends):
#     - Counted SELECTION_REFERENCE turns vs REFINEMENT turns
#     - Short conversation (≤3 turns) with SELECTION_REFERENCE → +1.0 (efficient)
#     - Long conversation (6+ turns) with only REFINEMENT → -0.5 (frustrating)
#     - User reached FEEDBACK with positive words → +0.8
#
# HOW IT FEEDS RL TRAINING:
#   Each collected signal is stored as an RLExperience document in MongoDB.
#   rl_offline_train.py reads these experiences and runs REINFORCE.
#   The experience contains:
#     - input_text: what DistilBERT classified (the [SEP]-joined conversation)
#     - predicted_label: what DistilBERT predicted
#     - total_reward: the real user signal
#   This is IDENTICAL format to the synthetic trajectories — just with real rewards.
#
# FASTAPI ENDPOINTS ADDED:
#   POST /api/rl/feedback          ← explicit thumbs up/down from frontend
#   GET  /api/rl/stats             ← for monitoring dashboard
#   POST /api/rl/session-complete  ← called when session ends
# =============================================================================

import os
import json
from datetime import datetime, timezone
from typing import Optional, Literal
from pydantic import BaseModel, Field
import uuid


def now_utc():
    return datetime.now(timezone.utc)

def new_id(prefix=""):
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# =============================================================================
# Data models for RL experiences
# =============================================================================

class RLExperience(BaseModel):
    """
    One real-user RL experience stored in MongoDB.
    Directly usable as training data for REINFORCE.
    """
    experience_id:    str = Field(default_factory=lambda: new_id("rlexp_"))
    session_id:       str
    user_id:          str
    turn_id:          str           # The user turn that was classified
    created_at:       datetime = Field(default_factory=now_utc)

    # DistilBERT prediction for this turn
    input_text:       str           # The [SEP]-joined classifier input
    predicted_label:  int           # 0-7 integer label
    label_name:       str           # e.g. "REFINEMENT"
    predicted_strategy: str         # "FULL" / "PARTIAL" / "NO"
    confidence:       float

    # Real user reward signal
    total_reward:     float         # The actual reward from user behaviour
    reward_source:    Literal[
        "explicit_thumbs_up",
        "explicit_thumbs_down",
        "implicit_next_turn",
        "session_outcome",
    ]
    reward_detail:    str           # Human-readable explanation

    # Context
    recommendation_id: Optional[str] = None
    article_ids:       list[str] = []


class ExplicitFeedbackRequest(BaseModel):
    """Request body for POST /api/rl/feedback"""
    session_id:        str
    user_id:           str
    recommendation_id: str = ""
    turn_id:           str = ""                # user turn_id (used when no recommendation)
    rating:            Literal["up", "down"]   # 👍 or 👎
    article_ids:       list[str] = []          # which items were shown


class SessionCompleteRequest(BaseModel):
    """Request body for POST /api/rl/session-complete"""
    session_id: str
    user_id:    str


# =============================================================================
# Reward computation from real signals
# =============================================================================

# Label name → integer ID (must match DistilBERT training order)
LABEL_NAME_TO_ID = {
    "INITIAL_REQUEST":     0,
    "REFINEMENT":          1,
    "ATTRIBUTE_QUESTION":  2,
    "EXPLANATION_WHY":     3,
    "COMPARISON":          4,
    "SELECTION_REFERENCE": 5,
    "FEEDBACK":            6,
    "CHITCHAT":            7,
}

# Implicit reward: what does the NEXT turn tell us about this classification?
# If DistilBERT predicted REFINEMENT and user next does SELECTION_REFERENCE,
# that means the refinement correctly led to a better recommendation.
NEXT_TURN_IMPLICIT_REWARD = {
    # Current label → next label → reward
    ("REFINEMENT",         "SELECTION_REFERENCE"): +0.7,  # refinement worked
    ("REFINEMENT",         "ATTRIBUTE_QUESTION"):  +0.3,  # still engaged
    ("REFINEMENT",         "COMPARISON"):          +0.3,  # still looking
    ("REFINEMENT",         "REFINEMENT"):          -0.3,  # still not satisfied
    ("REFINEMENT",         "FEEDBACK"):            -0.1,  # ended, ambiguous
    ("INITIAL_REQUEST",    "SELECTION_REFERENCE"): +1.0,  # perfect first shot
    ("INITIAL_REQUEST",    "ATTRIBUTE_QUESTION"):  +0.4,  # good engagement
    ("INITIAL_REQUEST",    "COMPARISON"):          +0.4,  # comparing options
    ("INITIAL_REQUEST",    "REFINEMENT"):          -0.1,  # first rec missed
    ("INITIAL_REQUEST",    "FEEDBACK"):            +0.5,  # immediate feedback
    ("ATTRIBUTE_QUESTION", "SELECTION_REFERENCE"): +0.8,  # question led to pick
    ("ATTRIBUTE_QUESTION", "REFINEMENT"):          -0.2,  # question not helpful
    ("EXPLANATION_WHY",    "SELECTION_REFERENCE"): +0.8,  # explanation convinced
    ("EXPLANATION_WHY",    "REFINEMENT"):          -0.1,  # not convinced
    ("COMPARISON",         "SELECTION_REFERENCE"): +0.9,  # comparison decisive
    ("COMPARISON",         "REFINEMENT"):          -0.1,  # comparison not helpful
}


def compute_explicit_reward(rating: str) -> tuple[float, str]:
    """Explicit 👍/👎 → reward."""
    if rating == "up":
        return +1.0, "User gave thumbs up — recommendation accepted"
    else:
        return -1.0, "User gave thumbs down — recommendation rejected"


def compute_implicit_reward(
    current_label: str,
    next_label: str,
) -> tuple[float, str]:
    """Next-turn behaviour → implicit reward."""
    reward = NEXT_TURN_IMPLICIT_REWARD.get((current_label, next_label), 0.0)
    detail = (
        f"After {current_label}, user sent {next_label} "
        f"→ implicit reward={reward:+.1f}"
    )
    return reward, detail


def compute_session_outcome_reward(
    turns: list[dict],
) -> tuple[float, str]:
    """
    Session-level reward from conversation shape.
    Analyses the full turn sequence to determine if the session was efficient
    and successful.
    """
    labels = [
        t.get("classification", {}).get("label", "CHITCHAT")
        for t in turns
        if t.get("role") == "user"
    ]

    n_turns        = len(labels)
    n_refinements  = labels.count("REFINEMENT")
    n_selections   = labels.count("SELECTION_REFERENCE")
    n_attributes   = labels.count("ATTRIBUTE_QUESTION")
    n_explanations = labels.count("EXPLANATION_WHY")
    n_comparisons  = labels.count("COMPARISON")
    n_feedback     = labels.count("FEEDBACK")

    # Short, efficient conversations that end in selection
    if n_turns <= 3 and n_selections >= 1:
        return +1.0, f"Short efficient conversation ({n_turns} turns) → user selected item"

    # Good engagement conversation
    if n_selections >= 1 and n_refinements <= 1:
        return +0.7, f"User selected item after {n_turns} turns ({n_refinements} refinements)"

    # Engaged conversation with attribute questions and comparisons
    if (n_attributes + n_explanations + n_comparisons) >= 2 and n_selections >= 1:
        return +0.8, "High engagement (questions/comparisons) led to selection"

    # Multiple refinements but ended in selection
    if n_selections >= 1 and n_refinements >= 2:
        return +0.3, f"User selected item but after {n_refinements} refinements (inefficient)"

    # Long conversation, no selection
    if n_turns >= 6 and n_selections == 0 and n_refinements >= 3:
        return -0.5, f"Long frustrating conversation ({n_turns} turns, {n_refinements} refinements, no selection)"

    # Gave up (no selection, many refinements)
    if n_refinements >= 4 and n_selections == 0:
        return -0.8, f"User gave up after {n_refinements} refinements"

    # Neutral — no strong signal
    return 0.0, f"Neutral session outcome ({n_turns} turns)"


# =============================================================================
# Signal collector service
# =============================================================================

class RLSignalCollector:
    """
    Collects real user RL signals and stores them as RLExperience documents.

    Used in two ways:
      1. Called from FastAPI endpoints (explicit feedback)
      2. Called from pipeline.py (implicit signals after each turn)
    """

    async def collect_explicit_feedback(
        self,
        request:    ExplicitFeedbackRequest,
        db,         # MongoDB database instance
        classifier_input: str = "",
        predicted_label:  int = 0,
        label_name:       str = "",
        confidence:       float = 0.0,
        predicted_strategy: str = "FULL",
        turn_id:          str = "",
    ) -> RLExperience:
        """
        Collect explicit 👍/👎 feedback from the frontend.
        Called when user clicks thumbs up or down on a recommendation.
        """
        reward, detail = compute_explicit_reward(request.rating)

        exp = RLExperience(
            session_id=         request.session_id,
            user_id=            request.user_id,
            turn_id=            turn_id or new_id("turn_"),
            input_text=         classifier_input,
            predicted_label=    predicted_label,
            label_name=         label_name,
            predicted_strategy= predicted_strategy,
            confidence=         confidence,
            total_reward=       reward,
            reward_source=      "explicit_thumbs_up" if request.rating == "up" else "explicit_thumbs_down",
            reward_detail=      detail,
            recommendation_id=  request.recommendation_id,
            article_ids=        request.article_ids,
        )

        await db.rl_experiences.insert_one(exp.model_dump(mode="json"))

        # Also update the RecommendationDocument outcome
        outcome = "accepted" if request.rating == "up" else "rejected"
        await db.recommendations.update_one(
            {"recommendation_id": request.recommendation_id},
            {"$set": {"outcome": outcome}}
        )

        print(
            f"[RL-Signal] Explicit feedback: {request.rating} "
            f"session={request.session_id[:12]} "
            f"reward={reward:+.1f}"
        )
        return exp

    async def collect_implicit_signal(
        self,
        session_id:        str,
        user_id:           str,
        prev_turn_id:      str,
        prev_label:        str,
        prev_input_text:   str,
        prev_label_id:     int,
        prev_strategy:     str,
        prev_confidence:   float,
        next_label:        str,
        db,
    ) -> Optional[RLExperience]:
        """
        Collect implicit signal: what did the user do AFTER a recommendation?
        Called from pipeline.py at the START of each new turn.
        Looks at the PREVIOUS turn's label and the CURRENT turn's label.
        """
        reward, detail = compute_implicit_reward(prev_label, next_label)

        # Skip neutral signals (0.0) to keep the buffer clean
        if reward == 0.0:
            return None

        exp = RLExperience(
            session_id=         session_id,
            user_id=            user_id,
            turn_id=            prev_turn_id,
            input_text=         prev_input_text,
            predicted_label=    prev_label_id,
            label_name=         prev_label,
            predicted_strategy= prev_strategy,
            confidence=         prev_confidence,
            total_reward=       reward,
            reward_source=      "implicit_next_turn",
            reward_detail=      detail,
        )

        await db.rl_experiences.insert_one(exp.model_dump(mode="json"))

        print(
            f"[RL-Signal] Implicit: {prev_label}→{next_label} "
            f"reward={reward:+.1f} session={session_id[:12]}"
        )
        return exp

    async def collect_session_outcome(
        self,
        session_id: str,
        user_id:    str,
        db,
    ) -> list[RLExperience]:
        """
        Collect session-level outcome signal when conversation ends.
        Analyses the full conversation and assigns rewards to key turns.
        Called when session status changes to completed/abandoned.
        """
        # Load the full session from MongoDB
        session = await db.sessions.find_one({"session_id": session_id})
        if not session:
            return []

        turns = session.get("turns", [])
        reward, detail = compute_session_outcome_reward(turns)

        # Apply session reward to all REFINEMENT and INITIAL_REQUEST turns
        # These are the turns where the classifier decision mattered most
        experiences = []
        for turn in turns:
            if turn.get("role") != "user":
                continue
            label = turn.get("classification", {}).get("label", "")
            if label not in ("INITIAL_REQUEST", "REFINEMENT"):
                continue

            label_id = LABEL_NAME_TO_ID.get(label, 0)
            strategy = turn.get("classification", {}).get("retrieval_strategy", "FULL")
            conf     = turn.get("classification", {}).get("confidence", 0.0)

            exp = RLExperience(
                session_id=         session_id,
                user_id=            user_id,
                turn_id=            turn.get("turn_id", new_id("turn_")),
                input_text=         turn.get("classifier_input", "") or turn.get("content", ""),
                predicted_label=    label_id,
                label_name=         label,
                predicted_strategy= strategy,
                confidence=         conf,
                total_reward=       reward,
                reward_source=      "session_outcome",
                reward_detail=      detail,
            )

            await db.rl_experiences.insert_one(exp.model_dump(mode="json"))
            experiences.append(exp)

        if experiences:
            print(
                f"[RL-Signal] Session outcome: {len(experiences)} experiences "
                f"reward={reward:+.1f} session={session_id[:12]}"
            )
        return experiences

    async def get_stats(self, db) -> dict:
        """Returns RL signal collection statistics for the monitoring endpoint."""
        total     = await db.rl_experiences.count_documents({})
        explicit  = await db.rl_experiences.count_documents(
            {"reward_source": {"$in": ["explicit_thumbs_up", "explicit_thumbs_down"]}}
        )
        implicit  = await db.rl_experiences.count_documents(
            {"reward_source": "implicit_next_turn"}
        )
        outcome   = await db.rl_experiences.count_documents(
            {"reward_source": "session_outcome"}
        )

        # Reward distribution
        pipeline = [
            {"$group": {"_id": None,
                "avg_reward":   {"$avg": "$total_reward"},
                "pos_count":    {"$sum": {"$cond": [{"$gt": ["$total_reward", 0]}, 1, 0]}},
                "neg_count":    {"$sum": {"$cond": [{"$lt": ["$total_reward", 0]}, 1, 0]}},
            }}
        ]
        agg = await db.rl_experiences.aggregate(pipeline).to_list(1)
        agg = agg[0] if agg else {}

        return {
            "total_experiences":    total,
            "explicit_feedback":    explicit,
            "implicit_signals":     implicit,
            "session_outcomes":     outcome,
            "avg_reward":           round(agg.get("avg_reward", 0.0) or 0.0, 3),
            "positive_signals":     agg.get("pos_count", 0),
            "negative_signals":     agg.get("neg_count", 0),
            "ready_for_rl":         total >= 100,
            "min_for_rl_training":  100,
        }

    async def export_for_training(self, db, output_path: str = "rl_buffer_real.jsonl") -> int:
        """
        Export all experiences to JSONL file for rl_offline_train.py.
        Same format as the synthetic buffer — fully compatible.
        """
        cursor = db.rl_experiences.find({})
        count  = 0
        with open(output_path, "w") as f:
            async for exp in cursor:
                exp.pop("_id", None)
                f.write(json.dumps({
                    "input_text":        exp.get("input_text", ""),
                    "predicted_label":   exp.get("predicted_label", 0),
                    "label_name":        exp.get("label_name", ""),
                    "total_reward":      exp.get("total_reward", 0.0),
                    "reward_source":     exp.get("reward_source", ""),
                    "session_id":        exp.get("session_id", ""),
                }) + "\n")
                count += 1
        print(f"[RL-Signal] Exported {count} real experiences → {output_path}")
        return count


# Module-level singleton
_collector: Optional[RLSignalCollector] = None

def get_rl_collector() -> RLSignalCollector:
    global _collector
    if _collector is None:
        _collector = RLSignalCollector()
    return _collector
