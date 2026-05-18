# m3_implementation/memory/models/schemas.py
#
# All Pydantic schemas for the memory module.
# These define the exact structure of every document stored in MongoDB.
#
# Why Pydantic v2?
# - Automatic type validation (catches bugs early)
# - Built-in JSON serialisation
# - FastAPI uses it for request/response validation
# - Clear, self-documenting code

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Literal
from pydantic import BaseModel, Field
import uuid


# ─── Helper ───────────────────────────────────────────────────────────────────

def now_utc() -> datetime:
    """Returns the current UTC time. Used as default for timestamp fields."""
    return datetime.now(timezone.utc)


def new_id(prefix: str = "") -> str:
    """Generates a unique ID string like 'sess_a3f2b1' or 'pref_9c8d7e'."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# ─── 1. User Preference Entry ─────────────────────────────────────────────────
# One entry in the user's preference list.
# Each entry represents one (attribute, value) preference pair.
# e.g. user likes Black colour, or dislikes Orange, or prefers Dress type.

class PreferenceEntry(BaseModel):
    pref_id: str = Field(default_factory=lambda: new_id("pref_"))

    # What attribute this preference is about
    # attribute_name matches the column names in your sample_articles.csv
    category: Literal[
        "colour", "product_type", "style", "occasion",
        "material", "price_range", "index_group", "garment_group"
    ]
    attribute_name: str   # e.g. "colour_group_name", "product_type_name"
    attribute_value: str  # e.g. "Black", "Dress", "casual"

    # Sentiment: 1.0 = loves it, -1.0 = hates it, 0.0 = neutral
    sentiment: float = Field(ge=-1.0, le=1.0)

    # Confidence: how sure we are about this preference
    # 1.0 = user explicitly said it, 0.3 = weakly inferred from behavior
    confidence: float = Field(ge=0.0, le=1.0)

    # Where did this preference come from?
    source: Literal["explicit", "implicit", "mixed"]

    # Tracking fields
    mention_count: int = 1
    decay_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    first_mentioned_at: datetime = Field(default_factory=now_utc)
    last_mentioned_at: datetime = Field(default_factory=now_utc)


# ─── 2. Style Profile ─────────────────────────────────────────────────────────
# High-level style summary for a user.
# Updated as preferences accumulate over multiple sessions.

class StyleProfile(BaseModel):
    primary_style: Optional[str] = None     # e.g. "casual", "minimalist"
    secondary_styles: list[str] = []        # e.g. ["classic", "sporty"]

    # Occasion preferences: score 0.0-1.0 for how much they shop for each
    occasion_preferences: dict[str, float] = {}
    # e.g. {"casual": 0.9, "work": 0.7, "party": 0.3, "formal": 0.2}

    # Size preferences per garment group (from your articles data)
    size_preferences: dict[str, str] = {}
    # e.g. {"tops": "M", "bottoms": "28", "dresses": "M"}


# ─── 3. Purchase Summary ──────────────────────────────────────────────────────
# Aggregated from sample_transactions.csv + sample_articles.csv.
# Stored here so we do not re-query PostgreSQL every time.

class PurchaseSummary(BaseModel):
    total_purchases: int = 0
    avg_price_normalized: float = 0.0    # Price scale from your transactions CSV
    top_product_types: list[str] = []    # e.g. ["Dress", "Vest top", "Trousers"]
    top_colours: list[str] = []          # e.g. ["Black", "White", "Dark Blue"]
    top_index_groups: list[str] = []     # e.g. ["Ladieswear"]
    last_purchase_at: Optional[datetime] = None


# ─── 4. User Document ─────────────────────────────────────────────────────────
# The main user profile document stored in the `users` MongoDB collection.
# One document per user, updated across all sessions.

class UserDocument(BaseModel):
    # MongoDB uses _id, but we store user_id as a separate searchable field
    user_id: str = Field(default_factory=lambda: new_id("user_"))
    customer_id: str          # From sample_customers.csv (the long hash string)

    # From sample_customers.csv
    club_member_status: Optional[str] = None   # e.g. "ACTIVE"
    fashion_news_frequency: Optional[str] = None  # e.g. "Regularly"
    age: Optional[int] = None
    postal_code: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=now_utc)
    last_active_at: datetime = Field(default_factory=now_utc)

    # Core preference memory
    # This list grows as the user has more conversations
    attribute_preferences: list[PreferenceEntry] = []
    disliked_attributes: list[PreferenceEntry] = []

    # High-level profile
    style_profile: StyleProfile = Field(default_factory=StyleProfile)
    purchase_summary: PurchaseSummary = Field(default_factory=PurchaseSummary)
    purchase_history: dict            = Field(default_factory=dict)

    class Config:
        # Allow extra fields (in case MongoDB returns _id or other fields)
        extra = "allow"


# ─── 5. Item In Context ───────────────────────────────────────────────────────
# Represents one of the currently-discussed items during a session.
# Fields come from sample_articles.csv.

class ItemInContext(BaseModel):
    article_id: str
    prod_name: str
    product_type_name: str
    colour_group_name: str
    index_group_name: Optional[str] = None
    section_name: Optional[str] = None
    garment_group_name: Optional[str] = None
    detail_desc: Optional[str] = None
    graphical_appearance_name: Optional[str] = None
    price: Optional[float] = None        # From transactions if available


# ─── 6. Dialogue State ────────────────────────────────────────────────────────
# The current "working memory" for an active session.
# Tracks what the user wants right now — constraints, items in focus, history.
# Inspired by RA-Rec's semi-structured dialogue state design.

class DialogueState(BaseModel):
    # Hard constraints: filters that MUST be satisfied
    # These become WHERE conditions in your retrieval query
    hard_constraints: dict[str, str | float | int] = {}
    # e.g. {"product_type_name": "Dress", "price_max": 50.0, "colour_group_name": "Black"}

    # Soft constraints: nice-to-have preferences for this session
    # These become ranking boosters in your retrieval query
    soft_constraints: dict[str, str] = {}
    # e.g. {"style": "casual", "occasion": "summer"}

    # Items currently being discussed
    # item_a = first recommended item, item_b = second recommended item
    currently_discussing: dict[str, ItemInContext | None] = {
        "item_a": None,
        "item_b": None
    }

    # Items rejected or accepted THIS session
    # Used to exclude rejected items from future recommendations
    rejected_items: list[str] = []      # list of article_ids
    accepted_items: list[str] = []      # list of article_ids

    # Summary of what the user is looking for (updated by LLM after each turn)
    intent_summary: Optional[str] = None


# ─── 7. Turn Classification ───────────────────────────────────────────────────
# The result from your DistilBERT classifier for a user turn.

class TurnClassification(BaseModel):
    label: str               # e.g. "ATTRIBUTE_QUESTION"
    retrieval_strategy: str  # "FULL", "PARTIAL", or "NO"
    confidence: float
    used_rules: bool = False # True if the rule-based fallback was used


# ─── 8. Conversation Turn ─────────────────────────────────────────────────────
# One message in the conversation — either from user or assistant.
# Embedded inside SessionDocument.

class ConversationTurn(BaseModel):
    turn_id: str = Field(default_factory=lambda: new_id("turn_"))
    turn_number: int          # Sequential: 1, 2, 3, 4...
    role: Literal["user", "assistant"]
    content: str              # The actual message text
    timestamp: datetime = Field(default_factory=now_utc)

    # Only set for user turns (assistant turns are not classified)
    classification: Optional[TurnClassification] = None

    # Entities extracted from this turn
    # Keys are article CSV field names, values are the extracted values
    entities: dict[str, str | float] = {}
    # e.g. {"colour_group_name": "Black", "product_type_name": "Dress", "price_max": 50.0}

    # Preferences that were created or updated because of this turn
    preferences_updated: list[str] = []   # list of pref_ids

    # For assistant turns: which recommendation was made (if any)
    recommendation_id: Optional[str] = None


# ─── 9. Session Document ──────────────────────────────────────────────────────
# One conversation session. Stored in the `sessions` MongoDB collection.
# Contains the full turn history embedded inside it.

class SessionDocument(BaseModel):
    session_id: str = Field(default_factory=lambda: new_id("sess_"))
    user_id: str
    status: Literal["active", "completed", "expired", "abandoned"] = "active"

    # Timestamps
    started_at: datetime = Field(default_factory=now_utc)
    last_activity_at: datetime = Field(default_factory=now_utc)
    ended_at: Optional[datetime] = None
    timeout_minutes: int = 30

    # The working memory for this session
    dialogue_state: DialogueState = Field(default_factory=DialogueState)

    # Full conversation history — embedded array of turns
    turns: list[ConversationTurn] = []
    turn_count: int = 0    # Maintained separately for fast queries

    class Config:
        extra = "allow"


# ─── 10. Explanation Claim ────────────────────────────────────────────────────
# One atomic, verifiable claim extracted from a system explanation.
# This is the core unit for contradiction detection.
# Inspired by FActScore's atomic fact decomposition.

class ExplanationClaim(BaseModel):
    claim_id: str = Field(default_factory=lambda: new_id("claim_"))

    # The claim as a self-contained natural language sentence
    # "self-contained" means it can be understood without context
    claim_text: str

    # What kind of claim is this?
    claim_type: Literal[
        "preference_match",   # "This matches the user's preference for black"
        "price_match",        # "This item is within the user's budget"
        "attribute_fact",     # "This item is made from cotton"
        "comparative",        # "This item is cheaper than the other option"
        "style_match"         # "This matches the user's casual style preference"
    ]

    # Which article attribute this claim is about
    # Matches column names from sample_articles.csv
    attribute: Optional[str] = None   # e.g. "colour_group_name", "price"

    # The actual value being claimed
    evidence_value: Optional[str | float] = None

    # Cross-references for contradiction detection
    # Links this claim to the preference it references
    user_preference_ref: Optional[str] = None  # pref_id from user's preferences

    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    # Current status — updated when contradictions are detected
    status: Literal[
        "active",        # Currently valid
        "retracted",     # Withdrawn because it was wrong
        "contradicted",  # A later claim contradicted this one
        "confirmed"      # User explicitly confirmed this
    ] = "active"


# ─── 11. Explanation Document ─────────────────────────────────────────────────
# Stored in the `explanations` MongoDB collection.
# One document per recommendation, containing all claims about that recommendation.

class ExplanationDocument(BaseModel):
    explanation_id: str = Field(default_factory=lambda: new_id("expl_"))
    recommendation_id: str    # Links to recommendations collection
    article_id: str           # The article this explanation is about
    session_id: str
    user_id: str
    turn_id: str              # Which assistant turn generated this explanation
    created_at: datetime = Field(default_factory=now_utc)

    # The full natural language explanation shown to the user
    full_explanation: str

    # Atomic claims extracted from the explanation
    claims: list[ExplanationClaim] = []

    # Log of contradictions detected for claims in this explanation
    contradiction_log: list[dict] = []

    class Config:
        extra = "allow"


# ─── 12. Recommendation Document ──────────────────────────────────────────────
# Stored in the `recommendations` MongoDB collection.
# Tracks every time the system recommended items.

class RecommendationDocument(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: new_id("rec_"))
    session_id: str
    user_id: str
    turn_id: str
    created_at: datetime = Field(default_factory=now_utc)

    # The items that were recommended
    items: list[ItemInContext] = []

    # What triggered this recommendation
    trigger_label: str          # DistilBERT label e.g. "INITIAL_REQUEST"
    retrieval_strategy: str     # "FULL", "PARTIAL", "NO"

    # Updated later when user reacts
    outcome: Literal["pending", "accepted", "rejected", "ignored"] = "pending"

    class Config:
        extra = "allow"


# ─── 13. Contradiction Log Entry ──────────────────────────────────────────────
# Stored in the `contradiction_log` MongoDB collection.

class ContradictionEntry(BaseModel):
    contradiction_id: str = Field(default_factory=lambda: new_id("contra_"))
    session_id: str
    user_id: str
    detected_at: datetime = Field(default_factory=now_utc)

    old_claim_id: str
    old_claim_text: str
    new_claim_text: str
    article_id: Optional[str] = None
    attribute: Optional[str] = None

    # DeBERTa-v3 NLI contradiction probability score
    nli_score: float

    # How was it resolved?
    resolution: Literal["retract_old", "update_old", "notify_user", "pending"]
    resolution_explanation: Optional[str] = None