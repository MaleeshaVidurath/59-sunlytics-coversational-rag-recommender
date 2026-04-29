# m3_implementation/memory/core/user_manager.py
#
# Manages long-term user profiles and preference memory.
#
# RESPONSIBILITIES:
#   - Create a user profile when a new user starts chatting
#   - Load an existing user profile (from Redis cache or MongoDB)
#   - Update preferences when a conversation turn reveals new information
#   - Apply time decay to preferences so old preferences fade gradually
#   - Build a preference summary for the enrichment layer
#
# PREFERENCE UPDATE RULES (inspired by MemoCRS and MGConvRex):
#   - INITIAL_REQUEST turn   → extract new hard/soft constraints as preferences
#   - REFINEMENT turn        → update existing preference or add new one
#   - FEEDBACK positive turn → reinforce the preferences of the accepted item
#   - FEEDBACK negative turn → add/strengthen a dislike for the item's attributes
#   - CHITCHAT / other turns → no preference update

import json
import os
import math
from datetime import datetime, timezone
from typing import Optional

from memory.db.mongo import get_db
from memory.db.redis_client import get_redis
from memory.models.schemas import (
    UserDocument, PreferenceEntry, StyleProfile,
    PurchaseSummary, now_utc, new_id
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _strip_mongo_id(doc: dict) -> dict:
    """
    Removes MongoDB's internal _id field from a document dict.
    Must be called before passing any MongoDB document to Pydantic,
    because bson.ObjectId cannot be serialized to JSON.
    Uses pop with a default of None so it is safe to call even if
    _id is not present.
    """
    doc.pop("_id", None)
    return doc


# ── Redis key helpers ─────────────────────────────────────────────────────────

def _user_prefs_key(user_id: str) -> str:
    """Redis key for the cached user preference profile."""
    return f"user:{user_id}:preferences"


def _get_prefs_cache_seconds() -> int:
    minutes = int(os.getenv("USER_PREFERENCES_CACHE_MINUTES", 60))
    return minutes * 60


# ── Main class ────────────────────────────────────────────────────────────────

class UserManager:
    """
    Manages long-term user profiles and preference memory.

    Usage:
        manager = UserManager()

        # Get or create a user profile
        user = await manager.get_or_create_user(customer_id="0001d44dbe...")

        # Update preferences after a turn
        await manager.update_preferences_from_entities(
            user_id=user.user_id,
            entities={"colour_group_name": "Black", "product_type_name": "Dress"},
            sentiment=0.9,
            source="explicit"
        )

        # Get preferences formatted for the enrichment layer
        prefs = await manager.get_preference_summary(user_id)
    """

    # ── User creation and retrieval ───────────────────────────────────────────

    async def get_or_create_user(
        self,
        customer_id: str,
        initial_data: dict | None = None
    ) -> UserDocument:
        """
        Gets an existing user or creates a new one.
        Called when a user first interacts with the chatbot.

        Args:
            customer_id:  The customer_id from your sample_customers.csv
                          (the long hash string like "0001d44dbe7f6c4b...")
            initial_data: Optional dict with fields from sample_customers.csv
                          e.g. {"age": 44, "club_member_status": "ACTIVE",
                                "fashion_news_frequency": "Regularly"}

        Returns:
            UserDocument — either the existing profile or a newly created one
        """
        db = get_db()

        # Try to find existing user in MongoDB
        existing = await db.users.find_one({"customer_id": customer_id})

        if existing:
            # User exists — update their last_active timestamp and return
            await db.users.update_one(
                {"customer_id": customer_id},
                {"$set": {"last_active_at": now_utc()}}
            )

            # Invalidate the Redis cache so fresh data is loaded next time
            user_id = existing.get("user_id")
            if user_id:
                redis = get_redis()
                await redis.delete(_user_prefs_key(user_id))

            # Remove MongoDB's _id field before passing to Pydantic —
            # ObjectId cannot be serialized to JSON
            return UserDocument.model_validate(_strip_mongo_id(existing))

        # User does not exist — create a new profile
        user = UserDocument(
            customer_id=customer_id,
            **(initial_data or {})
        )

        await db.users.insert_one(user.model_dump(mode="json"))
        print(f"New user created: {user.user_id} (customer: {customer_id[:16]}...)")
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[UserDocument]:
        """Gets a user by their internal user_id."""
        db = get_db()
        doc = await db.users.find_one({"user_id": user_id})
        if not doc:
            return None
        return UserDocument.model_validate(_strip_mongo_id(doc))

    async def get_user_by_customer_id(
        self,
        customer_id: str
    ) -> Optional[UserDocument]:
        """Gets a user by their customer_id from sample_customers.csv."""
        db = get_db()
        doc = await db.users.find_one({"customer_id": customer_id})
        if not doc:
            return None
        return UserDocument.model_validate(_strip_mongo_id(doc))

    # ── Preference loading ────────────────────────────────────────────────────

    async def get_preferences(
        self,
        user_id: str,
        force_refresh: bool = False
    ) -> UserDocument:
        """
        Gets the full user profile including all preferences.
        Uses Redis cache for speed — only hits MongoDB on cache miss or refresh.

        Args:
            user_id:       The user's internal ID
            force_refresh: If True, bypass cache and load fresh from MongoDB

        Returns:
            UserDocument with all preference fields populated
        """
        redis = get_redis()
        cache_key = _user_prefs_key(user_id)

        # ── Fast path: Redis cache ─────────────────────────────────────────
        if not force_refresh:
            cached = await redis.get(cache_key)
            if cached:
                return UserDocument.model_validate(json.loads(cached))

        # ── Cold path: MongoDB ─────────────────────────────────────────────
        db = get_db()
        doc = await db.users.find_one({"user_id": user_id})
        if not doc:
            raise ValueError(f"User {user_id} not found in database")

        # Remove _id before Pydantic validation
        _strip_mongo_id(doc)
        user = UserDocument.model_validate(doc)

        # Apply time decay to preference weights before caching
        await self._apply_time_decay(user_id, user)

        # Reload the updated document after decay was applied,
        # then cache it in Redis for future requests
        updated_doc = await db.users.find_one({"user_id": user_id})
        if updated_doc:
            _strip_mongo_id(updated_doc)
            user = UserDocument.model_validate(updated_doc)

        await redis.set(
            cache_key,
            user.model_dump_json(),
            ex=_get_prefs_cache_seconds()
        )

        return user

    async def get_user_by_id(self, user_id: str):
        """Returns the User document by user_id."""
        db = get_db()
        doc = await db.users.find_one({"user_id": user_id})
        if not doc:
            return None
        doc = _strip_mongo_id(doc)
        try:
            from memory.models.schemas import User
            return User.model_validate(doc)
        except Exception:
            return None

    async def get_purchase_history(self, user_id: str) -> dict:
        """
        Returns the full purchase_history dict for a user.
        Tries user_id first, then falls back to customer_id pattern.
        Returns {} if no history exists.
        """
        db  = get_db()
        # Try by user_id first (primary key used at runtime)
        doc = await db.users.find_one(
            {"user_id": user_id},
            {"purchase_history": 1, "customer_id": 1}
        )
        if doc and doc.get("purchase_history"):
            print(f"[USER-MGR] get_purchase_history: found by user_id={user_id[:20]}")
            return doc.get("purchase_history", {})

        # Fallback: try customer_id derived from user_id
        # user_id format: "user_hist_XXXXXXXX" where XXXXXXXX = first 8 chars of customer_id
        if user_id.startswith("user_hist_"):
            partial_cid = user_id[len("user_hist_"):]
            doc2 = await db.users.find_one(
                {"customer_id": {"$regex": f"^{partial_cid}"}},
                {"purchase_history": 1, "customer_id": 1}
            )
            if doc2 and doc2.get("purchase_history"):
                print(f"[USER-MGR] get_purchase_history: found by customer_id prefix={partial_cid}")
                return doc2.get("purchase_history", {})

        print(f"[USER-MGR] get_purchase_history: NO history found for user_id={user_id[:20]}")
        return {}

    async def get_purchase_history_by_customer(self, customer_id: str) -> dict:
        """
        Returns the full purchase_history dict for a customer_id.
        Used during user creation to pre-load history.
        """
        db  = get_db()
        doc = await db.users.find_one(
            {"customer_id": customer_id},
            {"purchase_history": 1}
        )
        if not doc:
            return {}
        return doc.get("purchase_history", {})

    async def get_preference_summary(self, user_id: str) -> dict:
        """
        Returns a clean, query-ready summary of the user's preferences.
        This is what the enrichment layer uses to build retrieval queries.

        Returns a dict with:
            liked_attributes:  List of preference dicts sorted by strength
            disliked_values:   Dict of attribute_name → [disliked values]
            hard_constraints:  Dict of attribute → value for very strong prefs
            style_profile:     The user's style profile summary
            purchase_summary:  Aggregated purchase behavior
            top_product_types: Most purchased product types
            top_colours:       Most purchased colours
        """
        user = await self.get_preferences(user_id)

        # Build liked attributes list with computed weight
        # Weight = sentiment × confidence × decay_weight
        liked = []
        for pref in user.attribute_preferences:
            if pref.sentiment > 0.0:
                weight = pref.sentiment * pref.confidence * pref.decay_weight
                liked.append({
                    "attribute_name": pref.attribute_name,
                    "attribute_value": pref.attribute_value,
                    "weight": round(weight, 3),
                    "sentiment": pref.sentiment,
                    "confidence": pref.confidence,
                    "source": pref.source,
                    "pref_id": pref.pref_id
                })

        # Sort by weight descending — strongest preferences first
        liked.sort(key=lambda x: x["weight"], reverse=True)

        # Build disliked values dict for easy exclusion in retrieval
        disliked: dict[str, list[str]] = {}
        for pref in user.disliked_attributes:
            attr = pref.attribute_name
            if attr not in disliked:
                disliked[attr] = []
            disliked[attr].append(pref.attribute_value)

        # Hard constraints: very strong, high-confidence, explicit preferences
        # These become mandatory WHERE filters in retrieval queries
        hard: dict[str, str] = {}
        for pref in liked:
            if (pref["sentiment"] > 0.85
                    and pref["confidence"] > 0.85
                    and pref["source"] == "explicit"):
                hard[pref["attribute_name"]] = pref["attribute_value"]

        return {
            "liked_attributes": liked,
            "disliked_values": disliked,
            "hard_constraints": hard,
            "style_profile": user.style_profile.model_dump(),
            "purchase_summary": user.purchase_summary.model_dump(),
            "top_product_types": user.purchase_summary.top_product_types,
            "top_colours": user.purchase_summary.top_colours
        }

    # ── Preference updates ────────────────────────────────────────────────────

    async def update_preferences_from_entities(
        self,
        user_id: str,
        entities: dict,
        sentiment: float,
        source: str = "explicit",
        confidence: float = 0.9
    ):
        """
        Updates user preferences based on entities extracted from a turn.

        Called after:
          - INITIAL_REQUEST: entities from the user's first request
          - REFINEMENT:      entities from the preference change message
          - FEEDBACK:        entities from the item being accepted/rejected

        The attribute_name values must match column names from sample_articles.csv:
          colour_group_name, product_type_name, index_group_name,
          garment_group_name, section_name

        Args:
            user_id:    The user to update
            entities:   Dict like {"colour_group_name": "Black",
                                   "product_type_name": "Dress"}
            sentiment:  Sentiment score for these entities.
                        Positive (e.g. 0.9) for likes,
                        Negative (e.g. -0.8) for dislikes
            source:     "explicit" if user stated it, "implicit" if inferred
            confidence: How confident we are (0.0–1.0)
        """
        if not entities:
            return

        db = get_db()
        redis = get_redis()

        # Map entity field names to preference categories.
        # Keys are column names from sample_articles.csv.
        category_map = {
            "colour_group_name":            "colour",
            "perceived_colour_master_name": "colour",
            "product_type_name":            "product_type",
            "index_group_name":             "index_group",
            "garment_group_name":           "garment_group",
            "section_name":                 "style",
            "graphical_appearance_name":    "style",
            "style":                        "style",
            "occasion":                     "occasion",
            "material":                     "material",
        }

        # Load current preferences to check for existing entries
        user = await self.get_preferences(user_id)
        updates_made = False

        for attr_name, attr_value in entities.items():
            # Skip numeric/price fields — they are not preference attributes
            if attr_name in ("price_max", "price_min"):
                continue

            category = category_map.get(attr_name)
            if not category:
                continue  # Unknown attribute — skip

            attr_value_str = str(attr_value)

            if sentiment >= 0:
                # ── Positive preference ────────────────────────────────────
                existing = next(
                    (p for p in user.attribute_preferences
                     if p.attribute_name == attr_name
                     and p.attribute_value == attr_value_str),
                    None
                )

                if existing:
                    # Reinforce: new sentiment = weighted average
                    # (existing × 0.7 + new × 0.3)
                    # This prevents one very strong signal from dominating
                    new_sentiment = (existing.sentiment * 0.7) + (sentiment * 0.3)
                    new_confidence = min(
                        (existing.confidence * 0.7) + (confidence * 0.3),
                        1.0
                    )
                    await db.users.update_one(
                        {
                            "user_id": user_id,
                            "attribute_preferences.pref_id": existing.pref_id
                        },
                        {
                            "$set": {
                                "attribute_preferences.$.sentiment":   new_sentiment,
                                "attribute_preferences.$.confidence":  new_confidence,
                                "attribute_preferences.$.last_mentioned_at": now_utc(),
                                "attribute_preferences.$.decay_weight": 1.0
                            },
                            "$inc": {
                                "attribute_preferences.$.mention_count": 1
                            }
                        }
                    )
                else:
                    # New preference — append to array
                    new_pref = PreferenceEntry(
                        category=category,
                        attribute_name=attr_name,
                        attribute_value=attr_value_str,
                        sentiment=sentiment,
                        confidence=confidence,
                        source=source
                    )
                    await db.users.update_one(
                        {"user_id": user_id},
                        {
                            "$push": {
                                "attribute_preferences":
                                    new_pref.model_dump(mode="json")
                            }
                        }
                    )

                # ── Conflict resolution (Problem 5 fix) ───────────────────
                # When a user explicitly states a NEW value for an attribute
                # (e.g. "show me WHITE ones" after previously preferring Black),
                # reduce the sentiment of conflicting values in the same
                # attribute slot. We do not delete — we reduce.
                # This follows Mem0's UPDATE-not-DELETE philosophy:
                #   new_conflicting_sentiment = old_sentiment × 0.4
                # (reduced but not zero — the old preference may return)
                # Only applies to explicit source — implicit signals are
                # weaker and do not override existing explicit preferences.
                if source == "explicit" and sentiment > 0.5:
                    for i, existing_pref in enumerate(user.attribute_preferences):
                        if (existing_pref.attribute_name == attr_name
                                and existing_pref.attribute_value != attr_value_str
                                and existing_pref.sentiment > 0.3):
                            # Reduce conflicting preference sentiment
                            reduced_sentiment = existing_pref.sentiment * 0.4
                            await db.users.update_one(
                                {
                                    "user_id": user_id,
                                    "attribute_preferences.pref_id": existing_pref.pref_id
                                },
                                {
                                    "$set": {
                                        "attribute_preferences.$.sentiment": reduced_sentiment,
                                        "attribute_preferences.$.decay_weight": 0.5,
                                    }
                                }
                            )

                updates_made = True

            else:
                # ── Negative preference (dislike) ──────────────────────────
                existing_dislike = next(
                    (p for p in user.disliked_attributes
                     if p.attribute_name == attr_name
                     and p.attribute_value == attr_value_str),
                    None
                )

                if existing_dislike:
                    # Strengthen the dislike (more negative)
                    new_sentiment = max(
                        (existing_dislike.sentiment * 0.7) + (sentiment * 0.3),
                        -1.0
                    )
                    await db.users.update_one(
                        {
                            "user_id": user_id,
                            "disliked_attributes.pref_id": existing_dislike.pref_id
                        },
                        {
                            "$set": {
                                "disliked_attributes.$.sentiment":
                                    new_sentiment,
                                "disliked_attributes.$.last_mentioned_at":
                                    now_utc()
                            },
                            "$inc": {
                                "disliked_attributes.$.mention_count": 1
                            }
                        }
                    )
                else:
                    # New dislike — append to disliked_attributes array
                    new_dislike = PreferenceEntry(
                        category=category,
                        attribute_name=attr_name,
                        attribute_value=attr_value_str,
                        sentiment=sentiment,
                        confidence=confidence,
                        source=source
                    )
                    await db.users.update_one(
                        {"user_id": user_id},
                        {
                            "$push": {
                                "disliked_attributes":
                                    new_dislike.model_dump(mode="json")
                            }
                        }
                    )
                updates_made = True

        if updates_made:
            # Invalidate Redis cache so next read gets fresh preferences
            await redis.delete(_user_prefs_key(user_id))
            print(f"Preferences updated for user {user_id}")

    async def update_style_profile(
        self,
        user_id: str,
        updates: dict
    ):
        """
        Updates the user's high-level style profile.

        Args:
            updates: Dict of fields to update in style_profile
                     e.g. {"primary_style": "casual",
                           "occasion_preferences": {"casual": 0.9}}
        """
        db = get_db()
        redis = get_redis()

        mongo_updates: dict = {}
        for key, value in updates.items():
            if isinstance(value, dict):
                # Merge nested dicts rather than replacing them entirely
                for subkey, subval in value.items():
                    mongo_updates[f"style_profile.{key}.{subkey}"] = subval
            else:
                mongo_updates[f"style_profile.{key}"] = value

        await db.users.update_one(
            {"user_id": user_id},
            {"$set": mongo_updates}
        )
        await redis.delete(_user_prefs_key(user_id))

    async def update_purchase_summary(
        self,
        user_id: str,
        article_data: dict
    ):
        """
        Updates the purchase summary when a user buys or accepts an item.
        article_data should contain fields from sample_articles.csv:
            product_type_name, colour_group_name, index_group_name, etc.
        """
        db = get_db()
        redis = get_redis()

        # Increment purchase count and update last purchase timestamp
        await db.users.update_one(
            {"user_id": user_id},
            {
                "$set": {"purchase_summary.last_purchase_at": now_utc()},
                "$inc": {"purchase_summary.total_purchases": 1}
            }
        )

        # Add to top product types (addToSet prevents duplicates)
        if article_data.get("product_type_name"):
            await db.users.update_one(
                {"user_id": user_id},
                {
                    "$addToSet": {
                        "purchase_summary.top_product_types":
                            article_data["product_type_name"]
                    }
                }
            )

        # Add to top colours
        if article_data.get("colour_group_name"):
            await db.users.update_one(
                {"user_id": user_id},
                {
                    "$addToSet": {
                        "purchase_summary.top_colours":
                            article_data["colour_group_name"]
                    }
                }
            )

        # Add to top index groups (e.g. Ladieswear, Menswear)
        if article_data.get("index_group_name"):
            await db.users.update_one(
                {"user_id": user_id},
                {
                    "$addToSet": {
                        "purchase_summary.top_index_groups":
                            article_data["index_group_name"]
                    }
                }
            )

        # Invalidate Redis cache
        await redis.delete(_user_prefs_key(user_id))

    # ── Time decay ────────────────────────────────────────────────────────────

    async def _apply_time_decay(
        self,
        user_id: str,
        user: UserDocument
    ):
        """
        Applies exponential time decay to preference decay_weights.

        The formula is:  weight = e^(−λ × days_since_last_mentioned)

        With λ = 0.0077, the half-life is 90 days:
          - A preference mentioned today    → decay_weight = 1.00
          - A preference mentioned 90 ago   → decay_weight = 0.50
          - A preference mentioned 180 ago  → decay_weight = 0.25
          - Floor at 0.10 — preferences never fully disappear

        This runs when preferences are loaded from MongoDB (i.e. on cache miss),
        not on every single request (they are cached in Redis for 60 minutes).
        """
        # λ = ln(2) / half_life_days = 0.693 / 90 ≈ 0.0077
        lambda_decay = 0.0077
        now = now_utc()
        db = get_db()

        decay_updates: dict = {}

        for i, pref in enumerate(user.attribute_preferences):
            # Ensure timezone-aware comparison
            last_mentioned = pref.last_mentioned_at
            if last_mentioned.tzinfo is None:
                last_mentioned = last_mentioned.replace(tzinfo=timezone.utc)

            days_ago = max((now - last_mentioned).days, 0)
            new_weight = math.exp(-lambda_decay * days_ago)
            new_weight = max(round(new_weight, 4), 0.1)  # Floor at 0.1
            decay_updates[f"attribute_preferences.{i}.decay_weight"] = new_weight

        if decay_updates:
            await db.users.update_one(
                {"user_id": user_id},
                {"$set": decay_updates}
            )
