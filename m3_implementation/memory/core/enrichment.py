# m3_implementation/memory/core/enrichment.py
#
# The enrichment layer — bridges DistilBERT classification and retrieval.
#
# WHAT IT DOES:
#   After DistilBERT classifies a user message into one of 8 labels,
#   this module queries the right memory for that specific label type
#   and returns an enriched context dict that the retrieval system uses.
#
# WHY THIS MATTERS:
#   Different labels need completely different information from memory.
#   ATTRIBUTE_QUESTION → needs the currently discussed items
#   REFINEMENT         → needs current recommendations + full preferences
#   EXPLANATION_WHY    → needs the stored explanation for that item
#   INITIAL_REQUEST    → needs long-term preferences + purchase history
#   FEEDBACK           → needs the item being reacted to (no retrieval)
#   CHITCHAT           → needs nothing (no retrieval at all)
#
# THE ENRICHED OUTPUT is a structured dict that your retrieval/RAG system
# consumes directly. It contains everything needed to either:
#   a) Run a new catalog search (FULL retrieval)
#   b) Look up details of existing items (PARTIAL retrieval)
#   c) Skip retrieval entirely (NO retrieval)
#
# ALSO handles:
#   - Updating dialogue state after enrichment
#     (e.g. updating hard_constraints after REFINEMENT)
#   - Triggering preference updates after FEEDBACK turns
#   - Preparing the input query that goes to RAG

import json
from typing import Optional

from memory.db.mongo import get_db
from memory.db.redis_client import get_redis
from memory.core.session_manager import SessionManager
from memory.core.turn_manager import TurnManager
from memory.core.user_manager import UserManager
from memory.models.schemas import DialogueState, ItemInContext, now_utc


def _session_state_key(session_id: str) -> str:
    return f"session:{session_id}:state"


class EnrichmentLayer:
    """
    Assembles memory context after DistilBERT classification.

    Usage in your FastAPI endpoint:
        enricher = EnrichmentLayer()
        enriched = await enricher.enrich(
            label="ATTRIBUTE_QUESTION",
            retrieval_strategy="PARTIAL",
            session_id="sess_abc",
            user_id="user_123",
            current_message="What material is it?",
            entities={}
        )
        # enriched is now ready to pass to your RAG system
    """

    def __init__(self):
        self.session_mgr = SessionManager()
        self.turn_mgr    = TurnManager()
        self.user_mgr    = UserManager()

    async def enrich(
        self,
        label: str,
        retrieval_strategy: str,
        session_id: str,
        user_id: str,
        current_message: str,
        entities: dict
    ) -> dict:
        """
        Main entry point. Called after every DistilBERT classification.

        Args:
            label:              DistilBERT label e.g. "ATTRIBUTE_QUESTION"
            retrieval_strategy: "FULL", "PARTIAL", or "NO"
            session_id:         Active session ID
            user_id:            User ID
            current_message:    The user's current message (cleaned text)
            entities:           Entities extracted from the current message
                                e.g. {"colour_group_name": "Black",
                                      "product_type_name": "Dress",
                                      "price_max": 50.0}

        Returns:
            An enriched context dict containing:
                label:              The classification label
                retrieval_strategy: "FULL" | "PARTIAL" | "NO"
                current_message:    The original user message
                memory_context:     Label-specific memory data
                retrieval_query:    Structured query for your RAG system
                                    (None if retrieval_strategy is "NO")
                side_effects:       List of memory updates that were triggered
        """

        # Always get the current dialogue state — every label needs it
        state = await self.session_mgr.get_dialogue_state(session_id)

        # Route to the right enrichment function based on label
        if label == "INITIAL_REQUEST":
            return await self._enrich_initial_request(
                session_id, user_id, current_message, entities, state
            )

        elif label == "REFINEMENT":
            return await self._enrich_refinement(
                session_id, user_id, current_message, entities, state
            )

        elif label == "ATTRIBUTE_QUESTION":
            return await self._enrich_attribute_question(
                session_id, user_id, current_message, entities, state
            )

        elif label == "EXPLANATION_WHY":
            return await self._enrich_explanation_why(
                session_id, user_id, current_message, entities, state
            )

        elif label == "COMPARISON":
            return await self._enrich_comparison(
                session_id, user_id, current_message, entities, state
            )

        elif label == "SELECTION_REFERENCE":
            return await self._enrich_selection_reference(
                session_id, user_id, current_message, entities, state
            )

        elif label == "FEEDBACK":
            return await self._enrich_feedback(
                session_id, user_id, current_message, entities, state
            )

        elif label == "CHITCHAT":
            return await self._enrich_chitchat(
                session_id, user_id, current_message
            )

        else:
            # Unknown label — treat as chitchat (safe fallback)
            return await self._enrich_chitchat(
                session_id, user_id, current_message
            )

    # ── Label-specific enrichment functions ──────────────────────────────────

    async def _enrich_initial_request(
        self,
        session_id: str,
        user_id: str,
        current_message: str,
        entities: dict,
        state: DialogueState
    ) -> dict:
        """
        INITIAL_REQUEST → FULL retrieval

        User is starting a fresh search. We need:
        - Long-term preferences to personalise the catalog search
        - Purchase history summary for additional personalisation
        - Any items rejected earlier this session to exclude them
        - The entities extracted from this message as new hard constraints

        Side effects:
        - Updates dialogue state with new hard constraints from entities
        - Creates/reinforces preferences from entities
        """
        side_effects = []

        # Load long-term preference summary
        pref_summary = await self.user_mgr.get_preference_summary(user_id)

        # Extract hard constraints from entities in this message
        # These override anything from previous turns
        new_hard_constraints = {}
        entity_field_map = {
            "product_type_name": "product_type_name",
            "colour_group_name": "colour_group_name",
            "index_group_name":  "index_group_name",
            "section_name":      "section_name",
            "price_max":         "price_max",
            "price_min":         "price_min",
            "style":             "style",
            "occasion":          "occasion"
        }
        for key, val in entities.items():
            if key in entity_field_map and val:
                new_hard_constraints[key] = val

        # Update dialogue state with new constraints
        if new_hard_constraints:
            await self.session_mgr.update_dialogue_state(
                session_id,
                {"hard_constraints": new_hard_constraints}
            )
            side_effects.append(f"Updated hard_constraints: {new_hard_constraints}")

        # Update long-term preferences from entities (explicit source, positive)
        pref_entities = {
            k: v for k, v in entities.items()
            if k not in ("price_max", "price_min")
        }
        if pref_entities:
            await self.user_mgr.update_preferences_from_entities(
                user_id=user_id,
                entities=pref_entities,
                sentiment=0.8,
                source="explicit",
                confidence=0.85
            )
            side_effects.append(f"Preferences updated from entities")

        # Build the retrieval query for your RAG system
        retrieval_query = self._build_full_retrieval_query(
            current_message=current_message,
            hard_constraints={
                **pref_summary.get("hard_constraints", {}),
                **new_hard_constraints
            },
            preference_boosts=pref_summary.get("liked_attributes", []),
            disliked_values=pref_summary.get("disliked_values", {}),
            excluded_article_ids=state.rejected_items
        )

        return {
            "label":              "INITIAL_REQUEST",
            "retrieval_strategy": "FULL",
            "current_message":    current_message,
            "memory_context": {
                "long_term_preferences":   pref_summary["liked_attributes"],
                "disliked_values":         pref_summary["disliked_values"],
                "style_profile":           pref_summary["style_profile"],
                "purchase_summary":        pref_summary["purchase_summary"],
                "top_product_types":       pref_summary["top_product_types"],
                "top_colours":             pref_summary["top_colours"],
                "new_constraints":         new_hard_constraints,
                "excluded_article_ids":    state.rejected_items,
            },
            "retrieval_query": retrieval_query,
            "side_effects":    side_effects
        }

    async def _enrich_refinement(
        self,
        session_id: str,
        user_id: str,
        current_message: str,
        entities: dict,
        state: DialogueState
    ) -> dict:
        """
        REFINEMENT → FULL retrieval

        User is changing or adding a preference. We need:
        - What was previously recommended (to understand what they are changing from)
        - Current hard/soft constraints (to understand the baseline)
        - Updated constraints from this message
        - Rejection history (to not re-recommend rejected items)
        - Full preference profile for the new search

        Side effects:
        - Updates hard_constraints in dialogue state
        - Updates long-term preferences with new signal
        """
        side_effects = []

        # Get both the preference summary and current state
        pref_summary = await self.user_mgr.get_preference_summary(user_id)
        current_items = state.currently_discussing

        # Extract new constraints from this refinement message
        new_constraints = {}
        for key, val in entities.items():
            if val and key not in ("style", "occasion"):
                new_constraints[key] = val

        # Merge new constraints on top of existing ones
        merged_constraints = {
            **state.hard_constraints,
            **new_constraints
        }

        # Update dialogue state
        if new_constraints:
            await self.session_mgr.update_dialogue_state(
                session_id,
                {"hard_constraints": merged_constraints}
            )
            side_effects.append(f"Merged constraints: {merged_constraints}")

        # Update preferences with medium confidence
        # (refinement signals are slightly less certain than direct statements)
        pref_entities = {
            k: v for k, v in entities.items()
            if k not in ("price_max", "price_min")
        }
        if pref_entities:
            await self.user_mgr.update_preferences_from_entities(
                user_id=user_id,
                entities=pref_entities,
                sentiment=0.75,
                source="explicit",
                confidence=0.80
            )
            side_effects.append("Preferences updated from refinement entities")

        retrieval_query = self._build_full_retrieval_query(
            current_message=current_message,
            hard_constraints=merged_constraints,
            preference_boosts=pref_summary.get("liked_attributes", []),
            disliked_values=pref_summary.get("disliked_values", {}),
            excluded_article_ids=state.rejected_items
        )

        return {
            "label":              "REFINEMENT",
            "retrieval_strategy": "FULL",
            "current_message":    current_message,
            "memory_context": {
                "previous_items":      {
                    "item_a": current_items.get("item_a"),
                    "item_b": current_items.get("item_b")
                },
                "previous_constraints":  state.hard_constraints,
                "updated_constraints":   merged_constraints,
                "new_changes":           new_constraints,
                "soft_constraints":      state.soft_constraints,
                "rejected_items":        state.rejected_items,
                "long_term_preferences": pref_summary["liked_attributes"],
                "disliked_values":       pref_summary["disliked_values"],
            },
            "retrieval_query": retrieval_query,
            "side_effects":    side_effects
        }

    async def _enrich_attribute_question(
        self,
        session_id: str,
        user_id: str,
        current_message: str,
        entities: dict,
        state: DialogueState
    ) -> dict:
        """
        ATTRIBUTE_QUESTION → PARTIAL retrieval

        User is asking about a specific property of an item already in context.
        We do NOT need a new catalog search — just look up the item's details.

        What we need:
        - The items currently being discussed (item_a and item_b)
        - The full article details from PostgreSQL (passed as a query to fetch)
        - Any relevant user preferences for that attribute type

        No side effects — asking a question does not change preferences.
        """
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        # Identify which item the question is about
        # Default to item_a (the first/primary recommended item)
        target_item = item_a

        # Check if the message references item_b specifically
        msg_lower = current_message.lower()
        if any(phrase in msg_lower for phrase in [
            "second", "option 2", "the other", "item b",
            item_b.prod_name.lower() if item_b else ""
        ]):
            target_item = item_b

        return {
            "label":              "ATTRIBUTE_QUESTION",
            "retrieval_strategy": "PARTIAL",
            "current_message":    current_message,
            "memory_context": {
                "target_item":   target_item.model_dump() if target_item else None,
                "item_a":        item_a.model_dump() if item_a else None,
                "item_b":        item_b.model_dump() if item_b else None,
                "question_about": self._identify_attribute_topic(current_message)
            },
            "retrieval_query": {
                "action":     "lookup_item_attribute",
                "article_id": target_item.article_id if target_item else None,
                "attribute":  self._identify_attribute_topic(current_message),
                "question":   current_message
            },
            "side_effects": []
        }

    async def _enrich_explanation_why(
        self,
        session_id: str,
        user_id: str,
        current_message: str,
        entities: dict,
        state: DialogueState
    ) -> dict:
        """
        EXPLANATION_WHY → PARTIAL retrieval

        User is asking why a specific item was recommended.
        We need to retrieve the stored explanation for that item
        and the preferences that drove the recommendation.

        What we need:
        - The explanation record from the explanations collection
        - The user preference profile (to explain which preferences matched)
        - The item currently in focus
        """
        db = get_db()
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")

        # Try to fetch the stored explanation for item_a
        existing_explanation = None
        if item_a:
            expl_doc = await db.explanations.find_one(
                {
                    "session_id": session_id,
                    "article_id": item_a.article_id
                },
                sort=[("created_at", -1)]  # Most recent explanation first
            )
            if expl_doc:
                expl_doc.pop("_id", None)
                existing_explanation = expl_doc

        # Get user preferences to explain the match
        pref_summary = await self.user_mgr.get_preference_summary(user_id)

        return {
            "label":              "EXPLANATION_WHY",
            "retrieval_strategy": "PARTIAL",
            "current_message":    current_message,
            "memory_context": {
                "item_in_focus":        item_a.model_dump() if item_a else None,
                "existing_explanation": existing_explanation,
                "matched_preferences":  pref_summary["liked_attributes"],
                "style_profile":        pref_summary["style_profile"],
                "hard_constraints":     state.hard_constraints
            },
            "retrieval_query": {
                "action":        "generate_explanation",
                "article_id":    item_a.article_id if item_a else None,
                "question":      current_message,
                "prior_claims":  (
                    existing_explanation.get("claims", [])
                    if existing_explanation else []
                )
            },
            "side_effects": []
        }

    async def _enrich_comparison(
        self,
        session_id: str,
        user_id: str,
        current_message: str,
        entities: dict,
        state: DialogueState
    ) -> dict:
        """
        COMPARISON → PARTIAL retrieval

        User is comparing two items already in context.
        We need both items' full details and user preferences
        (to explain which one better matches their profile).

        No new catalog search needed — items are already known.
        """
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        pref_summary = await self.user_mgr.get_preference_summary(user_id)

        return {
            "label":              "COMPARISON",
            "retrieval_strategy": "PARTIAL",
            "current_message":    current_message,
            "memory_context": {
                "item_a":               item_a.model_dump() if item_a else None,
                "item_b":               item_b.model_dump() if item_b else None,
                "user_preferences":     pref_summary["liked_attributes"],
                "hard_constraints":     state.hard_constraints,
                "comparison_dimension": self._identify_comparison_dimension(
                    current_message
                )
            },
            "retrieval_query": {
                "action":               "compare_items",
                "article_id_a":         item_a.article_id if item_a else None,
                "article_id_b":         item_b.article_id if item_b else None,
                "comparison_question":  current_message,
                "user_preference_weights": {
                    p["attribute_name"]: p["weight"]
                    for p in pref_summary["liked_attributes"]
                }
            },
            "side_effects": []
        }

    async def _enrich_selection_reference(
        self,
        session_id: str,
        user_id: str,
        current_message: str,
        entities: dict,
        state: DialogueState
    ) -> dict:
        """
        SELECTION_REFERENCE → PARTIAL retrieval

        User is pointing at a specific item using a pronoun or ordinal
        like "the first one", "that one", "the blue one".
        We resolve which item they mean and return its full details.

        Side effects:
        - May update currently_discussing to focus on the selected item
        """
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        # Resolve which item the user is referring to
        selected_item = self._resolve_item_reference(
            current_message, item_a, item_b
        )

        # Update dialogue state to reflect the focused item
        side_effects = []
        if selected_item and selected_item == item_b:
            # User selected item_b — swap so item_a is always the focus
            await self.session_mgr.update_dialogue_state(
                session_id,
                {"currently_discussing": {
                    "item_a": item_b.model_dump() if item_b else None,
                    "item_b": item_a.model_dump() if item_a else None
                }}
            )
            side_effects.append("Swapped item focus to selected item")

        return {
            "label":              "SELECTION_REFERENCE",
            "retrieval_strategy": "PARTIAL",
            "current_message":    current_message,
            "memory_context": {
                "selected_item":  (
                    selected_item.model_dump() if selected_item else None
                ),
                "item_a":         item_a.model_dump() if item_a else None,
                "item_b":         item_b.model_dump() if item_b else None,
                "resolution_confidence": (
                    "high" if selected_item else "low"
                )
            },
            "retrieval_query": {
                "action":     "get_item_details",
                "article_id": selected_item.article_id if selected_item else None,
                "question":   current_message
            },
            "side_effects": side_effects
        }

    async def _enrich_feedback(
        self,
        session_id: str,
        user_id: str,
        current_message: str,
        entities: dict,
        state: DialogueState
    ) -> dict:
        """
        FEEDBACK → NO retrieval

        User is reacting to a recommendation (positive, negative, or neutral).
        No new search needed — just update memory based on the reaction.

        Side effects (the most important side effects in the whole system):
        - Positive feedback → reinforce preferences of the accepted item
        - Negative feedback → add dislikes for the rejected item's attributes
        - Update accepted_items or rejected_items in dialogue state
        - Update recommendation outcome in recommendations collection
        """
        db = get_db()
        side_effects = []

        current_items = state.currently_discussing
        item_a = current_items.get("item_a")

        # Determine if feedback is positive or negative
        sentiment_score = self._classify_feedback_sentiment(current_message)
        is_positive = sentiment_score > 0.0

        # Update preference memory based on feedback
        if item_a:
            # Build entities from the item's attributes
            item_entities = {
                "colour_group_name":  item_a.colour_group_name,
                "product_type_name":  item_a.product_type_name,
            }
            if item_a.index_group_name:
                item_entities["index_group_name"] = item_a.index_group_name
            if item_a.garment_group_name:
                item_entities["garment_group_name"] = item_a.garment_group_name

            await self.user_mgr.update_preferences_from_entities(
                user_id=user_id,
                entities=item_entities,
                sentiment=sentiment_score,
                source="implicit",   # Feedback is implicit signal
                confidence=0.80
            )
            side_effects.append(
                f"Preferences updated from feedback "
                f"({'positive' if is_positive else 'negative'}): "
                f"{list(item_entities.keys())}"
            )

            # Update dialogue state
            if is_positive:
                updated_accepted = state.accepted_items + [item_a.article_id]
                await self.session_mgr.update_dialogue_state(
                    session_id,
                    {"accepted_items": updated_accepted}
                )
                side_effects.append(
                    f"Added {item_a.article_id} to accepted_items"
                )

                # Update purchase summary if it is a strong acceptance
                if sentiment_score > 0.7:
                    await self.user_mgr.update_purchase_summary(
                        user_id=user_id,
                        article_data={
                            "product_type_name":  item_a.product_type_name,
                            "colour_group_name":  item_a.colour_group_name,
                            "index_group_name":   item_a.index_group_name
                        }
                    )
                    side_effects.append("Purchase summary updated")

            else:
                updated_rejected = state.rejected_items + [item_a.article_id]
                await self.session_mgr.update_dialogue_state(
                    session_id,
                    {"rejected_items": updated_rejected}
                )
                side_effects.append(
                    f"Added {item_a.article_id} to rejected_items"
                )

            # Update the recommendation outcome in MongoDB
            await db.recommendations.update_one(
                {
                    "session_id": session_id,
                    "items.article_id": item_a.article_id,
                    "outcome": "pending"
                },
                {
                    "$set": {
                        "outcome": "accepted" if is_positive else "rejected"
                    }
                }
            )
            side_effects.append(
                f"Recommendation outcome: "
                f"{'accepted' if is_positive else 'rejected'}"
            )

        return {
            "label":              "FEEDBACK",
            "retrieval_strategy": "NO",
            "current_message":    current_message,
            "memory_context": {
                "item_reacted_to": item_a.model_dump() if item_a else None,
                "sentiment_score": sentiment_score,
                "is_positive":     is_positive,
                "feedback_type": (
                    "positive" if sentiment_score > 0.3
                    else "negative" if sentiment_score < -0.3
                    else "neutral"
                )
            },
            "retrieval_query": None,   # No retrieval for feedback
            "side_effects":    side_effects
        }

    async def _enrich_chitchat(
        self,
        session_id: str,
        user_id: str,
        current_message: str
    ) -> dict:
        """
        CHITCHAT → NO retrieval

        Greeting, thanks, off-topic message.
        Nothing to fetch from memory — just respond conversationally.
        """
        return {
            "label":              "CHITCHAT",
            "retrieval_strategy": "NO",
            "current_message":    current_message,
            "memory_context":     {},
            "retrieval_query":    None,
            "side_effects":       []
        }

    # ── Helper methods ────────────────────────────────────────────────────────

    def _build_full_retrieval_query(
        self,
        current_message: str,
        hard_constraints: dict,
        preference_boosts: list,
        disliked_values: dict,
        excluded_article_ids: list
    ) -> dict:
        """
        Builds the structured query dict for your RAG/retrieval system.
        This is the format your GNN RAG, Text RAG, or Multimodal RAG
        will receive when it needs to search the catalog.

        hard_constraints → mandatory WHERE filters
        preference_boosts → ranking weights (soft boosters)
        disliked_values   → values to penalise in ranking
        excluded_article_ids → articles to exclude entirely
        """
        return {
            "action":    "catalog_search",
            "user_query": current_message,

            # These become strict WHERE conditions in your PostgreSQL query
            # or mandatory filters in your vector search
            "filters": {
                k: v for k, v in hard_constraints.items()
                if v is not None
            },

            # These boost ranking scores for matching items
            # weight is already computed as sentiment × confidence × decay
            "preference_boosts": [
                {
                    "attribute": p["attribute_name"],
                    "value":     p["attribute_value"],
                    "weight":    p["weight"]
                }
                for p in preference_boosts
                if p["weight"] > 0.3  # Only include meaningful boosts
            ],

            # These penalise items with disliked attribute values
            "penalties": disliked_values,

            # These article_ids are completely excluded from results
            "exclude_article_ids": excluded_article_ids
        }

    def _identify_attribute_topic(self, message: str) -> str:
        """
        Identifies what attribute the user is asking about.
        Maps to column names from sample_articles.csv where possible.
        """
        msg = message.lower()

        if any(w in msg for w in ["material", "fabric", "made of", "cotton",
                                   "linen", "polyester", "wash", "care"]):
            return "material_and_care"

        if any(w in msg for w in ["colour", "color", "shade", "hue"]):
            return "colour_group_name"

        if any(w in msg for w in ["pocket", "pockets"]):
            return "pockets"

        if any(w in msg for w in ["size", "fit", "slim", "loose", "relaxed",
                                   "fitted", "run small", "run large"]):
            return "sizing_and_fit"

        if any(w in msg for w in ["sleeve", "neckline", "collar", "length",
                                   "style", "cut", "design"]):
            return "design_details"

        if any(w in msg for w in ["price", "cost", "expensive", "cheap"]):
            return "price"

        return "general_details"

    def _identify_comparison_dimension(self, message: str) -> str:
        """Identifies what dimension the user wants to compare on."""
        msg = message.lower()

        if any(w in msg for w in ["cheaper", "expensive", "price", "cost",
                                   "value", "afford"]):
            return "price"

        if any(w in msg for w in ["quality", "better", "best", "recommend"]):
            return "quality_and_recommendation"

        if any(w in msg for w in ["casual", "formal", "smart", "style",
                                   "occasion", "wear"]):
            return "style_and_occasion"

        if any(w in msg for w in ["material", "fabric", "comfortable"]):
            return "material"

        if any(w in msg for w in ["colour", "color"]):
            return "colour"

        return "overall"

    def _resolve_item_reference(
        self,
        message: str,
        item_a: Optional[ItemInContext],
        item_b: Optional[ItemInContext]
    ) -> Optional[ItemInContext]:
        """
        Resolves a vague reference like "the first one", "the blue one",
        "option 2" to a specific item.
        Returns the resolved ItemInContext, or item_a as default.
        """
        msg = message.lower()

        # Explicit ordinal references to item_b
        if any(phrase in msg for phrase in [
            "second", "option 2", "the other", "second one",
            "the 2nd", "number two", "item 2"
        ]):
            return item_b

        # Colour-based resolution
        if item_b and item_b.colour_group_name.lower() in msg:
            return item_b
        if item_a and item_a.colour_group_name.lower() in msg:
            return item_a

        # Name-based resolution
        if item_b and item_b.prod_name.lower() in msg:
            return item_b
        if item_a and item_a.prod_name.lower() in msg:
            return item_a

        # Default: item_a is always the primary focus
        return item_a

    def _classify_feedback_sentiment(self, message: str) -> float:
        """
        Classifies feedback message sentiment as a float on [-1.0, 1.0].

        Positive feedback  → +0.8 to +1.0 (reinforce preferences)
        Neutral feedback   → -0.1 to +0.1 (weak signal, small update)
        Negative feedback  → -0.6 to -0.9 (strengthen dislikes)

        Uses keyword matching — simple and transparent.
        Your system could replace this with a sentiment model later.
        """
        msg = message.lower()

        strong_positive = [
            "love", "perfect", "exactly", "amazing", "excellent",
            "wonderful", "great choice", "this is it", "i'll take",
            "yes please", "definitely", "absolutely"
        ]
        mild_positive = [
            "like", "nice", "good", "looks good", "that works",
            "suits me", "i'm happy", "okay i'll go", "i'll go with",
            "yes", "sure", "alright"
        ]
        neutral = [
            "maybe", "possibly", "could work", "not sure",
            "let me think", "i'll think", "on the fence"
        ]
        mild_negative = [
            "not really", "not my", "don't think", "hmm",
            "not convinced", "not keen", "not for me"
        ]
        strong_negative = [
            "hate", "don't like", "no", "not what", "doesn't suit",
            "not right", "wrong", "bad", "ugly", "horrible",
            "not impressed", "disappointed", "nah", "awful"
        ]

        if any(w in msg for w in strong_positive):
            return 0.9
        if any(w in msg for w in mild_positive):
            return 0.6
        if any(w in msg for w in strong_negative):
            return -0.8
        if any(w in msg for w in mild_negative):
            return -0.5
        if any(w in msg for w in neutral):
            return 0.0

        # Default: mild positive (ambiguous messages lean positive)
        return 0.3