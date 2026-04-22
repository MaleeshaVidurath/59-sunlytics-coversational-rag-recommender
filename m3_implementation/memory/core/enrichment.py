# m3_implementation/memory/core/enrichment.py
#
# The enrichment layer — bridges DistilBERT classification and retrieval.
#
# WHAT IT DOES:
#   After DistilBERT classifies a user message into one of 8 labels,
#   this module queries the right memory for that specific label type
#   and returns a standardised PipelineOutput structure that every
#   RAG module consumes identically.
#
# OUTPUT STRUCTURE (always the same shape regardless of label):
#   {
#       action:            str        — what the RAG should do
#       retrieval_strategy: str       — FULL | PARTIAL | NO
#       user_message:      str        — original user message
#       items_in_context:  dict       — item_a, item_b (may be None)
#       exclude_ids:       list       — rejected article_ids
#       payload:           dict       — action-specific data (see ACTION TAXONOMY)
#   }
#
# ACTION TAXONOMY:
#   "catalog_search"        ← INITIAL_REQUEST, REFINEMENT
#   "item_attribute_lookup" ← ATTRIBUTE_QUESTION
#   "explanation_generate"  ← EXPLANATION_WHY
#   "item_compare"          ← COMPARISON
#   "item_detail_lookup"    ← SELECTION_REFERENCE
#   None                    ← FEEDBACK, CHITCHAT (retrieval_input is None)
#
# HYBRID SIMILARITY (keyword + vector):
#   Three helper methods use keyword matching first (fast, catches obvious cases)
#   and fall back to sentence embedding similarity (all-MiniLM-L6-v2) when
#   keyword matching is inconclusive. This handles synonyms and paraphrases
#   that keyword lists miss (e.g. "what is it constructed from" → material).

import json
import os
from typing import Optional

from memory.db.mongo import get_db
from memory.db.redis_client import get_redis
from memory.core.session_manager import SessionManager
from memory.core.turn_manager import TurnManager
from memory.core.user_manager import UserManager
from memory.models.schemas import DialogueState, ItemInContext, now_utc


def _session_state_key(session_id: str) -> str:
    return f"session:{session_id}:state"


# ── Semantic similarity model (loaded once at module level) ───────────────────
# Uses all-MiniLM-L6-v2: 80MB, ~3-5ms per comparison on CPU.
# Loaded lazily on first use to avoid slowing down import time.
_similarity_model = None

def _get_similarity_model():
    """Loads the sentence embedding model on first call, then caches it."""
    global _similarity_model
    if _similarity_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[EnrichmentLayer] Sentence similarity model loaded.")
        except Exception as e:
            print(f"[EnrichmentLayer] Could not load similarity model: {e}")
            print("[EnrichmentLayer] Falling back to keyword-only matching.")
            _similarity_model = "unavailable"
    return _similarity_model if _similarity_model != "unavailable" else None


def _cosine_similarity(a, b) -> float:
    """Computes cosine similarity between two numpy vectors."""
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _best_match(query: str, candidates: list[str]) -> tuple[str, float]:
    """
    Returns the best matching candidate string and its similarity score.
    Uses sentence embeddings to find semantic similarity.
    """
    model = _get_similarity_model()
    if model is None:
        return candidates[0], 0.0

    embeddings = model.encode([query] + candidates)
    query_emb = embeddings[0]
    best_label, best_score = candidates[0], -1.0
    for i, candidate in enumerate(candidates):
        score = _cosine_similarity(query_emb, embeddings[i + 1])
        if score > best_score:
            best_score = score
            best_label = candidate
    return best_label, best_score


# ── Attribute topic detection (hybrid keyword + vector) ───────────────────────
# These keyword lists catch the obvious cases instantly.
# If no keyword matches, vector similarity handles synonyms/paraphrases.

_ATTRIBUTE_KEYWORDS = {
    "material_and_care": [
        "material", "fabric", "made of", "made from", "constructed from",
        "cotton", "linen", "polyester", "silk", "wool", "jersey", "denim",
        "synthetic", "natural", "breathable", "wash", "care", "machine wash",
        "dry clean", "what is it", "what's it"
    ],
    "colour": [
        "colour", "color", "shade", "hue", "tone", "tint", "available in",
        "come in", "other colours", "other colors"
    ],
    "sizing_and_fit": [
        "size", "fit", "slim", "loose", "relaxed", "fitted", "oversized",
        "runs small", "runs large", "true to size", "measurements", "dimensions"
    ],
    "pockets": [
        "pocket", "pockets", "storage", "carry"
    ],
    "design_details": [
        "sleeve", "neckline", "collar", "length", "style", "cut", "design",
        "pattern", "print", "embroidery", "buttons", "zipper", "lining"
    ],
    "price": [
        "price", "cost", "expensive", "cheap", "affordable", "value",
        "how much", "what does it cost"
    ],
    "availability": [
        "stock", "available", "in stock", "out of stock", "when", "restock"
    ],
}

# Anchor phrases for vector similarity fallback.
# One descriptive sentence per topic — the model compares the user's
# message against these descriptions.
_ATTRIBUTE_ANCHORS = {
    "material_and_care": "What material or fabric is this item made from and how do I care for it",
    "colour":            "What colour is this item and what colour options are available",
    "sizing_and_fit":    "What size should I get and how does this item fit",
    "pockets":           "Does this item have pockets or storage",
    "design_details":    "What does the design look like including neckline sleeve and style details",
    "price":             "How much does this item cost",
    "availability":      "Is this item in stock and available to buy",
    "general_details":   "Tell me more about this item in general",
}


def _identify_attribute_topic(message: str) -> str:
    """
    Identifies what attribute the user is asking about.
    Step 1: keyword matching (fast, handles obvious cases)
    Step 2: vector similarity fallback (handles synonyms/paraphrases)
    """
    msg = message.lower()

    # Step 1: keyword matching
    for topic, keywords in _ATTRIBUTE_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            return topic

    # Step 2: vector similarity fallback
    anchors = list(_ATTRIBUTE_ANCHORS.keys())
    anchor_sentences = list(_ATTRIBUTE_ANCHORS.values())
    best_topic, score = _best_match(message, anchor_sentences)
    if score > 0.30:
        return anchors[anchor_sentences.index(best_topic)]

    return "general_details"


# ── Comparison dimension detection (hybrid keyword + vector) ──────────────────

_COMPARISON_KEYWORDS = {
    "price":              ["cheaper", "expensive", "price", "cost", "value", "afford", "budget"],
    "quality":            ["quality", "better", "best", "recommend", "durable", "lasts"],
    "style_and_occasion": ["casual", "formal", "smart", "style", "occasion", "wear", "event"],
    "material":           ["material", "fabric", "comfortable", "breathable", "soft", "feel"],
    "colour":             ["colour", "color", "shade"],
    "fit":                ["fit", "size", "slim", "loose", "tight"],
    "overall":            ["overall", "general", "difference", "compare", "which", "versus", "vs"],
}

_COMPARISON_ANCHORS = {
    "price":              "Which item is cheaper or better value for money",
    "quality":            "Which item is better quality and more durable",
    "style_and_occasion": "Which item is more suitable for a casual or formal occasion",
    "material":           "Which item has better fabric or is more comfortable to wear",
    "colour":             "Which item has a better colour option",
    "fit":                "Which item has a better fit or sizing",
    "overall":            "Compare these two items overall and tell me which is better",
}


def _identify_comparison_dimension(message: str) -> str:
    """
    Identifies what dimension the user wants to compare on.
    Hybrid keyword → vector similarity.
    """
    msg = message.lower()

    for dimension, keywords in _COMPARISON_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            return dimension

    anchors = list(_COMPARISON_ANCHORS.keys())
    anchor_sentences = list(_COMPARISON_ANCHORS.values())
    best_dim, score = _best_match(message, anchor_sentences)
    if score > 0.30:
        return anchors[anchor_sentences.index(best_dim)]

    return "overall"


# ── Feedback sentiment classification (hybrid keyword + vector) ───────────────

_FEEDBACK_KEYWORDS = {
    "strong_positive": [
        "love", "perfect", "exactly what", "amazing", "excellent",
        "wonderful", "great choice", "this is it", "i'll take", "i will take",
        "yes please", "definitely", "absolutely", "brilliant", "fantastic"
    ],
    "mild_positive": [
        "like", "nice", "good", "looks good", "that works", "suits me",
        "happy with", "okay i'll", "i'll go with", "i'll go", "yes",
        "alright", "fine", "sounds good", "works for me"
    ],
    "neutral": [
        "maybe", "possibly", "could work", "not sure", "let me think",
        "i'll think", "on the fence", "unsure", "perhaps"
    ],
    "mild_negative": [
        "not really", "not my", "don't think so", "not convinced",
        "not keen", "not for me", "not ideal", "bit disappointed"
    ],
    "strong_negative": [
        "hate", "don't like", "i dislike", "no not", "not what i",
        "doesn't suit", "not right", "wrong", "bad", "ugly", "horrible",
        "not impressed", "disappointed", "nah", "awful", "terrible",
        "not what i wanted", "not happy"
    ],
}

# Sentiment scores for each bucket
_FEEDBACK_SCORES = {
    "strong_positive": 0.9,
    "mild_positive":   0.6,
    "neutral":         0.0,
    "mild_negative":  -0.5,
    "strong_negative": -0.8,
}

_FEEDBACK_ANCHORS = {
    "strong_positive": "I love this item it is perfect exactly what I wanted",
    "mild_positive":   "I like this item it looks good and works for me",
    "neutral":         "Maybe this could work I am not sure yet",
    "mild_negative":   "This is not really my style I am not convinced",
    "strong_negative": "I hate this it is not what I wanted at all",
}


def _classify_feedback_sentiment(message: str) -> float:
    """
    Returns a sentiment score on [-1.0, 1.0] for a feedback message.
    Hybrid keyword → vector similarity.
    """
    msg = message.lower().strip()

    # Step 1: keyword matching
    for bucket, keywords in _FEEDBACK_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            return _FEEDBACK_SCORES[bucket]

    # Step 2: vector similarity fallback
    buckets = list(_FEEDBACK_ANCHORS.keys())
    anchor_sentences = list(_FEEDBACK_ANCHORS.values())
    best_bucket, score = _best_match(message, anchor_sentences)
    if score > 0.35:
        return _FEEDBACK_SCORES[buckets[anchor_sentences.index(best_bucket)]]

    # Default: mild positive (ambiguous messages lean positive)
    return 0.3


# ── Item reference resolver ───────────────────────────────────────────────────

def _resolve_item_reference(
    message: str,
    item_a: Optional[ItemInContext],
    item_b: Optional[ItemInContext]
) -> Optional[ItemInContext]:
    """
    Resolves a vague reference like "the first one", "the blue one",
    "option 2" to a specific item. Returns item_a as default.
    """
    msg = message.lower()

    # Explicit ordinal references to item_b
    if any(phrase in msg for phrase in [
        "second", "option 2", "the other", "second one",
        "the 2nd", "number two", "item 2", "2nd one"
    ]):
        return item_b

    # Colour-based resolution — check item_b first (less default)
    if item_b and item_b.colour_group_name.lower() in msg:
        return item_b
    if item_a and item_a.colour_group_name.lower() in msg:
        return item_a

    # Name-based resolution
    if item_b and item_b.prod_name.lower() in msg:
        return item_b
    if item_a and item_a.prod_name.lower() in msg:
        return item_a

    # Default: item_a is the primary focus
    return item_a


# ── Helper: build items_in_context dict ───────────────────────────────────────

def _items_dict(item_a, item_b) -> dict:
    return {
        "item_a": item_a.model_dump() if item_a else None,
        "item_b": item_b.model_dump() if item_b else None,
    }


# ── Main EnrichmentLayer class ────────────────────────────────────────────────

class EnrichmentLayer:
    """
    Assembles memory context after DistilBERT classification and returns
    a standardised output structure for every label type.

    The output structure is always:
        {
            "label":              str
            "retrieval_strategy": str
            "retrieval_input":    dict | None
            "memory_context":     dict
            "side_effects":       list[str]
        }

    retrieval_input is always None when retrieval_strategy is "NO".
    retrieval_input always has the same envelope keys regardless of action.

    Usage:
        enricher = EnrichmentLayer()
        result = await enricher.enrich(
            label="ATTRIBUTE_QUESTION",
            retrieval_strategy="PARTIAL",
            session_id="sess_abc",
            user_id="user_123",
            current_message="What material is it?",
            entities={}
        )
        # Pass result["retrieval_input"] to your RAG system
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

        Returns the standardised enrichment output dict described in the
        class docstring. The "retrieval_input" key is what your RAG
        system consumes directly.
        """
        state = await self.session_mgr.get_dialogue_state(session_id)

        if label == "INITIAL_REQUEST":
            return await self._enrich_initial_request(
                session_id, user_id, current_message, entities, state)
        elif label == "REFINEMENT":
            return await self._enrich_refinement(
                session_id, user_id, current_message, entities, state)
        elif label == "ATTRIBUTE_QUESTION":
            return await self._enrich_attribute_question(
                session_id, user_id, current_message, entities, state)
        elif label == "EXPLANATION_WHY":
            return await self._enrich_explanation_why(
                session_id, user_id, current_message, entities, state)
        elif label == "COMPARISON":
            return await self._enrich_comparison(
                session_id, user_id, current_message, entities, state)
        elif label == "SELECTION_REFERENCE":
            return await self._enrich_selection_reference(
                session_id, user_id, current_message, entities, state)
        elif label == "FEEDBACK":
            return await self._enrich_feedback(
                session_id, user_id, current_message, entities, state)
        else:
            # CHITCHAT or unknown
            return await self._enrich_chitchat(
                session_id, user_id, current_message)

    # ── Base memory context (always included regardless of label) ─────────────

    async def _base_memory_context(
        self,
        user_id: str,
        state: DialogueState,
        include_preferences: bool = True
    ) -> dict:
        """
        Builds the memory_context dict that is always present in output.
        Includes dialogue state, style profile, and optionally preferences.
        """
        ctx = {
            "dialogue_state": {
                "hard_constraints":    state.hard_constraints,
                "soft_constraints":    state.soft_constraints,
                "rejected_items":      state.rejected_items,
                "accepted_items":      state.accepted_items,
                "intent_summary":      state.intent_summary,
            },
            "long_term_preferences":  [],
            "style_profile":          {},
            "preference_summary":     {},
            "existing_explanation":   None,
        }

        if include_preferences:
            try:
                pref_summary = await self.user_mgr.get_preference_summary(user_id)
                ctx["long_term_preferences"] = pref_summary.get("liked_attributes", [])
                ctx["style_profile"]         = pref_summary.get("style_profile", {})
                ctx["preference_summary"]    = pref_summary
            except Exception:
                pass

        return ctx

    # ── Retrieval input envelope builder ──────────────────────────────────────

    def _make_retrieval_input(
        self,
        action: str,
        retrieval_strategy: str,
        user_message: str,
        item_a: Optional[ItemInContext],
        item_b: Optional[ItemInContext],
        exclude_ids: list,
        payload: dict
    ) -> dict:
        """
        Builds the standardised retrieval_input envelope.
        This is the only place retrieval_input is constructed —
        guarantees consistent shape for every label type.
        """
        return {
            "action":             action,
            "retrieval_strategy": retrieval_strategy,
            "user_message":       user_message,
            "items_in_context":   _items_dict(item_a, item_b),
            "exclude_ids":        exclude_ids,
            "payload":            payload,
        }

    # ── Label-specific enrichment methods ─────────────────────────────────────

    async def _enrich_initial_request(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        """INITIAL_REQUEST → action: catalog_search"""
        side_effects = []

        pref_summary = await self.user_mgr.get_preference_summary(user_id)

        # Extract hard constraints from entities
        new_constraints = {
            k: v for k, v in entities.items()
            if k not in ("style", "occasion") and v is not None
        }

        if new_constraints:
            await self.session_mgr.update_dialogue_state(
                session_id, {"hard_constraints": new_constraints}
            )
            side_effects.append(f"Updated hard_constraints: {new_constraints}")

        # Update long-term preferences from entities
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
            side_effects.append("Preferences updated from entities")

        merged_filters = {
            **pref_summary.get("hard_constraints", {}),
            **new_constraints
        }

        memory_ctx = await self._base_memory_context(user_id, state)

        return {
            "label":              "INITIAL_REQUEST",
            "retrieval_strategy": "FULL",
            "retrieval_input": self._make_retrieval_input(
                action="catalog_search",
                retrieval_strategy="FULL",
                user_message=current_message,
                item_a=state.currently_discussing.get("item_a"),
                item_b=state.currently_discussing.get("item_b"),
                exclude_ids=state.rejected_items,
                payload={
                    # Hard constraints — mandatory WHERE conditions
                    "filters": merged_filters,
                    # Soft constraints from session state (style, occasion)
                    # NOT mandatory — use to contextualise search and ranking
                    "soft_constraints": {
                        k: v for k, v in state.soft_constraints.items()
                        if v is not None
                    },
                    # Long-term preference boosts — ranking weights
                    "preference_boosts": [
                        {
                            "attribute": p["attribute_name"],
                            "value":     p["attribute_value"],
                            "weight":    p["weight"]
                        }
                        for p in pref_summary.get("liked_attributes", [])
                        if p["weight"] > 0.3
                    ],
                    # Purchase history hints — from pre-loaded transaction history
                    "purchase_history_hints": await self._get_purchase_hints(user_id),
                    # Disliked values — rank lower in results
                    "penalties": pref_summary.get("disliked_values", {}),
                }
            ),
            "memory_context": memory_ctx,
            "side_effects":   side_effects,
        }

    async def _enrich_refinement(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        """REFINEMENT → action: catalog_search (same as INITIAL_REQUEST)"""
        side_effects = []

        pref_summary = await self.user_mgr.get_preference_summary(user_id)

        # New constraints from this message, merged on top of existing ones
        new_constraints = {
            k: v for k, v in entities.items()
            if k not in ("style", "occasion") and v is not None
        }
        merged_constraints = {**state.hard_constraints, **new_constraints}

        if new_constraints:
            await self.session_mgr.update_dialogue_state(
                session_id, {"hard_constraints": merged_constraints}
            )
            side_effects.append(f"Merged constraints: {merged_constraints}")

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
            side_effects.append("Preferences updated from refinement")

        memory_ctx = await self._base_memory_context(user_id, state)
        memory_ctx["previous_constraints"] = state.hard_constraints
        memory_ctx["new_changes"]          = new_constraints

        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        return {
            "label":              "REFINEMENT",
            "retrieval_strategy": "FULL",
            "retrieval_input": self._make_retrieval_input(
                action="catalog_search",
                retrieval_strategy="FULL",
                user_message=current_message,
                item_a=item_a,
                item_b=item_b,
                exclude_ids=state.rejected_items,
                payload={
                    # Hard constraints — merged old + new from this turn
                    "filters": merged_constraints,
                    # Soft constraints from session (style, occasion)
                    "soft_constraints": {
                        k: v for k, v in state.soft_constraints.items()
                        if v is not None
                    },
                    # Long-term preference boosts
                    "preference_boosts": [
                        {
                            "attribute": p["attribute_name"],
                            "value":     p["attribute_value"],
                            "weight":    p["weight"]
                        }
                        for p in pref_summary.get("liked_attributes", [])
                        if p["weight"] > 0.3
                    ],
                    # Purchase history hints — from pre-loaded transaction history
                    "purchase_history_hints": await self._get_purchase_hints(user_id),
                    # Disliked values
                    "penalties": pref_summary.get("disliked_values", {}),
                }
            ),
            "memory_context": memory_ctx,
            "side_effects":   side_effects,
        }

    async def _enrich_attribute_question(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        """ATTRIBUTE_QUESTION → action: item_attribute_lookup.
        
        Guard: if no items are in context, cannot look up an attribute.
        Reclassify as INITIAL_REQUEST so the user gets a recommendation first.
        """
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        # ── Guard: no items in context ────────────────────────────────────
        if item_a is None and item_b is None:
            memory_ctx = await self._base_memory_context(user_id, state)
            memory_ctx["needs_clarification"] = True
            memory_ctx["clarification_reason"] = (
                "User asked about item properties but no items have been "
                "recommended yet. Treating as a new search request."
            )
            return {
                "label":              "INITIAL_REQUEST",
                "retrieval_strategy": "FULL",
                "retrieval_input": self._make_retrieval_input(
                    action="catalog_search",
                    retrieval_strategy="FULL",
                    user_message=current_message,
                    item_a=None, item_b=None,
                    exclude_ids=state.rejected_items,
                    payload={
                        "filters": state.hard_constraints,
                        "preference_boosts": [],
                        "penalties": {},
                    }
                ),
                "memory_context": memory_ctx,
                "side_effects":   ["Reclassified: no items in context → INITIAL_REQUEST"],
            }

        # Resolve which item the question is about
        target_item = _resolve_item_reference(current_message, item_a, item_b)

        # Identify what attribute is being asked about (hybrid similarity)
        attribute_topic = _identify_attribute_topic(current_message)

        memory_ctx = await self._base_memory_context(
            user_id, state, include_preferences=False
        )

        return {
            "label":              "ATTRIBUTE_QUESTION",
            "retrieval_strategy": "PARTIAL",
            "retrieval_input": self._make_retrieval_input(
                action="item_attribute_lookup",
                retrieval_strategy="PARTIAL",
                user_message=current_message,
                item_a=item_a,
                item_b=item_b,
                exclude_ids=state.rejected_items,
                payload={
                    "article_id":      target_item.article_id if target_item else None,
                    "attribute_topic": attribute_topic,
                }
            ),
            "memory_context": memory_ctx,
            "side_effects":   [],
        }

    async def _enrich_explanation_why(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        """EXPLANATION_WHY → action: explanation_generate"""
        db = get_db()
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        # Fetch stored explanation for item_a if it exists
        existing_explanation = None
        if item_a:
            expl_doc = await db.explanations.find_one(
                {"session_id": session_id, "article_id": item_a.article_id},
                sort=[("created_at", -1)]
            )
            if expl_doc:
                expl_doc.pop("_id", None)
                existing_explanation = expl_doc

        pref_summary = await self.user_mgr.get_preference_summary(user_id)
        memory_ctx = await self._base_memory_context(user_id, state)
        memory_ctx["existing_explanation"] = existing_explanation

        return {
            "label":              "EXPLANATION_WHY",
            "retrieval_strategy": "PARTIAL",
            "retrieval_input": self._make_retrieval_input(
                action="explanation_generate",
                retrieval_strategy="PARTIAL",
                user_message=current_message,
                item_a=item_a,
                item_b=item_b,
                exclude_ids=state.rejected_items,
                payload={
                    "article_id":   item_a.article_id if item_a else None,
                    "prior_claims": (
                        existing_explanation.get("claims", [])
                        if existing_explanation else []
                    ),
                    "matched_prefs": pref_summary.get("liked_attributes", []),
                }
            ),
            "memory_context": memory_ctx,
            "side_effects":   [],
        }

    async def _enrich_comparison(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        """COMPARISON → action: item_compare"""
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        # Identify comparison dimension using hybrid similarity
        dimension = _identify_comparison_dimension(current_message)

        pref_summary = await self.user_mgr.get_preference_summary(user_id)
        memory_ctx = await self._base_memory_context(user_id, state)

        return {
            "label":              "COMPARISON",
            "retrieval_strategy": "PARTIAL",
            "retrieval_input": self._make_retrieval_input(
                action="item_compare",
                retrieval_strategy="PARTIAL",
                user_message=current_message,
                item_a=item_a,
                item_b=item_b,
                exclude_ids=state.rejected_items,
                payload={
                    "article_id_a":        item_a.article_id if item_a else None,
                    "article_id_b":        item_b.article_id if item_b else None,
                    "comparison_dimension": dimension,
                    "preference_weights": {
                        p["attribute_name"]: p["weight"]
                        for p in pref_summary.get("liked_attributes", [])
                    },
                }
            ),
            "memory_context": memory_ctx,
            "side_effects":   [],
        }

    async def _enrich_selection_reference(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        """SELECTION_REFERENCE → action: item_detail_lookup.
        
        Guard: if no items are in context (session just started or no
        recommendations have been made yet), we cannot resolve a reference.
        Return CHITCHAT with a clarification flag instead of crashing.
        """
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        # ── Guard: no items in context ────────────────────────────────────
        # If the user says "tell me more about the first one" but no items
        # have been recommended yet, we cannot resolve the reference.
        # Return a clarification response instead of article_id: None.
        if item_a is None and item_b is None:
            memory_ctx = await self._base_memory_context(
                user_id, state, include_preferences=False
            )
            memory_ctx["needs_clarification"] = True
            memory_ctx["clarification_reason"] = (
                "User referenced an item but no items have been recommended yet."
            )
            return {
                "label":              "CHITCHAT",
                "retrieval_strategy": "NO",
                "retrieval_input":    None,
                "memory_context":     memory_ctx,
                "side_effects":       ["Reclassified: no items in context"],
            }

        selected_item = _resolve_item_reference(current_message, item_a, item_b)

        # If user selected item_b, swap so item_a is always the focus
        side_effects = []
        if selected_item and item_b and selected_item.article_id == item_b.article_id:
            await self.session_mgr.update_dialogue_state(
                session_id,
                {"currently_discussing": {
                    "item_a": item_b.model_dump() if item_b else None,
                    "item_b": item_a.model_dump() if item_a else None,
                }}
            )
            side_effects.append("Item focus swapped to selected item")

        memory_ctx = await self._base_memory_context(
            user_id, state, include_preferences=False
        )

        return {
            "label":              "SELECTION_REFERENCE",
            "retrieval_strategy": "PARTIAL",
            "retrieval_input": self._make_retrieval_input(
                action="item_detail_lookup",
                retrieval_strategy="PARTIAL",
                user_message=current_message,
                item_a=item_a,
                item_b=item_b,
                exclude_ids=state.rejected_items,
                payload={
                    "article_id": selected_item.article_id if selected_item else None,
                }
            ),
            "memory_context": memory_ctx,
            "side_effects":   side_effects,
        }

    async def _get_purchase_hints(self, user_id: str) -> dict:
        """
        Returns purchase_history_hints from the pre-loaded customer profile.
        Falls back to empty hints if no profile exists.
        """
        try:
            user = await self.user_mgr.get_user_by_id(user_id)
            if user and hasattr(user, 'purchase_history') and user.purchase_history:
                ph = user.purchase_history
                return {
                    "top_colours": [
                        c["colour"] for c in ph.get("top_colours", [])
                    ],
                    "top_product_types": [
                        t["type"] for t in ph.get("top_product_types", [])
                    ],
                    "inferred_gender":      ph.get("inferred_gender"),
                    "budget_tier":          ph.get("price_stats", {}).get("budget_tier"),
                    "preferred_price_range":ph.get("price_stats", {}).get("preferred_range"),
                    "dominant_colour":      ph.get("dominant_colour"),
                    "dominant_type":        ph.get("dominant_product_type"),
                }
        except Exception:
            pass
        return {
            "top_colours": [], "top_product_types": [],
            "inferred_gender": None, "budget_tier": None,
            "preferred_price_range": None,
            "dominant_colour": None, "dominant_type": None,
        }

    async def _enrich_feedback(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        """FEEDBACK → no retrieval. Updates memory based on sentiment."""
        db = get_db()
        side_effects = []

        current_items = state.currently_discussing
        item_a = current_items.get("item_a")

        # Classify sentiment using hybrid keyword + vector similarity
        sentiment_score = _classify_feedback_sentiment(current_message)
        is_positive     = sentiment_score > 0.0

        if item_a:
            item_entities = {
                "colour_group_name": item_a.colour_group_name,
                "product_type_name": item_a.product_type_name,
            }
            if item_a.index_group_name:
                item_entities["index_group_name"] = item_a.index_group_name
            if item_a.garment_group_name:
                item_entities["garment_group_name"] = item_a.garment_group_name

            await self.user_mgr.update_preferences_from_entities(
                user_id=user_id,
                entities=item_entities,
                sentiment=sentiment_score,
                source="implicit",
                confidence=0.80
            )
            side_effects.append(
                f"Preferences updated from feedback "
                f"({'positive' if is_positive else 'negative'}): "
                f"{list(item_entities.keys())}"
            )

            if is_positive:
                updated_accepted = state.accepted_items + [item_a.article_id]
                await self.session_mgr.update_dialogue_state(
                    session_id, {"accepted_items": updated_accepted}
                )
                side_effects.append(f"Added {item_a.article_id} to accepted_items")

                if sentiment_score > 0.7:
                    await self.user_mgr.update_purchase_summary(
                        user_id=user_id,
                        article_data={
                            "product_type_name": item_a.product_type_name,
                            "colour_group_name": item_a.colour_group_name,
                            "index_group_name":  item_a.index_group_name,
                        }
                    )
                    side_effects.append("Purchase summary updated")
            else:
                updated_rejected = state.rejected_items + [item_a.article_id]
                await self.session_mgr.update_dialogue_state(
                    session_id, {"rejected_items": updated_rejected}
                )
                side_effects.append(f"Added {item_a.article_id} to rejected_items")

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
                f"Recommendation outcome: {'accepted' if is_positive else 'rejected'}"
            )

        memory_ctx = await self._base_memory_context(
            user_id, state, include_preferences=False
        )
        memory_ctx["feedback"] = {
            "sentiment_score": sentiment_score,
            "is_positive":     is_positive,
            "feedback_type": (
                "positive" if sentiment_score > 0.3
                else "negative" if sentiment_score < -0.3
                else "neutral"
            ),
            "item_reacted_to": item_a.model_dump() if item_a else None,
        }

        return {
            "label":              "FEEDBACK",
            "retrieval_strategy": "NO",
            "retrieval_input":    None,   # no retrieval for feedback
            "memory_context":     memory_ctx,
            "side_effects":       side_effects,
        }

    async def _enrich_chitchat(
        self, session_id, user_id, current_message
    ) -> dict:
        """CHITCHAT → no retrieval, minimal memory context.
        Passes user_message so the response generator knows what was said."""
        return {
            "label":              "CHITCHAT",
            "retrieval_strategy": "NO",
            "retrieval_input":    None,
            "memory_context":     {
                "user_message":          current_message,
                "dialogue_state":        {},
                "long_term_preferences": [],
                "style_profile":         {},
                "preference_summary":    {},
                "existing_explanation":  None,
            },
            "side_effects": [],
        }
