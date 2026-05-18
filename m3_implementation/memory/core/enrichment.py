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


# ── Price ceiling resolver for "cheaper than X" refinements ─────────────────
# When the user says "cheaper" or "less expensive", the LLM entity extractor
# guesses a price_max from general knowledge and gets it wrong.
# This resolver reads the *actual price* of the named (or current) item from
# context and sets price_max to strictly below that price.

_CHEAPER_KEYWORDS = frozenset([
    "cheaper", "less expensive", "more affordable", "lower price",
    "cheaper than", "not as expensive", "cost less", "budget option",
    "something cheaper", "a cheaper one",
])


def _resolve_cheaper_price(
    message: str,
    constraints: dict,
    item_a,
    item_b,
) -> dict:
    """
    Override price_max when the user asks for something cheaper.
    Reads the real price from context items instead of trusting the LLM guess.

    Priority:
      1. If a specific item name is mentioned → use that item's price
      2. Otherwise → use the minimum price among context items
    Sets price_max = reference_price - 0.01 (strictly cheaper).
    """
    msg_lower = message.lower()
    if not any(kw in msg_lower for kw in _CHEAPER_KEYWORDS):
        return constraints

    # Collect (item, price) pairs that have a known price
    items_with_price = []
    for item in (item_a, item_b):
        if item is None:
            continue
        price = getattr(item, "price", None)
        if price is None and hasattr(item, "model_dump"):
            price = item.model_dump().get("price")
        if price:
            items_with_price.append((item, float(price)))

    if not items_with_price:
        return constraints  # no known prices — keep LLM value

    # Check whether a specific item is named in the message
    msg_words = set(msg_lower.split())
    reference_price = None
    for item, price in items_with_price:
        name_words = (item.prod_name or "").lower().split()
        if any(w in msg_words for w in name_words if len(w) > 3):
            reference_price = price
            print(f"[ENRICH-REFINE] price_max override: '{item.prod_name}' "
                  f"(£{price:.2f}) named in message")
            break

    if reference_price is None:
        reference_price = min(price for _, price in items_with_price)
        print(f"[ENRICH-REFINE] price_max override: using min context price £{reference_price:.2f}")

    price_max = round(reference_price - 0.01, 2)
    print(f"[ENRICH-REFINE] price_max: LLM={constraints.get('price_max')} → context-derived={price_max}")
    return {**constraints, "price_max": price_max}


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


# ── Comparison item resolver ──────────────────────────────────────────────────

def _score_items_by_name(message: str, item_pool: list) -> list:
    """
    For each item in the pool check whether its product name appears in the
    message. Full-name substring match scores 100; word-overlap scores by
    count of matching name words (>=3 chars). Deduplicates by prod_name
    (one item per unique product name), ordered by score descending.
    Returns empty list if no item names are detected.

    Why no splitting by 'and/vs'?
    The user may write any connector or none at all. Checking item names
    directly is more reliable and works for any phrasing.
    """
    if not item_pool:
        return []

    msg_lower = message.lower()
    msg_words = {w for w in msg_lower.split() if len(w) >= 3}

    scored = []
    for item in item_pool:
        name_lower = (item.prod_name or "").lower()
        if name_lower and name_lower in msg_lower:
            score = 100          # full product-name substring found
        else:
            name_words = {w for w in name_lower.split() if len(w) >= 3}
            score = len(name_words & msg_words)
        scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] == 0:
        return []

    top_score = scored[0][0]
    # When strong full-name matches exist, drop items that only scored via
    # incidental shared words (e.g. every leggings item matching "leggings").
    min_score = (top_score // 2) if top_score >= 10 else 1

    seen_names: set  = set()
    result:     list = []
    for score, item in scored:
        if score < min_score:
            break
        name_key = (item.prod_name or "").lower()
        if name_key not in seen_names:
            seen_names.add(name_key)
            result.append(item)

    return result




async def _collect_session_items(session_id: str) -> list:
    """
    Returns all unique ItemInContext objects recommended in this session
    (INITIAL_REQUEST and REFINEMENT turns only), oldest first.
    Used when named items are not found in the last-turn pool.
    """
    try:
        db     = get_db()
        cursor = db.recommendations.find(
            {
                "session_id":    session_id,
                "trigger_label": {"$in": ["INITIAL_REQUEST", "REFINEMENT"]},
            },
            {"items": 1, "created_at": 1},
        ).sort("created_at", 1)
        docs = await cursor.to_list(length=50)

        seen_ids:  set  = set()
        all_items: list = []
        for doc in docs:
            for item_dict in (doc.get("items") or []):
                aid = item_dict.get("article_id")
                if aid and aid not in seen_ids:
                    seen_ids.add(aid)
                    try:
                        all_items.append(ItemInContext(**item_dict))
                    except Exception:
                        pass
        print(f"[ENRICH-COMPARE] session history: {len(all_items)} unique items")
        return all_items
    except Exception as e:
        print(f"[ENRICH-COMPARE] session history query failed: {e}")
        return []


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
        print(f"[ENRICH-INIT] ━━━ called msg='{current_message[:50]}' entities={entities}")
        """INITIAL_REQUEST → action: catalog_search"""
        side_effects = []

        pref_summary = await self.user_mgr.get_preference_summary(user_id)

        # Extract hard constraints from entities — quantity is NOT a DB filter
        new_constraints = {
            k: v for k, v in entities.items()
            if k not in ("style", "occasion", "quantity") and v is not None
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

        print(f"[ENRICH-INIT] returning retrieval_input")
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
                    # Quantity requested by user — from LLM entity extraction
                    "quantity": entities.get("quantity") if entities else None,
                }
            ),
            "memory_context": memory_ctx,
            "side_effects":   side_effects,
        }

    async def _enrich_refinement(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        print(f"[ENRICH-REFINE] ━━━ called msg='{current_message[:50]}' entities={entities}")
        """REFINEMENT → action: catalog_search (same as INITIAL_REQUEST)"""
        side_effects = []

        pref_summary = await self.user_mgr.get_preference_summary(user_id)

        # Extract items first so we can use their prices for constraint resolution
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        # New constraints from this message, merged on top of existing ones
        new_constraints = {
            k: v for k, v in entities.items()
            if k not in ("style", "occasion", "quantity") and v is not None
        }

        # Override LLM-guessed price_max when user says "cheaper [than X]"
        new_constraints = _resolve_cheaper_price(
            current_message, new_constraints, item_a, item_b
        )

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
        print(f"[ENRICH-ATTR] ━━━ called msg='{current_message[:50]}' entities={entities}")
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
        print(f"[ENRICH-WHY] ━━━ called msg='{current_message[:50]}' entities={entities}")
        """
        EXPLANATION_WHY → action: explanation_generate

        KEY FIX: Parse which item the user is asking about from their message.
        User may say "why CA Hugh Linen shirt" or "why the second one" or
        "why did you recommend option 2" — we must match the correct item.
        Default to item_a only if no specific item can be identified.
        """
        db = get_db()
        current_items = state.currently_discussing
        item_a = current_items.get("item_a")
        item_b = current_items.get("item_b")

        # Collect ALL recommended items (item_a … item_z)
        all_ctx_items = [
            v for k, v in sorted(current_items.items())
            if k.startswith("item_") and v is not None
        ]

        msg_lower = current_message.lower()
        msg_words = set(msg_lower.split())

        # ── Identify which specific item user is asking about ──────────────
        # Explicit ordinals first, then score ALL items by name-word overlap.
        # Avoids false matches on generic category words (e.g. "shorts"
        # appears in every item name when user asks "why JONES 5-PKT SHORTS").
        # When no name is mentioned at all → target_item = None (explain all).
        if any(ref in msg_lower for ref in ["first", "option 1", "1st", "number 1", "#1"]):
            target_item = item_a
        elif any(ref in msg_lower for ref in [
            "second", "option 2", "2nd", "number 2", "the other", "latter", "last one", "#2"
        ]):
            target_item = item_b
        else:
            best_item  = None
            best_score = 0
            for item in all_ctx_items:
                name  = (item.prod_name or "").lower()
                score = sum(1 for w in name.split() if len(w) > 3 and w in msg_words)
                if score > best_score:
                    best_score = score
                    best_item  = item
            target_item = best_item if best_score > 0 else None

        print(f"[ENRICH-WHY] target_item="
              f"'{target_item.prod_name if target_item else 'ALL ITEMS'}' "
              f"(scored from {len(all_ctx_items)} context items)")

        # ── Fetch stored explanation for target item ───────────────────────
        existing_explanation = None
        if target_item:
            expl_doc = await db.explanations.find_one(
                {"session_id": session_id, "article_id": target_item.article_id},
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
                item_a=target_item if target_item else item_a,
                item_b=item_b if target_item != item_b else item_a,
                exclude_ids=state.rejected_items,
                payload={
                    # Single item explanation
                    "article_id":     target_item.article_id if target_item else None,
                    # Full item data stored at recommendation time — assembler uses
                    # this to skip the DB query when detail_desc is already present.
                    "context_article": target_item.model_dump() if target_item else None,
                    # All-items summary: passed when user asks "why" with no product name
                    "all_item_ids":   (
                        None if target_item
                        else [it.article_id for it in all_ctx_items]
                    ),
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
        print(f"[ENRICH-COMPARE] ━━━ called msg='{current_message[:50]}' entities={entities}")
        """COMPARISON → action: item_compare"""
        current_items = state.currently_discussing

        # Collect ALL items stored in currently_discussing (item_a … item_z)
        all_ctx_items = [
            v for k, v in sorted(current_items.items())
            if k.startswith("item_") and v is not None
        ]
        print(f"[ENRICH-COMPARE] currently_discussing keys={list(current_items.keys())} "
              f"non-null items={len(all_ctx_items)}")

        # Identify comparison dimension using hybrid similarity
        dimension = _identify_comparison_dimension(current_message)

        pref_summary = await self.user_mgr.get_preference_summary(user_id)
        memory_ctx = await self._base_memory_context(user_id, state)

        # Step 1: try to match named items from the last-turn pool.
        resolved = _score_items_by_name(current_message, all_ctx_items)
        print(f"[ENRICH-COMPARE] name-match (last turn): {[it.prod_name for it in resolved]}")

        # Step 2: if last-turn pool gave <2 matches, check session history.
        # Covers the case where the user names items from an earlier turn.
        # If no names were mentioned either, session history also returns empty
        # so the generic fallback below still fires correctly.
        if len(resolved) < 2:
            hist_pool = await _collect_session_items(session_id)
            if hist_pool:
                hist_resolved = _score_items_by_name(current_message, hist_pool)
                if len(hist_resolved) >= 2:
                    resolved = hist_resolved
                    print(f"[ENRICH-COMPARE] name-match (session history): "
                          f"{[it.prod_name for it in resolved]}")

        if len(resolved) >= 2:
            compare_a    = resolved[0]
            compare_b    = resolved[1]
            compare_list = resolved
            print(f"[ENRICH-COMPARE] named {len(resolved)} item(s): "
                  f"{[it.prod_name for it in resolved]}")
        else:
            # No names mentioned — compare all last-turn items only
            compare_a    = all_ctx_items[0] if all_ctx_items else None
            compare_b    = all_ctx_items[1] if len(all_ctx_items) > 1 else None
            compare_list = all_ctx_items
            print(f"[ENRICH-COMPARE] generic: comparing all {len(all_ctx_items)} last-turn items")

        payload = {
            "article_id_a":         compare_a.article_id if compare_a else None,
            "article_id_b":         compare_b.article_id if compare_b else None,
            "comparison_dimension": dimension,
            "preference_weights": {
                p["attribute_name"]: p["weight"]
                for p in pref_summary.get("liked_attributes", [])
            },
        }
        if len(compare_list) > 2:
            payload["article_ids_list"] = [
                {
                    "article_id": it.article_id,
                    "prod_name":  it.prod_name,
                    "colour":     it.colour_group_name,
                    "price":      getattr(it, "price", None),
                }
                for it in compare_list
            ]

        return {
            "label":              "COMPARISON",
            "retrieval_strategy": "PARTIAL",
            "retrieval_input": self._make_retrieval_input(
                action="item_compare",
                retrieval_strategy="PARTIAL",
                user_message=current_message,
                item_a=compare_a,
                item_b=compare_b,
                exclude_ids=state.rejected_items,
                payload=payload,
            ),
            "memory_context": memory_ctx,
            "side_effects":   [],
        }

    async def _enrich_selection_reference(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        print(f"[ENRICH-SELECT] ━━━ called msg='{current_message[:50]}' entities={entities}")
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

        # Score ALL items in currently_discussing (item_a … item_z) by name-word overlap.
        # Pick the item with the highest score — avoids false matches on generic category
        # words like "sneaker" or "dress" that appear in multiple item names.
        all_ctx_items = [
            v for k, v in sorted(current_items.items())
            if k.startswith("item_") and v is not None
        ]
        msg_lower = current_message.lower()
        msg_words = set(msg_lower.split())
        best_item  = None
        best_score = 0
        for item in all_ctx_items:
            name = (item.prod_name or "").lower()
            score = sum(1 for w in name.split() if len(w) > 3 and w in msg_words)
            if score > best_score:
                best_score = score
                best_item  = item
        if best_item is not None:
            selected_item = best_item
            print(f"[ENRICH-SELECT] name-scored '{selected_item.prod_name}' "
                  f"(article_id={selected_item.article_id}, score={best_score}) "
                  f"from {len(all_ctx_items)} context items")
        else:
            # Fallback: handles ordinals ("second one"), colours, "the other one"
            selected_item = _resolve_item_reference(current_message, item_a, item_b)
            print(f"[ENRICH-SELECT] fallback resolved to "
                  f"'{selected_item.prod_name if selected_item else None}'")

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
                    "article_id":     selected_item.article_id if selected_item else None,
                    # Full item data stored at recommendation time — assembler uses
                    # this to skip the DB query when detail_desc is already present.
                    "context_article": selected_item.model_dump() if selected_item else None,
                }
            ),
            "memory_context": memory_ctx,
            "side_effects":   side_effects,
        }

    async def _get_purchase_hints(self, user_id: str) -> dict:
        """
        Returns purchase_history_hints from the pre-loaded customer profile.
        Falls back to empty hints if no profile exists.

        FIX: Uses get_purchase_history() which reads raw MongoDB dict directly,
        instead of get_user_by_id() which returns a Pydantic UserDocument that
        may strip the purchase_history field if not declared in the schema.
        """
        _empty = {
            "top_colours":           [],
            "top_product_types":     [],
            "inferred_gender":       None,
            "budget_tier":           None,
            "preferred_price_range": None,
            "dominant_colour":       None,
            "dominant_type":         None,
        }
        try:
            print(f"[ENRICH-HINTS] loading purchase_history for user_id={user_id[:20]}")
            ph = await self.user_mgr.get_purchase_history(user_id)
            print(f"[ENRICH-HINTS] raw purchase_history keys={list(ph.keys()) if ph else 'EMPTY'}")
            if not ph:
                print(f"[ENRICH-HINTS] no purchase_history found → returning empty hints")
                return _empty
            hints = {
                "top_colours": [
                    c["colour"] for c in ph.get("top_colours", [])
                    if isinstance(c, dict) and c.get("colour")
                ],
                "top_product_types": [
                    t["type"] for t in ph.get("top_product_types", [])
                    if isinstance(t, dict) and t.get("type")
                ],
                "inferred_gender":       ph.get("inferred_gender"),
                "budget_tier":           ph.get("price_stats", {}).get("budget_tier"),
                "preferred_price_range": ph.get("price_stats", {}).get("preferred_range"),
                "dominant_colour":       ph.get("dominant_colour"),
                "dominant_type":         ph.get("dominant_product_type"),
            }
            print(f"[ENRICH-HINTS] hints built: top_colours={hints['top_colours'][:3]} gender={hints['inferred_gender']} budget={hints['budget_tier']} dominant_colour={hints['dominant_colour']}")
            return hints
        except Exception as e:
            print(f"[ENRICH-HINTS] ERROR: {e} → returning empty hints")
            import traceback; traceback.print_exc()
            return _empty


    async def _enrich_feedback(
        self, session_id, user_id, current_message, entities, state
    ) -> dict:
        print(f"[ENRICH-FEEDBACK] ━━━ called msg='{current_message[:50]}' entities={entities}")
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
        print(f"[ENRICH-CHITCHAT] ━━━ called msg='{current_message[:50]}'")
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
