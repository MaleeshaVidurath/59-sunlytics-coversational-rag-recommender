# m3_implementation/memory/core/pipeline.py
#
# The unified memory pipeline — single entry point for every chat turn.
#
# THIS IS WHERE EVERYTHING CONNECTS:
#   1. Receive user message + session/user context
#   2. Retrieve last N turns from Redis for DistilBERT context
#   3. Call DistilBERT classifier → get label + retrieval strategy
#   4. Store the user turn with classification result
#   5. Call enrichment layer → get memory context for that label
#   6. Return everything the retrieval/RAG system needs

import sys
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load .env from project root

from memory.core.session_manager import SessionManager
from memory.core.turn_manager import TurnManager
from memory.core.user_manager import UserManager
from memory.core.enrichment import EnrichmentLayer
from memory.models.schemas import (
    TurnClassification, ItemInContext,
    RecommendationDocument, now_utc
)
from memory.db.mongo import get_db


def _load_distilbert_predictor():
    model_path_from_env = os.getenv("DISTILBERT_MODEL_PATH")
    if not model_path_from_env:
        print("[MemoryPipeline] DISTILBERT_MODEL_PATH not set — using fallback.")
        return None

    if not os.path.isabs(model_path_from_env):
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root  = os.path.normpath(
            os.path.join(this_file_dir, '..', '..', '..')
        )
        model_path = os.path.normpath(
            os.path.join(project_root, model_path_from_env)
        )
    else:
        model_path = model_path_from_env

    if not os.path.exists(model_path):
        print(f"[MemoryPipeline] Model path not found: {model_path}")
        return None

    distilbert_training_dir = os.path.normpath(
        os.path.join(model_path, '..', '..')
    )

    if distilbert_training_dir not in sys.path:
        sys.path.insert(0, distilbert_training_dir)

    try:
        import importlib
        import config as distilbert_config
        importlib.reload(distilbert_config)

        from predict import Predictor
        predictor = Predictor(model_dir=model_path)
        print(f"[MemoryPipeline] DistilBERT loaded from: {model_path}")
        return predictor
    except Exception as e:
        print(f"[MemoryPipeline] Could not load DistilBERT: {e}")
        return None


class MemoryPipeline:
    """
    Unified memory pipeline that connects DistilBERT to the enrichment layer.

    Two ways to instantiate:

    1. Auto-load DistilBERT from .env path (recommended):
         pipeline = MemoryPipeline()

    2. Pass a pre-loaded predictor (for testing or custom setups):
         from predict import Predictor
         predictor = Predictor(model_dir="path/to/model")
         pipeline = MemoryPipeline(distilbert_predictor=predictor)

    3. Force fallback classifier (for unit tests without the model):
         pipeline = MemoryPipeline(distilbert_predictor=None, use_fallback=True)
    """

    def __init__(
        self,
        distilbert_predictor=None,
        auto_load: bool = True
    ):
        """
        Args:
            distilbert_predictor: A pre-loaded Predictor instance.
                                  If provided, this is used directly.
            auto_load:            If True (default) and distilbert_predictor
                                  is None, automatically loads from .env path.
                                  Set to False to force the fallback classifier.
        """
        self.session_mgr = SessionManager()
        self.turn_mgr    = TurnManager()
        self.user_mgr    = UserManager()
        self.enricher    = EnrichmentLayer()

        if distilbert_predictor is not None:
            # Caller provided a pre-loaded predictor — use it directly
            self.predictor = distilbert_predictor
            print("[MemoryPipeline] Using provided DistilBERT predictor.")
        elif auto_load:
            # Auto-load from .env path
            self.predictor = _load_distilbert_predictor()
        else:
            # Explicitly forced fallback
            self.predictor = None
            print("[MemoryPipeline] Using fallback rule-based classifier.")

    async def process_turn(
        self,
        user_id: str,
        message: str,
        session_id: Optional[str] = None,
        customer_id: Optional[str] = None
    ) -> dict:
        """
        Processes one user message through the complete memory pipeline.

        Call this at the start of every chat request, before your RAG system.

        Args:
            user_id:     Internal user ID (from users collection).
            message:     The user's raw message text.
            session_id:  Optional — active session ID from frontend.
                         If None, the pipeline finds or creates one.
            customer_id: Optional — customer_id from sample_customers.csv.
                         Used to look up user_id if user_id is not known.

        Returns a dict with everything your RAG system needs:
            user_id:            Resolved user ID
            session_id:         Active session ID
            turn_id:            ID of the stored user turn
            label:              DistilBERT classification label
            retrieval_strategy: "FULL", "PARTIAL", or "NO"
            confidence:         DistilBERT confidence score (0.0 if fallback)
            used_rules:         True if rule-based fallback was used
            enriched_context:   Full memory context for this label type
            retrieval_query:    Structured query for your RAG system (None if NO)
            memory_context:     Label-specific memory data
            classifier_input:   Text fed to DistilBERT (for debugging)
        """

        # ── Step 1: Resolve user from customer_id if needed ───────────────
        if not user_id and customer_id:
            user = await self.user_mgr.get_user_by_customer_id(customer_id)
            if user:
                user_id = user.user_id
            else:
                user = await self.user_mgr.get_or_create_user(
                    customer_id=customer_id
                )
                user_id = user.user_id

        # ── Step 2: Get or create the active session ───────────────────────
        session = await self.session_mgr.get_or_create_session(
            user_id=user_id,
            session_id=session_id
        )
        active_session_id = session.session_id

        # ── Step 3: Get recent turns for DistilBERT context ───────────────
        # Fast Redis read — sub-millisecond
        history = await self.turn_mgr.get_turns_as_history(
            session_id=active_session_id,
            n=3
        )
        classifier_input = await self.turn_mgr.get_classifier_input(
            session_id=active_session_id,
            current_message=message
        )

        # ── Step 4: Run DistilBERT classification ──────────────────────────
        if self.predictor is not None:
            # Real DistilBERT model
            classification_result = self.predictor.predict(
                history=history,
                current_message=message
            )
            label              = classification_result["label_name"]
            retrieval_strategy = classification_result["retrieval_strategy"]
            confidence         = classification_result["confidence"]
            used_rules         = classification_result.get("used_rules", False)
        else:
            # Fallback keyword rules
            label, retrieval_strategy = self._fallback_classify(message, history)
            confidence  = 0.0
            used_rules  = True

        # ── Step 5: Extract entities from the message ─────────────────────
        entities = self._extract_entities(message)

        # ── Step 6: Store the user turn ────────────────────────────────────
        turn_classification = TurnClassification(
            label=label,
            retrieval_strategy=retrieval_strategy,
            confidence=confidence,
            used_rules=used_rules
        )
        user_turn = await self.turn_mgr.add_user_turn(
            session_id=active_session_id,
            user_id=user_id,
            content=message,
            classification=turn_classification,
            entities=entities
        )

        # ── Step 7: Enrich with memory context ────────────────────────────
        enriched = await self.enricher.enrich(
            label=label,
            retrieval_strategy=retrieval_strategy,
            session_id=active_session_id,
            user_id=user_id,
            current_message=message,
            entities=entities
        )

        return {
            "user_id":            user_id,
            "session_id":         active_session_id,
            "turn_id":            user_turn.turn_id,
            "label":              label,
            "retrieval_strategy": retrieval_strategy,
            "confidence":         confidence,
            "used_rules":         used_rules,
            "enriched_context":   enriched,
            "retrieval_query":    enriched.get("retrieval_query"),
            "memory_context":     enriched.get("memory_context", {}),
            "classifier_input":   classifier_input
        }

    async def store_response(
        self,
        session_id: str,
        user_id: str,
        bot_response: str,
        recommended_items: Optional[list[dict]] = None,
        trigger_label: str = "UNKNOWN",
        retrieval_strategy: str = "UNKNOWN"
    ) -> dict:
        """
        Stores the assistant's response after your RAG system generates it.

        Call this AFTER your RAG system returns, before sending to the user.

        Args:
            session_id:         Active session ID
            user_id:            User ID
            bot_response:       Full text response from your RAG system
            recommended_items:  List of item dicts. Each needs at minimum:
                                 article_id, prod_name, product_type_name,
                                 colour_group_name
            trigger_label:      DistilBERT label that triggered this response
            retrieval_strategy: Retrieval strategy used

        Returns dict with recommendation_id and turn_id.
        """
        db = get_db()

        recommendation_id = None
        if recommended_items:
            items = []
            for item_dict in recommended_items:
                try:
                    items.append(ItemInContext(**item_dict))
                except Exception:
                    pass

            rec_doc = RecommendationDocument(
                session_id=session_id,
                user_id=user_id,
                turn_id="pending",
                trigger_label=trigger_label,
                retrieval_strategy=retrieval_strategy,
                items=items
            )
            await db.recommendations.insert_one(
                rec_doc.model_dump(mode="json")
            )
            recommendation_id = rec_doc.recommendation_id

            # Update dialogue state with newly recommended items
            if len(items) >= 1:
                await self.session_mgr.update_dialogue_state(
                    session_id,
                    {
                        "currently_discussing": {
                            "item_a": items[0].model_dump(),
                            "item_b": items[1].model_dump() if len(items) >= 2 else None
                        }
                    }
                )

        # Store the assistant turn
        bot_turn = await self.turn_mgr.add_assistant_turn(
            session_id=session_id,
            user_id=user_id,
            content=bot_response,
            recommendation_id=recommendation_id
        )

        # Update recommendation with correct turn_id
        if recommendation_id:
            await db.recommendations.update_one(
                {"recommendation_id": recommendation_id},
                {"$set": {"turn_id": bot_turn.turn_id}}
            )

        return {
            "turn_id":           bot_turn.turn_id,
            "recommendation_id": recommendation_id
        }

    # ── Helper methods ────────────────────────────────────────────────────────

    def _fallback_classify(
        self,
        message: str,
        history: list[dict]
    ) -> tuple[str, str]:
        """
        Rule-based fallback classifier used when DistilBERT is not loaded.
        Checks patterns from most specific to most generic.
        Returns (label, retrieval_strategy).
        """
        msg = message.lower().strip()
        has_history = len(history) > 0

        # WHY questions — most specific
        if any(w in msg for w in ["why", "reason", "explain why",
                                   "why did you", "why is this"]):
            return "EXPLANATION_WHY", "PARTIAL"

        # Comparison
        if any(w in msg for w in ["which is better", "compare", " vs ",
                                   "versus", "difference between",
                                   "which one", "which would"]):
            return "COMPARISON", "PARTIAL"

        # Attribute questions about specific product properties
        if any(w in msg for w in ["what material", "what fabric",
                                   "what colour", "what color",
                                   "does it have pocket",
                                   "how does it fit", "is it cotton",
                                   "machine wash", "what size"]):
            return "ATTRIBUTE_QUESTION", "PARTIAL"

        # Selection / reference to a specific item
        if any(w in msg for w in ["tell me more", "more details", "more info",
                                   "more about", "first one", "second one",
                                   "option 1", "option 2", "the other one"]):
            return "SELECTION_REFERENCE", "PARTIAL"

        # Feedback — only when history exists
        if has_history and any(w in msg for w in [
            "i love", "i like", "perfect", "great", "amazing",
            "i don't like", "i hate", "not my style", "not for me",
            "i'll take", "yes please", "no thanks", "not impressed"
        ]):
            return "FEEDBACK", "NO"

        # Refinement — only when history exists.
        # Must be before chitchat so "show me in white" with history → REFINEMENT
        if has_history and any(w in msg for w in [
            "cheaper", "instead", "prefer", "different colour",
            "different color", "more formal", "more casual",
            "something in", "show me", "can you show", "also show",
            "in white", "in black", "in red", "in blue", "in green",
            "something else", "other options", "more options",
            "slim fit", "more colourful", "smaller", "bigger",
            "something more", "try a different", "change to",
            "can you find", "different style", "another option"
        ]):
            return "REFINEMENT", "FULL"

        # Chitchat
        if any(w in msg for w in [
            "hello", "hi there", "hey", "good morning", "good afternoon",
            "thanks", "thank you", "cheers", "bye", "goodbye",
            "that's helpful", "you've been", "great help", "see you"
        ]):
            return "CHITCHAT", "NO"

        # Default
        if has_history:
            return "REFINEMENT", "FULL"
        return "INITIAL_REQUEST", "FULL"

    def _extract_entities(self, message: str) -> dict:
        """
        Extracts structured entities from a user message.
        Maps to column names from sample_articles.csv.
        Replace with your NLP pipeline for production.
        """
        entities = {}
        msg = message.lower()

        # Colour — check multi-word colours first
        colour_map = {
            "dark blue": "Dark Blue", "light blue": "Light Blue",
            "light pink": "Light Pink", "dark green": "Dark Green",
            "dark grey": "Dark Grey", "off white": "Off White",
            "black": "Black", "white": "White", "red": "Red",
            "blue": "Blue", "pink": "Pink", "green": "Green",
            "yellow": "Yellow", "beige": "Beige", "grey": "Grey",
            "gray": "Grey", "brown": "Brown", "orange": "Orange",
            "purple": "Purple", "navy": "Dark Blue",
        }
        for keyword, value in colour_map.items():
            if keyword in msg:
                entities["colour_group_name"] = value
                break

        # Product type — check multi-word types first
        product_map = {
            "vest top": "Vest top", "t-shirt": "T-shirt",
            "tshirt": "T-shirt", "dress": "Dress", "dresses": "Dress",
            "trousers": "Trousers", "pants": "Trousers", "jeans": "Trousers",
            "top": "Top", "blouse": "Blouse", "shirt": "Shirt",
            "sweater": "Sweater", "jumper": "Sweater", "hoodie": "Hoodie",
            "jacket": "Jacket", "coat": "Jacket", "skirt": "Skirt",
            "shorts": "Shorts", "sneakers": "Sneakers", "shoes": "Sneakers",
            "boots": "Boots", "sandals": "Sandals", "bag": "Bag",
            "handbag": "Bag", "cardigan": "Cardigan", "blazer": "Blazer",
            "scarf": "Scarf", "hat": "Hat/beanie", "socks": "Socks",
            "leggings": "Leggings/Tights", "bra": "Bra",
        }
        for keyword, value in product_map.items():
            if keyword in msg:
                entities["product_type_name"] = value
                break

        # Price
        import re
        price_patterns = [
            (r'under\s+[£$€]?\s*(\d+)',     "price_max"),
            (r'below\s+[£$€]?\s*(\d+)',     "price_max"),
            (r'less than\s+[£$€]?\s*(\d+)', "price_max"),
            (r'[£$€]\s*(\d+)',               "price_max"),
            (r'(\d+)\s*(?:pounds|dollars|euros)', "price_max"),
        ]
        for pattern, field in price_patterns:
            match = re.search(pattern, msg)
            if match:
                entities[field] = float(match.group(1))
                break

        # Style — check multi-word first
        style_map = {
            "smart casual": "smart casual", "casual": "casual",
            "formal": "formal", "sporty": "sporty", "elegant": "elegant",
            "minimalist": "minimalist", "classic": "classic",
            "professional": "professional", "relaxed": "relaxed",
        }
        for keyword, value in style_map.items():
            if keyword in msg:
                entities["style"] = value
                break

        # Occasion — check multi-word first
        occasion_map = {
            "job interview": "job interview", "casual day": "casual day out",
            "date night": "date night", "wedding": "wedding",
            "office": "work", "work": "work", "gym": "gym",
            "beach": "beach", "party": "party", "date": "date night",
            "summer": "summer", "winter": "winter",
        }
        for keyword, value in occasion_map.items():
            if keyword in msg:
                entities["occasion"] = value
                break

        return entities
