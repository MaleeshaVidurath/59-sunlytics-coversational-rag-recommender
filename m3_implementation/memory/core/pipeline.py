# m3_implementation/memory/core/pipeline.py
#
# The unified memory pipeline — single entry point for every chat turn.
#
# FLOW:
#   User message
#       → Step 1: Resolve user_id (from customer_id if needed)
#       → Step 2: get_or_create_session (Redis first, MongoDB fallback)
#       → Step 3: get_turns_as_history() → last 3 exchanges from Redis
#       → Step 4: predictor.predict(history, message) → label + strategy
#       → Step 5: _extract_entities(message) → colour, product, price etc
#       → Step 6: add_user_turn() → store to Redis + MongoDB
#       → Step 7: enricher.enrich(label, ...) → label-specific memory context
#       → Return standardised output to RAG system
#
# OUTPUT STRUCTURE (always identical regardless of label):
#   {
#       user_id, session_id, turn_id,
#       label, retrieval_strategy, confidence, used_rules,
#       retrieval_input: {              ← pass this to your RAG system
#           action, retrieval_strategy, user_message,
#           items_in_context, exclude_ids, payload
#       } | None,
#       memory_context: {              ← use this to build your RAG prompt
#           dialogue_state, long_term_preferences,
#           style_profile, preference_summary, existing_explanation, ...
#       },
#       side_effects: [...],           ← memory updates that happened
#       classifier_input: str          ← debug: what DistilBERT received
#   }

import sys
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from memory.core.session_manager import SessionManager
from memory.core.turn_manager import TurnManager
from memory.core.user_manager import UserManager
from memory.core.enrichment import EnrichmentLayer
from memory.core.entity_extractor import (
    extract_entities, is_fashion_relevant, is_fashion_relevant_async
)
from memory.models.schemas import (
    TurnClassification, ItemInContext,
    RecommendationDocument, now_utc
)
from memory.db.mongo import get_db


def _load_distilbert_predictor():
    """
    Loads the trained DistilBERT Predictor from DISTILBERT_MODEL_PATH in .env.
    Returns the Predictor instance, or None if loading fails (uses fallback).
    """
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
    Unified memory pipeline connecting DistilBERT to the enrichment layer.

    Instantiate once at app startup — auto-loads DistilBERT from .env:
        pipeline = MemoryPipeline()

    Then call for every user message:
        result = await pipeline.process_turn(user_id, message, session_id)

    After RAG responds:
        await pipeline.store_response(session_id, user_id, bot_response, items)
    """

    def __init__(
        self,
        distilbert_predictor=None,
        auto_load: bool = True
    ):
        self.session_mgr = SessionManager()
        self.turn_mgr    = TurnManager()
        self.user_mgr    = UserManager()
        self.enricher    = EnrichmentLayer()

        if distilbert_predictor is not None:
            self.predictor = distilbert_predictor
            print("[MemoryPipeline] Using provided DistilBERT predictor.")
        elif auto_load:
            self.predictor = _load_distilbert_predictor()
        else:
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

        Args:
            user_id:     Internal user ID (from users collection).
            message:     The user's raw message text.
            session_id:  Optional active session ID. If None, one is
                         found or created automatically.
            customer_id: Optional customer_id from sample_customers.csv.
                         Used to resolve user_id if not already known.

        Returns the standardised output dict described in this module's
        docstring. Pass result["retrieval_input"] to your RAG system.
        """

        # ── Step 1: Resolve user ───────────────────────────────────────────
        if not user_id and customer_id:
            user = await self.user_mgr.get_user_by_customer_id(customer_id)
            if user:
                user_id = user.user_id
            else:
                user = await self.user_mgr.get_or_create_user(
                    customer_id=customer_id
                )
                user_id = user.user_id

        # ── Step 2: Session ────────────────────────────────────────────────
        session = await self.session_mgr.get_or_create_session(
            user_id=user_id,
            session_id=session_id
        )
        active_session_id = session.session_id

        print(f"[PIPELINE-2] session: session_id={session_id}")
        # ── Step 3: Get recent turns for DistilBERT context ───────────────
        history = await self.turn_mgr.get_turns_as_history(
            session_id=active_session_id,
            n=3
        )
        classifier_input = await self.turn_mgr.get_classifier_input(
            session_id=active_session_id,
            current_message=message
        )

        # ── Step 3b: Not-relevant input guard (4-stage hybrid classifier) ────
        # is_fashion_relevant_async runs:
        #   Stage 1 (0ms)    — Conversational bypass: continuation phrases
        #                      in active history always pass through.
        #   Stage 2 (0ms)    — Fast keyword allowlist + expanded blocklist.
        #   Stage 3 (3-5ms)  — Dual-pool mean-top-3 semantic scoring
        #                      (16 fashion anchors vs 12 off-topic anchors).
        #   Stage 4 (~150ms) — Groq LLM arbitration for ambiguous middle zone.
        # Passing `history` activates Stage 1 context-awareness.
        _is_relevant, _relevance_score, _guard_stage = (
            await is_fashion_relevant_async(message, history=history)
        )
        print(
            f"[PIPELINE-3b] FashionGuard: relevant={_is_relevant} "
            f"score={_relevance_score:.3f} stage={_guard_stage}"
        )
        if not _is_relevant:
            # Store the turn but return a refusal — no classification, no retrieval
            _refusal_turn = await self.turn_mgr.add_user_turn(
                session_id=active_session_id,
                user_id=user_id,
                content=message,
                classification=TurnClassification(
                    label="CHITCHAT",
                    retrieval_strategy="NO",
                    confidence=_relevance_score,
                    used_rules=True
                ),
                entities={}
            )
            return {
                "user_id":    user_id,
                "session_id": active_session_id,
                "turn_id":    _refusal_turn.turn_id,
                "label":              "CHITCHAT",
                "retrieval_strategy": "NO",
                "confidence":         _relevance_score,
                "used_rules":         True,
                "retrieval_input":    None,
                "memory_context": {
                    "not_relevant":           True,
                    "refusal_message":         (
                        "I can only help with fashion and clothing recommendations. "
                        "Please ask me about clothes, styles, or outfit advice!"
                    ),
                    "dialogue_state":          {},
                    "long_term_preferences":   [],
                    "style_profile":           {},
                    "preference_summary":      {},
                    "existing_explanation":    None,
                },
                "side_effects":       ["Not-relevant input — refusal returned"],
                "classifier_input":   classifier_input,
            }

        # ── Step 4a: Pre-classification guard for short/ambiguous messages ───────
        # Before DistilBERT, check for very short messages that are clearly
        # acknowledgments or filler. Prevents "ok", "yes", "sure" from
        # becoming spurious REFINEMENT calls that trigger catalog searches.
        pre_label, pre_strategy = self._pre_classify_short_message(
            message, history
        )
        if pre_label is not None:
            print(f"[PIPELINE-4a] PRE-CLASSIFIER fired: {pre_label} / {pre_strategy}")
            label              = pre_label
            retrieval_strategy = pre_strategy
            confidence         = 0.0
            used_rules         = True
            print("\n" + "="*60)
            print(f"[DBG-1] PRE-CLASSIFIER: label={label} strategy={retrieval_strategy}")
            print(f"[DBG-1] MSG: '{message[:60]}'")
        # ── Step 4b: DistilBERT classification ─────────────────────────────────
        elif self.predictor is not None:
            classification_result = self.predictor.predict(
                history=history,
                current_message=message
            )
            label              = classification_result["label_name"]
            retrieval_strategy = classification_result["retrieval_strategy"]
            confidence         = classification_result["confidence"]
            used_rules         = classification_result.get("used_rules", False)
            print(f"[PIPELINE-4b] DISTILBERT: label={label} conf={confidence:.1%} strategy={retrieval_strategy}")
            _top3 = sorted(classification_result.get("all_probabilities",{}).items(), key=lambda x:-x[1])[:3]
            print(f"[PIPELINE-4b] top3: {[(k,f'{v:.1%}') for k,v in _top3]}")
            print("\n" + "="*60)
            print(f"[DBG-1] DISTILBERT: label={label} conf={confidence:.1%} strategy={retrieval_strategy}")
            print(f"[DBG-1] MSG: '{message[:60]}'")

        # ── Step 4c: Product keyword override ───────────────────────────────────
        # DistilBERT misclassifies short product-keyword messages as CHITCHAT/FEEDBACK.
        # Training data gaps: "red short", "red one", "need red one" are all REFINEMENT
        # but were never seen in training as such.
        # This override catches them using colour+product and context-aware rules.

        if label in ("CHITCHAT", "FEEDBACK") and pre_label is None:
            _ml = message.lower().strip()

            # All product keywords including singular forms
            _PRODUCT_KW = {
                "dress", "dresses", "shirt", "shirts",
                "t-shirt", "t shirt", "tshirt", "tee",
                "top", "tops", "trouser", "trousers", "pants",
                "skirt", "skirts", "jacket", "jackets",
                "sweater", "sweaters", "coat", "coats",
                "blouse", "blouses", "jeans", "short", "shorts",
                "hoodie", "hoodies", "leggings", "suit", "suits",
                "formal wear", "casual wear", "sportswear",
                "knitwear", "wear", "outfit", "clothing", "clothes",
                "sock", "socks", "shoe", "shoes", "boot", "boots",
                "cardigan", "blazer", "blazers", "pant",
            }

            # Colour words — a colour word alone after items shown = refinement
            _COLOUR_KW = {
                "red", "blue", "white", "black", "pink", "green",
                "yellow", "grey", "gray", "beige", "navy", "brown",
                "orange", "purple", "light", "dark", "bright",
            }

            # Search intent words
            _SEARCH_KW = {
                "need", "want", "show", "find", "looking", "give",
                "get", "buy", "purchase", "search", "recommend",
                "like", "prefer", "another", "different", "instead",
                "other", "more",
            }

            _has_product = any(kw in _ml for kw in _PRODUCT_KW)
            _has_colour  = any(kw in _ml.split() for kw in _COLOUR_KW)
            _has_search  = any(kw in _ml for kw in _SEARCH_KW)
            _is_short    = len(_ml.split()) <= 4

            # Rule 1: product keyword present
            _should_override = _has_product and (_has_search or _is_short)

            # Rule 2: colour + "one" or "ones" = user wants different colour
            # e.g. "red one", "blue ones", "I like red one"
            if not _should_override and history:
                _colour_one = (
                    _has_colour and
                    any(w in _ml for w in ["one", "ones", "those", "that"])
                )
                if _colour_one:
                    _should_override = True

            # Rule 3: colour word + short message with history showing items
            # e.g. "red short" = wants red shorts after seeing white shorts
            if not _should_override and history and _has_colour and _is_short:
                # Check if bot previously showed items (items in context)
                _bot_turns = [t for t in history[-4:] if t.get("role") == "bot"]
                _bot_showed_items = any(
                    any(kw in (t.get("content","")).lower()
                        for kw in ["option 1", "option 2", "£", "found two"])
                    for t in _bot_turns
                )
                if _bot_showed_items:
                    _should_override = True

            if _should_override:
                # Only mark REFINEMENT if history has actual search turns
                _has_search_hist = any(
                    t.get("retrieval_strategy") == "FULL" or
                    any(kw in (t.get("content") or "").lower()
                        for kw in ["option 1", "option 2", "£", "found two"])
                    for t in history
                    if t.get("role") == "bot"
                )
                _new_label = "REFINEMENT" if (history and _has_search_hist) else "INITIAL_REQUEST"
                print(f"[DBG-1] OVERRIDE: {label} ({confidence:.1%}) → {_new_label}")
                label              = _new_label
                retrieval_strategy = "FULL"
                confidence         = 0.80
                used_rules         = True

        else:
            label, retrieval_strategy = self._fallback_classify(message, history)
            confidence  = 0.0
            used_rules  = True


        print(f"[DBG-2] ENTITY EXTRACTION: label={label}")
        print(f"[PIPELINE-5] starting entity extraction for label={label}")
        # ── Step 5: Entity extraction (three-tier: keyword → vector → LLM) ────
        # Tier 1 (keyword+regex) always runs first.
        # Tier 2 (vector similarity) fills missing colour/product_type.
        # Tier 3 (LLM) handles complex natural language when < 2 entities found.
        # Pass label so extraction is skipped for non-search turns
        # (ATTRIBUTE_QUESTION, COMPARISON, FEEDBACK, CHITCHAT etc.)
        entities = await extract_entities(message, label=label)

        # ── Step 6: Store user turn ────────────────────────────────────────
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
        print(f"[DBG-3] ENRICHMENT: calling for label={label}")
        print(f"[PIPELINE-6] calling enricher...")
        enriched = await self.enricher.enrich(
            label=label,
            retrieval_strategy=retrieval_strategy,
            session_id=active_session_id,
            user_id=user_id,
            current_message=message,
            entities=entities
        )

        # ── Return clean, standardised output ─────────────────────────────
        # No duplication — every key appears exactly once.
        # retrieval_input is the single thing your RAG system reads.
        # memory_context is what your RAG uses to build its prompt.
        return {
            # Identity
            "user_id":    user_id,
            "session_id": active_session_id,
            "turn_id":    user_turn.turn_id,

            # Classification
            "label":              label,
            "retrieval_strategy": retrieval_strategy,
            "confidence":         confidence,
            "used_rules":         used_rules,

            # What your RAG system receives — always same envelope shape
            # None when retrieval_strategy is "NO" (FEEDBACK, CHITCHAT)
            "retrieval_input": enriched.get("retrieval_input"),

            # Memory context for building the RAG prompt
            # Always present, always same top-level keys
            "memory_context": enriched.get("memory_context", {}),

            # Memory updates that happened as a side effect of this turn
            # Always present, may be empty list
            "side_effects": enriched.get("side_effects", []),
            "_debug_enriched": enriched,  # temp debug key

            # Debug: what was fed to DistilBERT
            "classifier_input": classifier_input,
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
        Stores the assistant's response after RAG generates it.

        Call this AFTER your RAG system returns, before sending to user.

        Args:
            session_id:         Active session ID
            user_id:            User ID
            bot_response:       Full text response from your RAG system
            recommended_items:  List of item dicts, each needs at minimum:
                                 article_id, prod_name, product_type_name,
                                 colour_group_name
            trigger_label:      DistilBERT label that triggered this response
            retrieval_strategy: Strategy used

        Returns dict with turn_id and recommendation_id.
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

        bot_turn = await self.turn_mgr.add_assistant_turn(
            session_id=session_id,
            user_id=user_id,
            content=bot_response,
            recommendation_id=recommendation_id
        )

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

    def _pre_classify_short_message(
        self,
        message: str,
        history: list
    ):
        """
        Pre-classification for very short messages (≤ 4 words).
        Returns (label, strategy) if the message is clearly an
        acknowledgment or filler, otherwise returns (None, None)
        to let DistilBERT handle it.

        This prevents common misclassifications:
          "ok"     with history → REFINEMENT (wrong) → now CHITCHAT
          "yes"    with history → REFINEMENT (wrong) → now FEEDBACK
          "great"  with history → REFINEMENT (wrong) → now FEEDBACK
          "thanks" anywhere    → REFINEMENT (wrong) → now CHITCHAT
        """
        msg = message.strip().lower()
        words = msg.split()
        word_count = len(words)

        if word_count > 4:
            return None, None   # Let DistilBERT handle longer messages

        has_history = len(history) > 0

        # Pure acknowledgments / fillers → CHITCHAT (no retrieval)
        pure_chitchat = {
            "ok", "okay", "alright", "sure", "right", "noted",
            "got it", "i see", "understood", "cool", "nice",
            "thanks", "thank you", "cheers", "bye", "goodbye",
            "hello", "hi", "hey", "good morning", "good afternoon",
            "great thanks", "ok thanks", "thank you", "many thanks",
        }
        if msg in pure_chitchat:
            return "CHITCHAT", "NO"

        # Short positive reactions to shown items → FEEDBACK positive
        # Only when history exists (there is something to react to)
        short_positive = {
            "yes", "yes please", "perfect", "great", "amazing",
            "love it", "i love it", "love this", "i like it",
            "yes that", "that one", "this one", "i'll take it",
            "i want it", "yes this", "yes please", "lovely",
            "wonderful", "excellent", "fantastic", "brilliant",
        }
        if has_history and msg in short_positive:
            return "FEEDBACK", "NO"

        # Short negative reactions → FEEDBACK negative
        short_negative = {
            "no", "nope", "nah", "no thanks", "not really",
            "don't like", "i hate it", "not good", "bad",
            "terrible", "awful", "no way", "not for me",
        }
        if has_history and msg in short_negative:
            return "FEEDBACK", "NO"

        return None, None   # Not a short-message case — let classifier decide

    def _fallback_classify(
        self,
        message: str,
        history: list
    ) -> tuple[str, str]:
        """Rule-based fallback when DistilBERT is not loaded."""
        msg = message.lower().strip()
        has_history = len(history) > 0

        if any(w in msg for w in ["why", "reason", "explain why",
                                   "why did you", "why is this"]):
            return "EXPLANATION_WHY", "PARTIAL"
        if any(w in msg for w in ["which is better", "compare", " vs ",
                                   "versus", "difference between",
                                   "which one", "which would"]):
            return "COMPARISON", "PARTIAL"
        if any(w in msg for w in ["what material", "what fabric",
                                   "what colour", "what color",
                                   "does it have pocket", "how does it fit",
                                   "is it cotton", "machine wash", "what size"]):
            return "ATTRIBUTE_QUESTION", "PARTIAL"
        if any(w in msg for w in ["tell me more", "more details", "more info",
                                   "more about", "first one", "second one",
                                   "option 1", "option 2", "the other one"]):
            return "SELECTION_REFERENCE", "PARTIAL"
        if has_history and any(w in msg for w in [
            "i love", "i like", "perfect", "great", "amazing",
            "i don't like", "i hate", "not my style", "not for me",
            "i'll take", "yes please", "no thanks", "not impressed"
        ]):
            return "FEEDBACK", "NO"
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
        if any(w in msg for w in [
            "hello", "hi there", "hey", "good morning", "good afternoon",
            "thanks", "thank you", "cheers", "bye", "goodbye",
            "that's helpful", "you've been", "great help", "see you"
        ]):
            return "CHITCHAT", "NO"

        if has_history:
            # Only override to REFINEMENT if there were previous FULL retrieval
            # turns (actual product searches), not just CHITCHAT/greetings.
            _has_search_history = any(
                t.get("retrieval_strategy") == "FULL" or
                any(kw in (t.get("content") or "").lower()
                    for kw in ["option 1", "option 2", "£", "found two", "found these"])
                for t in history
                if t.get("role") == "bot"
            )
            if _has_search_history:
                return "REFINEMENT", "FULL"
        return "INITIAL_REQUEST", "FULL"

    def _extract_entities(self, message: str) -> dict:
        """
        Extracts structured entities from a user message.
        Maps to column names from sample_articles.csv.
        """
        entities = {}
        msg = message.lower()

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
