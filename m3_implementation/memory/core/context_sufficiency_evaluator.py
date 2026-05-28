# m3_implementation/memory/core/context_sufficiency_evaluator.py
#
# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT SUFFICIENCY EVALUATOR (CSE) — v2
# ═══════════════════════════════════════════════════════════════════════════════
#
# Scientific basis:
#   Implements the information-theoretic tier assignment from:
#   - Joren et al. "Sufficient Context: A New Lens on RAG" (ICLR 2025)
#   - Jeong et al. "Adaptive-RAG" (NAACL 2024)
#   - Wang et al. "RAGate: Adaptive RAG for Conversational Systems" (NAACL 2025)
#
# Three-tier decision:
#   tier = NO       — CHITCHAT / FEEDBACK       (pure dialogue, no retrieval)
#   tier = FULL     — INITIAL_REQUEST / REFINEMENT  (catalog ANN search needed)
#   tier = PARTIAL  — ATTRIBUTE_QUESTION / EXPLANATION_WHY / COMPARISON /
#                     SELECTION_REFERENCE        (bounded DB lookup using context)
#
# Sub-level routing fields (stored alongside tier, do NOT change the tier label):
#
#   full_subtype:
#     "FULL_STANDARD"         — fresh ANN catalog search, no prior context
#     "FULL_WITH_EXCLUSIONS"  — ANN search + exclude already-seen article_ids
#       Used for:
#         INITIAL_REQUEST → similar questions found in same session
#         REFINEMENT      → prior constraints / discussing items exist
#
#   partial_subtype:
#     "PARTIAL_RECENT"   — needed context found in last 3 exchanges (Redis hot)
#     "PARTIAL_SESSION"  — needed context found earlier in session (MongoDB)
#
# Routing decision is carried entirely by tier + subtype + excluded_ids.
# No numerical scores are computed — the tier assignment IS the decision.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# ── Similarity threshold for duplicate question detection ──────────────────────
# Two INITIAL_REQUEST messages with cosine similarity >= this are considered
# semantically equivalent — recommendations from the prior one are excluded.
_SIMILAR_QUESTION_THRESHOLD = 0.75


def _cosine(a, b) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def _get_embed_model():
    """Reuses the all-MiniLM-L6-v2 singleton from entity_extractor (avoids loading twice)."""
    try:
        from memory.core.entity_extractor import _get_model
        return _get_model()
    except Exception:
        return None


@dataclass
class SufficiencyResult:
    """
    Output of the Context Sufficiency Evaluator.

    Routing fields (these drive downstream behaviour):
        tier              : retrieval tier — FULL | PARTIAL | NO
        full_subtype      : "FULL_STANDARD" | "FULL_WITH_EXCLUSIONS"
        partial_subtype   : "PARTIAL_RECENT" | "PARTIAL_SESSION"
        excluded_ids      : article_ids to exclude from the next search
        override          : True when CSE changed DistilBERT's default strategy

    Metadata (for logging and rationale display only):
        label             : DistilBERT label for this turn
        prior_strategy    : DistilBERT default before CSE override
        rationale         : human-readable explanation of the tier decision
        cached_recommendation : items from a matched prior INITIAL_REQUEST
    """
    tier:               str
    label:              str
    prior_strategy:     str
    override:           bool
    rationale:          str = ""

    full_subtype:         Optional[str] = None
    partial_subtype:      Optional[str] = None
    excluded_ids:         list = field(default_factory=list)

    # Populated when a similar prior INITIAL_REQUEST is found — used to show
    # the cached recommendation instead of doing a new catalog search.
    cached_recommendation: list = field(default_factory=list)


class ContextSufficiencyEvaluator:
    """
    Evaluates Sufficient(q, C) for each user turn.

    Connects to Redis (hot session cache, last 10 turns) and MongoDB
    (full session history) to measure how much relevant context already
    exists, then assigns the appropriate retrieval tier and sub-level.

    The evaluate() method is async because it queries Redis and MongoDB.

    Usage:
        cse = get_cse()
        result = await cse.evaluate(
            label="ATTRIBUTE_QUESTION",
            message="What material is it?",
            dialogue_state=state_dict,
            history=history_list,
            session_id="sess_abc123",
        )
        # result.tier            → "PARTIAL"
        # result.partial_subtype → "PARTIAL_RECENT"
    """

    async def evaluate(
        self,
        label:          str,
        message:        str,
        dialogue_state: dict,
        history:        list[dict],
        session_id:     str,
        confidence:     float = 0.0,
    ) -> SufficiencyResult:
        """
        Assign retrieval tier and sub-level for a single user turn.

        Args:
            label          : DistilBERT label for this turn
            message        : user's current message text
            dialogue_state : current session DialogueState as a plain dict
                             (from get_dialogue_state().model_dump())
            history        : last 3 exchanges from Redis
                             format: [{"role": "user"/"bot", "content": "..."}]
            session_id     : active session ID
            confidence     : DistilBERT confidence score

        Returns:
            SufficiencyResult — tier + subtype + excluded_ids + rationale
        """
        prior_strategy = self._label_to_default_strategy(label)

        # ── Route to the correct evaluator based on label group ────────────

        if label in ("CHITCHAT", "FEEDBACK"):
            result = self._eval_dialogue(label, prior_strategy, message, dialogue_state)

        elif label == "INITIAL_REQUEST":
            result = await self._eval_initial_request(
                message, session_id, prior_strategy, dialogue_state
            )

        elif label == "REFINEMENT":
            result = await self._eval_refinement(
                dialogue_state, session_id, prior_strategy
            )

        else:
            # ATTRIBUTE_QUESTION / EXPLANATION_WHY / COMPARISON / SELECTION_REFERENCE
            result = await self._eval_item_reference(
                label, message, dialogue_state, history, session_id, prior_strategy
            )

        # ── Debug logging ─────────────────────────────────────────────────
        print(f"[CSE] ━━━ Context Sufficiency Evaluation ━━━")
        print(f"[CSE] label={label}  prior_strategy={prior_strategy}")
        print(f"[CSE] tier={result.tier}  full_sub={result.full_subtype}  partial_sub={result.partial_subtype}")
        if result.excluded_ids:
            print(f"[CSE] excluded_ids ({len(result.excluded_ids)}): "
                  f"{result.excluded_ids[:5]}"
                  f"{'...' if len(result.excluded_ids) > 5 else ''}")
        if result.override:
            print(f"[CSE] OVERRIDE: {prior_strategy} → {result.tier}")
        print(f"[CSE] rationale: {result.rationale}")

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Label-group evaluators
    # ══════════════════════════════════════════════════════════════════════════

    def _eval_dialogue(
        self,
        label:          str,
        prior_strategy: str,
        message:        str = "",
        dialogue_state: dict = None,
    ) -> SufficiencyResult:
        """
        CHITCHAT / FEEDBACK — dialogue management turns.

        CHITCHAT:
            LLM answers entirely from parametric knowledge.
            I(A;K|q,C) = 0 — catalog adds nothing.
            All D-dimensions = 1.0 → S = 1.0 [Joren2025].

        FEEDBACK (sentiment-driven via Twitter-RoBERTa, Barbieri et al. EMNLP 2020):
            positive  → tier=NO  (user satisfied; LLM acknowledges)
            neutral   → tier=NO  (no preference change expressed)
            negative + items in context → tier=FULL/FULL_WITH_EXCLUSIONS
                        (user implicitly requests alternatives; reject item excluded)
            negative + no items → tier=NO (cannot improve without context items)
        """
        # ── CHITCHAT ──────────────────────────────────────────────────────────
        if label == "CHITCHAT":
            print(f"[CSE-DIALOGUE] CHITCHAT → tier=NO")
            return SufficiencyResult(
                tier="NO",
                label=label,
                prior_strategy=prior_strategy,
                override=(prior_strategy != "NO"),
                rationale=(
                    "CHITCHAT → NO retrieval. Pure dialogue turn — LLM answers "
                    "from parametric knowledge. [Joren2025: I(A;K|q,C)=0]"
                ),
            )

        # ── FEEDBACK ──────────────────────────────────────────────────────────
        try:
            from memory.core.feedback_sentiment_classifier import classify_feedback
            sentiment_label, sentiment_score = classify_feedback(message)
        except Exception as _e:
            print(f"[CSE-DIALOGUE] classify_feedback error: {_e} — defaulting to neutral")
            sentiment_label, sentiment_score = "neutral", 0.0

        ds         = dialogue_state or {}
        discussing = ds.get("currently_discussing") or {}
        # Collect ALL items shown in this turn (item_a … item_d), not just item_a.
        # "I don't like them" rejects the whole set, so every shown id must be excluded.
        all_items  = [
            v for k, v in sorted(discussing.items())
            if k.startswith("item_") and v is not None
        ]
        item_a = discussing.get("item_a")

        # positive ─────────────────────────────────────────────────────────────
        if sentiment_label == "positive":
            print(f"[CSE-DIALOGUE] FEEDBACK positive (sent={sentiment_score:+.3f}) → tier=NO")
            return SufficiencyResult(
                tier="NO", label=label,
                prior_strategy=prior_strategy,
                override=(prior_strategy != "NO"),
                rationale=(
                    f"FEEDBACK positive (sentiment={sentiment_score:+.3f}) → NO retrieval. "
                    f"User satisfied — LLM acknowledges from parametric knowledge. "
                    f"[Barbieri2020 TweetEval RoBERTa: {sentiment_label}]"
                ),
            )

        # neutral ──────────────────────────────────────────────────────────────
        if sentiment_label == "neutral":
            print(f"[CSE-DIALOGUE] FEEDBACK neutral (sent={sentiment_score:+.3f}) → tier=NO")
            return SufficiencyResult(
                tier="NO", label=label,
                prior_strategy=prior_strategy,
                override=(prior_strategy != "NO"),
                rationale=(
                    f"FEEDBACK neutral (sentiment={sentiment_score:+.3f}) → NO retrieval. "
                    f"No clear preference expressed — LLM acknowledges. "
                    f"[Barbieri2020 TweetEval RoBERTa: {sentiment_label}]"
                ),
            )

        # negative + items in context → new search with exclusions ─────────────
        if all_items:
            excluded_ids = []
            for _it in all_items:
                _aid = (
                    _it.get("article_id") if isinstance(_it, dict)
                    else getattr(_it, "article_id", None)
                )
                if _aid and str(_aid) not in excluded_ids:
                    excluded_ids.append(str(_aid))
            print(
                f"[CSE-DIALOGUE] FEEDBACK negative+items (sent={sentiment_score:+.3f}) "
                f"→ tier=FULL  excluded={excluded_ids}"
            )
            return SufficiencyResult(
                tier="FULL",
                label=label,
                prior_strategy=prior_strategy,
                override=True,
                full_subtype="FULL_WITH_EXCLUSIONS",
                excluded_ids=excluded_ids,
                rationale=(
                    f"FEEDBACK negative (sentiment={sentiment_score:+.3f}) + items in context "
                    f"→ FULL_WITH_EXCLUSIONS (tier overridden). "
                    f"User implicitly requests alternatives; excluded={excluded_ids}. "
                    f"[Barbieri2020 TweetEval; Adaptive-RAG Jeong2024]"
                ),
            )

        # negative + no items — cannot improve without context ─────────────────
        print(
            f"[CSE-DIALOGUE] FEEDBACK negative+no-items (sent={sentiment_score:+.3f}) → tier=NO"
        )
        return SufficiencyResult(
            tier="NO",
            label=label,
            prior_strategy=prior_strategy,
            override=(prior_strategy != "NO"),
            rationale=(
                f"FEEDBACK negative (sentiment={sentiment_score:+.3f}) but no items in context. "
                f"Cannot trigger exclusion-based search without context items. "
                f"LLM responds with acknowledgment. [Barbieri2020 TweetEval]"
            ),
        )

    async def _eval_initial_request(
        self,
        message:        str,
        session_id:     str,
        prior_strategy: str,
        dialogue_state: dict,
    ) -> SufficiencyResult:
        """
        INITIAL_REQUEST — normally FULL retrieval (candidate set unknown).

        Exception: if a semantically similar question was asked earlier in
        this session (cosine >= 0.75), the tier is PARTIAL — we can return
        the cached recommendation instead of doing a new catalog search.
        The excluded_ids are preserved so the user can request fresh results.

        Sub-level:
          PARTIAL_RECENT  — cached items found in last 3 exchanges (Redis)
          PARTIAL_SESSION — cached items found in earlier session history
          FULL_STANDARD   — fresh question, no similar prior question
        """
        print(f"[CSE-INIT] checking similar prior questions in session={session_id}")
        excluded_ids, cached_items = await self._find_similar_question_exclusions(
            current_message=message,
            session_id=session_id,
        )
        print(f"[CSE-INIT] excluded_ids_count={len(excluded_ids)}  cached_items_count={len(cached_items)}")

        if excluded_ids and cached_items:
            # Similar question found — return cached recommendation as PARTIAL
            items_recent    = self._cached_items_in_recent_history(dialogue_state, cached_items)
            partial_subtype = "PARTIAL_RECENT" if items_recent else "PARTIAL_SESSION"
            print(f"[CSE-INIT] similar question detected → tier=PARTIAL/{partial_subtype}")
            return SufficiencyResult(
                tier="PARTIAL",
                label="INITIAL_REQUEST",
                prior_strategy=prior_strategy,
                override=True,
                partial_subtype=partial_subtype,
                excluded_ids=excluded_ids,
                cached_recommendation=cached_items,
                rationale=(
                    f"INITIAL_REQUEST → {partial_subtype}. "
                    f"Similar question asked earlier in session — using cached recommendation "
                    f"({len(cached_items)} item(s)). Excluded IDs retained for optional new search. "
                    f"[Joren2025: Sufficient(q,C_t)=1 — bounded session lookup sufficient]"
                ),
            )

        # Fresh question — standard FULL retrieval
        print(f"[CSE-INIT] full_subtype=FULL_STANDARD  excluded_ids_count=0")
        return SufficiencyResult(
            tier="FULL",
            label="INITIAL_REQUEST",
            prior_strategy=prior_strategy,
            override=(prior_strategy != "FULL"),
            full_subtype="FULL_STANDARD",
            rationale=(
                "INITIAL_REQUEST → FULL_STANDARD. "
                "Candidate set entirely unknown — ANN catalog search required. "
                "[Jeong2024: multi-step retrieval; Joren2025: Sufficient(q,C_t)=0]"
            ),
        )

    async def _eval_refinement(
        self,
        dialogue_state: dict,
        session_id:     str,
        prior_strategy: str,
    ) -> SufficiencyResult:
        """
        REFINEMENT — always FULL retrieval, but with existing context.

        The user is modifying their search (different colour, lower price etc.)
        so the catalog must be re-queried. All article_ids ever recommended
        in this session are excluded so the user always sees fresh results.
        """
        has_constraints = bool(dialogue_state.get("hard_constraints", {}).get("product_type_name"))
        item_a, item_b = self._discussing_items(dialogue_state)
        has_items = bool(item_a or item_b)

        print(f"[CSE-REFINE] has_constraints={has_constraints}  has_items={has_items}  "
              f"hard_constraints={dialogue_state.get('hard_constraints', {})}")
        print(f"[CSE-REFINE] currently_discussing: "
              f"item_a='{item_a.get('prod_name', '—')}' (article_id={item_a.get('article_id', '—')})  "
              f"item_b='{item_b.get('prod_name', '—')}' (article_id={item_b.get('article_id', '—')})")

        # Exclude every article_id recommended so far in this session so the
        # user never sees the same product again when they refine their search.
        excluded_ids = await self._all_session_article_ids(session_id)
        print(f"[CSE-REFINE] excluded_ids (all session recommendations): {excluded_ids}")

        full_subtype = (
            "FULL_WITH_EXCLUSIONS" if (has_items or has_constraints)
            else "FULL_STANDARD"
        )
        print(f"[CSE-REFINE] full_subtype={full_subtype} → tier=FULL")

        excl_note = f"Excluded {len(excluded_ids)} article(s) from full session history. " if excluded_ids else ""
        return SufficiencyResult(
            tier="FULL",
            label="REFINEMENT",
            prior_strategy=prior_strategy,
            override=(prior_strategy != "FULL"),
            full_subtype=full_subtype,
            excluded_ids=excluded_ids,
            rationale=(
                f"REFINEMENT → {full_subtype}. "
                f"Constraints={'yes' if has_constraints else 'no'}  "
                f"items={'yes' if has_items else 'no'}. "
                f"{excl_note}"
                "Catalog re-search required with updated filters. "
                "[Joren2025: Sufficient(q,C_t)=0 — new ANN search needed]"
            ),
        )

    @staticmethod
    def _discussing_items(dialogue_state: dict) -> tuple[dict, dict]:
        """Returns (item_a, item_b) from currently_discussing as plain dicts."""
        discussing = dialogue_state.get("currently_discussing", {})
        item_a = discussing.get("item_a") or {}
        item_b = discussing.get("item_b") or {}
        if hasattr(item_a, "model_dump"):
            item_a = item_a.model_dump()
        if hasattr(item_b, "model_dump"):
            item_b = item_b.model_dump()
        return item_a, item_b

    async def _eval_item_reference(
        self,
        label:          str,
        message:        str,
        dialogue_state: dict,
        history:        list[dict],
        session_id:     str,
        prior_strategy: str,
    ) -> SufficiencyResult:
        """
        ATTRIBUTE_QUESTION / EXPLANATION_WHY / COMPARISON / SELECTION_REFERENCE

        These labels reference items that were already recommended. The tier
        depends on whether the relevant items exist in session memory and how
        recent they are.

        Decision tree:
          1. Items in dialogue_state.currently_discussing?
             ├─ Yes → are they in the last 3 exchanges (history)?
             │        ├─ Yes → PARTIAL_RECENT   
             │        └─ No  → PARTIAL_SESSION 
             └─ No  → any recommendations in full session (MongoDB)?
                      ├─ Yes → PARTIAL_SESSION 
                      └─ No  → FULL_STANDARD 
        """
        discussing = dialogue_state.get("currently_discussing", {})
        item_a = discussing.get("item_a")
        item_b = discussing.get("item_b")
        print(f"[CSE-ITEMREF] label={label}  msg='{message[:60]}'")
        print(f"[CSE-ITEMREF] dialogue_state.currently_discussing: "
              f"item_a={bool(item_a)} ('{(item_a or {}).get('prod_name', '—')}')  "
              f"item_b={bool(item_b)} ('{(item_b or {}).get('prod_name', '—')}')")

        # COMPARISON specifically requires both items
        needs_both = (label == "COMPARISON")
        has_sufficient = (bool(item_a) and bool(item_b)) if needs_both else bool(item_a or item_b)
        print(f"[CSE-ITEMREF] needs_both={needs_both}  has_sufficient={has_sufficient}")

        if not has_sufficient:
            # ── Fallback: check full session history in MongoDB ────────────
            print(f"[CSE-ITEMREF] no items in dialogue_state → querying MongoDB fallback (session={session_id})")
            session_items = await self._find_items_in_full_session(session_id)
            print(f"[CSE-ITEMREF] MongoDB fallback: found {len(session_items)} item(s) in session history")
            if session_items:
                print(f"[CSE-ITEMREF] MongoDB path → PARTIAL_SESSION")
                return SufficiencyResult(
                    tier="PARTIAL",
                    label=label,
                    prior_strategy=prior_strategy,
                    override=(prior_strategy != "PARTIAL"),
                    partial_subtype="PARTIAL_SESSION",
                    rationale=(
                        f"{label} → PARTIAL_SESSION. "
                        "Items not in current dialogue state but found in "
                        "session history (MongoDB). Bounded lookup sufficient. "
                        "[Roy2024: follow-up on previously retrieved items]"
                    ),
                )
            else:
                # No items anywhere in session — must do a fresh catalog search
                print("[CSE-ITEMREF] no items anywhere in session → escalating to FULL retrieval")
                return SufficiencyResult(
                    tier="FULL",
                    label=label,
                    prior_strategy=prior_strategy,
                    override=(prior_strategy != "FULL"),
                    full_subtype="FULL_STANDARD",
                    rationale=(
                        f"{label} → FULL (no items in session). "
                        f"No recommendations found in this session — "
                        f"catalog search required before referencing items. "
                        f"[Jeong2024: retrieval required]"
                    ),
                )

        # ── Resolve which specific item the user is asking about ──────────
        # Do this BEFORE checking completeness so we inspect the RIGHT item's
        # fields. "what material is option 2?" must check item_b, not item_a.
        target_item = self._resolve_target_item(message, item_a, item_b)
        target_name = (
            (target_item.get("prod_name") if isinstance(target_item, dict)
             else getattr(target_item, "prod_name", None)) or "unknown"
        )
        print(f"[CSE] resolved target item: '{target_name}' "
              f"(from label={label} msg='{message[:50]}')")

        # ── Items are in dialogue_state — check recency ────────────────────
        items_recent    = self._items_in_recent_history(history, target_item, None)
        partial_subtype = "PARTIAL_RECENT" if items_recent else "PARTIAL_SESSION"

        return SufficiencyResult(
            tier="PARTIAL",
            label=label,
            prior_strategy=prior_strategy,
            override=(prior_strategy != "PARTIAL"),
            partial_subtype=partial_subtype,
            rationale=(
                f"{label} → {partial_subtype}. "
                f"Target item: '{target_name}'. "
                f"Context {'from last 3 exchanges (Redis)' if items_recent else 'from earlier in session (MongoDB)'}. "
                "I(A;C_t) ≫ I(A;K\\C_t): bounded DB lookup sufficient. "
                "[Joren2025: Sufficient(q,C_t)=1; Roy2024: follow-up on known items]"
            ),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Memory helpers
    # ══════════════════════════════════════════════════════════════════════════

    async def _find_similar_question_exclusions(
        self,
        current_message: str,
        session_id:      str,
    ) -> tuple[list, list]:
        """
        For INITIAL_REQUEST: scan all prior INITIAL_REQUEST turns in this session.
        Any turn whose message has cosine similarity >= 0.75 with the current
        message is considered a duplicate question.

        Returns (excluded_ids, cached_items):
          excluded_ids  — article_id strings to exclude from any new search
          cached_items  — full item dicts from the matched prior recommendation
                          (used to show the cached result without a new search)
        """
        from memory.db.mongo import get_db
        db = get_db()

        sess_doc = await db.sessions.find_one(
            {"session_id": session_id},
            {"turns": 1},
        )
        if not sess_doc:
            return [], []

        all_turns = sess_doc.get("turns", [])

        prior_turns = [
            t for t in all_turns
            if t.get("role") == "user"
            and t.get("classification", {}).get("label") == "INITIAL_REQUEST"
            and t.get("content", "").strip()
        ]
        if not prior_turns:
            return [], []

        model = _get_embed_model()
        if model is None:
            return [], []

        messages = [t["content"] for t in prior_turns]
        try:
            all_embs = model.encode([current_message] + messages)
        except Exception as e:
            print(f"[CSE] Embedding error in similar-question check: {e}")
            return [], []

        current_emb = all_embs[0]
        similar_turn_numbers = []
        for i, t in enumerate(prior_turns):
            sim = _cosine(current_emb, all_embs[i + 1])
            if sim >= _SIMILAR_QUESTION_THRESHOLD:
                print(f"[CSE] Similar question (sim={sim:.3f}): '{t['content'][:60]}'")
                similar_turn_numbers.append(t.get("turn_number"))

        if not similar_turn_numbers:
            return [], []

        turn_map = {t.get("turn_number"): t for t in all_turns}
        coll_name = self._recommendations_coll_name()

        excluded_ids: list[str] = []
        cached_items: list[dict] = []
        for turn_num in similar_turn_numbers:
            bot_turn = turn_map.get(turn_num + 1)
            if not bot_turn or bot_turn.get("role") != "assistant":
                continue
            rec_id = bot_turn.get("recommendation_id")
            if not rec_id:
                continue
            rec = await db[coll_name].find_one(
                {"recommendation_id": rec_id},
                {"items": 1},
            )
            if rec:
                for item in rec.get("items", []):
                    aid = str(item.get("article_id", ""))
                    if aid and aid not in excluded_ids:
                        excluded_ids.append(aid)
                        cached_items.append(item)

        if excluded_ids:
            print(f"[CSE] {len(excluded_ids)} article(s) from similar prior question "
                  f"— will show as cached recommendation")
        return excluded_ids, cached_items

    @staticmethod
    def _cached_items_in_recent_history(
        dialogue_state: dict,
        cached_items:   list[dict],
    ) -> bool:
        """Returns True if any cached item was shown in the last recommended turn."""
        if not cached_items:
            return False
        discussing = (dialogue_state or {}).get("currently_discussing") or {}
        recent_ids = {
            str(v.get("article_id", "")).strip()
            for k, v in discussing.items()
            if k.startswith("item_") and isinstance(v, dict)
        }
        recent_ids.discard("")
        if not recent_ids:
            return False
        cached_ids = {
            str(item.get("article_id", "")).strip()
            for item in cached_items
        }
        cached_ids.discard("")
        return bool(recent_ids & cached_ids)

    def _recommendations_coll_name(self) -> str:
        """Returns the recommendations collection name for this CSE instance.
        M3 uses "recommendations"; M2/M1 subclasses override this."""
        return "recommendations"

    async def _find_items_in_full_session(self, session_id: str) -> list:
        """
        Checks the recommendations collection for any items recommended during
        this session. Used as fallback when dialogue_state.currently_discussing
        is empty. Returns the items list from the most recent recommendation, or [].
        """
        from memory.db.mongo import get_db
        db = get_db()
        rec = await db[self._recommendations_coll_name()].find_one(
            {"session_id": session_id},
            {"items": 1},
            sort=[("created_at", -1)],
        )
        return rec.get("items", []) if rec else []

    async def _all_session_article_ids(self, session_id: str) -> list[str]:
        """
        Returns every article_id recommended across ALL turns in this session.
        Used by REFINEMENT so the re-search never surfaces a product the user
        has already seen.
        """
        from memory.db.mongo import get_db
        db = get_db()
        cursor = db[self._recommendations_coll_name()].find(
            {"session_id": session_id},
            {"items": 1},
        )
        excluded: list[str] = []
        async for rec in cursor:
            for item in rec.get("items", []):
                aid = str(item.get("article_id", "")).strip()
                if aid and aid not in excluded:
                    excluded.append(aid)
        print(f"[CSE-REFINE] _all_session_article_ids: found {len(excluded)} unique IDs "
              f"across all recommendations in session={session_id}")
        return excluded

    # ══════════════════════════════════════════════════════════════════════════
    # Synchronous helpers (no I/O)
    # ══════════════════════════════════════════════════════════════════════════

    def _resolve_target_item(
        self,
        message: str,
        item_a:  Optional[dict],
        item_b:  Optional[dict],
    ) -> Optional[dict]:
        """
        Identifies which item the user is referring to in their message.

        Resolution order:
          1. Explicit ordinal reference ("second one", "option 2") → item_b
          2. Colour mention matching item's colour_group_name
          3. Product name word match
          4. Default → item_a (first / primary item)

        Works with plain dicts (dialogue_state stores items as model_dump()
        output, not ItemInContext objects).
        """
        msg = message.lower()

        if self._msg_refs_item_b(msg):
            return item_b

        b_colour = self._item_field(item_b, "colour_group_name")
        if b_colour and b_colour in msg:
            return item_b
        a_colour = self._item_field(item_a, "colour_group_name")
        if a_colour and a_colour in msg:
            return item_a

        # Check item_b first — name match is more significant for the non-default item
        if item_b and self._item_name_in_msg(item_b, msg):
            return item_b
        if item_a and self._item_name_in_msg(item_a, msg):
            return item_a

        return item_a or item_b  # default: first available

    @staticmethod
    def _msg_refs_item_b(msg: str) -> bool:
        """Returns True if the message explicitly references the second item."""
        return any(ref in msg for ref in [
            "second", "option 2", "the other", "second one",
            "the 2nd", "number two", "item 2", "2nd one", "#2",
            "latter", "last one",
        ])

    @staticmethod
    def _item_field(item: Optional[dict], field: str) -> str:
        """Safely reads a field from an item dict or object, returns lowercase."""
        if not item:
            return ""
        val = (
            item.get(field, "")
            if isinstance(item, dict)
            else getattr(item, field, "") or ""
        )
        return val.lower()

    @staticmethod
    def _item_name_in_msg(item: Optional[dict], msg: str) -> bool:
        """Returns True if any significant word of the item's name appears in msg."""
        name = (
            item.get("prod_name", "")
            if isinstance(item, dict)
            else getattr(item, "prod_name", "") or ""
        ).lower()
        return any(w in msg for w in name.split() if len(w) > 3)

    def _items_in_recent_history(
        self,
        history: list[dict],
        item_a:  Optional[dict],
        item_b:  Optional[dict],
    ) -> bool:
        """
        Checks whether the currently_discussing items appear in the recent
        turn history (last 3 exchanges, passed from pipeline's get_turns_as_history).

        Matches by product name words in bot turn content, and by generic
        recommendation markers (£, "option 1", etc.).
        """
        if not history:
            return False

        bot_content = " ".join(
            t.get("content", "").lower()
            for t in history
            if t.get("role") in ("assistant", "bot")
        )
        if not bot_content:
            return False

        # Check for item names in bot content
        for item in (item_a, item_b):
            if not item:
                continue
            name = (
                item.get("prod_name", "")
                if isinstance(item, dict)
                else getattr(item, "prod_name", "") or ""
            ).lower()
            if name and len(name) > 3:
                # Match on first 3 significant words (handles truncation)
                words = [w for w in name.split()[:4] if len(w) > 3]
                if any(w in bot_content for w in words):
                    return True

        # Generic recommendation markers (bot showed items)
        return any(
            kw in bot_content
            for kw in ["option 1", "option 2", "here are", "£", "found two", "found these"]
        )

    def _label_to_default_strategy(self, label: str) -> str:
        """Returns the DistilBERT default retrieval strategy for a label."""
        return {
            "INITIAL_REQUEST":    "FULL",
            "REFINEMENT":         "FULL",
            "ATTRIBUTE_QUESTION": "PARTIAL",
            "EXPLANATION_WHY":    "PARTIAL",
            "COMPARISON":         "PARTIAL",
            "SELECTION_REFERENCE":"PARTIAL",
            "FEEDBACK":           "NO",
            "CHITCHAT":           "NO",
        }.get(label, "FULL")


# ── Module-level singleton ─────────────────────────────────────────────────────
_cse_instance: Optional[ContextSufficiencyEvaluator] = None


def get_cse() -> ContextSufficiencyEvaluator:
    global _cse_instance
    if _cse_instance is None:
        _cse_instance = ContextSufficiencyEvaluator()
    return _cse_instance


# ── Member-specific CSE subclasses ─────────────────────────────────────────────
# Each member gets its own CSE instance that reads from its own recommendations
# collection.  The base class handles all logic; only the collection name differs.
# M3 uses the default base class.  M2 and M1 use the subclasses below.

class ContextSufficiencyEvaluatorM2(ContextSufficiencyEvaluator):
    """CSE for the M2 · Multimodal RAG member.
    Reads recommendations from 'm2_recommendations' so follow-up turns
    (ATTRIBUTE_QUESTION, EXPLANATION_WHY, etc.) find the right items."""

    def _recommendations_coll_name(self) -> str:
        return "m2_recommendations"


class ContextSufficiencyEvaluatorM1(ContextSufficiencyEvaluator):
    """CSE for the M1 · Graph RAG member.
    Reads recommendations from 'm1_recommendations'."""

    def _recommendations_coll_name(self) -> str:
        return "m1_recommendations"


_cse_m2_instance: Optional[ContextSufficiencyEvaluatorM2] = None
_cse_m1_instance: Optional[ContextSufficiencyEvaluatorM1] = None


def get_cse_for_model(model: str = "m3") -> ContextSufficiencyEvaluator:
    """Returns the singleton CSE for the given model member.

    Args:
        model: "m3" (default), "m2", or "m1"
    """
    global _cse_instance, _cse_m2_instance, _cse_m1_instance
    if model == "m2":
        if _cse_m2_instance is None:
            _cse_m2_instance = ContextSufficiencyEvaluatorM2()
        return _cse_m2_instance
    if model == "m1":
        if _cse_m1_instance is None:
            _cse_m1_instance = ContextSufficiencyEvaluatorM1()
        return _cse_m1_instance
    # default → m3
    if _cse_instance is None:
        _cse_instance = ContextSufficiencyEvaluator()
    return _cse_instance
