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
# Score dimensions (0.0–1.0, used to compute the sufficiency score):
#   D_SELF         : LLM can answer from parametric knowledge alone
#   D_ITEMS        : relevant items found in session context
#   D_RECENCY      : how recent the available context is
#   D_COMPLETENESS : how complete the available information is
#
# Score formula (for PARTIAL labels only):
#   S = 0.40·D_ITEMS + 0.35·D_RECENCY + 0.25·D_COMPLETENESS
#
# For CHITCHAT/FEEDBACK and INITIAL_REQUEST/REFINEMENT the tier is forced
# — score is still computed for transparency but does not drive the decision.
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

    Core fields:
        tier              : retrieval tier — FULL | PARTIAL | NO
        score             : sufficiency score 0.0–1.0
        label             : DistilBERT label for this turn
        prior_strategy    : original DistilBERT default strategy
        override          : whether CSE changed the strategy
        rationale         : human-readable scientific explanation

    Sub-level routing fields (do NOT change the main tier):
        full_subtype      : "FULL_STANDARD" | "FULL_WITH_EXCLUSIONS"
        partial_subtype   : "PARTIAL_RECENT" | "PARTIAL_SESSION"
        excluded_ids      : article_ids to exclude (from similar prior questions)

    Dimension scores (transparency / explainability):
        d_self_sufficient : parametric LLM knowledge sufficient
        d_items_available : relevant items found in session context
        d_info_recency    : recency of available context (1.0 = very recent)
        d_info_completeness: completeness of available information
    """
    tier:               str
    score:              float
    label:              str
    prior_strategy:     str
    override:           bool
    rationale:          str = ""

    full_subtype:         Optional[str] = None
    partial_subtype:      Optional[str] = None
    excluded_ids:         list = field(default_factory=list)

    d_self_sufficient:    float = 0.0
    d_items_available:    float = 0.0
    d_info_recency:       float = 0.0
    d_info_completeness:  float = 0.0


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
        # result.tier          → "PARTIAL"
        # result.partial_subtype → "PARTIAL_RECENT"
        # result.score         → 0.75
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
            SufficiencyResult — tier + sub-level + score + dimensions
        """
        prior_strategy = self._label_to_default_strategy(label)

        # ── Route to the correct evaluator based on label group ────────────

        if label in ("CHITCHAT", "FEEDBACK"):
            result = self._eval_dialogue(label, prior_strategy)

        elif label == "INITIAL_REQUEST":
            result = await self._eval_initial_request(
                message, session_id, prior_strategy
            )

        elif label == "REFINEMENT":
            result = await self._eval_refinement(
                dialogue_state, history, session_id, prior_strategy
            )

        else:
            # ATTRIBUTE_QUESTION / EXPLANATION_WHY / COMPARISON / SELECTION_REFERENCE
            result = await self._eval_item_reference(
                label, message, dialogue_state, history, session_id, prior_strategy
            )

        # ── Debug logging ─────────────────────────────────────────────────
        print(f"[CSE] ━━━ Context Sufficiency Evaluation ━━━")
        print(f"[CSE] label={label}  prior_strategy={prior_strategy}")
        print(f"[CSE] score={result.score:.4f}  tier={result.tier}  "
              f"full_sub={result.full_subtype}  partial_sub={result.partial_subtype}")
        print(f"[CSE] D: self={result.d_self_sufficient:.2f}  "
              f"items={result.d_items_available:.2f}  "
              f"recency={result.d_info_recency:.2f}  "
              f"completeness={result.d_info_completeness:.2f}")
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

    def _eval_dialogue(self, label: str, prior_strategy: str) -> SufficiencyResult:
        """
        CHITCHAT / FEEDBACK — pure dialogue management, no retrieval needed.
        LLM answers entirely from parametric knowledge (greeting, acknowledge,
        sentiment reaction). I(A;K|q,C) ≈ 0 — catalog adds nothing.
        """
        score = 0.95 if label == "CHITCHAT" else 0.90
        print(f"[CSE-DIALOGUE] label={label}  score={score}  → tier=NO  (no retrieval needed)")
        return SufficiencyResult(
            tier="NO",
            score=score,
            label=label,
            prior_strategy=prior_strategy,
            override=(prior_strategy != "NO"),
            full_subtype=None,
            partial_subtype=None,
            excluded_ids=[],
            d_self_sufficient=1.0,
            d_items_available=1.0,
            d_info_recency=1.0,
            d_info_completeness=1.0,
            rationale=(
                f"{label} → NO retrieval. S={score:.3f}. "
                f"Pure dialogue turn — LLM answers from parametric knowledge. "
                f"I(A;K|q,C)≈0: catalog adds no information. "
                f"[Joren2025: Sufficient(q,∅,param)=1]"
            ),
        )

    async def _eval_initial_request(
        self,
        message:        str,
        session_id:     str,
        prior_strategy: str,
    ) -> SufficiencyResult:
        """
        INITIAL_REQUEST — always FULL retrieval (candidate set unknown).

        Additionally checks whether the same or very similar question was
        asked earlier in this session. If found, the article_ids from the
        prior recommendation are loaded from MongoDB's recommendations
        collection and returned as excluded_ids so the retrieval engine
        does not repeat the same products.

        Sub-level:
          FULL_WITH_EXCLUSIONS — similar prior question found, exclude IDs
          FULL_STANDARD        — fresh question, no exclusions
        """
        print(f"[CSE-INIT] checking similar prior questions in session={session_id}")
        excluded_ids = await self._find_similar_question_exclusions(
            current_message=message,
            session_id=session_id,
        )
        full_subtype = "FULL_WITH_EXCLUSIONS" if excluded_ids else "FULL_STANDARD"
        print(f"[CSE-INIT] full_subtype={full_subtype}  excluded_ids_count={len(excluded_ids)}")

        return SufficiencyResult(
            tier="FULL",
            score=0.10,
            label="INITIAL_REQUEST",
            prior_strategy=prior_strategy,
            override=(prior_strategy != "FULL"),
            full_subtype=full_subtype,
            partial_subtype=None,
            excluded_ids=excluded_ids,
            d_self_sufficient=0.0,
            d_items_available=0.0,
            d_info_recency=0.0,
            d_info_completeness=0.0,
            rationale=(
                f"INITIAL_REQUEST → {full_subtype}. S=0.10. "
                f"Candidate set entirely unknown — ANN catalog search required. "
                + (f"Excluded {len(excluded_ids)} article(s) from similar prior question(s). "
                   if excluded_ids else "")
                + f"[Jeong2024: multi-step retrieval; Joren2025: Sufficient(q,C_t)=0]"
            ),
        )

    async def _eval_refinement(
        self,
        dialogue_state: dict,
        history:        list[dict],
        session_id:     str,
        prior_strategy: str,
    ) -> SufficiencyResult:
        """
        REFINEMENT — always FULL retrieval, but with existing context.

        The user is modifying their search (different colour, lower price etc.)
        so the catalog must be re-queried. All article_ids ever recommended
        in this session are excluded so the user always sees fresh results.
        """
        has_constraints = bool(dialogue_state.get("hard_constraints", {}))
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

        score = round(
            0.10
            + (0.08 if has_constraints else 0.0)
            + (0.07 if has_items else 0.0),
            4,
        )
        full_subtype = (
            "FULL_WITH_EXCLUSIONS" if (has_items or has_constraints)
            else "FULL_STANDARD"
        )
        print(f"[CSE-REFINE] full_subtype={full_subtype}  score={score}  → tier=FULL")

        excl_note = f"Excluded {len(excluded_ids)} article(s) from full session history. " if excluded_ids else ""
        return SufficiencyResult(
            tier="FULL",
            score=score,
            label="REFINEMENT",
            prior_strategy=prior_strategy,
            override=(prior_strategy != "FULL"),
            full_subtype=full_subtype,
            partial_subtype=None,
            excluded_ids=excluded_ids,
            d_self_sufficient=0.0,
            d_items_available=0.3 if has_items else 0.0,
            d_info_recency=0.5 if history else 0.0,
            d_info_completeness=0.4 if has_constraints else 0.0,
            rationale=(
                f"REFINEMENT → {full_subtype}. S={score:.3f}. "
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

    @staticmethod
    def _article_ids_from_items(*items: dict) -> list[str]:
        """Collects unique non-empty article_id strings from the given item dicts."""
        ids: list[str] = []
        for item in items:
            aid = str(item.get("article_id", "")).strip()
            if aid and aid not in ids:
                ids.append(aid)
        return ids

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
             │        ├─ Yes → PARTIAL_RECENT   (score ~0.70–0.79)
             │        └─ No  → PARTIAL_SESSION  (score ~0.44–0.79)
             └─ No  → any recommendations in full session (MongoDB)?
                      ├─ Yes → PARTIAL_SESSION  (score ~0.40–0.55)
                      └─ No  → FULL_STANDARD    (score  0.20)
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
                d_items = 0.60
                d_recency = 0.25
                d_completeness = self._info_completeness(label, message, dialogue_state)
                print(f"[CSE-ITEMREF] MongoDB path → d_items={d_items}  d_recency={d_recency}  "
                      f"d_completeness={d_completeness:.2f}  → PARTIAL_SESSION")
                score = round(
                    0.40 * d_items + 0.35 * d_recency + 0.25 * d_completeness,
                    4,
                )
                score = max(score, 0.40)  # always PARTIAL for this path
                return SufficiencyResult(
                    tier="PARTIAL",
                    score=score,
                    label=label,
                    prior_strategy=prior_strategy,
                    override=(prior_strategy != "PARTIAL"),
                    full_subtype=None,
                    partial_subtype="PARTIAL_SESSION",
                    excluded_ids=[],
                    d_self_sufficient=0.0,
                    d_items_available=d_items,
                    d_info_recency=d_recency,
                    d_info_completeness=d_completeness,
                    rationale=(
                        f"{label} → PARTIAL_SESSION. S={score:.3f}. "
                        f"Items not in current dialogue state but found in "
                        f"session history (MongoDB). Bounded lookup sufficient. "
                        f"[Roy2024: follow-up on previously retrieved items]"
                    ),
                )
            else:
                # No items anywhere in session — must do a fresh catalog search
                print("[CSE-ITEMREF] no items anywhere in session → escalating to FULL retrieval")
                return SufficiencyResult(
                    tier="FULL",
                    score=0.20,
                    label=label,
                    prior_strategy=prior_strategy,
                    override=(prior_strategy != "FULL"),
                    full_subtype="FULL_STANDARD",
                    partial_subtype=None,
                    excluded_ids=[],
                    d_self_sufficient=0.0,
                    d_items_available=0.0,
                    d_info_recency=0.0,
                    d_info_completeness=0.0,
                    rationale=(
                        f"{label} → FULL (no items in session). S=0.20. "
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
        d_items = 1.0
        d_completeness = self._info_completeness(
            label, message, dialogue_state, target_item=target_item
        )
        items_recent = self._items_in_recent_history(history, target_item, None)

        if items_recent:
            d_recency = 1.0
            partial_subtype = "PARTIAL_RECENT"
        else:
            d_recency = 0.40
            partial_subtype = "PARTIAL_SESSION"

        score = round(
            0.40 * d_items + 0.35 * d_recency + 0.25 * d_completeness,
            4,
        )
        # These labels are NEVER NO retrieval — cap score below NO threshold
        score = min(score, 0.79)

        return SufficiencyResult(
            tier="PARTIAL",
            score=score,
            label=label,
            prior_strategy=prior_strategy,
            override=(prior_strategy != "PARTIAL"),
            full_subtype=None,
            partial_subtype=partial_subtype,
            excluded_ids=[],
            d_self_sufficient=0.0,
            d_items_available=d_items,
            d_info_recency=d_recency,
            d_info_completeness=d_completeness,
            rationale=(
                f"{label} → {partial_subtype}. S={score:.3f}. "
                f"Target item: '{target_name}'. "
                f"Context {'from last 3 exchanges (Redis)' if items_recent else 'from earlier in session (MongoDB)'}. "
                f"D_items={d_items:.2f} D_recency={d_recency:.2f} D_completeness={d_completeness:.2f}. "
                f"I(A;C_t) ≫ I(A;K\\C_t): bounded DB lookup sufficient. "
                f"[Joren2025: Sufficient(q,C_t)=1; Roy2024: follow-up on known items]"
            ),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Memory helpers
    # ══════════════════════════════════════════════════════════════════════════

    async def _find_similar_question_exclusions(
        self,
        current_message: str,
        session_id:      str,
    ) -> list:
        """
        For INITIAL_REQUEST: scan all prior INITIAL_REQUEST turns in this session.
        Any turn whose message has cosine similarity >= 0.75 with the current
        message is considered a duplicate question. The article_ids from the
        bot response that followed that turn are returned as excluded_ids so
        the retrieval engine does not recommend the same items again.

        Flow:
          1. Load full session turns from MongoDB (beyond Redis 10-turn buffer)
          2. Embed current message + all prior INITIAL_REQUEST messages
          3. Compare cosine similarity — threshold 0.75
          4. For matching turns: load recommendation from db.recommendations
             using the bot turn's recommendation_id
          5. Return list of article_ids to exclude
        """
        from memory.db.mongo import get_db
        db = get_db()

        sess_doc = await db.sessions.find_one(
            {"session_id": session_id},
            {"turns": 1},
        )
        if not sess_doc:
            return []

        all_turns = sess_doc.get("turns", [])

        # Collect prior INITIAL_REQUEST user turns with non-empty content
        prior_turns = [
            t for t in all_turns
            if t.get("role") == "user"
            and t.get("classification", {}).get("label") == "INITIAL_REQUEST"
            and t.get("content", "").strip()
        ]
        if not prior_turns:
            return []

        model = _get_embed_model()
        if model is None:
            return []

        messages = [t["content"] for t in prior_turns]
        try:
            all_embs = model.encode([current_message] + messages)
        except Exception as e:
            print(f"[CSE] Embedding error in similar-question check: {e}")
            return []

        current_emb = all_embs[0]
        similar_turn_numbers = []
        for i, t in enumerate(prior_turns):
            sim = _cosine(current_emb, all_embs[i + 1])
            if sim >= _SIMILAR_QUESTION_THRESHOLD:
                print(f"[CSE] Similar question (sim={sim:.3f}): "
                      f"'{t['content'][:60]}'")
                similar_turn_numbers.append(t.get("turn_number"))

        if not similar_turn_numbers:
            return []

        # Build a map of turn_number → turn for quick lookup
        turn_map = {t.get("turn_number"): t for t in all_turns}

        excluded_ids = []
        for turn_num in similar_turn_numbers:
            # The bot response immediately follows the user turn (turn_number + 1)
            bot_turn = turn_map.get(turn_num + 1)
            if not bot_turn or bot_turn.get("role") != "assistant":
                continue
            rec_id = bot_turn.get("recommendation_id")
            if not rec_id:
                continue
            rec = await db.recommendations.find_one(
                {"recommendation_id": rec_id},
                {"items": 1},
            )
            if rec:
                for item in rec.get("items", []):
                    aid = str(item.get("article_id", ""))
                    if aid and aid not in excluded_ids:
                        excluded_ids.append(aid)

        if excluded_ids:
            print(f"[CSE] {len(excluded_ids)} article(s) excluded "
                  f"(similar prior questions in session)")
        return excluded_ids

    async def _find_items_in_full_session(self, session_id: str) -> list:
        """
        Checks MongoDB's recommendations collection for any items that were
        recommended during this session. Used as fallback when
        dialogue_state.currently_discussing is empty.

        Returns the items list from the most recent recommendation, or [].
        """
        from memory.db.mongo import get_db
        db = get_db()

        rec = await db.recommendations.find_one(
            {"session_id": session_id},
            {"items": 1},
            sort=[("created_at", -1)],
        )
        if rec:
            return rec.get("items", [])
        return []

    async def _all_session_article_ids(self, session_id: str) -> list[str]:
        """
        Returns every article_id recommended across ALL turns in this session.

        Queries all documents in db.recommendations for this session_id and
        collects every article_id, deduplicated. Used by REFINEMENT so that
        the re-search never surfaces a product the user has already seen,
        regardless of how many turns have passed since it was recommended.
        """
        from memory.db.mongo import get_db
        db = get_db()

        cursor = db.recommendations.find(
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

    def _info_completeness(
        self,
        label:          str,
        message:        str,
        dialogue_state: dict,
        target_item:    Optional[dict] = None,
    ) -> float:
        """
        Measures how complete the available session information is for
        answering this specific label without a new catalog search.

        Uses target_item (the resolved specific item the user asked about)
        to check the RIGHT item's fields — not always item_a.

        Returns a float in [0.0, 1.0].
        """
        item = self._resolve_item_dict(target_item, dialogue_state)
        dispatch = {
            "ATTRIBUTE_QUESTION":  lambda: self._completeness_attribute(message.lower(), item),
            "EXPLANATION_WHY":     lambda: self._completeness_explanation(dialogue_state, item),
            "COMPARISON":          lambda: self._completeness_comparison(dialogue_state, item),
            "SELECTION_REFERENCE": lambda: self._completeness_selection(item),
        }
        handler = dispatch.get(label)
        return handler() if handler else 0.50

    @staticmethod
    def _resolve_item_dict(
        target_item:    Optional[dict],
        dialogue_state: dict,
    ) -> dict:
        """Returns a plain dict for the target item, falling back to item_a."""
        item = target_item
        if not item:
            item = dialogue_state.get("currently_discussing", {}).get("item_a") or {}
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        return item or {}

    @staticmethod
    def _completeness_attribute(msg: str, item: dict) -> float:
        """Completeness score for ATTRIBUTE_QUESTION based on which attribute is asked."""
        if any(w in msg for w in ["price", "cost", "how much", "£"]):
            return 0.90 if item.get("price") else 0.40
        if any(w in msg for w in ["colour", "color"]):
            return 0.90 if item.get("colour_group_name") else 0.50
        if any(w in msg for w in ["type", "category", "kind"]):
            return 0.90 if item.get("product_type_name") else 0.50
        if any(w in msg for w in ["material", "fabric", "made"]):
            return 0.30  # never stored in session — always needs DB lookup
        if any(w in msg for w in ["description", "detail", "tell me about"]):
            return 0.80 if item.get("detail_desc") else 0.30
        return 0.50

    @staticmethod
    def _completeness_explanation(dialogue_state: dict, item: dict) -> float:
        """Completeness score for EXPLANATION_WHY."""
        prefs = dialogue_state.get("preference_profile", {})
        if prefs and item:
            return 0.80
        if item:
            return 0.55  # can explain based on item attributes alone
        return 0.30

    @staticmethod
    def _completeness_comparison(dialogue_state: dict, item: dict) -> float:
        """Completeness score for COMPARISON — requires both items."""
        item_b = dialogue_state.get("currently_discussing", {}).get("item_b") or {}
        if hasattr(item_b, "model_dump"):
            item_b = item_b.model_dump()
        return 0.80 if (item and item_b) else 0.30

    @staticmethod
    def _completeness_selection(item: dict) -> float:
        """Completeness score for SELECTION_REFERENCE."""
        if item.get("prod_name") and item.get("colour_group_name"):
            return 0.90
        return 0.50 if item else 0.30

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
