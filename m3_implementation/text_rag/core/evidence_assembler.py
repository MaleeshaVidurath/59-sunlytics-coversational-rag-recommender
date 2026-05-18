# m3_implementation/text_rag/core/evidence_assembler.py
#
# Routes each retrieval_input action to the correct database queries
# and assembles a structured evidence bundle for the LLM.
#
# THE EVIDENCE BUNDLE:
#   A structured dict that contains all facts the LLM is allowed to use.
#   The hallucination checker validates LLM output against this bundle.
#   Nothing the LLM says should go beyond what is in the evidence bundle.
#
# ACTION → EVIDENCE STRATEGY:
#   catalog_search      → Qdrant semantic search + PostgreSQL filter ranking
#   item_attribute_lookup→ PostgreSQL single article lookup
#   item_compare        → PostgreSQL two article lookup
#   explanation_generate→ PostgreSQL lookup + matched_prefs + prior_claims
#   item_detail_lookup  → PostgreSQL single article lookup
#   FEEDBACK/CHITCHAT   → no DB queries, memory_context only

import os
import sys
import re
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

_WORD_TO_DIGIT = {
    "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

def _extract_quantity(message: str, payload_qty: int = 0) -> int:
    """
    Extracts requested quantity from user message.
    e.g. "5 shirts"      -> 5  (honours the request exactly)
         "six shirts"     -> 6  (English word recognised)
         "I need a dress" -> 2  (default when no number given)
    No artificial cap — returns whatever the user asked for.
    Default: 2
    Accepts an optional payload_qty already extracted by the LLM entity extractor.
    """
    # Payload quantity from LLM entity extraction takes priority
    if payload_qty and payload_qty >= 2:
        return payload_qty

    # Normalise English number words to digits before regex matching
    msg = message.lower()
    for word, digit in _WORD_TO_DIGIT.items():
        msg = re.sub(rf"\b{word}\b", digit, msg)

    patterns = [
        r"\b([2-9])\s+(?:different\s+)?(?:colours?|colors?|options?|items?|pieces?|styles?)\b",
        r"\b([2-9])\s+(?:t[- ])?(?:dress|top|shirt|trouser|short|jacket|sweater|skirt|coat|blouse|bra|bras|sock|socks|shoe|shoes|boot|boots|jean|jeans|short|shorts|hoodie|hoodies|cardigan|cardigans|blazer|blazers|legging|leggings)s?\b",
        r"\b(?:show|find|get|give)\s+(?:me\s+)?([2-9])\b",
        r"\b(?:need|want|buy)\s+([2-9])\b",
        r"^([2-9])\s+",
    ]
    for pattern in patterns:
        m = re.search(pattern, msg)
        if m:
            return int(m.group(1))
    return 2  # default when no quantity mentioned


def _ensure_colour_diversity(articles: list, requested_qty: int) -> list:
    """
    Ensures returned items have diverse colours when multiple are requested.
    Takes the first item per unique colour, then fills with remaining.
    """
    if len(articles) <= 1:
        return articles
    seen_colours = set()
    diverse = []
    same_colour = []
    for art in articles:
        colour = art.get("colour_group_name", "").lower()
        if colour and colour not in seen_colours:
            seen_colours.add(colour)
            diverse.append(art)
        else:
            same_colour.append(art)
    result = diverse + same_colour
    return result[:requested_qty]
from text_rag.db.postgres_client import (
    get_article_by_id, get_articles_by_ids,
    search_articles_filtered, get_articles_for_comparison
)
from text_rag.db.qdrant_client import semantic_search
from text_rag.config import MAX_RECOMMENDATIONS


def _format_price(price) -> str:
    if price is None:
        return "Price not available"
    return f"£{float(price):.2f}"


def _article_summary(art: dict) -> dict:
    """Returns a clean summary dict of an article for the evidence bundle."""
    if not art:
        return {}
    return {
        "article_id":               str(art.get("article_id", "")),
        "name":                     art.get("prod_name", ""),
        "type":                     art.get("product_type_name", ""),
        "colour":                   art.get("colour_group_name", ""),
        "pattern":                  art.get("graphical_appearance_name", ""),
        "material_description":     art.get("detail_desc", ""),
        "garment_group":            art.get("garment_group_name", ""),
        "section":                  art.get("section_name", ""),
        "index_group":              art.get("index_group_name", ""),
        "price":                    _format_price(art.get("avg_price")),
        "price_raw":                art.get("avg_price"),
    }


class EvidenceAssembler:
    """
    Assembles the evidence bundle for each action type.
    Called by the RAG pipeline before LLM generation.
    """

    async def assemble(
        self,
        retrieval_input: dict,
        memory_context: dict
    ) -> dict:
        """
        Main entry point. Routes to the correct assembly method.

        Args:
            retrieval_input: From pipeline.process_turn() — may be None
            memory_context:  From pipeline.process_turn() — always present

        Returns evidence bundle dict.
        """
        if retrieval_input is None:
            return await self._assemble_no_retrieval(memory_context)

        action = retrieval_input.get("action")

        if action == "catalog_search":
            return await self._assemble_catalog_search(
                retrieval_input, memory_context
            )
        elif action == "item_attribute_lookup":
            return await self._assemble_attribute_lookup(
                retrieval_input, memory_context
            )
        elif action == "item_compare":
            return await self._assemble_comparison(
                retrieval_input, memory_context
            )
        elif action == "explanation_generate":
            return await self._assemble_explanation(
                retrieval_input, memory_context
            )
        elif action == "item_detail_lookup":
            return await self._assemble_detail_lookup(
                retrieval_input, memory_context
            )
        else:
            return await self._assemble_no_retrieval(memory_context)

    # ── catalog_search ─────────────────────────────────────────────────────────

    async def _assemble_catalog_search(
        self, ri: dict, mc: dict
    ) -> dict:
        """
        Assembles evidence for catalog search.
        Strategy:
          1. Qdrant semantic search to find semantically relevant articles
          2. PostgreSQL filtered search for structurally matching articles
          3. Merge and deduplicate, keeping top MAX_RECOMMENDATIONS
        """
        payload        = ri.get("payload", {})
        filters        = payload.get("filters", {})
        preference_boosts = payload.get("preference_boosts", [])
        purchase_hints = payload.get("purchase_history_hints", {})
        soft_constraints = payload.get("soft_constraints", {})
        exclude_ids    = ri.get("exclude_ids", [])
        user_message   = ri.get("user_message", "")
        payload_qty    = int(payload.get("quantity") or 0)

        # Step 1: Semantic search in Qdrant
        semantic_query = self._build_semantic_query(user_message, filters, soft_constraints)
        qdrant_results = semantic_search(
            query=semantic_query,
            filters=filters,
            exclude_ids=exclude_ids,
            penalties=payload.get("penalties", {}),
            top_k=max(20, _extract_quantity(user_message, payload_qty) * 5)
        )

        print(f"[ASSEMBLER-QDRANT] got {len(qdrant_results)} results")
        for _qr in qdrant_results[:3]: print(f"  [QDRANT] {str(_qr.get('article_id',_qr.get('id','?')))[:12]} {str(_qr.get('prod_name',_qr.get('name','?')))[:25]} {_qr.get('colour_group_name',_qr.get('colour','?'))}")
        # Step 2: PostgreSQL filtered search for ranking diversity
        # Fetch larger pool when user requests multiple items
        print(f"\n[ASSEMBLER-CATALOG] ━━━ catalog search ━━━")
        requested_qty = _extract_quantity(user_message, payload_qty)
        print(f"[ASSEMBLER-CATALOG] qty={requested_qty} msg='{user_message[:60]}'")
        print(f"[ASSEMBLER-CATALOG] filters={filters}")
        print(f"[ASSEMBLER-CATALOG] exclude_ids={exclude_ids}")
        print(f"[ASSEMBLER-CATALOG] purchase_hints={purchase_hints}")
        print(f"[ASSEMBLER-CATALOG] preference_boosts={preference_boosts}")
        search_limit  = max(20, requested_qty * 5)
        penalties     = payload.get("penalties", {})
        print(f"[ASSEMBLER-POSTGRES] searching limit={search_limit}")
        pg_results = await search_articles_filtered(
            filters=filters,
            exclude_ids=exclude_ids,
            preference_boosts=preference_boosts,
            purchase_hints=purchase_hints,
            penalties=penalties,
            limit=search_limit
        )

        print(f"[ASSEMBLER-POSTGRES] got {len(pg_results)} results")
        for _pr in pg_results[:3]: print(f"  [POSTGRES] {str(_pr.get('article_id','?'))[:12]} {str(_pr.get('prod_name','?'))[:25]} {_pr.get('colour_group_name','?')} £{_pr.get('avg_price','?')}")
        # Step 3: Merge — Qdrant results first (semantically relevant),
        # then add PostgreSQL results not already in Qdrant set
        seen_ids = set()
        merged   = []

        for art in qdrant_results:
            aid = str(art.get("article_id", ""))
            if aid not in seen_ids:
                seen_ids.add(aid)
                merged.append(art)

        for art in pg_results:
            aid = str(art.get("article_id", ""))
            if aid not in seen_ids:
                seen_ids.add(aid)
                merged.append(art)

        # Strip excluded IDs before selection — Qdrant's HasId filter can silently
        # fail on some client versions, so enforce the exclusion here as a hard gate.
        if exclude_ids:
            exclude_set = {str(x) for x in exclude_ids if x}
            before = len(merged)
            merged = [a for a in merged if str(a.get("article_id", "")) not in exclude_set]
            print(f"[ASSEMBLER-CATALOG] post-merge exclude filter: {before} → {len(merged)} items "
                  f"(removed {before - len(merged)} excluded)")

        # ── Filter relaxation fallback ──────────────────────────────────────
        # If full filters returned 0 results, retry with progressively looser
        # constraints so the user gets something rather than nothing.
        if not merged:
            # Pass 1: drop price constraints (most likely culprit)
            _price_keys = {"price_min", "price_max"}
            _relaxed = {k: v for k, v in filters.items() if k not in _price_keys}
            if _relaxed != filters:
                print(f"[ASSEMBLER-CATALOG] 0 results — retrying without price filters: {_relaxed}")
                _qr2 = semantic_search(
                    query=semantic_query, filters=_relaxed, exclude_ids=exclude_ids,
                    penalties=penalties, top_k=max(20, requested_qty * 5)
                )
                _pg2 = await search_articles_filtered(
                    filters=_relaxed, exclude_ids=exclude_ids,
                    preference_boosts=preference_boosts, purchase_hints=purchase_hints,
                    penalties=penalties, limit=search_limit
                )
                for art in _qr2 + _pg2:
                    aid = str(art.get("article_id", ""))
                    if aid not in seen_ids:
                        seen_ids.add(aid)
                        merged.append(art)
                print(f"[ASSEMBLER-CATALOG] after price-relaxed retry: {len(merged)} results")

        if not merged:
            # Pass 2: keep only product_type_name (broadest search)
            _minimal = {k: v for k, v in filters.items() if k == "product_type_name"}
            if _minimal and _minimal != filters:
                print(f"[ASSEMBLER-CATALOG] still 0 — retrying with product_type only: {_minimal}")
                _qr3 = semantic_search(
                    query=semantic_query, filters=_minimal, exclude_ids=exclude_ids,
                    penalties=penalties, top_k=max(20, requested_qty * 5)
                )
                _pg3 = await search_articles_filtered(
                    filters=_minimal, exclude_ids=exclude_ids,
                    preference_boosts=preference_boosts, purchase_hints=purchase_hints,
                    penalties=penalties, limit=search_limit
                )
                for art in _qr3 + _pg3:
                    aid = str(art.get("article_id", ""))
                    if aid not in seen_ids:
                        seen_ids.add(aid)
                        merged.append(art)
                print(f"[ASSEMBLER-CATALOG] after product_type-only retry: {len(merged)} results")

        # Apply colour diversity for multi-item requests
        if requested_qty > 2:
            top_articles = _ensure_colour_diversity(merged, requested_qty)
        else:
            top_articles = merged[:requested_qty]
        print(f"[DBG-4d] FINAL ITEMS: {len(top_articles)} selected from {len(merged)} merged")
        for _fa in top_articles:
            print(f"  [DBG-4d] → {_fa.get('article_id','?')} | {str(_fa.get('prod_name',_fa.get('name','?')))[:30]} | {_fa.get('colour_group_name',_fa.get('colour','?'))} | avg_price={_fa.get('avg_price','?')}")


        return {
            "action":          "catalog_search",
            "user_message":    user_message,
            "items":           [_article_summary(a) for a in top_articles],
            "filters_applied": filters,
            "soft_constraints":soft_constraints,
            "preference_boosts": preference_boosts,
            "purchase_hints":  purchase_hints,
            "user_preferences": mc.get("long_term_preferences", []),
            "style_profile":    mc.get("style_profile", {}),
            "result_count":    len(top_articles),
        }

    # ── item_attribute_lookup ──────────────────────────────────────────────────

    async def _assemble_attribute_lookup(
        self, ri: dict, mc: dict
    ) -> dict:
        """Fetches a single article and packages the relevant attribute."""
        print(f"\n[ASSEMBLER-ATTR] ━━━ attribute lookup ━━━")
        payload         = ri.get("payload", {})
        article_id      = payload.get("article_id")
        attribute_topic = payload.get("attribute_topic", "general_details")
        user_message    = ri.get("user_message", "")
        items_in_context= ri.get("items_in_context", {})

        article = None
        if article_id:
            article = await get_article_by_id(str(article_id))

        # Map attribute_topic to specific fields
        attribute_field_map = {
            "material_and_care":  ["detail_desc"],
            "colour_group_name":  ["colour_group_name", "graphical_appearance_name",
                                   "perceived_colour_master_name"],
            "sizing_and_fit":     ["detail_desc", "product_type_name"],
            "design_details":     ["detail_desc", "graphical_appearance_name",
                                   "garment_group_name"],
            "price":              ["avg_price"],
            "pockets":            ["detail_desc"],
            "availability":       ["article_id"],
            "general_details":    ["prod_name", "product_type_name",
                                   "colour_group_name", "detail_desc", "avg_price"],
        }
        relevant_fields = attribute_field_map.get(attribute_topic, ["detail_desc"])

        extracted = {}
        if article:
            for field in relevant_fields:
                val = article.get(field)
                if val is not None:
                    extracted[field] = (
                        _format_price(val) if field == "avg_price" else str(val)
                    )

        return {
            "action":          "item_attribute_lookup",
            "user_message":    user_message,
            "article":         _article_summary(article) if article else None,
            "attribute_topic": attribute_topic,
            "extracted_facts": extracted,
            "items_in_context":items_in_context,
        }

    # ── item_compare ───────────────────────────────────────────────────────────

    async def _assemble_comparison(
        self, ri: dict, mc: dict
    ) -> dict:
        """Fetches all context articles and assembles comparison evidence."""
        payload      = ri.get("payload", {})
        id_a         = payload.get("article_id_a")
        id_b         = payload.get("article_id_b")
        dimension    = payload.get("comparison_dimension", "overall")
        pref_weights = payload.get("preference_weights", {})
        user_message = ri.get("user_message", "")
        ids_list     = payload.get("article_ids_list")  # all context items if >2

        item_a, item_b = await get_articles_for_comparison(
            str(id_a) if id_a else "",
            str(id_b) if id_b else ""
        )

        # Fetch ALL context items when >2 were recommended
        items_all = None
        if ids_list and len(ids_list) > 2:
            all_ids = [entry["article_id"] for entry in ids_list]
            fetched = await get_articles_by_ids(all_ids)
            # Preserve order from ids_list; attach context prices where DB has none
            id_to_ctx = {entry["article_id"]: entry for entry in ids_list}
            items_all = []
            for art in fetched:
                aid = str(art.get("article_id", ""))
                ctx = id_to_ctx.get(aid, {})
                if art.get("avg_price") is None and ctx.get("price") is not None:
                    art = dict(art)
                    art["avg_price"] = ctx["price"]
                items_all.append(art)
            # Sort by price ascending for price comparisons
            if dimension == "price":
                items_all.sort(key=lambda x: float(x.get("avg_price") or 9999))

        # Build dimension-specific comparison facts
        comparison_facts = _build_comparison_facts(item_a, item_b, dimension)

        # For multi-item price comparison, override with full ranked list
        if items_all and dimension == "price":
            comparison_facts["all_items_ranked"] = [
                {
                    "name":   a.get("prod_name", ""),
                    "colour": a.get("colour_group_name", ""),
                    "price":  _format_price(a.get("avg_price")),
                }
                for a in items_all
            ]
            if items_all:
                cheapest = items_all[0]
                comparison_facts["cheaper_item"] = (
                    f"{cheapest.get('prod_name','')} ({cheapest.get('colour_group_name','')})"
                )
                comparison_facts["cheapest_price"] = _format_price(cheapest.get("avg_price"))

        return {
            "action":            "item_compare",
            "user_message":      user_message,
            "item_a":            _article_summary(item_a) if item_a else None,
            "item_b":            _article_summary(item_b) if item_b else None,
            "items_all":         items_all,
            "comparison_dimension": dimension,
            "comparison_facts":  comparison_facts,
            "preference_weights":pref_weights,
            "user_preferences":  mc.get("long_term_preferences", []),
        }

    # ── explanation_generate ───────────────────────────────────────────────────

    async def _assemble_explanation(
        self, ri: dict, mc: dict
    ) -> dict:
        """
        Assembles evidence for explanation generation.
        Fetches article from PostgreSQL and finds which user preferences match.
        """
        print(f"\n[ASSEMBLER-EXPLAIN] ━━━ _assemble_explanation ━━━")
        payload       = ri.get("payload", {})
        article_id    = payload.get("article_id")
        all_item_ids  = payload.get("all_item_ids")   # present when user asked "why" with no product name
        prior_claims  = payload.get("prior_claims", [])
        matched_prefs = payload.get("matched_prefs", [])
        user_message  = ri.get("user_message", "")
        items_ctx     = ri.get("items_in_context", {})

        print(f"[ASSEMBLER-EXPLAIN] article_id={article_id} all_item_ids={all_item_ids}")
        print(f"[ASSEMBLER-EXPLAIN] matched_prefs count={len(matched_prefs)}")
        print(f"[ASSEMBLER-EXPLAIN] prior_claims count={len(prior_claims)}")

        # ── All-items summary (no specific product named) ──────────────────
        if not article_id and all_item_ids:
            print(f"[ASSEMBLER-EXPLAIN] all-items summary for {len(all_item_ids)} items")
            fetched = await get_articles_by_ids(all_item_ids)
            articles_summary = [_article_summary(a) for a in fetched if a]
            print(f"[ASSEMBLER-EXPLAIN] fetched {len(articles_summary)} articles for summary")
            return {
                "action":            "explanation_generate",
                "user_message":      user_message,
                "article":           {},
                "articles":          articles_summary,   # multi-item path
                "prior_claims":      [],
                "confirmed_matches": [],
                "matched_prefs":     matched_prefs,
                "user_preferences":  mc.get("long_term_preferences", []),
                "style_profile":     mc.get("style_profile", {}),
            }

        context_art = payload.get("context_article") or {}
        article = None

        if context_art and context_art.get("detail_desc"):
            # Rich data already in session context — skip DB query entirely
            print(f"[ASSEMBLER-EXPLAIN] using stored context for article_id={article_id} (no DB query)")
            article = {
                "article_id":               str(context_art.get("article_id", "")),
                "prod_name":                context_art.get("prod_name", ""),
                "product_type_name":        context_art.get("product_type_name", ""),
                "colour_group_name":        context_art.get("colour_group_name", ""),
                "graphical_appearance_name":context_art.get("graphical_appearance_name", ""),
                "detail_desc":              context_art.get("detail_desc", ""),
                "garment_group_name":       context_art.get("garment_group_name", ""),
                "section_name":             context_art.get("section_name", ""),
                "index_group_name":         context_art.get("index_group_name", ""),
                "avg_price":                context_art.get("price"),
            }
        elif article_id:
            print(f"[ASSEMBLER-EXPLAIN] context missing detail_desc — querying DB for article_id={article_id}")
            article = await get_article_by_id(str(article_id))
            if article:
                print(f"[ASSEMBLER-EXPLAIN] article: name={article.get('prod_name')} colour={article.get('colour_group_name')} type={article.get('product_type_name')}")
            else:
                print(f"[ASSEMBLER-EXPLAIN] WARNING: article not found for id={article_id}")
                for slot in ['item_a', 'item_b']:
                    ctx_item = items_ctx.get(slot) or {}
                    if str(ctx_item.get('article_id','')) == str(article_id):
                        article = {
                            'article_id':        str(article_id),
                            'prod_name':         ctx_item.get('prod_name',''),
                            'product_type_name': ctx_item.get('product_type_name',''),
                            'colour_group_name': ctx_item.get('colour_group_name',''),
                            'avg_price':         ctx_item.get('price', 0),
                            'detail_desc':       ctx_item.get('detail_desc',''),
                        }
                        print(f"[ASSEMBLER-EXPLAIN] used items_in_context fallback for {slot}")
                        break

        # Find which user preferences are confirmed by the article attributes
        confirmed_matches = []
        if article and matched_prefs:
            for pref in matched_prefs:
                # Support both attribute_name (from enrichment) and attribute (legacy)
                attr  = pref.get("attribute_name") or pref.get("attribute")
                val   = pref.get("attribute_value") or pref.get("value")
                weight= pref.get("weight", 0)
                if attr and val and article.get(attr) == val:
                    confirmed_matches.append({
                        "attribute": attr,
                        "value":     val,
                        "weight":    weight,
                        "confirmed": True,
                    })

        print(f"[ASSEMBLER-EXPLAIN] confirmed_matches={len(confirmed_matches)}")
        for cm in confirmed_matches:
            print(f"  [CONFIRM] {cm['attribute']}={cm['value']} weight={cm['weight']:.2f}")

        return {
            "action":            "explanation_generate",
            "user_message":      user_message,
            "article":           _article_summary(article) if article else {},
            "prior_claims":      prior_claims,
            "confirmed_matches": confirmed_matches,
            "matched_prefs":     matched_prefs,
            "user_preferences":  mc.get("long_term_preferences", []),
            "style_profile":     mc.get("style_profile", {}),
        }

    # ── item_detail_lookup ─────────────────────────────────────────────────────

    async def _assemble_detail_lookup(
        self, ri: dict, mc: dict
    ) -> dict:
        """
        Fetches full article details.
        Uses context data stored at recommendation time when available (no DB query).
        Falls back to PostgreSQL only when detail_desc was not stored in context.
        """
        payload      = ri.get("payload", {})
        article_id   = payload.get("article_id")
        user_message = ri.get("user_message", "")
        context_art  = payload.get("context_article") or {}

        article = None

        if context_art and context_art.get("detail_desc"):
            # Rich data already in session context — skip DB query entirely
            print(f"[ASSEMBLER-DETAIL] using stored context for article_id={article_id} (no DB query)")
            article = {
                "article_id":               str(context_art.get("article_id", "")),
                "prod_name":                context_art.get("prod_name", ""),
                "product_type_name":        context_art.get("product_type_name", ""),
                "colour_group_name":        context_art.get("colour_group_name", ""),
                "graphical_appearance_name":context_art.get("graphical_appearance_name", ""),
                "detail_desc":              context_art.get("detail_desc", ""),
                "garment_group_name":       context_art.get("garment_group_name", ""),
                "section_name":             context_art.get("section_name", ""),
                "index_group_name":         context_art.get("index_group_name", ""),
                "avg_price":                context_art.get("price"),
            }
        elif article_id:
            print(f"[ASSEMBLER-DETAIL] context missing detail_desc — querying DB for article_id={article_id}")
            article = await get_article_by_id(str(article_id))

        return {
            "action":       "item_detail_lookup",
            "user_message": user_message,
            "article":      _article_summary(article) if article else None,
        }

    # ── no retrieval (FEEDBACK / CHITCHAT) ────────────────────────────────────

    @staticmethod
    def _build_semantic_query(user_message: str, filters: dict, soft_constraints: dict) -> str:
        """Appends key filter/constraint terms to the base message for Qdrant embedding."""
        parts = [user_message]
        for key in ("colour_group_name", "product_type_name"):
            val = filters.get(key)
            if val:
                parts.append(val)
        for key in ("style", "occasion"):
            val = soft_constraints.get(key)
            if val:
                parts.append(val)
        return " ".join(parts)

    async def _assemble_no_retrieval(self, mc: dict) -> dict:
        """For FEEDBACK and CHITCHAT — uses only memory_context."""
        return {
            "action":          "no_retrieval",
            "user_message":    mc.get("user_message", ""),
            "feedback":        mc.get("feedback"),
            "dialogue_state":  mc.get("dialogue_state", {}),
            "not_relevant":    mc.get("not_relevant", False),
            "refusal_message": mc.get("refusal_message"),
        }


# ── Comparison fact builder ────────────────────────────────────────────────────

def _build_comparison_facts(
    item_a: Optional[dict],
    item_b: Optional[dict],
    dimension: str
) -> dict:
    """
    Builds dimension-specific comparison facts from two articles.
    Only includes verifiable facts from the data — no inference.
    """
    if not item_a or not item_b:
        return {}

    facts = {
        "item_a_name":  item_a.get("prod_name", "Item A"),
        "item_b_name":  item_b.get("prod_name", "Item B"),
    }

    if dimension == "price":
        price_a = item_a.get("avg_price")
        price_b = item_b.get("avg_price")
        if price_a is not None and price_b is not None:
            facts["item_a_price"] = _format_price(price_a)
            facts["item_b_price"] = _format_price(price_b)
            if float(price_a) < float(price_b):
                facts["cheaper_item"] = item_a.get("prod_name")
                facts["price_difference"] = f"£{abs(float(price_a)-float(price_b)):.2f}"
            elif float(price_b) < float(price_a):
                facts["cheaper_item"] = item_b.get("prod_name")
                facts["price_difference"] = f"£{abs(float(price_a)-float(price_b)):.2f}"
            else:
                facts["same_price"] = True

    elif dimension == "colour":
        facts["item_a_colour"] = item_a.get("colour_group_name", "")
        facts["item_b_colour"] = item_b.get("colour_group_name", "")
        facts["item_a_pattern"]= item_a.get("graphical_appearance_name", "")
        facts["item_b_pattern"]= item_b.get("graphical_appearance_name", "")

    elif dimension == "material":
        facts["item_a_description"] = item_a.get("detail_desc", "")[:200]
        facts["item_b_description"] = item_b.get("detail_desc", "")[:200]

    elif dimension == "style_and_occasion":
        facts["item_a_section"]  = item_a.get("section_name", "")
        facts["item_b_section"]  = item_b.get("section_name", "")
        facts["item_a_garment"]  = item_a.get("garment_group_name", "")
        facts["item_b_garment"]  = item_b.get("garment_group_name", "")
        facts["item_a_desc_short"] = item_a.get("detail_desc", "")[:150]
        facts["item_b_desc_short"] = item_b.get("detail_desc", "")[:150]

    else:  # overall or fit or quality
        facts["item_a_price"]   = _format_price(item_a.get("avg_price"))
        facts["item_b_price"]   = _format_price(item_b.get("avg_price"))
        facts["item_a_colour"]  = item_a.get("colour_group_name", "")
        facts["item_b_colour"]  = item_b.get("colour_group_name", "")
        facts["item_a_type"]    = item_a.get("product_type_name", "")
        facts["item_b_type"]    = item_b.get("product_type_name", "")
        facts["item_a_desc_short"] = item_a.get("detail_desc", "")[:150]
        facts["item_b_desc_short"] = item_b.get("detail_desc", "")[:150]

    return facts

