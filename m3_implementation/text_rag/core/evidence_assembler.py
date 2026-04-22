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
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
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

        # Step 1: Semantic search in Qdrant
        # Build semantic query from user message + filter values
        semantic_query = user_message
        if filters.get("colour_group_name"):
            semantic_query += f" {filters['colour_group_name']}"
        if filters.get("product_type_name"):
            semantic_query += f" {filters['product_type_name']}"
        if soft_constraints.get("style"):
            semantic_query += f" {soft_constraints['style']}"
        if soft_constraints.get("occasion"):
            semantic_query += f" {soft_constraints['occasion']}"

        qdrant_results = semantic_search(
            query=semantic_query,
            filters=filters,
            exclude_ids=exclude_ids,
            top_k=10
        )

        # Step 2: PostgreSQL filtered search for ranking diversity
        pg_results = await search_articles_filtered(
            filters=filters,
            exclude_ids=exclude_ids,
            preference_boosts=preference_boosts,
            purchase_hints=purchase_hints,
            limit=10
        )

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

        # Take top MAX_RECOMMENDATIONS (2)
        top_articles = merged[:MAX_RECOMMENDATIONS]

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
        """Fetches both articles and assembles comparison evidence."""
        payload    = ri.get("payload", {})
        id_a       = payload.get("article_id_a")
        id_b       = payload.get("article_id_b")
        dimension  = payload.get("comparison_dimension", "overall")
        pref_weights = payload.get("preference_weights", {})
        user_message = ri.get("user_message", "")

        item_a, item_b = await get_articles_for_comparison(
            str(id_a) if id_a else "",
            str(id_b) if id_b else ""
        )

        # Build dimension-specific comparison facts
        comparison_facts = _build_comparison_facts(item_a, item_b, dimension)

        return {
            "action":            "item_compare",
            "user_message":      user_message,
            "item_a":            _article_summary(item_a) if item_a else None,
            "item_b":            _article_summary(item_b) if item_b else None,
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
        Critically includes prior_claims so the LLM cannot contradict them.
        """
        payload      = ri.get("payload", {})
        article_id   = payload.get("article_id")
        prior_claims = payload.get("prior_claims", [])
        matched_prefs= payload.get("matched_prefs", [])
        user_message = ri.get("user_message", "")

        article = None
        if article_id:
            article = await get_article_by_id(str(article_id))

        # Find which preferences actually match this article
        confirmed_matches = []
        if article and matched_prefs:
            for pref in matched_prefs:
                attr  = pref.get("attribute_name")
                val   = pref.get("attribute_value")
                weight= pref.get("weight", 0)
                if attr and article.get(attr) == val:
                    confirmed_matches.append({
                        "attribute": attr,
                        "value":     val,
                        "weight":    weight,
                        "confirmed": True,
                    })

        return {
            "action":            "explanation_generate",
            "user_message":      user_message,
            "article":           _article_summary(article) if article else None,
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
        """Fetches full article details."""
        payload      = ri.get("payload", {})
        article_id   = payload.get("article_id")
        user_message = ri.get("user_message", "")

        article = None
        if article_id:
            article = await get_article_by_id(str(article_id))

        return {
            "action":       "item_detail_lookup",
            "user_message": user_message,
            "article":      _article_summary(article) if article else None,
        }

    # ── no retrieval (FEEDBACK / CHITCHAT) ────────────────────────────────────

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

