"""
M2 Action Handlers — Implements each action type from the retrieval_input spec.

Each handler receives the full retrieval_input dict and returns a standardized
response dict with: action, success, response_text, items, error.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from shared.data_loader import data_loader
from m2_multimodal_rag.llm_generator import llm_generator
from m2_multimodal_rag.clip_embeddings import clip_encoder
from m2_multimodal_rag.faiss_index import faiss_db
from m2_multimodal_rag.hallucination_guard.regeneration_loop import generator_loop
from m2_multimodal_rag.hallucination_guard.layer_2_vlm_visual_verification import blip_verifier
from m2_multimodal_rag.cross_encoder_reranker import cross_encoder_reranker
from m2_multimodal_rag.diversity_bandit import diversity_bandit
from m2_multimodal_rag.knowledge_base.kb_retriever import kb_retriever
from m2_multimodal_rag.collaborative_filtering.cf_scorer import cf_scorer

# Attempt to load CF model at import time (silent if files not present)
cf_scorer.load()


# =====================================================================
# Accuracy Reporter (terminal diagnostic after every catalog_search)
# =====================================================================

_FUZZY_GROUPS = {
    "product_type_name": {
        "Dress":           ["dress"],
        "Top":             ["top", "t-shirt", "vest top", "blouse"],
        "Trousers":        ["trousers", "jeans"],
        "Jacket":          ["jacket", "coat", "blazer"],
        "Sweater":         ["sweater", "jumper", "knitwear"],
        "Hoodie":          ["hoodie", "sweatshirt"],
        "Shirt":           ["shirt"],
        "Blouse":          ["blouse"],
        "Leggings/Tights": ["leggings/tights", "leggings", "tights"],
    },
    "colour_group_name": {
        "Dark Blue": ["dark blue", "blue", "navy blue"],
        "White":     ["white", "off white"],
        "Black":     ["black"],
        "Red":       ["red", "dark red"],
        "Grey":      ["grey", "gray", "dark grey"],
    },
    "index_group_name": {
        "Ladieswear": ["ladieswear"],
        "Menswear":   ["menswear"],
        "Divided":    ["divided"],
        "Children":   ["children", "baby"],
    },
}


def _attr_match(item_val: str, expected_val: str, field: str) -> bool:
    iv = str(item_val).strip().lower()
    ev = str(expected_val).strip().lower()
    if iv == ev:
        return True
    for members in _FUZZY_GROUPS.get(field, {}).values():
        if ev in members and iv in members:
            return True
    return False


def _print_accuracy(items: list, filters: dict):
    """Prints a per-item accuracy report to the terminal using the request's hard filters."""
    checkable = {k: v for k, v in filters.items()
                 if k not in ("price_max", "price_min") and v is not None}

    print("\n" + "=" * 60)
    print("  [ACCURACY] Catalog Search Recommendation Report")
    print("=" * 60)

    if not checkable:
        print("  [ACCURACY] No checkable filters in this request — skipping score.")
        print("=" * 60 + "\n")
        return

    print("  Expected attributes from request filters:")
    for k, v in checkable.items():
        print(f"    {k}: {v}")

    if not items:
        print("  [ACCURACY] No items returned — cannot score.")
        print("=" * 60 + "\n")
        return

    item_accs = []
    for idx, item in enumerate(items, 1):
        hits = 0
        print(f"\n  Item {idx}: {item.get('prod_name', '?')}")
        print(f"  {'Attribute':<30} {'Expected':<18} {'Got':<20} Result")
        print(f"  {'-'*75}")
        for field, exp_val in checkable.items():
            got     = str(item.get(field, "")).strip()
            matched = _attr_match(got, str(exp_val), field)
            tick    = "[PASS]" if matched else "[FAIL]"
            print(f"  {field:<30} {str(exp_val):<18} {got:<20} {tick}")
            if matched:
                hits += 1
        acc = hits / len(checkable)
        item_accs.append(acc)
        print(f"  Item accuracy: {acc:.0%}")

    avg_acc  = sum(item_accs) / len(item_accs)
    full_hit = any(a >= 1.0 - 1e-9 for a in item_accs)
    part_hit = any(a >= 0.5 for a in item_accs)
    status   = "FULL HIT" if full_hit else ("PARTIAL HIT" if part_hit else "MISS")

    print(f"\n  {'─'*60}")
    print(f"  Result          : {status}")
    print(f"  Avg Accuracy    : {avg_acc:.0%}  ({avg_acc*100:.1f}% of expected attrs matched)")
    print(f"  Full Hit        : {'YES' if full_hit else 'NO'}  (all expected attrs matched in >=1 item)")
    print(f"  Items returned  : {len(items)}")
    print("=" * 60 + "\n")


# =====================================================================
# Shared Helpers
# =====================================================================

def _fetch_article(article_id: str) -> dict | None:
    """Fetches a single article's metadata from the articles CSV by article_id."""
    articles_df = data_loader.load_articles()
    try:
        match = articles_df[articles_df['article_id'] == int(article_id)]
    except (ValueError, TypeError):
        return None
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _format_article_for_response(metadata: dict) -> dict:
    """Formats raw CSV metadata into a clean response dict for the API."""
    return {
        "article_id":               str(metadata.get("article_id", "")).zfill(10),
        "prod_name":                metadata.get("prod_name", "Unknown"),
        "product_type_name":        metadata.get("product_type_name", "Unknown"),
        "product_group_name":       metadata.get("product_group_name", "Unknown"),
        "colour_group_name":        metadata.get("colour_group_name", "Unknown"),
        "department_name":          metadata.get("department_name", "Unknown"),
        "index_group_name":         metadata.get("index_group_name", "Unknown"),
        "detail_desc":              metadata.get("detail_desc", ""),
        "graphical_appearance_name":metadata.get("graphical_appearance_name", "Unknown"),
    }


def _call_llm(prompt: str) -> str | None:
    """Calls the cloud LLM (Groq) for a natural language response."""
    return llm_generator._call_llm(prompt, max_tokens=250)


def _vlm_verified_response(
    prompt: str,
    metadata: dict,
    article_id: str,
    max_attempts: int = 2,
) -> str | None:
    """
    Generates an LLM response, runs a self-reflection gate, then visually
    verifies it with ViLT against the product image.

    Fallback chain:
        1. Self-reflection fails → regenerate with corrective feedback
        2. VLM fails up to max_attempts → regenerate with visual feedback
        3. No image found → return self-reflection-cleared text as-is
    """
    response = _call_llm(prompt)
    if not response:
        return response

    # Self-reflection gate: LLM scores its own output (NOVELTY 4)
    passes_self, self_feedback = llm_generator.self_evaluate(response, metadata)
    if not passes_self:
        print(f"  [VLM] Self-reflection FAIL — regenerating. Reason: {self_feedback}")
        regenerated = _call_llm(
            prompt + f"\n\n[Your previous answer scored poorly: {self_feedback}. Improve it.]"
        )
        if regenerated:
            response = regenerated

    # VLM visual verification loop
    image_path = data_loader.get_image(article_id)
    if not image_path or not image_path.exists():
        print(f"  [VLM] No image for {article_id} — skipping visual gate.")
        return response

    for attempt in range(1, max_attempts + 1):
        is_valid, reason = blip_verifier.verify(str(image_path), response)
        if is_valid:
            print(f"  [VLM] Visual verification PASS (attempt {attempt}): {reason}")
            return response
        print(f"  [VLM] Visual verification FAIL (attempt {attempt}/{max_attempts}): {reason}")
        if attempt < max_attempts:
            corrected = _call_llm(
                prompt + f"\n\n[Visual check failed: {reason}. Correct your description.]"
            )
            if corrected:
                response = corrected

    return response


# =====================================================================
# HANDLER 1: catalog_search
# Triggered by: INITIAL_REQUEST, REFINEMENT  |  Strategy: FULL
# =====================================================================

def handle_catalog_search(retrieval_input: dict) -> dict:
    """
    6-phase search pipeline:

    Phase 1 — LLM Query Expansion + Multi-Vector CLIP Ensemble (NOVELTY 1)
    Phase 2 — Hard filter + boost / penalty / purchase_history scoring
    Phase 3 — Cross-Encoder Neural Re-ranking (MiniLM-BERT)
    Phase 4 — LLM Semantic Re-ranking (NOVELTY 2)
    Phase 5 — Thompson Sampling Diversity Bandit + MMR (NOVELTY 3)
    Phase 6 — Verified explanation generation with self-reflection gate (NOVELTY 4)
    """
    # --- Unpack request ---
    payload          = retrieval_input.get("payload", {})
    user_message     = retrieval_input.get("user_message", "")
    exclude_ids      = retrieval_input.get("exclude_ids", [])
    filters          = payload.get("filters", {})
    boosts           = payload.get("preference_boosts", [])
    penalties        = payload.get("penalties", {})
    soft_constraints = payload.get("soft_constraints", {})
    purchase_hints   = payload.get("purchase_history_hints", {})

    # --- Resolve requested number of items ---
    # Priority: payload.num_items > parsed from user_message > default 2
    # Capped at 6 to keep latency reasonable (each item runs VLM + LLM)
    _MAX_ITEMS = 6
    num_items = int(payload.get("num_items") or 2)
    if num_items <= 0:
        num_items = 2
    # Parse user message for explicit quantity ("show me 4 dresses", "give 3 options")
    import re as _re
    _qty_match = _re.search(r'\b([2-6])\s*(items?|options?|dresses?|outfits?|products?|tops?|pairs?)\b', user_message.lower())
    if _qty_match:
        num_items = int(_qty_match.group(1))
    num_items = min(num_items, _MAX_ITEMS)
    print(f"  [catalog_search] Requested items: {num_items}")

    print(f"  [catalog_search] Filters: {filters}")
    print(f"  [catalog_search] Soft constraints: {soft_constraints}")
    print(f"  [catalog_search] Purchase hints — dominant: "
          f"{purchase_hints.get('dominant_colour')}/{purchase_hints.get('dominant_type')}, "
          f"budget: {purchase_hints.get('budget_tier')}")
    print(f"  [catalog_search] Exclude IDs: {exclude_ids}")

    # --- Kansei style detection (NOVELTY 5 — Improvement 3) ---
    # Scans raw user message for emotional vocabulary (Nagamachi, 1995) so KB
    # scoring fires even when soft_constraints["style"] is absent.
    detected_kansei = kb_retriever.detect_kansei_from_message(user_message)
    inferred_style  = soft_constraints.get("style") or (detected_kansei[0] if detected_kansei else None)
    if detected_kansei and not soft_constraints.get("style"):
        print(f"  [KB] Kansei detected from message: {detected_kansei} → style='{inferred_style}'")

    # ---------------------------------------------------------------
    # PHASE 1 — LLM Query Expansion + Multi-Vector CLIP Ensemble
    # ---------------------------------------------------------------
    filter_terms = " ".join(str(v) for v in filters.values() if not isinstance(v, (int, float)))
    soft_terms   = " ".join(str(v) for v in soft_constraints.values() if v)

    # NOVELTY 5: inject psychology KB context into CLIP search text
    kb_query_context = kb_retriever.get_context(
        occasion=soft_constraints.get("occasion"),
        style_word=inferred_style,
    )
    clip_terms = kb_retriever.get_clip_terms(
        occasion=soft_constraints.get("occasion"),
        style_word=inferred_style,
    )
    if kb_query_context:
        print(f"  [KB] Query context injected: {kb_query_context[:80]}...")
    if clip_terms:
        print(f"  [KB] CLIP visual terms: {clip_terms[:80]}...")

    base_search_text = (
        f"{user_message} {filter_terms} {soft_terms} {kb_query_context} {clip_terms}".strip()
        or " ".join(str(v) for v in filters.values())
    )
    print(f"  [catalog_search] Base search text: '{base_search_text}'")

    expanded_queries = llm_generator.expand_query(base_search_text)
    query_vector     = clip_encoder.encode_expanded(expanded_queries)

    if query_vector is None:
        return {"action": "catalog_search", "success": False,
                "response_text": "I couldn't process your search request.",
                "items": [], "error": "CLIP encoding failed"}

    candidates = faiss_db.search(query_vector, top_k=50)
    if not candidates:
        return {"action": "catalog_search", "success": False,
                "response_text": "I couldn't find any items matching your search.",
                "items": [], "error": "No FAISS results"}

    # ---------------------------------------------------------------
    # PHASE 2 — Hard filter + boost / penalty / purchase history scoring
    # ---------------------------------------------------------------
    articles_df     = data_loader.load_articles()
    filtered_results = []

    for article_id, faiss_score in candidates:
        if article_id in exclude_ids or article_id.lstrip('0') in exclude_ids:
            continue

        try:
            article_row = articles_df[articles_df['article_id'] == int(article_id)]
        except (ValueError, TypeError):
            continue

        if article_row.empty:
            continue

        metadata = article_row.iloc[0].to_dict()

        # Hard filters — all must pass
        passes_filters = True
        for filter_key, filter_value in filters.items():
            if filter_key == "price_max":
                if metadata.get("price", float('inf')) > filter_value:
                    passes_filters = False; break
            elif filter_key == "price_min":
                if metadata.get("price", 0) < filter_value:
                    passes_filters = False; break
            else:
                if str(metadata.get(filter_key, "")).strip().lower() != str(filter_value).strip().lower():
                    passes_filters = False; break

        if not passes_filters:
            continue

        # Penalty score
        penalty_score = sum(
            0.3
            for penalty_key, penalty_values in penalties.items()
            for pv in penalty_values
            if str(metadata.get(penalty_key, "")).strip().lower() == str(pv).strip().lower()
        )

        # Preference boost score
        boost_score = sum(
            boost.get("weight", 0.0)
            for boost in boosts
            if str(metadata.get(boost.get("attribute", ""), "")).strip().lower()
               == str(boost.get("value", "")).strip().lower()
        )

        # Purchase history collaborative score
        # CF model (trained on 185,037 transactions) scores at item level.
        # Falls back to rule-based scoring if model files are not yet loaded.
        if cf_scorer._loaded:
            history_score = cf_scorer.score(article_id, purchase_hints, articles_df)
        else:
            history_score = 0.0
            if purchase_hints:
                item_colour = str(metadata.get('colour_group_name', '')).strip()
                item_type   = str(metadata.get('product_type_name', '')).strip()
                item_price  = float(metadata.get('price') or 0)

                top_colours = purchase_hints.get('top_colours') or []
                if item_colour in top_colours:
                    history_score += 0.12 * (1 - top_colours.index(item_colour) / max(len(top_colours), 1))

                if item_type in (purchase_hints.get('top_product_types') or []):
                    history_score += 0.08

                price_range = purchase_hints.get('preferred_price_range')
                if price_range and len(price_range) == 2 and price_range[0] <= item_price <= price_range[1]:
                    history_score += 0.08

        # Psychology KB score (NOVELTY 5)
        kb_score = kb_retriever.score(
            item_colour=metadata.get("colour_group_name", ""),
            item_type=metadata.get("product_type_name", ""),
            item_appearance=metadata.get("graphical_appearance_name", ""),
            occasion=soft_constraints.get("occasion"),
            style_word=inferred_style,
            index_group_name=metadata.get("index_group_name", ""),
        )

        filtered_results.append({
            "article_id":  article_id,
            "metadata":    metadata,
            "faiss_score": faiss_score,
            "final_score": faiss_score + boost_score - penalty_score + history_score + kb_score,
        })

    filtered_results.sort(key=lambda x: x["final_score"], reverse=True)

    # Fallback: if all candidates were filtered out, use top raw FAISS results
    if not filtered_results:
        print("  [catalog_search] Hard filters eliminated all candidates. Falling back to top FAISS results.")
        for article_id, faiss_score in candidates[:10]:
            if article_id not in exclude_ids:
                meta = _fetch_article(article_id)
                if meta:
                    filtered_results.append({
                        "article_id":  article_id,
                        "metadata":    meta,
                        "faiss_score": faiss_score,
                        "final_score": faiss_score,
                    })

    # ---------------------------------------------------------------
    # PHASE 3 — Cross-Encoder Neural Re-ranking (MiniLM-BERT)
    # ---------------------------------------------------------------
    print(f"  [catalog_search] Neural cross-encoder scoring top-{min(len(filtered_results), 20)} candidates...")
    neural_reranked = cross_encoder_reranker.rerank(
        query=base_search_text,
        candidates=filtered_results,
        top_k=20,
    )

    # ---------------------------------------------------------------
    # PHASE 4 — LLM Semantic Re-ranking (NOVELTY 2)
    # ---------------------------------------------------------------
    # NOVELTY 5: inject KB psychology context into LLM reranking
    kb_rerank_context = kb_retriever.get_context(
        occasion=soft_constraints.get("occasion"),
        style_word=inferred_style,
    )
    enriched_soft_constraints = dict(soft_constraints)
    if kb_rerank_context:
        enriched_soft_constraints["kb_psychology"] = kb_rerank_context

    print("  [catalog_search] LLM semantic re-ranking top-8 from neural stage...")
    reranked_results = llm_generator.rerank_candidates(
        user_message=user_message,
        candidates=neural_reranked,
        soft_constraints=enriched_soft_constraints,
        purchase_hints=purchase_hints,
    )

    # ---------------------------------------------------------------
    # PHASE 5 — Thompson Sampling Diversity Bandit + MMR (NOVELTY 3)
    # ---------------------------------------------------------------
    # Derive implicit feedback signals from session context:
    #   exclude_ids      → rejected items → more diversity → β increases
    #   items_in_context → kept items     → more relevance → α increases
    items_ctx      = retrieval_input.get("items_in_context") or {}
    retained_count = sum(1 for k in ("item_a", "item_b") if items_ctx.get(k))

    adaptive_lambda = diversity_bandit.sample_lambda(
        exclude_count=len(exclude_ids),
        retained_count=retained_count,
    )
    print(f"  [catalog_search] MMR with Thompson Sampling λ={adaptive_lambda:.3f}...")

    top_results = faiss_db.mmr_select(
        candidates=reranked_results,
        query_vector=query_vector,
        top_k=num_items,
        lambda_param=adaptive_lambda,
    ) or reranked_results[:num_items]

    # Colour harmony diagnostic for the selected pair (NOVELTY 5 — Improvement 4)
    if len(top_results) >= 2:
        c1 = top_results[0]["metadata"].get("colour_group_name", "")
        c2 = top_results[1]["metadata"].get("colour_group_name", "")
        h  = kb_retriever.harmony_score(c1, c2)
        label = "complementary" if h > 0.10 else ("analogous" if h > 0 else ("clashing" if h < 0 else "neutral"))
        print(f"  [KB] Colour harmony: {c1} × {c2} = {h:+.2f} ({label})")

    # ---------------------------------------------------------------
    # PHASE 6 — Verified explanation generation
    # ---------------------------------------------------------------
    response_items = []
    for result in top_results:
        aid  = result["article_id"]
        meta = result["metadata"]

        # NOVELTY 5: KB-grounded explanation fact injected into the prompt
        kb_explanation_fact = kb_retriever.get_explanation(
            item_colour=meta.get("colour_group_name", ""),
            item_type=meta.get("product_type_name", ""),
            occasion=soft_constraints.get("occasion"),
            style_word=inferred_style,
        )
        if kb_explanation_fact:
            print(f"  [KB] Explanation fact: {kb_explanation_fact[:80]}...")

        explanation, verification_trail = generator_loop.generate_faithful_explanation(
            article_id=aid,
            kb_fact=kb_explanation_fact,
        )

        item_response = _format_article_for_response(meta)
        item_response["explanation"]         = explanation
        item_response["score"]               = result["final_score"]
        item_response["verification_trail"]  = verification_trail
        response_items.append(item_response)

    # Natural language summary — handles any number of items
    n = len(response_items)
    if n == 0:
        summary = "I couldn't find any items matching all your criteria."
    elif n == 1:
        summary = (
            f"I found a great match: the {response_items[0]['prod_name']} "
            f"in {response_items[0]['colour_group_name']}."
        )
    elif n == 2:
        summary = (
            f"Based on your search, I found two great options: "
            f"the {response_items[0]['prod_name']} in {response_items[0]['colour_group_name']} "
            f"and the {response_items[1]['prod_name']} in {response_items[1]['colour_group_name']}."
        )
    else:
        item_list = ", ".join(
            f"the {item['prod_name']} in {item['colour_group_name']}"
            for item in response_items[:-1]
        )
        last = f"the {response_items[-1]['prod_name']} in {response_items[-1]['colour_group_name']}"
        summary = f"Based on your search, I found {n} great options: {item_list}, and {last}."

    _print_accuracy(response_items, filters)

    return {
        "action":        "catalog_search",
        "success":       len(response_items) > 0,
        "response_text": summary,
        "items":         response_items,
        "error":         None,
    }


# =====================================================================
# HANDLER 2: item_attribute_lookup
# Triggered by: ATTRIBUTE_QUESTION  |  Strategy: PARTIAL
# =====================================================================

def handle_attribute_lookup(retrieval_input: dict) -> dict:
    """Fetches an item and answers a specific attribute question about it."""
    payload          = retrieval_input.get("payload", {})
    user_message     = retrieval_input.get("user_message", "")
    article_id       = payload.get("article_id", "")
    attribute_topic  = payload.get("attribute_topic", "general_details")

    print(f"  [attribute_lookup] Article: {article_id}, Topic: {attribute_topic}")

    metadata = _fetch_article(article_id)
    if not metadata:
        return {"action": "item_attribute_lookup", "success": False,
                "response_text": f"I couldn't find item {article_id} in the catalog.",
                "items": [], "error": f"Article {article_id} not found"}

    item_info   = _format_article_for_response(metadata)
    detail_desc = metadata.get("detail_desc", "No detailed description available.")

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Here are the item details:\n"
        f"- Product: {item_info['prod_name']}\n"
        f"- Type: {item_info['product_type_name']}\n"
        f"- Colour: {item_info['colour_group_name']}\n"
        f"- Department: {item_info['department_name']}\n"
        f"- Appearance: {item_info['graphical_appearance_name']}\n"
        f"- Description: {detail_desc}\n\n"
        f"The customer is specifically asking about: {attribute_topic.replace('_', ' ')}.\n"
        f"Answer their question in 1-3 sentences using ONLY the information above. "
        f"If the information isn't available in the details, say so honestly."
    )

    print(f"  [attribute_lookup] Running self-reflection + VLM verification...")
    response_text = _vlm_verified_response(prompt, metadata, article_id)

    if not response_text:
        response_text = (
            f"The {item_info['prod_name']} is a {item_info['colour_group_name']} "
            f"{item_info['product_type_name']} from the {item_info['department_name']} department."
        )
        if attribute_topic == "material_and_care" and detail_desc:
            response_text = f"Here are the details: {detail_desc}"

    return {
        "action":        "item_attribute_lookup",
        "success":       True,
        "response_text": response_text,
        "items":         [item_info],
        "error":         None,
    }


# =====================================================================
# HANDLER 3: item_compare
# Triggered by: COMPARISON  |  Strategy: PARTIAL
# =====================================================================

def handle_item_compare(retrieval_input: dict) -> dict:
    """
    Compares two items on a specified dimension. Runs VLM verification in
    two phases — once anchored on item_a, once on item_b — to catch
    hallucinations about either item independently.
    """
    payload              = retrieval_input.get("payload", {})
    user_message         = retrieval_input.get("user_message", "")
    article_id_a         = payload.get("article_id_a") or ""
    article_id_b         = payload.get("article_id_b") or ""
    comparison_dimension = payload.get("comparison_dimension", "overall")
    preference_weights   = payload.get("preference_weights", {})

    if not article_id_a or not article_id_b:
        return {
            "action":        "item_compare",
            "success":       False,
            "response_text": ("I'd be happy to compare items for you! "
                              "Could you let me know which two items you'd like me to compare?"),
            "items":         [],
            "error":         "Missing article_id_a or article_id_b in payload",
        }

    print(f"  [item_compare] Comparing {article_id_a} vs {article_id_b} on '{comparison_dimension}'")

    meta_a = _fetch_article(article_id_a)
    meta_b = _fetch_article(article_id_b)

    if not meta_a or not meta_b:
        missing = article_id_a if not meta_a else article_id_b
        return {"action": "item_compare", "success": False,
                "response_text": f"I couldn't find item {missing} in the catalog.",
                "items": [], "error": f"Article {missing} not found"}

    item_a = _format_article_for_response(meta_a)
    item_b = _format_article_for_response(meta_b)

    pref_str = ""
    if preference_weights:
        pref_parts = [f"{k.replace('_', ' ')}: {v:.0%} importance" for k, v in preference_weights.items()]
        pref_str = f"\n\nThe customer's preferences: {', '.join(pref_parts)}."

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Compare these two items on the dimension of '{comparison_dimension}':\n\n"
        f"ITEM A — {item_a['prod_name']}:\n"
        f"  - Type: {item_a['product_type_name']}\n"
        f"  - Colour: {item_a['colour_group_name']}\n"
        f"  - Department: {item_a['department_name']}\n"
        f"  - Description: {meta_a.get('detail_desc', 'N/A')}\n\n"
        f"ITEM B — {item_b['prod_name']}:\n"
        f"  - Type: {item_b['product_type_name']}\n"
        f"  - Colour: {item_b['colour_group_name']}\n"
        f"  - Department: {item_b['department_name']}\n"
        f"  - Description: {meta_b.get('detail_desc', 'N/A')}\n"
        f"{pref_str}\n\n"
        f"Give a clear, concise comparison (2-4 sentences). "
        f"State which item is better for the customer and why, based on the comparison dimension."
    )

    # Phase 1: self-reflection + VLM verification anchored on item_a
    print(f"  [item_compare] Phase 1 — self-reflection + VLM verification (anchor: item_a)...")
    response_text = _vlm_verified_response(prompt, meta_a, article_id_a)

    # Phase 2: additional VLM pass anchored on item_b
    if response_text:
        image_path_b = data_loader.get_image(article_id_b)
        if image_path_b and image_path_b.exists():
            print(f"  [item_compare] Phase 2 — additional VLM verification (anchor: item_b)...")
            is_valid_b, reason_b = blip_verifier.verify(str(image_path_b), response_text)
            if is_valid_b:
                print(f"  [item_compare] Item B VLM PASS: {reason_b}")
            else:
                print(f"  [item_compare] Item B VLM FAIL: {reason_b} — regenerating with item_b feedback")
                corrected = _call_llm(
                    prompt + f"\n\n[Visual check on Item B failed: {reason_b}. "
                             f"Ensure your description of Item B is visually accurate.]"
                )
                if corrected:
                    response_text = corrected
        else:
            print(f"  [item_compare] Phase 2 — no image for item_b ({article_id_b}), skipping.")

    if not response_text:
        response_text = (
            f"Comparing the {item_a['prod_name']} ({item_a['colour_group_name']}) "
            f"and {item_b['prod_name']} ({item_b['colour_group_name']}) "
            f"on {comparison_dimension}: both are great options from their respective departments."
        )

    return {
        "action":        "item_compare",
        "success":       True,
        "response_text": response_text,
        "items":         [item_a, item_b],
        "error":         None,
    }


# =====================================================================
# HANDLER 4: explanation_generate
# Triggered by: EXPLANATION_WHY  |  Strategy: PARTIAL
# =====================================================================

def handle_explanation_generate(retrieval_input: dict) -> dict:
    """
    Generates a justified explanation for why an item was recommended,
    grounded in the user's matched_prefs and consistent with prior_claims.
    """
    payload        = retrieval_input.get("payload", {})
    user_message   = retrieval_input.get("user_message", "")
    article_id     = payload.get("article_id", "")
    prior_claims   = payload.get("prior_claims", [])
    matched_prefs  = payload.get("matched_prefs", [])

    print(f"  [explanation_generate] Article: {article_id}")
    print(f"  [explanation_generate] Prior claims: {len(prior_claims)}, Matched prefs: {len(matched_prefs)}")

    metadata = _fetch_article(article_id)
    if not metadata:
        return {"action": "explanation_generate", "success": False,
                "response_text": f"I couldn't find item {article_id} to explain.",
                "items": [], "error": f"Article {article_id} not found"}

    item_info = _format_article_for_response(metadata)

    # Build prior claims context (do not contradict active claims)
    claims_str = ""
    active_claims = [c for c in prior_claims if c.get("status") == "active"]
    if active_claims:
        claims_parts = [f"- {c['claim_text']} (type: {c['claim_type']})" for c in active_claims]
        claims_str = (
            "\n\nIMPORTANT — You have already told the customer these facts (do NOT contradict them):\n"
            + "\n".join(claims_parts)
        )

    # Build matched preferences context
    prefs_str = ""
    if matched_prefs:
        prefs_parts = [
            f"- {p['attribute_name'].replace('_', ' ')}: {p['attribute_value']} (weight: {p['weight']:.0%})"
            for p in matched_prefs
        ]
        prefs_str = "\n\nThe customer's preferences that match this item:\n" + "\n".join(prefs_parts)

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Explain why we recommended the {item_info['prod_name']}:\n"
        f"- Type: {item_info['product_type_name']}\n"
        f"- Colour: {item_info['colour_group_name']}\n"
        f"- Department: {item_info['department_name']}\n"
        f"- Description: {metadata.get('detail_desc', 'N/A')}\n"
        f"{prefs_str}{claims_str}\n\n"
        f"Generate a warm, conversational 2-3 sentence explanation. "
        f"Base it on the matched preferences above. "
        f"Do NOT contradict any prior claims listed above."
    )

    print(f"  [explanation_generate] Running self-reflection + VLM verification...")
    response_text = _vlm_verified_response(prompt, metadata, article_id)

    if not response_text:
        if matched_prefs:
            reasons = [
                f"it matches your preference for {p['attribute_value']} {p['attribute_name'].replace('_', ' ')}"
                for p in matched_prefs[:3]
            ]
            response_text = f"We recommended the {item_info['prod_name']} because " + ", and ".join(reasons) + "."
        else:
            response_text = (
                f"The {item_info['prod_name']} is a great {item_info['colour_group_name']} "
                f"{item_info['product_type_name']} that fits your style."
            )

    return {
        "action":        "explanation_generate",
        "success":       True,
        "response_text": response_text,
        "items":         [item_info],
        "error":         None,
    }


# =====================================================================
# HANDLER 5: item_detail_lookup
# Triggered by: SELECTION_REFERENCE  |  Strategy: PARTIAL
# =====================================================================

def handle_item_detail_lookup(retrieval_input: dict) -> dict:
    """
    Fetches and presents all details for an item the user pointed at.
    m3 has already resolved a reference like 'the first one' to an article_id.
    """
    payload      = retrieval_input.get("payload", {})
    user_message = retrieval_input.get("user_message", "")
    article_id   = payload.get("article_id", "")

    print(f"  [item_detail_lookup] Article: {article_id}")

    metadata = _fetch_article(article_id)
    if not metadata:
        return {"action": "item_detail_lookup", "success": False,
                "response_text": f"I couldn't find item {article_id} in the catalog.",
                "items": [], "error": f"Article {article_id} not found"}

    item_info   = _format_article_for_response(metadata)
    detail_desc = metadata.get("detail_desc", "")

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Present the full details of this item in a friendly, conversational way (3-4 sentences):\n"
        f"- Name: {item_info['prod_name']}\n"
        f"- Type: {item_info['product_type_name']}\n"
        f"- Colour: {item_info['colour_group_name']}\n"
        f"- Department: {item_info['department_name']}\n"
        f"- Category: {item_info['product_group_name']}\n"
        f"- Appearance: {item_info['graphical_appearance_name']}\n"
        f"- Description: {detail_desc}\n\n"
        f"Be informative and enthusiastic. Highlight the key selling points."
    )

    print(f"  [item_detail_lookup] Running self-reflection + VLM verification...")
    response_text = _vlm_verified_response(prompt, metadata, article_id)

    if not response_text:
        response_text = (
            f"Here are the details for the {item_info['prod_name']}: "
            f"It's a {item_info['colour_group_name']} {item_info['product_type_name']} "
            f"from the {item_info['department_name']} department. "
            f"{detail_desc}"
        )

    return {
        "action":        "item_detail_lookup",
        "success":       True,
        "response_text": response_text,
        "items":         [item_info],
        "error":         None,
    }


# =====================================================================
# HANDLER 6: No retrieval
# Triggered by: FEEDBACK, CHITCHAT  |  Strategy: NONE
# =====================================================================

def handle_no_retrieval(memory_context: dict) -> dict:
    """Handles turns where no retrieval is needed (FEEDBACK or CHITCHAT)."""
    feedback = memory_context.get("feedback")

    if feedback:
        sentiment_score  = feedback.get("sentiment_score", 0.0)
        is_positive      = feedback.get("is_positive", False)
        feedback_type    = feedback.get("feedback_type", "neutral")
        item_name        = feedback.get("item_reacted_to", {}).get("prod_name", "that item")

        print(f"  [no_retrieval] FEEDBACK: sentiment={sentiment_score:.1f}, type={feedback_type}")

        if is_positive:
            prompt   = (
                f"You are a friendly fashion assistant. The customer just expressed positive feedback "
                f"(sentiment: {sentiment_score:.1f}/1.0) about the {item_name}. "
                f"Write a brief, warm response (1-2 sentences) congratulating their choice "
                f"and asking if they'd like to see similar items or proceed to purchase."
            )
            fallback = (
                f"Great choice! The {item_name} is an excellent pick. "
                f"Would you like to see similar items, or shall I help with anything else?"
            )
        else:
            prompt   = (
                f"You are a friendly fashion assistant. The customer just expressed negative feedback "
                f"(sentiment: {sentiment_score:.1f}/1.0) about the {item_name}. "
                f"Write a brief, empathetic response (1-2 sentences) acknowledging their reaction "
                f"and offering to search for something different."
            )
            fallback = (
                f"I understand the {item_name} wasn't quite right. "
                f"Let me know what you'd prefer and I'll find something better for you!"
            )

        return {
            "action":        None,
            "success":       True,
            "response_text": _call_llm(prompt) or fallback,
            "items":         [],
            "error":         None,
        }

    # CHITCHAT path
    print("  [no_retrieval] CHITCHAT: Generating conversational response.")
    has_history = bool(memory_context.get("dialogue_state", {}).get("hard_constraints"))

    if has_history:
        prompt   = (
            "You are a friendly fashion assistant. The customer is making small talk "
            "during a shopping conversation. Respond briefly and warmly (1-2 sentences), "
            "then gently steer back to helping them find fashion items."
        )
        fallback = "Of course! I'm here to help. What kind of fashion items are you looking for today?"
    else:
        prompt   = (
            "You are a friendly fashion assistant. A new customer just greeted you. "
            "Welcome them warmly (1-2 sentences) and invite them to describe what "
            "kind of clothing or style they're looking for."
        )
        fallback = (
            "Welcome! I'm your fashion assistant. "
            "Tell me what you're looking for — a specific item, colour, or style — and I'll find the perfect match!"
        )

    return {
        "action":        None,
        "success":       True,
        "response_text": _call_llm(prompt) or fallback,
        "items":         [],
        "error":         None,
    }
