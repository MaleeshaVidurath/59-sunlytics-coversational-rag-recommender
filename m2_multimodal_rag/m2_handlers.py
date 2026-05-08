"""
M2 Action Handlers — Implements each action type from the retrieval_input spec.

Each handler receives the full retrieval_input dict and returns a standardized
response dict with: action, success, response_text, items, error.

These handlers reuse existing M2 components:
    - CLIP encoder + FAISS index for catalog_search vector retrieval
    - data_loader for article CSV lookups
    - llm_generator (Ollama llama3.1) for natural language responses
    - regeneration_loop for verified explanations
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from shared.data_loader import data_loader
from m2_multimodal_rag.llm_generator import llm_generator
from m2_multimodal_rag.clip_embeddings import clip_encoder
from m2_multimodal_rag.faiss_index import faiss_db
from m2_multimodal_rag.regeneration_loop import generator_loop


# =====================================================================
# Helper: Fetch article metadata by article_id from the articles CSV
# =====================================================================
def _fetch_article(article_id: str) -> dict | None:
    """
    Fetches a single article's metadata from the articles CSV.
    Returns a dict of the row, or None if not found.
    """
    articles_df = data_loader.load_articles()
    # article_id in CSV is integer, but retrieval_input sends it as string
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
        "article_id": str(metadata.get("article_id", "")).zfill(10),
        "prod_name": metadata.get("prod_name", "Unknown"),
        "product_type_name": metadata.get("product_type_name", "Unknown"),
        "product_group_name": metadata.get("product_group_name", "Unknown"),
        "colour_group_name": metadata.get("colour_group_name", "Unknown"),
        "department_name": metadata.get("department_name", "Unknown"),
        "index_group_name": metadata.get("index_group_name", "Unknown"),
        "detail_desc": metadata.get("detail_desc", ""),
        "graphical_appearance_name": metadata.get("graphical_appearance_name", "Unknown"),
    }


def _call_llm(prompt: str) -> str | None:
    """
    Utility to call the cloud LLM (Groq) for generating natural language responses.
    Reuses the llm_generator's configuration for consistency.
    """
    return llm_generator._call_llm(prompt, max_tokens=250)


def _rank_faiss_candidates(
    candidates: list,
    filters: dict | None = None,
    boosts: list | None = None,
    penalties: dict | None = None,
    exclude_ids: list | None = None,
    top_n: int = 2,
) -> list:
    """
    Shared post-processing for FAISS candidates:
    hard filters, preference boosts, penalties, excluded IDs, then top-N rank.
    """
    filters = filters or {}
    boosts = boosts or []
    penalties = penalties or {}
    exclude_ids = [str(item_id) for item_id in (exclude_ids or [])]

    articles_df = data_loader.load_articles()
    filtered_results = []

    for article_id, faiss_score in candidates:
        article_id = str(article_id).zfill(10)

        if article_id in exclude_ids or article_id.lstrip("0") in exclude_ids:
            continue

        try:
            article_row = articles_df[articles_df["article_id"] == int(article_id)]
        except (ValueError, TypeError):
            continue

        if article_row.empty:
            continue

        metadata = article_row.iloc[0].to_dict()

        passes_filters = True
        for filter_key, filter_value in filters.items():
            if filter_key == "price_max":
                article_price = metadata.get("price", float("inf"))
                if article_price > filter_value:
                    passes_filters = False
                    break
            elif filter_key == "price_min":
                article_price = metadata.get("price", 0)
                if article_price < filter_value:
                    passes_filters = False
                    break
            else:
                article_val = str(metadata.get(filter_key, "")).strip().lower()
                filter_val = str(filter_value).strip().lower()
                if article_val != filter_val:
                    passes_filters = False
                    break

        if not passes_filters:
            continue

        penalty_score = 0.0
        for penalty_key, penalty_values in penalties.items():
            article_val = str(metadata.get(penalty_key, "")).strip().lower()
            for penalty_value in penalty_values:
                if article_val == str(penalty_value).strip().lower():
                    penalty_score += 0.3

        boost_score = 0.0
        for boost in boosts:
            attr = boost.get("attribute", "")
            value = str(boost.get("value", "")).strip().lower()
            weight = boost.get("weight", 0.0)
            article_val = str(metadata.get(attr, "")).strip().lower()
            if article_val == value:
                boost_score += weight

        final_score = faiss_score + boost_score - penalty_score

        filtered_results.append({
            "article_id": article_id,
            "metadata": metadata,
            "faiss_score": faiss_score,
            "final_score": final_score,
        })

    filtered_results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = filtered_results[:top_n]

    if top_results:
        return top_results

    fallback_results = []
    for article_id, faiss_score in candidates[:top_n]:
        article_id = str(article_id).zfill(10)
        if article_id in exclude_ids or article_id.lstrip("0") in exclude_ids:
            continue

        metadata = _fetch_article(article_id)
        if metadata:
            fallback_results.append({
                "article_id": article_id,
                "metadata": metadata,
                "faiss_score": faiss_score,
                "final_score": faiss_score,
            })

    return fallback_results


def _mmr_rerank(candidates: list, top_n: int = 2, lambda_param: float = 0.7) -> list:
    """
    Maximum Marginal Relevance re-ranking.

    Selects top_n items that balance:
      - Relevance  : final_score (FAISS cosine sim + boosts - penalties)
      - Diversity  : penalises items whose CLIP vectors are too similar to
                     already-selected items

    lambda_param: 0 = pure diversity, 1 = pure relevance (default 0.7).

    Reference: Carbonell & Goldstein (1998) — MMR for query-based summarisation.
    """
    if len(candidates) <= top_n:
        return candidates

    max_score = max(c['final_score'] for c in candidates) or 1.0

    # Pre-fetch stored CLIP vectors from FAISS for inter-item similarity
    vecs = {}
    for c in candidates:
        v = faiss_db.get_vector(c['article_id'])
        if v is not None:
            vecs[c['article_id']] = v.flatten()

    selected, remaining = [], list(candidates)

    while len(selected) < top_n and remaining:
        best, best_mmr = None, float('-inf')

        for cand in remaining:
            relevance = cand['final_score'] / max_score

            if not selected:
                diversity_penalty = 0.0
            else:
                c_vec = vecs.get(cand['article_id'])
                if c_vec is None:
                    diversity_penalty = 0.0
                else:
                    sims = [
                        float(np.dot(c_vec, vecs[s['article_id']]))
                        for s in selected
                        if s['article_id'] in vecs
                    ]
                    diversity_penalty = max(sims) if sims else 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best = cand

        if best:
            selected.append(best)
            remaining.remove(best)

    print(f"  [MMR] Selected {len(selected)} diverse items from {len(candidates)} candidates (λ={lambda_param})")
    return selected


def _normalize_query_vector(query_vector: np.ndarray) -> np.ndarray:
    """Normalizes a single-vector batch for FAISS inner-product search."""
    norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (query_vector / norm).astype("float32")


# =====================================================================
# HANDLER 1: catalog_search
# Triggered by: INITIAL_REQUEST, REFINEMENT
# Strategy: FULL
# =====================================================================
def handle_catalog_search(retrieval_input: dict) -> dict:
    """
    Searches the product catalog using a hybrid approach:
    1. Use CLIP/FAISS vector search with the user's message (larger top_k)
    2. Post-filter results using structured 'filters' (hard constraints)
    3. Re-rank using 'preference_boosts' (soft weights)
    4. Apply 'penalties' (demote disliked attributes)
    5. Exclude 'exclude_ids' (already rejected items)
    6. Return top 2 items with verified explanations
    """
    payload = retrieval_input.get("payload", {})
    user_message = retrieval_input.get("user_message", "")
    exclude_ids = retrieval_input.get("exclude_ids", [])
    filters = payload.get("filters", {})
    boosts = payload.get("preference_boosts", [])
    penalties = payload.get("penalties", {})

    print(f"  [catalog_search] Filters: {filters}")
    print(f"  [catalog_search] Boosts: {boosts}")
    print(f"  [catalog_search] Penalties: {penalties}")
    print(f"  [catalog_search] Exclude IDs: {exclude_ids}")

    # ------------------------------------------------------------------
    # PHASE 1: CLIP/FAISS Vector Search (broader net)
    # ------------------------------------------------------------------
    # Build a search string from the user message + filter values for CLIP
    filter_terms = " ".join(str(v) for v in filters.values() if not isinstance(v, (int, float)))
    search_text = f"{user_message} {filter_terms}".strip()
    
    if not search_text:
        search_text = " ".join(str(v) for v in filters.values())

    print(f"  [catalog_search] CLIP search text: '{search_text}'")

    query_vector = clip_encoder.encode_text(search_text)
    if query_vector is None:
        return {
            "action": "catalog_search",
            "success": False,
            "response_text": "I couldn't process your search request.",
            "items": [],
            "error": "CLIP encoding failed",
        }

    # Retrieve a larger candidate pool for post-filtering
    candidates = faiss_db.search(query_vector, top_k=50)

    if not candidates:
        return {
            "action": "catalog_search",
            "success": False,
            "response_text": "I couldn't find any items matching your search.",
            "items": [],
            "error": "No FAISS results",
        }

    # ------------------------------------------------------------------
    # PHASE 2: Post-filter using structured filters (hard constraints)
    # ------------------------------------------------------------------
    articles_df = data_loader.load_articles()
    filtered_results = []

    for article_id, faiss_score in candidates:
        # Skip excluded items
        if article_id in exclude_ids or article_id.lstrip('0') in exclude_ids:
            continue

        # Fetch article metadata
        try:
            article_row = articles_df[articles_df['article_id'] == int(article_id)]
        except (ValueError, TypeError):
            continue

        if article_row.empty:
            continue

        metadata = article_row.iloc[0].to_dict()

        # Apply hard filter constraints
        passes_filters = True
        for filter_key, filter_value in filters.items():
            if filter_key == "price_max":
                article_price = metadata.get("price", float('inf'))
                if article_price > filter_value:
                    passes_filters = False
                    break
            elif filter_key == "price_min":
                article_price = metadata.get("price", 0)
                if article_price < filter_value:
                    passes_filters = False
                    break
            else:
                article_val = str(metadata.get(filter_key, "")).strip().lower()
                filter_val = str(filter_value).strip().lower()
                if article_val != filter_val:
                    passes_filters = False
                    break

        if not passes_filters:
            continue

        # Apply penalty check (demote but don't hard-exclude)
        penalty_score = 0.0
        for penalty_key, penalty_values in penalties.items():
            article_val = str(metadata.get(penalty_key, "")).strip().lower()
            for pv in penalty_values:
                if article_val == str(pv).strip().lower():
                    penalty_score += 0.3  # Significant demotion

        # Calculate boost score
        boost_score = 0.0
        for boost in boosts:
            attr = boost.get("attribute", "")
            value = str(boost.get("value", "")).strip().lower()
            weight = boost.get("weight", 0.0)
            article_val = str(metadata.get(attr, "")).strip().lower()
            if article_val == value:
                boost_score += weight

        # Combined score: FAISS similarity + boost - penalty
        final_score = faiss_score + boost_score - penalty_score

        filtered_results.append({
            "article_id": article_id,
            "metadata": metadata,
            "faiss_score": faiss_score,
            "final_score": final_score,
        })

    # Sort by final_score descending
    filtered_results.sort(key=lambda x: x["final_score"], reverse=True)

    # MMR re-ranking: select top 2 that balance relevance + diversity
    top_results = _mmr_rerank(filtered_results, top_n=2)

    if not top_results:
        # Fallback: if hard filters eliminated everything, return top FAISS results
        print("  [catalog_search] Hard filters eliminated all candidates. Falling back to top FAISS results.")
        fallback_results = []
        for article_id, faiss_score in candidates[:2]:
            if article_id not in exclude_ids:
                meta = _fetch_article(article_id)
                if meta:
                    fallback_results.append({
                        "article_id": article_id,
                        "metadata": meta,
                        "faiss_score": faiss_score,
                        "final_score": faiss_score,
                    })
        top_results = _mmr_rerank(fallback_results, top_n=2)

    # ------------------------------------------------------------------
    # PHASE 3: Generate verified explanations for the top items
    # ------------------------------------------------------------------
    response_items = []
    for result in top_results:
        aid = result["article_id"]
        meta = result["metadata"]

        # Use the regeneration loop for verified explanation
        explanation = generator_loop.generate_faithful_explanation(article_id=aid)
        
        item_response = _format_article_for_response(meta)
        item_response["explanation"] = explanation
        item_response["score"] = result["final_score"]
        response_items.append(item_response)

    # Build natural language summary
    if len(response_items) == 2:
        summary = (
            f"Based on your search, I found two great options: "
            f"the {response_items[0]['prod_name']} in {response_items[0]['colour_group_name']} "
            f"and the {response_items[1]['prod_name']} in {response_items[1]['colour_group_name']}."
        )
    elif len(response_items) == 1:
        summary = (
            f"I found a great match: the {response_items[0]['prod_name']} "
            f"in {response_items[0]['colour_group_name']}."
        )
    else:
        summary = "I couldn't find any items matching all your criteria."

    return {
        "action": "catalog_search",
        "success": len(response_items) > 0,
        "response_text": summary,
        "items": response_items,
        "error": None,
    }


# =====================================================================
# HANDLER 1B: image_catalog_search
# Triggered by: Frontend image upload directly to M2
# Strategy: FULL, M2-only
# =====================================================================
def handle_image_catalog_search(
    image_path: str,
    user_message: str = "",
    filters: dict | None = None,
    boosts: list | None = None,
    penalties: dict | None = None,
    exclude_ids: list | None = None,
) -> dict:
    """
    Searches the catalog from an uploaded image without requiring M1/M3 changes.
    Uses CLIP image retrieval as the primary signal and optional BLIP/VLM text
    understanding to handle noisy images or user hints like "find this jacket".
    """
    filters = filters or {}
    boosts = boosts or []
    penalties = penalties or {}
    exclude_ids = exclude_ids or []

    print(f"  [image_catalog_search] Image path: {image_path}")
    print(f"  [image_catalog_search] User hint: '{user_message}'")
    print(f"  [image_catalog_search] Filters: {filters}")
    print(f"  [image_catalog_search] Boosts: {boosts}")
    print(f"  [image_catalog_search] Penalties: {penalties}")
    print(f"  [image_catalog_search] Exclude IDs: {exclude_ids}")

    image_vector = clip_encoder.encode_image(image_path)
    if image_vector is None:
        return {
            "action": "image_catalog_search",
            "success": False,
            "response_text": "I couldn't process the uploaded image.",
            "items": [],
            "error": "CLIP image encoding failed",
        }

    visual_query = ""
    try:
        from m2_multimodal_rag.query_understanding import vlm_query_processor

        visual_query = vlm_query_processor.extract_search_query(
            text_query=user_message or None,
            image_path=image_path,
        )
        if visual_query == "IRRELEVANT_QUERY":
            visual_query = ""
        print(f"  [image_catalog_search] VLM visual query: '{visual_query}'")
    except Exception as e:
        print(f"  [image_catalog_search] VLM noise handling skipped: {e}")

    # Image vector is the main retrieval signal. Text is blended in only as a
    # lightweight hint so noisy backgrounds do not dominate the search.
    filter_terms = " ".join(str(v) for v in filters.values() if not isinstance(v, (int, float)))
    text_parts = []
    if visual_query:
        text_parts.append(visual_query)
    elif user_message:
        text_parts.append(user_message)
    if filter_terms:
        text_parts.append(filter_terms)

    query_vector = image_vector
    text_search = " ".join(text_parts).strip()
    if text_search:
        text_vector = clip_encoder.encode_text(text_search)
        if text_vector is not None:
            query_vector = _normalize_query_vector((0.75 * image_vector) + (0.25 * text_vector))
        print(f"  [image_catalog_search] Blended text hint: '{text_search}'")

    candidates = faiss_db.search(query_vector, top_k=50)
    if not candidates:
        return {
            "action": "image_catalog_search",
            "success": False,
            "response_text": "I couldn't find visually similar catalog items.",
            "items": [],
            "error": "No FAISS results",
        }

    # Get a larger pool then apply MMR to ensure diverse final selection
    mmr_pool = _rank_faiss_candidates(
        candidates=candidates,
        filters=filters,
        boosts=boosts,
        penalties=penalties,
        exclude_ids=exclude_ids,
        top_n=10,
    )
    top_results = _mmr_rerank(mmr_pool, top_n=2)

    response_items = []
    for result in top_results:
        aid = result["article_id"]
        meta = result["metadata"]

        explanation = generator_loop.generate_faithful_explanation(article_id=aid)

        item_response = _format_article_for_response(meta)
        item_response["explanation"] = explanation
        item_response["score"] = result["final_score"]
        item_response["retrieval_source"] = "image"
        response_items.append(item_response)

    if len(response_items) == 2:
        summary = (
            f"I found two visually similar options: "
            f"the {response_items[0]['prod_name']} in {response_items[0]['colour_group_name']} "
            f"and the {response_items[1]['prod_name']} in {response_items[1]['colour_group_name']}."
        )
    elif len(response_items) == 1:
        summary = (
            f"I found a visually similar match: the {response_items[0]['prod_name']} "
            f"in {response_items[0]['colour_group_name']}."
        )
    else:
        summary = "I couldn't find any catalog items similar to the uploaded image."

    return {
        "action": "image_catalog_search",
        "success": len(response_items) > 0,
        "response_text": summary,
        "items": response_items,
        "visual_query": visual_query,
        "error": None,
    }


# =====================================================================
# HANDLER 2: item_attribute_lookup
# Triggered by: ATTRIBUTE_QUESTION
# Strategy: PARTIAL
# =====================================================================
def handle_attribute_lookup(retrieval_input: dict) -> dict:
    """
    Fetches a specific item by article_id and answers a question about
    a specific attribute_topic (material_and_care, colour, sizing, etc.).
    """
    payload = retrieval_input.get("payload", {})
    user_message = retrieval_input.get("user_message", "")
    article_id = payload.get("article_id", "")
    attribute_topic = payload.get("attribute_topic", "general_details")

    print(f"  [attribute_lookup] Article: {article_id}, Topic: {attribute_topic}")

    metadata = _fetch_article(article_id)
    if not metadata:
        return {
            "action": "item_attribute_lookup",
            "success": False,
            "response_text": f"I couldn't find item {article_id} in the catalog.",
            "items": [],
            "error": f"Article {article_id} not found",
        }

    # Build a prompt for the LLM to answer the attribute question
    item_info = _format_article_for_response(metadata)
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

    response_text = _call_llm(prompt)
    if not response_text:
        # Fallback to template-based response
        response_text = (
            f"The {item_info['prod_name']} is a {item_info['colour_group_name']} "
            f"{item_info['product_type_name']} from the {item_info['department_name']} department."
        )
        if attribute_topic == "material_and_care" and detail_desc:
            response_text = f"Here are the details: {detail_desc}"

    return {
        "action": "item_attribute_lookup",
        "success": True,
        "response_text": response_text,
        "items": [item_info],
        "error": None,
    }


# =====================================================================
# HANDLER 3: item_compare
# Triggered by: COMPARISON
# Strategy: PARTIAL
# =====================================================================
def handle_item_compare(retrieval_input: dict) -> dict:
    """
    Compares two items currently in context on a specified dimension
    (price, quality, style_and_occasion, material, colour, fit, overall).
    Uses preference_weights to explain which item better matches the user.
    """
    payload = retrieval_input.get("payload", {})
    user_message = retrieval_input.get("user_message", "")
    article_id_a = payload.get("article_id_a", "")
    article_id_b = payload.get("article_id_b", "")
    comparison_dimension = payload.get("comparison_dimension", "overall")
    preference_weights = payload.get("preference_weights", {})

    print(f"  [item_compare] Comparing {article_id_a} vs {article_id_b} on '{comparison_dimension}'")

    meta_a = _fetch_article(article_id_a)
    meta_b = _fetch_article(article_id_b)

    if not meta_a or not meta_b:
        missing = article_id_a if not meta_a else article_id_b
        return {
            "action": "item_compare",
            "success": False,
            "response_text": f"I couldn't find item {missing} in the catalog.",
            "items": [],
            "error": f"Article {missing} not found",
        }

    item_a = _format_article_for_response(meta_a)
    item_b = _format_article_for_response(meta_b)

    # Build preference context string
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

    response_text = _call_llm(prompt)
    if not response_text:
        response_text = (
            f"Comparing the {item_a['prod_name']} ({item_a['colour_group_name']}) "
            f"and {item_b['prod_name']} ({item_b['colour_group_name']}) "
            f"on {comparison_dimension}: both are great options from their respective departments."
        )

    return {
        "action": "item_compare",
        "success": True,
        "response_text": response_text,
        "items": [item_a, item_b],
        "error": None,
    }


# =====================================================================
# HANDLER 4: explanation_generate
# Triggered by: EXPLANATION_WHY
# Strategy: PARTIAL
# =====================================================================
def handle_explanation_generate(retrieval_input: dict) -> dict:
    """
    Generates a justified explanation for why an item was recommended,
    grounded in the user's matched_prefs and consistent with prior_claims.
    """
    payload = retrieval_input.get("payload", {})
    user_message = retrieval_input.get("user_message", "")
    article_id = payload.get("article_id", "")
    prior_claims = payload.get("prior_claims", [])
    matched_prefs = payload.get("matched_prefs", [])

    print(f"  [explanation_generate] Article: {article_id}")
    print(f"  [explanation_generate] Prior claims: {len(prior_claims)}, Matched prefs: {len(matched_prefs)}")

    metadata = _fetch_article(article_id)
    if not metadata:
        return {
            "action": "explanation_generate",
            "success": False,
            "response_text": f"I couldn't find item {article_id} to explain.",
            "items": [],
            "error": f"Article {article_id} not found",
        }

    item_info = _format_article_for_response(metadata)

    # Build prior claims context
    claims_str = ""
    active_claims = [c for c in prior_claims if c.get("status") == "active"]
    if active_claims:
        claims_parts = [f"- {c['claim_text']} (type: {c['claim_type']})" for c in active_claims]
        claims_str = (
            f"\n\nIMPORTANT — You have already told the customer these facts (do NOT contradict them):\n"
            + "\n".join(claims_parts)
        )

    # Build matched preferences context
    prefs_str = ""
    if matched_prefs:
        prefs_parts = [
            f"- {p['attribute_name'].replace('_', ' ')}: {p['attribute_value']} "
            f"(weight: {p['weight']:.0%})"
            for p in matched_prefs
        ]
        prefs_str = (
            f"\n\nThe customer's preferences that match this item:\n"
            + "\n".join(prefs_parts)
        )

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Explain why we recommended the {item_info['prod_name']}:\n"
        f"- Type: {item_info['product_type_name']}\n"
        f"- Colour: {item_info['colour_group_name']}\n"
        f"- Department: {item_info['department_name']}\n"
        f"- Description: {metadata.get('detail_desc', 'N/A')}\n"
        f"{prefs_str}"
        f"{claims_str}\n\n"
        f"Generate a warm, conversational 2-3 sentence explanation. "
        f"Base it on the matched preferences above. "
        f"Do NOT contradict any prior claims listed above."
    )

    response_text = _call_llm(prompt)
    if not response_text:
        # Fallback: build explanation from matched_prefs
        if matched_prefs:
            reasons = [f"it matches your preference for {p['attribute_value']} {p['attribute_name'].replace('_', ' ')}"
                       for p in matched_prefs[:3]]
            response_text = (
                f"We recommended the {item_info['prod_name']} because "
                + ", and ".join(reasons) + "."
            )
        else:
            response_text = (
                f"The {item_info['prod_name']} is a great {item_info['colour_group_name']} "
                f"{item_info['product_type_name']} that fits your style."
            )

    return {
        "action": "explanation_generate",
        "success": True,
        "response_text": response_text,
        "items": [item_info],
        "error": None,
    }


# =====================================================================
# HANDLER 5: item_detail_lookup
# Triggered by: SELECTION_REFERENCE
# Strategy: PARTIAL
# =====================================================================
def handle_item_detail_lookup(retrieval_input: dict) -> dict:
    """
    Fetches and returns all details for a specific item the user pointed at.
    The m3 pipeline has already resolved 'the first one' / 'the blue one'
    into an article_id.
    """
    payload = retrieval_input.get("payload", {})
    user_message = retrieval_input.get("user_message", "")
    article_id = payload.get("article_id", "")

    print(f"  [item_detail_lookup] Article: {article_id}")

    metadata = _fetch_article(article_id)
    if not metadata:
        return {
            "action": "item_detail_lookup",
            "success": False,
            "response_text": f"I couldn't find item {article_id} in the catalog.",
            "items": [],
            "error": f"Article {article_id} not found",
        }

    item_info = _format_article_for_response(metadata)
    detail_desc = metadata.get("detail_desc", "")

    # Generate a natural description using the LLM
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

    response_text = _call_llm(prompt)
    if not response_text:
        response_text = (
            f"Here are the details for the {item_info['prod_name']}: "
            f"It's a {item_info['colour_group_name']} {item_info['product_type_name']} "
            f"from the {item_info['department_name']} department. "
            f"{detail_desc}"
        )

    return {
        "action": "item_detail_lookup",
        "success": True,
        "response_text": response_text,
        "items": [item_info],
        "error": None,
    }


# =====================================================================
# HANDLER 6: No retrieval (FEEDBACK / CHITCHAT)
# Triggered by: FEEDBACK, CHITCHAT
# Strategy: NO
# =====================================================================
def handle_no_retrieval(memory_context: dict) -> dict:
    """
    Handles turns where no retrieval is needed.
    - FEEDBACK: Responds based on sentiment in memory_context.feedback
    - CHITCHAT: Generates a conversational response
    """
    feedback = memory_context.get("feedback")

    # --- FEEDBACK PATH ---
    if feedback:
        sentiment_score = feedback.get("sentiment_score", 0.0)
        is_positive = feedback.get("is_positive", False)
        feedback_type = feedback.get("feedback_type", "neutral")
        item_reacted_to = feedback.get("item_reacted_to", {})
        item_name = item_reacted_to.get("prod_name", "that item")

        print(f"  [no_retrieval] FEEDBACK: sentiment={sentiment_score:.1f}, type={feedback_type}")

        if is_positive:
            prompt = (
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
            prompt = (
                f"You are a friendly fashion assistant. The customer just expressed negative feedback "
                f"(sentiment: {sentiment_score:.1f}/1.0) about the {item_name}. "
                f"Write a brief, empathetic response (1-2 sentences) acknowledging their reaction "
                f"and offering to search for something different."
            )
            fallback = (
                f"I understand the {item_name} wasn't quite right. "
                f"Let me know what you'd prefer and I'll find something better for you!"
            )

        response_text = _call_llm(prompt) or fallback

        return {
            "action": None,
            "success": True,
            "response_text": response_text,
            "items": [],
            "error": None,
        }

    # --- CHITCHAT PATH ---
    print("  [no_retrieval] CHITCHAT: Generating conversational response.")

    dialogue_state = memory_context.get("dialogue_state", {})
    has_history = bool(dialogue_state.get("hard_constraints"))

    if has_history:
        prompt = (
            "You are a friendly fashion assistant. The customer is making small talk "
            "during a shopping conversation. Respond briefly and warmly (1-2 sentences), "
            "then gently steer back to helping them find fashion items."
        )
        fallback = "Of course! I'm here to help. What kind of fashion items are you looking for today?"
    else:
        prompt = (
            "You are a friendly fashion assistant. A new customer just greeted you. "
            "Welcome them warmly (1-2 sentences) and invite them to describe what "
            "kind of clothing or style they're looking for."
        )
        fallback = (
            "Welcome! I'm your fashion assistant. "
            "Tell me what you're looking for — a specific item, colour, or style — and I'll find the perfect match!"
        )

    response_text = _call_llm(prompt) or fallback

    return {
        "action": None,
        "success": True,
        "response_text": response_text,
        "items": [],
        "error": None,
    }
