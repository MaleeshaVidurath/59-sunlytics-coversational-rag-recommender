"""
Dummy M3 -> M2 Request Tester
==============================
Sends requests to M2's /api/process endpoint in the exact format
that M3's EnrichmentLayer produces for all 5 action types:

  Cases 1-4 : catalog_search      (INITIAL_REQUEST / REFINEMENT)
  Case  5   : item_attribute_lookup (ATTRIBUTE_QUESTION)
  Case  6   : item_compare          (COMPARISON)
  Case  7   : explanation_generate  (EXPLANATION_WHY)
  Case  8   : item_detail_lookup    (SELECTION_REFERENCE)

M2 server must be running:
  uvicorn m2_multimodal_rag.backend.main:app --host 0.0.0.0 --port 8001

Usage:
  python m2_multimodal_rag/test_m3_request.py            # runs all 8 cases
  python m2_multimodal_rag/test_m3_request.py --case 5   # attribute lookup
  python m2_multimodal_rag/test_m3_request.py --case 6   # item compare
  python m2_multimodal_rag/test_m3_request.py --case 7   # explanation
  python m2_multimodal_rag/test_m3_request.py --case 8   # detail lookup
  python m2_multimodal_rag/test_m3_request.py --query "red dress for party"
"""

import sys
import json
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

M2_URL = "http://localhost:8001/api/process"


# =============================================================================
# Dummy M3 payloads — each simulates what M3 would send for a real user turn
# =============================================================================

DUMMY_CASES = [
    {
        "name": "INITIAL_REQUEST — black dress, no history",
        "description": "First-time user, asks for a black dress. No purchase history.",
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I want a black dress",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {
                        "colour_group_name":  "Black",
                        "product_type_name":  "Dress",
                    },
                    "soft_constraints":       {},
                    "preference_boosts":      [],
                    "penalties":              {},
                    "purchase_history_hints": {
                        "top_colours":           [],
                        "top_product_types":     [],
                        "inferred_gender":       None,
                        "budget_tier":           None,
                        "preferred_price_range": None,
                        "dominant_colour":       None,
                        "dominant_type":         None,
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {"colour_group_name": "Black", "product_type_name": "Dress"},
                    "soft_constraints": {},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "",
                },
                "long_term_preferences": [],
                "style_profile":         {},
                "preference_summary":    {},
                "existing_explanation":  None,
            },
        },
    },
    {
        "name": "INITIAL_REQUEST — blue jeans for men, with purchase history",
        "description": "User with purchase history asks for blue jeans. M3 supplies history hints.",
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I need blue jeans for men",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {
                        "colour_group_name": "Dark Blue",
                        "product_type_name": "Trousers",
                        "index_group_name":  "Menswear",
                    },
                    "soft_constraints":  {},
                    "preference_boosts": [
                        {"attribute": "colour_group_name", "value": "Dark Blue", "weight": 0.75},
                        {"attribute": "product_type_name", "value": "Trousers",  "weight": 0.60},
                    ],
                    "penalties": {},
                    "purchase_history_hints": {
                        "top_colours":           ["Dark Blue", "Black", "Grey"],
                        "top_product_types":     ["Trousers", "Shirt", "Jacket"],
                        "inferred_gender":       "Menswear",
                        "budget_tier":           "mid",
                        "preferred_price_range": [0.03, 0.12],
                        "dominant_colour":       "Dark Blue",
                        "dominant_type":         "Trousers",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {
                        "colour_group_name": "Dark Blue",
                        "product_type_name": "Trousers",
                        "index_group_name":  "Menswear",
                    },
                    "soft_constraints": {},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "Looking for blue jeans",
                },
                "long_term_preferences": [
                    {"attribute_name": "colour_group_name", "attribute_value": "Dark Blue", "weight": 0.75},
                    {"attribute_name": "product_type_name", "attribute_value": "Trousers",  "weight": 0.60},
                ],
                "style_profile":        {"dominant_style": "casual", "gender": "Menswear"},
                "preference_summary":   {},
                "existing_explanation": None,
            },
        },
    },
    {
        "name": "REFINEMENT — white top, with excluded items + soft constraints",
        "description": (
            "User previously saw two items (now excluded), "
            "refines to a white top for women with a casual occasion."
        ),
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "Show me something in white, a top for women",
                "items_in_context": {
                    "item_a": {
                        "article_id":        "0513701003",
                        "prod_name":         "V-NECK SS BASIC 3 PK",
                        "product_type_name": "T-shirt",
                        "colour_group_name": "Black",
                        "index_group_name":  "Menswear",
                        "garment_group_name": "Jersey Basic",
                    },
                    "item_b": {
                        "article_id":        "0817110002",
                        "prod_name":         "Peacock",
                        "product_type_name": "Blouse",
                        "colour_group_name": "Black",
                        "index_group_name":  "Ladieswear",
                        "garment_group_name": "Blouses",
                    },
                },
                "exclude_ids": ["0513701003", "0817110002"],
                "payload": {
                    "filters": {
                        "colour_group_name": "White",
                        "index_group_name":  "Ladieswear",
                    },
                    "soft_constraints": {
                        "occasion": "casual",
                    },
                    "preference_boosts": [
                        {"attribute": "colour_group_name", "value": "White",      "weight": 0.65},
                        {"attribute": "index_group_name",  "value": "Ladieswear", "weight": 0.50},
                    ],
                    "penalties": {
                        "colour_group_name": ["Black"],
                    },
                    "purchase_history_hints": {
                        "top_colours":           ["White", "Light Pink", "Beige"],
                        "top_product_types":     ["Top", "Blouse", "Dress"],
                        "inferred_gender":       "Ladieswear",
                        "budget_tier":           "low",
                        "preferred_price_range": [0.01, 0.06],
                        "dominant_colour":       "White",
                        "dominant_type":         "Top",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {
                        "colour_group_name": "White",
                        "index_group_name":  "Ladieswear",
                    },
                    "soft_constraints": {"occasion": "casual"},
                    "rejected_items":   ["0513701003", "0817110002"],
                    "accepted_items":   [],
                    "intent_summary":   "Looking for white casual top for women",
                },
                "previous_constraints": {"colour_group_name": "Black"},
                "new_changes":          {"colour_group_name": "White"},
                "long_term_preferences": [
                    {"attribute_name": "colour_group_name", "attribute_value": "White", "weight": 0.65},
                ],
                "style_profile":        {"dominant_style": "casual", "gender": "Ladieswear"},
                "preference_summary":   {},
                "existing_explanation": None,
            },
        },
    },
    {
        "name": "INITIAL_REQUEST — formal shirt for men, penalties on casual",
        "description": "User wants a formal men's shirt. Casual appearance penalised.",
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I need a formal shirt for work",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {
                        "product_type_name": "Shirt",
                        "index_group_name":  "Menswear",
                    },
                    "soft_constraints": {
                        "occasion": "work",
                        "style":    "formal",
                    },
                    "preference_boosts": [
                        {"attribute": "graphical_appearance_name", "value": "Solid", "weight": 0.70},
                        {"attribute": "colour_group_name",         "value": "White", "weight": 0.50},
                    ],
                    "penalties": {
                        "graphical_appearance_name": ["Stripe", "All over pattern"],
                    },
                    "purchase_history_hints": {
                        "top_colours":           ["White", "Dark Blue", "Black"],
                        "top_product_types":     ["Shirt", "Trousers", "Blazer"],
                        "inferred_gender":       "Menswear",
                        "budget_tier":           "mid",
                        "preferred_price_range": [0.04, 0.15],
                        "dominant_colour":       "White",
                        "dominant_type":         "Shirt",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {
                        "product_type_name": "Shirt",
                        "index_group_name":  "Menswear",
                    },
                    "soft_constraints": {"occasion": "work", "style": "formal"},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "Formal work shirt for men",
                },
                "long_term_preferences": [
                    {"attribute_name": "graphical_appearance_name", "attribute_value": "Solid", "weight": 0.70},
                ],
                "style_profile":        {"dominant_style": "formal", "gender": "Menswear"},
                "preference_summary":   {},
                "existing_explanation": None,
            },
        },
    },

    # -------------------------------------------------------------------------
    # Case 5: item_attribute_lookup  (ATTRIBUTE_QUESTION)
    # -------------------------------------------------------------------------
    {
        "name": "ATTRIBUTE_QUESTION — material query on V-neck T-shirt",
        "description": (
            "User asks about the material of item_a currently in context. "
            "M3 resolved attribute_topic to 'material_and_care'."
        ),
        "payload": {
            "retrieval_input": {
                "action":             "item_attribute_lookup",
                "retrieval_strategy": "PARTIAL",
                "user_message":       "what material is this top made of?",
                "items_in_context": {
                    "item_a": {
                        "article_id":        "0513701003",
                        "prod_name":         "V-NECK SS BASIC 3 PK",
                        "product_type_name": "T-shirt",
                        "colour_group_name": "Black",
                        "index_group_name":  "Menswear",
                    },
                    "item_b": None,
                },
                "exclude_ids": [],
                "payload": {
                    "article_id":      "0513701003",
                    "attribute_topic": "material_and_care",
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {},
                    "soft_constraints": {},
                    "rejected_items":   [],
                    "accepted_items":   ["0513701003"],
                    "intent_summary":   "Asking about material of top in context",
                },
                "long_term_preferences": [],
                "style_profile":         {},
                "preference_summary":    {},
                "existing_explanation":  None,
            },
        },
    },

    # -------------------------------------------------------------------------
    # Case 6: item_compare  (COMPARISON)
    # -------------------------------------------------------------------------
    {
        "name": "COMPARISON — V-neck T-shirt vs Peacock Blouse on style_and_occasion",
        "description": (
            "User has two items in context and asks which suits a casual day out better. "
            "M3 resolved comparison_dimension and supplied preference_weights."
        ),
        "payload": {
            "retrieval_input": {
                "action":             "item_compare",
                "retrieval_strategy": "PARTIAL",
                "user_message":       "which one is better for a casual day out?",
                "items_in_context": {
                    "item_a": {
                        "article_id":        "0513701003",
                        "prod_name":         "V-NECK SS BASIC 3 PK",
                        "product_type_name": "T-shirt",
                        "colour_group_name": "Black",
                        "index_group_name":  "Menswear",
                    },
                    "item_b": {
                        "article_id":        "0817110002",
                        "prod_name":         "Peacock",
                        "product_type_name": "Blouse",
                        "colour_group_name": "Black",
                        "index_group_name":  "Ladieswear",
                    },
                },
                "exclude_ids": [],
                "payload": {
                    "article_id_a":          "0513701003",
                    "article_id_b":          "0817110002",
                    "comparison_dimension":  "style_and_occasion",
                    "preference_weights": {
                        "style":   0.50,
                        "comfort": 0.30,
                        "price":   0.20,
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {},
                    "soft_constraints": {"occasion": "casual"},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "Comparing two items for casual occasion",
                },
                "long_term_preferences": [],
                "style_profile":         {"dominant_style": "casual"},
                "preference_summary":    {},
                "existing_explanation":  None,
            },
        },
    },

    # -------------------------------------------------------------------------
    # Case 7: explanation_generate  (EXPLANATION_WHY)
    # -------------------------------------------------------------------------
    {
        "name": "EXPLANATION_WHY — why was the V-neck T-shirt recommended?",
        "description": (
            "User asks 'why did you recommend this?' M3 supplies prior_claims "
            "(things already told to the user) and matched_prefs (why it was chosen)."
        ),
        "payload": {
            "retrieval_input": {
                "action":             "explanation_generate",
                "retrieval_strategy": "PARTIAL",
                "user_message":       "why did you recommend this one?",
                "items_in_context": {
                    "item_a": {
                        "article_id": "0513701003",
                        "prod_name":  "V-NECK SS BASIC 3 PK",
                    },
                    "item_b": None,
                },
                "exclude_ids": [],
                "payload": {
                    "article_id": "0513701003",
                    "prior_claims": [
                        {
                            "claim_text": "This is a classic black V-neck T-shirt",
                            "claim_type": "colour_claim",
                            "status":     "active",
                        },
                        {
                            "claim_text": "It suits a casual everyday style",
                            "claim_type": "occasion_claim",
                            "status":     "active",
                        },
                    ],
                    "matched_prefs": [
                        {"attribute_name": "colour_group_name", "attribute_value": "Black",    "weight": 0.80},
                        {"attribute_name": "product_type_name", "attribute_value": "T-shirt",  "weight": 0.65},
                        {"attribute_name": "index_group_name",  "attribute_value": "Menswear", "weight": 0.50},
                    ],
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {"colour_group_name": "Black"},
                    "soft_constraints": {},
                    "rejected_items":   [],
                    "accepted_items":   ["0513701003"],
                    "intent_summary":   "Requesting explanation for current recommendation",
                },
                "long_term_preferences": [
                    {"attribute_name": "colour_group_name", "attribute_value": "Black", "weight": 0.80},
                ],
                "style_profile":         {"dominant_style": "casual", "gender": "Menswear"},
                "preference_summary":    {},
                "existing_explanation":  "This is a classic black V-neck T-shirt",
            },
        },
    },

    # -------------------------------------------------------------------------
    # Case 8: item_detail_lookup  (SELECTION_REFERENCE)
    # -------------------------------------------------------------------------
    {
        "name": "SELECTION_REFERENCE — full details of Peacock Blouse",
        "description": (
            "User said 'tell me more about the second one'. "
            "M3 resolved 'the second one' → article_id 0817110002."
        ),
        "payload": {
            "retrieval_input": {
                "action":             "item_detail_lookup",
                "retrieval_strategy": "PARTIAL",
                "user_message":       "tell me more about the second one",
                "items_in_context": {
                    "item_a": {
                        "article_id": "0513701003",
                        "prod_name":  "V-NECK SS BASIC 3 PK",
                    },
                    "item_b": {
                        "article_id": "0817110002",
                        "prod_name":  "Peacock",
                    },
                },
                "exclude_ids": [],
                "payload": {
                    "article_id": "0817110002",
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {},
                    "soft_constraints": {},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "User wants full details of item_b (Peacock Blouse)",
                },
                "long_term_preferences": [],
                "style_profile":         {},
                "preference_summary":    {},
                "existing_explanation":  None,
            },
        },
    },
]


# =============================================================================
# HTTP sender
# =============================================================================

def send_request(payload: dict) -> dict:
    resp = requests.post(M2_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


# =============================================================================
# Display
# =============================================================================

def print_response(result: dict):
    items = result.get("items", [])
    print(f"\n  Status   : {'SUCCESS' if result.get('success') else 'FAILED'}")
    print(f"  Items    : {len(items)} returned")
    if result.get("response_text"):
        print(f"  Summary  : {result['response_text'][:120]}")

    for idx, item in enumerate(items, 1):
        print(f"\n  {'─'*56}")
        print(f"  Item {idx}  : {item.get('prod_name', '?')}")
        print(f"  Article : {item.get('article_id', '?')}")
        print(f"  Colour  : {item.get('colour_group_name', '?')}")
        print(f"  Type    : {item.get('product_type_name', '?')}")
        print(f"  Dept    : {item.get('department_name', '?')}")
        print(f"  Group   : {item.get('index_group_name', '?')}")
        print(f"  Appear  : {item.get('graphical_appearance_name', '?')}")
        score = item.get("score")
        if score is not None:
            print(f"  Score   : {score:.4f}")
        if item.get("explanation"):
            print(f"  Why     : {item['explanation'][:100]}...")


# =============================================================================
# Deterministic accuracy evaluation — no LLM required
# =============================================================================

_REQUIRED_ITEM_FIELDS = ["article_id", "prod_name", "colour_group_name", "product_type_name"]
_MIN_SIMILARITY_SCORE = 0.10
_UNCERTAINTY_WORDS    = ["i think", "maybe", "possibly", "i'm not sure", "i believe", "probably"]


def _check(checks: dict, name: str, passed: bool, detail: str):
    checks[name] = (passed, detail)


def _eval_catalog_search(ri: dict, result: dict, checks: dict):
    filters  = ri.get("payload", {}).get("filters", {})
    items    = result.get("items", [])
    exc_ids  = set(ri.get("exclude_ids", []))

    # Every requested filter must match every returned item
    if filters and items:
        compliant = 0
        for item in items:
            if all(item.get(k, "").lower() == v.lower() for k, v in filters.items()):
                compliant += 1
        pct = compliant / len(items) * 100
        _check(checks, "filter_precision",
               compliant == len(items),
               f"{compliant}/{len(items)} items match filters {list(filters.keys())}  ({pct:.0f}%)")
    else:
        _check(checks, "filter_precision", True, "no hard filters requested")

    # Similarity scores above noise floor
    if items:
        scores  = [item.get("score", 0) for item in items if item.get("score") is not None]
        avg_scr = sum(scores) / len(scores) if scores else 0
        _check(checks, "score_quality",
               avg_scr >= _MIN_SIMILARITY_SCORE,
               f"avg similarity={avg_scr:.4f}  (threshold={_MIN_SIMILARITY_SCORE})")

    # No excluded items leaked through
    returned_ids = {item.get("article_id") for item in items}
    leaked       = returned_ids & exc_ids
    _check(checks, "exclude_compliance",
           len(leaked) == 0,
           f"0 excluded IDs leaked" if not leaked else f"LEAKED: {leaked}")

    # Every item has an explanation (RAG grounding)
    with_exp = sum(1 for item in items if item.get("explanation"))
    _check(checks, "explanations_present",
           with_exp == len(items),
           f"{with_exp}/{len(items)} items have explanation")

    # Response text references returned product names
    resp = (result.get("response_text") or "").lower()
    named = sum(1 for item in items if (item.get("prod_name") or "").lower() in resp)
    _check(checks, "response_mentions_items",
           named > 0,
           f"{named}/{len(items)} item names mentioned in response")


def _eval_single_article(ri: dict, result: dict, checks: dict):
    requested_id = ri.get("payload", {}).get("article_id", "")
    items        = result.get("items", [])

    returned_ids = [item.get("article_id") for item in items]
    _check(checks, "article_id_match",
           requested_id in returned_ids,
           f"requested={requested_id}  returned={returned_ids}")

    resp = result.get("response_text") or ""
    _check(checks, "response_not_empty",
           len(resp.strip()) > 20,
           f"response length={len(resp)} chars")


def _eval_compare(ri: dict, result: dict, checks: dict):
    payload = ri.get("payload", {})
    id_a    = payload.get("article_id_a", "")
    id_b    = payload.get("article_id_b", "")
    items   = result.get("items", [])
    ids     = {item.get("article_id") for item in items}

    _check(checks, "article_a_present", id_a in ids, f"article_a={id_a}")
    _check(checks, "article_b_present", id_b in ids, f"article_b={id_b}")

    resp = result.get("response_text") or ""
    _check(checks, "response_not_empty", len(resp.strip()) > 20,
           f"response length={len(resp)} chars")

    dim  = payload.get("comparison_dimension", "")
    _check(checks, "dimension_mentioned",
           not dim or dim.lower() in resp.lower(),
           f"dimension '{dim}' {'found' if dim.lower() in resp.lower() else 'NOT found'} in response")


def _eval_explanation(ri: dict, result: dict, checks: dict):
    requested_id = ri.get("payload", {}).get("article_id", "")
    items        = result.get("items", [])
    returned_ids = [item.get("article_id") for item in items]

    _check(checks, "article_id_match",
           requested_id in returned_ids,
           f"requested={requested_id}  returned={returned_ids}")

    explanations = [item.get("explanation", "") for item in items]
    has_exp      = any(len(e) > 20 for e in explanations)
    _check(checks, "explanation_present", has_exp,
           f"explanation {'present' if has_exp else 'MISSING or too short'}")

    # Uncertainty / hallucination signal: LLM should not hedge on known product facts
    combined = " ".join(explanations).lower()
    hedges   = [w for w in _UNCERTAINTY_WORDS if w in combined]
    _check(checks, "no_uncertainty_language",
           len(hedges) == 0,
           f"no hedge words found" if not hedges else f"hedge words detected: {hedges}")


def _grade(pct: float) -> str:
    if pct >= 90: return "EXCELLENT"
    if pct >= 75: return "GOOD"
    if pct >= 50: return "FAIR"
    return "POOR"


def evaluate_accuracy(ri: dict, result: dict):
    """Deterministic, zero-LLM accuracy evaluation for any M2 action type."""
    action = ri.get("action", "")
    items  = result.get("items", [])
    checks: dict = {}

    # --- Universal checks ---
    _check(checks, "api_success",   result.get("success", False),  "M2 returned success=True")
    _check(checks, "has_results",   len(items) > 0,                f"{len(items)} items returned")

    # Field completeness — every item should have core fields
    if items:
        complete = sum(
            1 for item in items
            if all(item.get(f) for f in _REQUIRED_ITEM_FIELDS)
        )
        _check(checks, "field_completeness",
               complete == len(items),
               f"{complete}/{len(items)} items have all required fields")

    # --- Action-specific checks ---
    if action == "catalog_search":
        _eval_catalog_search(ri, result, checks)
    elif action == "item_attribute_lookup":
        _eval_single_article(ri, result, checks)
    elif action == "item_compare":
        _eval_compare(ri, result, checks)
    elif action == "explanation_generate":
        _eval_explanation(ri, result, checks)
    elif action == "item_detail_lookup":
        _eval_single_article(ri, result, checks)

    # --- Report ---
    passed  = sum(1 for ok, _ in checks.values() if ok)
    total   = len(checks)
    pct     = passed / total * 100 if total else 0
    grade   = _grade(pct)

    print(f"\n  {'─'*56}")
    print(f"  [ACCURACY]  Deterministic Evaluation  ({action})")
    print(f"  {'─'*56}")
    for name, (ok, detail) in checks.items():
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  {name:<30}  {detail}")
    print(f"  {'─'*56}")
    print(f"  Score : {passed}/{total} checks passed  ({pct:.0f}%)  [{grade}]")
    print(f"  {'─'*56}\n")


def _print_payload_summary(ri: dict):
    action  = ri.get("action", "unknown")
    payload = ri.get("payload", {})
    print("\n  --- M3 Payload Summary ---")
    print(f"  action          : {action}")
    print(f"  user_message    : {ri.get('user_message', '')}")

    if action == "catalog_search":
        hints = payload.get("purchase_history_hints", {})
        print(f"  filters         : {payload.get('filters', {})}")
        print(f"  soft_constraints: {payload.get('soft_constraints', {})}")
        print(f"  penalties       : {payload.get('penalties', {})}")
        print(f"  pref_boosts     : {len(payload.get('preference_boosts', []))} boosts")
        print(f"  purchase hints  : dominant={hints.get('dominant_colour')}/{hints.get('dominant_type')}  budget={hints.get('budget_tier')}")
        print(f"  exclude_ids     : {ri.get('exclude_ids', [])}")

    elif action == "item_attribute_lookup":
        print(f"  article_id      : {payload.get('article_id', '?')}")
        print(f"  attribute_topic : {payload.get('attribute_topic', '?')}")

    elif action == "item_compare":
        print(f"  article_id_a    : {payload.get('article_id_a', '?')}")
        print(f"  article_id_b    : {payload.get('article_id_b', '?')}")
        print(f"  dimension       : {payload.get('comparison_dimension', '?')}")
        print(f"  pref_weights    : {payload.get('preference_weights', {})}")

    elif action == "explanation_generate":
        print(f"  article_id      : {payload.get('article_id', '?')}")
        print(f"  matched_prefs   : {len(payload.get('matched_prefs', []))} prefs")
        print(f"  prior_claims    : {len(payload.get('prior_claims', []))} active claims")

    elif action == "item_detail_lookup":
        print(f"  article_id      : {payload.get('article_id', '?')}")


def run_case(case: dict):
    print(f"\n{'='*62}")
    print(f"  TEST: {case['name']}")
    print(f"  {case['description']}")

    ri = case["payload"]["retrieval_input"]
    _print_payload_summary(ri)

    print(f"{'='*62}")
    print(f"  Sending POST to {M2_URL} ...")

    try:
        result = send_request(case["payload"])
    except requests.exceptions.ConnectionError:
        print("\n  [ERROR] Cannot connect to M2 server.")
        print("  Start it with: uvicorn m2_multimodal_rag.backend.main:app --host 0.0.0.0 --port 8001")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return

    print_response(result)

    if result.get("success"):
        evaluate_accuracy(ri, result)

    print(f"{'='*62}")


# =============================================================================
# Custom query builder
# =============================================================================

_COLOUR_MAP = {
    "dark blue": "Dark Blue", "navy": "Dark Blue", "light blue": "Light Blue",
    "black": "Black", "white": "White", "red": "Red", "pink": "Pink",
    "green": "Green", "yellow": "Yellow", "grey": "Grey", "gray": "Grey",
    "beige": "Beige", "brown": "Brown", "orange": "Orange", "purple": "Purple",
    "blue": "Blue",
}
_PRODUCT_MAP = {
    "dress": "Dress", "dresses": "Dress", "jeans": "Trousers",
    "trousers": "Trousers", "pants": "Trousers", "shirt": "Shirt",
    "top": "Top", "blouse": "Blouse", "jacket": "Jacket", "coat": "Jacket",
    "sweater": "Sweater", "jumper": "Sweater", "hoodie": "Hoodie",
    "skirt": "Skirt", "shorts": "Shorts", "leggings": "Leggings/Tights",
    "t-shirt": "T-shirt", "tshirt": "T-shirt",
}
_GENDER_MAP = {
    "men": "Menswear", "man": "Menswear", "male": "Menswear",
    "women": "Ladieswear", "woman": "Ladieswear", "female": "Ladieswear",
}


def _build_custom_payload(query: str) -> dict:
    msg   = query.lower()
    filters: dict = {}

    for kw in sorted(_COLOUR_MAP, key=len, reverse=True):
        if kw in msg:
            filters["colour_group_name"] = _COLOUR_MAP[kw]
            break
    for kw in sorted(_PRODUCT_MAP, key=len, reverse=True):
        if kw in msg:
            filters["product_type_name"] = _PRODUCT_MAP[kw]
            break
    for kw, val in _GENDER_MAP.items():
        import re
        if re.search(r'\b' + kw + r'\b', msg):
            filters["index_group_name"] = val
            break

    return {
        "retrieval_input": {
            "action":             "catalog_search",
            "retrieval_strategy": "FULL",
            "user_message":       query,
            "items_in_context":   {"item_a": None, "item_b": None},
            "exclude_ids":        [],
            "payload": {
                "filters":                filters,
                "soft_constraints":       {},
                "preference_boosts":      [],
                "penalties":              {},
                "purchase_history_hints": {
                    "top_colours":           [],
                    "top_product_types":     [],
                    "inferred_gender":       None,
                    "budget_tier":           None,
                    "preferred_price_range": None,
                    "dominant_colour":       None,
                    "dominant_type":         None,
                },
            },
        },
        "memory_context": {
            "dialogue_state": {
                "hard_constraints": filters,
                "soft_constraints": {},
                "rejected_items":   [],
                "accepted_items":   [],
                "intent_summary":   "",
            },
            "long_term_preferences": [],
            "style_profile":         {},
            "preference_summary":    {},
            "existing_explanation":  None,
        },
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Send dummy M3-style catalog search requests to M2'
    )
    parser.add_argument('--case',  type=int, default=None,
                        help='Run a specific case by number (1-8)')
    parser.add_argument('--query', type=str, default=None,
                        help='Send a custom query (catalog_search payload auto-built)')
    parser.add_argument('--json',  action='store_true',
                        help='Print the full JSON payload sent to M2')
    args = parser.parse_args()

    print("=" * 62)
    print("  M3 -> M2 Request Tester  (all 5 action types)")
    print(f"  Target : {M2_URL}")
    print(f"  Cases  : {len(DUMMY_CASES)}  (1-4 catalog_search, 5 attr, 6 compare, 7 explain, 8 detail)")
    print("=" * 62)

    if args.query:
        payload = _build_custom_payload(args.query)
        case    = {
            "name":        f"Custom query: \"{args.query}\"",
            "description": "Auto-built M3-style payload from query text.",
            "payload":     payload,
        }
        if args.json:
            print("\n  Full JSON payload:")
            print(json.dumps(payload, indent=4))
        run_case(case)

    elif args.case is not None:
        idx = args.case - 1
        if idx < 0 or idx >= len(DUMMY_CASES):
            print(f"[ERROR] Case must be between 1 and {len(DUMMY_CASES)}.")
            sys.exit(1)
        case = DUMMY_CASES[idx]
        if args.json:
            print("\n  Full JSON payload:")
            print(json.dumps(case["payload"], indent=4))
        run_case(case)

    else:
        print(f"  Running all {len(DUMMY_CASES)} cases...\n")
        for case in DUMMY_CASES:
            run_case(case)


if __name__ == '__main__':
    main()
