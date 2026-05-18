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
import os
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
# RAGAS evaluation helpers
# =============================================================================

def _build_ragas_components():
    """
    Builds RAGAS LLM + embeddings backed by Groq + HuggingFace.
    Uses the same GROQ_API_KEY already in .env — no extra key needed.
    Returns (llm, embeddings) or (None, None) on failure.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        print("  [RAGAS] Install: pip install ragas langchain-groq langchain-huggingface")
        return None, None

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        print("  [RAGAS] GROQ_API_KEY not set — cannot run RAGAS.")
        return None, None

    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # LLM wrapper — try new API first, fall back to deprecated wrapper
    try:
        from ragas.llms import LangchainLLMWrapper
        llm = LangchainLLMWrapper(ChatGroq(model=groq_model, api_key=groq_key))
    except Exception as e:
        print(f"  [RAGAS] LLM setup failed: {e}")
        return None, None

    # Embeddings — try native RAGAS HuggingFace first, then LangChain wrapper
    try:
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmb
        emb = RagasHFEmb(model="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_huggingface import HuggingFaceEmbeddings
            emb = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
        except Exception as e:
            print(f"  [RAGAS] Embeddings setup failed: {e}")
            return None, None

    return llm, emb


def _ragas_contexts(items: list) -> list:
    contexts = []
    for item in items:
        parts = list(filter(None, [
            item.get("prod_name"),
            item.get("colour_group_name"),
            item.get("product_type_name"),
            item.get("detail_desc"),
        ]))
        if parts:
            contexts.append(" ".join(parts))
    return contexts or ["No item context available."]


def _ragas_answer(response_text: str, items: list) -> str:
    explanations = [item["explanation"] for item in items if item.get("explanation")]
    if explanations:
        return response_text + " " + " ".join(explanations)
    return response_text


def _ragas_metrics(llm, emb):
    try:
        from ragas.metrics import Faithfulness, ResponseRelevancy
        return (
            [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb, strictness=1)],
            "faithfulness",
            "response_relevancy",
        )
    except Exception:
        from ragas.metrics.collections import faithfulness, answer_relevancy
        faithfulness.llm            = llm
        answer_relevancy.llm        = llm
        answer_relevancy.embeddings = emb
        return [faithfulness, answer_relevancy], "faithfulness", "answer_relevancy"


def _ragas_dataset(user_message: str, answer: str, contexts: list):
    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        return EvaluationDataset(samples=[SingleTurnSample(
            user_input=user_message,
            response=answer,
            retrieved_contexts=contexts,
        )])
    except Exception:
        from datasets import Dataset
        return Dataset.from_dict({
            "question": [user_message],
            "answer":   [answer],
            "contexts": [contexts],
        })


def _ragas_grade(overall: float) -> str:
    if overall >= 0.85:
        return "EXCELLENT"
    if overall >= 0.70:
        return "GOOD"
    if overall >= 0.50:
        return "FAIR"
    return "POOR"


def _ragas_evaluate(user_message: str, response_text: str, items: list):
    """
    Runs RAGAS Faithfulness + ResponseRelevancy on a single M2 response.

    Faithfulness      — is the answer grounded in the retrieved item data?
    ResponseRelevancy — does the answer actually address what the user asked?

    Groq compatibility: max_workers=1 serialises calls (Groq rejects n>1 batches).
    Answer includes item explanations so RAGAS has specific verifiable claims.
    """
    try:
        from ragas import evaluate, RunConfig
    except ImportError:
        print("  [RAGAS] ragas not installed. Run: pip install ragas")
        return

    if not response_text:
        print("  [RAGAS] No response text — skipping.")
        return

    llm, emb = _build_ragas_components()
    if llm is None:
        return

    contexts          = _ragas_contexts(items)
    answer            = _ragas_answer(response_text, items)
    metrics, f_key, ar_key = _ragas_metrics(llm, emb)
    dataset           = _ragas_dataset(user_message, answer, contexts)
    run_cfg           = RunConfig(max_workers=1, max_retries=2, timeout=120)

    print("  [RAGAS] Evaluating (serialised calls, no concurrent batching)...")
    try:
        result  = evaluate(dataset, metrics=metrics, run_config=run_cfg)
        df      = result.to_pandas()

        if f_key in df.columns:
            f_score = float(df[f_key].iloc[0])
        else:
            f_score = 0.0
        if ar_key in df.columns:
            a_score = float(df[ar_key].iloc[0])
        else:
            a_score = 0.0
        overall = (f_score + a_score) / 2
        grade   = _ragas_grade(overall)

        print(f"\n  {'─'*56}")
        print("  [RAGAS] RAG Pipeline Evaluation")
        print(f"  {'─'*56}")
        print(f"  Faithfulness     : {f_score:.3f}  — answer grounded in item data")
        print(f"  Answer Relevancy : {a_score:.3f}  — answer addresses user question")
        print(f"  Overall          : {overall:.3f}  [{grade}]")
        print(f"  {'─'*56}\n")
    except Exception as e:
        print(f"  [RAGAS] Evaluation error: {e}")


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


def run_case(case: dict, run_ragas: bool = False):
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

    action = ri.get("action", "")
    if action == "catalog_search":
        print("\n  (Accuracy report printed above by M2 handler)")
    else:
        print("\n  (Self-reflection + VLM verification printed above by M2 handler)")

    if run_ragas and result.get("success"):
        _ragas_evaluate(
            user_message  = ri.get("user_message", ""),
            response_text = result.get("response_text", ""),
            items         = result.get("items", []),
        )

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
    parser.add_argument('--ragas', action='store_true',
                        help='Run RAGAS faithfulness + answer_relevancy after each case')
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
        run_case(case, run_ragas=args.ragas)

    elif args.case is not None:
        idx = args.case - 1
        if idx < 0 or idx >= len(DUMMY_CASES):
            print(f"[ERROR] Case must be between 1 and {len(DUMMY_CASES)}.")
            sys.exit(1)
        case = DUMMY_CASES[idx]
        if args.json:
            print("\n  Full JSON payload:")
            print(json.dumps(case["payload"], indent=4))
        run_case(case, run_ragas=args.ragas)

    else:
        print(f"  Running all {len(DUMMY_CASES)} cases...\n")
        for case in DUMMY_CASES:
            run_case(case, run_ragas=args.ragas)


if __name__ == '__main__':
    main()
