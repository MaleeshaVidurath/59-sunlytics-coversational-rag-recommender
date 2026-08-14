"""
Knowledge Base Test Script — NOVELTY 5
=======================================
Sends requests to M2's /api/process endpoint in the EXACT format
that M3 produces, specifically to exercise the 5 Fashion KB improvements.

PART 1 — KB Unit Tests (no server needed)
  Tests all 7 FashionKBRetriever methods directly.

PART 2 — KB Integration Tests (requires M2 server at port 8001)
  Each case is a real M3-style payload targeting one KB improvement:
    Case 1 : Occasion Rules         — interview, professional style
    Case 2 : Appearance Rules       — evening party, glamorous sequin  (Improvement 1)
    Case 3 : Gender-Aware Rules     — children's casual wear            (Improvement 2)
    Case 4 : Kansei Detection       — 'romantic' inferred from message  (Improvement 3)
    Case 5 : Colour Harmony logging — complement to a black item        (Improvement 4)
    Case 6 : CLIP Visual Terms      — beach holiday occasion            (Improvement 5)

M2 server:
  uvicorn m2_multimodal_rag.backend.main:app --host 0.0.0.0 --port 8001

Usage:
  python m2_multimodal_rag/testing_scripts/test_knowledge_base.py            # unit + all integration
  python m2_multimodal_rag/testing_scripts/test_knowledge_base.py --unit-only
  python m2_multimodal_rag/testing_scripts/test_knowledge_base.py --integ-only
  python m2_multimodal_rag/testing_scripts/test_knowledge_base.py --case 3
  python m2_multimodal_rag/testing_scripts/test_knowledge_base.py --json     # print full JSON payload
"""

import sys
import os
import json
import argparse
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

M2_URL = "http://localhost:8001/api/process"


# =============================================================================
# PART 1 — KB UNIT TESTS  (no server required)
# =============================================================================

def _ok(label: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}]  {label:<48}  {detail}")
    return passed


def run_unit_tests() -> int:
    """Runs deterministic unit tests on all KB methods. Returns failure count."""
    print("\n" + "=" * 62)
    print("  PART 1 — KB Unit Tests  (no server required)")
    print("=" * 62)

    try:
        from m2_multimodal_rag.knowledge_base.kb_retriever import kb_retriever
    except ImportError as e:
        print(f"  [ERROR] Cannot import kb_retriever: {e}")
        return 1

    failures = 0

    # ── score() — base scoring ────────────────────────────────────────────────
    print("\n  [score()] — Base KB scoring")
    s = kb_retriever.score("Black", "Blazer", "Solid",
                           occasion="interview", style_word="professional")
    failures += 0 if _ok("Black Blazer Solid / interview+professional → positive",
                          s > 0, f"score={s:+.4f}") else 1

    s2 = kb_retriever.score("Red", "T-shirt", "Front print",
                             occasion="interview", style_word=None)
    failures += 0 if _ok("Red T-shirt FrontPrint / interview → penalty",
                          s2 < 0, f"score={s2:+.4f}") else 1

    s3 = kb_retriever.score("Gold", "Dress", "Sequin",
                             occasion="evening", style_word="glamorous")
    failures += 0 if _ok("Gold Dress Sequin / evening+glamorous → strong positive",
                          s3 > 0.20, f"score={s3:+.4f}") else 1

    s4 = kb_retriever.score("Gold", "Dress", "Sequin",
                             occasion="gym", style_word="sporty")
    failures += 0 if _ok("Gold Dress Sequin / gym+sporty → penalty",
                          s4 < 0, f"score={s4:+.4f}") else 1

    # ── score() — Improvement 1: Appearance Rules ─────────────────────────────
    print("\n  [score()] — Improvement 1: Appearance Rules")
    base = kb_retriever.score("Black", "Dress", "",
                               occasion="evening", style_word="glamorous")
    with_sequin = kb_retriever.score("Black", "Dress", "Sequin",
                                     occasion="evening", style_word="glamorous")
    failures += 0 if _ok("Sequin boosts score vs no appearance (evening+glamorous)",
                          with_sequin > base,
                          f"base={base:+.4f}  with_sequin={with_sequin:+.4f}") else 1

    with_solid = kb_retriever.score("Black", "Dress", "Solid",
                                    occasion="interview", style_word="professional")
    base_int = kb_retriever.score("Black", "Dress", "",
                                  occasion="interview", style_word="professional")
    failures += 0 if _ok("Solid boosts score vs no appearance (interview+professional)",
                          with_solid > base_int,
                          f"base={base_int:+.4f}  with_solid={with_solid:+.4f}") else 1

    # ── score() — Improvement 2: Gender-Aware Rules ───────────────────────────
    print("\n  [score()] — Improvement 2: Gender-Aware Rules")
    neutral = kb_retriever.score("Pink", "Dress", "",
                                  occasion="date", style_word="romantic",
                                  index_group_name="Ladieswear")
    penalised = kb_retriever.score("Pink", "Dress", "",
                                    occasion="date", style_word="romantic",
                                    index_group_name="Menswear")
    failures += 0 if _ok("Pink Dress / Menswear scored lower than Ladieswear",
                          penalised < neutral,
                          f"Ladieswear={neutral:+.4f}  Menswear={penalised:+.4f}") else 1

    children_ok = kb_retriever.score("Yellow", "T-shirt", "Solid",
                                      occasion="casual", style_word=None,
                                      index_group_name="Children")
    children_dark = kb_retriever.score("Black", "Blazer", "Solid",
                                        occasion="casual", style_word=None,
                                        index_group_name="Children")
    failures += 0 if _ok("Children: Yellow T-shirt better scored than Black Blazer",
                          children_ok > children_dark,
                          f"Yellow T-shirt={children_ok:+.4f}  Black Blazer={children_dark:+.4f}") else 1

    # ── detect_kansei_from_message() — Improvement 3 ──────────────────────────
    print("\n  [detect_kansei_from_message()] — Improvement 3: Kansei Detection")
    words = kb_retriever.detect_kansei_from_message(
        "I want something chic and elegant for a night out"
    )
    failures += 0 if _ok("'chic elegant night out' detects 'elegant'",
                          "elegant" in words, f"detected={words}") else 1

    words2 = kb_retriever.detect_kansei_from_message(
        "looking for a romantic lace dress for a date"
    )
    failures += 0 if _ok("'romantic lace dress date' detects 'romantic'",
                          "romantic" in words2, f"detected={words2}") else 1

    words3 = kb_retriever.detect_kansei_from_message("show me a dress")
    failures += 0 if _ok("'show me a dress' → no Kansei words detected",
                          len(words3) == 0, f"detected={words3}") else 1

    words4 = kb_retriever.detect_kansei_from_message(
        "I need something sporty and athletic for the gym"
    )
    failures += 0 if _ok("'sporty athletic gym' detects 'sporty'",
                          "sporty" in words4, f"detected={words4}") else 1

    # ── harmony_score() — Improvement 4 ───────────────────────────────────────
    print("\n  [harmony_score()] — Improvement 4: Colour Harmony")
    h1 = kb_retriever.harmony_score("Black", "White")
    failures += 0 if _ok("Black × White → complementary (+0.15)",
                          abs(h1 - 0.15) < 0.01, f"score={h1:+.2f}") else 1

    h2 = kb_retriever.harmony_score("Grey", "Dark Grey")
    failures += 0 if _ok("Grey × Dark Grey → analogous (+0.08)",
                          abs(h2 - 0.08) < 0.01, f"score={h2:+.2f}") else 1

    h3 = kb_retriever.harmony_score("Red", "Pink")
    failures += 0 if _ok("Red × Pink → clashing (−0.12)",
                          abs(h3 + 0.12) < 0.01, f"score={h3:+.2f}") else 1

    h4 = kb_retriever.harmony_score("Dark Blue", "Orange")
    failures += 0 if _ok("Dark Blue × Orange → complementary (+0.15)",
                          abs(h4 - 0.15) < 0.01, f"score={h4:+.2f}") else 1

    h5 = kb_retriever.harmony_score("Black", "Black")
    failures += 0 if _ok("Same colour → neutral (0.0)",
                          h5 == 0.0, f"score={h5:+.2f}") else 1

    h6 = kb_retriever.harmony_score("Silver", "Gold")
    failures += 0 if _ok("Silver × Gold → clashing (−0.12) — check reverse",
                          h6 < 0, f"score={h6:+.2f}") else 1

    # ── get_clip_terms() — Improvement 5 ──────────────────────────────────────
    print("\n  [get_clip_terms()] — Improvement 5: CLIP Visual Terms")
    terms = kb_retriever.get_clip_terms(occasion="interview", style_word="professional")
    failures += 0 if _ok("interview+professional → non-empty clip terms",
                          len(terms) > 10, f"terms='{terms[:60]}'") else 1

    terms2 = kb_retriever.get_clip_terms(occasion="party", style_word="glamorous")
    failures += 0 if _ok("party+glamorous → contains 'sequin' or 'glamorous'",
                          any(w in terms2.lower() for w in ["sequin", "glamorous", "sparkle"]),
                          f"terms='{terms2[:60]}'") else 1

    terms3 = kb_retriever.get_clip_terms(occasion=None, style_word=None)
    failures += 0 if _ok("no occasion/style → empty string",
                          terms3 == "", f"terms='{terms3}'") else 1

    # ── get_context() ──────────────────────────────────────────────────────────
    print("\n  [get_context()]")
    ctx = kb_retriever.get_context(occasion="formal", style_word="elegant")
    failures += 0 if _ok("formal+elegant → non-empty context string",
                          len(ctx) > 20, f"ctx='{ctx[:80]}'") else 1

    ctx2 = kb_retriever.get_context(occasion=None, style_word=None)
    failures += 0 if _ok("no inputs → empty string",
                          ctx2 == "", f"ctx='{ctx2}'") else 1

    # ── get_explanation() ─────────────────────────────────────────────────────
    print("\n  [get_explanation()]")
    expl = kb_retriever.get_explanation("Black", "Blazer",
                                         occasion="interview", style_word="professional")
    failures += 0 if _ok("Black Blazer / interview+professional → non-empty",
                          len(expl) > 10, f"expl='{expl[:80]}'") else 1

    expl2 = kb_retriever.get_explanation("Red", "Dress",
                                          occasion="party", style_word="bold")
    failures += 0 if _ok("Red Dress / party+bold → non-empty",
                          len(expl2) > 10, f"expl='{expl2[:80]}'") else 1

    # ── Alias resolution ───────────────────────────────────────────────────────
    print("\n  [Alias resolution]")
    s_alias = kb_retriever.score("Dark Blue", "Shirt", "Solid",
                                  occasion="work", style_word="chic")
    s_canon = kb_retriever.score("Dark Blue", "Shirt", "Solid",
                                  occasion="office", style_word="elegant")
    failures += 0 if _ok("'work'→'office' and 'chic'→'elegant' aliases resolve correctly",
                          abs(s_alias - s_canon) < 0.01,
                          f"alias={s_alias:+.4f}  canonical={s_canon:+.4f}") else 1

    total  = 23
    passed = total - failures
    grade  = "EXCELLENT" if failures == 0 else ("GOOD" if failures <= 2 else "NEEDS REVIEW")
    print(f"\n  {'─'*58}")
    print(f"  Unit Test Result : {passed}/{total} passed  [{grade}]")
    print(f"  {'─'*58}\n")
    return failures


# =============================================================================
# PART 2 — KB INTEGRATION TEST CASES
# All cases are INITIAL_REQUEST — catalog_search / FULL retrieval — the exact
# body M3 sends to M2's /api/process endpoint for a first-turn user message.
#
# Format mirrors test_m3_request.py Cases 1-4:
#   • filters          = hard constraints extracted from the message (colour, type, gender)
#   • soft_constraints = occasion / style preferences
#   • preference_boosts= reinforce the filter values with weights
#   • purchase_history_hints = customer history from M3 memory pipeline
#   • dialogue_state.hard_constraints = same as filters
# =============================================================================

KB_CASES = [

    # ── Case 1: Occasion Rules (A/B/C) ────────────────────────────────────────
    # User explicitly says "black blazer" → M3 extracts colour+type as hard constraints.
    # KB Occasion scoring should boost Black Blazer Solid for interview+professional
    # and penalise bright/casual colours.
    {
        "name": "INITIAL_REQUEST — black blazer for job interview",
        "description": (
            "User asks for a black blazer for a job interview. "
            "M3 extracts colour=Black, type=Blazer as hard constraints. "
            "KB Occasion Rules should boost Black/Blazer/Solid and penalise Red/Yellow."
        ),
        "kb_improvement": "Occasion Rules (A/B/C) — base KB scoring",
        "expected": {
            "avoid_colours": ["Red", "Yellow", "Orange"],
        },
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I need a black blazer for a job interview",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {
                        "colour_group_name": "Black",
                        "product_type_name": "Blazer",
                    },
                    "soft_constraints": {
                        "occasion": "interview",
                        "style":    "professional",
                    },
                    "preference_boosts": [
                        {"attribute": "colour_group_name", "value": "Black",  "weight": 0.75},
                        {"attribute": "product_type_name", "value": "Blazer", "weight": 0.70},
                    ],
                    "penalties": {
                        "colour_group_name": ["Red", "Yellow", "Orange"],
                    },
                    "purchase_history_hints": {
                        "top_colours":           ["Black", "Dark Blue", "Grey"],
                        "top_product_types":     ["Blazer", "Shirt", "Trousers"],
                        "inferred_gender":       "Ladieswear",
                        "budget_tier":           "mid",
                        "preferred_price_range": [0.04, 0.15],
                        "dominant_colour":       "Black",
                        "dominant_type":         "Blazer",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {
                        "colour_group_name": "Black",
                        "product_type_name": "Blazer",
                    },
                    "soft_constraints": {"occasion": "interview", "style": "professional"},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "Black blazer for job interview",
                },
                "long_term_preferences": [
                    {"attribute_name": "colour_group_name", "attribute_value": "Black",  "weight": 0.75},
                    {"attribute_name": "product_type_name", "attribute_value": "Blazer", "weight": 0.70},
                ],
                "style_profile":        {"dominant_style": "professional", "gender": "Ladieswear"},
                "preference_summary":   {},
                "existing_explanation": None,
            },
        },
    },

    # ── Case 2: Appearance Rules (Improvement 1) ──────────────────────────────
    # User says "black sequin dress for a party" → colour=Black, type=Dress hard constraints.
    # KB Appearance Layer should further boost Sequin/Glittering for party+glamorous.
    {
        "name": "INITIAL_REQUEST — black sequin dress for a party",
        "description": (
            "User asks for a black sequin dress for a party. "
            "M3 extracts colour=Black, type=Dress as hard constraints. "
            "KB Appearance Rules (Improvement 1) should boost Sequin/Glittering appearance."
        ),
        "kb_improvement": "Improvement 1: APPEARANCE_RULES — graphical_appearance_name scoring",
        "expected": {
            "preferred_appearances": ["Sequin", "Glittering/Metallic"],
        },
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I want a black sequin dress for a party tonight",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {
                        "colour_group_name": "Black",
                        "product_type_name": "Dress",
                    },
                    "soft_constraints": {
                        "occasion": "party",
                        "style":    "glamorous",
                    },
                    "preference_boosts": [
                        {"attribute": "colour_group_name",         "value": "Black",  "weight": 0.65},
                        {"attribute": "product_type_name",         "value": "Dress",  "weight": 0.70},
                        {"attribute": "graphical_appearance_name", "value": "Sequin", "weight": 0.80},
                    ],
                    "penalties": {},
                    "purchase_history_hints": {
                        "top_colours":           ["Black", "Gold", "Dark Red"],
                        "top_product_types":     ["Dress", "Top", "Blouse"],
                        "inferred_gender":       "Ladieswear",
                        "budget_tier":           "high",
                        "preferred_price_range": [0.06, 0.25],
                        "dominant_colour":       "Black",
                        "dominant_type":         "Dress",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {
                        "colour_group_name": "Black",
                        "product_type_name": "Dress",
                    },
                    "soft_constraints": {"occasion": "party", "style": "glamorous"},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "Black sequin dress for party",
                },
                "long_term_preferences": [
                    {"attribute_name": "colour_group_name", "attribute_value": "Black", "weight": 0.65},
                    {"attribute_name": "product_type_name", "attribute_value": "Dress", "weight": 0.70},
                ],
                "style_profile":        {"dominant_style": "glamorous", "gender": "Ladieswear"},
                "preference_summary":   {},
                "existing_explanation": None,
            },
        },
    },

    # ── Case 3: Gender-Aware Rules (Improvement 2) ────────────────────────────
    # User says "blue T-shirt for my child" → colour=Blue, type=T-shirt, gender=Children.
    # KB Gender-Aware Rules should penalise dark/unusual colours for Children index group.
    {
        "name": "INITIAL_REQUEST — blue T-shirt for children",
        "description": (
            "User asks for a blue T-shirt for their child. "
            "M3 extracts colour=Blue, type=T-shirt, gender=Children as hard constraints. "
            "KB Gender-Aware Rules (Improvement 2) should penalise dark/unusual colours."
        ),
        "kb_improvement": "Improvement 2: GENDER_COLOUR_RULES — demographic-aware scoring",
        "expected": {
            "avoid_colours": ["Black", "Dark Grey"],
        },
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I need a blue T-shirt for my child",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {
                        "colour_group_name": "Blue",
                        "product_type_name": "T-shirt",
                        "index_group_name":  "Children Sizes 92-140",
                    },
                    "soft_constraints": {
                        "occasion": "casual",
                    },
                    "preference_boosts": [
                        {"attribute": "colour_group_name", "value": "Blue",    "weight": 0.70},
                        {"attribute": "product_type_name", "value": "T-shirt", "weight": 0.65},
                    ],
                    "penalties": {
                        "colour_group_name": ["Black", "Dark Grey"],
                    },
                    "purchase_history_hints": {
                        "top_colours":           ["Blue", "Yellow", "Light Blue"],
                        "top_product_types":     ["T-shirt", "Shorts"],
                        "inferred_gender":       "Children",
                        "budget_tier":           "low",
                        "preferred_price_range": [0.01, 0.05],
                        "dominant_colour":       "Blue",
                        "dominant_type":         "T-shirt",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {
                        "colour_group_name": "Blue",
                        "product_type_name": "T-shirt",
                        "index_group_name":  "Children Sizes 92-140",
                    },
                    "soft_constraints": {"occasion": "casual"},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "Blue T-shirt for child",
                },
                "long_term_preferences": [
                    {"attribute_name": "colour_group_name", "attribute_value": "Blue",    "weight": 0.70},
                    {"attribute_name": "product_type_name", "attribute_value": "T-shirt", "weight": 0.65},
                ],
                "style_profile":        {"gender": "Children"},
                "preference_summary":   {},
                "existing_explanation": None,
            },
        },
    },

    # ── Case 4: Kansei Detection from Message (Improvement 3) ─────────────────
    # User says "romantic lace dress for a date" → type=Dress hard constraint.
    # soft_constraints has occasion=date but NO style key — KB detects 'romantic'
    # from the raw user_message via detect_kansei_from_message().
    {
        "name": "INITIAL_REQUEST — romantic lace dress for a date night",
        "description": (
            "User asks for a romantic lace dress for a date night. "
            "M3 extracts type=Dress as hard constraint but does NOT set style in soft_constraints. "
            "KB Kansei detection (Improvement 3) infers style='romantic' from user_message."
        ),
        "kb_improvement": "Improvement 3: detect_kansei_from_message() → inferred_style",
        "expected": {
            "preferred_colours": ["Pink", "Red", "White", "Beige"],
        },
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I want a romantic lace dress for a date night",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {
                        "product_type_name": "Dress",
                    },
                    "soft_constraints": {
                        "occasion": "date",
                        # style is intentionally absent — KB infers 'romantic' from message
                    },
                    "preference_boosts": [
                        {"attribute": "product_type_name", "value": "Dress", "weight": 0.70},
                    ],
                    "penalties": {},
                    "purchase_history_hints": {
                        "top_colours":           ["Pink", "Red", "White"],
                        "top_product_types":     ["Dress", "Blouse"],
                        "inferred_gender":       "Ladieswear",
                        "budget_tier":           "mid",
                        "preferred_price_range": [0.04, 0.15],
                        "dominant_colour":       "Pink",
                        "dominant_type":         "Dress",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {"product_type_name": "Dress"},
                    "soft_constraints": {"occasion": "date"},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "Romantic lace dress for date night",
                },
                "long_term_preferences": [
                    {"attribute_name": "product_type_name", "attribute_value": "Dress", "weight": 0.70},
                ],
                "style_profile":        {"dominant_style": "romantic", "gender": "Ladieswear"},
                "preference_summary":   {},
                "existing_explanation": None,
            },
        },
    },

    # ── Case 5: Colour Harmony (Improvement 4) ────────────────────────────────
    # User asks for a white shirt for men — M3 extracts colour=White, type=Shirt, gender=Menswear.
    # When M2 returns 2 items, harmony_score() logs the colour pair coherence.
    # (Observe [KB] Colour harmony in M2 server logs after MMR phase.)
    {
        "name": "INITIAL_REQUEST — white shirt for men, casual occasion",
        "description": (
            "User asks for a white shirt for men. "
            "M3 extracts colour=White, type=Shirt, gender=Menswear as hard constraints. "
            "KB Colour Harmony (Improvement 4) logs pair score in server output after MMR."
        ),
        "kb_improvement": "Improvement 4: harmony_score() — colour pair coherence logged post-MMR",
        "expected": {
            "preferred_colours": ["White"],
        },
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I need a white shirt for men for a casual day",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {
                        "colour_group_name": "White",
                        "product_type_name": "Shirt",
                        "index_group_name":  "Menswear",
                    },
                    "soft_constraints": {
                        "occasion": "casual",
                    },
                    "preference_boosts": [
                        {"attribute": "colour_group_name", "value": "White",   "weight": 0.70},
                        {"attribute": "product_type_name", "value": "Shirt",   "weight": 0.65},
                        {"attribute": "index_group_name",  "value": "Menswear","weight": 0.50},
                    ],
                    "penalties": {},
                    "purchase_history_hints": {
                        "top_colours":           ["White", "Black", "Grey"],
                        "top_product_types":     ["Shirt", "Trousers", "T-shirt"],
                        "inferred_gender":       "Menswear",
                        "budget_tier":           "low",
                        "preferred_price_range": [0.01, 0.06],
                        "dominant_colour":       "White",
                        "dominant_type":         "Shirt",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {
                        "colour_group_name": "White",
                        "product_type_name": "Shirt",
                        "index_group_name":  "Menswear",
                    },
                    "soft_constraints": {"occasion": "casual"},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "White casual shirt for men",
                },
                "long_term_preferences": [
                    {"attribute_name": "colour_group_name", "attribute_value": "White",   "weight": 0.70},
                    {"attribute_name": "product_type_name", "attribute_value": "Shirt",   "weight": 0.65},
                    {"attribute_name": "index_group_name",  "attribute_value": "Menswear","weight": 0.50},
                ],
                "style_profile":        {"dominant_style": "casual", "gender": "Menswear"},
                "preference_summary":   {},
                "existing_explanation": None,
            },
        },
    },

    # ── Case 6: CLIP Visual Terms (Improvement 5) ─────────────────────────────
    # User says "clothes for a beach holiday" — no specific colour/type extracted.
    # KB get_clip_terms() injects beach-specific visual vocabulary into the CLIP query,
    # pulling swimwear/shorts/tropical items to the top of FAISS retrieval.
    {
        "name": "INITIAL_REQUEST — clothes for a beach holiday",
        "description": (
            "User asks for beach holiday clothes without naming a specific colour or type. "
            "M3 extracts no hard constraints. "
            "KB CLIP Terms (Improvement 5) enriches the FAISS query with beach visual vocabulary."
        ),
        "kb_improvement": "Improvement 5: get_clip_terms() — CLIP-optimised visual query enrichment",
        "expected": {
            "avoid_colours": ["Black", "Dark Grey"],
        },
        "payload": {
            "retrieval_input": {
                "action":             "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message":       "I need clothes for a beach holiday",
                "items_in_context":   {"item_a": None, "item_b": None},
                "exclude_ids":        [],
                "payload": {
                    "filters": {},
                    "soft_constraints": {
                        "occasion": "beach",
                    },
                    "preference_boosts": [
                        {"attribute": "colour_group_name", "value": "White",  "weight": 0.50},
                        {"attribute": "colour_group_name", "value": "Yellow", "weight": 0.40},
                    ],
                    "penalties": {
                        "colour_group_name": ["Black", "Dark Grey"],
                    },
                    "purchase_history_hints": {
                        "top_colours":           ["White", "Yellow", "Turquoise"],
                        "top_product_types":     ["Shorts", "T-shirt", "Swimsuit"],
                        "inferred_gender":       "Ladieswear",
                        "budget_tier":           "low",
                        "preferred_price_range": [0.01, 0.06],
                        "dominant_colour":       "White",
                        "dominant_type":         "Shorts",
                    },
                },
            },
            "memory_context": {
                "dialogue_state": {
                    "hard_constraints": {},
                    "soft_constraints": {"occasion": "beach"},
                    "rejected_items":   [],
                    "accepted_items":   [],
                    "intent_summary":   "Beach holiday clothing",
                },
                "long_term_preferences": [],
                "style_profile":        {"dominant_style": "casual", "gender": "Ladieswear"},
                "preference_summary":   {},
                "existing_explanation": None,
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
# Display helpers
# =============================================================================

def _print_payload_summary(ri: dict):
    payload = ri.get("payload", {})
    hints   = payload.get("purchase_history_hints", {})
    print("\n  --- M3 Payload Summary ---")
    print(f"  action           : {ri.get('action', 'unknown')}")
    print(f"  user_message     : {ri.get('user_message', '')}")
    print(f"  filters          : {payload.get('filters', {})}")
    print(f"  soft_constraints : {payload.get('soft_constraints', {})}")
    print(f"  penalties        : {payload.get('penalties', {})}")
    print(f"  pref_boosts      : {len(payload.get('preference_boosts', []))} boosts")
    print(f"  purchase hints   : dominant={hints.get('dominant_colour')}/{hints.get('dominant_type')}  "
          f"budget={hints.get('budget_tier')}  gender={hints.get('inferred_gender')}")
    print(f"  exclude_ids      : {ri.get('exclude_ids', [])}")
    ctx = ri.get("items_in_context", {})
    if ctx.get("item_a"):
        print(f"  item_a in context: {ctx['item_a'].get('prod_name')} "
              f"({ctx['item_a'].get('colour_group_name')})")


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
        print(f"  Appear  : {item.get('graphical_appearance_name', '?')}")
        print(f"  Gender  : {item.get('index_group_name', '?')}")
        score = item.get("score")
        if score is not None:
            print(f"  Score   : {score:.4f}")
        if item.get("explanation"):
            print(f"  Why     : {item['explanation'][:100]}...")


# =============================================================================
# Accuracy evaluation (deterministic, zero-LLM)
# =============================================================================

_REQUIRED_ITEM_FIELDS = ["article_id", "prod_name", "colour_group_name", "product_type_name"]
_MIN_SIMILARITY_SCORE = 0.10


def _check(checks: dict, name: str, passed: bool, detail: str):
    checks[name] = (passed, detail)


def _grade(pct: float) -> str:
    if pct >= 90: return "EXCELLENT"
    if pct >= 75: return "GOOD"
    if pct >= 50: return "FAIR"
    return "POOR"


def evaluate_accuracy(ri: dict, result: dict, expected: dict):
    """Deterministic KB accuracy evaluation — no LLM required."""
    items    = result.get("items", [])
    checks: dict = {}

    # ── Universal checks ──────────────────────────────────────────────────────
    _check(checks, "api_success",     result.get("success", False),   "M2 returned success=True")
    _check(checks, "has_results",     len(items) > 0,                  f"{len(items)} items returned")

    if items:
        complete = sum(1 for it in items if all(it.get(f) for f in _REQUIRED_ITEM_FIELDS))
        _check(checks, "field_completeness",
               complete == len(items),
               f"{complete}/{len(items)} items have all required fields")

    # Similarity scores above noise floor
    if items:
        scores  = [it.get("score", 0) for it in items if it.get("score") is not None]
        avg_scr = sum(scores) / len(scores) if scores else 0
        _check(checks, "score_quality",
               avg_scr >= _MIN_SIMILARITY_SCORE,
               f"avg similarity={avg_scr:.4f}  (threshold={_MIN_SIMILARITY_SCORE})")

    # No excluded items leaked through
    exc_ids      = set(ri.get("exclude_ids", []))
    returned_ids = {it.get("article_id") for it in items}
    leaked       = returned_ids & exc_ids
    _check(checks, "exclude_compliance",
           len(leaked) == 0,
           f"0 excluded IDs leaked" if not leaked else f"LEAKED: {leaked}")

    # All items have KB-grounded explanations
    with_exp = sum(1 for it in items if it.get("explanation"))
    _check(checks, "explanations_present",
           with_exp == len(items),
           f"{with_exp}/{len(items)} items have explanation")

    # Response text references returned product names
    resp  = (result.get("response_text") or "").lower()
    named = sum(1 for it in items if (it.get("prod_name") or "").lower() in resp)
    _check(checks, "response_mentions_items",
           named > 0,
           f"{named}/{len(items)} item names mentioned in response")

    # ── KB-specific checks ────────────────────────────────────────────────────
    avoid = expected.get("avoid_colours", [])
    if avoid and items:
        leaked_c = [it for it in items if it.get("colour_group_name") in avoid]
        _check(checks, "kb_penalised_colours_not_returned",
               len(leaked_c) == 0,
               f"0 penalised colours leaked" if not leaked_c
               else f"LEAKED: {[it['colour_group_name'] for it in leaked_c]}")

    pref = expected.get("preferred_colours", [])
    if pref and items:
        hit = any(it.get("colour_group_name") in pref for it in items)
        _check(checks, "kb_preferred_colours_returned",
               hit, f"at least 1 item with colour in {pref}")

    pref_app = expected.get("preferred_appearances", [])
    if pref_app and items:
        hit_app = any(it.get("graphical_appearance_name") in pref_app for it in items)
        _check(checks, "kb_preferred_appearance_returned",
               hit_app, f"at least 1 item with appearance in {pref_app}")

    # ── Report ────────────────────────────────────────────────────────────────
    passed = sum(1 for ok, _ in checks.values() if ok)
    total  = len(checks)
    pct    = passed / total * 100 if total else 0
    grade  = _grade(pct)

    print(f"\n  {'─'*56}")
    print(f"  [ACCURACY]  KB Evaluation  ({ri.get('action', '?')})")
    print(f"  {'─'*56}")
    for name, (ok, detail) in checks.items():
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  {name:<38}  {detail}")
    print(f"  {'─'*56}")
    print(f"  Score : {passed}/{total} checks passed  ({pct:.0f}%)  [{grade}]")
    print(f"  {'─'*56}\n")

    return passed, total


# =============================================================================
# Run one integration case
# =============================================================================

def run_case(case: dict, print_json: bool = False) -> tuple:
    print(f"\n{'='*62}")
    print(f"  TEST  : {case['name']}")
    print(f"  {case['description']}")
    print(f"  KB    : {case['kb_improvement']}")

    ri = case["payload"]["retrieval_input"]
    _print_payload_summary(ri)

    if print_json:
        print("\n  Full JSON payload:")
        print(json.dumps(case["payload"], indent=4))

    print(f"\n{'='*62}")
    print(f"  Sending POST to {M2_URL} ...")

    try:
        result = send_request(case["payload"])
    except requests.exceptions.ConnectionError:
        print("\n  [ERROR] Cannot connect to M2 server.")
        print("  Start it with: uvicorn m2_multimodal_rag.backend.main:app --host 0.0.0.0 --port 8001")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return 0, 1

    print_response(result)

    if result.get("success"):
        passed, total = evaluate_accuracy(ri, result, case.get("expected", {}))
    else:
        passed, total = 0, 1

    print(f"{'='*62}")
    return passed, total


# =============================================================================
# Run all integration cases
# =============================================================================

def run_integration_tests(case_num: int = None, print_json: bool = False) -> int:
    print("\n" + "=" * 62)
    print("  PART 2 — KB Integration Tests  (requires M2 server at :8001)")
    print(f"  Target : {M2_URL}")
    print(f"  Cases  : {len(KB_CASES)}  (1 occasion, 2 appearance, 3 gender, 4 kansei, 5 harmony, 6 clip)")
    print("=" * 62)

    if case_num is not None:
        idx = case_num - 1
        if idx < 0 or idx >= len(KB_CASES):
            print(f"[ERROR] Case must be 1–{len(KB_CASES)}.")
            return 1
        p, t = run_case(KB_CASES[idx], print_json)
        return 0 if p == t else 1

    total_pass = total_checks = 0
    for case in KB_CASES:
        p, t = run_case(case, print_json)
        total_pass   += p
        total_checks += t

    pct   = total_pass / total_checks * 100 if total_checks else 0
    grade = _grade(pct)
    print(f"\n{'='*62}")
    print(f"  INTEGRATION SUMMARY : {total_pass}/{total_checks} checks  ({pct:.0f}%)  [{grade}]")
    print(f"{'='*62}\n")
    return 0 if total_pass == total_checks else 1


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base test script — unit + M3-style integration"
    )
    parser.add_argument("--unit-only",  action="store_true",
                        help="Run unit tests only (no server needed)")
    parser.add_argument("--integ-only", action="store_true",
                        help="Run integration tests only (skip unit tests)")
    parser.add_argument("--case",       type=int, default=None,
                        help="Run a specific integration case (1–6)")
    parser.add_argument("--json",       action="store_true",
                        help="Print the full JSON payload sent to M2")
    args = parser.parse_args()

    print("=" * 62)
    print("  Knowledge Base Test Script — NOVELTY 5")
    print("  Sends M3-format payloads to M2's /api/process endpoint")
    print("=" * 62)

    failures = 0

    if not args.integ_only:
        failures += run_unit_tests()

    if not args.unit_only:
        failures += run_integration_tests(
            case_num=args.case,
            print_json=args.json,
        )

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
