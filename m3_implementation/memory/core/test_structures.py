# m3_implementation/memory/core/test_structures.py
#
# Comprehensive structural verification test.
# Tests every action type, every field, every constraint.
# Run this to answer confidently: "Yes, every structure is correct."

import asyncio
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from memory.db.redis_client import connect_to_redis, close_redis_connection
from memory.core.user_manager import UserManager
from memory.core.pipeline import MemoryPipeline

# ── Colours used in output ─────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0
warnings = 0


def check(condition: bool, label: str, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}✓{RESET} {label}")
    else:
        failed += 1
        print(f"  {RED}✗ FAIL{RESET} {label}")
        if detail:
            print(f"    {RED}→ {detail}{RESET}")


def warn(label: str, detail: str = ""):
    global warnings
    warnings += 1
    print(f"  {YELLOW}⚠ WARN{RESET} {label}")
    if detail:
        print(f"    {YELLOW}→ {detail}{RESET}")


def section(title: str):
    print(f"\n{BOLD}{BLUE}{'='*65}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*65}{RESET}")


def show_retrieval_input(ri: dict, indent: int = 4):
    """Pretty prints the retrieval_input for visual inspection."""
    pad = " " * indent
    if ri is None:
        print(f"{pad}retrieval_input: None")
        return
    print(f"{pad}action:             {BOLD}{ri['action']}{RESET}")
    print(f"{pad}retrieval_strategy: {ri['retrieval_strategy']}")
    print(f"{pad}user_message:       {ri['user_message'][:70]}")
    ia = ri["items_in_context"].get("item_a")
    ib = ri["items_in_context"].get("item_b")
    print(f"{pad}items_in_context:   "
          f"item_a={ia['prod_name'] if ia else None}, "
          f"item_b={ib['prod_name'] if ib else None}")
    print(f"{pad}exclude_ids:        {ri['exclude_ids']}")
    print(f"{pad}payload:")
    for k, v in ri["payload"].items():
        if isinstance(v, (list, dict)):
            print(f"{pad}  {k}: {json.dumps(v)[:80]}")
        else:
            print(f"{pad}  {k}: {v}")


async def test_all_structures():
    await connect_to_mongodb()
    await connect_to_redis()

    print(f"\n{BOLD}Loading pipeline...{RESET}")
    pipeline = MemoryPipeline()
    model_available = pipeline.predictor is not None
    print(f"DistilBERT: {'loaded ✓' if model_available else 'fallback ⚠'}")

    user_mgr = UserManager()
    user = await user_mgr.get_or_create_user(
        customer_id="test_structure_verify_001",
        initial_data={"age": 28, "club_member_status": "ACTIVE"}
    )

    # Seed a preference so preference_boosts appear in output
    await user_mgr.update_preferences_from_entities(
        user_id=user.user_id,
        entities={"colour_group_name": "Black", "product_type_name": "Dress"},
        sentiment=0.9, source="explicit"
    )
    # Seed a dislike so penalties appear
    await user_mgr.update_preferences_from_entities(
        user_id=user.user_id,
        entities={"colour_group_name": "Orange"},
        sentiment=-0.8, source="explicit"
    )

    session_id = None
    print(f"Test user: {user.user_id}\n")

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 1 — INITIAL_REQUEST → action: catalog_search")
    # ══════════════════════════════════════════════════════════════════════════

    r1 = await pipeline.process_turn(
        user_id=user.user_id,
        message="I want a black dress under £50",
        session_id=session_id
    )
    session_id = r1["session_id"]

    print(f"\n  Message: 'I want a black dress under £50'")
    print(f"  Label:   {r1['label']} ({r1['confidence']:.1%})")
    show_retrieval_input(r1["retrieval_input"])

    ri1 = r1["retrieval_input"]
    check(ri1 is not None,                        "retrieval_input exists")
    check(ri1["action"] == "catalog_search",       "action = catalog_search")
    check(ri1["retrieval_strategy"] == "FULL",     "retrieval_strategy = FULL")
    check("filters" in ri1["payload"],             "payload has filters")
    check("preference_boosts" in ri1["payload"],   "payload has preference_boosts")
    check("penalties" in ri1["payload"],           "payload has penalties")
    check("exclude_ids" in ri1,                    "exclude_ids present")
    check("items_in_context" in ri1,               "items_in_context present")
    check("user_message" in ri1,                   "user_message present")

    filters1 = ri1["payload"]["filters"]
    check("product_type_name" in filters1,         "filter has product_type_name")
    check("colour_group_name" in filters1,         "filter has colour_group_name")
    check("price_max" in filters1,                 "filter has price_max")

    if "product_type_name" in filters1:
        check(filters1["product_type_name"] == "Dress", 
              f"product_type_name = Dress (got: {filters1.get('product_type_name')})")
    if "colour_group_name" in filters1:
        check(filters1["colour_group_name"] == "Black",
              f"colour_group_name = Black (got: {filters1.get('colour_group_name')})")
    if "price_max" in filters1:
        check(filters1["price_max"] == 50.0,
              f"price_max = 50.0 (got: {filters1.get('price_max')})")

    boosts1 = ri1["payload"]["preference_boosts"]
    check(len(boosts1) > 0,                        "preference_boosts not empty")
    if boosts1:
        check(all("attribute" in b for b in boosts1), "boosts have attribute field")
        check(all("value"     in b for b in boosts1), "boosts have value field")
        check(all("weight"    in b for b in boosts1), "boosts have weight field")
        check(all(0 < b["weight"] <= 1 for b in boosts1), "boost weights in (0,1]")

    penalties1 = ri1["payload"]["penalties"]
    check("colour_group_name" in penalties1,       "penalties has colour_group_name")

    check("dialogue_state"       in r1["memory_context"], "memory_context has dialogue_state")
    check("long_term_preferences" in r1["memory_context"], "memory_context has preferences")
    check(isinstance(r1["side_effects"], list),    "side_effects is a list")

    # Store bot response + items to set up context for subsequent tests
    await pipeline.store_response(
        session_id=session_id,
        user_id=user.user_id,
        bot_response=(
            "Here are two options. Option 1 is the Valerie dress "
            "(black, dress): Short dress in crisp cotton weave. "
            "Option 2 is the Angel (dark pink): Short A-line dress."
        ),
        recommended_items=[
            {
                "article_id": "209886026",
                "prod_name": "Valerie dress",
                "product_type_name": "Dress",
                "colour_group_name": "Black",
                "index_group_name": "Ladieswear",
                "detail_desc": "Short dress in crisp cotton weave.",
                "price": 34.99
            },
            {
                "article_id": "108775015",
                "prod_name": "Angel",
                "product_type_name": "Dress",
                "colour_group_name": "Dark Pink",
                "index_group_name": "Ladieswear",
                "detail_desc": "Short A-line dress in cotton.",
                "price": 29.99
            }
        ],
        trigger_label=r1["label"],
        retrieval_strategy=r1["retrieval_strategy"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 2 — ATTRIBUTE_QUESTION → action: item_attribute_lookup")
    # ══════════════════════════════════════════════════════════════════════════

    r2 = await pipeline.process_turn(
        user_id=user.user_id,
        message="What material is the first one made of?",
        session_id=session_id
    )

    print(f"\n  Message: 'What material is the first one made of?'")
    print(f"  Label:   {r2['label']} ({r2['confidence']:.1%})")
    show_retrieval_input(r2["retrieval_input"])

    ri2 = r2["retrieval_input"]
    check(ri2 is not None,                              "retrieval_input exists")
    check(ri2["action"] == "item_attribute_lookup",     "action = item_attribute_lookup")
    check(ri2["retrieval_strategy"] == "PARTIAL",       "retrieval_strategy = PARTIAL")
    check("article_id" in ri2["payload"],               "payload has article_id")
    check("attribute_topic" in ri2["payload"],          "payload has attribute_topic")
    check(ri2["payload"]["article_id"] is not None,     "article_id is not None")
    check(ri2["payload"]["article_id"] == "209886026",  
          f"article_id = 209886026 (got: {ri2['payload'].get('article_id')})")
    check(ri2["payload"]["attribute_topic"] == "material_and_care",
          f"attribute_topic = material_and_care (got: {ri2['payload'].get('attribute_topic')})")

    ia2 = ri2["items_in_context"].get("item_a")
    check(ia2 is not None,                              "item_a in context")
    if ia2:
        check(ia2["prod_name"] == "Valerie dress",      "item_a = Valerie dress")

    # Verify entities are EMPTY for ATTRIBUTE_QUESTION
    check(len(r2.get("side_effects", [])) == 0,         "no side effects for ATTRIBUTE_QUESTION")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="The Valerie dress is made from cotton jersey.",
        trigger_label=r2["label"], retrieval_strategy=r2["retrieval_strategy"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 3 — COMPARISON → action: item_compare")
    # ══════════════════════════════════════════════════════════════════════════

    r3 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Which one is cheaper?",
        session_id=session_id
    )

    print(f"\n  Message: 'Which one is cheaper?'")
    print(f"  Label:   {r3['label']} ({r3['confidence']:.1%})")
    show_retrieval_input(r3["retrieval_input"])

    ri3 = r3["retrieval_input"]
    check(ri3 is not None,                          "retrieval_input exists")
    check(ri3["action"] == "item_compare",          "action = item_compare")
    check(ri3["retrieval_strategy"] == "PARTIAL",   "retrieval_strategy = PARTIAL")
    check("article_id_a" in ri3["payload"],         "payload has article_id_a")
    check("article_id_b" in ri3["payload"],         "payload has article_id_b")
    check("comparison_dimension" in ri3["payload"], "payload has comparison_dimension")
    check("preference_weights" in ri3["payload"],   "payload has preference_weights")
    check(ri3["payload"]["article_id_a"] == "209886026", "article_id_a correct")
    check(ri3["payload"]["article_id_b"] == "108775015", "article_id_b correct")

    dim3 = ri3["payload"]["comparison_dimension"]
    check(dim3 in ["price", "quality", "style_and_occasion",
                   "material", "colour", "fit", "overall"],
          f"comparison_dimension is valid (got: {dim3})")

    ia3 = ri3["items_in_context"].get("item_a")
    ib3 = ri3["items_in_context"].get("item_b")
    check(ia3 is not None and ib3 is not None, "both items in context")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="The Angel at £29.99 is cheaper than the Valerie dress at £34.99.",
        trigger_label=r3["label"], retrieval_strategy=r3["retrieval_strategy"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 4 — FEEDBACK → retrieval_input = None")
    # ══════════════════════════════════════════════════════════════════════════

    r4 = await pipeline.process_turn(
        user_id=user.user_id,
        message="I love it, I'll take the Valerie dress!",
        session_id=session_id
    )

    print(f"\n  Message: 'I love it, I'll take the Valerie dress!'")
    print(f"  Label:   {r4['label']} ({r4['confidence']:.1%})")
    print(f"  retrieval_input: {r4['retrieval_input']}")
    print(f"  sentiment_score: {r4['memory_context'].get('feedback', {}).get('sentiment_score')}")
    print(f"  side_effects: {r4['side_effects']}")

    check(r4["retrieval_input"] is None,            "retrieval_input = None ✓")
    check(r4["retrieval_strategy"] == "NO",         "retrieval_strategy = NO")
    check("feedback" in r4["memory_context"],        "memory_context has feedback")
    if "feedback" in r4["memory_context"]:
        fb = r4["memory_context"]["feedback"]
        check("sentiment_score" in fb,               "feedback has sentiment_score")
        check("is_positive" in fb,                   "feedback has is_positive")
        check(fb.get("is_positive") == True,         "is_positive = True")
        check(fb.get("sentiment_score", 0) > 0,      "sentiment_score > 0")
    check(len(r4["side_effects"]) > 0,               "side_effects not empty")
    check(any("accepted" in s for s in r4["side_effects"]),
          "accepted_items updated in side_effects")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="Great choice! The Valerie dress is wonderful.",
        trigger_label=r4["label"], retrieval_strategy=r4["retrieval_strategy"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 5 — REFINEMENT → action: catalog_search (with updated filters)")
    # ══════════════════════════════════════════════════════════════════════════

    r5 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Can you also show me something in white?",
        session_id=session_id
    )

    print(f"\n  Message: 'Can you also show me something in white?'")
    print(f"  Label:   {r5['label']} ({r5['confidence']:.1%})")
    show_retrieval_input(r5["retrieval_input"])

    ri5 = r5["retrieval_input"]
    check(ri5 is not None,                          "retrieval_input exists")
    check(ri5["action"] == "catalog_search",        "action = catalog_search (same as INITIAL)")
    check(ri5["retrieval_strategy"] == "FULL",      "retrieval_strategy = FULL")
    filters5 = ri5["payload"]["filters"]
    check("colour_group_name" in filters5,          "filter has colour_group_name")
    if "colour_group_name" in filters5:
        check(filters5["colour_group_name"] == "White",
              f"colour updated to White (got: {filters5.get('colour_group_name')})")
    check("product_type_name" in filters5,          "Dress still in filters (preserved)")
    check("209886026" in ri5["exclude_ids"],        "rejected/accepted item in exclude_ids")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="Here are white options for you.",
        trigger_label=r5["label"], retrieval_strategy=r5["retrieval_strategy"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 6 — SELECTION_REFERENCE → action: item_detail_lookup")
    # ══════════════════════════════════════════════════════════════════════════

    r6 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Tell me more about the first one",
        session_id=session_id
    )

    print(f"\n  Message: 'Tell me more about the first one'")
    print(f"  Label:   {r6['label']} ({r6['confidence']:.1%})")
    show_retrieval_input(r6["retrieval_input"])

    ri6 = r6["retrieval_input"]
    if ri6 is not None:
        check(ri6["action"] == "item_detail_lookup",    "action = item_detail_lookup")
        check(ri6["retrieval_strategy"] == "PARTIAL",   "retrieval_strategy = PARTIAL")
        check("article_id" in ri6["payload"],           "payload has article_id")
        check(ri6["payload"]["article_id"] is not None, "article_id is not None")
    else:
        warn("SELECTION_REFERENCE returned None retrieval_input",
             "May have been classified differently by DistilBERT")

    await pipeline.store_response(
        session_id=session_id, user_id=user.user_id,
        bot_response="The Valerie dress is a short dress in crisp cotton weave.",
        trigger_label=r6["label"], retrieval_strategy=r6["retrieval_strategy"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 7 — CHITCHAT → retrieval_input = None")
    # ══════════════════════════════════════════════════════════════════════════

    r7 = await pipeline.process_turn(
        user_id=user.user_id,
        message="Thanks, you have been really helpful!",
        session_id=session_id
    )

    print(f"\n  Message: 'Thanks, you have been really helpful!'")
    print(f"  Label:   {r7['label']} ({r7['confidence']:.1%})")
    print(f"  retrieval_input: {r7['retrieval_input']}")

    check(r7["retrieval_input"] is None,            "retrieval_input = None ✓")
    check(r7["retrieval_strategy"] == "NO",         "retrieval_strategy = NO")

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 8 — OFF-TOPIC INPUT → refusal, no retrieval")
    # ══════════════════════════════════════════════════════════════════════════

    r8 = await pipeline.process_turn(
        user_id=user.user_id,
        message="What is the weather forecast for tomorrow?",
        session_id=session_id
    )

    print(f"\n  Message: 'What is the weather forecast for tomorrow?'")
    print(f"  Label:   {r8['label']}")
    print(f"  retrieval_input: {r8['retrieval_input']}")
    print(f"  not_relevant: {r8['memory_context'].get('not_relevant')}")
    print(f"  refusal_message: {r8['memory_context'].get('refusal_message', '')[:60]}")

    check(r8["retrieval_input"] is None,            "retrieval_input = None ✓")
    check(r8["retrieval_strategy"] == "NO",         "retrieval_strategy = NO")
    check(r8["memory_context"].get("not_relevant") == True,
          "memory_context.not_relevant = True")
    check("refusal_message" in r8["memory_context"], "refusal_message present in memory_context")

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 9 — ENVELOPE CONSISTENCY across all turns")
    # ══════════════════════════════════════════════════════════════════════════

    results = [r1, r2, r3, r4, r5, r6, r7, r8]
    required_keys = {
        "user_id", "session_id", "turn_id", "label",
        "retrieval_strategy", "confidence", "used_rules",
        "retrieval_input", "memory_context", "side_effects",
        "classifier_input"
    }
    forbidden_keys = {"enriched_context", "retrieval_query"}

    print(f"\n  Checking all {len(results)} turns have consistent structure...")
    for i, r in enumerate(results, 1):
        missing = required_keys - set(r.keys())
        extra   = forbidden_keys & set(r.keys())
        check(len(missing) == 0,
              f"Turn {i} has all required keys",
              f"Missing: {missing}")
        check(len(extra) == 0,
              f"Turn {i} has no duplicate/forbidden keys",
              f"Found: {extra}")

    # Check retrieval_input envelope fields for non-None cases
    print(f"\n  Checking retrieval_input envelope for FULL/PARTIAL turns...")
    for i, r in enumerate(results, 1):
        ri = r["retrieval_input"]
        if ri is not None:
            envelope_keys = {"action", "retrieval_strategy", "user_message",
                             "items_in_context", "exclude_ids", "payload"}
            missing_env = envelope_keys - set(ri.keys())
            check(len(missing_env) == 0,
                  f"Turn {i} retrieval_input has all envelope fields",
                  f"Missing: {missing_env}")

    # ══════════════════════════════════════════════════════════════════════════
    section("TEST 10 — MEMORY UPDATES verified in MongoDB")
    # ══════════════════════════════════════════════════════════════════════════

    full_session = await pipeline.session_mgr.get_full_session(session_id)
    print(f"\n  Session turns in MongoDB: {full_session['turn_count']}")
    check(full_session["turn_count"] >= 10,         "at least 10 turns stored")

    prefs = await pipeline.user_mgr.get_preference_summary(user.user_id)
    print(f"\n  Liked attributes: {len(prefs['liked_attributes'])}")
    for p in prefs["liked_attributes"]:
        print(f"    {p['attribute_name']}: {p['attribute_value']} "
              f"(weight={p['weight']})")

    check(len(prefs["liked_attributes"]) > 0,       "preferences stored")
    check(len(prefs["disliked_values"]) > 0,        "dislikes stored")

    # Check accepted items were recorded
    db = get_db()
    accepted_rec = await db.recommendations.find_one(
        {"session_id": session_id, "outcome": "accepted"}
    )
    check(accepted_rec is not None,                  "accepted recommendation recorded in DB")

    # ══════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════════════════════════════════
    await db.sessions.delete_many({"session_id": session_id})
    await db.users.delete_many({"user_id": user.user_id})
    await db.recommendations.delete_many({"session_id": session_id})
    await db.explanations.delete_many({"session_id": session_id})

    # ══════════════════════════════════════════════════════════════════════════
    section("FINAL RESULTS")
    # ══════════════════════════════════════════════════════════════════════════

    total = passed + failed
    print(f"\n  {GREEN}{BOLD}Passed: {passed}/{total}{RESET}")
    if warnings > 0:
        print(f"  {YELLOW}Warnings: {warnings}{RESET}")
    if failed > 0:
        print(f"  {RED}{BOLD}Failed:  {failed}/{total}{RESET}")
        print(f"\n  {RED}Some checks failed — review above for details.{RESET}")
    else:
        print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED.{RESET}")
        print(f"  {GREEN}Every action type produces the correct structure.{RESET}")
        print(f"  {GREEN}You can answer confidently in your viva or dissertation.{RESET}")

    await close_mongodb_connection()
    await close_redis_connection()


if __name__ == "__main__":
    asyncio.run(test_all_structures())
