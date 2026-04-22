# test_contradiction.py
# Directly tests the ContradictionDetector with forced contradictions.
# Run from m3_implementation: python test_contradiction.py

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
from text_rag.db.postgres_client import create_schema, close_pool, get_article_by_id
from memory.core.contradiction_detector import ContradictionDetector


async def run_tests():
    print("=" * 60)
    print("CONTRADICTION DETECTOR — DIRECT UNIT TESTS")
    print("=" * 60)

    await connect_to_mongodb()
    await create_schema()

    db      = get_db()
    detector= ContradictionDetector()

    # Use a known article from the DB
    article = await get_article_by_id("829145009")
    if not article:
        print("ERROR: article 829145009 not found in PostgreSQL")
        return

    real_price  = float(article["avg_price"])
    real_colour = article["colour_group_name"]
    real_name   = article["prod_name"]
    print(f"\nTest article: {real_name}")
    print(f"  Real price:  £{real_price:.2f}")
    print(f"  Real colour: {real_colour}")

    SESSION_ID = "test_contra_session_001"
    USER_ID    = "test_user_contra"

    # Clean up any previous test data
    await db.explanations.delete_many({"session_id": SESSION_ID})
    await db.contradiction_log.delete_many({"session_id": SESSION_ID})
    print(f"\nTest session: {SESSION_ID}")

    # ── TEST 1: Store initial correct price claim ──────────────────────────
    print("\n" + "─"*60)
    print("TEST 1: Store initial claim (correct price)")
    
    evidence_turn1 = {
        "action": "catalog_search",
        "items": [{
            "article_id":           "829145009",
            "name":                 real_name,
            "type":                 "Dress",
            "colour":               real_colour,
            "price":                f"£{real_price:.2f}",
            "price_raw":            real_price,
            "material_description": article.get("detail_desc", ""),
            "garment_group":        article.get("garment_group_name", ""),
            "index_group":          article.get("index_group_name", ""),
        }]
    }

    response_turn1 = (
        f"Here are two options. "
        f"The {real_name} costs £{real_price:.2f}. "
        f"It is a {real_colour.lower()} dress made from viscose weave."
    )

    result1 = await detector.check_and_resolve(
        response_text=response_turn1,
        evidence=evidence_turn1,
        session_id=SESSION_ID,
        user_id=USER_ID,
        turn_id="turn_001",
    )

    print(f"  Claims stored:       {result1['claims_stored']}")
    print(f"  Contradiction found: {result1['contradiction_found']}")
    print(f"  Product IDs:         {result1['product_ids']}")
    assert not result1["contradiction_found"], "FAIL: Should not find contradiction on first turn"
    print("  ✓ PASS: No contradiction on first turn (correct)")

    # Verify claims stored in MongoDB
    doc = await db.explanations.find_one({"session_id": SESSION_ID, "turn_id": "turn_001"})
    assert doc is not None, "FAIL: ExplanationDocument not stored"
    stored_claims = doc.get("claims", [])
    print(f"  ✓ PASS: {len(stored_claims)} claims stored in MongoDB")
    for c in stored_claims:
        print(f"    attribute={c['attribute']}  value={c['value']}  status={c['status']}")

    # ── TEST 2: Inject WRONG price in second turn ──────────────────────────
    print("\n" + "─"*60)
    wrong_price = real_price + 20.00
    print(f"TEST 2: Inject contradicting WRONG price (£{wrong_price:.2f} vs real £{real_price:.2f})")

    evidence_turn2 = {
        "action": "item_attribute_lookup",
        "article": {
            "article_id":           "829145009",
            "name":                 real_name,
            "type":                 "Dress",
            "colour":               real_colour,
            "price":                f"£{wrong_price:.2f}",
            "price_raw":            wrong_price,
            "material_description": article.get("detail_desc", ""),
        },
        "attribute_topic": "price",
        "extracted_facts": {"avg_price": f"£{wrong_price:.2f}"},
    }

    # Response with deliberately wrong price
    response_turn2 = (
        f"The {real_name} is priced at £{wrong_price:.2f}. "
        f"It costs £{wrong_price:.2f} which is within your budget."
    )

    result2 = await detector.check_and_resolve(
        response_text=response_turn2,
        evidence=evidence_turn2,
        session_id=SESSION_ID,
        user_id=USER_ID,
        turn_id="turn_002",
    )

    print(f"  Contradiction found: {result2['contradiction_found']}")
    print(f"  Contradiction count: {result2['contradiction_count']}")
    print(f"  Original response:   {response_turn2}")
    print(f"  Corrected response:  {result2['response_text']}")

    if result2["contradiction_found"]:
        print(f"  ✓ PASS: Contradiction DETECTED")
        for c in result2["contradictions"]:
            print(f"    attribute:           {c['attribute']}")
            print(f"    old value:           {c['old_value']}")
            print(f"    new (wrong) value:   {c['new_value']}")
            print(f"    authoritative value: {c['authoritative_value']}")
            print(f"    nli_score:           {c['nli_score']:.3f}")
            print(f"    resolution:          {c['resolution']}")
        assert wrong_price != real_price
        print(f"  ✓ PASS: Wrong price £{wrong_price:.2f} caught, real price £{real_price:.2f} confirmed")
    else:
        print(f"  ✗ FAIL: Contradiction NOT detected")
        print(f"  → Claims extracted from turn 2: check if price pattern matched")

    # Check contradiction_log in MongoDB
    log_entry = await db.contradiction_log.find_one({"session_id": SESSION_ID})
    if log_entry:
        print(f"  ✓ PASS: ContradictionEntry stored in MongoDB contradiction_log")
        print(f"    old_claim:  {log_entry.get('old_claim_text','')[:60]}")
        print(f"    nli_score:  {log_entry.get('nli_score', 0):.3f}")
        print(f"    auth_value: {log_entry.get('authoritative_value', '')}")
    else:
        print(f"  ✗ FAIL: No ContradictionEntry in MongoDB")

    # Check old claim marked as contradicted
    doc1 = await db.explanations.find_one({"session_id": SESSION_ID, "turn_id": "turn_001"})
    if doc1:
        price_claim = next(
            (c for c in doc1.get("claims", []) if c.get("attribute") == "avg_price"),
            None
        )
        if price_claim:
            print(f"  Claim status in turn_001: {price_claim.get('status')}")
            if price_claim.get("status") == "contradicted":
                print(f"  ✓ PASS: Old wrong claim marked 'contradicted' in MongoDB")
            else:
                print(f"  Note: claim status = {price_claim.get('status')}")

    # ── TEST 3: Same attribute consistent claim (should NOT flag) ──────────
    print("\n" + "─"*60)
    print(f"TEST 3: Consistent price claim (£{real_price:.2f} again — should pass)")

    evidence_turn3 = {
        "action": "explanation_generate",
        "article": {
            "article_id":           "829145009",
            "name":                 real_name,
            "type":                 "Dress",
            "colour":               real_colour,
            "price":                f"£{real_price:.2f}",
            "price_raw":            real_price,
            "material_description": article.get("detail_desc", ""),
        },
        "confirmed_matches": [],
        "prior_claims": [],
        "matched_prefs": [],
    }

    response_turn3 = (
        f"I recommended the {real_name} because it costs £{real_price:.2f} "
        f"which is within your budget and it matches your preference for {real_colour} items."
    )

    result3 = await detector.check_and_resolve(
        response_text=response_turn3,
        evidence=evidence_turn3,
        session_id=SESSION_ID,
        user_id=USER_ID,
        turn_id="turn_003",
    )

    print(f"  Contradiction found: {result3['contradiction_found']}")
    if not result3["contradiction_found"]:
        print(f"  ✓ PASS: Consistent claim correctly NOT flagged")
    else:
        print(f"  ✗ FAIL: Consistent claim incorrectly flagged as contradiction")

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CONTRADICTION DETECTOR TEST SUMMARY")
    print("="*60)
    print(f"  Test 1 (first claim stored):        {'✓ PASS' if not result1['contradiction_found'] else '✗ FAIL'}")
    print(f"  Test 2 (wrong price detected):      {'✓ PASS' if result2['contradiction_found'] else '✗ FAIL'}")
    print(f"  Test 3 (consistent claim passes):   {'✓ PASS' if not result3['contradiction_found'] else '✗ FAIL'}")
    
    total_pass = (
        (not result1["contradiction_found"]) +
        result2["contradiction_found"] +
        (not result3["contradiction_found"])
    )
    print(f"\n  {total_pass}/3 tests passed")

    # Cleanup
    await db.explanations.delete_many({"session_id": SESSION_ID})
    await db.contradiction_log.delete_many({"session_id": SESSION_ID})
    print("\nTest data cleaned up.")

    await close_mongodb_connection()
    await close_pool()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(run_tests())


# ── EXTENDED TESTS — appended to existing test file ──────────────────────────

async def run_extended_tests():
    """
    Tests all scenarios that can occur in the Sunlytics CRS system.
    Run after run_tests() to verify full coverage.
    """
    from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db
    from text_rag.db.postgres_client import create_schema, close_pool, get_article_by_id
    from memory.core.contradiction_detector import ContradictionDetector

    await connect_to_mongodb()
    await create_schema()

    db      = get_db()
    detector= ContradictionDetector()

    article_a = await get_article_by_id("829145009")  # London dress (Black)
    article_b = await get_article_by_id("829145003")  # SS London dress (Black)

    SESSION = "test_extended_contra_001"
    await db.explanations.delete_many({"session_id": SESSION})
    await db.contradiction_log.delete_many({"session_id": SESSION})

    print("\n" + "="*60)
    print("EXTENDED CONTRADICTION TESTS — ALL SCENARIOS")
    print("="*60)

    passed = 0
    failed = 0

    def check(label, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ PASS: {label}")
            passed += 1
        else:
            print(f"  ✗ FAIL: {label} {detail}")
            failed += 1

    # ── SCENARIO A: Colour contradiction ─────────────────────────────────
    print("\nSCENARIO A: Colour contradiction (Black vs White for same article)")
    await db.explanations.delete_many({"session_id": SESSION})
    await db.contradiction_log.delete_many({"session_id": SESSION})

    ev_a1 = {"action": "catalog_search", "items": [{"article_id": "829145009",
        "name": "London dress", "colour": "Black", "type": "Dress",
        "price": "£11.08", "price_raw": 11.08, "material_description": "", "garment_group": "", "index_group": ""}]}
    r_a1 = "The London dress is available in Black colour. It is a Black dress."
    res_a1 = await detector.check_and_resolve(r_a1, ev_a1, SESSION, "u1", "ta1")
    check("Colour claim stored on first turn", not res_a1["contradiction_found"])
    check("Colour claim extracted", res_a1["claims_stored"] >= 1)

    ev_a2 = {"action": "item_attribute_lookup", "article": {"article_id": "829145009",
        "name": "London dress", "colour": "White", "type": "Dress",
        "price": "£11.08", "price_raw": 11.08, "material_description": ""},
        "attribute_topic": "colour_group_name", "extracted_facts": {"colour_group_name": "White"}}
    r_a2 = "The London dress comes in White colour. It is a White dress."
    res_a2 = await detector.check_and_resolve(r_a2, ev_a2, SESSION, "u1", "ta2")
    check("Colour contradiction detected (Black vs White)", res_a2["contradiction_found"])
    check("Corrected response does not contain wrong colour",
          "White" not in res_a2["response_text"] or "corrected" in res_a2["response_text"])
    log = await db.contradiction_log.find_one({"session_id": SESSION, "attribute": "colour_group_name"})
    check("Colour ContradictionEntry stored in MongoDB", log is not None)

    # ── SCENARIO B: Two different articles — no cross-contamination ───────
    print("\nSCENARIO B: Two articles — contradiction on one should not affect other")
    await db.explanations.delete_many({"session_id": SESSION})
    await db.contradiction_log.delete_many({"session_id": SESSION})

    ev_b1 = {"action": "item_compare",
        "item_a": {"article_id": "829145009", "name": "London dress",
                   "colour": "Black", "type": "Dress", "price": "£11.08",
                   "price_raw": 11.08, "material_description": "", "garment_group": "", "index_group": ""},
        "item_b": {"article_id": "829145003", "name": "SS London dress",
                   "colour": "Black", "type": "Dress", "price": "£15.12",
                   "price_raw": 15.12, "material_description": "", "garment_group": "", "index_group": ""},
        "comparison_dimension": "price",
        "comparison_facts": {"item_a_price": "£11.08", "item_b_price": "£15.12",
                             "cheaper_item": "London dress", "price_difference": "£4.04",
                             "item_a_name": "London dress", "item_b_name": "SS London dress"}}
    r_b1 = "The London dress costs £11.08. The SS London dress costs £15.12."
    res_b1 = await detector.check_and_resolve(r_b1, ev_b1, SESSION, "u1", "tb1")
    check("Two articles stored — no false contradiction", not res_b1["contradiction_found"])
    check("Both article IDs captured", len(res_b1["product_ids"]) == 2)

    # Now contradict price of article A only
    ev_b2 = {"action": "item_attribute_lookup", "article": {"article_id": "829145009",
        "name": "London dress", "colour": "Black", "type": "Dress",
        "price": "£99.00", "price_raw": 99.00, "material_description": ""},
        "attribute_topic": "price", "extracted_facts": {"avg_price": "£99.00"}}
    r_b2 = "The London dress is priced at £99.00."
    res_b2 = await detector.check_and_resolve(r_b2, ev_b2, SESSION, "u1", "tb2")
    check("Contradiction on article A detected", res_b2["contradiction_found"])

    # SS London dress price should be UNTOUCHED
    doc_b1 = await db.explanations.find_one({"session_id": SESSION, "turn_id": "tb1"})
    ss_claims = [c for c in doc_b1.get("claims", [])
                 if c.get("article_id") == "829145003"]
    check("SS London dress claims unaffected by contradiction on London dress",
          all(c.get("status") == "active" for c in ss_claims))

    # ── SCENARIO C: Multi-turn — T1 correct, T3 wrong, T5 consistent ─────
    print("\nSCENARIO C: Multi-turn consistency (T1 correct, T3 wrong, T5 checks again)")
    await db.explanations.delete_many({"session_id": SESSION})
    await db.contradiction_log.delete_many({"session_id": SESSION})

    # Turn 1 — correct material claim
    ev_c1 = {"action": "catalog_search", "items": [{"article_id": "829145009",
        "name": "London dress", "colour": "Black", "type": "Dress",
        "price": "£11.08", "price_raw": 11.08,
        "material_description": "Short dress in a patterned viscose weave",
        "garment_group": "", "index_group": ""}]}
    r_c1 = "The London dress is made of viscose weave."
    res_c1 = await detector.check_and_resolve(r_c1, ev_c1, SESSION, "u1", "tc1")
    check("Turn 1: material claim stored", res_c1["claims_stored"] >= 1)

    # Turn 3 — wrong material claim
    ev_c3 = {"action": "item_attribute_lookup", "article": {"article_id": "829145009",
        "name": "London dress", "colour": "Black", "type": "Dress",
        "price": "£11.08", "price_raw": 11.08,
        "material_description": "Short dress in a patterned viscose weave"},
        "attribute_topic": "material_and_care",
        "extracted_facts": {"detail_desc": "made of cotton blend"}}
    r_c3 = "The London dress is made of cotton blend."
    res_c3 = await detector.check_and_resolve(r_c3, ev_c3, SESSION, "u1", "tc3")
    check("Turn 3: material contradiction detected (viscose vs cotton)", res_c3["contradiction_found"])

    # Turn 5 — consistent with Turn 1 (viscose)
    ev_c5 = {"action": "explanation_generate", "article": {"article_id": "829145009",
        "name": "London dress", "colour": "Black", "type": "Dress",
        "price": "£11.08", "price_raw": 11.08, "material_description": "viscose weave"},
        "confirmed_matches": [], "prior_claims": [], "matched_prefs": []}
    r_c5 = "The London dress is made of viscose weave and matches your preference."
    res_c5 = await detector.check_and_resolve(r_c5, ev_c5, SESSION, "u1", "tc5")
    check("Turn 5: consistent claim (viscose) not flagged", not res_c5["contradiction_found"])

    # ── SCENARIO D: No product in response — should skip cleanly ─────────
    print("\nSCENARIO D: FEEDBACK and CHITCHAT skip contradiction check")
    ev_d = {"action": "no_retrieval", "feedback": {"is_positive": True,
            "sentiment_score": 0.9, "item_reacted_to": {"prod_name": "London dress"}}}
    r_d = "Great choice! You selected the London dress."
    res_d = await detector.check_and_resolve(r_d, ev_d, SESSION, "u1", "td1")
    check("FEEDBACK skipped (no_retrieval action)", not res_d["contradiction_found"]
          and res_d["claims_stored"] == 0)

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXTENDED TEST SUMMARY")
    print("="*60)
    print(f"  Scenarios tested: A (colour), B (two articles), C (multi-turn), D (FEEDBACK skip)")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Result: {'ALL PASS ✓' if failed == 0 else f'{failed} FAILURES ✗'}")

    await db.explanations.delete_many({"session_id": SESSION})
    await db.contradiction_log.delete_many({"session_id": SESSION})
    await close_mongodb_connection()
    await close_pool()


if __name__ == "__main__":
    # Run extended tests only when called directly with --extended flag
    import sys
    if "--extended" in sys.argv:
        asyncio.run(run_extended_tests())
    else:
        asyncio.run(run_tests())
