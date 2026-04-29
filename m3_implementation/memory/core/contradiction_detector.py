# m3_implementation/memory/core/contradiction_detector.py
#
# Contradiction Detector — Novel Research Contribution
#
# PURPOSE:
#   Ensures every bot response is consistent with all previous claims
#   made during the same session. Catches three types of contradictions:
#
#   TYPE 1 — Same product, same attribute, different value
#     e.g. "London dress costs £11" then later "London dress costs £15"
#
#   TYPE 2 — Preference claim contradiction
#     e.g. "recommended because you prefer Black" then later
#          "recommended because you prefer White" (if preference unchanged)
#
#   TYPE 3 — Cross-turn factual inconsistency
#     e.g. "made of cotton" then "made of polyester" for same item
#
# HOW IT WORKS:
#   1. Extract atomic claims from bot response using regex patterns
#   2. Load all active claims for same article_ids from MongoDB
#   3. For each new claim: check if same article + same attribute
#      has an existing claim with a different value
#   4. If candidate contradiction found: run NLI to confirm
#      (NLI contradiction score > NLI_CONTRADICTION_THRESHOLD = 0.70)
#   5. If confirmed: query PostgreSQL for authoritative truth
#   6. Mark wrong claim as "contradicted" in MongoDB
#   7. Correct the response text to use the authoritative value
#   8. Store ExplanationDocument (claims) and ContradictionEntry (events)
#   9. Return corrected response + full contradiction report
#
# INTEGRATION:
#   Called from text_rag/core/rag_pipeline.py after hallucination check
#   Only runs for actions that make factual product claims:
#   catalog_search, item_attribute_lookup, item_compare, explanation_generate
#
# OUTPUT STRUCTURE:
#   {
#     "response_text":          str   — corrected response (or original if no contradiction)
#     "contradiction_found":    bool
#     "contradiction_count":    int
#     "contradictions":         list  — details of each contradiction found
#     "claims_stored":          int   — number of new claims stored
#     "product_ids":            list  — article_ids mentioned in this response
#     "product_names":          list  — product names mentioned in this response
#   }

import re
import os
import sys
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.db.mongo import get_db
from text_rag.db.postgres_client import get_article_by_id
from text_rag.config import NLI_CONTRADICTION_THRESHOLD

# ── NLI model (shared with hallucination checker) ─────────────────────────────
_nli_model = None

def _get_nli():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        from text_rag.config import NLI_MODEL_NAME
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
        print("[ContradictionDetector] NLI model loaded.")
    return _nli_model


# ── Claim extraction patterns ──────────────────────────────────────────────────
# Each pattern extracts (attribute_name, value) from a sentence.
# Ordered by specificity — more specific patterns first.

CLAIM_PATTERNS = [
    # Price patterns
    (re.compile(
        r'(?:costs?|priced? at|price(?:d)? of|price is|for)\s+£([\d]+\.?\d*)',
        re.IGNORECASE),
     "avg_price", "£{0}"),

    (re.compile(
        r'£([\d]+\.?\d*)\s+(?:dress|top|jacket|skirt|trouser|sweater|item)',
        re.IGNORECASE),
     "avg_price", "£{0}"),

    # Colour patterns
    (re.compile(
        r'(?:colour is|colored?|in (?:a )?|comes in )\b'
        r'(black|white|red|blue|green|pink|grey|gray|beige|navy|brown|'
        r'orange|yellow|purple|light blue|dark blue|light pink|dark pink|'
        r'light beige|light grey|dark grey)\b',
        re.IGNORECASE),
     "colour_group_name", "{0}"),

    (re.compile(
        r'\b(black|white|red|navy|beige|grey|gray|pink|brown)\s+'
        r'(?:dress|top|jacket|skirt|trouser|sweater|blouse|shirt)',
        re.IGNORECASE),
     "colour_group_name", "{0}"),

    # Material / fabric patterns
    (re.compile(
        r'(?:made of|made from|fabric is|material is|in a?)\s+'
        r'([a-z\s]+(?:weave|knit|cotton|polyester|viscose|linen|silk|'
        r'wool|denim|jersey|nylon|blend|mix))',
        re.IGNORECASE),
     "material", "{0}"),

    # Pattern / graphical appearance
    (re.compile(
        r'(?:pattern is|features? (?:a )?|with (?:a )?)\b'
        r'(solid|striped|floral|printed|embroidered|plain|patterned|'
        r'all over pattern|denim)\b',
        re.IGNORECASE),
     "graphical_appearance_name", "{0}"),

    # Type patterns
    (re.compile(
        r'\b(?:it is|this is|a) (?:a )?'
        r'(dress|top|trouser|skirt|jacket|sweater|blouse|shirt|coat)\b',
        re.IGNORECASE),
     "product_type_name", "{0}"),

    # Length patterns
    (re.compile(
        r'\b(calf-length|knee-length|mini|midi|maxi|long|short)\s+'
        r'(?:dress|skirt)',
        re.IGNORECASE),
     "length", "{0}"),
]


def _extract_claims_from_text(
    text: str,
    article_id: str,
    article_name: str,
    turn_id: str,
) -> list[dict]:
    """
    Extracts atomic factual claims from response text using regex patterns.

    Each claim is:
      {article_id, article_name, attribute, value, claim_text, turn_id, status}

    Only extracts claims for sentences that mention the article name or
    contain a product attribute keyword. This avoids false positives from
    conversational sentences.
    """
    claims = []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    for sentence in sentences:
        if len(sentence) < 10:
            continue

        # Only process sentences likely about this specific product
        article_mentioned = (
            article_name.lower() in sentence.lower() or
            any(kw in sentence.lower() for kw in [
                "costs", "priced", "made of", "material", "colour",
                "fabric", "pattern", "dress", "top", "jacket"
            ])
        )
        if not article_mentioned:
            continue

        for pattern, attribute, value_template in CLAIM_PATTERNS:
            match = pattern.search(sentence)
            if match:
                raw_value = match.group(1).strip().lower()
                value     = value_template.format(raw_value)

                claims.append({
                    "article_id":   article_id,
                    "article_name": article_name,
                    "attribute":    attribute,
                    "value":        value,
                    "claim_text":   sentence.strip(),
                    "turn_id":      turn_id,
                    "status":       "active",
                    "created_at":   datetime.now(timezone.utc).isoformat(),
                })
                break  # one claim per sentence per attribute

    return claims


def _extract_product_refs(evidence: dict) -> list[tuple[str, str]]:
    """
    Extracts (article_id, product_name) pairs from the evidence bundle.
    Returns list of (article_id, name) tuples to check claims against.
    """
    refs = []
    action = evidence.get("action", "")

    if action == "catalog_search":
        for item in evidence.get("items", []):
            aid  = item.get("article_id", "")
            name = item.get("name", "")
            if aid and name:
                refs.append((str(aid), name))

    elif action in ("item_attribute_lookup", "item_detail_lookup",
                    "explanation_generate"):
        article = evidence.get("article") or {}
        aid     = article.get("article_id", "")
        name    = article.get("name", "")
        if aid and name:
            refs.append((str(aid), name))

    elif action == "item_compare":
        for key in ("item_a", "item_b"):
            item = evidence.get(key) or {}
            aid  = item.get("article_id", "")
            name = item.get("name", "")
            if aid and name:
                refs.append((str(aid), name))

    return refs


# ── PostgreSQL authoritative lookup ───────────────────────────────────────────

ATTRIBUTE_TO_DB_FIELD = {
    "avg_price":               "avg_price",
    "colour_group_name":       "colour_group_name",
    "product_type_name":       "product_type_name",
    "graphical_appearance_name": "graphical_appearance_name",
    "garment_group_name":      "garment_group_name",
    "material":                "detail_desc",   # material in detail_desc
    "length":                  "detail_desc",   # length in detail_desc
}


async def _get_authoritative_value(
    article_id: str,
    attribute: str
) -> Optional[str]:
    """
    Queries PostgreSQL for the authoritative value of an attribute.
    Returns the true value as a string, or None if not found.
    """
    article = await get_article_by_id(article_id)
    if not article:
        return None

    db_field = ATTRIBUTE_TO_DB_FIELD.get(attribute)
    if not db_field:
        return None

    val = article.get(db_field)
    if val is None:
        return None

    if attribute == "avg_price":
        return f"£{float(val):.2f}"

    return str(val).strip().lower()


# ── NLI contradiction confirmation ────────────────────────────────────────────

def _confirm_contradiction_nli(
    old_claim_text: str,
    new_claim_text: str,
) -> tuple[bool, float]:
    """
    Uses NLI to confirm whether two claims contradict each other.
    Returns (is_contradiction, contradiction_score).

    NLI labels for cross-encoder/nli-deberta-v3-base:
      0 = CONTRADICTION
      1 = NEUTRAL
      2 = ENTAILMENT
    """
    nli = _get_nli()
    scores = nli.predict([(old_claim_text, new_claim_text)])
    contradiction_score = float(scores[0][0])
    is_contradiction    = contradiction_score > NLI_CONTRADICTION_THRESHOLD
    return is_contradiction, contradiction_score


# ── Response text correction ──────────────────────────────────────────────────

def _correct_response_text(
    response_text: str,
    wrong_claim_text: str,
    attribute: str,
    authoritative_value: str,
    article_name: str,
    wrong_value: str = "",
) -> str:
    """
    Replaces all occurrences of the wrong value in the response.

    Strategy:
      1. Replace the extracted wrong claim sentence with a correction note.
      2. Replace ALL remaining occurrences of the raw wrong value string
         (e.g. the price "31.08") with the authoritative value.
         This catches related sentences like "It costs £31.08" that were
         not individually extracted as separate claims.
    """
    # Build correction note based on attribute type
    if attribute == "avg_price":
        correction_note = (
            f"Note: {article_name} is actually priced at "
            f"{authoritative_value} (corrected)."
        )
    elif attribute == "colour_group_name":
        correction_note = (
            f"Note: {article_name} is {authoritative_value} in colour (corrected)."
        )
    elif attribute == "material":
        correction_note = (
            f"Note: {article_name} is made from {authoritative_value} (corrected)."
        )
    elif attribute == "product_type_name":
        correction_note = (
            f"Note: {article_name} is a {authoritative_value} (corrected)."
        )
    else:
        correction_note = (
            f"Correction: the {attribute.replace('_',' ')} of "
            f"{article_name} is {authoritative_value}."
        )

    corrected = response_text

    # Step 1: Replace the extracted wrong sentence with correction note
    if wrong_claim_text and wrong_claim_text in corrected:
        corrected = corrected.replace(wrong_claim_text, correction_note, 1)
        corrected = corrected.replace(wrong_claim_text, "")

    # Step 2: Replace ALL remaining occurrences of the raw wrong value
    # This catches "It costs £31.08" even if not extracted as a claim
    if wrong_value and authoritative_value and wrong_value in corrected:
        corrected = corrected.replace(wrong_value, authoritative_value)

    # Step 3: Clean up spacing
    while "  " in corrected:
        corrected = corrected.replace("  ", " ")
    corrected = corrected.strip()

    return corrected


# ── MongoDB storage ───────────────────────────────────────────────────────────

async def _store_explanation_document(
    session_id: str,
    user_id: str,
    turn_id: str,
    response_text: str,
    claims: list[dict],
    article_ids: list[str],
):
    """
    Stores one ExplanationDocument per bot turn in MongoDB.
    Creates or updates the document for this turn.
    """
    db = get_db()
    doc = {
        "session_id":        session_id,
        "user_id":           user_id,
        "turn_id":           turn_id,
        "full_explanation":  response_text,
        "claims":            claims,
        "article_ids":       article_ids,
        "contradiction_log": [],
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }
    await db.explanations.update_one(
        {"turn_id": turn_id},
        {"$set": doc},
        upsert=True
    )


async def _load_prior_claims(
    session_id: str,
    article_ids: list[str],
) -> list[dict]:
    """
    Loads all active claims for the given article_ids from earlier turns
    in this session.
    """
    db = get_db()
    docs = await db.explanations.find(
        {
            "session_id": session_id,
            "article_ids": {"$in": article_ids},
        }
    ).to_list(length=50)

    prior_claims = []
    for doc in docs:
        for claim in doc.get("claims", []):
            if claim.get("status") == "active":
                prior_claims.append(claim)

    return prior_claims


async def _mark_claim_contradicted(
    session_id: str,
    old_claim: dict,
    new_claim_text: str,
    nli_score: float,
    authoritative_value: str,
):
    """
    Marks an existing claim as contradicted in MongoDB and logs the event.
    """
    db = get_db()

    # Update the claim status in the explanations collection
    await db.explanations.update_one(
        {
            "session_id": session_id,
            "claims.turn_id":   old_claim["turn_id"],
            "claims.attribute": old_claim["attribute"],
            "claims.article_id":old_claim["article_id"],
        },
        {
            "$set": {
                "claims.$.status":            "contradicted",
                "claims.$.contradicted_at":   datetime.now(timezone.utc).isoformat(),
                "claims.$.contradicted_by":   new_claim_text,
                "claims.$.authoritative_value": authoritative_value,
            }
        }
    )

    # Log contradiction event in contradiction_log collection
    entry = {
        "session_id":           session_id,
        "detected_at":          datetime.now(timezone.utc).isoformat(),
        "article_id":           old_claim["article_id"],
        "article_name":         old_claim.get("article_name", ""),
        "attribute":            old_claim["attribute"],
        "old_claim_text":       old_claim["claim_text"],
        "old_value":            old_claim["value"],
        "new_claim_text":       new_claim_text,
        "nli_score":            nli_score,
        "authoritative_value":  authoritative_value,
        "resolution":           "retract_old",
        "resolution_note":      f"PostgreSQL confirms: {authoritative_value}",
    }
    await db.contradiction_log.insert_one(entry)
    print(f"[ContradictionDetector] Contradiction logged: "
          f"{old_claim['attribute']} of {old_claim.get('article_name','?')} — "
          f"'{old_claim['value']}' vs '{new_claim_text[:50]}'")


# ── Main ContradictionDetector class ─────────────────────────────────────────

class ContradictionDetector:
    """
    Checks bot responses for cross-turn contradictions and resolves them.

    Called after hallucination check passes in rag_pipeline.py.
    Maintains consistent explanation history across the entire session.
    """

    async def check_and_resolve(
        self,
        response_text:   str,
        evidence:        dict,
        session_id:      str,
        user_id:         str,
        turn_id:         str,
    ) -> dict:
        """
        Main entry point. Checks response for contradictions with prior claims,
        corrects any found, stores all claims in MongoDB.

        Args:
            response_text: The hallucination-checked bot response
            evidence:      Evidence bundle from EvidenceAssembler
            session_id:    Current session ID
            user_id:       Current user ID
            turn_id:       Current bot turn ID

        Returns structured result dict.
        """
        action = evidence.get("action", "no_retrieval")
        print(f"\n[CONTRA] ━━━ check_and_resolve() called ━━━")
        print(f"[CONTRA] action={action} session={session_id[:12] if session_id else '?'} turn={turn_id}")
        print(f"[CONTRA] response_text: {repr(response_text[:120])}")

        # Only check actions that make factual product claims
        if action not in {
            "catalog_search", "item_attribute_lookup",
            "item_compare", "explanation_generate", "item_detail_lookup"
        }:
            print(f"[CONTRA] SKIP: action={action} not factual")
            return self._no_check_result(response_text)

        # Step 1: Find product references in this response
        product_refs = _extract_product_refs(evidence)
        if not product_refs:
            return self._no_check_result(response_text)

        article_ids   = [ref[0] for ref in product_refs]
        product_names = [ref[1] for ref in product_refs]

        # Step 2: Extract atomic claims from response
        all_new_claims = []
        for article_id, article_name in product_refs:
            claims = _extract_claims_from_text(
                text=response_text,
                article_id=article_id,
                article_name=article_name,
                turn_id=turn_id,
            )
            all_new_claims.extend(claims)

        print(f"[CONTRA] product_refs={product_refs}")
        print(f"[CONTRA] claims extracted: {len(all_new_claims)}")
        for _cl in all_new_claims: print(f"  [CONTRA-CLAIM] attr={_cl['attribute']} val={_cl['value']} text='{_cl['claim_text'][:60]}'")
        # Step 3: Load prior active claims for these articles
        prior_claims = await _load_prior_claims(session_id, article_ids)

        print(f"[CONTRA] prior active claims loaded: {len(prior_claims)}")
        for _pc in prior_claims: print(f"  [CONTRA-PRIOR] attr={_pc['attribute']} val={_pc['value']} turn={_pc.get('turn_id','?')}")
        # Step 4: Deduplicate new claims — keep only first per (article_id, attribute)
        # This prevents double-counting when same attribute appears twice in response
        seen_attr_keys  = set()
        deduped_claims  = []
        for claim in all_new_claims:
            key = (claim["article_id"], claim["attribute"])
            if key not in seen_attr_keys:
                seen_attr_keys.add(key)
                deduped_claims.append(claim)
        all_new_claims = deduped_claims

        print(f"[CONTRA] after dedup: {len(all_new_claims)} unique claims")
        # Check each deduplicated new claim against prior claims
        contradictions  = []
        corrected_text  = response_text
        claims_to_store = list(all_new_claims)

        for new_claim in all_new_claims:
            for prior in prior_claims:
                # Contradiction candidate: same article, same attribute,
                # different value
                if (
                    prior["article_id"] == new_claim["article_id"]
                    and prior["attribute"] == new_claim["attribute"]
                    and prior["value"].lower() != new_claim["value"].lower()
                    and prior["status"] == "active"
                ):
                    # Step 5: Confirm with NLI
                    is_contra, nli_score = _confirm_contradiction_nli(
                        old_claim_text=prior["claim_text"],
                        new_claim_text=new_claim["claim_text"],
                    )

                    if not is_contra:
                        continue

                    print(f"[CONTRA-DETECTED] ⚠ Article: {new_claim.get('article_name','?')} | attr={new_claim['attribute']} | old={prior['value']} vs new={new_claim['value']} NLI={nli_score:.3f}")

                    # Step 6: Query PostgreSQL for authoritative truth
                    auth_value = await _get_authoritative_value(
                        article_id=new_claim["article_id"],
                        attribute=new_claim["attribute"],
                    )

                    if auth_value is None:
                        auth_value = new_claim["value"]  # trust newer claim

                    # Step 7: Determine which claim is wrong
                    # Compare both claims against authoritative value
                    prior_matches = (
                        prior["value"].lower() in (auth_value or "").lower() or
                        (auth_value or "").lower() in prior["value"].lower()
                    )
                    new_matches = (
                        new_claim["value"].lower() in (auth_value or "").lower() or
                        (auth_value or "").lower() in new_claim["value"].lower()
                    )

                    if prior_matches and not new_matches:
                        # Prior claim is correct — new claim is wrong
                        # Correct the current response text
                        corrected_text = _correct_response_text(
                            response_text=corrected_text,
                            wrong_claim_text=new_claim["claim_text"],
                            attribute=new_claim["attribute"],
                            authoritative_value=auth_value,
                            article_name=new_claim.get("article_name", "the item"),
                            wrong_value=new_claim["value"],
                        )
                        # Mark new claim as contradicted before storing
                        new_claim["status"] = "contradicted"
                        wrong_claim   = new_claim
                        correct_claim = prior

                        # FIX 2: Also write ContradictionEntry for retract_new
                        # The new claim is wrong — log this event to MongoDB
                        db = get_db()
                        entry = {
                            "session_id":           session_id,
                            "detected_at":          datetime.now(timezone.utc).isoformat(),
                            "article_id":           new_claim["article_id"],
                            "article_name":         new_claim.get("article_name", ""),
                            "attribute":            new_claim["attribute"],
                            "old_claim_text":       prior["claim_text"],
                            "old_value":            prior["value"],
                            "new_claim_text":       new_claim["claim_text"],
                            "new_value":            new_claim["value"],
                            "nli_score":            nli_score,
                            "authoritative_value":  auth_value,
                            "resolution":           "retract_new",
                            "resolution_note":      (
                                f"Prior claim '{prior['value']}' matches DB. "
                                f"New claim '{new_claim['value']}' is wrong and corrected."
                            ),
                        }
                        await db.contradiction_log.insert_one(entry)
                        print(
                            f"[ContradictionDetector] ContradictionEntry stored "
                            f"(retract_new): {new_claim['attribute']} of "
                            f"{new_claim.get('article_name','?')} corrected."
                        )

                    else:
                        # New claim is correct (or both wrong — trust DB)
                        # Mark old claim as contradicted in MongoDB
                        await _mark_claim_contradicted(
                            session_id=session_id,
                            old_claim=prior,
                            new_claim_text=new_claim["claim_text"],
                            nli_score=nli_score,
                            authoritative_value=auth_value,
                        )
                        wrong_claim   = prior
                        correct_claim = new_claim

                    contradictions.append({
                        "article_id":          new_claim["article_id"],
                        "article_name":        new_claim.get("article_name", ""),
                        "attribute":           new_claim["attribute"],
                        "old_value":           prior["value"],
                        "new_value":           new_claim["value"],
                        "authoritative_value": auth_value,
                        "nli_score":           nli_score,
                        "wrong_claim":         wrong_claim["claim_text"],
                        "correct_claim":       correct_claim["claim_text"],
                        "resolution":          "retract_old" if not prior_matches else "retract_new",
                    })

        # Step 8: Store ExplanationDocument with all new claims
        await _store_explanation_document(
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            response_text=corrected_text,
            claims=claims_to_store,
            article_ids=article_ids,
        )

        # Step 9: Return final result
        contradiction_found = len(contradictions) > 0
        print(f"[CONTRA] ─── result: contradiction_found={contradiction_found} count={len(contradictions)} claims_stored={len(claims_to_store)}")
        if contradiction_found:
            print(f"[CONTRA] corrected response: {repr(corrected_text[:200])}")
            print(f"[ContradictionDetector] {len(contradictions)} contradiction(s) resolved.")

        return {
            "response_text":       corrected_text,
            "contradiction_found": contradiction_found,
            "contradiction_count": len(contradictions),
            "contradictions":      contradictions,
            "claims_stored":       len(claims_to_store),
            "product_ids":         article_ids,
            "product_names":       product_names,
        }

    def _no_check_result(self, response_text: str) -> dict:
        """Returns a pass-through result for non-factual actions."""
        return {
            "response_text":       response_text,
            "contradiction_found": False,
            "contradiction_count": 0,
            "contradictions":      [],
            "claims_stored":       0,
            "product_ids":         [],
            "product_names":       [],
        }
