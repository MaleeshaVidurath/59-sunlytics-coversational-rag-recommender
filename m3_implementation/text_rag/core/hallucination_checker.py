# m3_implementation/text_rag/core/hallucination_checker.py
#
# NLI-based hallucination detection for LLM responses.
#
# APPROACH — Option A: evidence-field-first
#   For each evidence field (colour, price, name, type, …):
#     1. If LLM usage map names sentence numbers for this field → use those sentences
#     2. Otherwise use MiniLM to find the best matching sentence in the response
#     3. If best MiniLM similarity < _MIN_SIMILARITY and field not in usage map → SKIP
#        (LLM did not mention this field — nothing to contradict)
#     4. Run DeBERTa on (evidence_fact, best_sentence) pair
#     5. If CONTRADICTION > _NLI_CONTRADICTION_THRESHOLD → hallucination
#
# IMPORTANT: is_unsupported (entailment < threshold) is NOT checked.
#   Reason: a sentence like "Red T-shirt at £45, stretch fabric" contains the
#   correct colour but also extra info (price, material) that the colour evidence
#   alone cannot confirm.  DeBERTa would score entailment low → false positive.
#   We only care: did the LLM CONTRADICT a field value we gave it?
#
# NLI LABELS from cross-encoder/nli-deberta-v3-base:
#   Label 0 = CONTRADICTION
#   Label 1 = NEUTRAL
#   Label 2 = ENTAILMENT

import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from text_rag.config import NLI_CONTRADICTION_THRESHOLD

# Sentence splitting, the item→sentence map and the MiniLM/DeBERTa singletons
# live in shared modules so the cross-turn consistency layer reads the response
# through exactly the same lens — one extraction, two consumers.
from text_rag.core.nli_model import get_nli_model as _get_nli_model
from text_rag.core.assertion_extractor import (
    get_embed_model        as _get_embed_model,
    split_sentences        as _split_sentences,
    build_item_sentence_map,
    norm_ws                as _norm_ws,
    get_catalog_names      as _get_catalog_names,
)

_MIN_SIMILARITY = 0.35  # MiniLM threshold below which a field is considered unused


# ── Sentence skip rules ────────────────────────────────────────────────────────

_SKIP_PATTERNS = [
    r'\b(short|long|relaxed|slim|fitted|woven|knit|cotton|stretch|denim)\b',
    r'\b(waist|crotch|pocket|hem|sleeve|collar|button|zip|fly)\b',
    r'\b(regular|classic|modern|style|design|detail|trim|finish)\b',
]
_SKIP_RE = re.compile('|'.join(_SKIP_PATTERNS), re.IGNORECASE)

_NON_FACTUAL_STARTS = [
    "here are", "i hope", "you might", "feel free", "let me know",
    "would you", "do you", "thank you", "these are", "please",
    "i'm happy", "i'd be", "of course", "great choice",
]


def _should_skip_sentence(sentence: str) -> bool:
    """
    Returns True for sentences that must never be NLI-checked:
    conversational openers and garment-description sentences.

    Exception: sentences containing a £ price are Option sentences
    (e.g. "Option 1: Solo bra, Black, £13.10, This bra's sporty design...")
    and must never be skipped even if they contain description words.
    """
    # Option sentences always contain a £ price — never skip these
    if "£" in sentence:
        return False
    s = sentence.lower()
    for start in _NON_FACTUAL_STARTS:
        if s.startswith(start):
            return True
    return bool(_SKIP_RE.search(sentence))


# ── Exact-value verification helpers (two-sided Gates 6/7) ────────────────────
# Name and price are exact values: they are either present in the item's
# sentence or they are not. String logic decides both directions — these
# fields NEVER go to NLI (DeBERTa is unreliable on numbers and proper names:
# "priced at £11.08" vs "£13.58" usually scores neutral, not contradiction).

_SLOT_GENERIC_STARTS = ("this ", "that ", "these ", "those ", "it ", "the ",
                        "here ", "there ", "i ", "we ", "you ")

def _find_wrong_name(sentence: str, true_name: str, evidence: dict) -> str | None:
    """Two-sided name check: returns a DIFFERENT product name found in the
    sentence, or None if no other name is present. Only called after the true
    name failed verbatim containment."""
    sent_norm = _norm_ws(sentence).lower()
    true_norm = _norm_ws(true_name).lower()

    def is_diff(candidate: str) -> bool:
        c = _norm_ws(candidate).lower()
        # substring relations (London dress / SS London dress) are ambiguous
        # truncations, not clear contradictions — never flag those
        return bool(c) and c != true_norm and c not in true_norm and true_norm not in c

    # 1. Names of other items in the SAME evidence — catches cross-item swaps
    other_names = []
    for it in (evidence.get("items") or []):
        other_names.append(it.get("name", ""))
    for key in ("article", "item_a", "item_b"):
        if evidence.get(key):
            other_names.append(evidence[key].get("name", ""))
    for n in other_names:
        if n and is_diff(n) and _norm_ws(n).lower() in sent_norm:
            return n

    # 2. Structured option sentences ("Option 1: <name>, ..." / "<name>: ...")
    #    always carry a £ price — parse the name slot and compare.
    if "£" in sentence:
        m = re.match(r"\s*(?:option\s+\d+\s*:\s*)?([^,:£]{3,60})[,:]",
                     _norm_ws(sentence), re.IGNORECASE)
        if m:
            slot = m.group(1).strip()
            if (slot and is_diff(slot)
                    and not slot.lower().startswith(_SLOT_GENERIC_STARTS)):
                return slot

    # 3. Known catalog names (case-sensitive match to avoid common-word hits)
    for n in _get_catalog_names():
        if is_diff(n) and n in sentence:
            return n
    return None


def _find_wrong_price(sentence: str, true_price: str, grounded: set = None) -> str | None:
    """Two-sided price check: returns a differing £value found in the
    sentence, or None if no £value is present. Only called after the true
    price failed verbatim containment.

    `grounded` holds £values that appear elsewhere in the evidence and are not
    item prices — most importantly the user's habitual spend range, which the
    ranker cites as a selection reason ("£4.59 is inside your usual
    £4.02-£9.07 range"). Quoting a grounded value is not an invented price.
    Passed only at response level; the locked-sentence branch stays strict so
    genuine cross-item price swaps are still caught."""
    sent_prices = re.findall(r"£[\d,]+(?:\.\d{1,2})?", sentence)
    allowed = grounded or set()
    wrong = [p for p in sent_prices if p != true_price and p not in allowed]
    return wrong[0] if wrong else None


def _grounded_prices(facts: list[dict]) -> set:
    """Every £value stated anywhere in the evidence facts."""
    found = set()
    for f in facts:
        found.update(re.findall(r"£[\d,]+(?:\.\d{1,2})?", f.get("text", "")))
    return found


# ── Evidence flattening ────────────────────────────────────────────────────────

def _flatten_evidence(evidence: dict) -> list[dict]:
    """
    Converts the evidence bundle into a list of
    {"field": str, "text": str} dicts.
    Each dict represents one checkable fact from one evidence field.
    """
    facts = []

    def add_article_facts(article: dict, prefix: str = "", item_idx: int = None):
        if not article:
            return
        name = article.get("name", "")
        _idx = {} if item_idx is None else {"item_idx": item_idx}
        if name:
            facts.append({"field": "name",
                           "text": f"{prefix}The item is called {name}.", **_idx})
        if article.get("type"):
            facts.append({"field": "type",
                           "text": f"{prefix}{name} is a {article['type']}." if name else f"{prefix}It is a {article['type']}.", **_idx})
        if article.get("colour"):
            facts.append({"field": "colour",
                           "text": f"{prefix}{name} is {article['colour']} in colour." if name else f"{prefix}The colour is {article['colour']}.", **_idx})
        if article.get("price"):
            facts.append({"field": "price",
                           "text": f"{prefix}{name} is priced at {article['price']}." if name else f"{prefix}The price is {article['price']}.", **_idx})
        if article.get("pattern"):
            facts.append({"field": "pattern",
                           "text": f"{prefix}{name} has a {article['pattern']} pattern." if name else f"{prefix}The pattern is {article['pattern']}.", **_idx})
        if article.get("index_group"):
            facts.append({"field": "index_group",
                           "text": f"{prefix}It is from {article['index_group']}.", **_idx})
        if article.get("section"):
            facts.append({"field": "section",
                           "text": f"{prefix}It is in the {article['section']} section.", **_idx})

    action = evidence.get("action", "")

    if action == "catalog_search":
        for item_idx, item in enumerate(evidence.get("items", [])):
            add_article_facts(item, item_idx=item_idx)
        for boost in evidence.get("preference_boosts", []):
            facts.append({
                "field": "preference",
                "text": (
                    f"User prefers {boost['attribute']}={boost['value']} "
                    f"with weight {boost['weight']:.2f}."
                ),
            })

    elif action in ("item_attribute_lookup", "item_detail_lookup"):
        add_article_facts(evidence.get("article"))
        for k, v in evidence.get("extracted_facts", {}).items():
            facts.append({"field": k, "text": f"{k}: {v}"})

    elif action == "item_compare":
        add_article_facts(evidence.get("item_a"), prefix="Option 1: ")
        add_article_facts(evidence.get("item_b"), prefix="Option 2: ")
        for k, v in evidence.get("comparison_facts", {}).items():
            facts.append({"field": k, "text": f"{k}: {v}"})

    elif action == "explanation_generate":
        add_article_facts(evidence.get("article"))

        # All-items path: the bundle carries `articles`, not `article`. Without
        # these the checker had zero facts to verify and passed everything.
        for item_idx, art in enumerate(evidence.get("articles", [])):
            add_article_facts(art, item_idx=item_idx)
            for reason in art.get("why", []):
                facts.append({
                    "field": "selection_reason",
                    "text":  reason,
                    "item_idx": item_idx,
                })

        # Single-item path reasons.
        for reason in evidence.get("ranking_reasons", []):
            facts.append({"field": "selection_reason", "text": reason})

        for match in evidence.get("confirmed_matches", []):
            facts.append({
                "field": match.get("attribute", "match"),
                "text": (
                    f"The item's {match['attribute']} is {match['value']}, "
                    f"which matches the user's preference."
                ),
            })
        for claim in evidence.get("prior_claims", []):
            if claim.get("status") == "active":
                facts.append({
                    "field": "prior_claim",
                    "text": f"Already stated to user: {claim['claim_text']}",
                })

    if not facts:
        facts.append({"field": "general", "text": "No specific product facts available."})

    return facts


# ── Best sentence finder (for a given evidence fact) ──────────────────────────

def _find_best_sentence(
    fact_text:           str,
    sentences:           list[str],
    option_sentence_map: dict = None,
    item_idx:            int  = None,
) -> tuple[str, float, int]:
    """
    Finds the best matching sentence in the LLM response for a given evidence fact.

    If option_sentence_map has an entry for this item → lock to that sentence.
    This prevents cross-item name collisions (e.g. two items sharing a name prefix).
    MiniLM similarity is always computed so the _MIN_SIMILARITY gate still works.

    Otherwise MiniLM scores all sentences and picks the highest.

    Returns (best_sentence_text, similarity_score, sentence_number_1indexed).
    Returns ("", 0.0, -1) if no sentences available.
    """
    if not sentences:
        return "", 0.0, -1

    import numpy as np
    model = _get_embed_model()

    # Item-to-sentence lock for catalog_search "Option N:" responses.
    # Locks item N's facts to only its own sentence, preventing cross-item collisions.
    if option_sentence_map and item_idx is not None:
        locked_sent_idx = option_sentence_map.get(item_idx)
        if locked_sent_idx is not None and locked_sent_idx < len(sentences):
            locked_sent = sentences[locked_sent_idx]
            embs = model.encode([fact_text, locked_sent])
            sim  = float(np.dot(embs[0], embs[1]) /
                         (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8))
            return locked_sent, sim, locked_sent_idx + 1

    # No lock — MiniLM scores all sentences, highest wins
    embeddings  = model.encode([fact_text] + sentences)
    fact_emb    = embeddings[0]
    sent_embs   = embeddings[1:]

    similarities = [
        float(np.dot(fact_emb, se) / (np.linalg.norm(fact_emb) * np.linalg.norm(se) + 1e-8))
        for se in sent_embs
    ]

    best_idx   = int(np.argmax(similarities))
    best_score = similarities[best_idx]
    return sentences[best_idx], best_score, best_idx + 1


# ── Main hallucination checker ─────────────────────────────────────────────────

class HallucinationChecker:
    """
    Checks LLM responses for hallucinations using NLI — evidence-field-first.
    Loops over each evidence fact, finds the LLM sentence that best matches it,
    and runs DeBERTa contradiction check on the pair.
    Only contradiction is flagged (not low entailment — see module docstring).
    """

    def check(
        self,
        response_text: str,
        evidence:      dict,
    ) -> dict:
        """
        Args:
            response_text: The LLM-generated response to check
            evidence:      The evidence bundle used to generate the response

        Returns:
            {
                "has_hallucination":   bool,
                "hallucination_score": float,
                "flagged_sentences":   list[dict],
                "all_checks":          list[dict],
                "n_checked":           int,
                "n_flagged":           int,
                "passed":              bool,
                "contradicted_fields": list[str],
            }
        """
        if not response_text or not response_text.strip():
            return self._empty_result()

        sentences        = _split_sentences(response_text)
        structured_facts = _flatten_evidence(evidence)
        nli_model        = _get_nli_model()

        # Build item→sentence map for catalog_search using MiniLM semantic matching.
        # Order-independent and format-independent — no reliance on "Option N:" numbering.
        option_sentence_map = {}
        if evidence.get("action") == "catalog_search":
            items = evidence.get("items", [])
            option_sentence_map = build_item_sentence_map(sentences, items)

        results             = []
        flagged             = []
        total_score         = 0.0
        contradicted_fields = set()
        checked_pairs       = set()   # (fact_text, sentence) pairs already checked

        print(f"\n[HALL-CHECK] ━━━ check() called ━━━")
        print(f"[HALL-CHECK] response len={len(response_text)}: {repr(response_text[:120])}")
        print(f"[HALL-CHECK] sentences={len(sentences)} evidence_facts={len(structured_facts)}")
        print(f"[HALL-CHECK] action={evidence.get('action','?')}")
        print(f"[HALL-CHECK] option_sentence_map={option_sentence_map}")
        for _f in structured_facts[:6]:
            print(f"  [HALL-FACT] [{_f['field']}] {_f['text'][:80]}")

        # For catalog_search, only name/colour/price are reliable enough to verify.
        # pattern, type, section, index_group all clash with LLM description text
        # and produce systematic false positives that cascade across retry attempts.
        _CATALOG_CORE_FIELDS = {"name", "colour", "price"}

        for fact in structured_facts:
            fact_text  = fact["text"]
            fact_field = fact["field"]
            fact_item  = fact.get("item_idx")

            if evidence.get("action") == "catalog_search" and fact_field not in _CATALOG_CORE_FIELDS:
                continue

            # If option_sentence_map was built but this item has no locked sentence,
            # the LLM didn't mention it in this response — skip to avoid cross-item
            # collisions where an unlocked item's fact matches another item's sentence.
            if option_sentence_map and fact_item is not None and fact_item not in option_sentence_map:
                continue

            # Find best matching sentence in LLM response for this evidence field
            best_sentence, similarity, sent_num = _find_best_sentence(
                fact_text, sentences,
                option_sentence_map=option_sentence_map,
                item_idx=fact_item,
            )

            if not best_sentence:
                continue

            # Skip if sentence is a description/non-factual type
            if _should_skip_sentence(best_sentence):
                print(f"[HALL-CHECK] SKIP [{fact_field}] sentence type: '{best_sentence[:60]}'")
                continue

            # Name/price: exact values — string logic decides BOTH directions
            # (two-sided gates). These fields never reach NLI: DeBERTa produces
            # false contradictions on correct values in structured sentences AND
            # misses wrong values (it scores "£11.08" vs "£13.58" as neutral).
            # They also deliberately BYPASS the MiniLM similarity gate below:
            # a swapped name destroys the very similarity the gate measures,
            # so low similarity is itself a symptom of this hallucination type.
            #
            # Locked items (catalog lock map): the locked sentence is
            #   authoritative for this item — verify against it directly.
            # Unlocked facts (compare/explanation/detail): verify at response
            #   level — only flag when the true value is absent from the WHOLE
            #   response and a different value of the same kind is present.
            #   (Response-level check avoids false flags when MiniLM's free
            #   search pairs item A's fact with item B's sentence, and avoids
            #   flagging derived values like "£4.04 more expensive".)
            is_locked = bool(option_sentence_map) and fact_item is not None \
                and fact_item in option_sentence_map

            def _record_pass():
                results.append({
                    "fact_field": fact_field, "fact_text": fact_text,
                    "sentence": best_sentence, "sentence_number": sent_num,
                    "similarity": similarity,
                    "nli_scores": {"contradiction": 0.0, "neutral": 0.0, "entailment": 1.0},
                    "passed": True, "is_contradiction": False,
                })

            def _record_flag(found_value):
                result = {
                    "fact_field": fact_field, "fact_text": fact_text,
                    "sentence": best_sentence, "sentence_number": sent_num,
                    "similarity": similarity,
                    "nli_scores": {"contradiction": 1.0, "neutral": 0.0, "entailment": 0.0},
                    "passed": False, "is_contradiction": True,
                    "method": "containment", "found_value": found_value,
                }
                results.append(result)
                flagged.append(result)
                contradicted_fields.add(fact_field)

            if fact_field == "name":
                name_m = re.search(r"The item is called (.+?)\.", fact_text)
                if name_m:
                    true_name = name_m.group(1)
                    if _norm_ws(true_name).lower() in _norm_ws(best_sentence).lower():
                        print(f"[HALL-CHECK] PASS [name] verbatim: '{true_name[:50]}'")
                        _record_pass()
                    elif is_locked:
                        # Locked sentence is this item's sentence — a different
                        # name here is a contradiction even if the true name
                        # appears elsewhere (cross-item swap).
                        wrong_name = _find_wrong_name(best_sentence, true_name, evidence)
                        if wrong_name is not None:
                            print(f"[HALL-CHECK] FLAG [name] locked-sentence: "
                                  f"expected '{true_name[:40]}' found '{wrong_name[:40]}'")
                            total_score += 1.0
                            _record_flag(wrong_name)
                        else:
                            print("[HALL-CHECK] SKIP [name] not stated in locked sentence")
                    elif _norm_ws(true_name).lower() in _norm_ws(response_text).lower():
                        print("[HALL-CHECK] SKIP [name] correct elsewhere in response")
                    else:
                        # True name absent from the ENTIRE response — if another
                        # product name is present, the LLM renamed the item.
                        wrong_name = (_find_wrong_name(best_sentence, true_name, evidence)
                                      or _find_wrong_name(response_text, true_name, evidence))
                        if wrong_name is not None:
                            print(f"[HALL-CHECK] FLAG [name] response-level: "
                                  f"expected '{true_name[:40]}' found '{wrong_name[:40]}'")
                            total_score += 1.0
                            _record_flag(wrong_name)
                        else:
                            print("[HALL-CHECK] SKIP [name] not stated in response")
                    continue

            if fact_field == "price":
                price_m = re.search(r"£[\d,]+(?:\.\d{1,2})?", fact_text)
                if price_m:
                    true_price = price_m.group(0)
                    if true_price in best_sentence:
                        print(f"[HALL-CHECK] PASS [price] verbatim: '{true_price}'")
                        _record_pass()
                    elif is_locked:
                        wrong_price = _find_wrong_price(best_sentence, true_price)
                        if wrong_price is not None:
                            print(f"[HALL-CHECK] FLAG [price] locked-sentence: "
                                  f"expected '{true_price}' found '{wrong_price}'")
                            total_score += 1.0
                            _record_flag(wrong_price)
                        else:
                            print("[HALL-CHECK] SKIP [price] not stated in locked sentence")
                    elif true_price in response_text:
                        print("[HALL-CHECK] SKIP [price] correct elsewhere in response")
                    else:
                        wrong_price = _find_wrong_price(
                            response_text, true_price, _grounded_prices(structured_facts)
                        )
                        if wrong_price is not None:
                            print(f"[HALL-CHECK] FLAG [price] response-level: "
                                  f"expected '{true_price}' found '{wrong_price}'")
                            total_score += 1.0
                            _record_flag(wrong_price)
                        else:
                            print("[HALL-CHECK] SKIP [price] not stated in response")
                    continue

            # Skip if LLM similarity is too low — field not mentioned in response
            if similarity < _MIN_SIMILARITY:
                print(
                    f"[HALL-CHECK] SKIP [{fact_field}] not used by LLM "
                    f"(sim={similarity:.3f}): '{best_sentence[:50]}'"
                )
                continue

            # Avoid duplicate (fact, sentence) pair checks
            pair_key = (fact_text, best_sentence)
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            print(
                f"[HALL-CHECK] NLI [{fact_field}] sim={similarity:.3f} "
                f"s{sent_num}: '{best_sentence[:60]}'"
            )
            print(f"[HALL-CHECK] vs fact: '{fact_text[:70]}'")

            scores = nli_model.predict([(fact_text, best_sentence)])
            score_dict = {
                "contradiction": float(scores[0][0]),
                "neutral":       float(scores[0][1]),
                "entailment":    float(scores[0][2]),
            }

            # Only contradiction is flagged — entailment not checked.
            # A sentence with extra info scores low entailment but is not a hallucination.
            is_hallucination = (
                score_dict["contradiction"] > NLI_CONTRADICTION_THRESHOLD
                and score_dict["contradiction"] > score_dict["entailment"]
            )

            result = {
                "fact_field":       fact_field,
                "fact_text":        fact_text,
                "sentence":         best_sentence,
                "sentence_number":  sent_num,
                "similarity":       similarity,
                "nli_scores":       score_dict,
                "passed":           not is_hallucination,
                "is_contradiction": is_hallucination,
            }
            results.append(result)

            print(
                f"[HALL-CHECK] NLI: contra={score_dict['contradiction']:.4f} "
                f"neutral={score_dict['neutral']:.4f} entail={score_dict['entailment']:.4f} "
                f"→ HALLUCINATION={is_hallucination}"
            )

            if is_hallucination:
                total_score += score_dict["contradiction"]
                flagged.append(result)
                contradicted_fields.add(fact_field)

        n_checked         = len(results)
        avg_score         = total_score / len(flagged) if flagged else 0.0
        has_hallucination = len(flagged) > 0

        print(
            f"[HALL-CHECK] RESULT: has_hallucination={has_hallucination} "
            f"score={round(avg_score,3)} n_checked={n_checked} n_flagged={len(flagged)} "
            f"contradicted_fields={list(contradicted_fields)}"
        )
        return {
            "has_hallucination":   has_hallucination,
            "hallucination_score": round(avg_score, 3),
            "flagged_sentences":   flagged,
            "all_checks":          results,
            "n_checked":           n_checked,
            "n_flagged":           len(flagged),
            "passed":              not has_hallucination,
            "contradicted_fields": list(contradicted_fields),
        }

    def _empty_result(self) -> dict:
        return {
            "has_hallucination":   False,
            "hallucination_score": 0.0,
            "flagged_sentences":   [],
            "all_checks":          [],
            "n_checked":           0,
            "n_flagged":           0,
            "passed":              True,
            "contradicted_fields": [],
        }
