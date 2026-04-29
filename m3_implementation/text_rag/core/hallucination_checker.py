# m3_implementation/text_rag/core/hallucination_checker.py
#
# NLI-based hallucination detection for LLM responses.
#
# HOW IT WORKS:
#   1. Split LLM response into individual sentences
#   2. For each sentence, find the most relevant evidence piece
#      using sentence embedding similarity
#   3. Run NLI (cross-encoder/nli-deberta-v3-base) on (evidence, sentence) pair
#   4. NLI returns 3 scores: CONTRADICTION, NEUTRAL, ENTAILMENT
#   5. If CONTRADICTION > 0.65 → sentence is a hallucination
#   6. If ENTAILMENT < 0.20 for factual sentences → possible hallucination
#   7. Aggregate: if any sentence fails → response flagged
#
# NLI LABELS from cross-encoder/nli-deberta-v3-base:
#   Label 0 = CONTRADICTION
#   Label 1 = NEUTRAL
#   Label 2 = ENTAILMENT
#
# FACTUAL vs NON-FACTUAL sentences:
#   Factual: contains numbers, product names, specific attributes
#   Non-factual: greetings, transitions, opinions ("I think", "you might")
#   Only factual sentences are strictly checked against evidence

import os
import sys
import re
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from text_rag.config import (
    NLI_MODEL_NAME, NLI_CONTRADICTION_THRESHOLD, NLI_ENTAILMENT_THRESHOLD
)

_nli_model   = None
_embed_model = None


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
        print(f"[HallucinationChecker] NLI model loaded: {NLI_MODEL_NAME}")
    return _nli_model


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ── Sentence splitting ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Splits text into sentences, filtering out very short ones."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


# ── Factual sentence detection ─────────────────────────────────────────────────

_FACTUAL_PATTERNS = [
    r'£\d',                          # price mentions only
    r'£[\d]+\.\d+',               # exact price format
]
# Only check price and name sentences — descriptions are too complex for NLI
# Description sentences contain partial/truncated text that NLI misreads
_SKIP_PATTERNS = [
    r'\b(short|long|relaxed|slim|fitted|woven|knit|cotton|stretch|denim)\b',
    r'\b(waist|crotch|pocket|hem|sleeve|collar|button|zip|fly)\b',
    r'\b(regular|classic|modern|style|design|detail|trim|finish)\b',
]
_SKIP_RE = re.compile('|'.join(_SKIP_PATTERNS), re.IGNORECASE)
_FACTUAL_RE = re.compile('|'.join(_FACTUAL_PATTERNS), re.IGNORECASE)

_NON_FACTUAL_STARTS = [
    "here are", "i hope", "you might", "feel free", "let me know",
    "would you", "do you", "thank you", "these are", "please",
    "i'm happy", "i'd be", "of course", "great choice",
]


def _is_factual_sentence(sentence: str) -> bool:
    """
    Returns True only if the sentence makes a VERIFIABLE factual claim.
    We only check price sentences — descriptions are too complex for NLI
    because they get truncated differently in evidence vs response.
    """
    s = sentence.lower()
    for start in _NON_FACTUAL_STARTS:
        if s.startswith(start):
            return False
    # Skip description-style sentences (contain garment detail words)
    if _SKIP_RE.search(sentence):
        return False
    # Only check sentences with price or clear product names
    return bool(_FACTUAL_RE.search(sentence))


# ── Evidence flattening ────────────────────────────────────────────────────────

def _flatten_evidence(evidence: dict) -> list[str]:
    """
    Converts the evidence bundle into a list of fact strings
    that can be compared against LLM sentences via NLI.
    """
    facts = []

    def add_article_facts(article: dict, prefix: str = ""):
        if not article:
            return
        if article.get("name"):
            facts.append(f"{prefix}The item is called {article['name']}.")
        if article.get("type"):
            facts.append(f"{prefix}It is a {article['type']}.")
        if article.get("colour"):
            facts.append(f"{prefix}The colour is {article['colour']}.")
        if article.get("price"):
            facts.append(f"{prefix}The price is {article['price']}.")
        if article.get("pattern"):
            facts.append(f"{prefix}The pattern is {article['pattern']}.")
        if article.get("material_description"):
            desc = article["material_description"][:300]
            facts.append(f"{prefix}Description: {desc}")
        if article.get("garment_group"):
            facts.append(f"{prefix}It belongs to {article['garment_group']}.")
        if article.get("index_group"):
            facts.append(f"{prefix}It is from {article['index_group']}.")

    action = evidence.get("action", "")

    if action == "catalog_search":
        for item in evidence.get("items", []):
            add_article_facts(item)
        for boost in evidence.get("preference_boosts", []):
            facts.append(
                f"User prefers {boost['attribute']}={boost['value']} "
                f"with weight {boost['weight']:.2f}."
            )

    elif action in ("item_attribute_lookup", "item_detail_lookup"):
        add_article_facts(evidence.get("article"))
        for k, v in evidence.get("extracted_facts", {}).items():
            facts.append(f"{k}: {v}")

    elif action == "item_compare":
        add_article_facts(evidence.get("item_a"), prefix="Option 1: ")
        add_article_facts(evidence.get("item_b"), prefix="Option 2: ")
        for k, v in evidence.get("comparison_facts", {}).items():
            facts.append(f"{k}: {v}")

    elif action == "explanation_generate":
        add_article_facts(evidence.get("article"))
        for match in evidence.get("confirmed_matches", []):
            facts.append(
                f"The item's {match['attribute']} is {match['value']}, "
                f"which matches the user's preference."
            )
        for claim in evidence.get("prior_claims", []):
            if claim.get("status") == "active":
                facts.append(f"Already stated to user: {claim['claim_text']}")

    if not facts:
        facts.append("No specific product facts available.")

    return facts


# ── Best evidence finder ───────────────────────────────────────────────────────

def _find_best_evidence(sentence: str, fact_list: list[str]) -> str:
    """
    Finds the most relevant evidence fact for a given sentence
    using cosine similarity between embeddings.
    """
    if not fact_list:
        return ""
    if len(fact_list) == 1:
        return fact_list[0]

    model      = _get_embed_model()
    embeddings = model.encode([sentence] + fact_list)
    sent_emb   = embeddings[0]
    fact_embs  = embeddings[1:]

    import numpy as np
    similarities = [
        float(np.dot(sent_emb, fe) / (np.linalg.norm(sent_emb) * np.linalg.norm(fe) + 1e-8))
        for fe in fact_embs
    ]
    best_idx = int(np.argmax(similarities))
    return fact_list[best_idx]


# ── Main hallucination checker ─────────────────────────────────────────────────

class HallucinationChecker:
    """
    Checks LLM responses for hallucinations using NLI.
    Returns a structured result with per-sentence scores and an overall flag.
    """

    def check(
        self,
        response_text: str,
        evidence: dict
    ) -> dict:
        """
        Checks a response for hallucinations.

        Args:
            response_text: The LLM-generated response to check
            evidence:      The evidence bundle used to generate the response

        Returns:
            {
                "has_hallucination": bool,
                "hallucination_score": float,  # 0.0-1.0 (1.0 = very bad)
                "flagged_sentences": list[dict],  # sentences that failed
                "all_sentences": list[dict],      # all sentence results
                "passed": bool,
            }
        """
        if not response_text or not response_text.strip():
            return self._empty_result()

        sentences   = _split_sentences(response_text)
        fact_list   = _flatten_evidence(evidence)
        nli_model   = _get_nli_model()

        results     = []
        flagged     = []
        total_score = 0.0

        print(f"\n[HALL-CHECK] ━━━ check() called ━━━")
        print(f"[HALL-CHECK] response_text len={len(response_text)}: {repr(response_text[:120])}")
        print(f"[HALL-CHECK] sentences={len(sentences)} facts={len(fact_list)}")
        print(f"[HALL-CHECK] action={evidence.get('action','?')} skip={evidence.get('action','?') in {'no_retrieval','explanation_generate'}}")
        for _fact in fact_list[:5]: print(f"  [HALL-FACT] {_fact[:80]}")
        for sentence in sentences:
            if not _is_factual_sentence(sentence):
                print(f"[HALL-CHECK] SKIP non-factual: '{sentence[:70]}'")
                # Non-factual sentence — skip NLI check
                results.append({
                    "sentence":       sentence,
                    "is_factual":     False,
                    "checked":        False,
                    "passed":         True,
                    "nli_scores":     None,
                    "best_evidence":  None,
                })
                continue

            # Find most relevant evidence
            best_evidence = _find_best_evidence(sentence, fact_list)
            if not best_evidence:
                continue

            print(f"[HALL-CHECK] NLI checking: '{sentence[:70]}'")
            print(f"[HALL-CHECK] vs evidence: '{best_evidence[:80]}'")
            # Run NLI: (premise=evidence, hypothesis=sentence)
            scores = nli_model.predict([(best_evidence, sentence)])
            # scores shape: (1, 3) — [contradiction, neutral, entailment]
            score_dict = {
                "contradiction": float(scores[0][0]),
                "neutral":       float(scores[0][1]),
                "entailment":    float(scores[0][2]),
            }

            contradiction_score = score_dict["contradiction"]
            entailment_score    = score_dict["entailment"]

            # Determine if this sentence is a hallucination
            is_contradiction = contradiction_score > NLI_CONTRADICTION_THRESHOLD
            is_unsupported   = entailment_score < NLI_ENTAILMENT_THRESHOLD

            is_hallucination = is_contradiction or is_unsupported
            sentence_score   = contradiction_score if is_contradiction else (
                1.0 - entailment_score if is_unsupported else 0.0
            )
            total_score += sentence_score

            result = {
                "sentence":         sentence,
                "is_factual":       True,
                "checked":          True,
                "passed":           not is_hallucination,
                "nli_scores":       score_dict,
                "best_evidence":    best_evidence,
                "is_contradiction": is_contradiction,
                "is_unsupported":   is_unsupported,
            }
            results.append(result)

            print(f"[HALL-CHECK] NLI scores: contra={score_dict['contradiction']:.4f} neutral={score_dict['neutral']:.4f} entail={score_dict['entailment']:.4f}")
            print(f"[HALL-CHECK] → is_contradiction={is_contradiction} is_unsupported={is_unsupported} HALLUCINATION={is_hallucination}")
            if is_hallucination:
                flagged.append(result)

        print(f"[HALL-CHECK] ─── loop done: checked={len([r for r in results if r['checked']])} flagged={len(flagged)}")
        factual_checked = [r for r in results if r["checked"]]
        n_checked       = len(factual_checked)
        avg_score       = total_score / n_checked if n_checked > 0 else 0.0
        has_hallucination = len(flagged) > 0

        print(f"[HALL-CHECK] RESULT: has_hallucination={has_hallucination} score={round(avg_score,3)} n_flagged={len(flagged)}")
        return {
            "has_hallucination":  has_hallucination,
            "hallucination_score":round(avg_score, 3),
            "flagged_sentences":  flagged,
            "all_sentences":      results,
            "n_factual_checked":  n_checked,
            "n_flagged":          len(flagged),
            "passed":             not has_hallucination,
        }

    def _empty_result(self) -> dict:
        return {
            "has_hallucination":  False,
            "hallucination_score":0.0,
            "flagged_sentences":  [],
            "all_sentences":      [],
            "n_factual_checked":  0,
            "n_flagged":          0,
            "passed":             True,
        }
