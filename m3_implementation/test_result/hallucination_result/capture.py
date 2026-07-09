# m3_implementation/test_result/hallucination_result/capture.py
#
# Evaluation capture hook — persists (evidence, response) pairs produced by
# the RAG pipeline into a JSONL file. These pairs are the raw material for
# the hallucination evaluation test set (see corrupt_cases.py, next step).
#
# Disabled by default: does nothing unless the environment variable
# EVAL_CAPTURE=1 is set. collect_cases.py sets it automatically.
#
# One JSONL line per generation attempt:
#   {
#     "captured_at":   ISO timestamp,
#     "session_id":    session the turn belongs to,
#     "user_message":  the user message that triggered this response,
#     "action":        catalog_search / item_compare / ...,
#     "attempt":       1-3 (regeneration attempt number),
#     "evidence":      full evidence bundle given to the LLM,
#     "response_text": the LLM response that was checked,
#     "checker":       summary of the hallucination check on this attempt
#   }

import json
import os
from datetime import datetime, timezone

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "captured_cases.jsonl")

# Item/article fields the checker's _flatten_evidence() actually reads,
# plus article_id for traceability and material_description for the
# LLM-judge baseline (responses often quote the description text).
_ITEM_FIELDS = (
    "article_id", "name", "type", "colour", "price", "pattern",
    "index_group", "section", "material_description",
)


def _slim_item(item: dict) -> dict:
    return {k: item.get(k) for k in _ITEM_FIELDS if item.get(k) is not None}


def slim_evidence(evidence: dict) -> dict:
    """Keeps only the evidence fields the hallucination check verifies against.

    Drops personalisation payloads (user_preferences, purchase_hints,
    style_profile, items_in_context, ...) that the checker never reads —
    they only bloat the capture file.
    """
    if not evidence:
        return {}
    slim = {"action": evidence.get("action", "")}

    if evidence.get("items") is not None:
        slim["items"] = [_slim_item(i) for i in evidence.get("items", []) if i]
    for key in ("article", "item_a", "item_b"):
        if evidence.get(key):
            slim[key] = _slim_item(evidence[key])
    for key in ("extracted_facts", "comparison_facts", "filters_applied",
                "preference_boosts", "confirmed_matches", "prior_claims"):
        if evidence.get(key):
            slim[key] = evidence[key]
    return slim


def capture_enabled() -> bool:
    return os.getenv("EVAL_CAPTURE", "0") == "1"


def capture_path() -> str:
    return os.getenv("EVAL_CAPTURE_PATH", _DEFAULT_PATH)


def capture_case(
    evidence:      dict,
    response_text: str,
    action:        str,
    attempt:       int,
    session_id:    str  = "",
    user_message:  str  = "",
    check_result:  dict = None,
) -> None:
    """Appends one evaluation case to the capture file. No-op unless EVAL_CAPTURE=1."""
    if not capture_enabled():
        return

    check = check_result or {}
    record = {
        "captured_at":   datetime.now(timezone.utc).isoformat(),
        "session_id":    session_id,
        "user_message":  user_message,
        "action":        action,
        "attempt":       attempt,
        "evidence":      slim_evidence(evidence),
        "response_text": response_text,
        "checker": {
            "passed":              check.get("passed"),
            "n_checked":           check.get("n_checked"),
            "n_flagged":           check.get("n_flagged"),
            "hallucination_score": check.get("hallucination_score"),
            "contradicted_fields": check.get("contradicted_fields", []),
        },
    }

    path = capture_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            # default=str handles Decimal / datetime values from PostgreSQL
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"[EvalCapture] case saved: action={action} attempt={attempt}")
    except Exception as e:
        print(f"[EvalCapture] write failed (non-fatal): {e}")
