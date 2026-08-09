"""
Latency logger for adaptive retrieval evaluation.
Appends one CSV row per turn to latency_log.csv in the project root.
"""
import csv
import time
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent.parent / "latency_log.csv"
_HEADERS  = [
    "timestamp", "session_id", "turn_id", "label",
    "tier", "sub_tier", "user_message",
    "memory_ms", "rag_ms", "total_ms", "response_status",
    "has_context", "context_source",
]


def _ensure_header():
    if not _LOG_PATH.exists():
        with open(_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_HEADERS)


def determine_tier(pipeline_output: dict) -> tuple[str, str]:
    """
    Returns (tier, sub_tier) from pipeline_output.
      tier     : NO | FULL | PARTIAL
      sub_tier : — | STANDARD | EXCLUSIONS | RECENT | SESSION
    """
    strategy    = pipeline_output.get("retrieval_strategy", "NO")
    cse         = pipeline_output.get("cse") or {}
    exclude_ids = (pipeline_output.get("retrieval_input") or {}).get("exclude_ids") or []

    if strategy == "NO":
        return "NO", "—"
    if strategy == "FULL":
        sub = "EXCLUSIONS" if exclude_ids else "STANDARD"
        return "FULL", sub
    if strategy == "PARTIAL":
        partial_sub = cse.get("partial_subtype", "")
        sub = "RECENT" if partial_sub == "PARTIAL_RECENT" else "SESSION"
        return "PARTIAL", sub
    return "UNKNOWN", "—"


def compute_response_status(rag_result: dict) -> str:
    """
    Derives a single status string from rag_result quality flags.
      OK          — clean response, no issues
      HALLUCINATION — hallucination flag raised and not resolved
      CONTRADICTION — cross-turn contradiction detected
      HALL+CONTRA — both flags raised
      ERROR       — rag_result is empty / missing (pipeline exception)
    """
    if not rag_result:
        return "ERROR"
    hall   = rag_result.get("hallucination_flag", False)
    contra = rag_result.get("contradiction_found", False)
    if hall and contra:
        return "HALL+CONTRA"
    if hall:
        return "HALLUCINATION"
    if contra:
        return "CONTRADICTION"
    return "OK"


def context_available(pipeline_output: dict) -> bool:
    """
    Was there anything in the session context window when this turn was routed?

    WHY THIS IS LOGGED
      The routing evaluation (E2) scores the chosen tier against a table keyed
      on the intent label alone. That table cannot be right on its own, because
      the correct tier also depends on conversation state:

          "tell me more about the first one"
              items in context  -> PARTIAL is correct (reuse them)
              context empty     -> PARTIAL is IMPOSSIBLE; escalating to FULL is
                                   the correct recovery, yet the label-only
                                   table still says PARTIAL and marks it a
                                   misroute.

      Recording this flag lets the expected tier be state-aware, so the context
      evaluator stops being penalised for the very overrides it exists to make.

    `items_in_context` is built by enrichment._items_dict() from the dialogue
    state's context window (up to 12 item slots), so a non-empty dict means the
    session genuinely had items available to reuse.
    """
    ri = (pipeline_output or {}).get("retrieval_input") or {}
    return bool(ri.get("items_in_context") or {})


def log_turn(
    session_id:      str,
    turn_id:         str,
    label:           str,
    tier:            str,
    sub_tier:        str,
    user_message:    str,
    memory_ms:       float,
    rag_ms:          float,
    total_ms:        float,
    response_status: str,
    has_context:     bool = None,
) -> None:
    _ensure_header()
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        session_id, turn_id, label,
        tier, sub_tier,
        user_message,
        round(memory_ms, 1),
        round(rag_ms, 1),
        round(total_ms, 1),
        response_status,
        "" if has_context is None else bool(has_context),
        # Rows written by this function observed the state directly. Rows
        # backfilled by the migration script are marked "derived" so the two
        # are never silently mixed in an analysis.
        "logged" if has_context is not None else "",
    ]
    with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    print(
        f"[LATENCY] tier={tier:<8} sub={sub_tier:<11} "
        f"memory={round(memory_ms):>5}ms  "
        f"rag={round(rag_ms):>5}ms  "
        f"total={round(total_ms):>5}ms  "
        f"status={response_status}  "
        f"ctx={'yes' if has_context else 'no'}"
    )
