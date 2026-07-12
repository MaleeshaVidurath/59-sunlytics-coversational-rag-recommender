# m3_implementation/test_result/contradiction_result/capture.py
#
# Contradiction-evaluation capture hook — Step 1 of the contradiction
# detector evaluation (see EVALUATION_PLAN.md, section 2, Experiment A).
#
# Persists one JSONL record per contradiction-checked turn, containing
# everything the injection script (Step 1 → corrupt_sessions.py) needs:
#
#   - product_refs      : ground-truth values from the current evidence bundle
#   - graph_before      : session-graph product nodes BEFORE this turn's
#                         update — tells us which products were established in
#                         PRIOR turns and with what values (first_seen_turn /
#                         last_seen_turn give the turn-distance stratification)
#   - turn_ordinal      : 1-based position of this checked turn within its
#                         session (in-process counter), so first_seen_turn ids
#                         can be resolved to ordinals and distance computed
#   - extracted_claims  : what Groq said the LLM wrote (live extraction)
#   - response_in/out   : response before / after the live detector's fix
#   - contradictions    : what the live detector found (usually empty — clean)
#
# Disabled by default: does nothing unless CONTRA_EVAL_CAPTURE=1.
# collect_sessions.py sets it automatically.

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "captured_sessions.jsonl")

# In-process per-session counter of checked turns. Valid because the
# collection driver runs all conversations sequentially in one process.
_session_counters: dict = defaultdict(int)

# Graph node fields worth keeping (everything _update_graph_nodes writes)
_NODE_FIELDS = (
    "name", "colour", "price", "product_type", "pattern",
    "index_group", "section", "garment_group",
    "first_seen_turn", "last_seen_turn",
)


def capture_enabled() -> bool:
    return os.getenv("CONTRA_EVAL_CAPTURE", "0") == "1"


def capture_path() -> str:
    return os.getenv("CONTRA_EVAL_CAPTURE_PATH", _DEFAULT_PATH)


def _slim_nodes(graph_before: dict) -> dict:
    """Keeps only product-node fields the evaluation needs."""
    slim = {}
    for aid, attrs in (graph_before or {}).items():
        if attrs.get("type") == "contradiction_event":
            continue
        slim[str(aid)] = {k: attrs.get(k) for k in _NODE_FIELDS if attrs.get(k)}
    return slim


def capture_case(
    session_id:        str,
    turn_id:           str,
    action:            str,
    product_refs:      list,
    graph_before:      dict,
    extracted_claims:  dict,
    response_in:       str,
    response_out:      str,
    contradictions:    list,
    collection_prefix: str = "m3",
) -> None:
    """Appends one contradiction-eval case. No-op unless CONTRA_EVAL_CAPTURE=1."""
    if not capture_enabled():
        return

    _session_counters[session_id] += 1

    record = {
        "captured_at":       datetime.now(timezone.utc).isoformat(),
        "session_id":        session_id,
        "turn_id":           turn_id,
        "turn_ordinal":      _session_counters[session_id],
        "action":            action,
        "product_refs":      product_refs,
        "graph_before":      _slim_nodes(graph_before),
        "extracted_claims":  extracted_claims,
        "response_in":       response_in,
        "response_out":      response_out,
        "live_contradictions": contradictions,
        "member_model":      collection_prefix,
    }

    path = capture_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            # default=str handles Decimal / datetime values from PostgreSQL
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"[ContraEvalCapture] case saved: session={session_id[:12]} "
              f"ordinal={record['turn_ordinal']} action={action} "
              f"graph_nodes={len(record['graph_before'])}")
    except Exception as e:
        print(f"[ContraEvalCapture] write failed (non-fatal): {e}")
