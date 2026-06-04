"""
extract_real_data.py
────────────────────
Extracts all classified user turns from MongoDB (sessions collection) into a
CSV file matching the column structure of v4_train_midSession.csv, with an
extra `confidence` column at the end.

Rules:
  - For each user turn with a stored classification, get the 2 turns that came
    immediately before it in the session (regardless of user/bot role).
  - Build input_text in the same [SEP]-joined format as the training data.
  - If no prior turns exist, input_text is just: CURRENT: <message>
  - No confidence filter — all turns are exported. Filter from `confidence` later.

Output: data/real_data_from_mongodb.csv  (default)

Usage:
    python extract_real_data.py
    python extract_real_data.py --output data/my_output.csv
"""

import os
import csv
import json
import argparse
from collections import Counter
from dotenv import load_dotenv
from pymongo import MongoClient

# ── Load .env from project root (3 levels up from this file) ─────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

# ── Label definitions (must match config.py order exactly) ───────────────────
LABEL_NAMES = [
    "INITIAL_REQUEST",       # 0 → FULL
    "REFINEMENT",            # 1 → FULL
    "ATTRIBUTE_QUESTION",    # 2 → PARTIAL
    "EXPLANATION_WHY",       # 3 → PARTIAL
    "COMPARISON",            # 4 → PARTIAL
    "SELECTION_REFERENCE",   # 5 → PARTIAL
    "FEEDBACK",              # 6 → NO
    "CHITCHAT",              # 7 → NO
]
LABEL_NAME_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

RETRIEVAL_STRATEGY_MAP = {
    "INITIAL_REQUEST":    "FULL",
    "REFINEMENT":         "FULL",
    "ATTRIBUTE_QUESTION": "PARTIAL",
    "EXPLANATION_WHY":    "PARTIAL",
    "COMPARISON":         "PARTIAL",
    "SELECTION_REFERENCE":"PARTIAL",
    "FEEDBACK":           "NO",
    "CHITCHAT":           "NO",
}

CSV_COLUMNS = [
    "input_text",
    "current_message",
    "conversation_history_json",
    "label",
    "label_name",
    "retrieval_strategy",
    "exchanges",
    "confidence",
]


def _normalize_role(role: str) -> str:
    """Map any role value to USER or BOT for the input_text format."""
    r = role.upper()
    if r in ("ASSISTANT", "BOT"):
        return "BOT"
    return "USER"


def _build_input_text(prior_turns: list, current_message: str) -> str:
    """
    Build the [SEP]-joined string used as model input.

    Format mirrors v4_train_midSession.csv training data:
      With prior turns:  USER: <p1> [SEP] BOT: <p2> [SEP] CURRENT: <msg>
      Without prior turns: CURRENT: <msg>
    """
    parts = []
    for turn in prior_turns:
        role    = _normalize_role(turn.get("role", "user"))
        content = turn.get("content", "").strip()
        if content:
            parts.append(f"{role}: {content}")
    parts.append(f"CURRENT: {current_message.strip()}")
    return " [SEP] ".join(parts)


def _build_row(turn: dict, prior_turns: list) -> dict:
    """Build one CSV row dict from a classified user turn and its prior context."""
    classification = turn["classification"]
    current_message = turn["content"].strip()

    input_text = _build_input_text(prior_turns, current_message)

    conv_history = [
        {"role": t.get("role", ""), "content": t.get("content", "")}
        for t in prior_turns
    ]

    label_name = classification.get("label", "")
    retrieval_strategy = classification.get(
        "retrieval_strategy",
        RETRIEVAL_STRATEGY_MAP.get(label_name, "")
    )

    return {
        "input_text":                input_text,
        "current_message":           current_message,
        "conversation_history_json": json.dumps(conv_history),
        "label":                     LABEL_NAME_TO_ID.get(label_name, ""),
        "label_name":                label_name,
        "retrieval_strategy":        retrieval_strategy,
        "exchanges":                 len(prior_turns) if prior_turns else None,
        "confidence":                classification.get("confidence", ""),
    }


def _is_classified_user_turn(turn: dict) -> bool:
    return (
        (turn.get("role") or "").lower() == "user"
        and turn.get("classification")
        and (turn.get("content") or "").strip()
    )


def extract_rows(sessions_collection) -> list:
    rows = []
    for session in sessions_collection.find({}, {"turns": 1, "session_id": 1}):
        turns = session.get("turns", [])
        for i, turn in enumerate(turns):
            if not _is_classified_user_turn(turn):
                continue
            prior_turns = turns[max(0, i - 2):i]
            rows.append(_build_row(turn, prior_turns))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract MongoDB turns to CSV")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "data", "real_data_from_mongodb.csv"),
        help="Output CSV file path (default: data/real_data_from_mongodb.csv)",
    )
    args = parser.parse_args()

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name     = os.getenv("MONGODB_DB_NAME", "sunlytics_crs")

    print(f"Connecting to MongoDB: {mongodb_url}  db={db_name}")
    client = MongoClient(mongodb_url)
    db     = client[db_name]

    print("Scanning sessions collection...")
    rows = extract_rows(db.sessions)
    client.close()

    print(f"  Extracted {len(rows)} classified user turns")

    if not rows:
        print("No rows found. Check that sessions exist and turns have classification data.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.output}")

    # ── Label distribution summary ────────────────────────────────────────────
    label_counts = Counter(r["label_name"] for r in rows)
    print("\nLabel distribution:")
    for label in LABEL_NAMES:
        count = label_counts.get(label, 0)
        bar   = "#" * min(count, 40)
        print(f"  {label:<25} {count:>4}  {bar}")

    # ── Confidence summary ────────────────────────────────────────────────────
    conf_values = [r["confidence"] for r in rows if isinstance(r["confidence"], (int, float))]
    if conf_values:
        avg_conf = sum(conf_values) / len(conf_values)
        low_conf = sum(1 for c in conf_values if c < 0.90)
        print("\nConfidence stats:")
        print(f"  Average confidence : {avg_conf:.4f}")
        print(f"  Turns < 0.90 conf  : {low_conf} / {len(conf_values)}")
        print("  (To get low-confidence only, filter: confidence < 0.90)")


if __name__ == "__main__":
    main()
