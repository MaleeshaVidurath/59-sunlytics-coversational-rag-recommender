"""
preprocess_simmc.py
────────────────────
Preprocesses the SIMMC 2.1 fashion dialogue dataset into a CSV file
matching the column structure of v4_train_midSession.csv, ready for
DistilBERT intent classifier retraining.

Source  : simmc2.1_dials_dstc11_train.json  (Facebook AI Research, EMNLP 2021)
Output  : data/real_data_simmc.csv

Preprocessing pipeline (9 steps):
  Step 0  Load raw data + filter fashion domain
  Step 1  Extract user turns with SIMMC act labels
  Step 2  Map SIMMC acts -> 8 intent labels
  Step 3  Filter visual reference turns
  Step 4  Filter very short turns (< 5 chars)
  Step 5  Normalize text
  Step 6  Build input_text with up to 2 prior turns [SEP] format
  Step 7  Build conversation_history_json + count exchanges
  Step 8  Save CSV
  Step 9  Print full report

Usage:
    python preprocess_simmc.py
    python preprocess_simmc.py --input path/to/simmc2.1_dials_dstc11_train.json
    python preprocess_simmc.py --output data/my_output.csv
"""

import os
import re
import csv
import json
import argparse
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_BASE_DIR, "data_sets_from_researchers",
                            "simc-upload", "simmc2-main", "data")
_DEFAULT_IN = os.path.join(_DATA_DIR, "simmc2.1_dials_dstc11_train.json")
_DEFAULT_OUT = os.path.join(_BASE_DIR, "data", "real_data_simmc.csv")

# ── Label mapping ─────────────────────────────────────────────────────────────
# Maps SIMMC 2.1 dialogue acts to our 8 intent labels.
#
# SIMMC act taxonomy:
#   REQUEST:GET          — user requests items matching a description
#   INFORM:REFINE        — user narrows/changes the current search
#   INFORM:GET           — user provides attributes to get a recommendation
#   ASK:GET              — user asks about an attribute of a shown item
#   REQUEST:COMPARE      — user requests comparison between two items
#   INFORM:DISAMBIGUATE  — user points to a specific item for more detail
#   REQUEST:ADD_TO_CART  — user selects/buys an item (positive feedback signal)
#
# Mapping rationale:
#   REQUEST:GET          -> INITIAL_REQUEST    : fresh product search query
#   INFORM:REFINE        -> REFINEMENT         : narrows same search category
#   INFORM:GET           -> REFINEMENT         : provides attributes to refine
#   ASK:GET              -> ATTRIBUTE_QUESTION : asks about item attribute
#   REQUEST:COMPARE      -> COMPARISON         : compares two shown items
#   INFORM:DISAMBIGUATE  -> SELECTION_REFERENCE: points to specific shown item
#   REQUEST:ADD_TO_CART  -> FEEDBACK           : positive selection signal
LABEL_MAP = {
    "REQUEST:GET":         ("INITIAL_REQUEST",    0, "FULL"),
    "INFORM:REFINE":       ("REFINEMENT",         1, "FULL"),
    "INFORM:GET":          ("REFINEMENT",         1, "FULL"),
    "ASK:GET":             ("ATTRIBUTE_QUESTION", 2, "PARTIAL"),
    "REQUEST:COMPARE":     ("COMPARISON",         4, "PARTIAL"),
    "INFORM:DISAMBIGUATE": ("SELECTION_REFERENCE",5, "PARTIAL"),
    "REQUEST:ADD_TO_CART": ("FEEDBACK",           6, "NO"),
}

CSV_COLUMNS = [
    "input_text",
    "current_message",
    "conversation_history_json",
    "label",
    "label_name",
    "retrieval_strategy",
    "exchanges",
]

# ── Visual reference filter patterns ─────────────────────────────────────────
# These patterns identify turns that reference items by physical position in
# the virtual shopping scene (e.g. "the one on the left wall", "second from
# the right"). Such turns are meaningless in a text-only CRS because the
# positional reference has no text equivalent.
_VISUAL_PATTERNS = [
    r"\b(left|right)\b.*\b(wall|row|side)\b",
    r"\b(top|bottom)\s+(left|right|row)\b",
    r"\b(second|third|fourth|fifth|first|1st|2nd|3rd|4th|5th)\s+from\s+the\s+(left|right)\b",
    r"\bthe\s+\w+\s+(on\s+the\s+)?(left|right|top|bottom|middle|front|back|far)\b",
    r"\bover\s+(there|here)\b",
    r"\bnext\s+to\s+(it|them|the)\b",
    r"\bin\s+the\s+(middle|center|corner)\b",
    r"\bhanging\s+on\s+the\b",
    r"\bthat\s+one\b",
    r"\bthose\s+(two|ones|items|jackets|dresses|shirts|blouses|hoodies|shoes|coats|skirts|pants|trousers)\b",
]
_VISUAL_RE = re.compile("|".join(_VISUAL_PATTERNS), re.IGNORECASE)

MIN_LENGTH = 5  # minimum character length for a valid user turn


# ── Text normalisation ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Applies minimal text normalization:
      - Strip leading/trailing whitespace
      - Collapse multiple spaces into one
      - Remove zero-width and non-printable characters
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", "", text)
    return text.strip()


# ── input_text builder ────────────────────────────────────────────────────────

def _build_input_text(prior_turns: list, current_message: str) -> str:
    """
    Builds the [SEP]-joined input string used as DistilBERT input.

    Format (with 2 prior turns):
        USER: <prev_user> [SEP] BOT: <prev_bot> [SEP] CURRENT: <message>

    Format (with 1 prior turn):
        USER: <prev_user> [SEP] CURRENT: <message>

    Format (no prior turns):
        CURRENT: <message>

    Roles map to:  user -> USER,  system/assistant -> BOT
    """
    parts = []
    for turn in prior_turns:
        role    = "BOT" if turn["role"] in ("system", "assistant", "bot") else "USER"
        content = _normalize(turn["content"])
        if content:
            parts.append(f"{role}: {content}")
    parts.append(f"CURRENT: {_normalize(current_message)}")
    return " [SEP] ".join(parts)


# ── Main preprocessing pipeline ───────────────────────────────────────────────

def preprocess(input_path: str, output_path: str) -> list:
    print("=" * 60)
    print("SIMMC 2.1 -> DistilBERT Training Data Preprocessing")
    print("=" * 60)

    # ── Step 0: Load raw data + filter fashion domain ──────────────────────
    print("\nStep 0 — Loading raw data and filtering fashion domain")
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    all_dialogues  = raw["dialogue_data"]
    fashion        = [d for d in all_dialogues if d["domain"] == "fashion"]
    furniture      = [d for d in all_dialogues if d["domain"] != "fashion"]

    total_turns_all     = sum(len(d["dialogue"]) for d in all_dialogues)
    total_turns_fashion = sum(len(d["dialogue"]) for d in fashion)

    print(f"  Total dialogues (all domains)  : {len(all_dialogues)}")
    print(f"  Furniture dialogues removed    : {len(furniture)}")
    print(f"  Fashion dialogues kept         : {len(fashion)}")
    print(f"  Total user turns (all)         : {total_turns_all}")
    print(f"  Total user turns (fashion)     : {total_turns_fashion}")

    # ── Step 1: Extract user turns with SIMMC act labels ───────────────────
    print("\nStep 1 — Extracting user turns with SIMMC act labels")
    raw_turns = []
    for dial in fashion:
        turns = dial["dialogue"]
        for idx, turn in enumerate(turns):
            act  = turn["transcript_annotated"]["act"]
            text = turn["transcript"]
            raw_turns.append({
                "dialogue_idx": dial["dialogue_idx"],
                "turn_idx":     turn["turn_idx"],
                "text":         text,
                "act":          act,
                "dialogue":     turns,   # full dialogue for context building
                "position":     idx,
            })

    act_dist = Counter(t["act"] for t in raw_turns)
    print(f"  Total user turns extracted : {len(raw_turns)}")
    print(f"  Act distribution:")
    for act, cnt in sorted(act_dist.items(), key=lambda x: -x[1]):
        print(f"    {act:<30} {cnt:>5}")

    # ── Step 2: Map SIMMC acts -> 8 intent labels ───────────────────────────
    print("\nStep 2 — Mapping SIMMC acts to 8 intent labels")
    labelled_turns = []
    unmapped = Counter()
    for t in raw_turns:
        if t["act"] in LABEL_MAP:
            label_name, label_id, strategy = LABEL_MAP[t["act"]]
            t["label_name"]         = label_name
            t["label"]              = label_id
            t["retrieval_strategy"] = strategy
            labelled_turns.append(t)
        else:
            unmapped[t["act"]] += 1

    print(f"  Turns after label mapping : {len(labelled_turns)}")
    if unmapped:
        print(f"  Unmapped acts (dropped)   : {dict(unmapped)}")
    else:
        print(f"  Unmapped acts             : none (all acts covered)")

    label_dist = Counter(t["label_name"] for t in labelled_turns)
    print(f"  Label distribution after mapping:")
    for label, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"    {label:<25} {cnt:>5}")

    # ── Step 3: Filter visual reference turns ──────────────────────────────
    print("\nStep 3 — Filtering visual reference turns")
    visual_examples = []
    non_visual = []
    for t in labelled_turns:
        if _VISUAL_RE.search(t["text"]):
            if len(visual_examples) < 5:
                visual_examples.append(t["text"])
        else:
            non_visual.append(t)

    removed_visual = len(labelled_turns) - len(non_visual)
    print(f"  Turns removed (visual refs)  : {removed_visual}")
    print(f"  Turns remaining              : {len(non_visual)}")
    print(f"  Example visual ref turns removed:")
    for ex in visual_examples:
        print(f"    - \"{ex[:90]}\"")

    # ── Step 4: Filter very short turns ────────────────────────────────────
    print(f"\nStep 4 — Filtering turns shorter than {MIN_LENGTH} characters")
    long_enough = [t for t in non_visual if len(t["text"].strip()) >= MIN_LENGTH]
    removed_short = len(non_visual) - len(long_enough)
    print(f"  Turns removed (too short) : {removed_short}")
    print(f"  Turns remaining           : {len(long_enough)}")

    # ── Step 5: Normalize text ─────────────────────────────────────────────
    print("\nStep 5 — Normalizing text")
    for t in long_enough:
        t["text"] = _normalize(t["text"])
    print(f"  Applied: strip whitespace, collapse spaces, remove non-ASCII")
    print(f"  Turns after normalization : {len(long_enough)}")

    # ── Step 6: Build input_text with up to 2 prior turns ─────────────────
    print("\nStep 6 — Building input_text with up to 2 prior turns [SEP] format")
    for t in long_enough:
        pos      = t["position"]
        dialogue = t["dialogue"]

        # Get up to 2 turns immediately before the current user turn
        raw_prior = dialogue[max(0, pos - 2):pos]
        prior_turns = []
        for p in raw_prior:
            user_text = p.get("transcript", "").strip()
            bot_text  = p.get("system_transcript", "").strip()
            if user_text:
                prior_turns.append({"role": "user",   "content": user_text})
            if bot_text:
                prior_turns.append({"role": "system", "content": bot_text})

        # Limit to last 2 turns for context
        prior_turns = prior_turns[-2:]
        t["prior_turns"]   = prior_turns
        t["input_text"]    = _build_input_text(prior_turns, t["text"])
        t["exchanges"]     = len(prior_turns) if prior_turns else None

    no_context  = sum(1 for t in long_enough if not t["prior_turns"])
    one_context = sum(1 for t in long_enough if len(t["prior_turns"]) == 1)
    two_context = sum(1 for t in long_enough if len(t["prior_turns"]) == 2)
    print(f"  Turns with no prior context  : {no_context}")
    print(f"  Turns with 1 prior turn      : {one_context}")
    print(f"  Turns with 2 prior turns     : {two_context}")

    # ── Step 7: Build conversation_history_json ────────────────────────────
    print("\nStep 7 — Building conversation_history_json")
    rows = []
    for t in long_enough:
        conv_history_json = json.dumps(t["prior_turns"])
        rows.append({
            "input_text":                t["input_text"],
            "current_message":           t["text"],
            "conversation_history_json": conv_history_json,
            "label":                     t["label"],
            "label_name":                t["label_name"],
            "retrieval_strategy":        t["retrieval_strategy"],
            "exchanges":                 t["exchanges"] if t["exchanges"] else "",
        })
    print(f"  Total rows ready for CSV : {len(rows)}")

    # ── Step 8: Save CSV ───────────────────────────────────────────────────
    print(f"\nStep 8 — Saving CSV")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {output_path}")

    # ── Step 9: Full report ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 9 — Final Report")
    print("=" * 60)

    final_label_dist = Counter(r["label_name"] for r in rows)
    print(f"\nFinal label distribution ({len(rows)} total rows):")
    label_order = [
        "INITIAL_REQUEST", "REFINEMENT", "ATTRIBUTE_QUESTION",
        "EXPLANATION_WHY", "COMPARISON", "SELECTION_REFERENCE",
        "FEEDBACK", "CHITCHAT",
    ]
    for label in label_order:
        cnt = final_label_dist.get(label, 0)
        bar = "#" * min(cnt // 50, 40)
        print(f"  {label:<25} {cnt:>5}  {bar}")

    missing = [l for l in label_order if final_label_dist.get(l, 0) == 0]
    if missing:
        print(f"\n  Missing labels (not in SIMMC): {missing}")
        print(f"  -> These will be filled from synthetic data in the mixing step.")

    print("\nSample rows per label:")
    shown = set()
    for r in rows:
        lbl = r["label_name"]
        if lbl not in shown:
            print(f"\n  [{lbl}]")
            print(f"    current_message : {r['current_message'][:80]}")
            print(f"    input_text      : {r['input_text'][:100]}")
            shown.add(lbl)
        if len(shown) == 6:
            break

    print("\n" + "=" * 60)
    print(f"Preprocessing complete. Output: {output_path}")
    print("=" * 60)

    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SIMMC 2.1 -> DistilBERT training CSV"
    )
    parser.add_argument(
        "--input",
        default=_DEFAULT_IN,
        help="Path to simmc2.1_dials_dstc11_train.json",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUT,
        help="Output CSV path (default: data/real_data_simmc.csv)",
    )
    args = parser.parse_args()
    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
