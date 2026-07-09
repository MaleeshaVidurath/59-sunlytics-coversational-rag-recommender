# m3_implementation/test_result/hallucination_result/corrupt_cases.py
#
# Step 2 of the hallucination checker evaluation — builds the labeled test set.
#
# METHOD (FactCC-style synthetic corruption; Kryscinski et al. 2020, HaluEval 2023):
#   Take each captured (evidence, response) pair and corrupt ONE field value in
#   the RESPONSE while leaving the EVIDENCE untouched. The corrupted variant is
#   guaranteed to be a hallucination by construction — the response now states
#   something the evidence contradicts. The uncorrupted original is kept as a
#   presumed-clean case (label noise possible; see audit note below).
#
# CORRUPTION TYPES (mirror the checker's core fields + its novelty target):
#   colour_swap        — item colour replaced with a colour NOT in the evidence
#   price_change       — £value replaced with a different price
#   name_swap          — product name replaced with a different catalog name
#   cross_item_swap    — colours of item A and item B exchanged (catalog_search
#                        with ≥2 items only). Targets the item→sentence lock map:
#                        every value still exists in the evidence, just attached
#                        to the wrong item.
#
# LABELS:
#   "hallucinated" — corrupted variants (ground truth certain)
#   "clean"        — original responses that the checker passed
#                    (presumed correct; manually audit via clean_audit.txt —
#                    do NOT treat checker-pass as ground truth, that is circular)
#   Originals the checker FLAGGED are excluded from the clean set and written
#   to flagged_for_review.jsonl for manual adjudication (they may be true
#   positives or false positives — deciding which requires a human).
#
# Deterministic: seeded RNG → same input file always yields the same test set.
#
# Run:  python test_result/hallucination_result/corrupt_cases.py

import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

_DIR         = os.path.dirname(os.path.abspath(__file__))
CAPTURED     = os.path.join(_DIR, "captured_cases.jsonl")
TEST_SET     = os.path.join(_DIR, "labeled_test_set.jsonl")
FLAGGED_OUT  = os.path.join(_DIR, "flagged_for_review.jsonl")
AUDIT_OUT    = os.path.join(_DIR, "clean_audit.txt")

SEED = 42

# H&M colour vocabulary for swaps (colour_group_name values)
COLOUR_POOL = [
    "Black", "White", "Red", "Blue", "Dark Blue", "Light Blue", "Pink",
    "Dark Pink", "Green", "Dark Green", "Yellow", "Grey", "Beige",
    "Brown", "Orange", "Purple",
]


# ── Evidence helpers ─────────────────────────────────────────────────────────

def _evidence_items(evidence: dict) -> list[dict]:
    """Returns the list of product dicts referenced by this evidence bundle."""
    action = evidence.get("action", "")
    if action == "catalog_search":
        return evidence.get("items", []) or []
    if action == "item_compare":
        return [x for x in (evidence.get("item_a"), evidence.get("item_b")) if x]
    article = evidence.get("article")
    return [article] if article else []


def _replace_once_ci(text: str, old: str, new: str) -> str | None:
    """Replaces ALL case-insensitive occurrences of `old` with `new`.
    Returns None if `old` does not occur in `text`."""
    if not old:
        return None
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    if not pattern.search(text):
        return None
    return pattern.sub(new, text)


# ── Corruption generators ────────────────────────────────────────────────────
# Each returns a list of variant dicts:
#   {"response_text", "corruption": {type, field, item_idx, original, corrupted}}

def _corrupt_colour(response: str, items: list[dict], rng: random.Random) -> list[dict]:
    variants = []
    evidence_colours = {str(i.get("colour", "")).lower() for i in items}
    for idx, item in enumerate(items):
        colour = item.get("colour") or ""
        if not colour:
            continue
        # Pick a replacement colour that appears NOWHERE in the evidence, so the
        # corrupted sentence cannot accidentally be true for another item.
        candidates = [c for c in COLOUR_POOL
                      if c.lower() != colour.lower()
                      and c.lower() not in evidence_colours
                      # avoid substring collisions like Blue → Dark Blue
                      and colour.lower() not in c.lower()
                      and c.lower() not in colour.lower()]
        if not candidates:
            continue
        new_colour = rng.choice(candidates)
        corrupted = _replace_once_ci(response, colour, new_colour)
        if corrupted:
            variants.append({
                "response_text": corrupted,
                "corruption": {"type": "colour_swap", "field": "colour",
                               "item_idx": idx, "original": colour,
                               "corrupted": new_colour},
            })
    return variants


def _corrupt_price(response: str, items: list[dict], rng: random.Random) -> list[dict]:
    variants = []
    for idx, item in enumerate(items):
        price = str(item.get("price") or "")
        m = re.search(r"£([\d,]+\.?\d*)", price)
        if not m:
            continue
        old_value = f"£{m.group(1)}"
        if old_value not in response:
            continue
        try:
            amount = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        new_amount = round(amount + rng.choice([2.5, 3.75, 5.0, 7.25, -2.0, -4.5]), 2)
        if new_amount <= 0 or abs(new_amount - amount) < 0.01:
            new_amount = round(amount + 6.0, 2)
        new_value = f"£{new_amount:.2f}"
        variants.append({
            "response_text": response.replace(old_value, new_value),
            "corruption": {"type": "price_change", "field": "price",
                           "item_idx": idx, "original": old_value,
                           "corrupted": new_value},
        })
    return variants


def _corrupt_name(response: str, items: list[dict], name_pool: list[str],
                  rng: random.Random) -> list[dict]:
    variants = []
    evidence_names = {str(i.get("name", "")).lower() for i in items}
    for idx, item in enumerate(items):
        name = item.get("name") or ""
        if not name:
            continue
        candidates = [n for n in name_pool
                      if n.lower() not in evidence_names
                      # avoid prefix collisions like "London dress"/"SS London dress"
                      and n.lower() not in name.lower()
                      and name.lower() not in n.lower()]
        if not candidates:
            continue
        new_name = rng.choice(candidates)
        corrupted = _replace_once_ci(response, name, new_name)
        if corrupted:
            variants.append({
                "response_text": corrupted,
                "corruption": {"type": "name_swap", "field": "name",
                               "item_idx": idx, "original": name,
                               "corrupted": new_name},
            })
    return variants


def _item_field_value(item: dict, field: str) -> str:
    """Returns the swappable text value of a field ('' if unusable)."""
    if field == "price":
        m = re.search(r"£[\d,]+\.?\d*", str(item.get("price") or ""))
        return m.group(0) if m else ""
    return str(item.get(field) or "")


def _corrupt_cross_item(response: str, evidence: dict) -> list[dict]:
    """Swaps a field value between two items (colour, price, and name — one
    variant per field). Both values remain present in the evidence — only the
    item association is wrong. This is the exact failure mode the
    item→sentence lock map targets."""
    if evidence.get("action") != "catalog_search":
        return []
    items = evidence.get("items", []) or []
    variants = []

    for field in ("colour", "price", "name"):
        done = False
        for i in range(len(items)):
            if done:
                break
            for j in range(i + 1, len(items)):
                v1 = _item_field_value(items[i], field)
                v2 = _item_field_value(items[j], field)
                if not v1 or not v2 or v1.lower() == v2.lower():
                    continue
                # substring pairs (Blue / Dark Blue, London dress / SS London
                # dress) make the swap ambiguous — skip
                if v1.lower() in v2.lower() or v2.lower() in v1.lower():
                    continue
                p1 = re.compile(re.escape(v1), re.IGNORECASE)
                p2 = re.compile(re.escape(v2), re.IGNORECASE)
                if not (p1.search(response) and p2.search(response)):
                    continue
                placeholder = "[[SWAP-PLACEHOLDER]]"
                swapped = p1.sub(placeholder, response)
                swapped = p2.sub(v1, swapped)
                swapped = swapped.replace(placeholder, v2)
                variants.append({
                    "response_text": swapped,
                    "corruption": {"type": "cross_item_swap", "field": field,
                                   "item_idx": [i, j], "original": f"{v1} / {v2}",
                                   "corrupted": f"{v2} / {v1}"},
                })
                done = True
                break
    return variants


# ── Main ─────────────────────────────────────────────────────────────────────

def build_test_set():
    rng = random.Random(SEED)

    with open(CAPTURED, encoding="utf-8") as f:
        captured = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(captured)} captured cases from {os.path.basename(CAPTURED)}")

    # Global name pool for name_swap — every product name seen in any evidence
    name_pool = sorted({
        item.get("name") for case in captured
        for item in _evidence_items(case["evidence"]) if item.get("name")
    })

    test_cases   = []
    flagged      = []
    audit_lines  = []
    case_counter = 0

    def add_case(label, source_idx, case, response_text, corruption=None):
        nonlocal case_counter
        case_counter += 1
        test_cases.append({
            "case_id":       f"case_{case_counter:04d}",
            "label":         label,                    # "clean" | "hallucinated"
            "corruption":    corruption,               # None for clean
            "source_case":   source_idx,
            "action":        case["action"],
            "user_message":  case["user_message"],
            "evidence":      case["evidence"],
            "response_text": response_text,
        })

    for src_idx, case in enumerate(captured):
        response = case["response_text"]
        evidence = case["evidence"]
        items    = _evidence_items(evidence)

        if not case["checker"].get("passed", False):
            # Checker flagged this original — needs human adjudication, keep out
            # of both the clean set and the corruption base.
            flagged.append({**case, "source_case": src_idx})
            continue

        # Clean case (presumed correct — subject to manual audit)
        add_case("clean", src_idx, case, response)
        audit_lines.append(
            f"case_{case_counter:04d} [{case['action']}] "
            f"user: {case['user_message']}\n"
            f"  evidence: " + " | ".join(
                f"{i.get('name','?')} / {i.get('colour','?')} / {i.get('price','?')}"
                for i in items) + "\n"
            f"  response: {response}\n"
        )

        # Hallucinated variants
        variants = (
            _corrupt_colour(response, items, rng)
            + _corrupt_price(response, items, rng)
            + _corrupt_name(response, items, name_pool, rng)
            + _corrupt_cross_item(response, evidence)
        )
        for v in variants:
            add_case("hallucinated", src_idx, case, v["response_text"], v["corruption"])

    with open(TEST_SET, "w", encoding="utf-8") as f:
        for tc in test_cases:
            f.write(json.dumps(tc, ensure_ascii=False, default=str) + "\n")

    with open(FLAGGED_OUT, "w", encoding="utf-8") as f:
        for fc in flagged:
            f.write(json.dumps(fc, ensure_ascii=False, default=str) + "\n")

    with open(AUDIT_OUT, "w", encoding="utf-8") as f:
        f.write("MANUAL AUDIT — verify each 'clean' response truly matches its "
                "evidence.\nMark any wrong ones; they must be relabeled before "
                "computing final metrics.\n\n")
        f.write("\n".join(audit_lines))

    # ── Summary ──────────────────────────────────────────────────────────────
    by_label = Counter(tc["label"] for tc in test_cases)
    by_type  = Counter(tc["corruption"]["type"] for tc in test_cases if tc["corruption"])
    by_action = Counter(tc["action"] for tc in test_cases)

    print(f"\nTest set written: {os.path.basename(TEST_SET)}")
    print(f"  total cases:  {len(test_cases)}")
    print(f"  clean:        {by_label.get('clean', 0)}")
    print(f"  hallucinated: {by_label.get('hallucinated', 0)}")
    print("\nBy corruption type:")
    for t, n in by_type.most_common():
        print(f"  {t:18s} {n}")
    print("\nBy action:")
    for a, n in by_action.most_common():
        print(f"  {a:25s} {n}")
    print(f"\nFlagged originals for manual review: {len(flagged)} "
          f"→ {os.path.basename(FLAGGED_OUT)}")
    print(f"Clean-case audit sheet: {os.path.basename(AUDIT_OUT)}")


if __name__ == "__main__":
    build_test_set()
