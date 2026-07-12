# m3_implementation/test_result/contradiction_result/corrupt_sessions.py
#
# Step 2 of the contradiction detector evaluation — builds the labeled test set
# (see EVALUATION_PLAN.md, section 2, Experiment A1).
#
# METHOD (synthetic cross-turn corruption; FactCC / HaluEval lineage, extended
# to the multi-turn setting of DECODE and Sato et al. 2024):
#   Each captured turn record holds the LLM response (response_in), the current
#   evidence ground truth (product_refs), and the session-graph state BEFORE the
#   turn (graph_before — products established in PRIOR turns). We corrupt ONE
#   attribute value of ONE product in the response, leaving the evidence/graph
#   ground truth untouched. The corrupted variant is a contradiction BY
#   CONSTRUCTION. Crucially, the product's ground truth may have been established
#   several turns earlier, so each contradiction carries a TURN DISTANCE
#   (how many turns back the truth entered the graph) — the axis of the
#   signature cross-turn figure.
#
# CASE LABELS:
#   "contradiction"  — corrupted variant (ground truth certain)
#   "clean"          — the original response (presumed correct)
#   "hard_negative"  — a benign subtype paraphrase injected on purpose
#                      ("Dress" -> "maxi dress"); NOT a contradiction. Stresses
#                      the NLI confirmation gate (must stay silent).
#
# CORRUPTION TYPES:
#   colour_drift     — product colour -> a colour absent from the session
#   price_drift      — GBP value -> a different value
#   name_drift       — product name -> a different catalog name
#   type_drift       — product_type -> a DIFFERENT real garment type (not subtype)
#   cross_item_swap  — two products' colour/price/name exchanged in one turn
#                      (both values still exist — association error only)
#
# TURN DISTANCE (d):
#   d = current turn_ordinal - ordinal(product.first_seen_turn)
#   d = 0  -> product introduced this same turn (same-turn contradiction)
#   d >= 1 -> cross-turn contradiction (only the session graph can catch it)
#
# Deterministic: seeded RNG -> same input file always yields the same test set.
#
# Run:  python test_result/contradiction_result/corrupt_sessions.py
#       (reads captured_sessions.jsonl by default; override with --input)

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

_DIR        = os.path.dirname(os.path.abspath(__file__))
CAPTURED    = os.path.join(_DIR, "captured_sessions.jsonl")
TEST_SET    = os.path.join(_DIR, "labeled_test_set.jsonl")
AUDIT_OUT   = os.path.join(_DIR, "clean_audit.txt")

SEED = 42

# H&M colour vocabulary for swaps (colour_group_name values)
COLOUR_POOL = [
    "Black", "White", "Red", "Blue", "Dark Blue", "Light Blue", "Pink",
    "Dark Pink", "Green", "Dark Green", "Yellow", "Grey", "Beige",
    "Brown", "Orange", "Purple",
]

# Distinct garment types for type_drift — each pair is genuinely different
# (NOT subtypes of one another), so a swap is a real contradiction.
TYPE_POOL = [
    "Dress", "Blouse", "Trousers", "Skirt", "Jacket", "Sweater",
    "Cardigan", "Shirt", "Hoodie", "Coat", "Shorts", "Bra", "Vest top",
]

# Benign subtype paraphrases for hard negatives — the LLM legitimately uses
# these; the detector's NLI gate must NOT flag them (entailment, not contra).
SUBTYPE_MAP = {
    "dress":     ["maxi dress", "short dress", "midi dress"],
    "bra":       ["sports bra", "soft bra"],
    "trousers":  ["tailored trousers", "slim trousers"],
    "jacket":    ["denim jacket", "light jacket"],
    "sweater":   ["knit sweater", "wool sweater"],
    "blouse":    ["blouse top"],
    "shirt":     ["casual shirt", "button-up shirt"],
    "hoodie":    ["pullover hoodie"],
    "cardigan":  ["knit cardigan"],
    "skirt":     ["a-line skirt", "mini skirt"],
    "shorts":    ["denim shorts"],
    "coat":      ["long coat", "winter coat"],
}


# ── Text helpers ─────────────────────────────────────────────────────────────

def _replace_ci(text: str, old: str, new: str) -> str | None:
    """Replaces ALL case-insensitive occurrences of `old` with `new`.
    Returns None if `old` does not occur in `text`."""
    if not old:
        return None
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    if not pattern.search(text):
        return None
    return pattern.sub(new, text)


def _price_token(price: str) -> str:
    """Extracts the '£XX.XX' token from a price string ('' if none)."""
    m = re.search(r"£[\d,]+\.?\d*", str(price or ""))
    return m.group(0) if m else ""


def _replace_type_only(response: str, refs: list, ptype: str, new_type: str) -> str | None:
    """Replaces the product_type token in the response WITHOUT touching product
    names that contain it (e.g. corrupts 'a Black Dress' but leaves 'London
    dress' intact). Product names are masked first, then restored."""
    masked = response
    masks = {}
    for k, ref in enumerate(refs):
        name = ref.get("name") or ""
        if name and name.lower() in masked.lower():
            ph = f"\x00NAME{k}\x00"
            masks[ph] = name
            masked = re.compile(re.escape(name), re.IGNORECASE).sub(ph, masked)
    corrupted = _replace_ci(masked, ptype, new_type)
    if corrupted is None:
        return None
    for ph, name in masks.items():
        corrupted = corrupted.replace(ph, name)
    return corrupted if corrupted != response else None


# ── Turn-distance computation ────────────────────────────────────────────────

def _build_turnid_ordinal_maps(records: list[dict]) -> dict:
    """Per session: { turn_id -> turn_ordinal }. Lets us translate a product's
    first_seen_turn (a turn_id) into an ordinal so distance can be computed."""
    maps: dict = defaultdict(dict)
    for rec in records:
        sid = rec.get("session_id", "")
        tid = rec.get("turn_id", "")
        ordv = rec.get("turn_ordinal")
        if sid and tid and ordv is not None:
            maps[sid][tid] = ordv
    return maps


def _turn_distance(rec: dict, article_id: str, turnid_ordinal: dict) -> int:
    """Distance between the current turn and the turn that first introduced this
    product. 0 = introduced this turn (or unknown first_seen). >=1 = cross-turn."""
    graph_before = rec.get("graph_before", {}) or {}
    node = graph_before.get(article_id)
    cur_ord = rec.get("turn_ordinal", 0)
    if not node:
        return 0  # not established earlier -> first mention is this turn
    first_seen = node.get("first_seen_turn", "")
    sid = rec.get("session_id", "")
    first_ord = turnid_ordinal.get(sid, {}).get(first_seen)
    if first_ord is None:
        return 0
    return max(0, cur_ord - first_ord)


# ── Corruption generators (single product) ───────────────────────────────────
# Each returns a list of variant dicts:
#   {"response_text", "corruption": {type, field, article_id, original, corrupted}}

def _corrupt_colour(response, refs, session_colours, rng):
    variants = []
    for ref in refs:
        colour = ref.get("colour") or ""
        if not colour:
            continue
        candidates = [
            c for c in COLOUR_POOL
            if c.lower() != colour.lower()
            and c.lower() not in session_colours
            and colour.lower() not in c.lower()
            and c.lower() not in colour.lower()
        ]
        if not candidates:
            continue
        new_colour = rng.choice(candidates)
        corrupted = _replace_ci(response, colour, new_colour)
        if corrupted:
            variants.append({
                "response_text": corrupted,
                "corruption": {"type": "colour_drift", "field": "colour",
                               "article_id": ref["article_id"],
                               "original": colour, "corrupted": new_colour},
            })
    return variants


def _corrupt_price(response, refs, rng):
    variants = []
    for ref in refs:
        tok = _price_token(ref.get("price"))
        if not tok or tok not in response:
            continue
        try:
            amount = float(tok[1:].replace(",", ""))
        except ValueError:
            continue
        new_amount = round(amount + rng.choice([2.5, 3.75, 5.0, 7.25, -2.0, -4.5]), 2)
        if new_amount <= 0 or abs(new_amount - amount) < 0.01:
            new_amount = round(amount + 6.0, 2)
        new_tok = f"£{new_amount:.2f}"
        variants.append({
            "response_text": response.replace(tok, new_tok),
            "corruption": {"type": "price_drift", "field": "price",
                           "article_id": ref["article_id"],
                           "original": tok, "corrupted": new_tok},
        })
    return variants


def _corrupt_name(response, refs, name_pool, session_names, rng):
    variants = []
    for ref in refs:
        name = ref.get("name") or ""
        if not name:
            continue
        candidates = [
            n for n in name_pool
            if n.lower() not in session_names
            and n.lower() not in name.lower()
            and name.lower() not in n.lower()
        ]
        if not candidates:
            continue
        new_name = rng.choice(candidates)
        corrupted = _replace_ci(response, name, new_name)
        if corrupted:
            variants.append({
                "response_text": corrupted,
                "corruption": {"type": "name_drift", "field": "name",
                               "article_id": ref["article_id"],
                               "original": name, "corrupted": new_name},
            })
    return variants


def _corrupt_type(response, refs, rng):
    variants = []
    for ref in refs:
        ptype = ref.get("product_type") or ""
        if not ptype:
            continue
        # Only corrupt when the DB type appears verbatim in the response
        if not re.search(rf"\b{re.escape(ptype)}\b", response, re.IGNORECASE):
            continue
        candidates = [
            t for t in TYPE_POOL
            if t.lower() != ptype.lower()
            and t.lower() not in ptype.lower()
            and ptype.lower() not in t.lower()
        ]
        if not candidates:
            continue
        new_type = rng.choice(candidates)
        corrupted = _replace_type_only(response, refs, ptype, new_type)
        if corrupted:
            variants.append({
                "response_text": corrupted,
                "corruption": {"type": "type_drift", "field": "product_type",
                               "article_id": ref["article_id"],
                               "original": ptype, "corrupted": new_type},
            })
    return variants


def _corrupt_cross_item(response, refs):
    """Swaps a field value between two products both mentioned in the response.
    Both values remain in the evidence — only the association is wrong."""
    variants = []
    if len(refs) < 2:
        return variants

    for field in ("colour", "price", "name"):
        done = False
        for i in range(len(refs)):
            if done:
                break
            for j in range(i + 1, len(refs)):
                v1 = _price_token(refs[i].get("price")) if field == "price" else str(refs[i].get(field) or "")
                v2 = _price_token(refs[j].get("price")) if field == "price" else str(refs[j].get(field) or "")
                if not v1 or not v2 or v1.lower() == v2.lower():
                    continue
                if v1.lower() in v2.lower() or v2.lower() in v1.lower():
                    continue
                p1 = re.compile(re.escape(v1), re.IGNORECASE)
                p2 = re.compile(re.escape(v2), re.IGNORECASE)
                if not (p1.search(response) and p2.search(response)):
                    continue
                ph = "[[SWAP-PH]]"
                swapped = p1.sub(ph, response)
                swapped = p2.sub(v1, swapped)
                swapped = swapped.replace(ph, v2)
                variants.append({
                    "response_text": swapped,
                    "corruption": {"type": "cross_item_swap", "field": field,
                                   "article_id": [refs[i]["article_id"], refs[j]["article_id"]],
                                   "original": f"{v1} / {v2}",
                                   "corrupted": f"{v2} / {v1}"},
                })
                done = True
                break
    return variants


def _hard_negatives(response, refs, rng):
    """Benign subtype paraphrases — labeled clean/hard_negative. The detector
    must NOT flag these (NLI entailment gate)."""
    variants = []
    for ref in refs:
        ptype = (ref.get("product_type") or "").lower()
        subs = SUBTYPE_MAP.get(ptype)
        if not subs:
            continue
        orig = ref.get("product_type") or ""
        if not re.search(rf"\b{re.escape(orig)}\b", response, re.IGNORECASE):
            continue
        new_sub = rng.choice(subs)
        corrupted = _replace_type_only(response, refs, orig, new_sub)
        if corrupted and corrupted != response:
            variants.append({
                "response_text": corrupted,
                "corruption": {"type": "subtype_paraphrase", "field": "product_type",
                               "article_id": ref["article_id"],
                               "original": orig, "corrupted": new_sub},
            })
    return variants


# ── Main ─────────────────────────────────────────────────────────────────────

def build_test_set(input_path: str, out_path: str, audit_path: str):
    rng = random.Random(SEED)

    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} captured turn records from {os.path.basename(input_path)}")

    turnid_ordinal = _build_turnid_ordinal_maps(records)

    # Global name pool for name_drift — every product name seen anywhere
    name_pool = sorted({
        ref.get("name") for rec in records
        for ref in rec.get("product_refs", []) if ref.get("name")
    })

    # Per-session colour/name sets so a swap can't accidentally match another
    # product the same session already established.
    session_colours = defaultdict(set)
    session_names   = defaultdict(set)
    for rec in records:
        sid = rec.get("session_id", "")
        for node in (rec.get("graph_before", {}) or {}).values():
            if node.get("colour"):
                session_colours[sid].add(node["colour"].lower())
            if node.get("name"):
                session_names[sid].add(node["name"].lower())
        for ref in rec.get("product_refs", []):
            if ref.get("colour"):
                session_colours[sid].add(ref["colour"].lower())
            if ref.get("name"):
                session_names[sid].add(ref["name"].lower())

    test_cases   = []
    audit_lines  = []
    case_counter = 0

    def add_case(label, rec, response_text, corruption, distance):
        nonlocal case_counter
        case_counter += 1
        test_cases.append({
            "case_id":       f"ccase_{case_counter:04d}",
            "label":         label,               # clean | contradiction | hard_negative
            "corruption":    corruption,          # None for clean
            "turn_distance": distance,            # None for clean/cross-item mixed
            "session_id":    rec.get("session_id", ""),
            "turn_id":       rec.get("turn_id", ""),
            "turn_ordinal":  rec.get("turn_ordinal"),
            "action":        rec.get("action", ""),
            "product_refs":  rec.get("product_refs", []),
            "graph_before":  rec.get("graph_before", {}),
            "response_text": response_text,        # corrupted (or clean for clean cases)
            "source_response": rec.get("response_in", ""),  # original clean text
        })

    for rec in records:
        response = rec.get("response_in", "")
        refs     = rec.get("product_refs", []) or []
        sid      = rec.get("session_id", "")
        if not response or not refs:
            continue

        # Clean case (presumed correct — subject to manual audit)
        add_case("clean", rec, response, None, None)
        audit_lines.append(
            f"ccase_{case_counter:04d} [{rec.get('action')}] ord={rec.get('turn_ordinal')}\n"
            f"  refs: " + " | ".join(
                f"{r.get('name','?')}/{r.get('colour','?')}/{r.get('price','?')}"
                for r in refs) + "\n"
            f"  response: {response[:200]}\n"
        )

        # Single-product contradictions (each carries its own turn distance)
        single = (
            _corrupt_colour(response, refs, session_colours[sid], rng)
            + _corrupt_price(response, refs, rng)
            + _corrupt_name(response, refs, name_pool, session_names[sid], rng)
            + _corrupt_type(response, refs, rng)
        )
        for v in single:
            aid = v["corruption"]["article_id"]
            dist = _turn_distance(rec, aid, turnid_ordinal)
            v["corruption"]["turn_distance"] = dist
            add_case("contradiction", rec, v["response_text"], v["corruption"], dist)

        # Cross-item swaps (association errors) — distance taken as the min of
        # the two products' distances (the "easier" one to have remembered).
        for v in _corrupt_cross_item(response, refs):
            aids = v["corruption"]["article_id"]
            dists = [_turn_distance(rec, a, turnid_ordinal) for a in aids]
            dist = min(dists) if dists else 0
            v["corruption"]["turn_distance"] = dist
            add_case("contradiction", rec, v["response_text"], v["corruption"], dist)

        # Hard negatives (benign subtype paraphrases)
        for v in _hard_negatives(response, refs, rng):
            add_case("hard_negative", rec, v["response_text"], v["corruption"], None)

    with open(out_path, "w", encoding="utf-8") as f:
        for tc in test_cases:
            f.write(json.dumps(tc, ensure_ascii=False, default=str) + "\n")

    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("MANUAL AUDIT — verify each 'clean' response truly matches its "
                "evidence before computing final metrics.\n\n")
        f.write("\n".join(audit_lines))

    # ── Summary ──────────────────────────────────────────────────────────────
    by_label = Counter(tc["label"] for tc in test_cases)
    by_type  = Counter(tc["corruption"]["type"]
                       for tc in test_cases if tc["corruption"])
    by_dist  = Counter(tc["turn_distance"]
                       for tc in test_cases
                       if tc["label"] == "contradiction")
    by_action = Counter(tc["action"] for tc in test_cases)

    print(f"\nTest set written: {os.path.basename(out_path)}")
    print(f"  total cases:    {len(test_cases)}")
    print(f"  clean:          {by_label.get('clean', 0)}")
    print(f"  contradiction:  {by_label.get('contradiction', 0)}")
    print(f"  hard_negative:  {by_label.get('hard_negative', 0)}")
    print("\nBy corruption type:")
    for t, n in by_type.most_common():
        print(f"  {t:20s} {n}")
    print("\nContradictions by turn distance (d):")
    for d in sorted(by_dist, key=lambda x: (x is None, x)):
        print(f"  d={d}: {by_dist[d]}")
    # collapsed 3+ view
    d3plus = sum(v for k, v in by_dist.items() if isinstance(k, int) and k >= 3)
    print(f"  [collapsed] d=0:{by_dist.get(0,0)}  d=1:{by_dist.get(1,0)}  "
          f"d=2:{by_dist.get(2,0)}  d>=3:{d3plus}")
    print("\nBy action:")
    for a, n in by_action.most_common():
        print(f"  {a:25s} {n}")
    print(f"\nClean-case audit sheet: {os.path.basename(audit_path)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CAPTURED, help="captured_sessions.jsonl path")
    ap.add_argument("--out",   default=TEST_SET, help="output labeled test set path")
    ap.add_argument("--audit", default=AUDIT_OUT, help="clean-audit sheet path")
    args = ap.parse_args()
    build_test_set(args.input, args.out, args.audit)
