# m3_implementation/test_result/contradiction_result_v2/audit_clean_cases.py
#
# INDEPENDENT AUDIT OF THE `clean` LABELS.
#
# WHY THIS EXISTS
#   The test set's 1,013 contradictions and 145 hard negatives are certain by
#   construction — a seeded script produced them. The 188 `clean` rows are not:
#   they are real LLM output, captured verbatim, and labelled clean on the
#   presumption that the assistant got them right.
#
#   That presumption is the one part of the label set nothing verifies. If an
#   original response did contain a contradiction, it sits in the negative class
#   and a silent detector scores a free true negative.
#
#   The detector cannot audit them — it would be grading the exam it sits. So
#   this referee is deliberately dumb and shares NO code with it: no NLI, no
#   embeddings, no thresholds. It compares literal values in the response
#   against the structured truth and nothing else.
#
# RULES  (a case FAILS if any rule produces a `problem`)
#   R1  every £ amount stated must be a known price, a difference between two
#       known prices (legitimate derived arithmetic — "£4.04 cheaper"), or a
#       budget figure the user asked for ("dresses under £50")
#   R2  every colour word must be a known colour — product names are blanked
#       first, because names embed colours ("Skinny Midprice No Fade Black")
#
#   R3  for multi-item actions, each item's name should appear. Recorded as a
#       `minor` note, never a failure: the LLM legitimately re-cases and
#       re-spaces names ("MINDY cardigan" -> "Mindy Cardigan").
#
# TRUTH USED
#   product_refs (this turn's evidence) UNION graph_before (what earlier turns
#   established) — a clean response may correctly refer to either.
#
# ISOLATION
#   Reads ../contradiction_result/labeled_test_set.jsonl. Writes only into this
#   folder. v1's reported artifacts are never modified.
#
# Run:  python test_result/contradiction_result_v2/audit_clean_cases.py

import json
import os
import re

_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SET = os.path.normpath(os.path.join(
    _DIR, "..", "contradiction_result", "labeled_test_set.jsonl"))
OUT_JSON = os.path.join(_DIR, "results_clean_audit.json")
OUT_TXT  = os.path.join(_DIR, "clean_audit_report.txt")

_PRICE = re.compile(r"£\s?[\d,]+(?:\.\d{1,2})?")

# A £ amount is a BUDGET, not a claim, when the user's constraint is being
# echoed. Checked against the text immediately before the amount.
_BUDGET = re.compile(
    r"(under|below|less than|cheaper than|up to|within|no more than|max(?:imum)?|"
    r"budget of|around|about|approximately|between)\s*$", re.IGNORECASE)

# Colour words the catalogue uses, plus the ones an LLM reaches for.
# Longest first so "Dark Blue" is consumed before "Blue".
_COLOURS = sorted([
    "Black", "White", "Off White", "Red", "Dark Red", "Blue", "Dark Blue",
    "Light Blue", "Navy", "Pink", "Dark Pink", "Light Pink", "Green",
    "Dark Green", "Yellow", "Grey", "Gray", "Dark Grey", "Light Grey",
    "Beige", "Light Beige", "Dark Beige", "Brown", "Dark Brown", "Light Brown",
    "Orange", "Purple", "Turquoise", "Silver", "Gold", "Khaki", "Burgundy",
    "Maroon", "Teal", "Cream", "Ivory", "Olive", "Mustard", "Lilac",
], key=len, reverse=True)

_MULTI_ITEM = {"catalog_search", "item_compare"}

# Garment nouns. A colour immediately followed by one of these, where the noun
# is NOT the product's own type, is a styling aside or a preference reference —
# "pairs well with black dresses" while selling a boot — not a claim about the
# item. Recorded as minor, never a failure.
_GARMENTS = ("dress", "dresses", "skirt", "skirts", "top", "tops", "shirt",
             "shirts", "blouse", "blouses", "trouser", "trousers", "jean",
             "jeans", "short", "shorts", "coat", "coats", "jacket", "jackets",
             "boot", "boots", "shoe", "shoes", "sneaker", "sneakers", "bag",
             "bags", "tee", "tees", "t-shirt", "cardigan", "cardigans",
             "sweater", "sweaters", "bra", "item", "items", "piece", "pieces")


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _price_val(s):
    m = _PRICE.search(str(s or ""))
    if not m:
        return None
    return round(float(m.group(0).replace("£", "").replace(" ", "").replace(",", "")), 2)


def truth_of(row) -> dict:
    """Known prices / names / colours: this turn's evidence + earlier turns."""
    prices, names, colours = set(), set(), set()

    def take(d):
        if not d:
            return
        p = _price_val(d.get("price"))
        if p is not None:
            prices.add(p)
        if d.get("name"):
            names.add(str(d["name"]))
        if d.get("colour"):
            colours.add(_norm(d["colour"]))

    for ref in row.get("product_refs") or []:
        take(ref)
    for _aid, g in (row.get("graph_before") or {}).items():
        take(g)
    return {"prices": prices, "names": names, "colours": colours}


def strip_names(text: str, names) -> str:
    """Blank product names before the colour scan — names embed colours."""
    out = text or ""
    for n in sorted({str(x) for x in names if x and len(str(x)) >= 3},
                    key=len, reverse=True):
        out = re.sub(re.escape(n), " ", out, flags=re.IGNORECASE)
    return out


def audit(row) -> tuple:
    """Returns (problems, minor)."""
    problems, minor = [], []
    t = truth_of(row)
    text = row.get("response_text", "")

    # ── R1 · prices ──────────────────────────────────────────────────────────
    allowed = set(t["prices"])
    vals = sorted(t["prices"])
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            allowed.add(round(abs(vals[i] - vals[j]), 2))

    for m in _PRICE.finditer(text):
        v = _price_val(m.group(0))
        if v is None or v in allowed:
            continue
        before = text[max(0, m.start() - 32):m.start()]
        if _BUDGET.search(before.rstrip()):
            minor.append(f"R1-minor budget figure {m.group(0)} "
                         f"(\"{before.strip()[-24:]} {m.group(0)}\")")
            continue
        problems.append(f"R1 price {m.group(0)} not in known "
                        f"{sorted(t['prices'])} nor a derived difference")

    # ── R2 · colours (names blanked first) ───────────────────────────────────
    # Two exemptions, both of which produced only false alarms in a first pass:
    #
    #  (a) HYPERNYM. "Brown" when the truth is "Yellowish Brown" is a more
    #      general rendering of the same colour, not a different one — the same
    #      reasoning the test set uses for its hard negatives ("Dress" ->
    #      "maxi dress" is benign). A component word of a known compound colour
    #      is therefore allowed.
    #
    #  (b) STYLING ASIDE. "pairs well with black dresses" while recommending a
    #      boot is advice about other garments, not a claim about this one.
    #      Detected as: the colour is followed by a garment noun that is not
    #      this product's type.
    known_types = {_norm(p.get("product_type"))
                   for p in (row.get("product_refs") or [])} | \
                  {_norm(g.get("product_type"))
                   for g in (row.get("graph_before") or {}).values()}
    known_types.discard("")
    components = set()
    for c in t["colours"]:
        components.update(c.split())

    scan = _norm(strip_names(text, t["names"]))
    for colour in _COLOURS:
        c = colour.lower()
        pat = r"\b" + re.escape(c) + r"\b"
        m = re.search(pat, scan)
        if not m:
            continue
        after = scan[m.end():m.end() + 18].strip()
        follower = after.split()[0].strip(".,:;!?") if after.split() else ""
        scan = re.sub(pat, " ", scan)

        if _norm(colour) in t["colours"]:
            continue
        if c in components:                                     # (a) hypernym
            minor.append(f"R2-minor '{colour}' is a component of a known "
                         f"compound colour {sorted(t['colours'])}")
            continue
        if follower in _GARMENTS and follower.rstrip("s") not in \
                {ty.rstrip("s") for ty in known_types}:         # (b) aside
            minor.append(f"R2-minor '{colour} {follower}' is a styling aside, "
                         f"not a claim about a {sorted(known_types)}")
            continue
        problems.append(f"R2 colour '{colour}' not in known "
                        f"{sorted(t['colours'])}")

    # ── R3 · names present (advisory only) ───────────────────────────────────
    if row.get("action") in _MULTI_ITEM:
        low = _norm(text)
        for n in {str(x) for x in ((r.get("name") for r in
                                    (row.get("product_refs") or []))) if x}:
            if _norm(n) not in low:
                words = _norm(n).split()
                prefix = " ".join(words[:2]) if len(words) >= 2 else words[0]
                if len(prefix) >= 6 and prefix in low:
                    minor.append(f"R3-minor name re-cased/truncated: '{n}'")
                else:
                    minor.append(f"R3-minor name '{n}' not found verbatim")
    return problems, minor


def main() -> None:
    rows = [json.loads(l) for l in open(TEST_SET, encoding="utf-8") if l.strip()]
    clean = [r for r in rows if r.get("label") == "clean"]
    print(f"{TEST_SET}\n  {len(rows)} rows · {len(clean)} labelled clean\n")

    failed, noted, results = [], 0, []
    for r in clean:
        problems, minor = audit(r)
        noted += bool(minor)
        results.append({"case_id": r["case_id"], "action": r.get("action"),
                        "problems": problems, "minor": minor,
                        "verdict": "FAIL" if problems else "pass"})
        if problems:
            failed.append((r, problems))

    print(f"  passed          : {len(clean) - len(failed)}")
    print(f"  FAILED          : {len(failed)}")
    print(f"  with minor notes: {noted}\n")

    lines = ["INDEPENDENT AUDIT OF THE `clean` LABELS",
             "=" * 72,
             "Rule-based. No NLI, no embeddings, no code shared with the detector.",
             f"{len(clean)} clean cases · {len(clean) - len(failed)} pass · "
             f"{len(failed)} fail", ""]

    for r, problems in failed:
        print("-" * 72)
        print(f"{r['case_id']}  [{r.get('action')}]")
        refs = " | ".join(f"{p.get('name')}/{p.get('colour')}/{p.get('price')}"
                          for p in (r.get("product_refs") or []))
        print(f"  evidence : {refs}")
        for p in problems:
            print(f"  PROBLEM  : {p}")
        print(f"  response : {r['response_text'][:150]}")
        lines += ["-" * 72, f"{r['case_id']}  [{r.get('action')}]",
                  f"  evidence : {refs}"]
        lines += [f"  PROBLEM  : {p}" for p in problems]
        lines += [f"  response : {r['response_text'][:300]}", ""]

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "purpose": "verify the presumed `clean` labels independently",
                "test_set": os.path.basename(TEST_SET),
                "n_clean": len(clean),
                "n_pass": len(clean) - len(failed),
                "n_fail": len(failed),
                "method": "literal value checks against product_refs + "
                          "graph_before; no model, no shared code",
            },
            "cases": results,
        }, f, indent=2)

    print("\n" + "=" * 72)
    print(f"  report -> {os.path.basename(OUT_TXT)}")
    print(f"  json   -> {os.path.basename(OUT_JSON)}")
    if failed:
        print(f"\n  {len(failed)} case(s) are mislabelled `clean` and should be "
              f"excluded or relabelled before scoring.")


if __name__ == "__main__":
    main()
