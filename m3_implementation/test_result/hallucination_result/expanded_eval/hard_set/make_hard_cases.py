# m3_implementation/test_result/hallucination_result/hard_set/make_hard_cases.py
#
# HARD corruption set — adversarial cases the checker was NOT tailor-made for.
#
# MOTIVATION: the standard set is dominated by exact-value corruptions, which
# the checker's string-logic gates solve near-perfectly by design. To probe
# the system's limits honestly (and answer "why are the numbers so high?"),
# this set corrupts responses in ways that evade string matching:
#
#   paraphrase_colour   colour replaced by a descriptive phrase of a WRONG
#                       colour using no colour-vocabulary word
#                       ("Black" → "a rich crimson shade")
#   paraphrase_price    £value replaced by a WRONG amount spelled in words,
#                       no £ symbol ("£11.08" → "about fifteen pounds")
#   fabricated_attribute an invented claim appended to the response
#                       ("It is fully machine washable.") — an UNSUPPORTED
#                       claim, not a contradiction. The checker's
#                       contradiction-only rule ignores these BY DESIGN, so
#                       low recall here quantifies that documented trade-off.
#                       Analysed separately (corruption["claim_class"]).
#
# The set contains HALLUCINATED rows only: clean behaviour (false positives)
# is already measured on the standard set with the identical clean cases, so
# hard-set reporting is detection-rate (recall) per corruption family.
#
# Deterministic (seed 77). Bases = the same checker-passed captured cases the
# standard set is built from.
#
# Run:  python test_result/hallucination_result/hard_set/make_hard_cases.py

import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

_DIR     = os.path.dirname(os.path.abspath(__file__))
CAPTURED = os.path.join(_DIR, "..", "captured_cases.jsonl")
HARD_SET = os.path.join(_DIR, "labeled_hard_set.jsonl")

SEED = 77

# Wrong-colour paraphrases. Deliberately avoid every referee/checker colour
# vocabulary word (black/white/red/blue/pink/green/yellow/grey/gray/beige/
# brown/orange/purple and compounds).
_COLOUR_PHRASES = {
    "Red":    "a rich crimson shade",
    "Blue":   "a soft azure shade",
    "Green":  "an earthy olive shade",
    "Yellow": "a warm mustard shade",
    "Pink":   "a delicate blush shade",
    "Grey":   "a smoky charcoal shade",
    "Brown":  "a warm chestnut shade",
    "Purple": "a deep violet shade",
    "White":  "a clean ivory shade",
    "Black":  "a deep midnight shade",
    "Orange": "a burnt amber shade",
}

# Invented claims + the evidence keyword that would make them legitimate
# (claim is only used when the keyword is absent from the evidence text).
_FABRICATED_CLAIMS = [
    ("It is fully machine washable.",              "wash"),
    ("It is made from 100% organic cotton.",       "organic"),
    ("It is completely waterproof.",               "waterproof"),
    ("It comes with a matching belt included.",    "belt"),
    ("It is currently available at a half-price discount.", "discount"),
]

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS  = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
          "eighty", "ninety"]


def _num_words(n: int) -> str:
    n = abs(n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + ("-" + _UNITS[unit] if unit else "")
    hundreds, rest = divmod(n, 100)
    words = _UNITS[hundreds] + " hundred"
    return words + (" and " + _num_words(rest) if rest else "")


def _items_of(evidence: dict) -> list[dict]:
    action = evidence.get("action", "")
    if action == "catalog_search":
        return evidence.get("items", []) or []
    if action == "item_compare":
        return [x for x in (evidence.get("item_a"), evidence.get("item_b")) if x]
    article = evidence.get("article")
    return [article] if article else []


def _evidence_text(evidence: dict) -> str:
    return json.dumps(evidence, ensure_ascii=False).lower()


# ── Hard corruption generators ───────────────────────────────────────────────

def _paraphrase_colour(response, items, rng):
    variants = []
    evidence_colours = {str(i.get("colour", "")).lower() for i in items}
    for idx, item in enumerate(items):
        colour = item.get("colour") or ""
        if not colour:
            continue
        pattern = re.compile(re.escape(colour), re.IGNORECASE)
        if not pattern.search(response):
            continue
        wrong_options = [c for c in _COLOUR_PHRASES
                         if c.lower() != colour.lower()
                         and c.lower() not in evidence_colours]
        if not wrong_options:
            continue
        wrong = rng.choice(wrong_options)
        variants.append({
            "response_text": pattern.sub(_COLOUR_PHRASES[wrong], response),
            "corruption": {"type": "paraphrase_colour", "field": "colour",
                           "item_idx": idx, "original": colour,
                           "corrupted": _COLOUR_PHRASES[wrong],
                           "claim_class": "contradiction"},
        })
    return variants


def _paraphrase_price(response, items, rng):
    variants = []
    for idx, item in enumerate(items):
        m = re.search(r"£([\d,]+)(?:\.\d{1,2})?", str(item.get("price") or ""))
        if not m:
            continue
        exact = re.search(r"£[\d,]+(?:\.\d{1,2})?", str(item.get("price") or "")).group(0)
        if exact not in response:
            continue
        actual = int(m.group(1).replace(",", ""))
        wrong = actual + rng.choice([4, 6, 9, -4, -6])
        if wrong < 2 or abs(wrong - actual) < 3:
            wrong = actual + 7
        phrase = f"about {_num_words(wrong)} pounds"
        variants.append({
            "response_text": response.replace(exact, phrase, 1),
            "corruption": {"type": "paraphrase_price", "field": "price",
                           "item_idx": idx, "original": exact,
                           "corrupted": phrase,
                           "claim_class": "contradiction"},
        })
    return variants


def _fabricated_attribute(response, evidence, rng):
    ev_text = _evidence_text(evidence)
    candidates = [(claim, kw) for claim, kw in _FABRICATED_CLAIMS
                  if kw not in ev_text and kw not in response.lower()]
    if not candidates:
        return []
    claim, kw = rng.choice(candidates)
    return [{
        "response_text": response.rstrip() + " " + claim,
        "corruption": {"type": "fabricated_attribute", "field": "extra_claim",
                       "item_idx": None, "original": "(nothing — claim invented)",
                       "corrupted": claim, "keyword": kw,
                       "claim_class": "unsupported"},
    }]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)
    with open(CAPTURED, encoding="utf-8") as f:
        captured = [json.loads(l) for l in f if l.strip()]
    bases = [c for c in captured if c["checker"].get("passed", False)]
    print(f"Loaded {len(captured)} captured cases → {len(bases)} clean bases")

    rows, counter = [], 0
    for src_idx, case in enumerate(bases):
        response = case["response_text"]
        evidence = case["evidence"]
        items    = _items_of(evidence)
        variants = (_paraphrase_colour(response, items, rng)
                    + _paraphrase_price(response, items, rng)
                    + _fabricated_attribute(response, evidence, rng))
        for v in variants:
            counter += 1
            rows.append({
                "case_id":       f"hard_{counter:04d}",
                "label":         "hallucinated",
                "corruption":    v["corruption"],
                "source_case":   src_idx,
                "action":        case["action"],
                "user_message":  case["user_message"],
                "evidence":      evidence,
                "response_text": v["response_text"],
            })

    with open(HARD_SET, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    by_type = Counter(r["corruption"]["type"] for r in rows)
    print(f"\nHard set written: {os.path.basename(HARD_SET)}")
    print(f"  total hallucinated rows: {len(rows)}")
    for t, n in by_type.most_common():
        print(f"  {t:22s} {n}")
    print("\nNote: hallucinated-only set — clean/false-positive behaviour is "
          "measured on the standard set (identical clean cases).")


if __name__ == "__main__":
    main()
