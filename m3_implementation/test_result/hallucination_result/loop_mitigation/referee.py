# m3_implementation/test_result/hallucination_result/loop_mitigation/referee.py
#
# Independent grader for the loop-mitigation experiment.
#
# WHY THIS EXISTS: the retry loop only ships responses its own checker approved,
# so the checker cannot grade the loop's final output — it would mark its own
# homework and report 100% success by definition. This referee is deliberately
# simple, model-free, and shares NO code with the checker: it verifies final
# responses against the structured database truth with literal value checks.
#
# RULES (a response is WRONG if any rule fails):
#   R1  every £value stated in the response must be one of the evidence prices
#   R2  every colour-vocabulary word in the response must be one of the
#       evidence colours (longest-match first so "Dark Blue" ≠ "Blue")
#   R3  for multi-item actions (catalog_search, item_compare) every evidence
#       item's name must appear in the response (the generator prompts always
#       list all items; a missing name means the LLM renamed or dropped one)
#
# KNOWN LIMITATION (state in write-up): value-set checks cannot detect two
# correct values swapped BETWEEN items in a regenerated response, or lies
# paraphrased outside the colour vocabulary. Both are rare under the strict
# regeneration prompts (exact-copy instruction); noted as a threat to validity.

import re

# Same colour vocabulary the corruption generator draws from, plus the
# catalog's compound colour names. Longest first for masking.
_COLOUR_VOCAB = sorted([
    "Black", "White", "Red", "Blue", "Dark Blue", "Light Blue", "Pink",
    "Dark Pink", "Green", "Dark Green", "Yellow", "Grey", "Gray", "Beige",
    "Brown", "Orange", "Purple", "Dark Grey", "Light Pink", "Off White",
    "Dark Red", "Light Beige",
], key=len, reverse=True)

_MULTI_ITEM_ACTIONS = {"catalog_search", "item_compare"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _evidence_items(evidence: dict) -> list[dict]:
    action = evidence.get("action", "")
    if action == "catalog_search":
        return evidence.get("items", []) or []
    if action == "item_compare":
        return [x for x in (evidence.get("item_a"), evidence.get("item_b")) if x]
    article = evidence.get("article")
    return [article] if article else []


def _price_value(price_str: str) -> float | None:
    m = re.search(r"£([\d,]+(?:\.\d{1,2})?)", price_str or "")
    return float(m.group(1).replace(",", "")) if m else None


def grade(evidence: dict, response_text: str) -> tuple[bool, list[str], list[str]]:
    """Grades a final shipped response against the evidence truth.

    Returns (correct, problems, minor_issues).
    - problems     → response counted WRONG (hallucination reached the user)
    - minor_issues → fidelity blemishes reported separately, NOT hallucinations
                     (e.g. a truncated but unambiguous item name)
    """
    problems = []
    minor = []
    items = _evidence_items(evidence)
    resp_norm = _norm(response_text)

    # R1 — prices. Derived arithmetic is legitimate: the LLM may state the
    # DIFFERENCE between two evidence prices ("£4.04 cheaper"), so pairwise
    # absolute differences are allowed values too.
    evidence_price_vals = [v for v in (_price_value(str(it.get("price") or ""))
                                       for it in items) if v is not None]
    allowed = {round(v, 2) for v in evidence_price_vals}
    for i in range(len(evidence_price_vals)):
        for j in range(i + 1, len(evidence_price_vals)):
            allowed.add(round(abs(evidence_price_vals[i] - evidence_price_vals[j]), 2))
    for p in set(re.findall(r"£[\d,]+(?:\.\d{1,2})?", response_text)):
        v = _price_value(p)
        if v is not None and round(v, 2) not in allowed:
            problems.append(f"R1 price {p} not in evidence or derived diffs {sorted(allowed)}")

    # R2 — colours (mask longest matches so compounds don't double-count)
    evidence_colours = {_norm(it.get("colour")) for it in items if it.get("colour")}
    masked = resp_norm
    for colour in _COLOUR_VOCAB:
        c = colour.lower()
        pattern = r"\b" + re.escape(c) + r"\b"
        if re.search(pattern, masked):
            if c not in evidence_colours:
                problems.append(f"R2 colour '{colour}' not in evidence {sorted(evidence_colours)}")
            masked = re.sub(pattern, " ", masked)

    # R3 — item names (multi-item actions only). A TRUNCATED name whose
    # leading words still appear ("Regular Printed tee" for
    # "Regular Printed tee 9.99 B2") identifies the item unambiguously —
    # that is a fidelity blemish, not a hallucination. A fully absent or
    # replaced name stays a problem.
    if evidence.get("action") in _MULTI_ITEM_ACTIONS:
        for it in items:
            name = _norm(it.get("name"))
            if not name or name in resp_norm:
                continue
            words = name.split()
            prefix = " ".join(words[:2]) if len(words) >= 2 else words[0]
            if len(prefix) >= 8 and prefix in resp_norm:
                minor.append(f"R3-minor item name truncated: '{it.get('name')}'")
            else:
                problems.append(f"R3 item name '{it.get('name')}' missing from response")

    return (len(problems) == 0, problems, minor)
