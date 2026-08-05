# m3_implementation/text_rag/core/assertion_extractor.py
#
# Shared response-analysis layer: ONE extraction pass, TWO consumers.
#
# WHY THIS EXISTS:
#   The turn-local faithfulness check (HallucinationChecker) and the cross-turn
#   consistency check (ContradictionDetector) both need the same two things:
#     (a) the response split into sentences,
#     (b) which sentence describes which recommended item.
#   They used to compute this twice by two different mechanisms — MiniLM
#   greedy assignment in the checker, a second Groq LLM call in the detector.
#   That was duplicated work, duplicated latency, and a duplicated failure mode
#   (the detector silently gave up whenever the Groq free tier rate-limited).
#
#   This module owns both steps. The checker imports the sentence splitter and
#   the item→sentence map; the detector imports those plus `extract_assertions`,
#   which turns the map into typed (article_id, attribute, value) statements.
#
# EXTRACTION IS DETERMINISTIC — no LLM call:
#   price         £-token regex inside the item's own sentence
#   name          verbatim containment, else another evidence/catalog name
#   colour        H&M colour_group_name vocabulary lookup
#   product_type  H&M product_type_name vocabulary lookup
#   Vocabularies are loaded once from the articles CSV, so any value the
#   catalogue can hold is a value we can recognise in the response.
#
# SENTENCE SCOPE:
#   An assertion is always tied to the sentence it was read from. That lets a
#   correction rewrite only that sentence, which is what makes cross-item value
#   swaps repairable — a response-wide string replace cannot fix them because
#   both values are legitimately present somewhere in the text.

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

_embed_model = None

# Minimum MiniLM similarity for a sentence to be accepted as describing an item.
MIN_LOCK_SIM = 0.30


def get_embed_model():
    """Process-wide MiniLM sentence encoder."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ── Text normalisation ────────────────────────────────────────────────────────

def norm_ws(text: str) -> str:
    """
    Normalises a product name for comparison.

    Collapses whitespace runs (catalog names may contain double spaces) and
    tightens spacing around hyphens. Catalog entries carry stray spacing such
    as 'Victoria Pull- On TRS', which any LLM renders as 'Victoria Pull-On
    TRS'. Without this, a name check reads that as a swapped product.
    """
    t = re.sub(r"\s+", " ", text or "").strip()
    return re.sub(r"\s*-\s*", "-", t)


def normalise_value(val) -> str:
    """Lower-cased, whitespace-collapsed form used for value equality tests."""
    if val is None:
        return ""
    val = str(val).lower().strip()
    val = re.sub(r'[-\s]+', ' ', val)
    val = re.sub(r'£\s*', '£', val)
    return val.strip()


def values_differ(a, b) -> bool:
    """
    True when two values of the same attribute genuinely disagree.
    Prices compare numerically so £11.10 and £11.1 are the same value.
    Empty on either side means "no signal" — never a disagreement.
    """
    na, nb = normalise_value(a), normalise_value(b)
    if not na or not nb:
        return False
    if na.startswith('£') and nb.startswith('£'):
        try:
            return abs(float(na[1:].replace(',', '')) -
                       float(nb[1:].replace(',', ''))) > 0.05
        except (ValueError, TypeError):
            return na != nb
    return na != nb


# ── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Splits text into sentences, filtering out very short ones."""
    # Also split on newlines immediately before "Option N:" so each catalog option
    # always starts its own sentence even when the LLM prefixes with an intro line
    # that has no trailing period (e.g. "Here are your options:\nOption 1: ...").
    sentences = re.split(r'(?<=[.!?])\s+|\n(?=Option\s+\d+\s*:)', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


# ── Item → sentence assignment ────────────────────────────────────────────────

def build_item_sentence_map(
    sentences: list[str],
    items:     list[dict],
    verbose:   bool = True,
) -> dict:
    """
    Maps each evidence item to the sentence that best describes it using
    MiniLM semantic similarity on rich item descriptions (name + colour + price + type).

    Order-independent: does not rely on "Option N:" numbering or LLM response order.
    Format-independent: works regardless of how the LLM structures its response.
    Robust to single-field errors: if LLM writes wrong colour, name+price still pull
    the match to the correct sentence.

    Greedy assignment — highest similarity pair is assigned first, each sentence
    can only be claimed by one item.  Items with no sentence above MIN_LOCK_SIM
    are left out of the map (LLM didn't mention them in this response).

    `verbose` controls only the diagnostic printing; the returned map is
    identical either way.
    """
    if not sentences or not items:
        return {}

    import numpy as np
    model = get_embed_model()

    # Rich description per item using fields that reliably appear in LLM responses.
    # Deliberately excludes section/index_group/garment_group — LLM never mentions them.
    item_descs = []
    for item in items:
        parts = [
            item.get("name", ""),
            item.get("colour", ""),
            str(item.get("price", "")),
            item.get("type", ""),
        ]
        item_descs.append(" ".join(p for p in parts if p))

    # Encode items and sentences together in one batch for efficiency
    all_texts  = item_descs + sentences
    embeddings = model.encode(all_texts)
    item_embs  = embeddings[:len(items)]
    sent_embs  = embeddings[len(items):]

    # Build full similarity matrix
    scores = []
    for item_idx, item_emb in enumerate(item_embs):
        for sent_idx, sent_emb in enumerate(sent_embs):
            sim = float(
                np.dot(item_emb, sent_emb) /
                (np.linalg.norm(item_emb) * np.linalg.norm(sent_emb) + 1e-8)
            )
            scores.append((sim, item_idx, sent_idx))

    # Greedy assignment: highest similarity first, no sentence used twice
    scores.sort(reverse=True)
    option_map = {}
    used_sents = set()

    for sim, item_idx, sent_idx in scores:
        if sim < MIN_LOCK_SIM:
            break  # sorted descending — nothing below threshold is useful
        if item_idx not in option_map and sent_idx not in used_sents:
            option_map[item_idx] = sent_idx
            used_sents.add(sent_idx)

    if not verbose:
        return option_map

    # Print matrix AFTER assignment so ASSIGNED vs best-raw conflict is visible.
    # "best-raw" means this was the highest score for that item in isolation,
    # but another item claimed the sentence first in greedy assignment.
    print("\n[MINILM-SIM] ━━━ similarity matrix (items × sentences) ━━━")
    for item_idx, desc in enumerate(item_descs):
        assigned_sent = option_map.get(item_idx)
        item_scores = sorted(
            [(sent_idx, sim) for sim, i, sent_idx in scores if i == item_idx],
            key=lambda x: x[0],
        )
        best_sim = max(sim for _, sim in item_scores) if item_scores else 0
        print(f"  item[{item_idx}] {desc[:60]}")
        for sent_idx, sim in item_scores:
            tags = []
            if sent_idx == assigned_sent:
                tags.append("ASSIGNED")
            elif sim == best_sim:
                tags.append("best-raw")
            tag_str = f"  ◀ {' | '.join(tags)}" if tags else ""
            print(f"    sent[{sent_idx}] sim={sim:.4f}{tag_str}  {sentences[sent_idx][:80]}")

    print("\n[ITEM-MAP] ━━━ item→sentence assignments ━━━")
    for item_idx in sorted(option_map.keys()):
        sent_idx = option_map[item_idx]
        print(f"  item[{item_idx}] desc : {item_descs[item_idx]}")
        print(f"  item[{item_idx}] sent[{sent_idx}]: {sentences[sent_idx][:120]}")

    return option_map


# ── Catalogue vocabularies (loaded once from the articles CSV) ────────────────

_vocab_cache: dict = {}

# Colour words an LLM reaches for that are NOT H&M colour_group_name values.
# The catalogue only knows 49 colour groups — "Navy", "Maroon" and "Teal" are
# not among them — so a response that drifts to one of these would otherwise be
# unreadable, and drift toward a word outside the catalogue is precisely the
# drift worth catching.
#
# Deliberately excludes polysemous words that appear in ordinary garment
# description ("sand", "rose", "peach", "camel", "stone", "blush"): inside a
# product sentence those describe material or motif far more often than colour,
# and a false read here would be reported as a contradiction.
_EXTRA_COLOUR_TERMS = (
    "Navy", "Navy Blue", "Teal", "Maroon", "Burgundy", "Charcoal",
    "Cream", "Ivory", "Khaki", "Olive", "Mustard", "Lilac", "Lavender",
    "Magenta", "Crimson", "Scarlet", "Indigo", "Aqua", "Mint", "Coral",
    "Tan", "Brown", "Dark Brown", "Light Brown", "Beige Brown",
)


def _load_vocabularies() -> dict:
    """
    Reads the articles CSV once and returns the value vocabularies the
    extractor recognises in free text:
        names   — prod_name             (case-sensitive match, common words abound)
        colours — colour_group_name     (case-insensitive, word-boundary)
        types   — product_type_name     (case-insensitive, word-boundary)

    Returns empty sets if the CSV is unavailable; extraction then falls back to
    evidence-local values only, which still covers the common case.
    """
    global _vocab_cache
    if _vocab_cache:
        return _vocab_cache

    names, colours, types = set(), set(), set()
    try:
        import csv
        from text_rag.config import ARTICLES_CSV
        with open(ARTICLES_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                n = norm_ws(row.get('prod_name') or '')
                if len(n) >= 5:
                    names.add(n)
                c = norm_ws(row.get('colour_group_name') or '')
                if len(c) >= 3:
                    colours.add(c)
                t = norm_ws(row.get('product_type_name') or '')
                if len(t) >= 3:
                    types.add(t)
        print(f"[AssertionExtractor] vocab loaded: {len(names)} names, "
              f"{len(colours)} colours, {len(types)} types")
    except Exception as e:
        print(f"[AssertionExtractor] vocabularies unavailable: {e}")

    colours.update(_EXTRA_COLOUR_TERMS)

    _vocab_cache = {
        "names":   frozenset(names),
        # Longest first so "Dark Blue" wins over "Blue" on the same span.
        "colours": sorted(colours, key=len, reverse=True),
        "types":   sorted(types,   key=len, reverse=True),
    }
    return _vocab_cache


def get_catalog_names() -> frozenset:
    """All prod_name values from the articles CSV (original casing)."""
    return _load_vocabularies()["names"]


def _find_vocab_value(text: str, vocabulary: list, prefer: str = "") -> str:
    """
    Finds a vocabulary term inside `text`, matched case-insensitively on word
    boundaries. If `prefer` (the item's known value) is present it always wins,
    so a correct statement is never re-read as some other term that happens to
    also appear. Otherwise the longest matching term is returned.
    """
    if not text:
        return ""
    if prefer and re.search(rf"\b{re.escape(prefer)}\b", text, re.IGNORECASE):
        return prefer
    for term in vocabulary:
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            return term
    return ""


def _find_name(text: str, true_name: str, other_names: list) -> str:
    """
    Returns the product name stated in `text`: the item's own name when present,
    otherwise a different name belonging to another evidence item or to the
    catalogue. Substring relations ('London dress' vs 'SS London dress') are
    ambiguous truncations, never treated as a different product.
    """
    t = norm_ws(text)
    true_norm = norm_ws(true_name).lower()
    if true_norm and true_norm in t.lower():
        return true_name

    def is_diff(candidate: str) -> bool:
        c = norm_ws(candidate).lower()
        return bool(c) and c != true_norm and c not in true_norm and true_norm not in c

    for n in other_names:
        if n and is_diff(n) and norm_ws(n).lower() in t.lower():
            return n
    for n in get_catalog_names():
        if is_diff(n) and n in text:
            return n
    return ""


_PRICE_RE = re.compile(r"£[\d,]+(?:\.\d{1,2})?")


def _find_price(text: str, true_price: str) -> str:
    """
    Returns the price stated in `text`. The item's own price wins when present;
    otherwise the first £-token found (a swapped or invented price).
    """
    tokens = _PRICE_RE.findall(text or "")
    if not tokens:
        return ""
    true_token = ""
    m = _PRICE_RE.search(str(true_price or ""))
    if m:
        true_token = m.group(0)
    if true_token and true_token in tokens:
        return true_token
    return tokens[0]


# ── Public extraction API ─────────────────────────────────────────────────────

# Attributes this module can read out of free text. Kept deliberately narrow:
# section / index_group / garment_group are catalogue-internal taxonomy that no
# LLM response ever states, so "extracting" them only ever produced noise.
EXTRACTABLE_ATTRIBUTES = ("name", "colour", "price", "product_type")


def extract_assertions(
    response_text: str,
    items:         list[dict],
    verbose:       bool = False,
) -> dict:
    """
    Reads the factual statements a response makes about a set of candidate items.

    Args:
        response_text: the assistant text to analyse
        items:         candidate products, each a dict with keys
                       article_id, name, colour, price, product_type
                       (`price` may be either "£12.34" or a bare number)
        verbose:       print the MiniLM assignment matrix

    Returns:
        {
          "sentences":  list[str],
          "item_sentence_map": {item_idx: sentence_idx},
          "assertions": [
              {"article_id", "attribute", "value",
               "sentence_idx", "sentence", "scope"}, ...
          ],
        }

    `scope` is "sentence" when the value was read from the item's own locked
    sentence, or "response" when the whole reply was used because exactly one
    candidate item exists (an unambiguous pronoun reference such as
    "It's Navy." on a follow-up turn where no product name is repeated).
    """
    out = {"sentences": [], "item_sentence_map": {}, "assertions": []}
    if not response_text or not response_text.strip() or not items:
        return out

    sentences = split_sentences(response_text)
    out["sentences"] = sentences

    # The map builder expects the checker's field names; items arrive with the
    # ledger's field names, so translate for the similarity description only.
    map_items = [
        {
            "name":   it.get("name", ""),
            "colour": it.get("colour", ""),
            "price":  it.get("price", ""),
            "type":   it.get("product_type", "") or it.get("type", ""),
        }
        for it in items
    ]
    item_map = build_item_sentence_map(sentences, map_items, verbose=verbose)
    out["item_sentence_map"] = item_map

    vocab       = _load_vocabularies()
    all_names   = [it.get("name", "") for it in items]

    for idx, item in enumerate(items):
        aid = str(item.get("article_id", "")).strip()
        if not aid:
            continue

        if len(items) == 1:
            # Single candidate — the whole reply is about it, so read the whole
            # reply. Locking to one sentence would silently drop every fact
            # stated elsewhere: a detail answer spreads type, colour, pattern
            # and price across separate lines, and a one-sentence scope sees
            # only whichever line the matcher happened to pick. With one item
            # there is no cross-item collision to protect against anyway.
            scope, text, sent_idx = "response", response_text, None
        elif item_map.get(idx) is not None:
            sent_idx = item_map[idx]
            scope, text = "sentence", sentences[sent_idx]
        else:
            # Several candidates and this one won no sentence: the response did
            # not describe it. Guessing here is how cross-item false positives
            # are born, so we say nothing about it.
            continue

        other_names = [n for i, n in enumerate(all_names) if i != idx and n]

        found = {
            "name":         _find_name(text, item.get("name", ""), other_names),
            "price":        _find_price(text, item.get("price", "")),
            "colour":       _find_vocab_value(text, vocab["colours"],
                                              prefer=item.get("colour", "")),
            "product_type": _find_vocab_value(
                text, vocab["types"],
                prefer=item.get("product_type", "") or item.get("type", "")),
        }

        for attribute in EXTRACTABLE_ATTRIBUTES:
            value = found.get(attribute, "")
            if not value:
                continue
            out["assertions"].append({
                "article_id":   aid,
                "attribute":    attribute,
                "value":        value,
                "sentence_idx": sent_idx,
                "sentence":     text if scope == "sentence" else "",
                "scope":        scope,
            })

    return out


def replace_in_scope(
    response_text: str,
    sentences:     list[str],
    sentence_idx,
    wrong_value:   str,
    correct_value: str,
) -> str:
    """
    Replaces `wrong_value` with `correct_value`, confined to the sentence the
    wrong value was read from.

    Scoping is what makes cross-item swaps repairable: when two products have
    exchanged colours, both colours are legitimately present in the response, so
    a global replace corrupts the sentence that was right. Rewriting only the
    offending sentence fixes exactly one side of the swap, and the other side is
    fixed by its own assertion in the same pass.

    Falls back to a whole-response replace when no sentence scope is known
    (single-item pronoun references), preserving the previous behaviour there.
    """
    if not wrong_value or wrong_value == correct_value:
        return response_text

    pattern = re.compile(re.escape(wrong_value), re.IGNORECASE)

    if sentence_idx is None or not sentences or sentence_idx >= len(sentences):
        return pattern.sub(correct_value, response_text)

    target = sentences[sentence_idx]
    if wrong_value not in target and not pattern.search(target):
        # The value moved (an earlier fix rewrote this sentence) — fall back.
        return pattern.sub(correct_value, response_text)

    fixed = pattern.sub(correct_value, target)
    # Replace this one occurrence of the sentence inside the full response.
    if target in response_text:
        return response_text.replace(target, fixed, 1)
    return pattern.sub(correct_value, response_text)
