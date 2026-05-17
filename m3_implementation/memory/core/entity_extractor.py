# m3_implementation/memory/core/entity_extractor.py
#
# Three-tier entity extraction grounded in actual sample_articles.csv values.
#
# KEY DESIGN DECISIONS:
#
# 1. ONLY extract for INITIAL_REQUEST and REFINEMENT labels.
#    For all other labels (ATTRIBUTE_QUESTION, COMPARISON, FEEDBACK etc.)
#    call extract_entities() returns {} immediately — no extraction needed
#    because those labels use items already in dialogue state, not new searches.
#
# 2. Only extract fields that exist in sample_articles.csv OR are used
#    as soft constraints (price, occasion, style).
#    Hard filter fields (map to CSV columns):
#      product_type_name, colour_group_name, graphical_appearance_name,
#      index_group_name, garment_group_name
#    Soft constraint fields (not in CSV, used in RAG prompt only):
#      price_max, price_min, occasion, style
#
# 3. All extracted values are validated against the exact set of values
#    from sample_articles.csv. Invalid values are silently dropped.
#    This prevents LLM hallucination from creating filter values that
#    do not exist in the database.
#
# 4. Tier 3 (Ollama LLM) always runs but NEVER overrides price_max/price_min
#    found by Tier 1 regex — regex is more reliable for numbers.
#
# TIER STRATEGY:
#   Tier 1: keyword + regex  → instant, handles obvious inputs
#   Tier 2: vector similarity → handles synonyms (midnight→Dark Blue)
#   Tier 3: Ollama LLM        → handles complex NL (pastel shade, graduation)
#   Merge:  Tier 3 > Tier 2 > Tier 1  (except price: Tier 1 wins)

import os
import json
import re
import asyncio
import numpy as np
import httpx

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Groq settings for entity extraction
# Uses llama-3.1-8b-instant (same model, better accuracy for JSON extraction)
LLM_PROVIDER  = os.getenv("LLM_PROVIDER",  "groq")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_ENTITY_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Labels that require entity extraction ─────────────────────────────────────
# Only these two labels result in a catalog search.
# All other labels work with items already in context.
EXTRACTION_LABELS = {"INITIAL_REQUEST", "REFINEMENT"}

# ── Valid values from sample_articles.csv ─────────────────────────────────────
# These are the ONLY values that will be accepted as filter values.
# Any value not in these sets is dropped before returning.

VALID_PRODUCT_TYPES = {
    "Alice band", "Backpack", "Bag", "Ballerinas", "Beanie", "Belt",
    "Bikini top", "Blazer", "Blouse", "Bodysuit", "Bootie", "Boots", "Bra",
    "Cap", "Cap/peaked", "Cardigan", "Coat", "Cross-body bag", "Dress",
    "Dungarees", "Flat shoe", "Flat shoes", "Flip flop", "Gloves",
    "Hair clip", "Hair string", "Hat/beanie", "Hat/brim", "Heeled sandals",
    "Heels", "Hoodie", "Jacket", "Jumpsuit/Playsuit", "Leggings/Tights",
    "Moccasins", "Necklace", "Night gown", "Polo shirt", "Pumps",
    "Pyjama set", "Robe", "Sandals", "Scarf", "Shirt", "Shorts",
    "Shoulder bag", "Skirt", "Slippers", "Sneakers", "Socks", "Sunglasses",
    "Sweater", "Swimsuit", "Swimwear bottom", "Swimwear top", "T-shirt",
    "Tailored Waistcoat", "Tie", "Top", "Tote bag", "Trousers",
    "Underwear body", "Underwear bottom", "Underwear set", "Vest top",
    "Wallet", "Watch", "Wedge", "Weekend/Gym bag",
}

VALID_COLOURS = {
    "Beige", "Black", "Blue", "Bronze/Copper", "Dark Beige", "Dark Blue",
    "Dark Green", "Dark Grey", "Dark Orange", "Dark Pink", "Dark Purple",
    "Dark Red", "Dark Turquoise", "Dark Yellow", "Gold", "Green",
    "Greenish Khaki", "Grey", "Greyish Beige", "Light Beige", "Light Blue",
    "Light Green", "Light Grey", "Light Orange", "Light Pink", "Light Purple",
    "Light Red", "Light Turquoise", "Light Yellow", "Off White", "Orange",
    "Other", "Pink", "Purple", "Red", "Silver", "Turquoise", "White",
    "Yellow", "Yellowish Brown",
}

VALID_GRAPHICAL = {
    "All over pattern", "Argyle", "Chambray", "Check", "Colour blocking",
    "Contrast", "Denim", "Dot", "Embroidery", "Front print", "Jacquard",
    "Lace", "Melange", "Mesh", "Other pattern", "Placement print",
    "Sequin", "Solid", "Stripe", "Transparent",
}

VALID_INDEX_GROUPS = {
    "Baby/Children", "Divided", "Ladieswear", "Menswear", "Sport",
}

VALID_GARMENT_GROUPS = {
    "Accessories", "Blouses", "Dressed", "Dresses Ladies",
    "Jersey Basic", "Jersey Fancy", "Knitwear", "Outdoor", "Shirts",
    "Shoes", "Shorts", "Skirts", "Socks and Tights", "Swimwear",
    "Trousers", "Trousers Denim", "Under-, Nightwear",
}

VALID_OCCASIONS = {
    "casual", "formal", "work", "gym", "beach", "party",
    "wedding", "date night", "summer", "winter", "casual day out",
}

VALID_STYLES = {
    "casual", "formal", "smart casual", "sporty", "elegant",
    "minimalist", "classic", "relaxed", "trendy",
}


def _validate_entities(entities: dict) -> dict:
    """
    Validates all extracted entities against known valid values.
    Drops any field whose value is not in the valid set.
    This is the final gate that prevents invalid filter values
    from reaching the retrieval system.
    """
    validated = {}

    if "product_type_name" in entities:
        if entities["product_type_name"] in VALID_PRODUCT_TYPES:
            validated["product_type_name"] = entities["product_type_name"]

    if "colour_group_name" in entities:
        if entities["colour_group_name"] in VALID_COLOURS:
            validated["colour_group_name"] = entities["colour_group_name"]

    if "graphical_appearance_name" in entities:
        if entities["graphical_appearance_name"] in VALID_GRAPHICAL:
            validated["graphical_appearance_name"] = entities["graphical_appearance_name"]

    if "index_group_name" in entities:
        if entities["index_group_name"] in VALID_INDEX_GROUPS:
            validated["index_group_name"] = entities["index_group_name"]

    if "garment_group_name" in entities:
        if entities["garment_group_name"] in VALID_GARMENT_GROUPS:
            validated["garment_group_name"] = entities["garment_group_name"]

    # Price: accept any positive float
    for price_field in ("price_max", "price_min"):
        if price_field in entities:
            try:
                val = float(entities[price_field])
                if val > 0:
                    validated[price_field] = val
            except (TypeError, ValueError):
                pass

    # Soft constraints: validate against allowed values
    if "occasion" in entities:
        if entities["occasion"] in VALID_OCCASIONS:
            validated["occasion"] = entities["occasion"]

    if "style" in entities:
        if entities["style"] in VALID_STYLES:
            validated["style"] = entities["style"]

    return validated


# ── Advanced Fashion Relevance Classifier ─────────────────────────────────────
#
# 4-stage hybrid approach — no training needed, leverages existing infrastructure:
#
#   Stage 1 (0ms)   — Conversational bypass: continuation phrases in active
#                     sessions always pass (they refer to items in context).
#   Stage 2 (0ms)   — Fast keyword gate: rich allowlist (fashion keywords) +
#                     expanded blocklist (word-boundary regex patterns).
#   Stage 3 (3-5ms) — Dual-pool semantic scoring: mean-of-top-3 cosine scores
#                     from 16 fashion vs 12 off-topic anchors. More robust than
#                     the old single-max approach.
#   Stage 4 (≈150ms)— Groq LLM arbitration: only fires for the genuinely
#                     ambiguous middle zone. Re-uses existing GROQ_API_KEY.
#
# Why not train a model?
#   → No labeled data, high maintenance, overkill for a guard layer.
# Why not NLI cross-encoder only?
#   → 50-100× slower than bi-encoder; unacceptable for every turn.
# Why mean-of-top-3 instead of max?
#   → Single best anchor is brittle; averaging top-3 anchors per pool
#     smooths outliers and gives a more reliable signal.

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[EntityExtractor] Sentence model loaded.")
        except Exception as e:
            print(f"[EntityExtractor] Sentence model unavailable: {e}")
            _model = "unavailable"
    return None if _model == "unavailable" else _model


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def _best_match_index(query: str, candidates: list) -> tuple:
    """Returns (best_index, score). Index maps to the KEY, not the description."""
    model = _get_model()
    if model is None:
        return 0, 0.0
    embeddings = model.encode([query] + candidates)
    query_emb  = embeddings[0]
    best_idx, best_score = 0, -1.0
    for i in range(len(candidates)):
        score = _cosine(query_emb, embeddings[i + 1])
        if score > best_score:
            best_score = score
            best_idx   = i
    return best_idx, best_score


# ── Catalog-derived term sets (built at module load from sample_articles.csv) ──
# Two fast frozensets used by FashionGuard Stage 2c/2d:
#
#   _CATALOG_TYPE_TERMS  — every unique product_type_name token (lowercased).
#       Catches "short", "top", "bra", "tights", "hoodie" etc. that are missing
#       from the hand-written allowlist but exist in the actual catalogue.
#
#   _CATALOG_NAME_TOKENS — significant words from prod_name values.
#       Catches catalogue-specific identifiers like "thuhin", "whisper",
#       "robban", "babette" that signal a product reference even when the
#       semantic model has never seen those words.
#
# Both sets are O(1) frozensets so the per-request check is negligible.

_GENERIC_FASHION_WORDS = frozenset({
    # colours / patterns  – too common to be a reliable product-name signal
    "black", "white", "grey", "gray", "blue", "pink", "brown", "green",
    "beige", "light", "dark", "solid", "melange", "denim", "khaki",
    # garment adjectives
    "basic", "classic", "slim", "loose", "fitted", "short", "long", "mini",
    "maxi", "cropped", "wide", "narrow", "thick", "thin", "soft", "warm",
    "sport", "sports", "casual", "elegant", "fancy", "trendy", "smart",
    # garment anatomy
    "waist", "collar", "sleeve", "pocket", "strap", "lining", "seam",
    "layer", "panel", "front", "back", "inner", "outer", "lace",
    # materials
    "cotton", "polyester", "nylon", "jersey", "knit", "woven",
    # size / quantity
    "extra", "large", "small", "plus", "piece", "pair", "single",
    "women", "ladies", "girls", "boys", "mens", "kids", "baby",
    # common English words likely in names
    "with", "and", "the", "for", "from", "that", "this", "have",
    "will", "your", "over", "under", "more", "less", "very",
})


def _build_catalog_terms() -> tuple:
    """
    Reads sample_articles.csv once at import time and returns two frozensets:
      (product_type_terms, prod_name_tokens)

    product_type_terms  : each word from every unique product_type_name value
    prod_name_tokens    : significant words (>= 5 chars, alpha, not generic)
                          from every unique prod_name value
    """
    import csv as _csv
    _csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "shared", "main_data_set", "sample_articles.csv")
    )
    type_terms:  set = set()
    name_tokens: set = set()
    try:
        with open(_csv_path, encoding="utf-8") as _f:
            for _row in _csv.DictReader(_f):
                # product_type_name — add full value and each word
                _pt = _row.get("product_type_name", "").strip().lower()
                if _pt:
                    type_terms.add(_pt)
                    for _w in _pt.split("/"):        # "cap/peaked" → "cap", "peaked"
                        for _t in _w.split():
                            if len(_t) >= 3:
                                type_terms.add(_t.strip("-"))

                # prod_name — extract significant tokens only
                _pn = _row.get("prod_name", "").strip().lower()
                for _w in _pn.split():
                    _w = re.sub(r"[^a-z]", "", _w)  # strip punctuation
                    if (len(_w) >= 5
                            and _w.isalpha()
                            and _w not in _GENERIC_FASHION_WORDS):
                        name_tokens.add(_w)
    except Exception as _e:
        print(f"[FashionGuard] Warning: could not load catalog terms: {_e}")
    print(f"[FashionGuard] Catalog terms loaded: "
          f"{len(type_terms)} type terms, {len(name_tokens)} name tokens")
    return frozenset(type_terms), frozenset(name_tokens)


_CATALOG_TYPE_TERMS, _CATALOG_NAME_TOKENS = _build_catalog_terms()

# ── Stage 1: Continuation bypass phrases ──────────────────────────────────────
# When these appear in a message that has conversation history, the message
# is always a continuation (references prior items) — never off-topic.
_CONTINUATION_PHRASES = [
    "thanks", "thank you", "cheers", "great", "ok", "okay", "perfect",
    "awesome", "brilliant", "wonderful", "that helps", "very helpful",
    "tell me more", "more about", "more details", "more info",
    "why this", "why the", "explain", "what material", "what fabric",
    "how much", "the price", "is it", "does it", "tell me about",
    "first one", "second one", "option 1", "option 2",
    "which one", "compare", "the first", "the second",
    "yes please", "yes", "no", "nope", "love it", "hate it",
    "show me", "another one", "different", "instead",
]

# ── Stage 2a: Fast allowlist — always fashion-relevant ────────────────────────
# Plain substring match (already lowercased). Contains H&M product types,
# fashion intent verbs, occasions and style words.
_ALLOWLIST_PHRASES = [
    # Product types (singular + plural)
    "dress", "blouse", "trousers", "pants", "jeans", "skirt",
    "jacket", "coat", "hoodie", "sweater", "jumper", "cardigan",
    "vest top", "shorts", "leggings", "tights", "blazer",
    "sneakers", "boots", "sandals", "heels", "loafers",
    "handbag", "backpack", "scarf", "beanie",
    "swimsuit", "bikini", "activewear", "sportswear",
    # Must keep "shirt" after "t-shirt" / "blouse" to avoid partial match issues
    "t-shirt", "tshirt", "polo shirt", "shirt",
    # Generic fashion signals
    "outfit", "clothing", "fashion", "wardrobe", "apparel", "garment",
    # Style / aesthetic words
    "trendy", "elegant", "smart casual", "minimalist",
    # Material / attribute queries about a clothing item
    "what fabric", "what material", "machine wash", "cotton", "linen",
    # Action phrases unambiguously about fashion
    "what to wear", "something to wear", "dressed for",
    "recommend me", "show me clothes", "find me a",
    # H&M direct reference
    "h&m",
]

# ── Stage 2b: Expanded blocklist — clear off-topic signals ────────────────────
# Word-boundary regex to avoid false positives (e.g. "warm" ≠ "war").
_BLOCKLIST_PATTERNS = [
    # Weather
    r'\bweather\b', r'\bforecast\b', r'\btemperature\b', r'\bhumidity\b',
    # Humour
    r'\bjoke\b', r'\bfunny\b', r'\blaugh\b', r'\briddle\b',
    # Sports
    r'\bfootball\b', r'\bsoccer\b', r'\bcricket\b', r'\bbasketball\b',
    r'\btennis\b', r'\bmatch result\b', r'\bsports score\b',
    # Politics / geography
    r'\bpresident\b', r'\belection\b', r'\bpolitics\b', r'\bgovernment\b',
    r'\bcapital city\b',
    # Food and drink — FIX: was missing entirely from blocklist patterns
    # Stage 0 only caught single-word exact matches ("food" alone).
    # These patterns now catch "I need food", "order me pizza", "I want coffee" etc.
    r'\bfood\b', r'\beat\b', r'\beating\b', r'\bhungry\b',
    r'\bmeal\b', r'\blunch\b', r'\bdinner\b', r'\bbreakfast\b', r'\bsnack\b',
    r'\bdrink\b', r'\bbeverage\b', r'\bcafe\b', r'\bcanteen\b',
    r'\bpizza\b', r'\bburger\b', r'\bcoffee\b', r'\btea\b',
    r'\bsushi\b', r'\bpasta\b', r'\bsoup\b', r'\bcake\b', r'\bwine\b', r'\bbeer\b',
    r'\border food\b', r'\bfood delivery\b', r'\btakeaway\b', r'\btakeout\b',
    r'\brestaurant\b', r'\bmenu\b', r'\brecipe\b',
    # Cooking
    r'\bcooking\b', r'\bbaking\b', r'\bingredients\b',
    # Programming / homework
    r'\bhomework\b', r'\bprogramming\b', r'\balgorithm\b', r'\bdebugging\b',
    r'\bwrite code\b', r'\bfix bug\b', r'\bpython error\b',
    # Finance / crypto
    r'\bbitcoin\b', r'\bcrypto\b', r'\bstock price\b', r'\binvest\b',
    # Medical
    r'\bdoctor\b', r'\bhospital\b', r'\bsymptom\b', r'\bmedication\b',
    # Language
    r'\btranslate\b', r'\bgrammar\b', r'\blanguage lesson\b',
    # Travel bookings
    r'\bbook a flight\b', r'\bbook a hotel\b', r'\bbook me a flight\b',
    r'\bflight to\b', r'\bhotel booking\b', r'\btravel insurance\b',
    # Time / news
    r'\bcurrent time\b', r'\bwhat time is\b', r'\bwhat date is\b',
    r'\bnews today\b', r'\bcurrent events\b', r'\bbreaking news\b',
]

# ── Stage 0: Exact word blocklist ───────────────────────────────────────────
# Exact whole-message match (after stripping). Catches single or double words
# that semantic scoring misclassifies because of incidental word overlap
# (e.g. 'washroom' scores high fashion due to 'machine washable' in anchors).
# Keep this set focused on genuinely ambiguous short off-topic words.
_EXACT_WORD_BLOCKLIST = {
    # Bathroom / household
    "washroom", "wash room", "bathroom", "toilet", "restroom", "shower",
    "kitchen", "bedroom", "living room", "dining room", "garage",
    # Food
    "pizza", "burger", "food", "lunch", "dinner", "breakfast", "coffee",
    "beer", "wine", "restaurant", "cake", "soup", "pasta", "sushi",
    # Transport
    "car", "bus", "taxi", "train", "flight", "bike", "motorcycle",
    "uber", "lyft", "vehicle", "driving",
    # Tech / general
    "computer", "laptop", "phone", "wifi", "internet", "website",
    "google", "facebook", "twitter", "youtube",
    # Sports / games
    "football", "cricket", "tennis", "basketball", "chess", "gaming",
    # Animals
    "dog", "cat", "bird", "fish", "pet",
    # Finance
    "money", "bitcoin", "crypto", "bank", "loan", "salary",
    # Medical
    "medicine", "doctor", "hospital", "pain", "sick", "headache",
}

# ── Stage 3: Dual-pool anchor sentences ───────────────────────────────────────
# 16 fashion anchors — cover product discovery, attributes, refinement,
# comparison, feedback, price/budget, style/occasion (H&M specific).
_FASHION_ANCHOR_POOL = [
    "I want to buy a dress shirt jacket trousers shoes or bag",
    "Show me casual or formal clothing options from the collection",
    "Find me something to wear for a party wedding office or beach",
    "I need an outfit for a special occasion or everyday use",
    "What material fabric colour or size is this clothing item",
    "Does it have pockets how does it fit is it machine washable",
    "Tell me more about this fashion item its design and style details",
    "Show me something similar but in a different colour or style",
    "I prefer something more casual formal sporty or elegant",
    "Which of these two clothing items is better for my needs",
    "Can you compare these two fashion recommendations for me",
    "I love this item it is perfect I want to buy it",
    "I don't like this show me different fashion options",
    "I have a budget of fifty pounds show me affordable clothing",
    "Recommend fashion items under thirty or forty pounds",
    "I prefer minimalist classic trendy or smart casual clothing styles",
]

# 12 off-topic anchor sentences covering common chatbot misuse categories.
_OFF_TOPIC_ANCHOR_POOL = [
    "What is the weather forecast temperature today or tomorrow",
    "Tell me a joke funny story or riddle to make me laugh",
    "Who won the football cricket basketball tennis match score",
    "What is the capital of a country or who is the president",
    "Help me write code fix a bug or explain a programming concept",
    "What happened in the news politics or current world events",
    "How do I cook bake or prepare a recipe or meal",
    "What is the current date time or timezone",
    "Give me financial advice about stocks crypto or bitcoin investment",
    "I have a medical symptom what doctor hospital should I see",
    "Translate this sentence to another language or fix my grammar",
    "Book a hotel flight or recommend travel destinations for a trip",
]

# Threshold constants
_SEMANTIC_FASHION_MIN = 0.18   # below this → off-topic regardless of margin
_SEMANTIC_MARGIN      = 0.10   # off-topic must beat fashion by this to reject
_AMBIGUITY_HIGH       = 0.28   # above this → confident fashion, skip Groq
# FIX: minimum margin fashion must have OVER off-topic to be accepted at Stage 3.
# Old behaviour: any fashion_score > off_topic_score was accepted (margin=0.001 was enough).
# "I need food" had margin=0.033 — too thin, basically noise — but still passed.
# New behaviour: fashion must lead by at least 0.08 to be confidently accepted.
# Scores within 0.08 of each other are genuinely ambiguous → escalate to Stage 4 Groq.
_SEMANTIC_FASHION_WIN_MARGIN = 0.08


def _semantic_scores(message: str) -> tuple:
    """
    Dual-pool mean-of-top-3 cosine scoring.
    Returns (fashion_score, offtopic_score).
    """
    model = _get_model()
    if model is None:
        return 0.5, 0.0  # no model → default allow

    all_anchors = _FASHION_ANCHOR_POOL + _OFF_TOPIC_ANCHOR_POOL
    embeddings  = model.encode([message] + all_anchors)
    msg_emb     = embeddings[0]
    n_f         = len(_FASHION_ANCHOR_POOL)

    f_scores = sorted(
        [_cosine(msg_emb, embeddings[i + 1]) for i in range(n_f)],
        reverse=True
    )
    o_scores = sorted(
        [_cosine(msg_emb, embeddings[i + 1 + n_f])
         for i in range(len(_OFF_TOPIC_ANCHOR_POOL))],
        reverse=True
    )
    return sum(f_scores[:3]) / 3, sum(o_scores[:3]) / 3


async def _groq_relevance_check(message: str) -> tuple:
    """
    Stage 4: Groq LLM arbitration for genuinely ambiguous messages.
    Returns (is_relevant: bool, confidence: float).
    Only called when semantic scores are in the ambiguous middle zone.
    """
    if not GROQ_API_KEY:
        return True, 0.5

    prompt = (
        "You are a domain guard for an H&M fashion shopping assistant. "
        "Reply with ONLY JSON: {\"is_fashion\": true/false, \"reason\": \"one sentence\"}.\n\n"
        "Is this message about fashion, clothing, style, accessories, or H&M shopping?\n"
        f"Message: \"{message}\""
    )
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": GROQ_ENTITY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 60,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
            )
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            is_fashion = bool(parsed.get("is_fashion", True))
            conf = 0.92 if is_fashion else 0.08
            print(f"[FashionGuard] Groq verdict: is_fashion={is_fashion} reason={parsed.get('reason','')}")
            return is_fashion, conf
    except Exception as e:
        print(f"[FashionGuard] Groq fallback error: {e}")
    return True, 0.5  # on any error → default allow (prefer false-negatives over false-positives)


def _fg_stage0(msg: str, msg_words: set, history: list) -> tuple | None:
    """Stage 0: exact word blocklist. Returns result tuple or None to continue."""
    blocked = msg_words & _EXACT_WORD_BLOCKLIST
    if not blocked:
        return None
    if not history:
        print(f"[FashionGuard] Stage0-word-block: '{msg}' matched {blocked}")
        return False, 0.0, "stage0_exact_block"
    if not any(p in msg for p in _CONTINUATION_PHRASES):
        print(f"[FashionGuard] Stage0-word-block (history): '{msg}' matched {blocked}")
        return False, 0.0, "stage0_exact_block"
    return None


def _fg_stage1(msg: str, history: list) -> tuple | None:
    """Stage 1: continuation phrase bypass (requires history)."""
    if history:
        for phrase in _CONTINUATION_PHRASES:
            if phrase in msg:
                return True, 1.0, "stage1_continuation"
    return None


def _fg_stage2_allow(msg: str, msg_words: set, history: list) -> tuple | None:
    """Stages 2a/2c/2d: allowlist, catalog type terms, catalog name tokens."""
    for phrase in _ALLOWLIST_PHRASES:
        if phrase in msg:
            return True, 0.95, "stage2_allowlist"
    type_hit = msg_words & _CATALOG_TYPE_TERMS
    if type_hit:
        print(f"[FashionGuard] Stage2c-catalog-type: matched {type_hit} in '{msg[:60]}'")
        return True, 0.95, "stage2_catalog_type"
    if history:
        name_hit = msg_words & _CATALOG_NAME_TOKENS
        if name_hit:
            print(f"[FashionGuard] Stage2d-catalog-name: matched {name_hit} in '{msg[:60]}'")
            return True, 0.93, "stage2_catalog_name"
    return None


def _fg_stage2b_block(msg: str) -> tuple | None:
    """Stage 2b: regex blocklist."""
    for pattern in _BLOCKLIST_PATTERNS:
        if re.search(pattern, msg):
            print(f"[FashionGuard] Stage2-blocklist: pattern={pattern!r} msg='{msg[:60]}'")
            return False, 0.05, "stage2_blocklist"
    return None


def _fg_stage3(message: str, msg: str) -> tuple[tuple | None, float, float]:
    """Stage 3: dual-pool semantic scoring. Returns (result_or_None, f_score, o_score)."""
    f_score, o_score = _semantic_scores(message)
    margin = f_score - o_score
    print(f"[FashionGuard] Stage3 fashion={f_score:.3f} offtopic={o_score:.3f} "
          f"margin={margin:.3f} msg='{msg[:40]}'")
    if o_score > f_score + _SEMANTIC_MARGIN:
        return (False, o_score, "stage3_semantic"), f_score, o_score
    if f_score < _SEMANTIC_FASHION_MIN:
        return (False, f_score, "stage3_semantic"), f_score, o_score
    if f_score >= _AMBIGUITY_HIGH:
        return (True, f_score, "stage3_semantic"), f_score, o_score
    if margin < _SEMANTIC_FASHION_WIN_MARGIN:
        print(f"[FashionGuard] Stage3-thin-margin: f={f_score:.3f} o={o_score:.3f} "
              f"margin={margin:.3f} < {_SEMANTIC_FASHION_WIN_MARGIN} → escalating to Groq")
    return None, f_score, o_score


async def is_fashion_relevant_async(
    message: str,
    history: list = None
) -> tuple:
    """
    Full 4-stage fashion relevance classifier (async).
    Returns (is_relevant: bool, confidence: float, stage: str).
    Each stage is extracted into its own _fg_stageX helper for clarity.
    """
    msg = " ".join(message.lower().split())
    msg_words = set(msg.split())

    for check in (
        _fg_stage0(msg, msg_words, history),
        _fg_stage1(msg, history),
        _fg_stage2_allow(msg, msg_words, history),
        _fg_stage2b_block(msg),
    ):
        if check is not None:
            return check

    decided, f_score, o_score = _fg_stage3(message, msg)
    if decided is not None:
        return decided

    print(f"[FashionGuard] Stage4-Groq: f={f_score:.3f} o={o_score:.3f} msg='{message[:60]}'")
    is_rel, conf = await _groq_relevance_check(message)
    return is_rel, conf, "stage4_groq"


def is_fashion_relevant(message: str) -> tuple:
    """
    Synchronous 3-stage fashion relevance check (no Groq, no history).
    Returns (is_relevant: bool, confidence: float).

    No blanket short-message bypass — single unknown words are still
    evaluated. Only allowlist hits get fast-allowed.
    """
    msg = message.lower().strip()

    # Stage 0: word-level blocklist (sync version)
    _msg_words = set(msg.split())
    _blocked_words = _msg_words & _EXACT_WORD_BLOCKLIST
    if _blocked_words:
        print(f"[FashionGuard] Stage0-word-block: '{msg}' matched {_blocked_words}")
        return False, 0.0

    # Stage 2a: allowlist (works for any length including single words)
    for phrase in _ALLOWLIST_PHRASES:
        if phrase in msg:
            return True, 0.95

    # Stage 2b: blocklist
    for pattern in _BLOCKLIST_PATTERNS:
        if re.search(pattern, msg):
            return False, 0.05

    # Stage 3: semantic scoring — runs for ALL remaining messages
    f_score, o_score = _semantic_scores(message)
    print(f"[FashionGuard] sync: f={f_score:.3f} o={o_score:.3f} msg='{msg[:40]}'")

    if o_score > f_score + _SEMANTIC_MARGIN:
        return False, o_score
    if f_score < _SEMANTIC_FASHION_MIN:
        return False, f_score

    return True, f_score



# ── Tier 1: Keyword + regex ───────────────────────────────────────────────────
# Maps user language to exact values from sample_articles.csv

_COLOUR_MAP = {
    # Multi-word first
    "dark blue": "Dark Blue", "light blue": "Light Blue",
    "light pink": "Light Pink", "dark pink": "Dark Pink",
    "dark green": "Dark Green", "dark grey": "Dark Grey",
    "dark gray": "Dark Grey", "light grey": "Light Grey",
    "light green": "Light Green", "light purple": "Light Purple",
    "off white": "Off White", "off-white": "Off White",
    "greyish beige": "Greyish Beige", "yellowish brown": "Yellowish Brown",
    # Single word
    "black": "Black", "white": "White", "red": "Red", "blue": "Blue",
    "pink": "Pink", "green": "Green", "yellow": "Yellow", "beige": "Beige",
    "grey": "Grey", "gray": "Grey", "brown": "Yellowish Brown",
    "orange": "Orange", "purple": "Purple", "turquoise": "Turquoise",
    "navy": "Dark Blue", "cream": "Off White", "ivory": "Off White",
    "teal": "Dark Green", "coral": "Dark Pink", "khaki": "Greenish Khaki",
    "camel": "Beige", "burgundy": "Dark Red", "lilac": "Light Purple",
    "mint": "Light Green", "nude": "Beige", "tan": "Beige",
    "charcoal": "Dark Grey", "mustard": "Dark Yellow", "rust": "Dark Orange",
    "wine": "Dark Red", "gold": "Gold", "silver": "Silver",
    "multicolour": "Other", "multicolor": "Other", "colourful": "Other",
}

_PRODUCT_MAP = {
    # Multi-word first (more specific patterns before generic ones)
    "vest top": "Vest top", "t-shirt": "T-shirt", "tshirt": "T-shirt",
    "tank top": "Vest top", "midi skirt": "Skirt", "maxi skirt": "Skirt",
    "mini skirt": "Skirt", "midi dress": "Dress", "maxi dress": "Dress",
    "mini dress": "Dress", "shirt dress": "Dress", "wrap dress": "Dress",
    "body suit": "Bodysuit", "swimsuit": "Swimsuit", "swim suit": "Swimsuit",
    "gym wear": "Leggings/Tights", "activewear": "Leggings/Tights",
    "sports bra": "Bra", "rain coat": "Jacket", "raincoat": "Jacket",
    "trench coat": "Jacket", "denim jacket": "Jacket",
    "leather jacket": "Jacket", "puffer jacket": "Jacket",
    "polo shirt": "Polo shirt", "tote bag": "Tote bag",
    "cross-body bag": "Cross-body bag", "shoulder bag": "Shoulder bag",
    "weekend bag": "Weekend/Gym bag", "gym bag": "Weekend/Gym bag",
    "heeled sandals": "Heeled sandals", "flat shoe": "Flat shoes",
    "jumpsuit": "Jumpsuit/Playsuit", "playsuit": "Jumpsuit/Playsuit",
    "dungarees": "Dungarees", "pyjama": "Pyjama set",
    # Single word
    "dress": "Dress", "dresses": "Dress", "skirt": "Skirt",
    "trousers": "Trousers", "pants": "Trousers", "jeans": "Trousers",
    "top": "Top", "blouse": "Blouse", "shirt": "Shirt",
    "sweater": "Sweater", "jumper": "Sweater", "knit": "Sweater",
    "hoodie": "Hoodie", "sweatshirt": "Hoodie",
    "jacket": "Jacket", "coat": "Jacket", "blazer": "Blazer",
    "cardigan": "Cardigan", "shorts": "Shorts",
    "leggings": "Leggings/Tights", "tights": "Leggings/Tights",
    "sneakers": "Sneakers", "trainers": "Sneakers", "shoes": "Sneakers",
    "boots": "Boots", "sandals": "Sandals", "heels": "Heels",
    "loafers": "Flat shoes", "bag": "Bag", "handbag": "Bag",
    "backpack": "Backpack", "scarf": "Scarf",
    "hat": "Hat/beanie", "beanie": "Beanie", "cap": "Cap",
    "socks": "Socks", "bra": "Bra", "bikini": "Bikini top",
    "swimwear": "Swimsuit", "robe": "Robe",
    "belt": "Belt", "watch": "Watch", "wallet": "Wallet",
    "sunglasses": "Sunglasses", "gloves": "Gloves",
}

_GRAPHICAL_MAP = {
    # Maps user words to exact graphical_appearance_name values
    "floral": "Front print", "flower": "Front print", "flowers": "Front print",
    "stripe": "Stripe", "stripes": "Stripe", "striped": "Stripe",
    "check": "Check", "checked": "Check", "plaid": "Check",
    "tartan": "Check", "gingham": "Check",
    "dot": "Dot", "dots": "Dot", "polka dot": "Dot", "spotted": "Dot",
    "embroidery": "Embroidery", "embroidered": "Embroidery",
    "lace": "Lace",
    "sequin": "Sequin", "sequins": "Sequin", "sparkle": "Sequin",
    "denim": "Denim",
    "mesh": "Mesh",
    "animal print": "Front print", "leopard": "Front print",
    "solid": "Solid", "plain": "Solid",
    "pattern": "All over pattern", "patterned": "All over pattern",
    "geometric": "All over pattern",
}

_INDEX_GROUP_MAP = {
    "women": "Ladieswear", "womens": "Ladieswear", "ladies": "Ladieswear",
    "womenswear": "Ladieswear", "ladieswear": "Ladieswear",
    "men": "Menswear", "mens": "Menswear", "menswear": "Menswear",
    "kids": "Baby/Children", "children": "Baby/Children", "baby": "Baby/Children",
    "sport": "Sport", "sports": "Sport",
    "divided": "Divided", "teen": "Divided", "young": "Divided",
}

_OCCASION_MAP = {
    "job interview": "work", "casual day": "casual day out",
    "date night": "date night", "girls night": "party",
    "night out": "party", "garden party": "party",
    "black tie": "formal", "wedding": "wedding",
    "graduation": "formal", "prom": "formal", "gala": "formal",
    "the office": "work", "office": "work", "work": "work",
    "business": "work", "meeting": "work", "interview": "work",
    "gym": "gym", "workout": "gym", "exercise": "gym",
    "yoga": "gym", "running": "gym",
    "beach": "beach", "holiday": "beach", "vacation": "beach",
    "pool": "beach", "resort": "beach",
    "party": "party", "birthday": "party", "celebration": "party",
    "festival": "casual", "concert": "casual",
    "date": "date night", "dinner": "date night",
    "brunch": "casual", "lunch": "casual",
    "summer": "summer", "winter": "winter",
    "christmas": "formal", "new year": "formal",
}

_STYLE_MAP = {
    "smart casual": "smart casual", "business casual": "smart casual",
    "casual": "casual", "everyday": "casual", "relaxed": "relaxed",
    "laid back": "relaxed", "formal": "formal", "smart": "smart casual",
    "professional": "formal", "corporate": "formal",
    "sporty": "sporty", "athletic": "sporty", "active": "sporty",
    "elegant": "elegant", "sophisticated": "elegant", "chic": "elegant",
    "minimalist": "minimalist", "minimal": "minimalist",
    "classic": "classic", "timeless": "classic",
    "trendy": "trendy", "fashionable": "trendy", "stylish": "trendy",
    "bohemian": "relaxed", "boho": "relaxed",
    "streetwear": "casual", "street style": "casual",
}

_NUMBER_WORDS = {
    "five": 5, "ten": 10, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "twenty five": 25, "thirty": 30,
    "thirty-five": 35, "forty": 40, "forty-five": 45,
    "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}

_PRICE_CONTEXT_WORDS = frozenset([
    "pound", "dollar", "euro", "budget",
    "under", "below", "less", "cheap", "afford",
])
_PRICE_MIN_WORDS = frozenset(["over", "above", "more than", "at least"])
# Word-set (not substring) so "cheap" does NOT match inside "cheaper"
_BUDGET_WORDS  = frozenset(["budget", "cheap", "affordable"])
_LUXURY_WORDS  = frozenset(["luxury", "premium", "designer", "high-end"])

_PRICE_MAX_PATTERNS = [
    r'under\s+[£$€]?\s*(\d+(?:\.\d+)?)',
    r'below\s+[£$€]?\s*(\d+(?:\.\d+)?)',
    r'less\s+than\s+[£$€]?\s*(\d+(?:\.\d+)?)',
    r'[£$€]\s*(\d+(?:\.\d+)?)',
    r'(\d+(?:\.\d+)?)\s*(?:pounds?|dollars?|euros?|gbp)',
]


def _price_from_number_word(msg: str, value: float) -> dict:
    key = "price_min" if any(kw in msg for kw in _PRICE_MIN_WORDS) else "price_max"
    return {key: float(value)}


def _extract_price(msg: str) -> dict:
    for pattern in _PRICE_MAX_PATTERNS:
        m = re.search(pattern, msg)
        if m:
            return {"price_max": float(m.group(1))}

    for word, value in _NUMBER_WORDS.items():
        if re.search(rf'\b{re.escape(word)}\b', msg) and \
                any(kw in msg for kw in _PRICE_CONTEXT_WORDS):
            return _price_from_number_word(msg, value)

    msg_words = set(msg.split())
    if any(kw in msg_words for kw in _BUDGET_WORDS):
        return {"price_max": 35.0}
    if any(kw in msg_words for kw in _LUXURY_WORDS):
        return {"price_min": 80.0}
    return {}


def extract_entities_keyword(message: str) -> dict:
    """Tier 1: keyword + regex. Always runs."""
    entities = {}
    msg = message.lower()
    for kw, val in _COLOUR_MAP.items():
        if kw in msg:
            entities["colour_group_name"] = val
            break
    for kw, val in _PRODUCT_MAP.items():
        if kw in msg:
            entities["product_type_name"] = val
            break
    entities.update(_extract_price(msg))
    for kw, val in _OCCASION_MAP.items():
        if kw in msg:
            entities["occasion"] = val
            break
    for kw, val in _STYLE_MAP.items():
        if kw in msg:
            entities["style"] = val
            break
    for kw, val in _GRAPHICAL_MAP.items():
        if kw in msg:
            entities["graphical_appearance_name"] = val
            break
    for kw, val in _INDEX_GROUP_MAP.items():
        if kw in msg:
            entities["index_group_name"] = val
            break
    return entities


# ── Tier 2: Vector similarity ─────────────────────────────────────────────────

_COLOUR_VECTOR_ANCHORS = {
    "Black":         "Black dark midnight jet ebony",
    "White":         "White bright pure snow clean",
    "Dark Blue":     "Dark blue navy midnight indigo cobalt deep blue",
    "Light Blue":    "Light blue sky blue pale blue baby blue powder blue",
    "Red":           "Red crimson scarlet",
    "Dark Red":      "Dark red wine burgundy maroon deep red",
    "Pink":          "Pink rose blush fuchsia hot pink",
    "Light Pink":    "Light pink pastel pink pale pink soft pink blush",
    "Dark Pink":     "Dark pink deep rose raspberry",
    "Green":         "Green olive sage forest emerald",
    "Dark Green":    "Dark green bottle green forest green hunter green",
    "Light Green":   "Light green mint sage pale green",
    "Yellow":        "Yellow lemon sunshine bright yellow",
    "Dark Yellow":   "Dark yellow mustard gold ochre",
    "Beige":         "Beige nude tan cream sand natural",
    "Grey":          "Grey gray ash stone pebble",
    "Dark Grey":     "Dark grey charcoal slate graphite",
    "Light Grey":    "Light grey pale grey silver grey",
    "Yellowish Brown": "Brown chocolate caramel rust earth toned earthy warm",
    "Dark Orange":   "Orange amber burnt orange terracotta warm toned rust",
    "Purple":        "Purple plum violet",
    "Light Purple":  "Light purple lilac lavender",
    "Turquoise":     "Turquoise teal aqua cyan",
    "Gold":          "Gold metallic golden",
    "Silver":        "Silver metallic chrome",
    "Off White":     "Off white ivory cream ecru natural white",
}

_PRODUCT_VECTOR_ANCHORS = {
    "Dress":           "Dress gown frock midi maxi mini wrap shift",
    "Skirt":           "Skirt midi maxi mini pencil A-line pleated",
    "Trousers":        "Trousers pants jeans slacks chinos wide leg",
    "Blouse":          "Blouse shirt top button-up formal feminine",
    "Shirt":           "Shirt button-down oxford formal casual",
    "Top":             "Top basic tee plain simple everyday",
    "T-shirt":         "T-shirt tee casual basic cotton jersey",
    "Vest top":        "Vest top tank top cami spaghetti strap sleeveless",
    "Sweater":         "Sweater jumper knit knitwear pullover warm cosy",
    "Hoodie":          "Hoodie sweatshirt hooded casual warm",
    "Jacket":          "Jacket coat outerwear denim leather puffer rain",
    "Blazer":          "Blazer suit jacket formal smart professional",
    "Cardigan":        "Cardigan knit open-front cosy layering",
    "Shorts":          "Shorts hot pants denim casual summer",
    "Leggings/Tights": "Leggings tights activewear gym sports workout",
    "Sneakers":        "Sneakers trainers shoes casual footwear",
    "Boots":           "Boots ankle knee-high chelsea cowboy footwear",
    "Sandals":         "Sandals heels mules flip flops summer footwear",
    "Bag":             "Bag handbag tote backpack purse clutch",
    "Scarf":           "Scarf wrap shawl neckwear",
    "Hat/beanie":      "Hat cap beanie headwear",
    "Swimsuit":        "Swimsuit bikini swimwear beach pool bathing suit",
}

_CLOTHING_HINT_WORDS = {
    "wear", "outfit", "clothing", "clothes", "garment", "style",
    "fashion", "item", "piece", "something to", "looking for",
    "need a", "want a", "show me", "find me", "recommend",
}


def extract_entities_vector(message: str, missing_fields: list) -> dict:
    """Tier 2: vector similarity. Returns KEY names, not descriptions."""
    model = _get_model()
    if model is None:
        return {}

    entities = {}

    if "colour_group_name" in missing_fields:
        colour_hint_words = [
            "colour", "color", "shade", "tone", "hue", "pastel", "bright",
            "dark", "light", "deep", "pale", "neutral", "warm", "cool",
            "muted", "vibrant", "earthy", "toned", "metallic",
        ]
        if any(w in message.lower() for w in colour_hint_words):
            names  = list(_COLOUR_VECTOR_ANCHORS.keys())
            descs  = list(_COLOUR_VECTOR_ANCHORS.values())
            idx, score = _best_match_index(message, descs)
            if score > 0.35:
                colour_name = names[idx]
                if colour_name in VALID_COLOURS:
                    entities["colour_group_name"] = colour_name

    if "product_type_name" in missing_fields:
        msg_lower = message.lower()
        clothing_hint = any(w in msg_lower for w in _CLOTHING_HINT_WORDS)
        if clothing_hint:
            names  = list(_PRODUCT_VECTOR_ANCHORS.keys())
            descs  = list(_PRODUCT_VECTOR_ANCHORS.values())
            idx, score = _best_match_index(message, descs)
            if score > 0.45:
                product_name = names[idx]
                if product_name in VALID_PRODUCT_TYPES:
                    entities["product_type_name"] = product_name

    return entities


# ── Tier 3: Ollama LLM ────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Extract fashion shopping attributes from this message.
Return ONLY a JSON object. No explanation. No extra text.

STRICT RULES:
1. colour_group_name must be EXACTLY one of: Black, White, Red, Dark Red, Blue, Dark Blue, Light Blue, Pink, Light Pink, Dark Pink, Green, Dark Green, Light Green, Yellow, Dark Yellow, Beige, Grey, Dark Grey, Light Grey, Orange, Dark Orange, Purple, Light Purple, Turquoise, Gold, Silver, Off White, Yellowish Brown.
   - pastel → Light Pink. Navy/midnight → Dark Blue. Earthy/warm toned → Yellowish Brown.
   - NEVER put pattern words (Floral, Stripe) in colour_group_name.
2. product_type_name must be EXACTLY one of: Dress, Skirt, Trousers, Blouse, Shirt, Top, T-shirt, Vest top, Sweater, Hoodie, Jacket, Blazer, Cardigan, Shorts, Leggings/Tights, Sneakers, Boots, Sandals, Bag, Tote bag, Backpack, Scarf, Hat/beanie, Swimsuit, Bra, Heels, Flat shoes.
   - midi/maxi/mini skirt → Skirt. Midi/wrap/maxi dress → Dress. Activewear → Leggings/Tights.
3. graphical_appearance_name must be EXACTLY one of: Solid, Stripe, Check, Dot, Front print, All over pattern, Lace, Denim, Sequin, Embroidery.
   - Floral → Front print. Leopard/animal print → Front print. Polka dot → Dot. Plaid → Check.
4. price_max and price_min must be numbers. thirty=30, fifty=50, forty=40.
5. occasion must be one of: casual, formal, work, gym, beach, party, wedding, date night, summer, winter.
   - graduation/prom/gala → formal. office/meeting/interview → work. workout/yoga → gym. holiday/resort → beach.
6. style must be one of: casual, formal, smart casual, sporty, elegant, minimalist, classic, relaxed, trendy.
7. Return {{}} for: greetings (hi, hello, thanks, ok, yes, no), questions about item properties (what material, what size), comparisons (which is better), feedback reactions (I love it, I hate it).
8. Only include fields clearly stated. Do not invent or guess.

Examples:
"Something floral in a pastel shade" → {{"graphical_appearance_name":"Front print","colour_group_name":"Light Pink"}}
"A midi skirt for the office under £40" → {{"product_type_name":"Skirt","occasion":"work","price_max":40}}
"Under thirty pounds" → {{"price_max":30}}
"For my sister graduation" → {{"occasion":"formal","style":"elegant"}}
"Warm toned and earthy" → {{"colour_group_name":"Yellowish Brown"}}
"Thanks!" → {{}}
"What material is it?" → {{}}
"I love it" → {{}}
"Navy dress" → {{"colour_group_name":"Dark Blue","product_type_name":"Dress"}}

Message: {message}
JSON:"""


async def extract_entities_llm(message: str) -> dict:
    """
    Tier 3: LLM entity extraction.
    Routes to Groq or local Ollama based on LLM_PROVIDER env variable.
    Groq uses llama-3.1-8b-instant (better JSON accuracy than llama3.2:3b).
    Ollama uses llama3.2:3b (smaller, faster for local use).
    """
    if len(message.strip().split()) <= 2:
        return {}

    prompt = EXTRACTION_PROMPT.format(message=message)

    try:
        if LLM_PROVIDER == "groq":
            # ── Groq API ──────────────────────────────────────────────────────
            print(f"[ENTITY-LLM] calling Groq: model={GROQ_ENTITY_MODEL} msg='{message[:50]}'")
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type":  "application/json",
                    },
                    json={
                        "model":       GROQ_ENTITY_MODEL,
                        "messages":    [
                            {"role": "system", "content": "You are a fashion entity extractor. Return ONLY valid JSON. No explanation."},
                            {"role": "user",   "content": prompt},
                        ],
                        "max_tokens":  150,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    }
                )
            if response.status_code != 200:
                print(f"[ENTITY-LLM] Groq error {response.status_code}: {response.text[:100]}")
                return {}
            raw = response.json()["choices"][0]["message"]["content"].strip()
            print(f"[ENTITY-LLM] Groq response: {raw[:100]}")
        else:
            # ── Local Ollama ──────────────────────────────────────────────────
            print(f"[ENTITY-LLM] calling Ollama: model={OLLAMA_MODEL} msg='{message[:50]}'")
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model":  OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.0, "num_predict": 150},
                    }
                )
            if response.status_code != 200:
                return {}
            raw = response.json().get("response", "").strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        entities = json.loads(raw)
        return {k: v for k, v in entities.items()
                if v is not None and str(v).strip() != ""}
    except Exception:
        return {}


# ── Unified extraction ─────────────────────────────────────────────────────────

async def extract_entities(message: str, label: str = None) -> dict:
    print(f"\n[ENTITY] ━━━ extract_entities ━━━ msg='{message[:60]}' label={label}")
    """
    Main extraction function.

    Args:
        message: The user's message text
        label:   Optional DistilBERT label. If provided and not in
                 EXTRACTION_LABELS, returns {} immediately — no extraction
                 needed for non-search turns.

    Returns validated dict of entities. All values are guaranteed to be
    valid for use as retrieval filters against sample_articles.csv.
    """
    # Only extract for catalog search labels
    if label is not None and label not in EXTRACTION_LABELS:
        return {}

    # Tier 1
    tier1 = extract_entities_keyword(message)

    # Tier 2: fill missing colour and product type
    missing = [f for f in ["colour_group_name", "product_type_name"]
               if f not in tier1]
    tier2 = extract_entities_vector(message, missing)

    # Tier 3: LLM always runs
    tier3 = await extract_entities_llm(message)

    # Merge: Tier 1 base, Tier 2 fills gaps, Tier 3 overrides except price
    merged = {**tier1}
    for k, v in tier2.items():
        if k not in merged:
            merged[k] = v
    price_protected = bool(tier1.keys() & {"price_max", "price_min"})
    for k, v in tier3.items():
        if k in {"price_max", "price_min"} and price_protected:
            pass  # Protect Tier 1 price
        else:
            merged[k] = v

    # Final validation: drop any value not in the predefined valid sets
    return _validate_entities(merged)


# ── Test harness ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # Label, message, expected_note
        ("INITIAL_REQUEST", "I want a black dress under £50",     "Black, Dress, price_max=50"),
        ("INITIAL_REQUEST", "Something floral in a pastel shade", "Front print, Light Pink"),
        ("INITIAL_REQUEST", "A midi-length skirt for the office", "Skirt, work"),
        ("INITIAL_REQUEST", "Under thirty pounds please",         "price_max=30"),
        ("INITIAL_REQUEST", "For my sister's graduation",         "formal"),
        ("INITIAL_REQUEST", "Show me casual dresses",             "casual, Dress"),
        ("INITIAL_REQUEST", "A trendy outfit for a beach holiday under £40", "beach, price_max=40"),
        ("INITIAL_REQUEST", "Something warm toned and earthy",    "Yellowish Brown"),
        ("INITIAL_REQUEST", "Navy dress",                         "Dark Blue, Dress"),
        ("INITIAL_REQUEST", "Something minimalist in white",      "minimalist, White"),
        # These should return {} — wrong label
        ("ATTRIBUTE_QUESTION", "What material is it?",            "should be {}"),
        ("COMPARISON",         "Which one is cheaper?",           "should be {}"),
        ("FEEDBACK",           "I love it!",                      "should be {}"),
        ("CHITCHAT",           "Thanks!",                         "should be {}"),
        # Off-topic — caught by relevance check before extraction
        (None, "What is the weather today?",  "off-topic"),
        (None, "Who won the football match?", "off-topic"),
    ]

    async def run():
        print("ENTITY EXTRACTION TEST — FINAL VERSION")
        print("="*60)
        for label, msg, note in test_cases:
            relevant, conf = is_fashion_relevant(msg)
            print(f"\nLabel:{label or 'N/A':20} '{msg}'")
            print(f"  Expected: {note}")
            if not relevant:
                print(f"  → Off-topic detected ({conf:.2f})")
                continue
            entities = await extract_entities(msg, label=label)
            if entities:
                for k, v in entities.items():
                    print(f"  {k}: {v}")
            else:
                print("  {} (no entities / skipped)")

    asyncio.run(run())
