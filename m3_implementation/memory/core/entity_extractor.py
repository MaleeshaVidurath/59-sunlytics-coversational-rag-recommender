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


# ── Fashion relevance detection ────────────────────────────────────────────────

_FASHION_ANCHORS = [
    "I want to buy clothes shoes bags or fashion accessories",
    "Show me dresses tops shirts jackets trousers shoes",
    "I need an outfit to wear for an event or occasion",
    "What colour material fabric or style is this clothing item",
    "I like or dislike this fashion recommendation show me something else",
    "Tell me more details about this clothing item recommendation",
    "I am looking for something casual formal sporty elegant to wear",
    "Recommend me fashion items within my budget price range",
]

_OFF_TOPIC_ANCHORS = [
    "What is the weather temperature forecast today",
    "Tell me a funny joke or story",
    "Who scored in the football soccer cricket match",
    "What is the capital city country president prime minister",
    "Help me with coding programming homework assignment",
    "What happened in the news today current events",
    "How do I cook this recipe bake ingredients",
    "What time is it current date",
]

# Word boundary patterns — prevents "show"→"show me dresses", "war"→"warm"
_OFF_TOPIC_PATTERNS = [
    r'\bweather\b', r'\bforecast\b', r'\btemperature\b',
    r'\bjoke\b', r'\bfunny\b', r'\blaugh\b',
    r'\bfootball\b', r'\bsoccer\b', r'\bcricket\b', r'\bbasketball\b',
    r'\bpresident\b', r'\belection\b', r'\bpolitics\b',
    r'\brecipe\b', r'\bcooking\b', r'\bbaking\b',
    r'\bhomework\b', r'\bprogramming\b', r'\balgorithm\b',
    r'\bbitcoin\b', r'\bcrypto\b', r'\bstock price\b',
    r'\bdoctor\b', r'\bhospital\b', r'\bsymptom\b',
    r'\btranslate\b', r'\bgrammar\b',
]

_RELEVANCE_THRESHOLD = 0.20
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


def is_fashion_relevant(message: str) -> tuple:
    """
    Returns (is_relevant: bool, confidence: float).
    Uses word-boundary regex first, then vector similarity.
    Short messages (≤2 words) are always considered relevant.
    """
    msg = message.lower().strip()

    if len(msg.split()) <= 2:
        return True, 1.0

    for pattern in _OFF_TOPIC_PATTERNS:
        if re.search(pattern, msg):
            return False, 0.0

    model = _get_model()
    if model is None:
        return True, 0.5

    all_anchors = _FASHION_ANCHORS + _OFF_TOPIC_ANCHORS
    embeddings  = model.encode([message] + all_anchors)
    msg_emb     = embeddings[0]

    fashion_scores  = [_cosine(msg_emb, embeddings[i + 1])
                       for i in range(len(_FASHION_ANCHORS))]
    offtopic_scores = [_cosine(msg_emb, embeddings[i + 1 + len(_FASHION_ANCHORS)])
                       for i in range(len(_OFF_TOPIC_ANCHORS))]

    max_fashion  = max(fashion_scores)
    max_offtopic = max(offtopic_scores)

    if max_offtopic > max_fashion + 0.20:
        return False, max_offtopic
    if max_fashion < _RELEVANCE_THRESHOLD:
        return False, max_fashion

    return True, max_fashion


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


def _extract_price(msg: str) -> dict:
    result = {}
    patterns_max = [
        r'under\s+[£$€]?\s*(\d+(?:\.\d+)?)',
        r'below\s+[£$€]?\s*(\d+(?:\.\d+)?)',
        r'less\s+than\s+[£$€]?\s*(\d+(?:\.\d+)?)',
        r'[£$€]\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:pounds?|dollars?|euros?|gbp)',
    ]
    for pattern in patterns_max:
        m = re.search(pattern, msg)
        if m:
            result["price_max"] = float(m.group(1))
            return result
    # Text number words — only when price context present
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf'\b{re.escape(word)}\b', msg):
            if any(kw in msg for kw in [
                "pound", "dollar", "euro", "budget",
                "under", "below", "less", "cheap", "afford"
            ]):
                if any(kw in msg for kw in
                       ["over", "above", "more than", "at least"]):
                    result["price_min"] = float(value)
                else:
                    result["price_max"] = float(value)
                return result
    if any(kw in msg for kw in ["budget", "cheap", "affordable"]):
        result["price_max"] = 35.0
    elif any(kw in msg for kw in ["luxury", "premium", "high-end", "designer"]):
        result["price_min"] = 80.0
    return result


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
