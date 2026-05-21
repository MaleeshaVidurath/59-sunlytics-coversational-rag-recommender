"""
Fashion Psychology Knowledge Base — NOVELTY 5

Three-layer expert knowledge base grounded in peer-reviewed fashion psychology research:

  Layer 1 — COLOR_PSYCHOLOGY
    Maps colour_group_name values to psychological emotions, occasion suitability,
    and Kansei emotional word associations.
    Sources:
      - Kodzoman D. The Psychology of Clothing: Meaning of Colors, Body Image and
        Gender Expression in Fashion. Textile & Leather Review. 2019;2(2):90-103.
      - Guo Hui et al. Research on the Application of Color Psychology in Fashion
        Design. Clausius Scientific Press, 2023.
      - Soroka I. Colours in clothes and psychological functioning. Sapienza:
        International Journal of Interdisciplinary Studies. 2024;5(4).
      - Dantas I.J. et al. The psychological dimension of colors: a systematic
        literature review. Research, Society and Development. 2022;11(5).

  Layer 2 — OCCASION_RULES
    Maps occasion keywords (from soft_constraints) to preferred and avoided
    colour_group_name and product_type_name values, grounded in occasion semantics.
    Sources:
      - Usip P.U., Osang F.B. & Konyeha S. An Ontology-Driven Fashion Recommender
        System for Occasion-Specific Apparels. Journal of Advances in Mathematical
        & Computational Sciences. 2020;8(1):67-76.
      - Kodzoman D. (2019) — formal vs casual clothing psychology findings.

  Layer 3 — KANSEI_MAPPING
    Maps emotional/style words to preferred colour_group_name, product_type_name,
    and graphical_appearance_name values using Kansei Engineering methodology.
    Sources:
      - Nagamachi M. Kansei Engineering: A new ergonomic consumer-oriented technology
        for product development. Int. J. Industrial Ergonomics. 1995;15:3-11.
      - Lu H., Chen Y. & Du J. An Interactive System Based on Kansei Engineering to
        Support Clothing Design Process. Research Journal of Applied Sciences,
        Engineering and Technology. 2013;6(24):4531-4535.

All keys use exact H&M dataset column values (verified against sample_articles.csv).
"""

# ============================================================
# LAYER 1 — COLOR PSYCHOLOGY
# key = exact colour_group_name value in H&M dataset
# ============================================================
COLOR_PSYCHOLOGY = {
    "Black": {
        "emotions":        ["authority", "power", "sophistication", "elegance"],
        "good_occasions":  ["interview", "office", "formal", "evening", "party", "date"],
        "bad_occasions":   ["beach", "gym"],
        "kansei_match":    ["elegant", "professional", "bold", "sophisticated", "minimalist", "glamorous"],
        "psych_note":      "Conveys authority and competence — ideal for high-stakes settings (Kodzoman, 2019)",
    },
    "Dark Blue": {
        "emotions":        ["trust", "calm", "reliability", "rationality", "intelligence"],
        "good_occasions":  ["interview", "office", "formal", "date", "casual"],
        "bad_occasions":   ["beach"],
        "kansei_match":    ["professional", "classic", "calm", "elegant", "trustworthy"],
        "psych_note":      "Most universally trusted colour — projects credibility (Guo Hui et al., 2023)",
    },
    "Blue": {
        "emotions":        ["calm", "trust", "openness", "clarity"],
        "good_occasions":  ["casual", "office", "date"],
        "bad_occasions":   [],
        "kansei_match":    ["calm", "classic", "casual", "fresh"],
        "psych_note":      "Cool inward colour evoking the ocean and calmness (Guo Hui et al., 2023)",
    },
    "White": {
        "emotions":        ["purity", "cleanliness", "freshness", "simplicity", "openness"],
        "good_occasions":  ["casual", "office", "wedding", "summer", "beach"],
        "bad_occasions":   [],
        "kansei_match":    ["minimalist", "clean", "fresh", "simple", "classic"],
        "psych_note":      "Projects clarity and openness; light-coloured clothing brings a bright relaxed feeling (Guo Hui et al., 2023)",
    },
    "Off White": {
        "emotions":        ["warmth", "softness", "understated elegance"],
        "good_occasions":  ["casual", "office", "wedding", "date"],
        "bad_occasions":   [],
        "kansei_match":    ["minimalist", "elegant", "classic", "soft"],
        "psych_note":      "Warm version of white — approachable and refined",
    },
    "Grey": {
        "emotions":        ["neutrality", "balance", "professionalism", "calm"],
        "good_occasions":  ["office", "interview", "casual", "formal"],
        "bad_occasions":   ["party", "evening"],
        "kansei_match":    ["professional", "minimalist", "calm", "classic"],
        "psych_note":      "Safe and versatile — suits most professional and neutral contexts (Kodzoman, 2019)",
    },
    "Dark Grey": {
        "emotions":        ["authority", "seriousness", "professionalism"],
        "good_occasions":  ["office", "interview", "formal"],
        "bad_occasions":   ["party", "beach"],
        "kansei_match":    ["professional", "sophisticated", "minimalist"],
        "psych_note":      "Dark colours convey modesty and composure (Guo Hui et al., 2023)",
    },
    "Light Grey": {
        "emotions":        ["softness", "calm", "approachability"],
        "good_occasions":  ["casual", "office"],
        "bad_occasions":   ["formal_evening"],
        "kansei_match":    ["casual", "minimalist", "calm", "soft"],
        "psych_note":      "Light-coloured clothing brings a bright relaxed feeling (Guo Hui et al., 2023)",
    },
    "Red": {
        "emotions":        ["excitement", "passion", "energy", "confidence", "stimulation"],
        "good_occasions":  ["date", "party", "evening"],
        "bad_occasions":   ["interview", "office", "formal_conservative"],
        "kansei_match":    ["bold", "romantic", "energetic", "glamorous", "passionate"],
        "psych_note":      "Stimulates adrenaline — powerful for social settings, too aggressive for professional (Guo Hui et al., 2023; Kodzoman, 2019)",
    },
    "Dark Red": {
        "emotions":        ["depth", "elegance", "passion", "luxury"],
        "good_occasions":  ["evening", "date", "party", "formal"],
        "bad_occasions":   ["interview", "office"],
        "kansei_match":    ["elegant", "glamorous", "romantic", "bold"],
        "psych_note":      "Deep red adds sophistication to red's passion (Kodzoman, 2019)",
    },
    "Light Red": {
        "emotions":        ["playfulness", "energy", "warmth"],
        "good_occasions":  ["casual", "party", "date"],
        "bad_occasions":   ["interview", "office"],
        "kansei_match":    ["bold", "playful", "casual"],
        "psych_note":      "Lighter red retains energy while feeling more approachable",
    },
    "Pink": {
        "emotions":        ["softness", "romance", "playfulness", "femininity", "warmth"],
        "good_occasions":  ["date", "casual", "party"],
        "bad_occasions":   ["interview", "formal_conservative"],
        "kansei_match":    ["romantic", "soft", "playful", "feminine"],
        "psych_note":      "Evokes warmth and approachability — strong romantic signal (Kodzoman, 2019)",
    },
    "Dark Pink": {
        "emotions":        ["confidence", "romance", "warmth"],
        "good_occasions":  ["date", "party", "evening"],
        "bad_occasions":   ["interview", "office"],
        "kansei_match":    ["romantic", "bold", "feminine", "glamorous"],
        "psych_note":      "Bold version of pink — confident romantic statement",
    },
    "Light Pink": {
        "emotions":        ["gentleness", "softness", "innocence"],
        "good_occasions":  ["casual", "date"],
        "bad_occasions":   ["interview", "office"],
        "kansei_match":    ["soft", "romantic", "gentle", "feminine"],
        "psych_note":      "Very gentle and approachable feminine tone",
    },
    "Beige": {
        "emotions":        ["warmth", "naturalness", "understated", "calm"],
        "good_occasions":  ["casual", "office", "date", "formal"],
        "bad_occasions":   ["party", "evening"],
        "kansei_match":    ["classic", "minimalist", "elegant", "natural", "calm"],
        "psych_note":      "Neutral warm tone — versatile and understated",
    },
    "Dark Beige": {
        "emotions":        ["earthiness", "warmth", "reliability"],
        "good_occasions":  ["casual", "office"],
        "bad_occasions":   [],
        "kansei_match":    ["natural", "classic", "calm"],
        "psych_note":      "Grounded, earthy neutral",
    },
    "Light Beige": {
        "emotions":        ["lightness", "softness", "warmth"],
        "good_occasions":  ["casual", "summer"],
        "bad_occasions":   [],
        "kansei_match":    ["minimalist", "natural", "fresh"],
        "psych_note":      "Light warm neutral — airy and fresh",
    },
    "Yellow": {
        "emotions":        ["happiness", "optimism", "energy", "warmth", "creativity"],
        "good_occasions":  ["casual", "beach", "summer"],
        "bad_occasions":   ["interview", "office", "formal"],
        "kansei_match":    ["bold", "playful", "energetic", "casual"],
        "psych_note":      "Brightest colour — signals happiness and energy but lacks gravitas (Guo Hui et al., 2023)",
    },
    "Dark Yellow": {
        "emotions":        ["warmth", "earthiness", "creativity"],
        "good_occasions":  ["casual"],
        "bad_occasions":   ["interview", "formal"],
        "kansei_match":    ["bold", "earthy", "casual"],
        "psych_note":      "Warmer, more subdued yellow — creative but grounded",
    },
    "Green": {
        "emotions":        ["naturalness", "peace", "balance", "freshness", "youth"],
        "good_occasions":  ["casual", "outdoor", "beach"],
        "bad_occasions":   ["formal", "evening"],
        "kansei_match":    ["natural", "fresh", "casual", "calm"],
        "psych_note":      "Natural colour symbolising youth, peace and balance (Guo Hui et al., 2023)",
    },
    "Dark Green": {
        "emotions":        ["sophistication", "naturalness", "depth"],
        "good_occasions":  ["casual", "office", "outdoor"],
        "bad_occasions":   ["party", "evening"],
        "kansei_match":    ["natural", "elegant", "calm", "sophisticated"],
        "psych_note":      "Deep green adds sophistication to nature's palette",
    },
    "Light Green": {
        "emotions":        ["freshness", "youth", "playfulness"],
        "good_occasions":  ["casual", "beach", "summer"],
        "bad_occasions":   ["formal", "interview"],
        "kansei_match":    ["fresh", "playful", "casual", "natural"],
        "psych_note":      "Fresh and youthful — spring energy",
    },
    "Purple": {
        "emotions":        ["creativity", "mystery", "luxury", "spirituality"],
        "good_occasions":  ["evening", "party", "date"],
        "bad_occasions":   ["office", "interview"],
        "kansei_match":    ["glamorous", "creative", "romantic", "bold"],
        "psych_note":      "Purple adds mystery and authority — historically a colour of royalty",
    },
    "Dark Purple": {
        "emotions":        ["luxury", "depth", "mystery", "elegance"],
        "good_occasions":  ["evening", "formal", "party"],
        "bad_occasions":   ["office", "interview"],
        "kansei_match":    ["glamorous", "elegant", "sophisticated"],
        "psych_note":      "Deep purple — luxurious and dramatic",
    },
    "Orange": {
        "emotions":        ["cheerfulness", "warmth", "energy", "creativity", "friendliness"],
        "good_occasions":  ["casual", "beach", "summer"],
        "bad_occasions":   ["interview", "formal", "office"],
        "kansei_match":    ["bold", "playful", "energetic", "casual"],
        "psych_note":      "Energetic and sociable — too casual for professional settings",
    },
    "Gold": {
        "emotions":        ["luxury", "glamour", "success", "warmth"],
        "good_occasions":  ["evening", "party", "formal"],
        "bad_occasions":   ["casual", "office", "interview"],
        "kansei_match":    ["glamorous", "bold", "elegant"],
        "psych_note":      "Gold signals luxury and celebration (Guo Hui et al., 2023)",
    },
    "Turquoise": {
        "emotions":        ["freshness", "creativity", "calm", "tropical"],
        "good_occasions":  ["casual", "beach", "summer"],
        "bad_occasions":   ["interview", "formal"],
        "kansei_match":    ["fresh", "playful", "bold", "casual"],
        "psych_note":      "Tropical and refreshing — leisure and creativity",
    },
    "Light Blue": {
        "emotions":        ["calm", "freshness", "trust", "openness"],
        "good_occasions":  ["casual", "office", "summer"],
        "bad_occasions":   [],
        "kansei_match":    ["fresh", "casual", "calm", "classic"],
        "psych_note":      "Lighter blue — friendly and approachable version of trust",
    },
    "Silver": {
        "emotions":        ["modernity", "sophistication", "neutrality"],
        "good_occasions":  ["evening", "party"],
        "bad_occasions":   ["casual", "office"],
        "kansei_match":    ["glamorous", "modern", "elegant"],
        "psych_note":      "Metallic and contemporary — festive and sophisticated",
    },
}


# ============================================================
# LAYER 2 — OCCASION RULES
# key = occasion keyword from soft_constraints["occasion"]
# Sources: Usip et al. (2020); Kodzoman (2019)
# ============================================================
OCCASION_RULES = {
    "interview": {
        "preferred_colours":  ["Black", "Dark Blue", "Grey", "Dark Grey", "White", "Off White", "Beige"],
        "preferred_types":    ["Blazer", "Shirt", "Trousers", "Dress", "Blouse", "Coat"],
        "preferred_appearance": ["Solid", "Stripe", "Check"],
        "avoid_colours":      ["Red", "Yellow", "Orange", "Light Green", "Light Pink", "Gold", "Turquoise"],
        "avoid_types":        ["Hoodie", "Leggings/Tights", "Shorts", "T-shirt", "Swimsuit"],
        "psych_basis":        "Neutral and cool tones project competence and reliability (Kodzoman, 2019)",
        "score_weight":       0.18,
        "clip_visual_terms":  "structured blazer tailored shirt business professional neat clean crisp formal suit",
    },
    "office": {
        "preferred_colours":  ["Dark Blue", "Grey", "Dark Grey", "Black", "White", "Off White", "Beige", "Dark Beige"],
        "preferred_types":    ["Shirt", "Blouse", "Trousers", "Blazer", "Dress", "Sweater", "Cardigan"],
        "preferred_appearance": ["Solid", "Stripe", "Check"],
        "avoid_colours":      ["Orange", "Yellow", "Turquoise", "Gold", "Silver"],
        "avoid_types":        ["Hoodie", "Shorts", "Leggings/Tights", "T-shirt"],
        "psych_basis":        "Formal clothing conveys professionalism and rational thinking (Kodzoman, 2019)",
        "score_weight":       0.15,
        "clip_visual_terms":  "business casual smart polished workwear trousers blouse office professional tailored",
    },
    "date": {
        "preferred_colours":  ["Red", "Dark Red", "Dark Blue", "Black", "Pink", "Dark Pink", "Dark Purple"],
        "preferred_types":    ["Dress", "Blouse", "Top", "Skirt", "Jumpsuit/Playsuit"],
        "preferred_appearance": ["Solid", "Lace", "Dot", "All over pattern"],
        "avoid_colours":      ["Dark Grey", "Beige", "Greenish Khaki"],
        "avoid_types":        ["Hoodie", "Trousers", "Blazer"],
        "psych_basis":        "Warm tones and fitted silhouettes signal romantic intent (Kodzoman, 2019)",
        "score_weight":       0.15,
        "clip_visual_terms":  "romantic date night dress flirty feminine flattering lace stylish evening outfit",
    },
    "party": {
        "preferred_colours":  ["Red", "Black", "Dark Red", "Gold", "Dark Pink", "Purple", "Dark Purple", "Silver"],
        "preferred_types":    ["Dress", "Top", "Blouse", "Jumpsuit/Playsuit", "Skirt"],
        "preferred_appearance": ["Sequin", "Glittering/Metallic", "Lace", "Solid", "All over pattern"],
        "avoid_colours":      ["Beige", "Dark Grey", "Greenish Khaki"],
        "avoid_types":        ["Blazer", "Coat"],
        "psych_basis":        "Bold colours and statement pieces create social presence (Usip et al., 2020)",
        "score_weight":       0.15,
        "clip_visual_terms":  "party dress sequin sparkle bold festive glamorous night out colorful statement",
    },
    "casual": {
        "preferred_colours":  ["White", "Grey", "Light Grey", "Light Blue", "Beige", "Light Beige", "Green", "Light Green"],
        "preferred_types":    ["Top", "T-shirt", "Hoodie", "Sweater", "Trousers", "Shorts", "Jeans", "Cardigan"],
        "preferred_appearance": ["Solid", "Stripe", "All over pattern", "Front print"],
        "avoid_colours":      ["Gold", "Silver"],
        "avoid_types":        ["Blazer", "Coat"],
        "psych_basis":        "Casual clothing signals intimacy and familiarity (Kodzoman, 2019)",
        "score_weight":       0.10,
        "clip_visual_terms":  "casual everyday relaxed comfortable t-shirt jeans hoodie streetwear weekend outfit",
    },
    "formal": {
        "preferred_colours":  ["Black", "Dark Blue", "Grey", "Dark Grey", "White", "Gold", "Dark Red"],
        "preferred_types":    ["Dress", "Blazer", "Shirt", "Blouse", "Trousers", "Coat", "Tailored Waistcoat"],
        "preferred_appearance": ["Solid", "Stripe"],
        "avoid_colours":      ["Orange", "Yellow", "Light Green", "Turquoise"],
        "avoid_types":        ["Hoodie", "Shorts", "T-shirt", "Leggings/Tights"],
        "psych_basis":        "Formal attire signals social hierarchy and respect (Kodzoman, 2019)",
        "score_weight":       0.18,
        "clip_visual_terms":  "formal elegant gown suit tailored sophisticated refined luxury ceremony evening wear",
    },
    "evening": {
        "preferred_colours":  ["Black", "Dark Red", "Gold", "Dark Purple", "Silver", "Dark Blue"],
        "preferred_types":    ["Dress", "Blouse", "Top", "Jumpsuit/Playsuit", "Skirt"],
        "preferred_appearance": ["Sequin", "Glittering/Metallic", "Lace", "Solid"],
        "avoid_colours":      ["Beige", "Light Grey", "Light Green"],
        "avoid_types":        ["Hoodie", "Shorts", "T-shirt"],
        "psych_basis":        "Evening wear signals glamour and celebration (Usip et al., 2020)",
        "score_weight":       0.15,
        "clip_visual_terms":  "evening gown cocktail dress glamorous sophisticated night out dinner elegant sparkle",
    },
    "beach": {
        "preferred_colours":  ["White", "Light Blue", "Yellow", "Turquoise", "Light Green", "Orange", "Light Pink"],
        "preferred_types":    ["Swimsuit", "Shorts", "T-shirt", "Dress", "Sarong", "Bikini top", "Swimwear bottom"],
        "preferred_appearance": ["Solid", "Stripe", "Dot", "All over pattern"],
        "avoid_colours":      ["Black", "Dark Grey", "Dark Red"],
        "avoid_types":        ["Blazer", "Coat", "Trousers", "Hoodie"],
        "psych_basis":        "Bright vacation colours express joy and openness (Guo Hui et al., 2023)",
        "score_weight":       0.12,
        "clip_visual_terms":  "beach swimwear summer vacation bikini shorts colorful tropical light airy holiday",
    },
    "gym": {
        "preferred_colours":  ["Black", "Grey", "Dark Grey", "Blue", "Dark Blue"],
        "preferred_types":    ["Leggings/Tights", "Shorts", "T-shirt", "Hoodie", "Top"],
        "preferred_appearance": ["Solid", "Stripe"],
        "avoid_colours":      ["Gold", "Silver", "Sequin"],
        "avoid_types":        ["Dress", "Blazer", "Blouse", "Shirt"],
        "psych_basis":        "Functional dark tones signal focus and performance",
        "score_weight":       0.10,
        "clip_visual_terms":  "athletic sportswear workout activewear leggings gym performance functional sporty",
    },
    "wedding": {
        "preferred_colours":  ["White", "Off White", "Beige", "Light Pink", "Gold", "Dark Blue"],
        "preferred_types":    ["Dress", "Blouse", "Skirt", "Blazer"],
        "preferred_appearance": ["Lace", "Solid", "Embroidery"],
        "avoid_colours":      ["Black", "Red"],
        "avoid_types":        ["Hoodie", "T-shirt", "Shorts"],
        "psych_basis":        "Wedding occasions call for delicate, celebratory tones (Usip et al., 2020)",
        "score_weight":       0.15,
        "clip_visual_terms":  "wedding bridal ceremony elegant lace white floral delicate dress formal celebration",
    },
}


# ============================================================
# LAYER 3 — KANSEI EMOTIONAL MAPPING
# key = emotional/style word from soft_constraints["style"]
#       or detected in user_message
# Sources: Nagamachi (1995); Lu et al. (2013)
# ============================================================
KANSEI_MAPPING = {
    "elegant": {
        "preferred_colours":    ["Black", "Dark Blue", "White", "Off White", "Beige", "Dark Red", "Dark Grey"],
        "preferred_types":      ["Dress", "Blazer", "Blouse", "Trousers", "Coat", "Skirt"],
        "preferred_appearance": ["Solid", "Stripe"],
        "avoid_types":          ["Hoodie", "Leggings/Tights", "Shorts", "T-shirt"],
        "avoid_appearance":     ["Front print", "All over pattern"],
        "description":          "Structured silhouettes, solid colours, refined details (Lu et al., 2013)",
        "score_weight":         0.12,
    },
    "casual": {
        "preferred_colours":    ["Grey", "White", "Light Blue", "Beige", "Green", "Light Grey"],
        "preferred_types":      ["Top", "T-shirt", "Hoodie", "Sweater", "Trousers", "Shorts", "Cardigan"],
        "preferred_appearance": ["Solid", "Stripe", "Front print"],
        "avoid_types":          ["Blazer", "Coat"],
        "avoid_appearance":     ["Sequin", "Glittering/Metallic"],
        "description":          "Relaxed fits, comfortable fabrics, muted tones (Lu et al., 2013)",
        "score_weight":         0.10,
    },
    "bold": {
        "preferred_colours":    ["Red", "Dark Red", "Yellow", "Orange", "Dark Pink", "Dark Purple", "Gold"],
        "preferred_types":      ["Dress", "Jacket", "Blouse", "Top", "Coat"],
        "preferred_appearance": ["All over pattern", "Front print", "Colour blocking", "Sequin"],
        "avoid_colours":        ["Beige", "Dark Grey", "Greenish Khaki"],
        "avoid_appearance":     ["Solid"],
        "description":          "Statement colours, eye-catching patterns, strong silhouettes (Nagamachi, 1995)",
        "score_weight":         0.12,
    },
    "romantic": {
        "preferred_colours":    ["Pink", "Dark Pink", "Light Pink", "Red", "White", "Beige"],
        "preferred_types":      ["Dress", "Blouse", "Top", "Skirt"],
        "preferred_appearance": ["Lace", "Dot", "Solid", "All over pattern", "Embroidery"],
        "avoid_types":          ["Hoodie", "Shorts", "Blazer"],
        "avoid_appearance":     ["Check", "Stripe"],
        "description":          "Soft colours, feminine silhouettes, delicate details (Lu et al., 2013)",
        "score_weight":         0.12,
    },
    "sporty": {
        "preferred_colours":    ["Grey", "Black", "Blue", "Dark Blue", "Light Grey"],
        "preferred_types":      ["T-shirt", "Shorts", "Leggings/Tights", "Hoodie", "Top", "Sweater"],
        "preferred_appearance": ["Solid", "Stripe"],
        "avoid_types":          ["Dress", "Blazer", "Blouse"],
        "avoid_appearance":     ["Sequin", "Lace", "Glittering/Metallic"],
        "description":          "Functional and dynamic — performance-oriented (Nagamachi, 1995)",
        "score_weight":         0.10,
    },
    "professional": {
        "preferred_colours":    ["Dark Blue", "Grey", "Dark Grey", "Black", "White", "Beige"],
        "preferred_types":      ["Blazer", "Shirt", "Blouse", "Trousers", "Dress", "Coat"],
        "preferred_appearance": ["Solid", "Stripe", "Check"],
        "avoid_types":          ["Hoodie", "Shorts", "Leggings/Tights"],
        "avoid_appearance":     ["All over pattern", "Front print", "Sequin"],
        "description":          "Structured and polished — signals competence and authority (Kodzoman, 2019)",
        "score_weight":         0.12,
    },
    "minimalist": {
        "preferred_colours":    ["Black", "White", "Grey", "Beige", "Dark Grey", "Off White"],
        "preferred_types":      ["T-shirt", "Top", "Trousers", "Dress", "Sweater", "Coat"],
        "preferred_appearance": ["Solid"],
        "avoid_appearance":     ["All over pattern", "Front print", "Sequin", "Embroidery", "Lace"],
        "description":          "Clean lines, neutral palette, no excess ornamentation (Lu et al., 2013)",
        "score_weight":         0.10,
    },
    "glamorous": {
        "preferred_colours":    ["Black", "Gold", "Dark Red", "Dark Purple", "Silver"],
        "preferred_types":      ["Dress", "Blouse", "Top", "Jumpsuit/Playsuit", "Skirt"],
        "preferred_appearance": ["Sequin", "Glittering/Metallic", "Lace", "Metallic"],
        "avoid_types":          ["Hoodie", "T-shirt", "Shorts"],
        "avoid_appearance":     ["Solid", "Stripe"],
        "description":          "Opulent materials, statement silhouettes, high-impact details (Nagamachi, 1995)",
        "score_weight":         0.12,
    },
    "classic": {
        "preferred_colours":    ["Dark Blue", "White", "Black", "Grey", "Beige", "Off White"],
        "preferred_types":      ["Shirt", "Trousers", "Blazer", "Dress", "Sweater", "Coat"],
        "preferred_appearance": ["Solid", "Stripe", "Check"],
        "avoid_appearance":     ["All over pattern", "Sequin", "Glittering/Metallic"],
        "description":          "Timeless cuts, muted palette, enduring style (Lu et al., 2013)",
        "score_weight":         0.10,
    },
    "feminine": {
        "preferred_colours":    ["Pink", "Light Pink", "White", "Beige", "Light Purple", "Red"],
        "preferred_types":      ["Dress", "Skirt", "Blouse", "Top"],
        "preferred_appearance": ["Lace", "Dot", "Embroidery", "All over pattern"],
        "avoid_types":          ["Blazer", "Trousers", "Hoodie"],
        "description":          "Soft silhouettes, delicate details, warm light tones (Kodzoman, 2019)",
        "score_weight":         0.10,
    },
    "edgy": {
        "preferred_colours":    ["Black", "Dark Grey", "Dark Red", "Dark Purple"],
        "preferred_types":      ["Jacket", "Dress", "Top", "Blazer", "Trousers"],
        "preferred_appearance": ["Solid", "Check", "Stripe"],
        "avoid_colours":        ["Light Pink", "Light Green", "Yellow", "Beige"],
        "description":          "Dark palette, sharp cuts, unconventional details (Nagamachi, 1995)",
        "score_weight":         0.10,
    },
    "natural": {
        "preferred_colours":    ["Beige", "Green", "Light Beige", "Dark Beige", "Greenish Khaki", "Light Green", "Yellowish Brown"],
        "preferred_types":      ["Top", "Trousers", "Dress", "Shirt", "Sweater", "Cardigan"],
        "preferred_appearance": ["Solid", "Melange", "Check"],
        "avoid_colours":        ["Gold", "Silver", "Sequin"],
        "description":          "Earthy tones, organic textures, grounded palette (Guo Hui et al., 2023)",
        "score_weight":         0.10,
    },
}


# ============================================================
# KANSEI WORD ALIASES — maps user phrasing to KB keys
# Handles synonyms from user natural language input
# ============================================================
KANSEI_ALIASES = {
    "chic":         "elegant",
    "classy":       "elegant",
    "sophisticated":"elegant",
    "refined":      "elegant",
    "laid back":    "casual",
    "relaxed":      "casual",
    "everyday":     "casual",
    "chill":        "casual",
    "statement":    "bold",
    "vibrant":      "bold",
    "striking":     "bold",
    "daring":       "bold",
    "cute":         "romantic",
    "lovely":       "romantic",
    "feminine":     "romantic",
    "sweet":        "romantic",
    "athletic":     "sporty",
    "active":       "sporty",
    "workout":      "sporty",
    "business":     "professional",
    "formal":       "professional",
    "work":         "professional",
    "smart":        "professional",
    "simple":       "minimalist",
    "clean":        "minimalist",
    "understated":  "minimalist",
    "luxurious":    "glamorous",
    "glam":         "glamorous",
    "fancy":        "glamorous",
    "timeless":     "classic",
    "traditional":  "classic",
    "preppy":       "classic",
    "earthy":       "natural",
    "organic":      "natural",
    "boho":         "natural",
    "bohemian":     "natural",
}

# ============================================================
# OCCASION ALIASES — maps user phrasing to KB keys
# ============================================================
OCCASION_ALIASES = {
    "job interview":    "interview",
    "work interview":   "interview",
    "work":             "office",
    "workplace":        "office",
    "business":         "office",
    "dinner":           "evening",
    "night out":        "evening",
    "gala":             "evening",
    "event":            "evening",
    "celebration":      "party",
    "birthday":         "party",
    "festival":         "casual",
    "everyday":         "casual",
    "weekend":          "casual",
    "swimming":         "beach",
    "summer":           "beach",
    "holiday":          "beach",
    "vacation":         "beach",
    "workout":          "gym",
    "exercise":         "gym",
    "sport":            "gym",
    "training":         "gym",
    "romantic dinner":  "date",
    "first date":       "date",
    "wedding":          "wedding",
    "ceremony":         "formal",
    "graduation":       "formal",
    "black tie":        "formal",
}


# ============================================================
# LAYER 4 — APPEARANCE RULES  (NOVELTY 5 — Improvement 1)
# key = exact graphical_appearance_name value in H&M dataset
# Maps visual texture/pattern to occasion and Kansei suitability.
# Sources: Nagamachi (1995); Usip et al. (2020); Lu et al. (2013)
# ============================================================
APPEARANCE_RULES = {
    "Solid": {
        "occasion_fit":   ["interview", "office", "formal", "casual", "gym", "date"],
        "kansei_fit":     ["minimalist", "classic", "elegant", "professional", "sporty"],
        "occasion_avoid": [],
        "kansei_avoid":   ["glamorous", "bold"],
        "score_weight":   0.05,
        "description":    "Clean and versatile — the universal safe choice (Lu et al., 2013)",
    },
    "Stripe": {
        "occasion_fit":   ["office", "casual", "interview"],
        "kansei_fit":     ["classic", "casual", "professional"],
        "occasion_avoid": ["evening", "wedding"],
        "kansei_avoid":   ["glamorous", "romantic"],
        "score_weight":   0.05,
        "description":    "Classic pattern — smart without being formal (Lu et al., 2013)",
    },
    "Check": {
        "occasion_fit":   ["office", "casual", "interview"],
        "kansei_fit":     ["classic", "professional", "natural"],
        "occasion_avoid": ["evening", "wedding", "party"],
        "kansei_avoid":   ["glamorous", "romantic", "feminine"],
        "score_weight":   0.05,
        "description":    "Traditional heritage pattern — smart casual",
    },
    "Sequin": {
        "occasion_fit":   ["evening", "party"],
        "kansei_fit":     ["glamorous", "bold"],
        "occasion_avoid": ["interview", "office", "casual", "gym", "wedding"],
        "kansei_avoid":   ["minimalist", "natural", "professional", "classic"],
        "score_weight":   0.08,
        "description":    "High-impact — evening and celebration only (Usip et al., 2020)",
    },
    "Lace": {
        "occasion_fit":   ["date", "evening", "wedding", "party"],
        "kansei_fit":     ["romantic", "feminine", "elegant"],
        "occasion_avoid": ["interview", "office", "gym"],
        "kansei_avoid":   ["sporty", "minimalist", "professional", "edgy"],
        "score_weight":   0.07,
        "description":    "Delicate and feminine texture (Kodzoman, 2019)",
    },
    "Denim": {
        "occasion_fit":   ["casual"],
        "kansei_fit":     ["casual", "classic", "natural", "edgy"],
        "occasion_avoid": ["interview", "formal", "evening", "wedding"],
        "kansei_avoid":   ["glamorous", "elegant"],
        "score_weight":   0.06,
        "description":    "Casual workwear-inspired — strictly leisure contexts",
    },
    "Embroidery": {
        "occasion_fit":   ["wedding", "date", "party", "evening"],
        "kansei_fit":     ["romantic", "feminine", "natural", "elegant"],
        "occasion_avoid": ["gym", "interview", "office"],
        "kansei_avoid":   ["minimalist", "sporty", "professional"],
        "score_weight":   0.06,
        "description":    "Artisan detail — warmth and craftsmanship (Lu et al., 2013)",
    },
    "All over pattern": {
        "occasion_fit":   ["casual", "beach", "party"],
        "kansei_fit":     ["bold", "casual", "natural"],
        "occasion_avoid": ["interview", "formal", "office"],
        "kansei_avoid":   ["minimalist", "professional", "classic"],
        "score_weight":   0.06,
        "description":    "Statement print — expressive and relaxed",
    },
    "Front print": {
        "occasion_fit":   ["casual", "beach"],
        "kansei_fit":     ["casual", "bold", "edgy"],
        "occasion_avoid": ["interview", "formal", "office", "evening", "wedding"],
        "kansei_avoid":   ["minimalist", "elegant", "professional", "classic"],
        "score_weight":   0.05,
        "description":    "Graphic front — casual self-expression",
    },
    "Glittering/Metallic": {
        "occasion_fit":   ["evening", "party"],
        "kansei_fit":     ["glamorous", "bold"],
        "occasion_avoid": ["interview", "office", "casual", "gym", "wedding"],
        "kansei_avoid":   ["minimalist", "natural", "professional"],
        "score_weight":   0.08,
        "description":    "Metallic sheen — high glamour nighttime (Nagamachi, 1995)",
    },
    "Dot": {
        "occasion_fit":   ["casual", "date"],
        "kansei_fit":     ["romantic", "feminine", "casual"],
        "occasion_avoid": ["interview", "formal"],
        "kansei_avoid":   ["minimalist", "professional"],
        "score_weight":   0.05,
        "description":    "Playful polka dot — approachable femininity",
    },
    "Melange": {
        "occasion_fit":   ["casual", "office"],
        "kansei_fit":     ["natural", "casual", "minimalist"],
        "occasion_avoid": ["evening", "party", "formal"],
        "kansei_avoid":   ["glamorous", "bold"],
        "score_weight":   0.04,
        "description":    "Heathered texture — relaxed and understated",
    },
    "Colour blocking": {
        "occasion_fit":   ["casual", "party"],
        "kansei_fit":     ["bold", "edgy"],
        "occasion_avoid": ["interview", "formal", "office"],
        "kansei_avoid":   ["minimalist", "classic", "professional"],
        "score_weight":   0.06,
        "description":    "Geometric contrast — modern statement",
    },
    "Jacquard/woven": {
        "occasion_fit":   ["formal", "office", "evening"],
        "kansei_fit":     ["elegant", "classic", "professional"],
        "occasion_avoid": ["gym", "casual", "beach"],
        "kansei_avoid":   ["sporty", "casual"],
        "score_weight":   0.06,
        "description":    "Woven structure — refined and formal (Nagamachi, 1995)",
    },
    "Animal print": {
        "occasion_fit":   ["party", "evening", "casual"],
        "kansei_fit":     ["bold", "edgy", "glamorous"],
        "occasion_avoid": ["interview", "formal", "office", "wedding"],
        "kansei_avoid":   ["minimalist", "classic", "professional", "natural"],
        "score_weight":   0.07,
        "description":    "Wild pattern — bold and expressive statement",
    },
    "Mixed": {
        "occasion_fit":   ["casual", "party"],
        "kansei_fit":     ["bold", "casual", "edgy"],
        "occasion_avoid": ["interview", "formal"],
        "kansei_avoid":   ["minimalist", "professional"],
        "score_weight":   0.04,
        "description":    "Mixed pattern — eclectic and casual",
    },
}


# ============================================================
# LAYER 5 — GENDER-AWARE COLOUR/TYPE RULES  (NOVELTY 5 — Improvement 2)
# key = exact index_group_name value in H&M dataset
# Applies demographic-specific colour expectations as KB penalties.
# Sources: Kodzoman (2019); gender expression in fashion
# ============================================================
GENDER_COLOUR_RULES = {
    "Menswear": {
        "unusual_colours":  ["Pink", "Light Pink", "Dark Pink", "Purple", "Light Red"],
        "usual_colours":    ["Dark Blue", "Blue", "Black", "Grey", "Dark Grey", "White",
                             "Beige", "Dark Green", "Dark Red", "Light Blue"],
        "preferred_types":  ["Shirt", "Trousers", "Blazer", "Sweater", "Hoodie",
                             "T-shirt", "Shorts", "Coat", "Jacket"],
        "unusual_types":    ["Dress", "Skirt", "Blouse"],
        "penalty_weight":   0.10,
        "description":      "Conventional menswear palette — darker neutral tones (Kodzoman, 2019)",
    },
    "Ladieswear": {
        "unusual_colours":  [],
        "usual_colours":    [],
        "preferred_types":  ["Dress", "Blouse", "Skirt", "Top"],
        "unusual_types":    [],
        "penalty_weight":   0.0,
        "description":      "No KB colour restrictions applied to Ladieswear",
    },
    "Divided": {
        "unusual_colours":  [],
        "usual_colours":    [],
        "preferred_types":  ["T-shirt", "Hoodie", "Shorts", "Top", "Trousers", "Jacket"],
        "unusual_types":    [],
        "penalty_weight":   0.04,
        "description":      "Young fashion — casual streetwear focus",
    },
    "Children": {
        "unusual_colours":  ["Black", "Dark Grey", "Dark Red", "Gold", "Silver"],
        "usual_colours":    ["Blue", "Light Blue", "Yellow", "Light Pink", "White",
                             "Light Green", "Orange", "Turquoise"],
        "preferred_types":  ["T-shirt", "Shorts", "Dress", "Leggings/Tights"],
        "unusual_types":    ["Blazer", "Coat"],
        "penalty_weight":   0.08,
        "description":      "Children's wear prefers bright playful colours",
    },
}


# ============================================================
# LAYER 6 — COLOR HARMONY  (NOVELTY 5 — Improvement 4)
# Encodes colour wheel relationships for multi-item recommendation sets.
# Used by harmony_score() to evaluate how well two recommended items pair.
# Sources: Guo Hui et al. (2023); traditional colour wheel theory
# ============================================================
COLOR_HARMONY = {
    "Black": {
        "complementary":  ["White", "Off White", "Gold", "Silver", "Red"],
        "analogous":      ["Dark Grey", "Grey"],
        "clashing":       [],
        "harmony_note":   "Black is universally harmonious — pairs with almost any colour",
    },
    "Dark Blue": {
        "complementary":  ["White", "Off White", "Orange", "Gold", "Beige"],
        "analogous":      ["Blue", "Dark Grey", "Black", "Light Blue"],
        "clashing":       ["Dark Green", "Dark Purple"],
        "harmony_note":   "Navy pairs with neutrals and warm accent tones",
    },
    "Blue": {
        "complementary":  ["White", "Orange", "Gold", "Beige"],
        "analogous":      ["Dark Blue", "Light Blue", "Grey"],
        "clashing":       ["Green", "Dark Green"],
        "harmony_note":   "Blue is most harmonious with warm neutrals and orange tones",
    },
    "White": {
        "complementary":  ["Black", "Dark Blue", "Red", "Gold"],
        "analogous":      ["Off White", "Light Grey", "Light Beige"],
        "clashing":       [],
        "harmony_note":   "White is universally versatile — neutral anchor",
    },
    "Off White": {
        "complementary":  ["Dark Blue", "Black", "Beige", "Gold"],
        "analogous":      ["White", "Light Beige", "Light Grey"],
        "clashing":       [],
        "harmony_note":   "Warm white — pairs especially well with warm tones",
    },
    "Grey": {
        "complementary":  ["Black", "White", "Dark Blue", "Red", "Pink"],
        "analogous":      ["Dark Grey", "Light Grey"],
        "clashing":       [],
        "harmony_note":   "Mid-grey is a safe neutral pairing base",
    },
    "Dark Grey": {
        "complementary":  ["White", "Gold", "Red", "Light Blue"],
        "analogous":      ["Black", "Grey"],
        "clashing":       [],
        "harmony_note":   "Dark grey pairs well with light and warm accent tones",
    },
    "Light Grey": {
        "complementary":  ["White", "Dark Blue", "Pink", "Beige"],
        "analogous":      ["Grey", "Dark Grey"],
        "clashing":       [],
        "harmony_note":   "Light grey is a soft neutral — easy to pair",
    },
    "Red": {
        "complementary":  ["White", "Black", "Dark Blue", "Gold"],
        "analogous":      ["Dark Red", "Orange", "Dark Pink"],
        "clashing":       ["Pink", "Orange"],
        "harmony_note":   "Red needs neutral anchors to avoid visual competition",
    },
    "Dark Red": {
        "complementary":  ["White", "Off White", "Black", "Gold", "Beige"],
        "analogous":      ["Red", "Dark Pink", "Dark Purple"],
        "clashing":       ["Orange", "Light Pink"],
        "harmony_note":   "Burgundy pairs with neutrals and earthy tones",
    },
    "Light Red": {
        "complementary":  ["White", "Black", "Dark Blue"],
        "analogous":      ["Red", "Dark Pink", "Orange"],
        "clashing":       ["Pink", "Yellow"],
        "harmony_note":   "Light red pairs with strong neutrals",
    },
    "Pink": {
        "complementary":  ["White", "Grey", "Dark Blue", "Black"],
        "analogous":      ["Light Pink", "Dark Pink", "Red"],
        "clashing":       ["Orange", "Yellow", "Red"],
        "harmony_note":   "Pink pairs with grey or white to avoid over-softness",
    },
    "Dark Pink": {
        "complementary":  ["Black", "White", "Dark Blue", "Grey"],
        "analogous":      ["Pink", "Red", "Purple"],
        "clashing":       ["Orange", "Yellow", "Green"],
        "harmony_note":   "Hot pink needs strong neutrals to ground it",
    },
    "Light Pink": {
        "complementary":  ["White", "Grey", "Light Blue", "Beige"],
        "analogous":      ["Pink", "Dark Pink"],
        "clashing":       ["Orange", "Yellow"],
        "harmony_note":   "Pale pink is harmonious with neutrals and blues",
    },
    "Beige": {
        "complementary":  ["Dark Blue", "Black", "Gold", "Dark Red"],
        "analogous":      ["Off White", "Dark Beige", "Light Beige"],
        "clashing":       [],
        "harmony_note":   "Beige works beautifully with navy and earth tones",
    },
    "Dark Beige": {
        "complementary":  ["Dark Blue", "Black", "Gold"],
        "analogous":      ["Beige", "Light Beige", "Dark Green"],
        "clashing":       [],
        "harmony_note":   "Dark beige is a grounded earthy neutral",
    },
    "Light Beige": {
        "complementary":  ["Dark Blue", "Black", "Dark Green"],
        "analogous":      ["Beige", "Off White", "White"],
        "clashing":       [],
        "harmony_note":   "Light beige is airy and fresh — pairs with darks",
    },
    "Yellow": {
        "complementary":  ["Dark Blue", "Purple", "Black"],
        "analogous":      ["Orange", "Dark Yellow", "Gold"],
        "clashing":       ["Pink", "Light Green"],
        "harmony_note":   "Yellow pops against dark navy or purple",
    },
    "Dark Yellow": {
        "complementary":  ["Dark Blue", "Black"],
        "analogous":      ["Yellow", "Orange", "Beige"],
        "clashing":       ["Pink", "Green"],
        "harmony_note":   "Mustard pairs with dark neutrals and earthy tones",
    },
    "Green": {
        "complementary":  ["Red", "Dark Red", "White", "Beige"],
        "analogous":      ["Dark Green", "Light Green"],
        "clashing":       ["Blue", "Dark Blue"],
        "harmony_note":   "Green pairs with earth tones and red accents",
    },
    "Dark Green": {
        "complementary":  ["Beige", "Off White", "Gold", "Dark Red"],
        "analogous":      ["Green", "Greenish Khaki"],
        "clashing":       ["Dark Blue", "Dark Purple"],
        "harmony_note":   "Forest green is elegant with cream and gold",
    },
    "Light Green": {
        "complementary":  ["White", "Light Pink", "Beige"],
        "analogous":      ["Green", "Yellow", "Turquoise"],
        "clashing":       ["Orange", "Red"],
        "harmony_note":   "Mint green pairs with neutrals and pastels",
    },
    "Purple": {
        "complementary":  ["Yellow", "Gold", "White", "Black"],
        "analogous":      ["Dark Purple", "Dark Pink", "Dark Blue"],
        "clashing":       ["Orange", "Red"],
        "harmony_note":   "Purple pairs with gold and white for elegance",
    },
    "Dark Purple": {
        "complementary":  ["Gold", "White", "Off White", "Black"],
        "analogous":      ["Purple", "Dark Blue", "Dark Red"],
        "clashing":       ["Dark Green", "Orange"],
        "harmony_note":   "Deep purple is richly complemented by gold and ivory",
    },
    "Orange": {
        "complementary":  ["Dark Blue", "White", "Black"],
        "analogous":      ["Red", "Yellow", "Dark Yellow"],
        "clashing":       ["Pink", "Green"],
        "harmony_note":   "Orange needs cool-toned neutrals to balance it",
    },
    "Gold": {
        "complementary":  ["Black", "Dark Blue", "White", "Dark Green"],
        "analogous":      ["Yellow", "Orange", "Beige"],
        "clashing":       ["Silver", "Light Grey"],
        "harmony_note":   "Gold elevates navy and black — classic luxury pairing",
    },
    "Turquoise": {
        "complementary":  ["White", "Orange", "Beige"],
        "analogous":      ["Light Blue", "Blue", "Light Green"],
        "clashing":       ["Pink", "Purple"],
        "harmony_note":   "Turquoise pairs with warm neutrals and earthy accents",
    },
    "Light Blue": {
        "complementary":  ["White", "Beige", "Off White", "Orange"],
        "analogous":      ["Blue", "Dark Blue", "Turquoise"],
        "clashing":       ["Green"],
        "harmony_note":   "Light blue is a versatile soft neutral — easy to pair",
    },
    "Silver": {
        "complementary":  ["Black", "Dark Blue", "White"],
        "analogous":      ["Grey", "Light Grey"],
        "clashing":       ["Gold", "Beige"],
        "harmony_note":   "Silver metallic is cool-toned — avoid warm gold clashes",
    },
}
