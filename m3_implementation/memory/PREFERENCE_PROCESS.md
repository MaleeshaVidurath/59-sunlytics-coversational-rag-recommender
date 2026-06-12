# User Preference Process — Sunlytics CRS
### m3_implementation/memory/

This document explains exactly how user preferences are **stored**, **updated**, **decayed**, and **used** to personalise recommendations. All details are taken directly from the source code.

---

## 1. What Is a Preference?

A preference is one (attribute, value) pair with a strength score.

**Example:** User likes Black colour → one `PreferenceEntry`

```
attribute_name  = "colour_group_name"
attribute_value = "Black"
sentiment       = 0.8       ← how much they like it (-1.0 to +1.0)
confidence      = 0.85      ← how sure we are (0.0 to 1.0)
source          = "explicit" ← user said it directly
mention_count   = 1         ← how many times mentioned
decay_weight    = 1.0       ← freshness multiplier (fades over time)
```

**Schema file:** [memory/models/schemas.py](models/schemas.py) — `PreferenceEntry` class

---

## 2. What Attributes Can Be Preferences?

Only attributes that map to columns in `sample_articles.csv`:

| attribute_name | category | Example value |
|----------------|----------|---------------|
| `colour_group_name` | colour | `"Black"`, `"White"`, `"Dark Blue"` |
| `perceived_colour_master_name` | colour | `"Blue"` |
| `product_type_name` | product_type | `"Dress"`, `"T-shirt"`, `"Blazer"` |
| `index_group_name` | index_group | `"Ladieswear"`, `"Menswear"` |
| `garment_group_name` | garment_group | `"Dresses Ladies"`, `"Knitwear"` |
| `section_name` | style | `"Divided"`, `"H&M+"` |
| `graphical_appearance_name` | style | `"Solid"`, `"Stripe"`, `"Check"` |
| `style` | style | `"casual"`, `"formal"` |
| `occasion` | occasion | `"work"`, `"beach"`, `"party"` |
| `material` | material | `"cotton"`, `"linen"` |

**Price fields (`price_max`, `price_min`) are NOT saved as preferences** — they are session filters only.

---

## 3. Where Are Preferences Stored?

**Two storage layers:**

| Layer | What | How long |
|-------|------|----------|
| **MongoDB** (`users` collection) | Full user document with all preferences — permanent | Forever |
| **Redis** cache | JSON copy of the user document | 60 minutes (then re-loaded from MongoDB) |

Inside MongoDB, each user document (`UserDocument`) has two separate preference lists:

- `attribute_preferences` → things the user **likes** (sentiment ≥ 0)
- `disliked_attributes` → things the user **dislikes** (sentiment < 0)

**Code file:** [memory/core/user_manager.py](core/user_manager.py)

---

## 4. When Are Preferences Updated?

Preferences are updated after each user turn, depending on the turn type (label):

| Turn Label | When Called | Sentiment Used | Source | Confidence |
|-----------|------------|---------------|--------|-----------|
| INITIAL_REQUEST | User's first request | 0.8 | explicit | 0.85 |
| REFINEMENT | User changes something | 0.75 | explicit | 0.80 |
| FEEDBACK positive | User says "I like it" | score from RoBERTa (~0.9) | implicit | 0.80 |
| FEEDBACK negative | User says "I don't like it" | score from RoBERTa (~−0.8) | implicit | 0.80 |
| CHITCHAT / others | Small talk | **not updated** | — | — |

**Note:** Price entities (`price_max`, `price_min`) are **never saved as preferences** even during INITIAL_REQUEST or REFINEMENT.

**Code file:** [memory/core/enrichment.py](core/enrichment.py) — calls `update_preferences_from_entities()` per label

---

## 5. How Does a Preference Get Created or Updated?

**Code:** `UserManager.update_preferences_from_entities()` in [memory/core/user_manager.py](core/user_manager.py)

### For a LIKED attribute (sentiment ≥ 0):

**Step 1:** Check if this (attribute_name, attribute_value) pair already exists in `attribute_preferences`

**If it already exists → reinforce it:**
```
new_sentiment  = (existing_sentiment × 0.7) + (new_sentiment × 0.3)
new_confidence = min((existing_confidence × 0.7) + (new_confidence × 0.3), 1.0)
decay_weight   = reset to 1.0   ← refreshed, so it won't fade
mention_count  += 1
last_mentioned_at = now
```

**If it is new → create it:**
```
New PreferenceEntry added to attribute_preferences array in MongoDB
```

**Step 2 (Conflict resolution):** If `source="explicit"` and `sentiment > 0.5`, reduce any other value for the same attribute:
```
Example: User preferred "Black", now says "show me WHITE ones"
→ Black sentiment = Black_sentiment × 0.4   (reduced but NOT deleted)
→ Black decay_weight = 0.5
→ White added as new preference with sentiment=0.8
```

### For a DISLIKED attribute (sentiment < 0):

**If this dislike already exists → strengthen it:**
```
new_sentiment = max((existing_sentiment × 0.7) + (new_sentiment × 0.3), −1.0)
mention_count += 1
last_mentioned_at = now
```

**If it is new → create it:**
```
New PreferenceEntry added to disliked_attributes array in MongoDB
```

### After any update:
The Redis cache for this user is **deleted** so the next read gets fresh data from MongoDB.

---

## 6. How Does Feedback Update Preferences?

**Code:** `_enrich_feedback()` in [memory/core/enrichment.py](core/enrichment.py)

When user says "I like them" or "I don't like them":

**Step 1:** Classify sentiment using Twitter-RoBERTa:
```python
sentiment_label, sentiment_score = classify_feedback(current_message)
# sentiment_label: "positive" / "neutral" / "negative"
# sentiment_score: float in [-1.0, +1.0]
```

**Step 2:** Extract attributes from the currently shown item (`item_a`):
```python
item_entities = {
    "colour_group_name": item_a.colour_group_name,
    "product_type_name": item_a.product_type_name,
    "index_group_name":  item_a.index_group_name,    ← if available
    "garment_group_name": item_a.garment_group_name, ← if available
}
```

**Step 3:** Update preferences using those item attributes + sentiment score:
```python
update_preferences_from_entities(
    entities=item_entities,
    sentiment=sentiment_score,   ← e.g. +0.876 (positive) or -0.954 (negative)
    source="implicit",
    confidence=0.80
)
```

**Step 4:** Update session state:
- **Positive:** Add `item_a.article_id` to `accepted_items`. If score > 0.7, update `purchase_summary`.
- **Negative:** Add ALL currently shown items to `rejected_items`. Mark recommendation as `"rejected"`.
- **Neutral:** No session state changes.

**Step 5 — Negative only:** Trigger a NEW catalog search excluding all rejected items.

---

## 7. Time Decay — Preferences Fade Over Time

**Code:** `UserManager._apply_time_decay()` in [memory/core/user_manager.py](core/user_manager.py)

Old preferences gradually lose influence using exponential decay:

```
decay_weight = e^(−0.0077 × days_since_last_mentioned)
```

| Days since mentioned | decay_weight |
|---------------------|:------------:|
| 0 days (today) | 1.00 |
| 30 days | 0.79 |
| 90 days | 0.50 |
| 180 days | 0.25 |
| Very old | 0.10 (minimum floor — never deleted) |

**When decay runs:**
- Only when loading from MongoDB (cache miss)
- NOT on every request (Redis cache serves for 60 minutes before decay recalculates)

**Why a floor of 0.10?**
Preferences are never fully deleted — a user who hasn't mentioned something in 2 years still has a weak signal, which is better than nothing.

---

## 8. How Preferences Are Read for Retrieval

**Code:** `UserManager.get_preference_summary()` in [memory/core/user_manager.py](core/user_manager.py)

Before every catalog search, preferences are converted into a structured summary:

```
For each liked preference:
    weight = sentiment × confidence × decay_weight
    → sorted by weight descending (strongest first)
    → only preferences with weight > 0.3 are passed to retrieval

Hard constraints (mandatory filters):
    → only if sentiment > 0.85 AND confidence > 0.85 AND source = "explicit"

Disliked values:
    → grouped by attribute: {"colour_group_name": ["Orange", "Pink"]}
```

**Example output:**
```python
{
    "liked_attributes": [
        {"attribute_name": "colour_group_name", "attribute_value": "Black", "weight": 0.680},
        {"attribute_name": "product_type_name", "attribute_value": "Dress",  "weight": 0.612},
    ],
    "disliked_values": {
        "colour_group_name": ["Orange"]
    },
    "hard_constraints": {
        "product_type_name": "Dress"    ← if sentiment > 0.85 and confidence > 0.85
    },
    "style_profile": {...},
    "purchase_summary": {...},
    "top_product_types": ["Dress", "Blouse"],
    "top_colours": ["Black", "White"]
}
```

---

## 9. How Preferences Become Search Instructions

**Code:** `_enrich_initial_request()` and `_enrich_refinement()` in [memory/core/enrichment.py](core/enrichment.py)

The preference summary is converted into a retrieval payload sent to the RAG system:

```python
payload = {
    # HARD FILTERS — only show items matching these exactly
    "filters": {
        "product_type_name": "Dress",     ← from current message entities
        "colour_group_name": "Black",     ← from hard_constraints
        "price_max": 40.0                 ← from current message
    },

    # PREFERENCE BOOSTS — rank items higher if they match these
    "preference_boosts": [
        {"attribute": "colour_group_name", "value": "Black", "weight": 0.680},
        {"attribute": "product_type_name", "value": "Dress",  "weight": 0.612},
    ],

    # PURCHASE HISTORY HINTS — from historical transaction data
    "purchase_history_hints": {
        "top_colours":          ["Black", "White", "Navy"],
        "top_product_types":    ["Dress", "Blouse"],
        "inferred_gender":      "female",
        "budget_tier":          "mid",
        "preferred_price_range": [15.0, 45.0],
        "dominant_colour":      "Black",
        "dominant_type":        "Dress"
    },

    # PENALTIES — rank items lower if they match these
    "penalties": {
        "colour_group_name": ["Orange"]
    },

    # EXCLUSIONS — never show these items again (rejected this session)
    "exclude_ids": ["article_123", "article_456"]
}
```

---

## 10. Full Flow — One User Turn End to End

Using **"I want a black dress under £40"** as an example:

```
User: "I want a black dress under £40"
         ↓
[1] Intent Classification (DistilBERT)
    → label = "INITIAL_REQUEST"
         ↓
[2] Entity Extraction (entity_extractor.py)
    → colour_group_name = "Black"
    → product_type_name = "Dress"
    → price_max = 40.0
         ↓
[3] Enrichment — _enrich_initial_request()
    │
    ├── Load preference summary from user profile
    │   → liked: [Black(0.68), Dress(0.61)]
    │   → disliked: [Orange]
    │   → hard_constraints: {product_type_name: "Dress"}
    │
    ├── Save new hard constraints to session:
    │   → {colour_group_name: "Black", product_type_name: "Dress", price_max: 40.0}
    │
    ├── Update long-term preferences (save to MongoDB):
    │   → Black: sentiment=0.8, confidence=0.85, source="explicit"
    │   → Dress: sentiment=0.8, confidence=0.85, source="explicit"
    │   → (price_max skipped — never saved as preference)
    │
    └── Build retrieval payload:
        → filters: {colour: Black, type: Dress, price_max: 40.0}
        → preference_boosts: [{Black, 0.68}, {Dress, 0.61}]
        → purchase_history_hints: {top_colours: [Black, White], budget: mid}
        → penalties: {colour: [Orange]}
        → exclude_ids: []
         ↓
[4] RAG searches catalog using this payload → returns items
         ↓
[5] Items shown to user, stored in session as currently_discussing
```

---

## 11. Two Types of Memory

| | Session Memory | Long-term Memory |
|--|---------------|-----------------|
| **What** | What user wants RIGHT NOW | What user has liked across ALL sessions |
| **Stored in** | `DialogueState` inside `SessionDocument` | `UserDocument.attribute_preferences` |
| **Lives** | This conversation only (30 min timeout) | Permanently in MongoDB |
| **Content** | hard_constraints, soft_constraints, rejected_items, accepted_items, currently_discussing | PreferenceEntry list with sentiment + decay |
| **Code** | `session_manager.py`, `schemas.py:DialogueState` | `user_manager.py`, `schemas.py:UserDocument` |

---

## 12. Key Design Rules

| Rule | Detail |
|------|--------|
| **Never delete preferences** | Old preferences are reduced (× 0.4 or decay), not removed. User may return to old preferences. |
| **Hard constraints need high confidence** | Only `sentiment > 0.85` AND `confidence > 0.85` AND `source = "explicit"` become mandatory filters |
| **Only boosts with weight > 0.3** | Weak preferences (very old or low-confidence) are ignored for retrieval to avoid noise |
| **Merge, not replace** | New sentiment = (old × 0.7) + (new × 0.3) — prevents one strong signal from dominating |
| **Feedback uses item attributes** | When user says "I don't like them", the system reads the shown item's colour and type and saves those as dislikes — not the words the user said |
| **Price is never a preference** | `price_max` and `price_min` are session-only filters. Budget tier comes from purchase history only. |

---

## 13. Files Involved

| File | Role |
|------|------|
| [memory/models/schemas.py](models/schemas.py) | `PreferenceEntry`, `UserDocument`, `DialogueState` data structures |
| [memory/core/user_manager.py](core/user_manager.py) | Create/load user, update preferences, apply decay, build preference summary |
| [memory/core/enrichment.py](core/enrichment.py) | Per-label enrichment, builds retrieval payload from preferences |
| [memory/core/feedback_sentiment_classifier.py](core/feedback_sentiment_classifier.py) | Twitter-RoBERTa classifier — converts feedback text to sentiment score |
| [memory/core/session_manager.py](core/session_manager.py) | Manage `DialogueState` (session memory) |
| [memory/core/entity_extractor.py](core/entity_extractor.py) | Extract fashion attributes from user messages |
| [memory/core/pipeline.py](core/pipeline.py) | Orchestrates all steps per turn |
