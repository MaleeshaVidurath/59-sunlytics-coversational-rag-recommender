# Entity Extractor — Full Process Documentation
### Sunlytics CRS — M3 Memory Module

`m3_implementation/memory/core/entity_extractor.py`

The entity extractor has two independent responsibilities that run at different
points in the pipeline: **FashionGuard** (relevance classification) and
**Entity Extraction** (attribute extraction for retrieval filters).

---

## Overview — Two Jobs in One File

| Job | Function | Called at | Purpose |
|-----|----------|-----------|---------|
| FashionGuard | `is_fashion_relevant_async()` | Pipeline Step 3b | Block off-topic messages before classification |
| Entity Extraction | `extract_entities()` | Pipeline Step 5 | Extract filter attributes for Qdrant catalog search |

Both jobs share the same `all-MiniLM-L6-v2` singleton — the model is loaded once
and reused across both jobs to avoid loading it twice into memory.

---

## Part 1 — FashionGuard

### Purpose

FashionGuard runs **before DistilBERT classification**. If the user's message is
not about fashion, shopping, or clothing, the pipeline skips all downstream steps
(classification, retrieval, response generation) and returns a refusal message
immediately. This prevents the system from processing irrelevant queries.

What it blocks is off-topic *subject matter*, not everything that lacks fashion
vocabulary. Conversational turns that belong in a shopping session — follow-ups
that reference items already shown (Stage 1) and greetings (Stage 1b) — pass
through and are handled as `CHITCHAT`.

### Staged Architecture

```
User message
     │
     ├─ Stage 0 (instant) — Exact word blocklist
     │     Whole-message word set intersected with 60+ known off-topic words
     │     (pizza, football, doctor, car, coffee, etc.)
     │     → REJECT immediately if match found and no continuation phrase
     │
     ├─ Stage 1 (instant) — Continuation phrase bypass
     │     Only fires when conversation history exists
     │     "thanks", "show me more", "which one", "yes please", "first one"
     │     → PASS immediately (these reference items already in context)
     │
     ├─ Stage 1b (instant) — Greeting bypass
     │     Needs NO history — a greeting is usually the first message sent
     │     Every word must be a greeting: exact match against a small token
     │     set, or repeated letters collapsed and matched against stems
     │     ("hyy" → "hy", "heyyy" → "hey", "helloo" → "helo")
     │     → PASS (routed to CHITCHAT, which produces a warm welcome)
     │
     ├─ Stage 2 (instant) — Keyword gate
     │     2a. Allowlist substring match:
     │         "dress", "jacket", "outfit", "h&m", "t-shirt", "sneakers" → PASS
     │     2b. Regex blocklist (word-boundary patterns):
     │         r'\bweather\b', r'\bfood\b', r'\bcooking\b' → REJECT
     │     2c. Catalog type terms (loaded from sample_articles.csv at startup):
     │         Every token from product_type_name column → PASS
     │     2d. Catalog product name tokens (significant words ≥ 5 chars):
     │         Unique identifiers from prod_name column → PASS (with history)
     │     2e. Full product names (all 21k names from prod_name):
     │         Substring match against the message → PASS
     │
     ├─ Stage 3 (3–5ms) — Dual-pool mean-of-top-3 semantic scoring
     │     MiniLM encodes: [message] + 16 fashion anchors + 12 off-topic anchors
     │     fashion_score  = mean cosine of top-3 fashion anchor similarities
     │     offtopic_score = mean cosine of top-3 off-topic anchor similarities
     │
     │     Decision rules:
     │       f_score ≥ 0.28                    → PASS (confident fashion)
     │       o_score > f_score + 0.10          → REJECT (off-topic leads clearly)
     │       f_score < 0.18                    → REJECT (below minimum fashion floor)
     │       margin < 0.08 (ambiguous zone)    → escalate to Stage 4
     │
     └─ Stage 4 (~150ms) — Groq LLM arbitration
           Only fires for genuinely ambiguous messages (not caught by Stages 0–3)
           Prompt: "Is this message about fashion, clothing, style or H&M shopping?"
           Returns: {"is_fashion": true/false, "reason": "one sentence"}
           On any error → defaults to PASS (prefer false negatives over false positives)
```

### Why Greetings Need Their Own Stage

A greeting carries no fashion vocabulary, so it scores near zero on the
semantic stage and was being answered with the off-topic refusal:

| message | fashion score | outcome before Stage 1b |
|---------|---------------|-------------------------|
| `hi`    | 0.08          | REJECT — below the 0.18 floor |
| `hyy`   | 0.17          | REJECT — below the 0.18 floor |

Stage 1 could not catch these either: it requires conversation history, and a
greeting is normally the very first message of a session, when there is none.

Lowering the semantic floor was not an option — it is what keeps genuinely
off-topic single words out. A greeting is not weak evidence of fashion intent;
it is a different kind of message, so it gets its own gate.

Two design details keep the gate tight:

- **Every** word must be a greeting, not merely one of them. So `hi` passes,
  while `hi, what is a good pasta recipe` falls through to the stages below and
  is still rejected. Stage 1b also sits after Stage 0, so blocklisted words win
  regardless.
- Typed greetings are full of doubled letters, and enumerating every spelling
  is a losing game. Runs of a repeated letter are collapsed to one and matched
  against stems instead. Words like `good` and `all` stay on the exact-match
  list, since collapsing would mangle them into `god` and `al`.

Once a greeting passes, nothing else is needed: `_pre_classify_short_message`
already labels it `CHITCHAT`, and the response generator already holds a
warm-welcome prompt. Both paths existed; the guard was simply rejecting the
message before either could run.

### Known Issue — Stage 2e Matches Substrings

Stage 2e checks whether any of ~21,600 catalog product names appears anywhere
in the message, as a plain substring with no word-boundary check. The catalog
contains names as short as one character, so this matches almost any text:

```
[FashionGuard] Stage2e-product-name: matched 's' in 'i need 3 shorts'
[FashionGuard] Stage2e-product-name: matched 'mo' in 'good morning'
```

Usually harmless, because an earlier stage has already decided. But when the
message reaches Stage 2e undecided, it can pass off-topic text — `shhh` is
admitted on the strength of the product name `s`.

Fix when addressed: require a word-boundary match and ignore catalog names
shorter than three characters.

### Why Mean-of-Top-3 Instead of Max

Earlier versions used the single highest cosine score against the anchor pool.
A single anchor is brittle — one irrelevant anchor can mislead the score.
Mean-of-top-3 averages the three best-matching anchors from each pool,
smoothing outliers and giving a more reliable signal for borderline inputs.

### Why Not Train a Custom Classifier

| Alternative | Problem |
|-------------|---------|
| Train a binary fashion/off-topic classifier | No labelled data; high maintenance; overkill for a guard layer |
| NLI cross-encoder only (DeBERTa) | 50–100× slower than bi-encoder; unacceptable for every turn |
| Keyword rules only | Cannot handle synonyms, informal phrasing, or novel off-topic inputs |
| LLM only | 150ms+ per call; too slow when 90% of messages are obviously fashion-related |

The cascade gives near-instant decisions (0ms) for easy cases and reserves the
expensive LLM call only for the genuinely ambiguous ~5%.

> **Implementation note.** `is_fashion_relevant_async` collects the stage
> results into a tuple before iterating, so Python evaluates *every* stage on
> every message even after one has already decided. The first non-`None` result
> still wins, so the verdict is correct — but the cheap stages are not actually
> short-circuiting, which is why decided messages still log lines from later
> stages. Stage 4 (Groq) is called after the loop and is not affected, so no
> API calls are wasted.

### Pipeline Outcome When Not Relevant

```python
# pipeline.py Step 3b
_is_relevant, _relevance_score, _guard_stage = (
    await is_fashion_relevant_async(message, history=history)
)

if not _is_relevant:
    # Store turn as CHITCHAT, return refusal — no DistilBERT, no Qdrant, no Groq
    return {
        "label":           "CHITCHAT",
        "retrieval_strategy": "NO",
        "memory_context": {
            "not_relevant": True,
            "refusal_message": "I can only help with fashion and clothing recommendations..."
        }
    }
```

---

## Part 2 — Entity Extraction

### Purpose

After DistilBERT classifies the message (Pipeline Step 4), entity extraction
converts the raw user message into structured filter attributes for Qdrant.
These become the **pre-filters** that Qdrant applies before cosine similarity
scoring — only articles matching the extracted colour, type, and price are scored.

### Label Gate — Extract Only for Search Turns

```python
EXTRACTION_LABELS = {"INITIAL_REQUEST", "REFINEMENT"}

async def extract_entities(message, label=None):
    if label is not None and label not in EXTRACTION_LABELS:
        return {}   # no extraction needed
```

All other labels (ATTRIBUTE_QUESTION, COMPARISON, FEEDBACK, SELECTION_REFERENCE,
CHITCHAT) work with items already in the dialogue state — no new catalog search,
no extraction needed.

### Three-Tier Extraction Architecture

```
User message (INITIAL_REQUEST or REFINEMENT only)
     │
     ├─ Tier 1: Keyword + regex (instant, always runs)
     │     Lookup maps for colour, product type, pattern, occasion, style, index group
     │     Regex patterns for price extraction (£, $, €, number words)
     │     e.g. "black dress under £50"
     │       → colour_group_name: "Black"
     │       → product_type_name: "Dress"
     │       → price_max: 50.0
     │
     ├─ Tier 2: MiniLM vector similarity (fills gaps from Tier 1)
     │     Only runs for colour/product_type if Tier 1 did not find them
     │     Encodes message + anchor descriptions, picks best cosine match
     │     Handles synonyms Tier 1 misses:
     │       "midnight blue" → Dark Blue (score > 0.35)
     │       "activewear"    → Leggings/Tights (score > 0.45)
     │
     ├─ Tier 3: Groq LLM (always runs for complex NL)
     │     Handles context that keywords and vectors cannot:
     │       "for my sister's graduation" → {occasion: formal, style: elegant}
     │       "something earthy and warm toned" → {colour: Yellowish Brown}
     │     Model: llama-3.1-8b-instant, temperature=0, response_format=json_object
     │
     └─ Merge + Validate
           Base: Tier 1
           Fill: Tier 2 (only adds colour/product_type if missing)
           Override: Tier 3 overrides all except price_max/price_min
                     (regex is more reliable for numbers than LLM)
           Final: _validate_entities() — drops any value not in valid sets
```

### Tier 1 — Keyword + Regex Detail

Seven lookup maps cover all filterable fields:

| Map | Example entries | Target field |
|-----|----------------|--------------|
| `_COLOUR_MAP` | `"navy" → "Dark Blue"`, `"burgundy" → "Dark Red"` | `colour_group_name` |
| `_PRODUCT_MAP` | `"activewear" → "Leggings/Tights"`, `"trench coat" → "Jacket"` | `product_type_name` |
| `_GRAPHICAL_MAP` | `"floral" → "Front print"`, `"plaid" → "Check"` | `graphical_appearance_name` |
| `_INDEX_GROUP_MAP` | `"ladies" → "Ladieswear"`, `"teen" → "Divided"` | `index_group_name` |
| `_OCCASION_MAP` | `"graduation" → "formal"`, `"yoga" → "gym"` | `occasion` |
| `_STYLE_MAP` | `"boho" → "relaxed"`, `"streetwear" → "casual"` | `style` |
| `_PRICE_MAX_PATTERNS` | `r'under\s+[£$€]?\s*(\d+)'` + number words | `price_max` / `price_min` |

Multi-word patterns (e.g. `"vest top"`, `"cross-body bag"`) are always checked
before single-word patterns to prevent partial matches.

### Tier 2 — Vector Similarity Detail

Two anchor dictionaries map valid database values to rich descriptive strings:

```python
_COLOUR_VECTOR_ANCHORS = {
    "Dark Blue": "Dark blue navy midnight indigo cobalt deep blue",
    "Off White":  "Off white ivory cream ecru natural white",
    ...
}
_PRODUCT_VECTOR_ANCHORS = {
    "Leggings/Tights": "Leggings tights activewear gym sports workout",
    "Dress":            "Dress gown frock midi maxi mini wrap shift",
    ...
}
```

For each missing field, the message is encoded alongside all anchor descriptions.
The best cosine match is accepted only above a confidence threshold
(colour: 0.35, product: 0.45) — below threshold, nothing is added.

A `_CLOTHING_HINT_WORDS` gate ensures product type matching only fires when
the message is actually about clothing (`"wear"`, `"outfit"`, `"show me"`,
`"need a"` etc.) — preventing false product matches in unrelated messages.

### Tier 3 — LLM Detail

The LLM receives a strict prompt with:
- Exact allowed values for every field
- Mapping rules for ambiguous inputs (e.g. `pastel → Light Pink`, `graduation → formal`)
- Explicit instruction to return `{}` for greetings, attribute questions, comparisons, and feedback
- `temperature=0` for deterministic output
- `response_format: json_object` to prevent free text

Tier 3 always runs (for INITIAL_REQUEST / REFINEMENT) but its price output is
discarded if Tier 1 already found a price. Regex extraction of explicit numbers
(`£50`, `under thirty`) is more reliable than LLM interpretation.

### Merge and Validate

```
merged = Tier 1 (base)
       + Tier 2 values where Tier 1 had no colour/product_type
       + Tier 3 values (overrides Tier 1/2) EXCEPT price_max/price_min

_validate_entities(merged)
    → product_type_name must be in VALID_PRODUCT_TYPES (76 values from CSV)
    → colour_group_name must be in VALID_COLOURS (40 values from CSV)
    → graphical_appearance_name must be in VALID_GRAPHICAL (20 values)
    → index_group_name must be in VALID_INDEX_GROUPS (5 values)
    → garment_group_name must be in VALID_GARMENT_GROUPS (17 values)
    → price_max / price_min must be positive float
    → occasion must be in VALID_OCCASIONS (11 values)
    → style must be in VALID_STYLES (9 values)
    → anything else is silently dropped
```

This final validation gate **prevents LLM hallucination from reaching the
retrieval system**. If the LLM returns `"colour_group_name": "Midnight Blue"`
(not a valid CSV value), it is dropped — Qdrant only receives values that
actually exist in the database.

---

## Extraction Examples

| User message | Tier 1 | Tier 2 | Tier 3 | Final |
|---|---|---|---|---|
| `"black dress under £50"` | Black, Dress, price_max=50 | — | — | `{colour: Black, product: Dress, price_max: 50}` |
| `"something floral in a pastel shade"` | Front print | — | Light Pink | `{pattern: Front print, colour: Light Pink}` |
| `"for my sister's graduation"` | — | — | formal, elegant | `{occasion: formal, style: elegant}` |
| `"navy dress"` | Dress | — | Dark Blue | `{colour: Dark Blue, product: Dress}` |
| `"activewear under thirty"` | Leggings/Tights, price_max=30 | — | — | `{product: Leggings/Tights, price_max: 30}` |
| `"Thanks!"` (FEEDBACK label) | — | — | — | `{}` (label gate skips) |
| `"What material is it?"` (ATTRIBUTE_QUESTION) | — | — | — | `{}` (label gate skips) |

---

## How Entities Flow into the Pipeline

```
Pipeline Step 3b
    is_fashion_relevant_async(message, history)
    → not relevant → refusal response, pipeline stops
    → relevant → continue

Pipeline Step 4
    DistilBERT classifier → label (e.g. INITIAL_REQUEST)

Pipeline Step 5
    extract_entities(message, label=label)
    → {} for non-search labels
    → {colour, product, price, occasion, style, ...} for INITIAL_REQUEST/REFINEMENT

Pipeline Step 6
    entities stored with the user turn in PostgreSQL turn history

Pipeline Step 7 (RAG Pipeline)
    entities passed as pre-filters to Qdrant semantic_search()
    Qdrant filters: colour_group_name="Black" AND product_type_name="Dress"
    THEN scores remaining articles by cosine similarity to query embedding
    → only articles matching all filter values are ranked and returned
```

---

## Why This Three-Tier Design

| Tier | Why needed |
|------|-----------|
| Tier 1 (keyword) | Handles the majority of cases (60–70%) instantly with no model inference overhead |
| Tier 2 (vector) | Catches synonyms and informal phrasing that exact keyword maps miss — user vocabulary is unpredictable |
| Tier 3 (LLM) | Handles complex contextual NL that neither keywords nor embedding similarity can resolve |

A single tier would fail: keyword-only misses synonyms, vector-only misses
price and pattern, LLM-only adds 500ms+ to every single turn for cases that
a simple regex could solve in microseconds.

The cascade design means fast simple cases resolve in Tier 1 and only genuinely
complex cases escalate to the expensive Tier 3 LLM call.

---

## Valid Value Sets

All valid values are loaded from `sample_articles.csv` at module import time:

| Field | Count | Source |
|-------|-------|--------|
| `product_type_name` | 76 types | CSV unique values |
| `colour_group_name` | 40 colours | CSV unique values |
| `graphical_appearance_name` | 20 patterns | CSV unique values |
| `index_group_name` | 5 groups | CSV unique values |
| `garment_group_name` | 17 groups | CSV unique values |
| `occasion` | 11 values | Curated list |
| `style` | 9 values | Curated list |

---

## File Location

| Component | Location |
|-----------|----------|
| Entity extractor | `m3_implementation/memory/core/entity_extractor.py` |
| Pipeline integration | `m3_implementation/memory/core/pipeline.py` lines 44–45, 217, 489 |
| Catalog source data | `shared/main_data_set/sample_articles.csv` |
