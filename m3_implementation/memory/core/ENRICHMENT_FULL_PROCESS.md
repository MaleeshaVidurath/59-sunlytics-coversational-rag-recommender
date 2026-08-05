# Enrichment Layer — Full Process Documentation
### Sunlytics CRS — M3 Memory Module

`m3_implementation/memory/core/enrichment.py`

---

## 1. What Is the Enrichment Layer

The Enrichment Layer is the **bridge between DistilBERT classification and the RAG pipeline**.

DistilBERT tells the system *what the user wants* (the intent label).
Enrichment tells the system *exactly what to retrieve and from where*, based on what
is already stored in session memory.

For every user turn, enrichment:
1. Reads the current session state (dialogue_state, hard/soft constraints, accepted/rejected items)
2. Reads long-term user preferences (liked attributes, disliked values, purchase history)
3. Resolves which specific items the user is referring to
4. Builds a fully standardised retrieval_input that the RAG pipeline consumes directly
5. Applies side effects (updates dialogue_state, preferences, recommendation outcomes)

---

## 2. Standardised Output — Always the Same Shape

Every label returns the same envelope regardless of what happens internally:

```python
{
    "label":              str,          # DistilBERT label
    "retrieval_strategy": str,          # "FULL" | "PARTIAL" | "NO"
    "retrieval_input": {                # None when strategy is NO
        "action":             str,      # what the RAG assembler should do
        "retrieval_strategy": str,
        "user_message":       str,      # original user message text
        "items_in_context":   dict,     # item_a, item_b ... from dialogue_state
        "exclude_ids":        list,     # rejected article_ids (never show again)
        "payload":            dict,     # action-specific data (see taxonomy below)
    } | None,
    "memory_context":     dict,         # dialogue state, preferences, style profile
    "side_effects":       list[str],    # human-readable log of what state was updated
}
```

The RAG pipeline consumes `retrieval_input` directly.
The ResponseGenerator uses `memory_context` to build LLM prompts.
State updates are already applied inside enrichment before it returns —
`side_effects` is a log only.

---

## 3. Action Taxonomy — One Action Per Label

| Label | Action | Retrieval Strategy |
|-------|--------|--------------------|
| INITIAL_REQUEST | `catalog_search` | FULL — Qdrant ANN vector search |
| REFINEMENT | `catalog_search` | FULL — Qdrant ANN vector search |
| FEEDBACK (negative + items) | `catalog_search` | FULL — with exclusions |
| ATTRIBUTE_QUESTION | `item_attribute_lookup` | PARTIAL — item already in memory |
| EXPLANATION_WHY | `explanation_generate` | PARTIAL — item already in memory |
| COMPARISON | `item_compare` | PARTIAL — both items already in memory |
| SELECTION_REFERENCE | `item_detail_lookup` | PARTIAL — item already in memory |
| FEEDBACK (positive / neutral) | None | NO |
| CHITCHAT | None | NO |

---

## 4. Base Memory Context (Always Included)

Every label builds a `_base_memory_context` that is always present in the output:

```python
{
    "dialogue_state": {
        "hard_constraints":  {...},   # colour, product_type, price — DB filter conditions
        "soft_constraints":  {...},   # style, occasion — preference hints
        "rejected_items":    [...],   # article_ids never shown again
        "accepted_items":    [...],   # article_ids user liked
        "intent_summary":    str,     # LLM-written summary of session intent
    },
    "long_term_preferences":  [...],  # liked attribute-value pairs with weights
    "style_profile":          {...},  # aggregated style signals
    "preference_summary":     {...},  # full preference summary from user profile
    "existing_explanation":   None,   # populated for EXPLANATION_WHY only
}
```

For PARTIAL labels that only need the item context (ATTRIBUTE_QUESTION, SELECTION_REFERENCE),
`include_preferences=False` is passed to avoid the preference DB query — not needed
for item lookups.

---

## 5. Per-Label Enrichment — Full Detail

### 5.1 INITIAL_REQUEST → `catalog_search`

Builds a full catalog search payload with filters, preference boosts, and penalties.

**Steps:**

1. Extract hard constraints from entities (colour_group_name, product_type_name,
   price_max, price_min etc.) — `style`, `occasion`, and `quantity` are excluded
   from DB filters (they are soft constraints or metadata)

2. Update `dialogue_state.hard_constraints` in Redis/MongoDB with the new constraints

3. Update long-term user preferences from entities (sentiment=0.8, confidence=0.85,
   source="explicit") — colour and product type preferences recorded immediately

4. Load full preference profile from MongoDB for boosting and penalties

**Payload sent to RAG:**
```python
payload = {
    "filters":        {colour_group_name, product_type_name, price_max, ...},  # hard WHERE
    "soft_constraints": {style, occasion},    # used in LLM prompt, not as DB filters
    "preference_boosts": [                    # rank items matching these attributes higher
        {"attribute": "colour_group_name", "value": "Black", "weight": 0.85},
        ...
    ],
    "purchase_history_hints": {              # from H&M transaction history loaded at login
        "top_colours":           [...],
        "top_product_types":     [...],
        "inferred_gender":       str,
        "budget_tier":           str,
        "preferred_price_range": str,
        "dominant_colour":       str,
        "dominant_type":         str,
    },
    "penalties":  {attribute: value},        # disliked attributes ranked lower
    "quantity":   int | None,                # how many items user requested
}
```

---

### 5.2 REFINEMENT → `catalog_search`

Same action as INITIAL_REQUEST but merges new constraints **on top of** the
session's existing hard_constraints rather than replacing them.

**Key difference — `_resolve_cheaper_price()`:**

When the user says *"show me something cheaper"*, the LLM entity extractor
guesses a price_max from general language knowledge and gets it wrong.
`_resolve_cheaper_price()` overrides this:

```
"show me something cheaper"
     ↓
Is any cheaper-keyword in message?
     ↓
Read actual prices of items in currently_discussing
     ↓
Is a specific item named? → use that item's price
Otherwise              → use minimum price among all context items
     ↓
price_max = reference_price − 0.01 (strictly cheaper)
```

Example: items are £34.99 and £29.99 → `price_max = 28.99`

The memory_context includes `previous_constraints` and `new_changes` so the
LLM knows exactly what changed between the old and new search.

**Side effects:** merged hard_constraints saved to dialogue_state; preference
profile updated from refinement entities (sentiment=0.75, confidence=0.80).

---

### 5.3 ATTRIBUTE_QUESTION → `item_attribute_lookup`

Determines which item and which attribute the user is asking about.

**Guard — no items in context:**
If `currently_discussing` is empty, the user is asking about a property before
any recommendations have been made. Enrichment escalates to FULL `catalog_search`
and sets `memory_context.needs_clarification = True`.

**Item resolution:**
`_resolve_item_reference_checked()` — returns the target item AND a flag
`is_default=True` when no real match was found in the message (fell back to item_a).

If `is_default=True` → search full session history in MongoDB for a better match.
If found in history → use that item with `use_historical_items=True`.
If not found → escalate to FULL retrieval.

**Attribute topic detection — hybrid keyword + vector:**

```
Step 1: Keyword matching (instant)
    "material", "fabric", "made of", "cotton" → material_and_care
    "size", "fit", "slim", "oversized"         → sizing_and_fit
    "pocket", "pockets"                        → pockets
    "price", "how much", "cost"                → price
    ... etc.

Step 2 (fallback): MiniLM vector similarity
    Compare message against 8 anchor descriptions
    e.g. "what is it constructed from"
    → matches "What material or fabric is this item made from and how do I care for it"
    → topic = material_and_care
```

**Available topics:** `material_and_care`, `colour`, `sizing_and_fit`, `pockets`,
`design_details`, `price`, `availability`, `general_details`

**Payload:** `{article_id, attribute_topic}`

---

### 5.4 EXPLANATION_WHY → `explanation_generate`

Determines which specific item to explain and retrieves any prior explanation
to prevent contradicting claims.

**Item resolution:**
`_resolve_item_reference()` returns a target, but for EXPLANATION_WHY the resolution
is checked more strictly — if no ordinal/price reference and no name word scoring
finds a match, `target_item = None` (explain all items), not the default item_a.
Session history fallback applies if no specific item identified in current context.

**Prior claims retrieval:**
```python
expl_doc = await db.explanations.find_one(
    {"session_id": session_id, "article_id": target_item.article_id},
    sort=[("created_at", -1)]
)
```
The `prior_claims` from any stored explanation are passed in the payload so the
ResponseGenerator cannot generate explanations that contradict what was said
in an earlier turn about the same item.

**Payload:** `{article_id, context_article, all_item_ids (if explain-all), prior_claims, matched_prefs}`

---

### 5.5 COMPARISON → `item_compare`

Identifies which two items to compare and on what dimension.

**Comparison dimension detection — hybrid keyword + vector:**

```
Keyword first:
    "cheaper", "price", "cost"    → price
    "quality", "better", "durable" → quality
    "casual", "formal", "occasion" → style_and_occasion
    "material", "fabric", "soft"   → material
    "fit", "size", "slim"          → fit
    "compare", "which", "overall"  → overall

MiniLM fallback (for paraphrases):
    "which one gives me more for my money?" → price
    "which is more versatile to style?"     → style_and_occasion
```

**Item pair resolution — priority order:**
```
1. Ordinal:  "option 3 and option 4" → item_list[2], item_list[3]
2. Name scoring:
     Full name match = +100 points
     Per name-word overlap = +n points
     Colour match = +50 points
     Price match = +50 points
   → top 2 scoring items
3. One identified + permissive second search in remaining items
4. Generic fallback: item_a vs item_b from currently_discussing
```

If current context cannot identify both items, queries full session history
(MongoDB) for 2 matching items.

When more than 2 items are compared, `article_ids_list` and `context_items_list`
are included in the payload so the RAG can show a multi-item comparison table.

**Payload:** `{article_id_a, article_id_b, context_article_a, context_article_b, comparison_dimension, preference_weights}`

---

### 5.6 SELECTION_REFERENCE → `item_detail_lookup`

Resolves which item the user selected and promotes it to `item_a` position.

**Guard — no items in context:**
Returns a CHITCHAT response with `needs_clarification=True` — user referenced
an item before any recommendations were made.

**Item resolution — two-method combined approach:**

```
_score_items_by_name(message, all_ctx_items)
    → name/colour/price word scoring → ranked list

_resolve_item_reference(message, *all_ctx_items)
    → ordinal → price match → colour match → name match → default

Decision priority:
    1. Both methods agree     → high confidence, use the agreed item
    2. Ordinal/price signal   → trust _resolve_item_reference
    3. Name score only        → use name-scored result
    4. No name match          → use _resolve_item_reference result
```

If the result is a default fallback (no real match) → search session history.

**Item promotion:**
The selected item is promoted to `item_a` position in `dialogue_state.currently_discussing`.
The item previously at `item_a` is moved to the selected item's original slot.
This ensures all subsequent turns (ATTRIBUTE_QUESTION, EXPLANATION_WHY, FEEDBACK)
default to the user's chosen item without needing to resolve again.

```
before: item_a=DressA, item_b=DressB, item_c=DressC
user: "tell me more about option 3"
after:  item_a=DressC, item_b=DressB, item_c=DressA  ← DressC promoted
```

**Payload:** `{article_id, context_article}`

---

### 5.7 FEEDBACK → `catalog_search` or NO

Classifies sentiment via Twitter-RoBERTa (Barbieri et al., EMNLP 2020) and
branches on the result.

**Sentiment → Memory update:**

| Sentiment | Memory side effect |
|-----------|-------------------|
| positive | `accepted_items` += article_id; recommendation outcome = "accepted"; purchase summary updated if score > 0.7 |
| negative | ALL shown items → `rejected_items`; recommendation outcome = "rejected" for each |
| neutral | Preferences updated only; no accepted/rejected changes; item stays "pending" |

**Retrieval decision:**

```
negative + items in context
    → catalog_search (FULL)
    → exclude_ids = all shown items + all previously rejected
    → payload includes current hard_constraints + preference_boosts + feedback_context

positive or neutral
    → retrieval_strategy = "NO"
    → retrieval_input = None
    → LLM acknowledges from parametric knowledge
```

**Why exclude ALL shown items on negative:**
*"I don't like them"* rejects the entire set, not just item_a. Enrichment collects
every `item_*` key from `currently_discussing` into the exclusion list.

---

### 5.8 CHITCHAT → NO retrieval

Returns minimal memory_context with no preference lookup, no dialogue_state query.
Passes only `user_message` so the response generator knows what was said.

---

## 6. Item Reference Resolution — Shared Logic

All four PARTIAL labels resolve in two stages: **choose the pool**, then
**resolve within it**.

### Stage 1 — Choosing the pool (`_resolve_pool`)

```
Start with the context window (dialogue_state.currently_discussing,
the 12 most recently recommended items, newest first)
    ↓
Score the message against it (_score_items_by_name_scored)
    ↓
Best score >= 50, or message contains an ordinal / price?
    ├─ Yes → window is enough. Return it. No MongoDB call.
    └─ No  → query MongoDB for every recommendation in this session,
             append the ones not already in the window, and resolve
             over the merged pool.
```

The threshold of 50 is the point on the scoring scale below which a match is
not a real reference: naming the product outright scores 100, and naming its
colour or price is worth 50 each, so anything lower is incidental word overlap.

**This fall-through is unconditional.** Any product recommended at any point in
the session stays referenceable for the rest of it — the window is a cache, and
MongoDB is the pool.

> **Superseded:** the fall-through used to be gated on the window producing *no
> match at all*. On a pool of same-category items, one shared word ("shorts")
> was enough to prevent it, so a product from an earlier turn was answered with
> whichever item happened to occupy `item_a`.

### Stage 2 — Resolving within the pool

```
Priority 1 — Ordinal reference
    "first", "option 1", "1st", "number one"    → newest turn's item 0
    "second", "option 2", "2nd", "the other"    → newest turn's item 1
    "third" ... "eighth"                         → newest turn's item 2..7

Priority 2 — Price match
    "£34.99" appears in message → item whose price == 34.99

Priority 3 — Colour match
    "the blue one" → item whose colour_group_name.lower() in message

Priority 4 — Name word overlap
    Significant words (>3 chars) from prod_name found in message
    Scored: full name = +100, per word = +1, colour = +50, price = +50

Priority 5 — Default
    pool[0] — the first item of the most recent recommendation
```

**Ordinals are scoped to the newest recommendation turn** (`_newest_turn_items`).
The pool spans several turns, so position in it is not the same as position in
what the user was last shown; "option 2" always means the second item of the
latest offer. Each item's `rec_turn` stamp is what makes this possible, which is
also why promoting a resolved item to `item_a` is safe — the answer comes from
the stamp, not the slot.

Priority 5 is the deliberate fallback for a message with no item reference at
all ("tell me more"): the most recently recommended item is the right default.

---

## 7. How Enrichment Connects to the Full Pipeline

```
Pipeline Step 4:  DistilBERT classification
                  → label = "ATTRIBUTE_QUESTION", confidence = 0.94

Pipeline Step 4b: CSE (Context Sufficiency Evaluator)
                  For AQ / COMP / EW / SELREF:
                    → CSE calls enrichment INTERNALLY at Step 4b
                    → Stores pre-built output in SufficiencyResult
                    → Pipeline uses pre-built output, skips Step 7 enrichment call
                    → Avoids duplicate MongoDB round-trip for same turn

                  For INITIAL_REQUEST / REFINEMENT:
                    → CSE only assigns tier (needs entity extraction first)
                    → Enrichment called fresh at Step 7

Pipeline Step 5:  Entity extraction → entities = {colour_group_name: "Black", ...}
                  (only for INITIAL_REQUEST / REFINEMENT)

Pipeline Step 6:  Turn stored in MongoDB with classification + entities

Pipeline Step 7:  enricher.enrich(label, strategy, session_id, user_id, message, entities)
                  → returns retrieval_input + memory_context + side_effects

Pipeline Step 7a: retrieval_input → EvidenceAssembler
                  action = "catalog_search"
                  payload.filters → Qdrant pre-filter
                  payload.preference_boosts → PostgreSQL ranking
                  → assembled evidence bundle

Pipeline Step 7b: evidence + memory_context → ResponseGenerator (Groq LLM)
                  → LLM prompt includes dialogue_state, preferences, items
                  → response text generated

Pipeline Step 7c: response → HallucinationChecker → ContradictionDetector
                  → final response returned to user
```

---

## 8. Purchase History Hints

`_get_purchase_hints(user_id)` reads the pre-loaded customer transaction profile
from MongoDB (loaded from H&M customer transaction CSV at startup). Returns:

```python
{
    "top_colours":           ["Black", "Blue", "White"],  # most purchased colours
    "top_product_types":     ["T-shirt", "Trousers"],     # most purchased types
    "inferred_gender":       "Ladieswear",                # from purchase patterns
    "budget_tier":           "mid-range",                 # spending pattern
    "preferred_price_range": "£20–£50",
    "dominant_colour":       "Black",
    "dominant_type":         "T-shirt",
}
```

These hints are passed to the RAG pipeline for INITIAL_REQUEST and REFINEMENT
so catalog results are personalised to the user's real purchase history even
before they have expressed any preference in the current session.

---

## 9. File Location

| Component | Location |
|-----------|----------|
| Enrichment layer | `m3_implementation/memory/core/enrichment.py` |
| Pipeline integration | `m3_implementation/memory/core/pipeline.py` |
| CSE (calls enrichment at Step 4b) | `m3_implementation/memory/core/context_sufficiency_evaluator.py` |
| Session state | `m3_implementation/memory/core/session_manager.py` |
| User preferences | `m3_implementation/memory/core/user_manager.py` |
| Feedback sentiment | `m3_implementation/memory/core/feedback_sentiment_classifier.py` |
| MongoDB session store | `m3_implementation/memory/db/mongo.py` |
