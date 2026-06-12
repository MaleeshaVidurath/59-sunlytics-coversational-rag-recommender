# Context Sufficiency Evaluator (CSE) — Full Process Documentation
### Sunlytics CRS — M3 Memory Module

`m3_implementation/memory/core/context_sufficiency_evaluator.py`

---

## 1. What Is the CSE

The Context Sufficiency Evaluator decides, for every user turn, **how much
retrieval is actually needed** given what already exists in session memory.

The central question it answers:

> **Is the current context sufficient to answer the query — or must we retrieve more?**

This is formalised as `Sufficient(q, C_t)` from information theory:
- `q`   — the user's current query
- `C_t` — all context accumulated so far in this session

If context is already sufficient (`Sufficient=1`) → bounded lookup, no full search.
If context is insufficient (`Sufficient=0`) → full catalog retrieval required.

### Academic Basis

| Paper | Contribution to CSE design |
|-------|---------------------------|
| Joren et al. "Sufficient Context: A New Lens on RAG" (ICLR 2025) | The `Sufficient(q, C_t)` formalism — tier assignment IS the sufficiency decision |
| Jeong et al. "Adaptive-RAG" (NAACL 2024) | Multi-step adaptive retrieval — different queries need different retrieval depths |
| Wang et al. "RAGate: Adaptive RAG for Conversational Systems" (NAACL 2025) | Applying adaptive retrieval to conversational recommendation specifically |
| Barbieri et al. "TweetEval" (EMNLP 2020) | Twitter-RoBERTa sentiment model used for FEEDBACK tier assignment |
| Roy et al. (2024) | Follow-up queries on known items → bounded lookup sufficient |

---

## 2. Three-Tier Decision System

The CSE assigns every user turn to exactly one retrieval tier:

| Tier | Labels | What happens |
|------|--------|-------------|
| **NO** | CHITCHAT, FEEDBACK (pos/neutral) | No retrieval — LLM answers from its parametric knowledge alone |
| **FULL** | INITIAL_REQUEST, REFINEMENT, FEEDBACK (neg+items) | Full ANN catalog search against Qdrant vector database |
| **PARTIAL** | ATTRIBUTE_QUESTION, EXPLANATION_WHY, COMPARISON, SELECTION_REFERENCE | Bounded lookup — items already in session memory, no catalog search needed |

### Sub-levels

Each tier has sub-levels that carry additional routing information:

**FULL sub-levels:**

| Sub-level | Meaning |
|-----------|---------|
| `FULL_STANDARD` | Fresh catalog search — no exclusions, no prior similar question |
| `FULL_WITH_EXCLUSIONS` | Catalog search with `excluded_ids` — already-seen article_ids are blocked from results |

**PARTIAL sub-levels:**

| Sub-level | Meaning |
|-----------|---------|
| `PARTIAL_RECENT` | Items needed are in the last 3 exchanges (Redis hot cache) |
| `PARTIAL_SESSION` | Items needed are in earlier session history (MongoDB) |

---

## 3. SufficiencyResult — What the CSE Returns

```python
@dataclass
class SufficiencyResult:
    tier:               str          # "FULL" | "PARTIAL" | "NO"
    label:              str          # DistilBERT label that triggered this evaluation
    prior_strategy:     str          # DistilBERT default before CSE override
    override:           bool         # True when CSE changed the default strategy
    rationale:          str          # human-readable explanation (with paper citations)

    full_subtype:         Optional[str]   # "FULL_STANDARD" | "FULL_WITH_EXCLUSIONS"
    partial_subtype:      Optional[str]   # "PARTIAL_RECENT" | "PARTIAL_SESSION"
    excluded_ids:         list            # article_ids to exclude from next search

    cached_recommendation: list          # items from a matched prior INITIAL_REQUEST
    enriched_retrieval_input: Optional[dict]   # pre-built retrieval input (skips re-enrichment)
    enriched_memory_context:  Optional[dict]   # pre-built memory context
    enriched_side_effects:    Optional[list]   # pre-built side effects
```

The `override` field is important — it records whenever the CSE changed what
DistilBERT's default strategy was. For example, a negative FEEDBACK turn has a
DistilBERT default of `NO` but CSE overrides it to `FULL_WITH_EXCLUSIONS`.

---

## 4. Where in the Pipeline CSE Runs

The CSE runs at **Pipeline Step 4b** — after DistilBERT classification and
before entity extraction:

```
Step 3b: FashionGuard (off-topic check)
Step 4:  DistilBERT intent classification → label + confidence
Step 4b: CSE evaluate(label, message, dialogue_state, history, session_id)
         → tier + subtype + excluded_ids + rationale
Step 5:  Entity extraction (only for INITIAL_REQUEST / REFINEMENT)
Step 6:  Store user turn
Step 7:  RAG pipeline with tier + entities + excluded_ids
```

---

## 5. Per-Label Evaluation — How Each Label Is Processed

### 5.1 CHITCHAT → Always NO

```
CHITCHAT
    ↓
tier = NO
rationale: "LLM answers from parametric knowledge. I(A;K|q,C)=0"
```

The LLM knows how to greet, say goodbye, and handle small talk entirely from
its own training. Retrieving from the fashion catalog adds nothing here.
`I(A;K|q,C) = 0` — mutual information between the answer and the catalog,
given the query and context, is zero.

---

### 5.2 FEEDBACK — Sentiment-Driven Tier Assignment

FEEDBACK turns are evaluated using **Twitter-RoBERTa** sentiment classification
(`cardiffnlp/twitter-roberta-base-sentiment-latest`, Barbieri et al. EMNLP 2020).

```
FEEDBACK message
    ↓
Twitter-RoBERTa → sentiment: positive | neutral | negative
    ↓
positive  →  tier = NO
             "User satisfied — LLM acknowledges."

neutral   →  tier = NO
             "No clear preference change expressed."

negative + items in context
          →  tier = FULL
             full_subtype = FULL_WITH_EXCLUSIONS
             excluded_ids = all currently_discussing item_a … item_d
             "User implicitly requests alternatives."
             (ALL shown items excluded, not just item_a)

negative + no items in context
          →  tier = NO
             "Cannot trigger exclusion-based search without context items."
```

**Why exclude ALL items on negative feedback:**
If the user said *"I don't like them"*, they rejected the entire set shown
(item_a through item_d if all were shown). Excluding only item_a would still
surface item_b which they also rejected. The CSE collects every
`item_*` key from `currently_discussing` to build the exclusion list.

---

### 5.3 INITIAL_REQUEST — Full or Cached

INITIAL_REQUEST normally requires a full catalog search — the candidate set
is entirely unknown. But the CSE checks whether the same question was asked
before in this session using **cosine similarity** (threshold = 0.75).

```
INITIAL_REQUEST message
    ↓
Query MongoDB for all prior INITIAL_REQUEST turns in this session
    ↓
MiniLM encodes: [current_message] + [prior message 1] + [prior message 2] + ...
    ↓
Cosine similarity between current and each prior
    ↓
sim ≥ 0.75 found?
    ├─ Yes → similar question detected
    │         → Look up the bot turn immediately after that prior turn
    │         → Retrieve cached recommendation items from MongoDB
    │         → tier = PARTIAL
    │           excluded_ids = those article_ids (retained for optional new search)
    │           cached_recommendation = the prior items
    │           subtype = PARTIAL_RECENT (if items in current dialogue_state)
    │                   = PARTIAL_SESSION (if found only in MongoDB history)
    │
    └─ No  → fresh question
              tier = FULL
              full_subtype = FULL_STANDARD
```

**Why 0.75 threshold:**
Below 0.75 the questions are related but not the same (e.g. *"black dress"* vs
*"dark dress"*) — different enough that a new search may surface better results.
At 0.75 or above the questions are semantically equivalent and showing the same
recommendations makes sense.

---

### 5.4 REFINEMENT — Always FULL with Exclusions

REFINEMENT means the user is modifying their existing search (different colour,
lower price, different style). The catalog must be re-queried with updated filters.

```
REFINEMENT turn
    ↓
Check dialogue_state:
    has_constraints = hard_constraints.product_type_name is set
    has_items = item_a or item_b in currently_discussing
    ↓
_all_session_article_ids(session_id)
    → all article_ids recommended across ALL turns in this session (MongoDB)
    ↓
full_subtype:
    FULL_WITH_EXCLUSIONS  — if has_items or has_constraints (prior context exists)
    FULL_STANDARD         — if no items and no constraints at all

tier = FULL
excluded_ids = every article_id the user has ever seen in this session
```

**Why exclude the entire session history:**
On refinement, the user explicitly wants *something different*. Showing any
previously recommended product, even briefly mentioned, would feel repetitive.
The exclusion list spans all recommendation turns in the session, not just the
current `currently_discussing` items.

---

### 5.5 ATTRIBUTE_QUESTION / EXPLANATION_WHY / COMPARISON / SELECTION_REFERENCE → PARTIAL

These four labels all reference items that were already recommended. The tier
is always PARTIAL — bounded lookup, no new catalog search. The subtype depends
on where the items are found.

**Decision tree:**

```
Item-reference label (AQ, EW, COMP, SELREF)
    ↓
Call enrichment-backed helper first (pre-builds retrieval_input + memory_context)
    ↓
Items in dialogue_state.currently_discussing?
    ├─ Yes → check recency:
    │         Items in last 3 exchanges (Redis)?
    │           ├─ Yes → PARTIAL_RECENT
    │           └─ No  → PARTIAL_SESSION
    │
    └─ No  → query MongoDB for any session recommendations
              ├─ Found → PARTIAL_SESSION
              └─ Not found → FULL_STANDARD (user asked without prior recommendations)
```

**COMPARISON** specifically requires **both** item_a and item_b. If only one
item is in context, the CSE falls through to the MongoDB fallback.

**SELECTION_REFERENCE** enrichment can return `NO` (no items exist at all) —
in that case the CSE falls through to the MongoDB fallback rather than
returning a PARTIAL with no items.

---

## 6. Target Item Resolution

For item-reference labels, the CSE identifies **which specific item** the user
is asking about before checking recency. This ensures the right item's presence
in history is checked.

```
_resolve_target_item(message, item_a, item_b)
    ↓
Step 1: Ordinal reference?
    "second", "option 2", "the 2nd", "latter", "last one" → item_b
    ↓
Step 2: Colour mention?
    message contains item_b's colour_group_name → item_b
    message contains item_a's colour_group_name → item_a
    ↓
Step 3: Product name word match?
    significant word from item_b's prod_name in message → item_b
    significant word from item_a's prod_name in message → item_a
    ↓
Step 4: Default → item_a (first / primary item)
```

**Example:**
- *"What fabric is the second one?"* → Step 1 matches "second" → item_b
- *"Tell me more about the blue dress"* → Step 2 matches Blue colour → blue item
- *"What about the Stark coat?"* → Step 3 matches "stark" in prod_name → that item

---

## 7. Memory Layers — Redis and MongoDB

The CSE reads from two memory stores with different scope:

| Layer | Store | Contents | Latency |
|-------|-------|----------|---------|
| Hot cache | Redis | Last 10 turns of this session | ~1ms |
| Full history | MongoDB | All turns since session start | ~5–20ms |

**`_items_in_recent_history(history, item_a, item_b)`**
— checks whether product names from the items appear in bot turn content
from the last 3 exchanges (the `history` list passed from the pipeline).
Matches on first 3+ significant words of the product name, plus generic
markers like `"£"`, `"option 1"`, `"here are"`.

**`_find_similar_question_exclusions(message, session_id)`**
— queries MongoDB for all prior INITIAL_REQUEST turns, encodes them with
MiniLM, and returns any that match at cosine ≥ 0.75.

**`_all_session_article_ids(session_id)`**
— queries the `recommendations` collection for every recommendation made in
this session, collects all article_ids across all turns.

**`_find_items_in_full_session(session_id)`**
— fallback query for the most recent recommendation document in this session.

---

## 8. Enrichment Integration

For ATTRIBUTE_QUESTION, COMPARISON, EXPLANATION_WHY, and SELECTION_REFERENCE,
the CSE does not just assign a tier — it calls the relevant enrichment function
from `enrichment.py` internally and stores the result in `SufficiencyResult`.

```python
# CSE calls enrichment:
_enrich_result = await EnrichmentLayer()._enrich_attribute_question(
    session_id, user_id, message, {}, state_obj
)

# Stores pre-built output in the result:
SufficiencyResult(
    ...
    enriched_retrieval_input = _enrich_result["retrieval_input"],
    enriched_memory_context  = _enrich_result["memory_context"],
    enriched_side_effects    = _enrich_result["side_effects"],
)
```

**Why:** The pipeline would call enrichment anyway in Step 7. By calling it
inside the CSE at Step 4b, the result is carried forward and the pipeline
skips the duplicate enrichment call — one less async database round-trip per
turn for these label types.

If the enrichment call fails for any reason, the CSE returns `None` for that
helper and falls through to the generic item-reference path, so the pipeline
degrades gracefully.

---

## 9. Full Evaluation Flow

```
Pipeline Step 4b: CSE.evaluate(label, message, dialogue_state, history, session_id)
    │
    ├─ label == CHITCHAT or FEEDBACK
    │       ↓
    │    _eval_dialogue()
    │       ├─ CHITCHAT → tier=NO
    │       └─ FEEDBACK → Twitter-RoBERTa sentiment
    │                      pos/neutral → tier=NO
    │                      neg+items   → tier=FULL, FULL_WITH_EXCLUSIONS, excluded_ids
    │                      neg+no-items → tier=NO
    │
    ├─ label == INITIAL_REQUEST
    │       ↓
    │    _eval_initial_request()
    │       ├─ MiniLM similarity vs all prior INITIAL_REQUEST turns
    │       ├─ sim ≥ 0.75 → tier=PARTIAL, cached_recommendation
    │       └─ no match   → tier=FULL, FULL_STANDARD
    │
    ├─ label == REFINEMENT
    │       ↓
    │    _eval_refinement()
    │       ├─ _all_session_article_ids() → excluded_ids
    │       └─ tier=FULL, FULL_WITH_EXCLUSIONS (or FULL_STANDARD if no context)
    │
    └─ label == AQ / EW / COMP / SELREF
            ↓
         _eval_item_reference()
            ├─ _dispatch_enrichment_cse() → calls enrichment, stores pre-built output
            │       ↓
            │   enrichment-backed helpers:
            │     ATTRIBUTE_QUESTION   → _eval_attr_q_cse()
            │     COMPARISON           → _eval_comparison_cse() (requires both items)
            │     EXPLANATION_WHY      → _eval_explanation_why_cse()
            │     SELECTION_REFERENCE  → _eval_selection_ref_cse()
            │
            ├─ items in dialogue_state?
            │    ├─ Yes → _resolve_target_item() → _items_in_recent_history()
            │    │          → PARTIAL_RECENT or PARTIAL_SESSION
            │    └─ No  → _find_items_in_full_session() (MongoDB fallback)
            │               ├─ Found → PARTIAL_SESSION
            │               └─ None  → FULL_STANDARD
            │
            └─ SufficiencyResult with tier + subtype + enriched output
```

---

## 10. Per-Label Summary Table

| Label | Default | CSE Output | Sub-level | Exclusions | Condition |
|-------|---------|-----------|-----------|-----------|-----------|
| CHITCHAT | NO | NO | — | none | always |
| FEEDBACK positive | NO | NO | — | none | Twitter-RoBERTa positive |
| FEEDBACK neutral | NO | NO | — | none | Twitter-RoBERTa neutral |
| FEEDBACK negative + items | NO | **FULL** *(override)* | FULL_WITH_EXCLUSIONS | all shown items | CSE override |
| FEEDBACK negative + no items | NO | NO | — | none | no context |
| INITIAL_REQUEST (fresh) | FULL | FULL | FULL_STANDARD | none | no similar question |
| INITIAL_REQUEST (repeat) | FULL | **PARTIAL** *(override)* | PARTIAL_RECENT / SESSION | cached item ids | cosine ≥ 0.75 |
| REFINEMENT | FULL | FULL | FULL_WITH_EXCLUSIONS | all session ids | items or constraints exist |
| REFINEMENT (no prior) | FULL | FULL | FULL_STANDARD | none | no constraints, no items |
| ATTRIBUTE_QUESTION | PARTIAL | PARTIAL | PARTIAL_RECENT / SESSION | none | items in context |
| EXPLANATION_WHY | PARTIAL | PARTIAL | PARTIAL_RECENT / SESSION | none | items in context |
| COMPARISON | PARTIAL | PARTIAL | PARTIAL_RECENT / SESSION | none | both items in context |
| SELECTION_REFERENCE | PARTIAL | PARTIAL | PARTIAL_RECENT / SESSION | none | items in context |
| Any item-ref, no items | PARTIAL | **FULL** *(override)* | FULL_STANDARD | none | no session history |

---

## 11. Multiple CSE Instances (M1 / M2 / M3)

The system has three RAG pipeline members (M1 Graph RAG, M2 Multimodal RAG,
M3 Text RAG). Each member has its own recommendations MongoDB collection:

```
M3 (default)  → "recommendations"
M2            → "m2_recommendations"
M1            → "m1_recommendations"
```

Each member gets its own CSE singleton via `get_cse_for_model("m1"|"m2"|"m3")`.
The only difference between instances is which collection they query — all
tier assignment logic is identical.

This ensures that follow-up turns (`ATTRIBUTE_QUESTION`, `EXPLANATION_WHY` etc.)
find items from the correct pipeline member's recommendations, not from another
member's output.

---

## 12. File Location

| Component | Location |
|-----------|----------|
| CSE implementation | `m3_implementation/memory/core/context_sufficiency_evaluator.py` |
| Pipeline integration (Step 4b) | `m3_implementation/memory/core/pipeline.py` |
| Enrichment layer (called by CSE) | `m3_implementation/memory/core/enrichment.py` |
| Feedback sentiment classifier | `m3_implementation/memory/core/feedback_sentiment_classifier.py` |
| Session memory (MongoDB) | `m3_implementation/memory/db/mongo.py` |
