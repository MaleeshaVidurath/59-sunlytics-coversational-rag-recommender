# M3 Memory Architecture — Redis & MongoDB

## Overview

The system uses two databases together. They serve completely different purposes and are always kept in sync.

```
                    ┌─────────────────────────────────────┐
Every chat turn →   │  Redis   (hot cache, RAM, fast)     │  ← current session
                    │  MongoDB (permanent store, disk)    │  ← all history
                    └─────────────────────────────────────┘
```

**Redis** = what is happening RIGHT NOW in this conversation (sub-millisecond access)  
**MongoDB** = the permanent record of everything that ever happened (full history)

---

## Redis — Hot Session Cache

Redis is an in-memory key-value store. Everything has a TTL (time-to-live) and expires. Three types of keys per session plus one user-level key.

---

### Key 1 — Session State Hash

```
key:  session:{session_id}:state
type: Hash (dict of named fields)
TTL:  30 minutes — reset on every turn
```

| Field | Example value | Purpose |
|---|---|---|
| `user_id` | `"user_f137c16f"` | Who owns this session |
| `status` | `"active"` | Lifecycle state |
| `started_at` | `"2026-05-20T16:30:00Z"` | When session began |
| `last_activity_at` | `"2026-05-20T16:35:00Z"` | Time of last message |
| `dialogue_state` | `"{...json...}"` | Full working memory (see below) |

The `dialogue_state` field is a JSON string of the entire `DialogueState` object:

```json
{
  "hard_constraints": {"product_type_name": "Sneakers"},
  "soft_constraints": {"style": "casual"},
  "currently_discussing": {
    "item_a": {
      "article_id": "748748007",
      "prod_name": "Winter sneaker",
      "product_type_name": "Sneakers",
      "colour_group_name": "Dark Yellow",
      "price": 14.11,
      "rec_turn": 6
    },
    "item_b": {
      "article_id": "599716003",
      "prod_name": "OL SOUTH PQ sneaker",
      "colour_group_name": "Light Beige",
      "price": 29.24,
      "rec_turn": 6
    },
    "item_c": {
      "article_id": "612345001",
      "prod_name": "Suede trainer",
      "colour_group_name": "Black",
      "price": 22.50,
      "rec_turn": 2
    },
    "item_d": null
  },
  "rejected_items": [],
  "accepted_items": [],
  "intent_summary": "User wants casual sneakers",
  "awaiting_new_recommendation_consent": false,
  "pending_new_rec_excluded_ids": [],
  "pending_original_message": null
}
```

This is the "working memory" — everything the system needs to understand the current conversation state. It is read and written on every single turn.

### `currently_discussing` — the context window

`currently_discussing` is a **rolling window of the 12 most recently
recommended items, newest first** (`CONTEXT_WINDOW_ITEMS` in `pipeline.py`),
spanning slots `item_a` … `item_l`. In the example above `item_a`/`item_b` came
from turn 6 and `item_c` is still held from turn 2.

Each item stores its complete record — description, price, and the ranker's
justification strings — roughly 630 B per item, so a full window is about 8 KB.
Keeping the full record is what lets `item_detail_lookup` answer from memory
with no database query at all.

`_push_context_window` maintains it: new items go on the front, earlier turns'
items shift back, an item recommended again returns to the front with a fresh
`rec_turn`, and the oldest fall off past 12. All 12 slots are rewritten every
time — including trailing `null`s — because `update_dialogue_state` deep-merges
dicts, so an omitted key would strand a dropped item.

> **Superseded:** this dict used to be wiped and refilled with only the current
> turn's items. That kept ordinal references honest, but made anything
> recommended earlier in the session unreferenceable. `rec_turn` now separates
> the two concerns — ordinals read the stamp, name references search the whole
> window — so history can be retained without ordinals losing their meaning.

Items older than the window are not lost; they live in the `recommendations`
collection and are recovered on demand by `enrichment._resolve_pool`.

---

### Key 2 — Turns List

```
key:  session:{session_id}:turns
type: List (ordered queue, oldest first)
TTL:  30 minutes — reset on every turn
max:  10 turns kept (oldest auto-trimmed by ltrim)
```

Each item in the list is a JSON string of one `ConversationTurn`. User turn example:

```json
{
  "turn_id": "turn_4894e5f9",
  "turn_number": 1,
  "role": "user",
  "content": "I need 3 shoes",
  "timestamp": "2026-05-20T16:33:00Z",
  "classification": {
    "label": "INITIAL_REQUEST",
    "retrieval_strategy": "FULL",
    "confidence": 0.874,
    "used_rules": false
  },
  "entities": {
    "product_type_name": "Sneakers",
    "occasion": "casual",
    "style": "sporty"
  },
  "preferences_updated": ["pref_a3b2c1"],
  "recommendation_id": null
}
```

Bot (assistant) turn example:

```json
{
  "turn_id": "turn_9f8e7d",
  "turn_number": 2,
  "role": "assistant",
  "content": "Hi there, I'd be happy to help you find the perfect shoes...",
  "timestamp": "2026-05-20T16:33:05Z",
  "classification": null,
  "entities": {},
  "preferences_updated": [],
  "recommendation_id": "rec_4a3b2c"
}
```

**Why only 10?** Redis lives in RAM — keeping every turn for every active user would be expensive. MongoDB holds all turns permanently. The last 10 is enough for DistilBERT context building, CSE recency checking, and consent detection.

---

### Key 3 — Active Session Pointer

```
key:   user:{user_id}:active_session
type:  String
TTL:   30 minutes
value: "sess_56ce7f97"
```

Maps user → their current session ID. When a new message arrives with no `session_id`, the system reads this key to resume the correct session instantly without a MongoDB query.

---

### Key 4 — User Preferences Cache

```
key:  user:{user_id}:preferences
type: String (JSON serialised UserDocument)
TTL:  60 minutes
```

A cached snapshot of the user's long-term preference profile from MongoDB. The enrichment layer reads this on every turn instead of hitting MongoDB. Invalidated whenever preferences are updated so stale data never persists longer than one turn.

---

## MongoDB — Permanent Store

MongoDB is the source of truth. Nothing expires. Five collections:

---

### Collection 1 — `users`

One document per user. Never deleted.

```json
{
  "user_id": "user_f137c16f",
  "customer_id": "0001d44dbe7f6c4b...",

  "club_member_status": "ACTIVE",
  "fashion_news_frequency": "Regularly",
  "age": 28,
  "postal_code": "SE10 0BE",

  "created_at": "2026-05-01T10:00:00Z",
  "last_active_at": "2026-05-20T16:35:00Z",

  "attribute_preferences": [
    {
      "pref_id": "pref_a3b2c1",
      "category": "colour",
      "attribute_name": "colour_group_name",
      "attribute_value": "Black",
      "sentiment": 0.9,
      "confidence": 0.8,
      "source": "explicit",
      "mention_count": 3,
      "decay_weight": 1.0,
      "first_mentioned_at": "2026-05-01T10:00:00Z",
      "last_mentioned_at": "2026-05-20T16:33:00Z"
    }
  ],

  "disliked_attributes": [
    {
      "pref_id": "pref_b4c3d2",
      "category": "colour",
      "attribute_name": "colour_group_name",
      "attribute_value": "Orange",
      "sentiment": -0.8,
      "confidence": 0.7,
      "source": "explicit",
      "mention_count": 1,
      "decay_weight": 0.9
    }
  ],

  "style_profile": {
    "primary_style": "casual",
    "secondary_styles": ["sporty"],
    "occasion_preferences": {"casual": 0.9, "work": 0.3},
    "size_preferences": {"tops": "M", "bottoms": "28"}
  },

  "purchase_summary": {
    "total_purchases": 42,
    "avg_price_normalized": 18.50,
    "top_product_types": ["Blouse", "Trousers", "Sweater"],
    "top_colours": ["White", "Black", "Light Beige"],
    "top_index_groups": ["Ladieswear"]
  },

  "purchase_history": {
    "top_colours": ["White", "Black", "Light Beige", "Light Pink"],
    "dominant_colour": "White",
    "top_product_types": ["Blouse", "Trousers", "Sweater"],
    "dominant_product_type": "Blouse",
    "inferred_gender": "female",
    "budget_tier": "mid",
    "preferred_price_range": [15.12, 25.71],
    "price_stats": {"min": 2.09, "max": 59.90, "mean": 18.50},
    "recency_score": 0.85
  }
}
```

**Indexes:** `customer_id` (unique).

**Preference fields explained:**

| Field | Meaning |
|---|---|
| `sentiment` | `1.0` = loves it, `-1.0` = hates it, `0.0` = neutral |
| `confidence` | `1.0` = user said it explicitly, `0.3` = weakly inferred |
| `source` | `explicit` / `implicit` / `mixed` |
| `mention_count` | How many times this preference has been referenced |
| `decay_weight` | Starts at 1.0, decreases over time as preferences age |

---

### Collection 2 — `sessions`

One document per conversation session. Updated on every turn.

```json
{
  "session_id": "sess_56ce7f97",
  "user_id": "user_f137c16f",
  "status": "active",

  "started_at": "2026-05-20T16:30:00Z",
  "last_activity_at": "2026-05-20T16:35:00Z",
  "ended_at": null,
  "timeout_minutes": 30,

  "dialogue_state": {
    "hard_constraints": {"product_type_name": "Sneakers"},
    "soft_constraints": {},
    "currently_discussing": {
      "item_a": {"article_id": "748748007", "prod_name": "Winter sneaker", ...},
      "item_b": {"article_id": "599716003", "prod_name": "OL SOUTH PQ sneaker", ...}
    },
    "rejected_items": [],
    "accepted_items": [],
    "intent_summary": "User wants casual sneakers",
    "awaiting_new_recommendation_consent": false,
    "pending_new_rec_excluded_ids": [],
    "pending_original_message": null
  },

  "turns": [
    {
      "turn_id": "turn_4894e5f9",
      "turn_number": 1,
      "role": "user",
      "content": "I need 3 shoes",
      "classification": {"label": "INITIAL_REQUEST", "retrieval_strategy": "FULL", ...},
      "entities": {"product_type_name": "Sneakers"}
    },
    {
      "turn_id": "turn_9f8e7d",
      "turn_number": 2,
      "role": "assistant",
      "content": "Hi there, I'd be happy to help...",
      "recommendation_id": "rec_4a3b2c"
    }
  ],
  "turn_count": 2
}
```

**Indexes:** `session_id` (unique), `(user_id, started_at)`, `status`.

**Session status values:**

| Status | Meaning |
|---|---|
| `active` | Conversation is ongoing |
| `completed` | User explicitly ended the session |
| `expired` | Session timed out after 30 minutes of inactivity |
| `abandoned` | Session was left without a clean end |

The `turns` array grows with every message. Redis keeps the last 10; MongoDB keeps every single turn forever. The CSE's similar-question detection queries this collection to read all prior `INITIAL_REQUEST` user turns.

---

### Collection 3 — `recommendations`

One document per recommendation event — every time the bot showed products to the user.

```json
{
  "recommendation_id": "rec_4a3b2c",
  "session_id": "sess_56ce7f97",
  "user_id": "user_f137c16f",
  "turn_id": "turn_9f8e7d",
  "created_at": "2026-05-20T16:33:05Z",

  "items": [
    {
      "article_id": "748748007",
      "prod_name": "Winter sneaker",
      "product_type_name": "Sneakers",
      "colour_group_name": "Dark Yellow",
      "index_group_name": "Baby/Children",
      "section_name": "Kids & Baby Shoes",
      "garment_group_name": "Shoes",
      "detail_desc": "Hi-tops with a lightly padded edge...",
      "graphical_appearance_name": "Solid",
      "price": 14.11
    },
    { "article_id": "599716003", "prod_name": "OL SOUTH PQ sneaker", ... },
    { "article_id": "549978007", "prod_name": "Kendal fancy slipon SG", ... }
  ],

  "trigger_label": "INITIAL_REQUEST",
  "retrieval_strategy": "FULL",
  "outcome": "pending",
  "user_turn_id": "turn_4894e5f9"
}
```

**Indexes:** `session_id`, `(user_id, created_at)`.

**This collection is read by:**
- CSE `_find_similar_question_exclusions` — fetches cached items when a similar question is detected
- CSE `_all_session_article_ids` — collects all article_ids to exclude on REFINEMENT searches
- CSE `_find_items_in_full_session` — fallback when `dialogue_state.currently_discussing` is empty
- Enrichment `_resolve_pool` / `_collect_session_items` — widens the resolution pool
  beyond the context window when a message references an item that has aged out of it.
  This is what guarantees any product recommended in the session stays referenceable.
- `chat.py` — finds the latest `recommendation_id` for RL feedback linking

---

### Collection 4 — `explanations`

One document per explanation generated (EXPLANATION_WHY label turns).

```json
{
  "explanation_id": "expl_7f8e9d",
  "recommendation_id": "rec_4a3b2c",
  "article_id": "748748007",
  "session_id": "sess_56ce7f97",
  "user_id": "user_f137c16f",
  "turn_id": "turn_...",
  "created_at": "2026-05-20T16:34:00Z",

  "full_explanation": "This sneaker was recommended because it matches your preference for casual style...",

  "claims": [
    {
      "claim_id": "claim_1a2b",
      "claim_text": "This item is Dark Yellow in colour.",
      "claim_type": "attribute_fact",
      "attribute": "colour_group_name",
      "evidence_value": "Dark Yellow",
      "user_preference_ref": null,
      "confidence": 1.0,
      "status": "active"
    },
    {
      "claim_id": "claim_2b3c",
      "claim_text": "This matches your preference for casual style.",
      "claim_type": "style_match",
      "attribute": "occasion",
      "evidence_value": "casual",
      "user_preference_ref": "pref_a3b2c1",
      "confidence": 0.85,
      "status": "active"
    }
  ],
  "contradiction_log": []
}
```

**Claim status values:** `active` / `retracted` / `contradicted` / `confirmed`

---

### Collection 5 — `contradiction_log`

One document per contradiction detected between claims across turns.

```json
{
  "contradiction_id": "contra_9z8y7x",
  "session_id": "sess_56ce7f97",
  "user_id": "user_f137c16f",
  "detected_at": "2026-05-20T16:36:00Z",

  "old_claim_id": "claim_1a2b",
  "old_claim_text": "This item costs £14.11.",
  "new_claim_text": "This item costs £12.00.",
  "article_id": "748748007",
  "attribute": "price",

  "nli_score": 0.94,

  "resolution": "retract_old",
  "resolution_explanation": "Price updated — old claim retracted."
}
```

**Resolution values:** `retract_old` / `update_old` / `notify_user` / `pending`

---

## How the Two Layers Work Together Per Turn

```
Message arrives
      │
      ▼
1. get_or_create_session()
   → Check Redis: user:{id}:active_session            (~1ms)
   → If found: load state from Redis Hash              (~1ms)
   → If not in Redis: load from MongoDB, warm Redis    (~10ms)
   → If not in MongoDB: create new session in both

      │
      ▼
2. get_recent_turns()   [for DistilBERT + CSE]
   → lrange session:{id}:turns -6 -1                  (~1ms from Redis)
   → Fallback: MongoDB $slice if Redis is cold

      │
      ▼
3. Step 3.5 — consent check
   → get_dialogue_state() from Redis Hash              (~1ms)
   → Reads awaiting_new_recommendation_consent flag

      │
      ▼
4. DistilBERT classifies → label + strategy

      │
      ▼
5. CSE evaluates
   → get_dialogue_state() from Redis                   (~1ms)
   → _find_similar_question_exclusions() → MongoDB sessions    (~10ms)
   → _all_session_article_ids()          → MongoDB recommendations (~10ms)
   → _find_items_in_full_session()       → MongoDB recommendations (~5ms)

      │
      ▼
6. Entity extraction + add_user_turn()
   → rpush + ltrim  → Redis turns list                 (~1ms)
   → $push + $inc   → MongoDB sessions.turns array     (~5ms)

      │
      ▼
7. update_dialogue_state()   [enricher updates hard_constraints etc.]
   → hset dialogue_state field → Redis Hash            (~1ms)
   → $set dialogue_state.*    → MongoDB sessions       (~5ms)

      │
      ▼
8. RAG generates response

      │
      ▼
9. store_response()   [after RAG]
   → add_assistant_turn()  → Redis list + MongoDB sessions.turns
   → insert recommendation → MongoDB recommendations
   → _push_context_window() → Redis Hash + MongoDB sessions
      (prepends this turn's items to the 12-item window, stamps rec_turn)
```

---

## Why This Split

| Need | Solution | Reason |
|---|---|---|
| Read dialogue state 3× per turn | Redis Hash hget | Sub-ms, no network DB round trip |
| Last 10 turns for DistilBERT context | Redis List lrange | O(1) slice from RAM |
| All INITIAL_REQUEST turns ever asked | MongoDB sessions.turns | Full history, Redis only has last 10 |
| All items ever recommended this session | MongoDB recommendations | Permanent, survives Redis restart |
| User preferences across sessions | MongoDB users + Redis cache (60 min TTL) | Preferences change slowly, cache is safe |
| Survive Redis restart | `_warm_redis_from_mongodb()` | MongoDB is always the source of truth |
| Session timeout cleanup | Redis TTL auto-expiry | No background cleanup job needed |
| RL feedback linking | MongoDB recommendations.user_turn_id | Permanent cross-collection reference |

**The core rule: Redis is the speed layer for things needed right now. MongoDB is the truth layer for everything that ever happened.**
