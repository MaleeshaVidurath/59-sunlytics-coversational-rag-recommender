# Contradiction Detector — Full Cross-Turn Workflow

## Overview

The contradiction detector ensures that every bot response is consistent with
the database evidence used to generate it — both within a single turn and
**across all previous turns of the same session**.

It uses three components together:

| Component | Role |
|---|---|
| **NetworkX DiGraph** | Session memory graph — one per session, persisted in MongoDB across turns |
| **Groq (llama-3.1-8b-instant)** | Extracts what name/colour/price the LLM actually wrote in the response |
| **DeBERTa NLI (cross-encoder/nli-deberta-v3-base)** | Confirms a suspected mismatch is a real semantic contradiction, not a surface difference |

The detector is called from `rag_pipeline.py` after the hallucination checker
passes. It only runs for factual actions:
`catalog_search`, `item_detail_lookup`, `item_attribute_lookup`,
`item_compare`, `explanation_generate`.

---

## Why cross-turn detection matters

The LLM generates each response independently — it has no memory of what it
said two turns ago. If a product was shown in Turn 1 as Black and the LLM
mentions it again in Turn 4 as Navy, there is no hallucination checker that
catches this. The session graph does.

```
Turn 1: "SC COLUMBUS blouse — Black — £57.14"   ← stored in graph
Turn 4: "SC COLUMBUS blouse — Navy — £57.14"    ← Groq extracts "Navy"
                                                 ← graph node says "Black"
                                                 ← values_contradict() fires
                                                 ← NLI confirms
                                                 ← response corrected before user sees it
```

---

## Session Graph — Structure

One MongoDB document per session in `db.session_graphs`:

```json
{
  "session_id": "sess_eea8c1bf",
  "graph_data": {
    "nodes": [
      {
        "id": "733839003",
        "name": "SC COLUMBUS blouse",
        "colour": "Black",
        "price": "£57.14",
        "first_seen_turn": "turn_bd25beee",
        "last_seen_turn":  "turn_bd25beee",
        "session_id": "sess_eea8c1bf"
      },
      ...
    ],
    "links": [
      {
        "source": "733839003",
        "target": "733839003_contra_colour_turn_xyz",
        "attribute": "colour",
        "old_value": "Navy",
        "new_value": "Black",
        "nli_score": 0.87
      }
    ]
  }
}
```

- **Nodes** = products seen at any point in the session (ground truth from DB)
- **Edges** = contradiction events (only added when a confirmed contradiction is found)

Sessions are completely isolated — two sessions never share a graph.

---

## Step-by-Step Workflow (per turn)

### Step 1 — Extract ground truth from evidence bundle

```
Evidence bundle (PostgreSQL / Qdrant — always accurate)
        ↓
_extract_product_refs(evidence)
        ↓
[
  { article_id: "733839003", name: "SC COLUMBUS blouse", colour: "Black",       price: "£57.14" },
  { article_id: "733838002", name: "SC UTAH blouse",     colour: "Light Beige", price: "£70.59" }
]
```

Handles field naming differences between sources:
- `colour_group_name` (PostgreSQL field) OR `colour` (internal assembler key)
- `prod_name` OR `name`
- `price` OR `avg_price`

---

### Step 2 — Load session graph from MongoDB

```
MongoDB db.session_graphs → find_one({ session_id: "sess_eea8c1bf" })
        ↓
NetworkX DiGraph  (empty DiGraph if first turn of session)

[GRAPH] nodes=4 edges=0   ← products seen in prior turns already in graph
```

The graph carries the full product history of the session. Products seen in
Turn 1 are still in the graph at Turn 5.

---

### Step 3 — Update graph nodes with current evidence

```
_update_graph_nodes(graph, product_refs, turn_id, session_id)
        ↓
[GRAPH] nodes=6            ← 2 new product nodes added
```

**If the product node already exists** (seen in a prior turn):
values are **overwritten** with the latest DB evidence. The DB is always right.

**If the product node is new**:
a new node is created with all evidence fields.

```
node "733839003" = {
    name:            "SC COLUMBUS blouse"
    colour:          "Black"
    price:           "£57.14"
    first_seen_turn: "turn_bd25beee"
    last_seen_turn:  "turn_bd25beee"
    session_id:      "sess_eea8c1bf"
}
```

---

### Step 4 — Extract LLM claims via Groq

```
response_text  +  product list (article_id → name)
        ↓
Groq API  —  llama-3.1-8b-instant  temperature=0.0
        ↓
{
  "733839003": { "name": "SC COLUMBUS blouse", "colour": "Black",       "price": "£57.14" },
  "733838002": { "name": "SC UTAH blouse",     "colour": "Light Beige", "price": "£70.59" }
}
```

The prompt gives Groq the article IDs and product names explicitly so it can
return **article_id-keyed JSON**. This avoids all name-matching ambiguity —
even if two products share the same name (e.g. two variants of "Charlotte
lowback bra"), Groq keys the output by article_id.

If Groq fails or returns invalid JSON → check is skipped gracefully.
The original response is returned to the user with no crash.

---

### Step 5 — Compare extracted claims vs graph node values

For every product Groq extracted, compare `colour`, `price`, and `name`
against the graph node (which holds the DB ground truth):

```
article_id = "733839003"

  graph node  (evidence truth):
      colour = "Black"    price = "£57.14"    name = "SC COLUMBUS blouse"

  groq extracted:
      colour = "Black"    price = "£57.14"    name = "SC COLUMBUS blouse"

  values_contradict("Black",           "Black")           → False  ✓ skip
  values_contradict("£57.14",          "£57.14")          → False  ✓ skip
  values_contradict("SC COLUMBUS blouse","SC COLUMBUS blouse") → False  ✓ skip

→ No contradiction for this product.
```

**`values_contradict()` normalisation rules:**

- Both values lowercased, stripped, whitespace/hyphen-collapsed
- `£` whitespace normalised (£ 57.14 → £57.14)
- Prices: float comparison with tolerance ±0.05 (avoids rounding false positives)
- Text: exact equality after normalisation
- Either value empty → always False (can't contradict what is unknown)

---

### Step 5b — NLI confirmation gate

`values_contradict()` is strict — it catches `"Utah" ≠ "Utah blouse"` as a
mismatch even though they refer to the same product. DeBERTa NLI runs as a
second gate to confirm the mismatch is semantically contradictory:

```
Premise:    "The SC UTAH blouse is Light Beige and costs £70.59."
Hypothesis: "The SC UTAH blouse is Blue in colour."   ← extracted wrong value

DeBERTa NLI scores:
  Label 0 = CONTRADICTION score
  Label 1 = NEUTRAL score
  Label 2 = ENTAILMENT score

contra_score > 0.5  → CONFIRMED contradiction
contra_score ≤ 0.5  → false positive, skip
```

**Example of NLI correctly rejecting a false positive (from real log):**
```
[CONTRA-CANDIDATE] 877274003 | name | evidence='Utah' extracted='Utah blouse'
[CONTRA] NLI not confirmed (score=-3.889) — skip
```
"Utah" vs "Utah blouse" — `values_contradict()` flagged it, NLI score=-3.889
(deep in ENTAILMENT territory) → correctly skipped, no correction applied.

---

### Step 5c — Fix response text (confirmed contradictions only)

```python
corrected_text = _fix_response_text(
    response_text = corrected_text,
    wrong_value   = extracted_val,   # what LLM wrote  e.g. "Navy"
    correct_value = evidence_val,    # what DB says    e.g. "Black"
)
```

Replaces the wrong value with the correct evidence value in the response.
Tries exact match first, then case-insensitive. The corrected text is what
the user ultimately sees.

---

### Step 5d — Record contradiction in graph and MongoDB

**Graph edge added:**
```
"733839003"  ──[colour | old=Navy | new=Black | nli=0.87]──▶  "733839003_contra_colour_turn_xyz"
```

**MongoDB `db.contradiction_log` entry:**
```json
{
  "session_id":      "sess_eea8c1bf",
  "turn_id":         "turn_xyz",
  "article_id":      "733839003",
  "article_name":    "SC COLUMBUS blouse",
  "attribute":       "colour",
  "evidence_value":  "Black",
  "extracted_value": "Navy",
  "nli_score":       0.87,
  "resolution":      "response_corrected"
}
```

---

### Step 6 — Persist updated graph to MongoDB

```
nx.node_link_data(graph)  →  JSON  →  db.session_graphs.update_one(upsert=True)
```

The graph is saved back with all new nodes and contradiction edges. The next
turn of this session loads this updated graph.

---

## Full Pipeline Diagram

```
LLM response text  +  evidence bundle (DB ground truth)
        │
        ▼
Step 1: _extract_product_refs(evidence)
        → normalised refs: [{ article_id, name, colour, price }, ...]

        ▼
Step 2: _load_graph(session_id)
        → NetworkX DiGraph from MongoDB
        → contains all products seen in prior turns of this session

        ▼
Step 3: _update_graph_nodes(graph, product_refs)
        → add new product nodes, overwrite existing ones with latest DB values

        ▼
Step 4: _extract_claims_groq(response_text, product_refs)
        → Groq reads the LLM response
        → returns { article_id: { colour, price, name } }
        → failure → skip check, save graph, return original response

        ▼
Step 5: for each product in extracted claims:
          │
          ├─ values_contradict(graph_node_value, extracted_value)?
          │       NO  → skip attribute
          │       YES ↓
          │
          ├─ _confirm_with_nli(premise, hypothesis)
          │       contra_score ≤ 0.5  → false positive, skip
          │       contra_score > 0.5  ↓
          │
          ├─ [CONTRA-DETECTED]
          │       _fix_response_text()          → corrected response
          │       graph.add_edge()              → contradiction edge in graph
          │       _log_contradiction()          → MongoDB contradiction_log

        ▼
Step 6: _save_graph(session_id, graph)
        → persisted for next turn

        ▼
Return corrected response text to user
```

---

## Cross-Turn Example — 3 Turns in One Session

```
────────────────────────────────────────────────────────
TURN 1   "I need blouse"   →  catalog_search
────────────────────────────────────────────────────────
Evidence:  733839003 SC COLUMBUS blouse  Black      £57.14
           733838002 SC UTAH blouse      Light Beige £70.59

Graph before: nodes=4  (products from earlier session turns)
Graph after:  nodes=6  (2 blouses added as new nodes)

Groq extracted:
  733839003 → colour=Black      price=£57.14   ✓ matches graph
  733838002 → colour=Light Beige price=£70.59  ✓ matches graph

Result: found=False  No contradiction.

────────────────────────────────────────────────────────
TURN 2   "why SC UTAH blouse"   →  catalog_search (misclassified)
────────────────────────────────────────────────────────
Evidence:  733838002 SC UTAH blouse   Light Beige £70.59
           877274003 Utah             White       £20.13

Graph before: nodes=6
Graph after:  nodes=7  (877274003 Utah added)

Groq extracted:
  733838002 → colour=light beige  price=£70.59
              normalise("Light Beige") = normalise("light beige") = "light beige"
              → values_contradict = False  ✓ matches

  877274003 → name=Utah blouse
              normalise("Utah") ≠ normalise("Utah blouse")
              → values_contradict = True → CANDIDATE
              → NLI: contra_score = -3.889  ≤ 0.5  → false positive, skip

Result: found=False  NLI correctly rejected the name surface difference.

────────────────────────────────────────────────────────
TURN 3   "why this Utah blouse"   →  explanation_generate
────────────────────────────────────────────────────────
Evidence:  733838002 SC UTAH blouse  Light Beige £70.59

Graph before: nodes=7
Graph after:  nodes=7  (node 733838002 already exists — values refreshed)

Groq extracted:
  733838002 → colour=Light Beige  price=£70.59  ✓ matches graph

Result: found=False  Clean pass.

Graph after 3 turns:
  nodes = 7  (all products seen across this session)
  edges = 0  (no confirmed contradictions)
```

---

## What Happens When a Contradiction IS Detected

```
Hypothetical Turn 4:  LLM drifts and writes "Navy" instead of "Black"
────────────────────────────────────────────────────────────────────────
Graph node 733839003:  colour = "Black"   (stored from Turn 1)
Groq extracts:         colour = "Navy"

values_contradict("Black", "Navy") → True  → CANDIDATE

NLI:
  Premise:    "The SC COLUMBUS blouse is Black and costs £57.14."
  Hypothesis: "The SC COLUMBUS blouse is Navy in colour."
  contra_score = 0.87  >  0.5  → CONFIRMED

[CONTRA-DETECTED] 733839003 | colour | evidence='Black' | extracted='Navy' | NLI=0.870

_fix_response_text():
  "...SC COLUMBUS blouse in Navy..." → "...SC COLUMBUS blouse in Black..."

Graph edge added:
  "733839003" ──[colour | old=Navy | new=Black | nli=0.87]──▶ "733839003_contra_colour_turn_4"

MongoDB contradiction_log entry written.
Graph saved.

User receives corrected response with "Black" — drift invisible to user.
```

---

## Design Decisions

| Decision | Reason |
|---|---|
| DB evidence is always truth | PostgreSQL / Qdrant never hallucinate — only the LLM does |
| Groq returns article_id-keyed JSON | No name-matching needed — avoids same-name variant confusion |
| `values_contradict()` + NLI two-gate system | String comparison catches mismatches; NLI filters false positives like "Utah" vs "Utah blouse" |
| Graph node values overwritten each turn | Latest DB query is always more reliable than a value stored 5 turns ago |
| Graph persisted across turns (MongoDB) | Enables detection of drift that spans multiple turns in the same session |
| Groq failure → graceful skip | System never crashes — user gets original (uncorrected) response |
| One graph per session_id | Sessions are fully isolated — user A's history never affects user B |
