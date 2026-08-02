# Contradiction Detector — Full Cross-Turn Workflow

## Overview

The contradiction detector ensures that every bot response is consistent with
the database evidence used to generate it — both within a single turn and
**across all previous turns of the same session**.

It uses three components together:

| Component | Role |
|---|---|
| **NetworkX DiGraph** | Session memory graph — one per session, persisted in MongoDB across turns |
| **Groq (llama-3.1-8b-instant)** | Extracts what attribute values the LLM actually wrote in the response |
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

## Attributes Tracked Per Product (graph node fields)

Every product node in the session graph stores these fields from DB evidence:

| Field | DB source key | Assembler summary key | Notes |
|---|---|---|---|
| `name` | `prod_name` | `name` | |
| `colour` | `colour_group_name` | `colour` | fallback tried in order |
| `price` | `avg_price` | `price` | formatted as `£XX.XX` |
| `product_type` | `product_type_name` | `type` | fallback tried in order |
| `pattern` | `graphical_appearance_name` | `pattern` | stored but **not checked** (see below) |
| `index_group` | `index_group_name` | `index_group` | |
| `section` | `section_name` | `section` | |
| `garment_group` | `garment_group_name` | `garment_group` | |

### Field naming — why two columns

The evidence assembler (`evidence_assembler.py`) uses shortened internal key
names in `_article_summary()` — e.g. `"type"` instead of `"product_type_name"`,
`"colour"` instead of `"colour_group_name"`. Context items stored in session
memory use the PostgreSQL column names. `_item_to_ref()` tries both:

```python
colour       = item.get("colour_group_name") or item.get("colour", "")
product_type = item.get("product_type_name") or item.get("type", "")
```

### `_CHECKABLE_FIELDS` — what is actually compared

Not all stored fields are checked. The tuple controls the comparison loop:

```python
_CHECKABLE_FIELDS = (
    "colour", "price", "name",
    "product_type", "index_group", "section", "garment_group",
)
```

**`pattern` is intentionally excluded.** The LLM describes product style/shape
using words like *"short"*, *"calf-length"*, *"one-shoulder design"* — Groq
reads these as the pattern value. The real pattern field (e.g. `"All over
pattern"`) is never written literally in responses, so checking it would produce
only false positives. Pattern is stored in the graph for completeness but never
compared.

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
        "name":          "SC COLUMBUS blouse",
        "colour":        "Black",
        "price":         "£57.14",
        "product_type":  "Blouse",
        "pattern":       "Solid",
        "index_group":   "Ladieswear",
        "section":       "Special Collections",
        "garment_group": "Unknown",
        "first_seen_turn": "turn_bd25beee",
        "last_seen_turn":  "turn_bd25beee",
        "session_id":      "sess_eea8c1bf"
      }
    ],
    "links": [
      {
        "source":    "733839003",
        "target":    "733839003_contra_colour_turn_xyz",
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
  {
    article_id:    "733839003",
    name:          "SC COLUMBUS blouse",
    colour:        "Black",
    price:         "£57.14",
    product_type:  "Blouse",
    pattern:       "Solid",
    index_group:   "Ladieswear",
    section:       "Special Collections",
    garment_group: "Unknown"
  },
  ...
]
```

All available fields are captured. Fields not present in the evidence item
default to empty string — empty evidence values are never checked
(`values_contradict()` returns False when evidence is empty).

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
all 8 field values are **overwritten** with the latest DB evidence. The DB is always right.

**If the product node is new**:
a new node is created with all 8 evidence fields plus metadata.

```
node "733839003" = {
    name:          "SC COLUMBUS blouse"
    colour:        "Black"
    price:         "£57.14"
    product_type:  "Blouse"
    pattern:       "Solid"
    index_group:   "Ladieswear"
    section:       "Special Collections"
    garment_group: "Unknown"
    first_seen_turn: "turn_bd25beee"
    last_seen_turn:  "turn_bd25beee"
    session_id:      "sess_eea8c1bf"
}
```

---

### Step 4 — Extract LLM claims via Groq

The Groq prompt includes the DB attribute values per product so Groq knows
what to look for, and instructs it to **only extract fields explicitly
mentioned in the response**:

```
Products shown in this response (with their correct database values):
- 733839003: SC COLUMBUS blouse
  [colour=Black, price=£57.14, product_type=Blouse, pattern=Solid,
   index_group=Ladieswear, section=Special Collections, garment_group=Unknown]

Bot response text:
...

Extract the values the bot wrote for each product.
IMPORTANT: Only extract a field if it is explicitly mentioned in the response.
```

Groq returns article_id-keyed JSON:

```json
{
  "733839003": { "name": "SC COLUMBUS blouse", "colour": "Black", "price": "£57.14", "product_type": "Blouse" },
  "733838002": { "name": "SC UTAH blouse",     "colour": "Light Beige", "price": "£70.59" }
}
```

**Groq only returns fields it found mentioned.** If the LLM didn't mention
`index_group`, Groq omits it — and the comparison loop skips it automatically.
This prevents false positives for unmentioned fields.

If Groq fails or returns invalid JSON → check is skipped gracefully.
The original response is returned to the user with no crash.

---

### Step 5 — Compare extracted claims vs graph node values

The comparison loop iterates over `_CHECKABLE_FIELDS` dynamically:

```python
for attribute in _CHECKABLE_FIELDS:        # colour, price, name, product_type, ...
    extracted_val = extracted_fields.get(attribute, "")
    if not extracted_val:
        continue    # Groq didn't find this field mentioned — skip

    evidence_val = node.get(attribute, "")

    if not values_contradict(evidence_val, extracted_val):
        continue    # values match — skip

    → CANDIDATE  (proceed to NLI confirmation)
```

**`values_contradict()` normalisation rules:**

- Both values lowercased, stripped, whitespace/hyphen-collapsed
- `£` whitespace normalised (`£ 57.14` → `£57.14`)
- Prices: float comparison with tolerance ±0.05 (avoids rounding false positives)
- Text: exact equality after normalisation
- Either value empty → always False (can't contradict what is unknown)

---

### Step 5b — NLI confirmation gate

`values_contradict()` is a strict string comparison — it catches surface
differences that are not real contradictions. DeBERTa NLI runs as a second
gate to confirm the mismatch is semantically contradictory.

The **premise** is built from all available node fields (richer context = better NLI scores):

```
premise = "The SC UTAH blouse is Light Beige and is a Blouse and costs £70.59."
```

The **hypothesis** is attribute-specific:

| Attribute | Hypothesis template |
|---|---|
| `colour` | `The {name} is {extracted_val} in colour.` |
| `price` | `The {name} costs {extracted_val}.` |
| `name` | `The product is called {extracted_val}.` |
| `product_type` | `The {name} is a {extracted_val}.` |
| `index_group` | `The {name} is from the {extracted_val} category.` |
| `section` | `The {name} belongs to the {extracted_val} section.` |
| `garment_group` | `The {name} is in the {extracted_val} garment group.` |

```
contra_score > 0.5  → CONFIRMED contradiction
contra_score ≤ 0.5  → false positive, skip
```

**Examples of NLI correctly rejecting false positives (from real logs):**

```
[CONTRA-CANDIDATE] 877274003 | name | evidence='Utah' extracted='Utah blouse'
[CONTRA] NLI not confirmed (score=-3.889) — skip
→ "Utah blouse" is not a contradiction of "Utah" — it's the same product

[CONTRA-CANDIDATE] 858407002 | product_type | evidence='Bra' extracted='sports bra'
[CONTRA] NLI not confirmed (score=-2.637) — skip
→ "sports bra" is a subtype of "Bra" — NLI sees entailment, not contradiction
```

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
        → normalised refs with all 8 fields per product

        ▼
Step 2: _load_graph(session_id)
        → NetworkX DiGraph from MongoDB
        → contains all products seen in prior turns of this session

        ▼
Step 3: _update_graph_nodes(graph, product_refs)
        → add new product nodes (all 8 fields)
        → overwrite existing nodes with latest DB values

        ▼
Step 4: _extract_claims_groq(response_text, product_refs)
        → prompt includes DB values for all fields per product
        → Groq extracts only fields explicitly mentioned in response
        → returns { article_id: { field: value, ... } }
        → failure → skip check, save graph, return original response

        ▼
Step 5: for each product in extracted claims:
          for each attribute in _CHECKABLE_FIELDS:
          │
          ├─ extracted_val missing → skip (field not mentioned)
          │
          ├─ values_contradict(node_value, extracted_value)?
          │       NO  → skip
          │       YES ↓
          │
          ├─ _confirm_with_nli(node_dict, extracted_val, attribute)
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

## Cross-Turn Example — 5-Item Catalog Then Detail Lookup

```
────────────────────────────────────────────────────────
TURN 1   "I need 5 bras"   →  catalog_search
────────────────────────────────────────────────────────
Evidence (5 items):
  858407002  Solo Assymetric bra   Black      Bra  £13.10
  640497002  ABBY bra              Other Pink Bra  £14.11
  644763002  Eden cut padding      Dark Pink  Bra  £9.07
  684087008  Kelly Softbra 2pk     Dark Red   Bra  £14.31
  704767004  Pumpkin bra           Dark Blue  Bra  £16.16

Graph before: nodes=7   (products from earlier session turns)
Graph after:  nodes=12  (5 bra nodes added)

Groq extracted (LLM used descriptive language for product_type):
  858407002 → colour=Black      price=£13.10  product_type="sports bra"
  640497002 → colour=Other Pink price=£14.11  product_type="fully lined sports top"
  ...

  product_type "Bra" ≠ "sports bra" → CANDIDATE for every product
  NLI scores: -2.637, -0.387, -2.686, -1.659, -2.127 → all ≤ 0.5 → all skipped

  colour and price: all 5 match exactly → no candidates

Result: found=False  NLI correctly filters LLM subtype descriptions.

────────────────────────────────────────────────────────
TURN 2   "tell me more about Kelly Softbra 2pk"  →  item_detail_lookup
────────────────────────────────────────────────────────
Evidence:  684087008 Kelly Softbra 2pk  Dark Red  Bra  £14.31

Graph before: nodes=12
Graph after:  nodes=12  (684087008 already exists — values refreshed)

Groq extracted:
  684087008 → colour=Dark Red  price=£14.31  product_type=Bra  pattern=Solid

  LLM wrote "* Type: Bra" in structured bullet format
  → Groq extracted exact DB value "Bra" this time
  → values_contradict("Bra", "Bra") = False → no candidate

Result: found=False  Clean pass.
```

---

## What Happens When a Contradiction IS Detected

```
Hypothetical Turn:  LLM drifts and writes "Navy" instead of "Black"
────────────────────────────────────────────────────────────────────
Graph node 733839003:  colour = "Black"   (stored from earlier turn)
Groq extracts:         colour = "Navy"

values_contradict("Black", "Navy") → True  → CANDIDATE

NLI:
  Premise:    "The SC COLUMBUS blouse is Black and is a Blouse and costs £57.14."
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

## Known Behaviours and Limitations

### product_type generates frequent false positive candidates in catalog_search

The LLM uses descriptive subtype language in catalog responses — *"This sports
bra has..."*, *"This lace bra offers..."*. Groq extracts `"sports bra"` or
`"lace bra"` as the `product_type` value. The DB stores `"Bra"`.
`values_contradict()` fires every time. NLI correctly rejects all of them
(a sports bra is a subtype of Bra — entailment, not contradiction).

In `item_detail_lookup` responses the LLM uses structured format (`* Type: Bra`)
so Groq extracts the exact DB value and no candidate is raised.

### pattern is stored but not checked

`pattern` is captured in graph nodes and shown to Groq in the prompt, but it
is excluded from `_CHECKABLE_FIELDS`. The LLM uses words like *"short"*,
*"calf-length"* or *"one-shoulder design"* which Groq reads as the pattern
value — these would produce only false positives against the real DB value
(e.g. `"All over pattern"`).

### Only 3 core attributes reliably detected across all action types

| Attribute | catalog_search | item_detail_lookup | explanation_generate |
|---|---|---|---|
| colour | reliable | reliable | reliable |
| price | reliable | reliable | reliable |
| name | reliable (NLI filters "X" vs "X skirt") | reliable | reliable |
| product_type | noisy (subtype language) | reliable (structured format) | reliable |
| index_group / section / garment_group | rarely mentioned by LLM | sometimes | rarely |

---

## Design Decisions

| Decision | Reason |
|---|---|
| DB evidence is always truth | PostgreSQL / Qdrant never hallucinate — only the LLM does |
| Groq returns article_id-keyed JSON | No name-matching needed — avoids same-name variant confusion (e.g. two bra variants) |
| Groq prompt includes DB values per field | Groq knows what to look for; extracts only explicitly mentioned fields |
| `_CHECKABLE_FIELDS` controls what is compared | Pattern excluded — LLM description words cause only false positives |
| `values_contradict()` + NLI two-gate system | String comparison catches mismatches; NLI filters false positives like "Bra" vs "sports bra" |
| NLI premise built from all node fields | Richer context improves NLI accuracy vs single-field premise |
| Graph node values overwritten each turn | Latest DB query is always more reliable than a value stored 5 turns ago |
| Graph persisted across turns (MongoDB) | Enables detection of drift that spans multiple turns in the same session |
| Groq failure → graceful skip | System never crashes — user gets original (uncorrected) response |
| One graph per session_id | Sessions are fully isolated — user A's history never affects user B |

---

## Evaluation Details

The contradiction detector was evaluated as a **classifier** (does the response
contradict an earlier turn?) on a labelled set built by synthetic cross-turn
corruption: 188 clean factual turns captured from 37 live multi-turn sessions,
each corrupted by changing one attribute value in a later turn, plus benign
subtype paraphrases ("Dress" -> "maxi dress") as hard negatives. Headline numbers
are on a stratified **599-case sample (450 contradictions, 149 negatives)**.

### Baselines (chosen to cover the established approaches)

| Baseline | What it is | Basis |
|---|---|---|
| **String-only (-NLI)** | our pipeline with the DeBERTa NLI gate removed - string comparison decides alone | ablation (isolates the NLI gate's contribution) |
| **History-NLI (unstructured)** | DeBERTa NLI over every (session fact, response sentence) pair | SummaC-style (Laban et al., TACL 2022) |
| **Utterance-pair NLI (structured)** | NLI on (fact about product X, sentence mentioning X) pairs | DECODE (Nie et al., ACL 2021) |
| **LLM judge** | Groq judges directly whether the response contradicts the session facts | LLM-as-a-judge (RAGAS / CoRE style) |

The four baselines represent the three main schools of contradiction detection
(unstructured NLI, structured NLI, LLM-as-judge) plus an ablation of our own
system - so a win over them shows the graph + extraction + NLI-gate architecture
adds value across the board, not just against a single weak baseline.

### Detection results (599-case sample, positive = contradiction)

| System | Precision | Recall | F1 | Balanced acc. |
|---|---|---|---|---|
| **Ours (graph + NLI)** | **0.98** | **0.90** | **0.94** | **0.92** |
| String-only (-NLI) | 0.84 | 0.86 | 0.85 | 0.69 |
| History-NLI | 0.76 | 0.96 | 0.85 | 0.53 |
| Utterance-pair NLI | 0.77 | 0.91 | 0.84 | 0.55 |
| LLM judge | 0.96 | 0.85 | 0.90 | 0.87 |

Ours has the **highest precision (0.98)** and the **best balanced accuracy (0.92)** -
ahead of the LLM judge. The two NLI-over-text baselines reach high recall
(0.91-0.96) only by falsely flagging 66-74 of the 84 clean responses, collapsing
their balanced accuracy to ~0.53-0.55 (near chance). Removing the NLI gate
(String-only) raises false alarms sharply and drops balanced accuracy 0.92 -> 0.69,
showing the gate is the core of the two-stage design.

### Correction results (system ON vs OFF, independent referee)

| Metric | Value |
|---|---|
| User-facing contradiction rate, detector OFF | 100% (450/450) |
| User-facing contradiction rate, detector ON | **11.8% (53/450)** |
| Detection rate | 0.90 (405/450) |
| P(correct fix given detected) | 0.98 (397/405) |

Turning the detector on cuts user-facing contradictions from 100% to **11.8%**,
and when it detects one it produces a correct, consistent response **98%** of the
time. Figures: `figures/figc1.png` (detection), `figures/figc5.png` (correction).

> Note: for the evaluation only, claim extraction ran on Llama 4 Scout (not the
> production llama-3.1-8b-instant) to avoid a free-tier rate limit; the detection
> logic is unchanged.
