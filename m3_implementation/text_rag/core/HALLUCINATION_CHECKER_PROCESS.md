# Hallucination Checker — Full Process Documentation

## Overview

The hallucination checker verifies that the LLM response does not contradict the
evidence used to generate it. It uses two models:

- **MiniLM (`all-MiniLM-L6-v2`)** — sentence embedding model for semantic similarity
- **DeBERTa (`cross-encoder/nli-deberta-v3-base`)** — NLI cross-encoder for contradiction detection

The checker is called after every LLM generation attempt. If a hallucination is
detected, the RAG pipeline retries with increased strictness (max 3 attempts).

---

## Model Roles — Why Two Models

The two models serve completely different purposes and neither can replace the other.

### MiniLM — Encoding and Similarity Only

MiniLM is a **bi-encoder**: it converts each piece of text independently into a numerical vector (embedding). Cosine similarity is then computed between those vectors as a standard math operation — MiniLM itself has no built-in similarity function.

```
text → MiniLM.encode() → vector → cosine_similarity() → score 0.0–1.0
```

MiniLM is used in two places:
- **Stage 3** — encodes all item descriptions and all sentences together, then cosine similarity assigns each item to its matching sentence (the lock map)
- **Gate 3** — encodes a single fact and all sentences, then cosine similarity picks the best matching sentence for that fact

**MiniLM cannot detect contradiction.** Cosine similarity only measures how close two texts are in meaning — it cannot tell whether one contradicts the other. For example:

- Fact: *"The item is Black in colour."*
- Sentence: *"This dress comes in Red."*

These score **high** on cosine similarity (both discuss colour) but the meaning is opposite. MiniLM would wrongly suggest they match.

### DeBERTa — Contradiction Detection

DeBERTa is a **cross-encoder**: it takes both the fact and the sentence together as a single input and attends across both simultaneously. This allows it to reason about the *relationship* between them — not just encode each independently.

`cross-encoder/nli-deberta-v3-base` classifies every (fact, sentence) pair as one of:
- **Entailment** — the sentence confirms the fact
- **Neutral** — the sentence does not mention the fact
- **Contradiction** — the sentence directly contradicts the fact

Only a **Contradiction** label with score above threshold triggers a hallucination flag.

### Why DeBERTa Specifically

**1. Cross-encoder architecture**
DeBERTa attends to both the fact and the sentence together, so it understands their relationship. A bi-encoder like MiniLM encodes each text separately and cannot reason about relationships between them.

**2. Fine-tuned on NLI benchmarks**
`cross-encoder/nli-deberta-v3-base` is pre-trained on large NLI datasets (MNLI, SNLI). It has already learned what contradiction, entailment, and neutral mean across thousands of real examples. No custom training data is needed.

**3. Disentangled attention**
DeBERTa separates **content** and **position** into different attention streams. This makes it significantly better than BERT or RoBERTa at detecting subtle semantic differences — exactly the case where two sentences both discuss the same attribute (e.g. colour) but one contradicts the other.

### Why Not Other Approaches

| Alternative | Problem |
|-------------|---------|
| Cosine similarity only | Measures closeness, not contradiction — opposite meanings can score high |
| Rule-based / regex | Cannot handle natural language variation — *"comes in red"* vs *"red coloured"* |
| LLM self-verification | Too slow, adds another API call per retry, expensive |
| BERT / RoBERTa NLI | Lower accuracy on subtle contradictions than DeBERTa's disentangled attention |

In summary: **MiniLM finds which sentence to check. DeBERTa decides whether that sentence contradicts the fact.**

---

## Input

| Parameter | Description |
|---|---|
| `response_text` | The LLM-generated response string |
| `evidence` | The evidence bundle used to generate the response (contains `action`, items, article, etc.) |

---

## Stage 1 — Split Response into Sentences

**Function:** `_split_sentences(text)`

The response text is split into individual sentences using two rules:

**Rule 1** — Split on `.` `!` `?` followed by whitespace:
```
"This is sentence one. This is sentence two."
                      ↑ split here
```

**Rule 2** — Split on newline immediately before `Option N:`:
```
"Here are your options:\nOption 1: ..."
                        ↑ split here
```
This handles catalog responses where the LLM writes an intro line without a
trailing period before the first option.

**Rule 3** — Discard any piece shorter than 15 characters.

**Example output (catalog_search, 4 items):**
```
s0: "Hello, I've got four great options for you."
s1: "Here's what I recommend:"
s2: "Option 1: Charlotte lowback bra, Black, £17.78, This strapless..."
s3: "Option 2: Charlotte lowback bra, White, £18.14, This strapless..."
s4: "Option 3: Eden cut padding, Dark Pink, £9.07, This lace bra..."
s5: "Option 4: Kelly Softbra 2pk, Dark Red, £14.31, These soft..."
```

---

## Stage 2 — Flatten Evidence into Checkable Facts

**Function:** `_flatten_evidence(evidence)`

Converts the evidence bundle into a flat list of `{ field, text, item_idx }` dicts.
Each dict represents one verifiable claim.

**Per action type:**

| Action | Facts generated |
|---|---|
| `catalog_search` | name, type, colour, price, pattern, index_group, section — per item, with `item_idx` |
| `item_detail_lookup` | name, type, colour, price, pattern, index_group, section — single item |
| `item_attribute_lookup` | same as above + `extracted_facts` key-value pairs |
| `item_compare` | "Option 1: ..." facts for item_a + "Option 2: ..." facts for item_b + comparison_facts |
| `explanation_generate` | single item facts + confirmed_matches + active prior_claims |

**Example facts for one catalog_search item:**
```
{ field: "name",    text: "The item is called Charlotte lowback bra.", item_idx: 0 }
{ field: "colour",  text: "Charlotte lowback bra is Black in colour.", item_idx: 0 }
{ field: "price",   text: "Charlotte lowback bra is priced at £17.78.", item_idx: 0 }
{ field: "type",    text: "Charlotte lowback bra is a Bra.", item_idx: 0 }
{ field: "pattern", text: "Charlotte lowback bra has a Solid pattern.", item_idx: 0 }
```

---

## Stage 3 — Build Item→Sentence Lock Map (catalog_search only)

**Function:** `_build_item_sentence_map(sentences, items)`

Only runs when `action == "catalog_search"`. Locks each evidence item to exactly
one LLM sentence before any fact checking begins. This prevents cross-item
collisions when multiple items share the same name (e.g. four "Stark wool coat"
variants).

### How it works

**Step 1 — Build rich description per item:**
```
item[0]: "Charlotte lowback bra  Black  17.78  Bra"
item[1]: "Charlotte lowback bra  White  18.14  Bra"
item[2]: "Eden cut padding  Dark Pink  9.07  Bra"
item[3]: "Kelly Softbra 2pk  Dark Red  14.31  Bra"
```
Uses: name + colour + price + type. Excludes section/index_group/garment_group
(LLM never mentions these).

**Step 2 — Encode all descriptions and sentences together with MiniLM:**
```
embeddings = MiniLM.encode([item_descs... + sentences...])
```

**Step 3 — Compute cosine similarity for every (item, sentence) pair:**
```
(sim=0.8105, item[0], sent[2])
(sim=0.8043, item[1], sent[2])
(sim=0.7900, item[1], sent[3])
(sim=0.7731, item[0], sent[3])
...
```

**Step 4 — Greedy assignment (highest similarity first, no reuse):**
```
0.8105  item[0] → sent[2]  ✓  both free         → ASSIGN
0.8043  item[1] → sent[2]  ✗  sent[2] taken      → SKIP
0.7900  item[1] → sent[3]  ✓  both free          → ASSIGN
0.7731  item[0] → sent[3]  ✗  item[0] assigned   → SKIP
...
```
Items with no pair above sim=0.30 are excluded from the map (LLM didn't mention them).

**Result:**
```
option_sentence_map = { 0: 2, 1: 3, 2: 4, 3: 5 }
```
item[0]→sent[2], item[1]→sent[3], item[2]→sent[4], item[3]→sent[5]

**Matrix print shows:**
- `◀ ASSIGNED` — the sentence this item was assigned to
- `◀ best-raw` — the raw highest score when it was lost to another item in greedy assignment

---

## Stage 4 — Check Each Fact (Main Loop)

For every fact, the following gates are evaluated **in order**. A fact exits at
the first gate that applies.

### Gate 1 — Field Filter (catalog_search only)

```python
_CATALOG_CORE_FIELDS = {"name", "colour", "price"}

if action == "catalog_search" and field not in _CATALOG_CORE_FIELDS:
    → SKIP
```

For catalog responses only `name`, `colour`, and `price` are checked.
`type`, `pattern`, `section`, `index_group` are excluded because:
- `pattern` ("Solid pattern") conflicts with LLM description text ("solid colour")
- `type` and `section` are never mentioned in LLM option sentences
- These fields caused systematic false positives across retry attempts

For all other actions (item_detail_lookup, item_compare, etc.) all fields are checked.

### Gate 2 — Unlocked Item Skip (catalog_search only)

```python
if option_sentence_map and fact_item not in option_sentence_map:
    → SKIP
```

If the item→sentence map was built but this item has no entry, the LLM didn't
mention that item in this response. Skipping prevents an unlocked item's facts
from matching a different item's sentence.

### Gate 3 — Find Best Sentence

**`_find_best_sentence(fact_text, sentences, option_sentence_map, item_idx)`**

- **With lock** (catalog_search): use the pre-assigned sentence from `option_sentence_map`.
  Compute sim(fact_text, locked_sentence) for the threshold gate only.
- **Without lock** (all other actions): MiniLM scores fact_text against every sentence,
  picks the highest similarity sentence.

### Gate 4 — Sentence Type Skip

**`_should_skip_sentence(sentence)`**

Skips sentences that are conversational or garment-description type:

- Conversational openers: "here are", "i hope", "you might", "feel free",
  "let me know", "would you", "thank you", etc.
- Garment description words: "waist", "pocket", "hem", "sleeve", "collar",
  "woven", "knit", "cotton", "slim", "fitted", etc.

**Exception:** any sentence containing `£` is never skipped — these are
Option sentences that must always be checked.

### Gates 6/7 — Two-Sided Exact-Value Verification (name & price)

**These gates run BEFORE the similarity gate and their facts NEVER reach NLI.**
Name and price are exact values: they are either present or they are not, so
string logic decides *both* directions. DeBERTa is unreliable in both
directions on exact values — it produces false contradictions on correct
values in structured sentences AND scores wrong values ("priced at £11.08"
vs "£13.58") as *neutral*, missing real hallucinations.

They also deliberately bypass Gate 5: a swapped name destroys the very
fact–sentence similarity that gate measures, so low similarity is itself a
symptom of this hallucination type — filtering on it would hide exactly the
cases that must be caught.

**Decision tree (name):**

```
true name in sentence (whitespace-normalised, case-insensitive)?
├── YES → PASS
├── item is LOCKED (catalog lock map):
│     the locked sentence is authoritative for this item —
│     a DIFFERENT product name here → CONTRADICTION (cross-item swap)
│     no name at all               → skip
└── item is UNLOCKED (compare / explanation / detail):
      true name elsewhere in the WHOLE response → skip (name correct,
        MiniLM free search merely paired the fact with another sentence)
      true name absent from the whole response AND a different name
        present → CONTRADICTION (the LLM renamed the item)
      no name anywhere → skip (nothing to contradict)
```

A "different product name" is found by `_find_wrong_name()`, in order:
1. names of **other items in the same evidence** (catches cross-item swaps),
2. the name slot of structured option sentences (`"Option N: <name>,"` —
   only when the sentence carries a £, and generic openers are excluded),
3. the full **catalog name list** from `sample_articles.csv` (case-sensitive,
   ≥5 chars). Names in a substring relation with the true name
   ("London dress" / "SS London dress") are treated as ambiguous
   truncations and never flagged.

**Decision tree (price):** identical shape — verbatim £value → PASS;
locked sentence containing a different £value → CONTRADICTION; unlocked:
true price anywhere in the response → skip, otherwise any other £value →
CONTRADICTION; no £value → skip.

Two implementation details that matter: all name comparisons are
**whitespace-normalised** (catalog names may contain double spaces, e.g.
`"Printed  tee 9.99"`, which used to defeat verbatim matching), and the
price regex is `£[\d,]+(?:\.\d{1,2})?` — the earlier `£[\d,.]+` swallowed
the sentence's trailing period (`"£11.08."`), so the verbatim pass rarely
fired and prices silently fell through to NLI.

Containment flags carry a synthetic score `contradiction = 1.0` and
`"method": "containment"` in the result dict.

### Gate 5 — Similarity Threshold (semantic fields only)

```python
_MIN_SIMILARITY = 0.35

if similarity < 0.35:
    → SKIP  (LLM did not mention this field)
```

If the best MiniLM similarity between the fact and any sentence is below 0.35,
the LLM did not mention this field at all. Nothing to contradict — skip.
Name and price facts never reach this gate (see Gates 6/7 above).

### Gate 8 — Duplicate Pair Skip

```python
checked_pairs = set()
if (fact_text, sentence) already in checked_pairs:
    → SKIP
```

Avoids running DeBERTa twice on the same (fact, sentence) combination.

### Gate 9 — DeBERTa NLI Check

Only `colour` and other non-name/price fields reach this gate.

```python
scores = deberta_model.predict([(fact_text, best_sentence)])
# scores[0] = [contradiction_logit, neutral_logit, entailment_logit]

is_hallucination = (
    contradiction > NLI_CONTRADICTION_THRESHOLD
    AND contradiction > entailment
)
```

**Why only contradiction, not low entailment:**
A sentence like "Red T-shirt at £45, stretch fabric" contains the correct colour
but also extra info (price, material) that the colour evidence alone cannot confirm.
DeBERTa would score entailment low → false positive. Only a direct contradiction
is flagged.

**Label mapping for `cross-encoder/nli-deberta-v3-base`:**
```
Label 0 = CONTRADICTION
Label 1 = NEUTRAL
Label 2 = ENTAILMENT
```

---

## Stage 5 — Final Result

```python
has_hallucination = len(flagged) > 0
hallucination_score = avg contradiction score of flagged facts

return {
    "has_hallucination":   bool,
    "hallucination_score": float,
    "flagged_sentences":   list of flagged fact dicts,
    "all_checks":          list of all checked fact dicts,
    "n_checked":           int,
    "n_flagged":           int,
    "passed":              bool,
    "contradicted_fields": list of field names that were contradicted,
}
```

---

## Retry Logic (in RAG Pipeline)

```
Attempt 1 → strictness=0 (friendly, descriptive)
    ↓ hallucination detected?
Attempt 2 → strictness=1 (accurate + friendly, copy values exactly)
    ↓ hallucination detected?
Attempt 3 → strictness=2 (bullet-only, minimal description)
    ↓ always accepted regardless of hallucination result
```

---

## Full Flow Diagram

```
response_text + evidence
        │
        ├─ Stage 1: _split_sentences()
        │     → [s0, s1, s2, ...]
        │
        ├─ Stage 2: _flatten_evidence()
        │     → [fact0, fact1, fact2, ...]
        │
        ├─ Stage 3: _build_item_sentence_map()  ← catalog_search only
        │     MiniLM similarity matrix
        │     Greedy assignment
        │     → { item_idx → sent_idx }
        │
        └─ Stage 4: for each fact:
              │
              ├─ Gate 1: field filter (catalog_search: name/colour/price only)
              ├─ Gate 2: unlocked item skip
              ├─ Gate 3: _find_best_sentence() — locked or free MiniLM search
              ├─ Gate 4: _should_skip_sentence() — conversational / garment desc
              ├─ Gates 6/7: name & price — TWO-SIDED value logic
              │             (bypass Gate 5, never reach NLI)
              │             → PASS / CONTRADICTION / skip
              ├─ Gate 5: similarity < 0.35 → skip (semantic fields only)
              ├─ Gate 8: duplicate pair skip
              └─ Gate 9: DeBERTa NLI (colour & other semantic fields)
                          → PASS or HALLUCINATION
                              │
                    ┌─────────┴──────────┐
                 all pass           any flagged
                    │                    │
              return passed      return has_hallucination=True
                                 → RAG pipeline retries
```

---

## Action Type Summary

| Action | Item lock | Field filter | Name containment | Price containment |
|---|---|---|---|---|
| `catalog_search` | MiniLM greedy | name, colour, price only | works | works |
| `item_detail_lookup` | not needed (1 item) | all fields | works | works |
| `item_attribute_lookup` | not needed (1 item) | all fields | works | works |
| `item_compare` | not needed (free search) | all fields | works (handles "Option 1:" prefix) | works |
| `explanation_generate` | not needed (1 item) | all fields | works | works |

---

## Novelty Assessment

Each individual component of this checker is established in the literature:
- NLI for hallucination detection is a known technique (Ji et al., 2023)
- DeBERTa cross-encoder for NLI is standard use of an existing model
- Retry/regeneration loops appear in various RAG systems

The genuine research contribution lies in the **system design applied to conversational recommendation**:

| Component | Contribution |
|---|---|
| Evidence-field-first loop | Most NLI hallucination work checks the full response against the full document. This system checks each evidence field (colour, price, name) individually against the specific sentence that mentions it |
| Item→sentence lock map | Greedy MiniLM assignment before NLI prevents cross-item collisions in multi-item catalog responses. No prior work addresses this specific problem for CRS |
| Contradiction-only, not entailment | Deliberate design decision to avoid false positives from descriptive extra information. Justified by the asymmetric cost of false positives in a recommender context |
| Gate filtering pipeline | Exact-value fields (name, price) verified by string logic, NLI reserved for semantic fields like colour where verbatim match is not sufficient |
| Two-sided exact-value gates | Value logic decides BOTH directions for name/price (present → pass, different value → contradiction), bypassing NLI entirely for exact values and bypassing the similarity gate whose signal a swapped value destroys. Evaluation-driven: raised recall on name/price corruptions from ~0.39 to 0.95/0.98 at precision 1.0 |
| Application domain | NLI hallucination checking evaluated within a fashion conversational recommender is not a prior existing system |

The framing for dissertation write-up: *"We apply and adapt NLI-based contradiction detection to the specific challenges of multi-item conversational recommendation, introducing an item-sentence locking mechanism and field-level contradiction checking to address cross-item collision and false positive problems that arise in structured catalog response formats."*

---

## Evaluation (completed)

The full evaluation lives in `m3_implementation/test_result/hallucination_result/`
(implementation: `HALLUCINATION_EVALUATION_PROCESS.md`, results: `RESULTS.md`,
loop experiment: `loop_mitigation/LOOP_RESULTS.md`, figures 1-8 under
`figures/`). Methodology: FactCC/HaluEval-style synthetic corruption of real
pipeline outputs — 238 labeled cases (33 clean + 205 corrupted), evaluated
against a naive all-pairs NLI baseline (SummaC-style) and an LLM-judge
baseline (RAGAS-style).

### Detection accuracy (positive class = hallucinated)

| System | Precision | Recall | F1 | Balanced acc. | False alarms (33 clean) |
|---|---|---|---|---|---|
| **This checker (v3)** | **1.000** | 0.951 | **0.975** | **0.976** | **0** |
| Naive NLI (no gates) | 0.872 | 0.961 | 0.914 | 0.541 | 29 |
| LLM judge (Groq) | 0.975 | 0.956 | 0.966 | 0.902 | 5 |

The evaluation was itself used to refine the checker (v1 -> v3 on the same
test set): the original one-sided containment gates scored recall 0.571
because DeBERTa misses exact-value mismatches; the two-sided gates plus the
similarity-gate bypass raised F1 from 0.727 to 0.975 at precision 1.000.

### Detect-reject-regenerate loop (induced-failure on/off experiment)

| | Hallucinated responses reaching the user |
|---|---|
| Loop OFF | 205 / 205 (100%) |
| **Loop ON** | **16 / 205 (7.8%)** |

P(correct final | detected) = 96.9%; 94.4% of detected hallucinations were
fixed by a single regeneration; average loop cost 0.41 s per detected case.
Final outputs were graded by an independent model-free referee, not the
checker itself.

---

## Reference Papers

- Kryscinski et al. (2020) — "Evaluating the Factual Consistency of Abstractive Text Summarization" (FactCC) — synthetic corruption methodology
- Li et al. (2023) — "HaluEval: A Large-Scale Hallucination Evaluation Benchmark" — injection-based evaluation
- Laban et al. (2022) — "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization" — NLI baseline design, balanced accuracy
- Manakul et al. (2023) — "SelfCheckGPT" — hallucination detection evaluation framing
- Es et al. (2023) — "RAGAS: Automated Evaluation of Retrieval Augmented Generation" — LLM-judge faithfulness baseline
- Yan et al. (2024) — "Corrective Retrieval Augmented Generation" (CRAG) — system on/off mitigation evaluation
- Ji et al. (2023) — "Survey of Hallucination in Natural Language Generation" — hallucination taxonomy
