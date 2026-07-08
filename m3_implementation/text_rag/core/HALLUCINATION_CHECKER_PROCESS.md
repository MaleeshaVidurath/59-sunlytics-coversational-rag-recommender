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

### Gate 5 — Similarity Threshold

```python
_MIN_SIMILARITY = 0.35

if similarity < 0.35:
    → SKIP  (LLM did not mention this field)
```

If the best MiniLM similarity between the fact and any sentence is below 0.35,
the LLM did not mention this field at all. Nothing to contradict — skip.

### Gate 6 — Name Containment Check

```python
if fact_field == "name":
    extract name from: "The item is called X."  (works with "Option 1: The item..." prefix too)
    if name found verbatim (case-insensitive) in sentence:
        → PASS immediately, no NLI
```

If the product name appears verbatim in the sentence it is definitively correct.
Bypasses DeBERTa which produces false contradictions when multiple items share
a name or when sentences have long descriptive tails.

### Gate 7 — Price Containment Check

```python
if fact_field == "price":
    extract £value from fact_text using regex
    if £value found verbatim in sentence:
        → PASS immediately, no NLI
```

Same reasoning as name: a price value either appears in the sentence or it doesn't.
DeBERTa was producing false contradictions on correct prices in structured list
format responses.

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
              ├─ Gate 5: similarity < 0.35 → skip (field not mentioned)
              ├─ Gate 6: name verbatim containment → PASS
              ├─ Gate 7: price verbatim containment → PASS
              ├─ Gate 8: duplicate pair skip
              └─ Gate 9: DeBERTa NLI → PASS or HALLUCINATION
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
| 9-gate filtering pipeline | Name and price verified by string containment (exact values), NLI reserved for semantic fields like colour where verbatim match is not sufficient |
| Application domain | NLI hallucination checking evaluated within a fashion conversational recommender is not a prior existing system |

The framing for dissertation write-up: *"We apply and adapt NLI-based contradiction detection to the specific challenges of multi-item conversational recommendation, introducing an item-sentence locking mechanism and field-level contradiction checking to address cross-item collision and false positive problems that arise in structured catalog response formats."*

---

## Evaluation Plan

### 1. Hallucination Detection Accuracy

Requires ground-truth labels. Two collection methods:

**Method A — Synthetic injection (recommended):**
1. Collect real system responses that the checker passed as correct
2. Manually corrupt one field per response (e.g. change "Black" → "Red" in the LLM response while keeping evidence as Black)
3. Run the checker on both the clean and the corrupted version
4. Record whether the injection was detected

**Method B — Manual annotation:**
1. Run 50–100 real conversations through the full system
2. Manually compare each LLM response against its evidence bundle
3. Label each checked fact as correct or hallucinated
4. Compare labels with checker decisions

**Metrics:**

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

Where:
- TP = checker flagged hallucination, fact was genuinely wrong
- FP = checker flagged hallucination, fact was actually correct (false alarm)
- FN = checker passed fact, fact was actually wrong (missed hallucination)
- TN = checker passed fact, fact was genuinely correct

---

### 2. Loop Behaviour Metrics

Measurable directly from runtime logs — no labeling required.

| Metric | Definition | How to measure |
|---|---|---|
| First-pass acceptance rate | % of responses that pass hallucination check on attempt 1 | Count attempt=1 passes / total responses |
| Retry rate | % of responses that required at least one retry | Count responses with attempt >= 2 / total |
| Convergence rate | % of responses that passed by attempt 3 | Count final passes / total (should be ~100% since attempt 3 is always accepted) |
| Average attempts per response | Mean number of LLM calls needed | Sum of attempt numbers / total responses |
| Field contradiction frequency | Which evidence fields are most often contradicted | Aggregate `contradicted_fields` across all flagged results |
| Strictness escalation success rate | % of retried responses that pass on attempt 2 | Count attempt=2 passes / total retries |

These metrics characterise the detect-reject-regenerate loop as a system, independent of whether individual flags are correct.

---

### 3. False Positive Rate

A checker that flags everything achieves 100% recall but is not useful. False positive rate must be measured separately.

**Procedure:**
1. Collect 30–50 LLM responses that are manually verified as fully correct
2. Run the checker on each
3. Count how many are incorrectly flagged

```
False Positive Rate (FPR) = FP / (FP + TN)
```

The 9-gate design (Gates 4–7) is specifically engineered to suppress false positives from conversational text, garment descriptions, and exact-value fields. This evaluation validates that design decision.

---

### 4. Strictness Escalation Effectiveness

The retry loop escalates strictness across three attempts. This evaluation checks whether escalation actually causes the LLM to correct the detected contradiction.

**Procedure:**
1. Collect all cases where attempt 1 was rejected
2. Compare attempt 2 response: did the contradicted field change to match evidence?
3. Repeat for attempt 2 → attempt 3

**Metric:**
```
P(correction | retry) = % of retried responses where the flagged field was corrected
```

A high value confirms that the strictness prompt works. A low value would indicate the retry prompt needs revision.

---

### 5. NLI Threshold Sensitivity

The checker uses `NLI_CONTRADICTION_THRESHOLD` (config value) to decide what contradiction score counts as a hallucination. This threshold is a hyperparameter and must be justified.

**Procedure:**
1. Run the checker over the labelled test set at multiple threshold values (e.g. 0.4, 0.5, 0.6, 0.7, 0.8)
2. Compute Precision and Recall at each threshold
3. Plot the Precision-Recall curve

The operating threshold should be chosen at the point of best F1 or at the target Precision depending on acceptable false positive cost. This also shows the threshold choice is data-driven, not arbitrary.

---

### 6. Ablation Study

Remove one design component at a time and measure the change in false positive rate and detection accuracy. This validates the individual contribution of each gate.

| Ablation | What is removed | Expected effect if component is effective |
|---|---|---|
| No item→sentence lock (Stage 3) | All facts use free MiniLM search | Increased FP from cross-item name collisions in catalog responses |
| No Gate 6 — name containment | Name facts go to DeBERTa | Increased FP when items share name prefixes or sentences have long descriptive tails |
| No Gate 7 — price containment | Price facts go to DeBERTa | Increased FP on correctly priced structured list responses |
| No Gate 4 — sentence type skip | Conversational openers and garment descriptions are NLI-checked | Increased FP from description words misread as contradictions |
| No contradiction-only filter | Low entailment also triggers hallucination flag | Very high FP from descriptive extra information in LLM sentences |
| Full system | All gates active | Baseline |

Each ablation is run on the same labelled test set. The difference in FP rate between the ablated version and the full system quantifies the contribution of that component.

---

### Recommended Minimum Evaluation for Dissertation

Given time constraints, prioritise in this order:

| Priority | Evaluation | Why |
|---|---|---|
| 1 | Synthetic injection test (50 cases) | Directly measures Precision, Recall, F1 — core claim of the checker |
| 2 | Loop metrics from 50 real conversations | Retry rate, convergence rate — characterises the detect-reject-regenerate loop |
| 3 | False positive check on 30 clean responses | Validates the 9-gate false positive suppression design |
| 4 | Ablation on item→sentence locking | Validates the most novel architectural component |
| 5 | Threshold sensitivity curve | Justifies the NLI threshold hyperparameter choice |

Items 1–3 together provide detection accuracy, loop behaviour, and false positive validation — sufficient for a complete evaluation chapter. Items 4 and 5 strengthen the research contribution claims.

---

### Reference Papers for Evaluation Section

- Ji et al. (2023) — "Survey of Hallucination in Natural Language Generation" — hallucination taxonomy and evaluation methods
- Maynez et al. (2020) — "On Faithfulness and Factuality in Abstractive Summarization" — faithfulness evaluation
- He et al. (2023) — "HaluEval: A Large-Scale Hallucination Evaluation Benchmark" — injection-based evaluation methodology
- Es et al. (2023) — "RAGAS: Automated Evaluation of Retrieval Augmented Generation" — RAG-specific evaluation framework
- Laurer et al. (2022) — "Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer of Pre-trained Language Models" — NLI cross-encoder benchmarks including DeBERTa variants
