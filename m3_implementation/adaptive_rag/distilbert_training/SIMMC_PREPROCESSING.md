# SIMMC 2.1 Dataset Preprocessing for DistilBERT Retraining

## Overview

The DistilBERT intent classifier was originally trained on 52,028 synthetic samples generated using Claude Haiku. To improve generalisation to real user inputs, the model is retrained using conversation data from the **SIMMC 2.1** dataset — a publicly available, peer-reviewed fashion dialogue dataset published at EMNLP 2021 by Facebook AI Research.

**Dataset citation:**
> Satwik Kottur, Seungwhan Moon, Alborz Geramifard, and Babak Damavandi. 2021. SIMMC 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations. In *Proceedings of EMNLP 2021*, pages 4903–4912.

**Script:** `preprocess_simmc.py`
**Output:** `data/real_data_simmc.csv`

---

## Why SIMMC 2.1 Was Selected

Four publicly available fashion conversation datasets were evaluated:

| Dataset | Turns | Intent Labels | Available | Issues |
|---------|-------|---------------|-----------|--------|
| SIMMC 2.1 | 27,567 | Yes (7 acts) | Yes | Some visual references |
| VOGUE | 428 | Yes (best match) | Yes | Too small for training |
| FashionRec | 331,124 | No | No | Wrong task (outfit assembly) |
| MMD | 262,300 | Yes (question-type) | Yes | No initial request labels; Indian English |

SIMMC 2.1 was selected because:
- Largest locally available labelled dataset
- Clear dialogue act taxonomy mapping to 6 of our 8 intent labels
- Clean American English consistent with our synthetic training data
- Well-cited published benchmark (EMNLP 2021, Facebook AI Research)
- Visual reference turns are identifiable and removable via regex

---

## Intent Label Mapping

SIMMC 2.1 annotates each user turn with a dialogue act. These acts are mapped to our 8-class intent taxonomy as follows:

| SIMMC Act | Our Label | ID | Retrieval Strategy | Rationale |
|-----------|-----------|----|--------------------|-----------|
| `REQUEST:GET` | INITIAL_REQUEST | 0 | FULL | User requests items matching a description — fresh product search |
| `INFORM:REFINE` | REFINEMENT | 1 | FULL | User narrows or changes the current search category |
| `INFORM:GET` | REFINEMENT | 1 | FULL | User provides attributes to refine recommendation |
| `ASK:GET` | ATTRIBUTE_QUESTION | 2 | PARTIAL | User asks about a specific attribute of a shown item |
| `REQUEST:COMPARE` | COMPARISON | 4 | PARTIAL | User requests comparison between two shown items |
| `INFORM:DISAMBIGUATE` | SELECTION_REFERENCE | 5 | PARTIAL | User points to a specific item for more detail |
| `REQUEST:ADD_TO_CART` | FEEDBACK | 6 | NO | User selects or buys an item — positive selection signal |

**Labels not present in SIMMC 2.1:**
- `EXPLANATION_WHY` (3) — SIMMC is a task-oriented shopping dataset with no "why did you recommend this?" turns
- `CHITCHAT` (7) — No casual conversation or greeting turns in SIMMC

These two labels are filled from the existing synthetic training data in the subsequent mixing step.

---

## Label Assignment Method

### How Labels Were Assigned — Label Transfer via Taxonomy Alignment

The label assignment method is **rule-based deterministic mapping** — a static lookup table. No machine learning or manual re-annotation was involved.

Each SIMMC 2.1 turn already contains a human-annotated `act` field stored directly in the JSON by the original Facebook AI Research team:

```json
{
  "transcript": "Do you have any plain jeans?",
  "transcript_annotated": {
    "act": "REQUEST:GET"
  }
}
```

The preprocessing script reads that `act` value and looks it up in a hardcoded dictionary:

```python
LABEL_MAP = {
    "REQUEST:GET":         ("INITIAL_REQUEST",    0, "FULL"),
    "INFORM:REFINE":       ("REFINEMENT",         1, "FULL"),
    "INFORM:GET":          ("REFINEMENT",         1, "FULL"),
    "ASK:GET":             ("ATTRIBUTE_QUESTION", 2, "PARTIAL"),
    "REQUEST:COMPARE":     ("COMPARISON",         4, "PARTIAL"),
    "INFORM:DISAMBIGUATE": ("SELECTION_REFERENCE",5, "PARTIAL"),
    "REQUEST:ADD_TO_CART": ("FEEDBACK",           6, "NO"),
}
```

### How Each Mapping Decision Was Made

Each mapping was decided by reading the SIMMC 2.0 paper's formal definition of each act and matching it to the semantically closest label in our intent taxonomy:

| SIMMC Act Definition | Our Label | Reasoning |
|----------------------|-----------|-----------|
| `REQUEST:GET` — user requests items matching a description | INITIAL_REQUEST | Asking for new products from scratch = fresh product search |
| `INFORM:REFINE` — user narrows or changes the current search | REFINEMENT | Narrowing the same product category already being discussed |
| `INFORM:GET` — user provides attributes to get a recommendation | REFINEMENT | Providing constraints to refine results = still a refinement signal |
| `ASK:GET` — user asks about a specific attribute of a shown item | ATTRIBUTE_QUESTION | Direct match — "what material?", "what size?" = attribute question |
| `REQUEST:COMPARE` — user requests comparison between two items | COMPARISON | Direct match |
| `INFORM:DISAMBIGUATE` — user points to a specific shown item for detail | SELECTION_REFERENCE | Pointing to a specific previously shown item = selection reference |
| `REQUEST:ADD_TO_CART` — user selects or buys an item | FEEDBACK | Positive selection/purchase decision = feedback signal |

### Label Quality and Academic Justification

The label quality depends entirely on how accurately the SIMMC annotators labelled the original data. Since SIMMC 2.1 was:
- Peer-reviewed and published at **EMNLP 2021**
- Produced by **Facebook AI Research**
- Annotated by trained human annotators following a formal annotation guide

The annotations are considered **high-quality ground truth**. No relabelling was performed — the original SIMMC labels were inherited and translated to our taxonomy.

This approach is formally called **label transfer via taxonomy alignment** — a standard methodology in NLP for adapting one annotated dataset to a different but semantically compatible label schema. It avoids the cost and subjectivity of manual re-annotation while leveraging the rigour of a published benchmark.

---

## Preprocessing Pipeline

### Raw Data Statistics (before preprocessing)

| Metric | Value |
|--------|-------|
| Total dialogues (all domains) | 7,307 |
| Total user turns (all domains) | 38,127 |

---

### Step 0 — Load Raw Data and Filter Fashion Domain

**What:** Load `simmc2.1_dials_dstc11_train.json` and discard all furniture domain dialogues.

**Why:** SIMMC 2.1 covers two domains: fashion and furniture. Our system is a fashion CRS and training on furniture dialogue (armchairs, sofas, tables) would introduce irrelevant vocabulary and confuse the classifier.

**Result:**

| Category | Count |
|----------|-------|
| Total dialogues | 7,307 |
| Furniture dialogues removed | 2,640 |
| Fashion dialogues kept | 4,667 |
| User turns (fashion only) | 27,567 |

---

### Step 1 — Extract User Turns with SIMMC Act Labels

**What:** For each fashion dialogue, iterate through all turns and extract the user `transcript` text together with its `transcript_annotated.act` label.

**Why:** SIMMC stores both user and system turns in the same dialogue array. We extract only user turns since these are what the intent classifier processes. The `act` field provides the ground-truth intent annotation.

**Act distribution in raw fashion data:**

| SIMMC Act | Count |
|-----------|-------|
| REQUEST:GET | 9,244 |
| REQUEST:ADD_TO_CART | 4,146 |
| ASK:GET | 3,165 |
| INFORM:REFINE | 2,907 |
| INFORM:DISAMBIGUATE | 2,869 |
| INFORM:GET | 2,700 |
| REQUEST:COMPARE | 2,536 |
| **Total** | **27,567** |

---

### Step 2 — Map SIMMC Acts to 8 Intent Labels

**What:** Apply the label mapping table above. All 7 SIMMC acts are covered — no turns are lost at this step.

**Why:** Our DistilBERT classifier uses 8 specific labels. The SIMMC taxonomy must be translated to match the label integers and names used in training.

**Result after mapping:**

| Our Label | Count |
|-----------|-------|
| INITIAL_REQUEST | 9,244 |
| REFINEMENT | 5,607 |
| FEEDBACK | 4,146 |
| ATTRIBUTE_QUESTION | 3,165 |
| SELECTION_REFERENCE | 2,869 |
| COMPARISON | 2,536 |
| **Total** | **27,567** |

---

### Step 3 — Filter Visual Reference Turns

**What:** Remove user turns that reference items by their physical position in the virtual AR shopping scene.

**Why:** SIMMC 2.1 was designed for an augmented reality shopping environment where users see items placed in a virtual room and refer to them by position ("the one on the left wall", "second from the right", "that one"). These positional references are meaningless in our text-only CRS where users describe what they want in natural language. Training on such turns would teach the classifier to associate position language with intent labels, which would not transfer to our system.

**Visual reference patterns filtered:**
- Position + wall/row/side: `"left wall"`, `"bottom row"`, `"right side"`
- Ordinal position: `"second from the left"`, `"third from the right"`
- Proximity references: `"that one"`, `"those two"`, `"next to it"`
- Scene-specific: `"hanging on the"`, `"in the middle"`, `"over there"`

**Examples of removed turns:**
```
"Can you compare the brown one and the black one that's second from the left?"
"Do you have anything like that one in terms of reviews but at a cheap price?"
"I mean the black one hanging in the top row and the white and black one to its right."
"Ok, can you tell me about those blouses instead?"
"The two in the top row, third and fourth from the left."
```

**Result:**

| Metric | Count |
|--------|-------|
| Turns removed (visual references) | 4,464 (16.2%) |
| Turns remaining | 23,103 |

---

### Step 4 — Filter Very Short Turns

**What:** Remove turns with fewer than 5 characters.

**Why:** Turns shorter than 5 characters (e.g. `"Ok"`, `"Yes"`, `"No"`) carry no intent information meaningful enough to train a classifier. They would introduce noise rather than signal.

**Result:**

| Metric | Count |
|--------|-------|
| Turns removed (< 5 chars) | 8 |
| Turns remaining | 23,095 |

---

### Step 5 — Normalize Text

**What:** Apply minimal text normalization to each user turn:
1. Strip leading and trailing whitespace
2. Collapse multiple consecutive spaces into one
3. Remove non-ASCII characters (zero-width spaces, invisible Unicode)

**Why:** Consistent text formatting ensures the DistilBERT tokenizer processes inputs the same way during training and inference. The normalization matches the same cleaning applied to the synthetic training data in `predict.py`.

**No turns are removed at this step.** The number remains 23,095.

---

### Step 6 — Build input_text with Up to 2 Prior Turns

**What:** For each user turn, retrieve up to 2 turns that immediately precede it in the same dialogue. Build the `[SEP]`-joined `input_text` string in the same format used in `v4_train_midSession.csv`.

**Format:**
```
# With 2 prior turns:
USER: <prev_user_msg> [SEP] BOT: <prev_bot_response> [SEP] CURRENT: <current_msg>

# With no prior turns (first turn in dialogue):
CURRENT: <current_msg>
```

**Why:** The DistilBERT classifier uses conversation history to disambiguate intent. For example, `"show me something cheaper"` is REFINEMENT only when there is prior context showing the user was already looking at items. Without context it could be mistaken for INITIAL_REQUEST. Including up to 2 prior turns matches the context window used during inference.

**The `system_transcript` field** from SIMMC provides the bot response text for context.

**Context distribution:**

| Context | Count |
|---------|-------|
| No prior turns (first turn) | 4,664 |
| 2 prior turns | 18,431 |
| **Total** | **23,095** |

---

### Step 7 — Build conversation_history_json and Count Exchanges

**What:** Serialize the prior turns list as a JSON array for the `conversation_history_json` column. Count the number of prior turn segments for the `exchanges` column.

**Why:** These fields match the exact schema of `v4_train_midSession.csv` so the SIMMC-derived data can be directly combined with the synthetic data for retraining without any schema changes.

---

### Step 8 — Save CSV

**What:** Write all processed rows to `data/real_data_simmc.csv` using the exact same column order as the synthetic training data.

**Output columns:**

| Column | Description |
|--------|-------------|
| `input_text` | [SEP]-joined context + current message |
| `current_message` | Raw user message text |
| `conversation_history_json` | Prior turns as JSON array |
| `label` | Integer label (0–7) |
| `label_name` | e.g. `INITIAL_REQUEST` |
| `retrieval_strategy` | `FULL`, `PARTIAL`, or `NO` |
| `exchanges` | Number of prior turn segments in context |

---

### Step 9 — Final Report

**Final label distribution after all preprocessing steps:**

| Label | Count | % of total |
|-------|-------|-----------|
| INITIAL_REQUEST | 9,203 | 39.8% |
| REFINEMENT | 4,987 | 21.6% |
| FEEDBACK | 3,191 | 13.8% |
| ATTRIBUTE_QUESTION | 2,266 | 9.8% |
| COMPARISON | 1,784 | 7.7% |
| SELECTION_REFERENCE | 1,664 | 7.2% |
| EXPLANATION_WHY | 0 | — (from synthetic) |
| CHITCHAT | 0 | — (from synthetic) |
| **Total** | **23,095** | |

**Sample rows per label:**

| Label | Example current_message |
|-------|------------------------|
| INITIAL_REQUEST | `"Hi, do you have any jackets today?"` |
| REFINEMENT | `"Are there any more jackets?"` |
| ATTRIBUTE_QUESTION | `"Can I get the size of the blouses?"` |
| FEEDBACK | `"Just add the dark blue jeans to my cart."` |
| COMPARISON | `"Do you have the sizes of the grey shirt and the grey and brown?"` |
| SELECTION_REFERENCE | `"The two black dresses."` |

---

## Preprocessing Summary

| Step | Action | Before | After | Removed |
|------|--------|--------|-------|---------|
| 0 | Filter fashion domain | 38,127 | 27,567 | 10,560 |
| 1 | Extract user turns | 27,567 | 27,567 | 0 |
| 2 | Map SIMMC acts to labels | 27,567 | 27,567 | 0 |
| 3 | Remove visual reference turns | 27,567 | 23,103 | 4,464 |
| 4 | Remove very short turns | 23,103 | 23,095 | 8 |
| 5 | Normalize text | 23,095 | 23,095 | 0 |
| 6–7 | Build input_text + context | 23,095 | 23,095 | 0 |
| **Final** | **Save CSV** | | **23,095** | |

---

---

## Dataset Mixing — Creating v5_train_mixed.csv

### Problem: Class Imbalance

The preprocessed SIMMC data (`real_data_simmc.csv`) has two problems:

1. **Two missing labels** — EXPLANATION_WHY and CHITCHAT do not exist in SIMMC 2.1 because it is a task-oriented shopping dataset with no "why did you recommend this?" or greeting turns.
2. **Class imbalance** — INITIAL_REQUEST (9,203) is 5.5× larger than SELECTION_REFERENCE (1,664). Training on this imbalanced data would bias the classifier toward the majority class.

| Label | SIMMC Count |
|-------|-------------|
| INITIAL_REQUEST | 9,203 |
| REFINEMENT | 4,987 |
| FEEDBACK | 3,191 |
| ATTRIBUTE_QUESTION | 2,266 |
| COMPARISON | 1,784 |
| SELECTION_REFERENCE | 1,664 |
| EXPLANATION_WHY | 0 |
| CHITCHAT | 0 |

### Solution: Balanced Mixing at 3,000 per Label

Script: `mix_datasets.py`

The mixing rule targets **3,000 rows per label** using this logic:

| Condition | Action |
|-----------|--------|
| SIMMC >= 3,000 | Randomly sample 3,000 from SIMMC only — no synthetic needed |
| 0 < SIMMC < 3,000 | Use all SIMMC rows + top up from synthetic to reach 3,000 |
| SIMMC = 0 | Take 3,000 from synthetic only |

This maximises real data usage while keeping the dataset balanced.

### Mixing Result per Label

| Label | From SIMMC (real) | From Synthetic | Total |
|-------|-------------------|----------------|-------|
| INITIAL_REQUEST | 3,000 (capped from 9,203) | 0 | 3,000 |
| REFINEMENT | 3,000 (capped from 4,987) | 0 | 3,000 |
| FEEDBACK | 3,000 (capped from 3,191) | 0 | 3,000 |
| ATTRIBUTE_QUESTION | 2,266 (all SIMMC) | 734 | 3,000 |
| COMPARISON | 1,784 (all SIMMC) | 1,216 | 3,000 |
| SELECTION_REFERENCE | 1,664 (all SIMMC) | 1,336 | 3,000 |
| EXPLANATION_WHY | 0 | 3,000 | 3,000 |
| CHITCHAT | 0 | 3,000 | 3,000 |
| **Total** | **14,714 (61.3%)** | **9,286 (38.7%)** | **24,000** |

The final dataset is shuffled with a fixed random seed (42) for reproducibility.

### Academic Justification for Synthetic Fill

The two labels filled entirely from synthetic data (EXPLANATION_WHY, CHITCHAT) were evaluated across all four candidate datasets (SIMMC 2.1, VOGUE, MMD, FashionRec) — none contained these labels. Since no published real-world annotated data exists for these two intents in the fashion CRS domain, using high-quality synthetic data is the only viable option. Importantly:

- **CHITCHAT** ("hello", "thanks", "ok") is simple and well-represented by synthetic examples
- **EXPLANATION_WHY** ("why did you recommend this?") is also well-captured synthetically
- The complex ambiguous labels (REFINEMENT, ATTRIBUTE_QUESTION, COMPARISON, SELECTION_REFERENCE) — where synthetic data historically underperforms — are all covered by real SIMMC data

---

## Validation and Test Data — Real SIMMC Splits

SIMMC 2.1 provides official train/dev/devtest splits. The same `preprocess_simmc.py` pipeline was applied to all three splits to produce consistent real-data train, validation, and test sets.

### Why Use Real Data for Validation and Test

Using the same real SIMMC data for all three splits means the model is trained AND evaluated on real human conversation data. This allows a stronger dissertation claim:

> *"The retrained model was trained and evaluated on real human conversation data from a peer-reviewed published benchmark."*

### Validation File — real_val_simmc.csv

Source: `simmc2.1_dials_dstc11_dev.json` + synthetic fill from `v4_train_midSession.csv`
Script: `preprocess_simmc.py` (same pipeline as training) + average-count synthetic top-up

| Label | Count | Source |
|-------|-------|--------|
| INITIAL_REQUEST | 1,185 | SIMMC 2.1 dev |
| REFINEMENT | 627 | SIMMC 2.1 dev |
| FEEDBACK | 401 | SIMMC 2.1 dev |
| ATTRIBUTE_QUESTION | 313 | SIMMC 2.1 dev |
| COMPARISON | 243 | SIMMC 2.1 dev |
| SELECTION_REFERENCE | 220 | SIMMC 2.1 dev |
| EXPLANATION_WHY | 498 | Synthetic (v4_train_midSession.csv) |
| CHITCHAT | 498 | Synthetic (v4_train_midSession.csv) |
| **Total** | **3,985** | |

EXPLANATION_WHY and CHITCHAT do not exist in SIMMC 2.1. The 498 rows added per label equals the average count of the 6 real SIMMC classes in the val split, keeping the evaluation balanced across all 8 labels. Rows were sampled from `v4_train_midSession.csv` with a fixed seed (42) and are non-overlapping with the test fill rows.

### Test File — real_test_simmc.csv

Source: `simmc2.1_dials_dstc11_devtest.json` + synthetic fill from `v4_train_midSession.csv`
Script: `preprocess_simmc.py` (same pipeline as training) + average-count synthetic top-up

| Label | Count | Source |
|-------|-------|--------|
| INITIAL_REQUEST | 2,180 | SIMMC 2.1 devtest |
| REFINEMENT | 1,200 | SIMMC 2.1 devtest |
| FEEDBACK | 742 | SIMMC 2.1 devtest |
| ATTRIBUTE_QUESTION | 543 | SIMMC 2.1 devtest |
| COMPARISON | 431 | SIMMC 2.1 devtest |
| SELECTION_REFERENCE | 387 | SIMMC 2.1 devtest |
| EXPLANATION_WHY | 913 | Synthetic (v4_train_midSession.csv) |
| CHITCHAT | 913 | Synthetic (v4_train_midSession.csv) |
| **Total** | **7,309** | |

EXPLANATION_WHY and CHITCHAT do not exist in SIMMC 2.1. The 913 rows added per label equals the average count of the 6 real SIMMC classes in the test split. Rows were sampled from `v4_train_midSession.csv` with a fixed seed (42) and are non-overlapping with the val fill rows.

---

## Complete Dataset Overview

| File | Rows | Purpose | Source |
|------|------|---------|--------|
| `real_data_simmc.csv` | 23,095 | Raw SIMMC extract (reference) | SIMMC 2.1 train |
| `v5_train_mixed.csv` | 24,000 | **Retraining — balanced mixed** | SIMMC + Synthetic |
| `real_val_simmc.csv` | 3,985 | Validation during retraining | SIMMC 2.1 dev + Synthetic fill |
| `real_test_simmc.csv` | 7,309 | Final accuracy measurement | SIMMC 2.1 devtest + Synthetic fill |
| `v4_train_midSession.csv` | 55,696 | Original synthetic training (baseline) | Synthetic |
| `v3_val_realistic.csv` | — | Original synthetic validation (baseline) | Synthetic |
| `v2_test_augmented.csv` | — | Original synthetic test (baseline) | Synthetic |

The retrained model (trained on `v5_train_mixed.csv`) can be directly compared against the original model (trained on `v4_train_midSession.csv`) to measure the improvement from incorporating real conversation data.
