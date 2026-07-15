# Real User Data Collection & Ensemble Majority Vote Labelling

## Overview

The DistilBERT intent classifier was initially trained on **52,028 synthetic samples** generated using a Python script with defined conversation structures and Claude Haiku. While the model achieves high accuracy on synthetic test data, synthetic samples cannot fully capture the diversity and unpredictability of real user messages in a live fashion CRS.

This document describes the approach used to collect real user turns from live sessions, identify uncertain predictions, assign correct labels via an ensemble majority vote, and use that corrected data to improve the model.

---

## The Problem

When DistilBERT classifies a real user message, it outputs a **confidence score** (0.0 – 1.0) alongside the predicted label. A low confidence score signals that the model was uncertain — the predicted label may be incorrect.

- **Confidence ≥ 0.90** → model is confident; label is likely correct
- **Confidence < 0.90** → model is uncertain; label needs verification

Out of 838 real classified user turns collected, **588 (70%)** had confidence below 0.90, indicating the synthetic training data leaves a significant gap when applied to real conversations.

---

## Solution: Real Data + Ensemble Majority Vote

### Step 1 — Extract Real User Turns from MongoDB

Script: `extract_real_data.py`

Reads all sessions from MongoDB (`sessions` collection), finds every user turn that has a stored classification, and writes a CSV matching the structure of the training data.

**Column structure** (same as `v4_train_midSession.csv` + `confidence`):

| Column | Description |
|--------|-------------|
| `input_text` | [SEP]-joined context string fed to DistilBERT |
| `current_message` | The raw user message |
| `conversation_history_json` | Prior turns as JSON array |
| `label` | Integer label (0–7) predicted by DistilBERT |
| `label_name` | e.g. `INITIAL_REQUEST` |
| `retrieval_strategy` | `FULL`, `PARTIAL`, or `NO` |
| `exchanges` | Number of prior turn segments in context |
| `confidence` | DistilBERT confidence score for predicted label |

**How `input_text` is built:**
- Get the 2 turns immediately before the current user turn in the session
- Format: `USER: [prev1] [SEP] BOT: [prev2] [SEP] CURRENT: [message]`
- If no prior turns exist: `CURRENT: [message]`

```bash
python extract_real_data.py
# Output: data/real_data_from_mongodb.csv
```

---

### Step 2 — Ensemble Majority Vote for Low-Confidence Rows

Script: `ensemble_label.py`

For every row where `confidence < 0.90`, the same `input_text` is sent to **3 independent free LLMs**. Each LLM returns one of the 8 intent labels. The label that receives the most votes becomes the `majority_vote`.

**Voters used:**

| Voter | Model | API Key in `.env` |
|-------|-------|-------------------|
| Google Gemini | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| Mistral AI | `mistral-small-latest` | `MISTRAL_API_KEY` |
| Ollama (local) | `llama3.2` | No key needed |

**Voting logic:**
- 3 voters → need at least **2 out of 3 to agree** for a valid majority vote
- If all 3 disagree → `majority_vote` is left empty (row excluded from retraining)
- If only 1–2 voters are available (missing API keys) → single/2-voter agreement accepted

**Why 3 independent LLMs?**
Using multiple models from different architectures reduces the chance of a systematic labelling error. A label that 2 or more models agree on independently is far more likely to be correct than a single model's output.

**New column added to CSV:**

| Column | Description |
|--------|-------------|
| `majority_vote` | Ensemble-agreed label (one of 8 label names), or empty if no majority |

```bash
# Add keys to .env first:
# GEMINI_API_KEY=your_key
# MISTRAL_API_KEY=your_key

python ensemble_label.py
# Updates: data/real_data_from_mongodb.csv  (adds majority_vote column)
```

The script saves a checkpoint every 25 rows. If interrupted, re-running it skips already-labelled rows.

---

### Step 3 — Filter Verified Real Data

After running the ensemble script, filter the CSV to keep only rows where a majority vote was obtained:

```python
import pandas as pd

df = pd.read_csv("data/real_data_from_mongodb.csv")

# Rows with a confirmed majority vote label
verified = df[df["majority_vote"].notna() & (df["majority_vote"] != "")]

# Use majority_vote as the corrected label for retraining
verified["label_name"] = verified["majority_vote"]
```

---

### Step 4 — Mix with Synthetic Data and Retrain

Mix the verified real data with the original synthetic training data. A higher mix ratio of real data improves generalisation to actual user inputs.

**Recommended mix strategy:**

```python
import pandas as pd

synthetic = pd.read_csv("data/v4_train_midSession.csv")
real      = pd.read_csv("data/real_data_from_mongodb.csv")

# Keep only rows with a confirmed majority vote
real_verified = real[real["majority_vote"].notna() & (real["majority_vote"] != "")].copy()
real_verified["label_name"] = real_verified["majority_vote"]

# Combine: e.g. 90% synthetic + 10% real
MIX_RATIO = 0.10
n_real = int(len(synthetic) * MIX_RATIO / (1 - MIX_RATIO))
real_sample = real_verified.sample(min(n_real, len(real_verified)), random_state=42)

combined = pd.concat([synthetic, real_sample], ignore_index=True).sample(frac=1, random_state=42)
combined.to_csv("data/v5_train_mixed.csv", index=False)

print(f"Synthetic rows : {len(synthetic)}")
print(f"Real rows      : {len(real_sample)}")
print(f"Total          : {len(combined)}")
```

Then retrain using `v5_train_mixed.csv` as the training file.

---

## Why Not Use Groq?

Groq is already integrated in the live prediction pipeline (`predict.py`) as the real-time judge that verifies every DistilBERT prediction. Using Groq again for ensemble labelling would mean the same system that made the original prediction is also verifying it — providing no independent signal. The three voters (Gemini, Mistral, Ollama) are all from different model families to ensure genuinely independent votes.

---

## Files

| File | Purpose |
|------|---------|
| `extract_real_data.py` | Extract all classified user turns from MongoDB into CSV |
| `ensemble_label.py` | Query 3 LLMs for low-confidence rows, write majority_vote |
| `data/real_data_from_mongodb.csv` | Output CSV with confidence + majority_vote columns |

---

## Quick Start

```bash
# 1. Extract real turns from MongoDB
python extract_real_data.py

# 2. Add free API keys to project .env
#    GEMINI_API_KEY=...
#    MISTRAL_API_KEY=...

# 3. Run ensemble labelling (safe to interrupt and resume)
python ensemble_label.py

# 4. Check results
python -c "
import pandas as pd
df = pd.read_csv('data/real_data_from_mongodb.csv')
print(df['majority_vote'].value_counts())
print('Labelled:', df['majority_vote'].notna().sum(), '/', len(df))
"
```
