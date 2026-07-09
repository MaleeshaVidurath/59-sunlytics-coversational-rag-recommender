# Synthetic Training Data Generation — User Personas & Process

## Overview

The DistilBERT 8-class intent classifier was initially trained on **52,028 synthetic samples** generated using **Claude Haiku** (`claude-haiku-4-5`). Synthetic data was chosen because no single publicly available dataset covers all 8 intent classes in the fashion CRS domain — particularly `EXPLANATION_WHY` and `CHITCHAT`.

**Script:** `generate_training_data.py`
**Output:** `data/v4_train_midSession.csv`

---

## Why Synthetic Data

- No existing labelled dataset covers all 8 intent classes in a fashion conversational recommender context
- `EXPLANATION_WHY` ("why did you recommend this?") and `CHITCHAT` (greetings, farewells) do not appear in any evaluated real-world fashion dataset (SIMMC 2.1, VOGUE, MMD, FashionRec)
- LLM-assisted generation allows precise control over label boundaries, linguistic diversity, and production input format

---

## Step 1 — Six User Personas

Each generated sample is written in the voice of one of six user personas. Personas are rotated uniformly across all 8 intent classes so the classifier learns to handle linguistic diversity, not just canonical phrasing.

The 6 personas represent distinct real segments of e-commerce users:

### 1. Casual Texter
**Description:** Young adult who texts casually. Uses abbreviations, no capitalisation, drops punctuation, very short messages.

**Style rules:**
- Use lowercase throughout
- Abbreviate where natural: `u` for `you`, `thx` for `thanks`, `pls` for `please`, `ngl` for `not gonna lie`
- Drop end punctuation
- Keep messages under 8 words where possible
- Use fragments: `cheaper?`, `first one?`, `why tho`

**Examples:** `need smth cheaper` · `which would u pick` · `first one ngl`

---

### 2. Busy Professional
**Description:** Adult in a hurry. Direct, minimal words, task-focused, no small talk. Still grammatically correct but very brief.

**Style rules:**
- Short, direct sentences
- No pleasantries
- Imperative mood: `Show me`, `Find`, `Give me`
- May include specific constraints: price, colour, occasion
- No filler words

**Examples:** `Show me black options under £40` · `Need something for a meeting` · `Which is better quality`

---

### 3. Teenager
**Description:** Teenager shopping for fashion. Uses slang, expressive, asks short rhetorical questions, dramatic reactions.

**Style rules:**
- Use teen slang: `omg`, `literally`, `lowkey`, `slay`, `vibe`
- Expressive punctuation: `!!` but also no punctuation at all
- Short questions: `wait why tho?`, `is it cute tho`
- Mix capitalisation inconsistently

**Examples:** `omg which one tho` · `lowkey need smth cheaper` · `first one slay`

---

### 4. Non-Native Speaker
**Description:** User whose first language is not English. Simplified grammar, dropped articles, direct word-for-word phrasing.

**Style rules:**
- Drop articles: `show me black dress` not `show me a black dress`
- Simple verb forms: `I want` not `I would like`
- May repeat question with `yes?` or `no?`
- Avoid idioms, use literal phrasing
- Short sentences with simple vocabulary

**Examples:** `show me black dress` · `I want more cheap` · `is good quality yes?`

---

### 5. Polite Shopper
**Description:** Considerate adult who uses full sentences, please/thank you, and polite question forms. Formal register.

**Style rules:**
- Full sentences with proper capitalisation and punctuation
- Use `Could you`, `Would you mind`, `Please`
- Express appreciation: `Thank you`, `That would be lovely`
- Explain reasoning: `I was thinking something more...`

**Examples:** `Could you show me something a little less expensive, please?` · `I was wondering whether it comes in other colours.`

---

### 6. Indecisive Browser
**Description:** Uncertain shopper who hedges, asks comparative questions, and expresses doubt.

**Style rules:**
- Hedging language: `maybe`, `not sure`, `I think`, `hmm`
- Comparative questions: `which is better?`, `what do you think?`
- Backtracking: `actually`, `on second thought`
- Vague preferences: `something nicer`, `a bit different`

**Examples:** `hmm not sure, maybe something cheaper?` · `actually, can you show me something different`

---

## Step 2 — Product Catalogue Grounding

Bot responses in the generated conversations always reference **real H&M product names, colours, and prices** drawn from the 41,794-item product catalogue. This grounds the conversation in authentic domain content rather than invented placeholders, making the classifier learn from realistic product dialogue.

Example bot response used in prompts:
> *"Option 1 is the Spring Wrap dress (black, dress): Wrap dress in woven fabric with a V-neck, £34.99. Option 2 is the Carnival Shift dress (red, dress): Knee-length shift dress, £29.99."*

---

## Step 3 — Prompt Templates Per Intent Class

Each of the 8 intent classes uses a structured generation prompt that specifies:
- **(a)** the intent label and its retrieval meaning
- **(b)** the required output format (`current_message` + `input_text` as JSON array)
- **(c)** the active persona's style rules
- **(d)** boundary constraints to prevent class overlap

### Key boundary constraints enforced in prompts

| Boundary | Rule |
|----------|------|
| REFINEMENT vs FEEDBACK | REFINEMENT must imply a new/different suggestion is wanted. FEEDBACK is a pure reaction with no new request. |
| INITIAL_REQUEST vs REFINEMENT | Mid-session INITIAL_REQUEST must switch product category entirely — not refine the existing one. |
| SELECTION_REFERENCE | Must reference `first`, `second`, `option 1/2`, or a specific product name/colour — not a vague pointer. |
| CHITCHAT | 65% standalone (no prior context), 35% mid-conversation remarks. |

---

## Step 4 — Generation Loop

- **Model:** Claude Haiku (`claude-haiku-4-5-20251001`)
- **Batch size:** 10 samples per API call
- **Persona rotation:** uniform across all samples per class
- **Random seed:** 42 (reproducibility)

### Target raw counts per class

| Label | Raw Generated | After Balancing |
|-------|:-------------:|:---------------:|
| INITIAL_REQUEST | 11,440 | 7,772 |
| REFINEMENT | 7,772 | 7,772 |
| ATTRIBUTE_QUESTION | 5,944 | 5,944 |
| EXPLANATION_WHY | 5,880 | 5,880 |
| COMPARISON | 5,882 | 5,882 |
| SELECTION_REFERENCE | 5,951 | 5,951 |
| FEEDBACK | 6,461 | 6,461 |
| CHITCHAT | 6,366 | 6,366 |
| **Total** | **55,696** | **52,028** |

**Why INITIAL_REQUEST is over-generated (11,440 raw):**
47% of INITIAL_REQUEST samples are deliberately generated as mid-session examples — where the user switches product category mid-conversation (e.g. from dresses to shoes). This fixes a production misclassification where `"I want a coat"` (said after a dress conversation) was predicted as REFINEMENT instead of INITIAL_REQUEST.

**Balancing:** INITIAL_REQUEST is downsampled from 11,440 to 7,772 at training time (matching REFINEMENT, the largest other class) to prevent classifier bias toward the majority class.

---

## Output Format

Every generated sample matches the exact production input format used during live inference:

```
# With prior conversation context:
USER: <prev_user_msg> [SEP] BOT: <bot_response> [SEP] CURRENT: <current_message>

# First turn (no prior context):
CURRENT: <current_message>
```

This ensures training and inference formats are identical — the classifier sees the same input structure in training as it does when deployed.
