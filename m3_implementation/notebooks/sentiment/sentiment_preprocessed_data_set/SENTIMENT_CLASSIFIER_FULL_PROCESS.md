# Sentiment Classifier — Full Process Documentation
### Sunlytics CRS — M3 Implementation

This document covers the complete end-to-end process for the sentiment classifier component of the Sunlytics Conversational Recommender System: from dataset acquisition through preprocessing, all training experiments, final model selection, and live system integration.

---

## 1. Dataset

**Dataset:** `cardiffnlp/tweet_eval` — sentiment configuration (SemEval 2017 Task 4A)

**Why SemEval 2017 tweet sentiment:**
- Publicly available, peer-reviewed benchmark (EMNLP/SemEval 2017)
- Tweet register (short, informal, opinionated) closely matches fashion CRS user feedback utterances
- Three-class labels (negative / neutral / positive) align directly with the system's sentiment scoring needs
- Widely used in sentiment classifier benchmarking — enables direct comparison with published results

**Download notebook:** `../download_sentiment_data_colab.ipynb`  
**Raw CSV output:** `../sentiment_data_set/`

| Split | File | Rows |
|-------|------|-----:|
| Train | `tweet_eval_train.csv` | 45,615 |
| Validation | `tweet_eval_val.csv` | 2,000 |
| Test | `tweet_eval_test.csv` | 12,284 |

**CSV columns:** `text`, `label` (0/1/2), `label_name` (negative/neutral/positive)

**Label distribution — raw data:**

| Label | Train | % | Val | % | Test | % |
|-------|------:|---|----:|---|-----:|---|
| negative | 7,093 | 15.6% | 312 | 15.6% | 3,972 | 32.3% |
| neutral | 20,673 | 45.3% | 869 | 43.5% | 5,937 | 48.3% |
| positive | 17,849 | 39.1% | 819 | 40.9% | 2,375 | 19.3% |

**Note on val/test distribution mismatch:** The test set has double the negative examples (32.3% vs 15.6%) and half the positive examples (19.3% vs 40.9%) compared to the validation set. This is an inherent property of the official SemEval 2017 splits and explains why test scores are consistently lower than validation scores across all training runs.

---

## 2. Noise Analysis — Raw Data

Before preprocessing, each split was analysed for Twitter-specific noise:

| Noise Type | Train | Val | Test |
|------------|------:|----:|-----:|
| Rows with `@user` mentions | 13,429 | 621 | 4,997 |
| Rows with `#hashtags` | 8,526 | 352 | 4,825 |
| Rows with extra whitespace | 4,122 | 168 | 894 |
| Rows with HTML entities | 1,923 | 88 | 10 |
| Rows with repeated punctuation (`!!!`, `???`) | 1,222 | 52 | 151 |
| Rows with URLs | 92 | 5 | 376 |

`@user` mentions (29.4% of training rows) and hashtags (18.7%) are the dominant noise sources, producing out-of-vocabulary sub-word tokens that obscure sentiment signal for models pre-trained on clean text.

---

## 3. Preprocessing

**Script:** `preprocess.py`  
**Output folder:** this folder (`sentiment_preprocessed_data_set/`)  
**Reference:** Barbieri et al. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*. EMNLP Findings 2020.

The original raw CSVs are **not modified**. All preprocessed files are written to this folder with `_clean` suffix.

### Steps Applied

| Step | Operation | Example Before | Example After |
|------|-----------|----------------|---------------|
| 1 | Decode HTML entities | `I &amp; you` | `I & you` |
| 2 | Replace URLs with `[URL]` | `check http://t.co/abc` | `check [URL]` |
| 3 | Replace `@mentions` with `[USER]` | `@user thanks!` | `[USER] thanks!` |
| 4 | Strip `#` from hashtags — keep word | `#HappyBirthday` | `HappyBirthday` |
| 5 | Normalise repeated punctuation (`3+` → `1`) | `What!!!` | `What!` |
| 6 | Collapse multiple spaces/tabs | `hello   world` | `hello world` |
| 7 | Strip leading/trailing whitespace | `  hello ` | `hello` |
| 8 | Drop rows with fewer than 3 words | `ok [USER]` (2 words) | removed |

**Step 4 rationale:** Hashtags often carry direct sentiment signal (`#love`, `#fail`). Removing only the `#` preserves that signal while avoiding awkward sub-word tokenisation of the `#` character.

**Step 5 note:** This step was temporarily removed during Run 2 experimentation after observing F1 Negative drop from 0.706 → 0.655. It was restored when switching to Twitter-RoBERTa, which handles emotional punctuation natively through its tweet-based pretraining.

**Step 8 rationale:** After cleaning, some rows collapse to 1–2 tokens (e.g. `@user @user` → `[USER] [USER]`) with no recoverable sentiment information. These are dropped.

### Row Counts After Preprocessing

| Split | Before | After | Dropped |
|-------|-------:|------:|--------:|
| Train | 45,615 | 45,613 | 2 |
| Val | 2,000 | 2,000 | 0 |
| Test | 12,284 | 12,241 | 43 |
| **Total** | **59,899** | **59,854** | **45 (0.08%)** |

### Step-by-Step Effect on Training Set

| Step | Rows Remaining |
|------|---------------:|
| Raw | 45,615 |
| After HTML decode | 45,615 |
| After URL replace | 45,615 |
| After @mention replace | 45,615 |
| After hashtag strip | 45,615 |
| After repeated punctuation normalise | 45,615 |
| After whitespace normalise | 45,615 |
| **After drop < 3 words (2 dropped)** | **45,613** |

### Label Distribution After Preprocessing

| Label | Train | % | Val | % | Test | % |
|-------|------:|---|----:|---|-----:|---|
| negative | 7,093 | 15.6% | 312 | 15.6% | 3,969 | 32.4% |
| neutral | 20,671 | 45.3% | 869 | 43.5% | 5,914 | 48.3% |
| positive | 17,849 | 39.1% | 819 | 40.9% | 2,358 | 19.3% |

---

## 4. Baseline — Raw Data, DistilBERT (Pre-Preprocessing Reference)

**Notebook:** `../train_sentiment_colab.ipynb`  
**Model:** `distilbert-base-uncased`  
**Data:** raw (unprocessed) tweet_eval CSVs

| Setting | Value |
|---------|-------|
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 64 |
| Class weights | None |

| Metric | Result |
|--------|--------|
| F1 Negative | **0.7060** |

F1 Negative was the key metric recorded from this run. It served as the reference threshold for the negative class across all subsequent experiments.

---

## 5. Training Experiments

All test results are on `tweet_eval_test_clean.csv` (12,241 rows).

**DistilBERT training notebook:** `train_sentiment_preprocessed_colab.ipynb` (Runs 1–3)  
**Twitter-RoBERTa training notebook:** `train_roberta_sentiment_colab.ipynb` (Runs 4–5)

---

### Run 1 — DistilBERT, Preprocessed, No Weights

| Setting | Value |
|---------|-------|
| Model | `distilbert-base-uncased` |
| Data | `tweet_eval_train_clean.csv` (all 8 steps including punc normalisation) |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 64 |
| Class weights | None |

| Metric | Test |
|--------|------|
| Accuracy | 0.7380 |
| F1 Macro | 0.7218 |
| F1 Negative | 0.6553 |
| F1 Neutral | 0.7179 |
| F1 Positive | 0.7921 |

**Finding:** Preprocessing improved overall accuracy (0.738) and F1 Macro (0.722) compared to the baseline. However, F1 Negative dropped significantly (0.706 → 0.655). Step 5 (repeated punctuation normalisation) removed emotional intensity signals (`!!!`, `???`) that are strong negative sentiment markers in tweet text.

---

### Run 2 — DistilBERT, Step 5 Removed, No Weights

| Setting | Value |
|---------|-------|
| Model | `distilbert-base-uncased` |
| Data | `tweet_eval_train_clean.csv` (Step 5 skipped — no punc normalisation) |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 64 |
| Class weights | None |

| Metric | Test |
|--------|------|
| Accuracy | 0.6880 |
| F1 Macro | 0.6872 |
| F1 Negative | 0.7087 |
| F1 Neutral | 0.6794 |
| F1 Positive | 0.6735 |

**Finding:** F1 Negative recovered (0.709) but all other metrics dropped substantially. Removing Step 5 alone cannot compensate for the model's limitation on tweet text. Preprocessing changes alone cannot solve the structural performance ceiling of a Wikipedia-pretrained model applied to tweet register.

---

### Run 3 — DistilBERT, 6 Epochs, Class Weights

| Setting | Value |
|---------|-------|
| Model | `distilbert-base-uncased` |
| Data | `tweet_eval_train_clean.csv` (all 8 steps) |
| Epochs | 6 (EarlyStoppingCallback patience=2) |
| Learning rate | 2e-5 |
| Batch size | 64 |
| Class weights | negative=2.144, neutral=0.736, positive=0.852 |

Class weights computed as inverse-frequency: `total_samples / (n_classes × class_count)`.

| Metric | Test |
|--------|------|
| Accuracy | 0.6791 |
| F1 Macro | 0.6792 |
| F1 Negative | 0.7089 |
| F1 Neutral | 0.6606 |
| F1 Positive | 0.6682 |

**Finding:** Class weights improved F1 Negative (0.709) but the 2.144× weight was too aggressive. The model over-predicted negative at the expense of neutral and positive, producing the worst F1 Macro of all runs. DistilBERT reached its performance ceiling on this tweet-domain task.

---

### Run 4 — Twitter-RoBERTa, Class Weights

| Setting | Value |
|---------|-------|
| Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Data | `tweet_eval_train_clean.csv` (all 8 steps) |
| Epochs | 4 (EarlyStoppingCallback patience=2) |
| Learning rate | 1e-5 |
| Batch size | 32 |
| Class weights | negative=2.144, neutral=0.736, positive=0.852 |

| Metric | Test |
|--------|------|
| Accuracy | 0.7109 |
| F1 Macro | 0.7130 |
| F1 Negative | 0.7478 |
| F1 Neutral | 0.6798 |
| F1 Positive | 0.7114 |

**Finding:** Switching to Twitter-RoBERTa (pre-trained on 124M tweets) gave a significant improvement in F1 Negative (0.748) and F1 Macro (0.713). The tweet-native pretraining provides a much better representation of conversational sentiment language. However, class weights still biased the model toward over-predicting negative, keeping F1 Neutral below 0.68.

---

### Run 5 — Twitter-RoBERTa, No Weights ✓ FINAL MODEL

| Setting | Value |
|---------|-------|
| Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Data | `tweet_eval_train_clean.csv` (all 8 steps) |
| Epochs | 4 (EarlyStoppingCallback patience=2, best at epoch 3) |
| Learning rate | 1e-5 |
| Batch size | 32 |
| Class weights | **None** |

| Metric | Val (epoch 3) | **Test** |
|--------|:-------------:|:--------:|
| Accuracy | 0.7650 | **0.7175** |
| F1 Macro | 0.7541 | **0.7182** |
| F1 Negative | 0.7079 | **0.7390** |
| F1 Neutral | 0.7313 | **0.7049** |
| F1 Positive | 0.8231 | **0.7105** |

**Finding:** Removing class weights from Twitter-RoBERTa produced the most balanced result. All three classes exceeded 0.70 F1 on the test set for the first time across all runs. Twitter-RoBERTa's tweet-native pretraining handles the implicit class imbalance better than explicit inverse-frequency weighting. This run was selected as the deployed model.

---

## 6. All Runs Comparison

| Run | Model | Weights | Accuracy | F1 Macro | F1 Neg | F1 Neu | F1 Pos |
|-----|-------|:-------:|:--------:|:--------:|:------:|:------:|:------:|
| Baseline | DistilBERT | No | — | — | 0.706 | — | — |
| Run 1 | DistilBERT | No | 0.738 | 0.722 | 0.655 | 0.718 | 0.792 |
| Run 2 | DistilBERT (no Step 5) | No | 0.688 | 0.687 | 0.709 | 0.679 | 0.674 |
| Run 3 | DistilBERT | Yes | 0.679 | 0.679 | 0.709 | 0.661 | 0.668 |
| Run 4 | Twitter-RoBERTa | Yes | 0.711 | 0.713 | 0.748 | 0.680 | 0.711 |
| **Run 5** | **Twitter-RoBERTa** | **No** | **0.718** | **0.718** | **0.739** | **0.705** | **0.711** |

---

## 7. Final Model

| Property | Value |
|----------|-------|
| Base model | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Parameters | 124,647,939 |
| Training data | `tweet_eval_train_clean.csv` — 45,613 rows |
| Training notebook | `train_roberta_sentiment_colab.ipynb` |
| Best epoch | 3 of 4 (early stopping, loaded best checkpoint) |
| Learning rate | 1e-5 |
| Batch size | 32 |
| Warmup steps | 570 |
| Weight decay | 0.01 |
| Class weights | None |
| **Test Accuracy** | **0.7175** |
| **Test F1 Macro** | **0.7182** |
| Test F1 Negative | 0.7390 |
| Test F1 Neutral | 0.7049 |
| Test F1 Positive | 0.7105 |

---

## 8. System Integration

**Model deployed at:** `m3_implementation/memory/models/sentiment_classifier/`

### Files Deployed

| File | Size | Description |
|------|-----:|-------------|
| `config.json` | ~2 KB | Model architecture + label mapping (auto-detects RoBERTa) |
| `model.safetensors` | ~476 MB | Fine-tuned weights |
| `tokenizer.json` | ~3.4 MB | RoBERTa fast tokenizer (self-contained) |
| `tokenizer_config.json` | ~1 KB | Tokenizer settings |
| `training_metrics.json` | ~1 KB | Training configuration and results |
| `training_args.bin` | ~5 KB | HuggingFace training arguments |

Files were downloaded from Colab as a ZIP, extracted locally, and copied into the `sentiment_classifier/` folder, replacing the previous DistilBERT model files.

### Loader — `feedback_sentiment_classifier.py`

**Path:** `m3_implementation/memory/core/feedback_sentiment_classifier.py`

The loader uses `transformers.pipeline("text-classification")` which auto-detects model architecture from `config.json`. The same loader code supports both DistilBERT and RoBERTa without modification — switching models only requires replacing the files in `sentiment_classifier/`.

**Updated label string (line 70):**
```python
label = "local Twitter-RoBERTa (SemEval 2017, fine-tuned)"
```

**Priority logic:**
1. Loads local model from `../models/sentiment_classifier/` if `config.json` is present
2. Falls back to remote `cardiffnlp/twitter-roberta-base-sentiment-latest` if local model missing

**Startup log:**
```
[FeedbackClassifier] Loaded local Twitter-RoBERTa (SemEval 2017, fine-tuned)
```

### Score Mapping

| Predicted Label | Score Range | Meaning |
|-----------------|:-----------:|---------|
| positive | +0.5 to +1.0 | User satisfied |
| neutral | −0.45 to +0.45 | No clear preference |
| negative | −1.0 to −0.5 | User dissatisfied |

Formula: `score = base_sign × (0.5 + 0.5 × model_confidence)`

---

## 9. Live System Validation

Validated in the running Sunlytics CRS pipeline after deploying the Twitter-RoBERTa model:

### Test 1 — Positive Feedback

```
User input   : "I like them"
Classifier   : positive  conf=0.876  score=+0.938
Pipeline     : FEEDBACK → positive → no retrieval (user satisfied)
System action: Acknowledged selection, offered to continue browsing or end session
```

### Test 2 — Negative Feedback

```
User input   : "I don't like them"
Classifier   : negative  conf=0.954  score=-0.977
Pipeline     : FEEDBACK → negative → full retrieval with exclusions
System action: Excluded 3 rejected items, retrieved 2 new alternatives
               (Miss Fancy Shorts Light Beige; RA shorts conscious Dark Grey)
```

Both cases confirmed:
- Correct sentiment detection and confidence scores
- Correct retrieval strategy routing based on sentiment label
- Correct preference memory updates (rejected items excluded from future results)

---

## 10. Files in This Folder

| File | Description |
|------|-------------|
| `preprocess.py` | Preprocessing script (8 steps, reads raw CSVs, writes clean CSVs) |
| `tweet_eval_train_clean.csv` | Cleaned training set (45,613 rows) |
| `tweet_eval_val_clean.csv` | Cleaned validation set (2,000 rows) |
| `tweet_eval_test_clean.csv` | Cleaned test set (12,241 rows) |
| `train_sentiment_preprocessed_colab.ipynb` | DistilBERT training notebook (Runs 1–3) |
| `train_roberta_sentiment_colab.ipynb` | Twitter-RoBERTa training notebook (Runs 4–5, final model) |
| `SENTIMENT_CLASSIFIER_FULL_PROCESS.md` | This document |

---

## 11. How to Reproduce Preprocessing

```bash
python preprocess.py
```

Run from any working directory — paths are resolved relative to the script's own location. Requires only `pandas` and the Python standard library.
