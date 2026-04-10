# DistilBERT Retrieval Trigger Classifier

Fine-tuning DistilBERT for adaptive per-turn retrieval triggering in a conversational fashion recommender system.

---

## What this code does

This project trains a text classifier that reads a fashion chatbot conversation and decides — for each user turn — whether the system should perform **full catalog retrieval**, **partial metadata lookup**, or **skip retrieval entirely**. This is the core contribution described in the dissertation: adaptive retrieval triggering for conversational recommender systems (CRS).

The classifier takes the last 1–3 conversation turns plus the current user message as input and predicts one of eight labels:

| Label | Class | Retrieval Strategy |
|-------|-------|--------------------|
| 0 | INITIAL_REQUEST | FULL |
| 1 | REFINEMENT | FULL |
| 2 | ATTRIBUTE_QUESTION | PARTIAL |
| 3 | EXPLANATION_WHY | PARTIAL |
| 4 | COMPARISON | PARTIAL |
| 5 | SELECTION_REFERENCE | PARTIAL |
| 6 | FEEDBACK | NO |
| 7 | CHITCHAT | NO |

---

## Repository structure

```
distilbert_training/
│
├── data/                          ← Put your CSV files here
│   ├── v2_train_clean.csv
│   ├── v2_val_clean.csv
│   └── v2_test_clean.csv
│
├── outputs/                       ← Created automatically during training
│   ├── best_model/                ← Saved model checkpoint
│   └── results/                   ← Evaluation reports and plots
│
├── config.py                      ← All hyperparameters and paths
├── dataset.py                     ← PyTorch Dataset class
├── train.py                       ← Training loop
├── evaluate.py                    ← Test set evaluation + plots
├── predict.py                     ← Inference on new conversations
└── requirements.txt               ← Python dependencies
```

---

## Step-by-step setup

### Step 1 — Clone or create the project folder

If you are adding this to an existing repo, copy all `.py` files and `requirements.txt` into a folder called `distilbert_training/`. Then create a `data/` subfolder inside it.

### Step 2 — Place your data files

Copy the three clean CSV files into the `data/` folder:

```
data/v2_train_clean.csv    (6,720 samples)
data/v2_val_clean.csv      (1,440 samples)
data/v2_test_clean.csv     (1,440 samples)
```

### Step 3 — Create a Python virtual environment

A virtual environment keeps your project's dependencies isolated from other Python projects on your machine. Always use one for a research project.

```bash
# Create the environment (do this once)
python -m venv venv

# Activate it — Linux / Mac:
source venv/bin/activate

# Activate it — Windows:
venv\Scripts\activate

# You should now see (venv) at the start of your terminal prompt
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, HuggingFace Transformers, scikit-learn, and the other libraries needed. It may take 2–5 minutes. The first time you run training, the DistilBERT pre-trained weights (~250 MB) will also be downloaded automatically from HuggingFace Hub and cached locally.

### Step 5 — Check config.py before training

Open `config.py` and verify:

- `DATA_DIR` points to where your CSV files are. By default it expects a `data/` subfolder next to the scripts — this is correct if you followed Step 2.
- `BATCH_SIZE = 32` works for most GPUs with 6 GB+ VRAM. If you get an "out of memory" error, change it to `16`.
- Everything else can stay at the default values for your first run.

---

## Running the code

### Training

```bash
python train.py
```

What you will see: a progress bar for each epoch, showing the current batch loss. After each epoch, the validation loss and macro-F1 are printed. Every time the macro-F1 improves, the message "✓ New best model saved" appears. Training stops automatically when the validation score stops improving for 3 consecutive epochs (early stopping).

Expected training time: approximately 8–15 minutes on a GPU, 45–90 minutes on CPU only.

When training finishes, two things are saved:
- `outputs/best_model/` — the model checkpoint with the highest validation macro-F1
- `outputs/results/training_history.json` — loss and F1 per epoch

### Evaluation on the test set

```bash
python evaluate.py
```

This loads the best saved checkpoint and runs it on the held-out test set. It produces:
- `outputs/results/test_classification_report.txt` — precision, recall, F1 for each of the 8 classes
- `outputs/results/confusion_matrix.png` — visual heatmap (use this in your dissertation)
- `outputs/results/training_curves.png` — loss and F1 curves over training epochs
- `outputs/results/test_metrics_summary.json` — all metrics as a JSON file

### Running inference on a new conversation

```bash
python predict.py
```

This runs four built-in demo conversations and prints the predicted label, retrieval strategy, and confidence score for each. To integrate the classifier into your CRS pipeline, import the `Predictor` class:

```python
from predict import Predictor

# Load once at startup
predictor = Predictor()

# Call for each conversation turn
history = [
    {"role": "user", "content": "Show me some black dresses"},
    {"role": "bot",  "content": "Here are two options ..."},
]
result = predictor.predict(history, "What material is the first one?")

print(result["retrieval_strategy"])   # "PARTIAL"
print(result["label_name"])           # "ATTRIBUTE_QUESTION"
print(result["confidence"])           # e.g. 0.9312
```

---

## Expected results

Based on the dataset characteristics (9,600 balanced samples, 8 classes, clean context-aware input), you should expect approximately:

- Accuracy: 88–94%
- Macro-F1: 0.87–0.93
- The easiest labels for the model to learn are INITIAL_REQUEST, CHITCHAT, and FEEDBACK, because they have very strong lexical signals.
- The hardest labels are ATTRIBUTE_QUESTION vs SELECTION_REFERENCE, because both appear after a recommendation and involve questions about presented items. The conversation context (history) is what distinguishes them, which is why we always feed the history into the model.

---

## Troubleshooting

**CUDA out of memory:** Reduce `BATCH_SIZE` from 32 to 16 in `config.py`.

**FileNotFoundError for CSV files:** Check that `DATA_DIR` in `config.py` points to the correct folder and that the CSV filenames match exactly.

**Model not found during evaluate.py:** You must run `train.py` first so that the model checkpoint exists in `outputs/best_model/`.

**Slow training on CPU:** Training on CPU is expected to be slow (45–90 min). For a student project, Google Colab provides a free GPU — upload your `data/` folder and all `.py` files there. See the Colab section below.

---

## Running on Google Colab (free GPU)

If you do not have a local GPU, Google Colab gives you a free T4 GPU which will reduce training time to approximately 3–5 minutes.

1. Go to https://colab.research.google.com and create a new notebook.
2. In Runtime → Change runtime type, select **GPU**.
3. Upload all `.py` files and your `data/` folder using the file panel on the left.
4. In a code cell, run:

```python
!pip install transformers datasets scikit-learn pandas matplotlib seaborn tqdm accelerate
!python train.py
!python evaluate.py
```

5. After training, download `outputs/` to your local machine for your records.
