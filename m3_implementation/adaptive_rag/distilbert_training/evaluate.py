# =============================================================================
# evaluate.py
# =============================================================================
# Run this AFTER train.py has completed.
# It loads the best saved model checkpoint and evaluates it on the held-out
# test set, which the model has NEVER seen during training or validation.
#
# Outputs produced:
#   outputs/results/test_classification_report.txt  ← per-class precision/recall/F1
#   outputs/results/confusion_matrix.png            ← heatmap visualisation
#   outputs/results/test_metrics_summary.json       ← all metrics as JSON
# =============================================================================

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TEST_FILE, MODEL_SAVE_DIR, RESULTS_DIR,
    LABEL_NAMES, BATCH_SIZE, SEED,
)
from dataset import RetrievalDataset

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """
    Saves a heatmap of the confusion matrix.

    The confusion matrix shows, for each true class (rows),
    how many samples were predicted as each class (columns).
    The diagonal = correct predictions. Off-diagonal = errors.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalise each row to show percentages instead of raw counts.
    # This makes it easier to spot where the model confuses classes,
    # regardless of how many samples each class has.
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm,
        annot=True,           # write the value inside each cell
        fmt=".2f",            # format as 2 decimal places
        cmap="Blues",         # colour scale: white = 0, dark blue = 1.0
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")


def plot_training_curves(history_path: str, save_path: str):
    """
    Loads the training history JSON and plots loss and F1 curves.
    This is useful for your dissertation to visualise the training process.
    """
    if not os.path.exists(history_path):
        print("  Training history not found, skipping curve plot.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation F1 curve
    ax2.plot(epochs, history["val_macro_f1"], "g-o", label="Val Macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_title("Validation Macro-F1 per Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("DistilBERT Training History — Retrieval Trigger Classifier", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load the best saved model ─────────────────────────────────────────────
    # We load from MODEL_SAVE_DIR, which is where train.py saved the best
    # checkpoint. The tokenizer is also loaded from there to guarantee
    # it is the exact same tokenizer used during training.
    print(f"\nLoading best model from: {MODEL_SAVE_DIR}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_SAVE_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
    model.to(device)
    model.eval()   # ALWAYS set to eval mode before evaluating

    # ── Build test DataLoader ─────────────────────────────────────────────────
    print(f"Loading test set: {TEST_FILE}")
    test_dataset = RetrievalDataset(TEST_FILE, tokenizer)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,           # never shuffle the test set
        num_workers=2,
        pin_memory=True,
    )
    print(f"  Test samples: {len(test_dataset)}")

    # ── Run inference on the full test set ────────────────────────────────────
    all_preds  = []
    all_labels = []

    print("\nRunning inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Compute metrics ───────────────────────────────────────────────────────
    accuracy        = accuracy_score(all_labels, all_preds)
    macro_f1        = f1_score(all_labels, all_preds, average="macro")
    weighted_f1     = f1_score(all_labels, all_preds, average="weighted")
    per_class_f1    = f1_score(all_labels, all_preds, average=None)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy        : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Macro-F1        : {macro_f1:.4f}")
    print(f"  Weighted-F1     : {weighted_f1:.4f}")
    print()
    print("  Per-class F1:")
    for i, (name, f1) in enumerate(zip(LABEL_NAMES, per_class_f1)):
        print(f"    {name:<25} F1 = {f1:.4f}")

    # ── Full classification report ────────────────────────────────────────────
    report = classification_report(
        all_labels,
        all_preds,
        target_names=LABEL_NAMES,
        digits=4,
    )
    print("\nFull classification report:")
    print(report)

    report_path = os.path.join(RESULTS_DIR, "test_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("TEST SET CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)

    # ── Training curves ───────────────────────────────────────────────────────
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    curves_path  = os.path.join(RESULTS_DIR, "training_curves.png")
    plot_training_curves(history_path, curves_path)

    # ── Save all metrics as JSON ──────────────────────────────────────────────
    # This JSON file is useful for your dissertation — you can load it directly
    # into tables or charts without re-running evaluation.
    metrics = {
        "accuracy":    round(float(accuracy),    4),
        "macro_f1":    round(float(macro_f1),    4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class_f1": {
            name: round(float(f1), 4)
            for name, f1 in zip(LABEL_NAMES, per_class_f1)
        },
    }
    metrics_path = os.path.join(RESULTS_DIR, "test_metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics summary saved to: {metrics_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
