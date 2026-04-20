# =============================================================================
# train.py
# =============================================================================
# The main training script. Run this file to fine-tune DistilBERT on your
# retrieval classification dataset.
#
# What this script does, step by step:
#   1. Sets random seeds for reproducibility
#   2. Loads the tokenizer and model
#   3. Builds Dataset and DataLoader objects for train and validation splits
#   4. Configures the AdamW optimiser and a learning rate scheduler
#   5. Runs the training loop for up to NUM_EPOCHS epochs
#   6. After each epoch, evaluates on the validation set
#   7. Saves the best model checkpoint (based on validation macro-F1)
#   8. Stops early if validation performance stops improving
#   9. Saves training history (loss and F1 per epoch) for later plotting
# =============================================================================

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from config import (
    TRAIN_FILE, VAL_FILE, MODEL_SAVE_DIR, RESULTS_DIR,
    PRETRAINED_MODEL, NUM_LABELS, LABEL_NAMES,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    WEIGHT_DECAY, WARMUP_RATIO, EARLY_STOPPING_PATIENCE, SEED,
)
from dataset import RetrievalDataset


# =============================================================================
# STEP 1 — Reproducibility
# =============================================================================
# Setting seeds in all four places (Python random, NumPy, PyTorch CPU,
# and PyTorch GPU) ensures that every run produces the exact same results.
# This is critical for a research project where you need to report results.

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# STEP 2 — Device selection
# =============================================================================
# PyTorch can run computations on CPU or on a GPU (CUDA).
# GPU training is typically 10-50x faster for transformer models.
# This function automatically picks the GPU if available, otherwise CPU.

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


# =============================================================================
# STEP 3 — Single epoch training function
# =============================================================================
# This function runs ONE pass through the entire training dataset.
# It is called once per epoch from the main training loop below.

def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimiser,
    scheduler,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch. Returns the average training loss.

    The training loop works like this for each batch:
      a) Move the batch tensors to the device (GPU or CPU)
      b) Forward pass: feed input_ids and attention_mask through the model
         → the model outputs logits (raw scores for each of the 8 classes)
      c) The loss is computed automatically inside the model when you pass labels
         (it uses CrossEntropyLoss internally)
      d) Backward pass: compute gradients via backpropagation
      e) Clip gradients to prevent them from becoming too large (exploding gradients)
      f) Update the model weights using the optimiser
      g) Update the learning rate using the scheduler
      h) Zero out the gradients to prepare for the next batch
    """
    model.train()   # puts the model in training mode (enables dropout, etc.)
    total_loss = 0.0

    # tqdm wraps the dataloader to show a live progress bar
    progress_bar = tqdm(dataloader, desc="  Training", leave=False)

    for batch in progress_bar:
        # Move all tensors in the batch from CPU RAM to GPU VRAM
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Zero gradients from the previous batch.
        # If you don't do this, gradients accumulate across batches,
        # which gives completely wrong weight updates.
        optimiser.zero_grad()

        # Forward pass through DistilBERT.
        # When labels are provided, the model returns (loss, logits).
        # loss  = CrossEntropyLoss(logits, labels) computed internally
        # logits = raw scores of shape [batch_size, 8]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        # Backward pass: compute gradients of loss w.r.t. all parameters
        loss.backward()

        # Gradient clipping: if any gradient magnitude exceeds 1.0,
        # scale it down. This prevents "exploding gradients" which can
        # destabilise training, especially in the early epochs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model weights based on gradients
        optimiser.step()

        # Update the learning rate according to the warmup/linear-decay schedule
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)   # average loss over all batches


# =============================================================================
# STEP 4 — Evaluation function
# =============================================================================
# This function evaluates the model on a given dataset split (val or test).
# It runs in "no_grad" mode, meaning PyTorch does NOT compute gradients —
# this saves memory and time during evaluation.

def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """
    Evaluates the model. Returns (avg_loss, macro_f1, all_preds, all_labels).

    We return all_preds and all_labels so the calling code can compute
    a full classification report and confusion matrix.
    """
    model.eval()   # puts the model in evaluation mode (disables dropout)
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():   # no gradient computation needed during evaluation
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss   = outputs.loss
            logits = outputs.logits   # shape: [batch_size, 8]

            total_loss += loss.item()

            # Convert logits to predicted class labels.
            # argmax(dim=1) picks the index of the highest score for each sample.
            preds = torch.argmax(logits, dim=1)

            # Move tensors back to CPU and convert to Python lists for sklearn
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss   = total_loss / len(dataloader)
    macro_f1   = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, macro_f1, all_preds, all_labels


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    set_seed(SEED)
    device = get_device()

    # Create output directories if they don't exist yet
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    # The tokenizer converts raw text into integer token IDs.
    # We load it from HuggingFace Hub using the model name string.
    # It will be downloaded to a local cache the first time (~200KB).
    print(f"\nLoading tokenizer: {PRETRAINED_MODEL}")
    tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL)

    # ── Build Datasets ────────────────────────────────────────────────────────
    print("Building datasets...")
    train_dataset = RetrievalDataset(TRAIN_FILE, tokenizer)
    val_dataset   = RetrievalDataset(VAL_FILE,   tokenizer)

    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Val samples   : {len(val_dataset)}")

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    # DataLoader wraps a Dataset and yields batches of samples.
    # shuffle=True for training means the order of samples is randomised
    # each epoch, which helps the model generalise better.
    # num_workers=2 means 2 background processes pre-load data in parallel,
    # so the GPU is never waiting for data to arrive.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,   # speeds up CPU→GPU tensor transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,   # can use larger batch for eval (no gradients)
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Load the pre-trained model ────────────────────────────────────────────
    # DistilBertForSequenceClassification is DistilBERT with a classification
    # head added on top. The head is a simple linear layer that maps the
    # [CLS] token representation (size 768) to num_labels (8) output scores.
    #
    # id2label and label2id are stored in the model config — this means when
    # you save and reload the model, it knows the names of your classes.
    print(f"\nLoading model: {PRETRAINED_MODEL}")
    model = DistilBertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_LABELS,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )
    model.to(device)   # move model weights to GPU if available

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # AdamW is the standard optimiser for transformer fine-tuning.
    # It is Adam with a corrected weight decay implementation.
    # weight_decay adds L2 regularisation to prevent overfitting.
    #
    # We do NOT apply weight decay to bias parameters or LayerNorm weights —
    # this is the standard practice from the original BERT fine-tuning paper
    # because these parameters behave differently from regular weights.
    no_decay = ["bias", "LayerNorm.weight"]
    optimiser_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimiser = torch.optim.AdamW(optimiser_grouped_parameters, lr=LEARNING_RATE)

    # ── Learning rate scheduler ───────────────────────────────────────────────
    # The scheduler controls how the learning rate changes over training.
    # We use a linear warmup then linear decay schedule:
    #   - For the first warmup_steps, LR increases from 0 to LEARNING_RATE
    #   - After warmup, LR decreases linearly back to 0 by the end of training
    # Warmup prevents large unstable updates in the very first steps when
    # gradients can be noisy.
    total_steps   = len(train_loader) * NUM_EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs          : {NUM_EPOCHS}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Learning rate   : {LEARNING_RATE}")
    print(f"  Total steps     : {total_steps}")
    print(f"  Warmup steps    : {warmup_steps}")
    print(f"  Early stopping  : patience = {EARLY_STOPPING_PATIENCE} epochs")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_macro_f1": [],
    }

    best_val_f1        = 0.0
    epochs_no_improve  = 0   # counter for early stopping

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 40)

        # ── Train for one epoch ───────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, optimiser, scheduler, device
        )

        # ── Evaluate on validation set ────────────────────────────────────────
        val_loss, val_macro_f1, val_preds, val_labels = evaluate(
            model, val_loader, device
        )

        # ── Log results ───────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        print(f"  Train loss    : {train_loss:.4f}")
        print(f"  Val loss      : {val_loss:.4f}")
        print(f"  Val macro-F1  : {val_macro_f1:.4f}")

        # ── Save best model ───────────────────────────────────────────────────
        # We save the model whenever validation macro-F1 improves.
        # Macro-F1 is the right metric here because it treats all 8 classes
        # equally regardless of their frequency.
        if val_macro_f1 > best_val_f1:
            best_val_f1       = val_macro_f1
            epochs_no_improve = 0

            # save_pretrained saves all model weights AND the config
            # (including id2label / label2id mappings you set above)
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)

            print(f"  ✓ New best model saved (val macro-F1 = {best_val_f1:.4f})")

            # Also save a per-class classification report for this best epoch
            report = classification_report(
                val_labels,
                val_preds,
                target_names=LABEL_NAMES,
                digits=4,
            )
            report_path = os.path.join(RESULTS_DIR, "best_epoch_val_report.txt")
            with open(report_path, "w") as f:
                f.write(f"Best epoch: {epoch}\n\n")
                f.write(report)

        else:
            epochs_no_improve += 1
            print(f"  No improvement. Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        # ── Early stopping check ──────────────────────────────────────────────
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after epoch {epoch}.")
            print(f"Best val macro-F1: {best_val_f1:.4f}")
            break

    # ── Save training history ─────────────────────────────────────────────────
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\n" + "=" * 60)
    print(f"Training complete. Best val macro-F1: {best_val_f1:.4f}")
    print(f"Best model saved to: {MODEL_SAVE_DIR}")
    print("=" * 60)
    print("\nNext step: run  python evaluate.py  to test on the held-out test set.")


if __name__ == "__main__":
    main()
