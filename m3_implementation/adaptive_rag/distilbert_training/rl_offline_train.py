# =============================================================================
# rl_offline_train.py
# =============================================================================
# Offline REINFORCE fine-tuning using REAL USER SIGNALS collected from
# actual conversations — NOT synthetic data.
#
# DATA SOURCE:
#   Real user interactions stored in MongoDB rl_experiences collection,
#   exported via POST /api/rl/export → rl_buffer_real.jsonl
#
#   Each line in the JSONL file is one real user experience:
#     {
#       "input_text":      "[SEP]-joined conversation DistilBERT classified",
#       "predicted_label": 1,           ← what DistilBERT predicted (0-7)
#       "label_name":      "REFINEMENT",
#       "total_reward":    0.7,         ← real user signal (not synthetic)
#       "reward_source":   "implicit_next_turn",
#       "session_id":      "sess_abc123"
#     }
#
# REWARD SOURCES (all real, no synthetic):
#   explicit_thumbs_up   → user clicked 👍  → reward +1.0
#   explicit_thumbs_down → user clicked 👎  → reward -1.0
#   implicit_next_turn   → next message reveals recommendation quality
#                          INITIAL_REQUEST → SELECTION_REFERENCE → +1.0
#                          REFINEMENT      → REFINEMENT           → -0.3
#   session_outcome      → full conversation shape analysis
#                          short + selection → +1.0, many refinements → -0.5
#
# ALGORITHM: REINFORCE with combined loss
#   L = L_SFT + λ·L_REINFORCE - β·H(π)
#   L_SFT       = CrossEntropy(logits, predicted_label)  keeps SFT accuracy
#   L_REINFORCE = -(G - baseline)·log π(a|s)            real reward signal
#   H(π)        = entropy bonus prevents policy collapse
#
# SAFEGUARDS:
#   1. Freeze transformer — only 596K classifier head params updated
#   2. Low LR (5e-6) — 4x lower than SFT to prevent catastrophic forgetting
#   3. Safe rollback — only saves if val F1 improves over SFT baseline
#   4. Accuracy drop alert — warns if per-turn accuracy degrades > 0.5%
#
# RUN:
#   # First export real data from your running server:
#   curl -X POST http://localhost:8000/api/rl/export
#
#   # Then train (from distilbert_training folder):
#   python rl_offline_train.py
#   python rl_offline_train.py --buffer rl_buffer_real.jsonl --epochs 3
#   python rl_offline_train.py --min-experiences 50   (lower threshold for testing)
# =============================================================================

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from config import (
    MODEL_SAVE_DIR, VAL_FILE, RESULTS_DIR,
    LABEL_NAMES, BATCH_SIZE, MAX_LEN, SEED,
)
from dataset import RetrievalDataset

# ── RL hyperparameters ─────────────────────────────────────────────────────────
RL_LR            = 5e-6   # 4x lower than SFT (2e-5)
RL_LAMBDA        = 0.1    # Weight of RL loss vs SFT loss
RL_ENTROPY_BETA  = 0.01   # Entropy bonus coefficient
RL_EPOCHS        = 3
RL_BATCH_SIZE    = 16
RL_MAX_GRAD_NORM = 0.5
RL_WARMUP_RATIO  = 0.05
RL_FREEZE_TRANSFORMER = True
MIN_EXPERIENCES  = 100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset — wraps real user experiences from JSONL
# =============================================================================

class RealExperienceDataset(Dataset):
    """
    Loads real user RL experiences from the JSONL file exported by
    POST /api/rl/export.

    Each experience was generated from a REAL user interaction — not synthetic.
    The reward reflects actual user behaviour: did they find what they wanted?
    """

    def __init__(self, jsonl_path: str, tokenizer, max_len: int = MAX_LEN):
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.experiences = []

        if not os.path.exists(jsonl_path):
            print(f"[RL] Buffer not found: {jsonl_path}")
            print(f"[RL] Run: curl -X POST http://localhost:8000/api/rl/export")
            return

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if (d.get("input_text") and
                        d.get("total_reward") is not None and
                        d.get("predicted_label") is not None):
                        self.experiences.append(d)
                except Exception:
                    pass

        if not self.experiences:
            print(f"[RL] No valid experiences in {jsonl_path}")
            return

        rewards = [e["total_reward"] for e in self.experiences]
        print(f"\n[RL] Loaded {len(self.experiences)} real user experiences")
        print(f"  Reward: mean={np.mean(rewards):+.3f}  "
              f"min={np.min(rewards):+.3f}  max={np.max(rewards):+.3f}")

        sources = {}
        for e in self.experiences:
            src = e.get("reward_source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        source_labels = {
            "explicit_thumbs_up":   "Thumbs up",
            "explicit_thumbs_down": "Thumbs down",
            "implicit_next_turn":   "Implicit next-turn",
            "session_outcome":      "Session outcome",
        }
        print(f"  Signal breakdown:")
        for src, count in sources.items():
            print(f"    {source_labels.get(src, src)}: {count}")

        pos = sum(1 for r in rewards if r > 0)
        neg = sum(1 for r in rewards if r < 0)
        print(f"  Positive signals: {pos}  |  Negative signals: {neg}")

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        exp = self.experiences[idx]
        enc = self.tokenizer(
            exp["input_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "action":         torch.tensor(exp["predicted_label"], dtype=torch.long),
            "reward":         torch.tensor(float(exp["total_reward"]), dtype=torch.float),
        }


def _collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "action":         torch.stack([b["action"] for b in batch]),
        "reward":         torch.stack([b["reward"] for b in batch]),
    }


# =============================================================================
# Layer freezing
# =============================================================================

def freeze_transformer(model):
    """Freeze all transformer layers. Only train classifier head (~596K params)."""
    for name, param in model.named_parameters():
        if "classifier" in name or "pre_classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[RL] Frozen: {frozen:,} params | Trainable: {trainable:,} params")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_val(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
    return (
        f1_score(all_labels, all_preds, average="macro"),
        accuracy_score(all_labels, all_preds),
    )


# =============================================================================
# One RL training epoch on real data
# =============================================================================

def train_rl_epoch(model, loader, optimiser, scheduler, device, baseline_state):
    """
    One epoch of REINFORCE on real user experience data.
    Rewards come from actual user behaviour — thumbs, next-turn behaviour,
    session outcomes — not from designed synthetic patterns.
    """
    model.train()
    total_loss = sft_total = rl_total = ent_total = 0.0
    n = 0

    # Compute running baseline for variance reduction
    all_r = []
    for batch in loader:
        all_r.extend(batch["reward"].tolist())
    epoch_mean = float(np.mean(all_r)) if all_r else 0.0
    baseline_state["value"] = (
        0.95 * baseline_state["value"] + 0.05 * epoch_mean
    )
    current_baseline = baseline_state["value"]

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        actions        = batch["action"].to(device)
        rewards        = batch["reward"].to(device)

        optimiser.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Supervised loss (anchors to SFT knowledge)
        L_sft = F.cross_entropy(logits, actions)

        # REINFORCE with advantage
        log_probs        = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        advantages       = rewards - current_baseline
        L_rl             = -(advantages * action_log_probs).mean()

        # Entropy bonus (prevents collapsing to one label)
        probs    = F.softmax(logits, dim=-1)
        entropy  = -(probs * log_probs).sum(dim=-1).mean()
        L_entropy = -RL_ENTROPY_BETA * entropy

        L = L_sft + RL_LAMBDA * L_rl + L_entropy
        L.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            RL_MAX_GRAD_NORM,
        )
        optimiser.step()
        scheduler.step()

        total_loss += L.item()
        sft_total  += L_sft.item()
        rl_total   += L_rl.item()
        ent_total  += entropy.item()
        n          += 1

    d = max(n, 1)
    return total_loss/d, sft_total/d, rl_total/d, ent_total/d


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Offline REINFORCE fine-tuning on real user signals."
    )
    parser.add_argument("--buffer",          default="rl_buffer_real.jsonl")
    parser.add_argument("--epochs",          type=int, default=RL_EPOCHS)
    parser.add_argument("--min-experiences", type=int, default=MIN_EXPERIENCES)
    parser.add_argument("--output-dir",      default=None)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[RL] ══════════════════════════════════════════")
    print(f"[RL] Real-User REINFORCE Fine-tuning")
    print(f"[RL] ══════════════════════════════════════════")
    print(f"[RL] Device : {device}")
    print(f"[RL] Buffer : {args.buffer}")

    output_dir = args.output_dir or MODEL_SAVE_DIR
    rl_results_dir = os.path.join(RESULTS_DIR, "rl_real")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(rl_results_dir, exist_ok=True)

    # Load model
    print(f"\n[RL] Loading SFT model: {MODEL_SAVE_DIR}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_SAVE_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
    model.to(device)

    if RL_FREEZE_TRANSFORMER:
        freeze_transformer(model)

    # Load real experience buffer
    exp_dataset = RealExperienceDataset(args.buffer, tokenizer, MAX_LEN)

    if len(exp_dataset) < args.min_experiences:
        print(f"\n[RL] Only {len(exp_dataset)} real experiences collected.")
        print(f"[RL] Minimum: {args.min_experiences}. Keep using the app.")
        print(f"[RL] Check: GET http://localhost:8000/api/rl/stats")
        return

    exp_loader = DataLoader(
        exp_dataset, batch_size=RL_BATCH_SIZE,
        shuffle=True, num_workers=0, collate_fn=_collate,
    )

    # Validation set (same as SFT — fair baseline comparison)
    val_dataset = RetrievalDataset(VAL_FILE, tokenizer)
    val_loader  = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0
    )

    # Pre-RL baseline
    print(f"\n[RL] ── Pre-RL baseline ──")
    baseline_f1, baseline_acc = evaluate_val(model, val_loader, device)
    print(f"  Val macro-F1 : {baseline_f1:.4f}")
    print(f"  Val accuracy : {baseline_acc*100:.2f}%")

    # Optimiser — only trainable params (classifier head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser    = torch.optim.AdamW(trainable_params, lr=RL_LR, weight_decay=0.01)
    total_steps  = len(exp_loader) * args.epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_steps * RL_WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # Training loop
    baseline_state = {"value": 0.0}
    best_f1        = baseline_f1
    history = {"train_loss":[], "sft_loss":[], "rl_loss":[], "entropy":[],
               "val_f1":[], "val_accuracy":[]}

    print(f"\n[RL] ── Training on {len(exp_dataset)} real experiences ──")
    print(f"[RL] λ={RL_LAMBDA}  β={RL_ENTROPY_BETA}  LR={RL_LR}  "
          f"batch={RL_BATCH_SIZE}  epochs={args.epochs}")
    print("=" * 55)

    for epoch in range(1, args.epochs + 1):
        print(f"\n[RL] Epoch {epoch}/{args.epochs}")
        print("-" * 40)

        t_loss, s_loss, r_loss, ent = train_rl_epoch(
            model, exp_loader, optimiser, scheduler, device, baseline_state
        )
        val_f1, val_acc = evaluate_val(model, val_loader, device)

        history["train_loss"].append(t_loss)
        history["sft_loss"].append(s_loss)
        history["rl_loss"].append(r_loss)
        history["entropy"].append(ent)
        history["val_f1"].append(val_f1)
        history["val_accuracy"].append(val_acc)

        print(f"  Loss: total={t_loss:.4f}  sft={s_loss:.4f}  "
              f"rl={r_loss:.4f}  entropy={ent:.4f}")
        print(f"  Val F1: {val_f1:.4f}  (Δ={val_f1-baseline_f1:+.4f})")
        print(f"  Val Acc: {val_acc*100:.2f}%  "
              f"(Δ={(val_acc-baseline_acc)*100:+.2f}pp)")

        if (baseline_acc - val_acc) > 0.005:
            print(f"  ⚠ Accuracy dropped. Consider more real data or lower RL_LAMBDA.")

        if val_f1 >= best_f1:
            best_f1 = val_f1
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  ✓ Best model saved → {output_dir}")
        else:
            print(f"  – No improvement. Previous best kept.")

    # Final summary
    print(f"\n{'='*55}")
    print(f"  REAL-USER RL RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Before':>8}  {'After':>8}  {'Δ':>8}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*8}")
    final_acc = history["val_accuracy"][-1] if history["val_accuracy"] else baseline_acc
    print(f"  {'Val macro-F1':<30} {baseline_f1:>8.4f}  {best_f1:>8.4f}  "
          f"{best_f1-baseline_f1:>+8.4f}")
    print(f"  {'Val accuracy':<30} {baseline_acc:>8.4f}  {final_acc:>8.4f}  "
          f"{final_acc-baseline_acc:>+8.4f}")
    print(f"  {'Real experiences':<30} {'':>8}  {len(exp_dataset):>8}")
    print(f"  {'Params updated':<30} {'':>8}  {'~596K':>8}")
    print(f"{'='*55}")

    if best_f1 >= baseline_f1:
        print(f"\n  ✓ Real-user RL improved the model.")
        print(f"  Restart server: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print(f"\n  ⚠ No improvement. Original checkpoint preserved.")
        print(f"  Collect more real user signals before next run.")

    # Save results
    results_path = os.path.join(rl_results_dir, "rl_real_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "approach":          "real_user_reinforce",
            "data_source":       args.buffer,
            "n_real_experiences": len(exp_dataset),
            "signal_sources":    _count_sources(args.buffer),
            "baseline_f1":       round(baseline_f1, 4),
            "best_f1":           round(best_f1, 4),
            "improvement":       round(best_f1 - baseline_f1, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "epochs_run":        args.epochs,
            "history":           history,
            "hyperparams": {
                "lr": RL_LR, "lambda": RL_LAMBDA,
                "entropy_beta": RL_ENTROPY_BETA,
                "batch_size": RL_BATCH_SIZE,
                "freeze_transformer": RL_FREEZE_TRANSFORMER,
            },
        }, f, indent=2)
    print(f"[RL] Results → {results_path}")


def _count_sources(jsonl_path):
    sources = {}
    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    s = d.get("reward_source", "unknown")
                    sources[s] = sources.get(s, 0) + 1
                except Exception:
                    pass
    except Exception:
        pass
    return sources


if __name__ == "__main__":
    main()
