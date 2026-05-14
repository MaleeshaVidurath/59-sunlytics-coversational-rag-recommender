"""
rl_finetune.py  (MERGED FINAL VERSION)
=======================================
REINFORCE fine-tuning for the DistilBERT CRS intent classifier.

Research contribution
---------------------
Supervised fine-tuning (SFT) optimises for per-turn label accuracy — it
asks "did the model pick the right category?" but not "did the model's
decision lead to a good recommendation outcome?". This script adds a
REINFORCE policy-gradient objective so the classifier learns to prefer
action sequences that maximise cumulative user satisfaction over a full
conversation — directly addressing the simulation-to-real gap inherent in
synthetic training data.

Algorithm
---------
  Combined loss:  L = L_SFT + λ · L_REINFORCE - β · H(π)
  L_SFT         = CrossEntropy(logits, true_label)         supervised signal
  L_REINFORCE   = -(G_t - baseline) · log π_θ(a_t | s_t) policy gradient
  H(π)          = -Σ π(a|s) log π(a|s)                    entropy bonus
  G_t           = Σ_k γ^k · r_{t+k}                       discounted return

Three safeguards against catastrophic forgetting:
  1. Selective layer freezing  — only pre_classifier + classifier updated
  2. Low RL learning rate      — 5e-6 (vs SFT 2e-5)
  3. Low RL lambda             — 0.1 keeps SFT loss dominant
  4. Entropy regularisation    — β=0.01 prevents policy collapse

Usage
-----
  python rl_finetune.py                           (uses default paths)
  python rl_finetune.py --sft_model_dir <path>   (custom SFT checkpoint)
  python rl_finetune.py --n_trajectories 5000    (quick smoke test)
"""

import os
import sys
import json
import random
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BASE_DIR, DATA_DIR, LABEL_NAMES, MAX_LEN,
    NUM_LABELS, OUTPUT_DIR, SEED,
)
from rl_config import (
    CONVERSATION_PATTERNS, LABEL_TURN_REWARDS, PATTERN_WEIGHTS,
    RL_BATCH_SIZE, RL_ENTROPY_BETA, RL_EPOCHS, RL_FREEZE_TRANSFORMER,
    RL_GAMMA, RL_LAMBDA, RL_LR, RL_MAX_GRAD_NORM,
    RL_MIN_TRAJ_LEN, RL_N_TRAJECTORIES, RL_OUTPUT_DIR, RL_RESULTS_DIR,
    RL_REWARD_CLIP, RL_SEED, RL_WARMUP_RATIO,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_ID_TO_NAME: Dict[int, str] = {i: n for i, n in enumerate(LABEL_NAMES)}
LABEL_NAME_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(LABEL_NAMES)}

_total_w    = sum(PATTERN_WEIGHTS)
PATTERN_PROBS = [w / _total_w for w in PATTERN_WEIGHTS]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Turn:
    input_ids:      torch.Tensor
    attention_mask: torch.Tensor
    label_id:       int
    label_name:     str


@dataclass
class Trajectory:
    turns:           List[Turn]
    terminal_reward: float
    returns:         List[float]


# =============================================================================
# Reward computation
# =============================================================================

def compute_discounted_returns(
    label_sequence:  List[str],
    terminal_reward: float,
    gamma:           float = RL_GAMMA,
    reward_clip:     float = RL_REWARD_CLIP,
) -> List[float]:
    """
    Compute G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … + γ^(T-t)·R_T

    r_t = per-label intrinsic reward (from LABEL_TURN_REWARDS)
    R_T = terminal reward added only to the last turn
    """
    T = len(label_sequence)
    rewards = [LABEL_TURN_REWARDS.get(lbl, 0.0) for lbl in label_sequence]
    rewards[-1] += terminal_reward

    returns: List[float] = [0.0] * T
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = float(np.clip(G, -reward_clip, reward_clip))
    return returns


# =============================================================================
# Trajectory generation
# =============================================================================

def generate_trajectories(
    df:             pd.DataFrame,
    tokenizer,
    n_trajectories: int = RL_N_TRAJECTORIES,
    max_len:        int = MAX_LEN,
) -> List[Trajectory]:
    """
    Generate synthetic multi-turn conversation trajectories from training rows.

    Each trajectory follows a conversation pattern from rl_config.py.
    For every label slot we sample a real training row with that label,
    tokenise its input_text, and attach the discounted return vector.
    """
    random.seed(RL_SEED)
    np.random.seed(RL_SEED)

    label_pools: Dict[str, List[dict]] = {}
    for lbl in LABEL_NAMES:
        rows = df[df["label_name"] == lbl].to_dict("records")
        if rows:
            label_pools[lbl] = rows

    missing = [l for l in LABEL_NAMES if l not in label_pools]
    if missing:
        log.warning("Labels with no training samples: %s", missing)

    pattern_list    = [(seq, r) for seq, r in CONVERSATION_PATTERNS]
    pattern_indices = list(range(len(pattern_list)))

    trajectories: List[Trajectory] = []
    skipped = 0

    for _ in range(n_trajectories):
        idx             = np.random.choice(pattern_indices, p=PATTERN_PROBS)
        label_seq, term_r = pattern_list[idx]

        turns: List[Turn] = []
        for lbl in label_seq:
            pool = label_pools.get(lbl)
            if not pool:
                break
            row = random.choice(pool)
            enc = tokenizer(
                str(row["input_text"]),
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            turns.append(Turn(
                input_ids=enc["input_ids"].squeeze(0),
                attention_mask=enc["attention_mask"].squeeze(0),
                label_id=int(row["label"]),
                label_name=lbl,
            ))

        if len(turns) < RL_MIN_TRAJ_LEN:
            skipped += 1
            continue

        returns = compute_discounted_returns(
            [t.label_name for t in turns], term_r
        )
        trajectories.append(Trajectory(
            turns=turns,
            terminal_reward=term_r,
            returns=returns,
        ))

    log.info("Generated %d trajectories (%d skipped).", len(trajectories), skipped)
    rewards = [t.terminal_reward for t in trajectories]
    log.info(
        "Terminal reward stats — mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards),
    )
    return trajectories


# =============================================================================
# Dataset — flattened (state, action, return) triples
# =============================================================================

class TrajectoryDataset(Dataset):
    """Flattens Trajectory objects into individual (turn, G_t) samples."""

    def __init__(self, trajectories: List[Trajectory]):
        self.input_ids_list:      List[torch.Tensor] = []
        self.attention_mask_list: List[torch.Tensor] = []
        self.label_ids:           List[int]          = []
        self.returns:             List[float]        = []

        for traj in trajectories:
            for turn, G_t in zip(traj.turns, traj.returns):
                self.input_ids_list.append(turn.input_ids)
                self.attention_mask_list.append(turn.attention_mask)
                self.label_ids.append(turn.label_id)
                self.returns.append(G_t)

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.input_ids_list[idx],
            "attention_mask": self.attention_mask_list[idx],
            "label_id":       torch.tensor(self.label_ids[idx],  dtype=torch.long),
            "G_t":            torch.tensor(self.returns[idx],    dtype=torch.float32),
        }


# =============================================================================
# Layer freezing (SAFEGUARD 1)
# =============================================================================

def apply_layer_freezing(model) -> Tuple[int, int]:
    """
    Freeze all transformer layers, keep only the classification head trainable.

    DistilBERT layer layout:
      distilbert.embeddings             → FROZEN (23M params)
      distilbert.transformer.layer[0-5] → FROZEN (43M params)
      pre_classifier  (768×768 Linear) → TRAINABLE (590K params)
      classifier      (768×8 Linear)   → TRAINABLE (6K params)

    Returns (frozen_count, trainable_count)
    """
    for name, param in model.named_parameters():
        if "classifier" in name or "pre_classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, trainable


# =============================================================================
# Evaluation helpers
# =============================================================================

class _CsvDataset(Dataset):
    def __init__(self, source, tokenizer):
        df = pd.read_csv(source) if isinstance(source, (str, Path)) else source
        self.texts  = df["input_text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(
            str(self.texts[idx]),
            max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate_per_turn(model, source, tokenizer, batch_size: int = 64) -> dict:
    """Standard per-turn accuracy evaluation (same metric as SFT training)."""
    model.eval()
    ds     = _CsvDataset(source, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    return {
        "accuracy":    accuracy_score(all_labels, all_preds),
        "macro_f1":    f1_score(all_labels, all_preds, average="macro",    zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "report":      classification_report(
            all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0
        ),
    }


def evaluate_conversation_metrics(
    model, trajectories: List[Trajectory], sample_size: int = 500
) -> dict:
    """
    Three conversation-level RL metrics (beyond per-turn accuracy):

    conversation_success_rate (CSR)
        Fraction of turns where the model's argmax matches the
        trajectory-defining label — measures policy-trajectory alignment.

    mean_policy_expected_reward
        E_π[G_t] = Σ_t π(a_t|s_t)·G_t / T per trajectory, then averaged.
        Directly measures whether the policy gravitates toward high-reward paths.

    mean_policy_entropy
        H(π) = -Σ π log π averaged across all turns.
        Monitors for policy collapse (should remain > 0.1 throughout training).
    """
    model.eval()
    rng    = random.Random(RL_SEED)
    sample = rng.sample(trajectories, min(sample_size, len(trajectories)))

    csr_hits, policy_rewards, entropies = [], [], []

    with torch.no_grad():
        for traj in sample:
            traj_policy_reward = 0.0
            traj_correct       = 0

            for turn, G_t in zip(traj.turns, traj.returns):
                ids      = turn.input_ids.unsqueeze(0).to(DEVICE)
                mask     = turn.attention_mask.unsqueeze(0).to(DEVICE)
                logits   = model(input_ids=ids, attention_mask=mask).logits[0]
                probs    = F.softmax(logits, dim=-1)
                log_probs= F.log_softmax(logits, dim=-1)

                traj_policy_reward += probs[turn.label_id].item() * G_t
                if torch.argmax(probs).item() == turn.label_id:
                    traj_correct += 1
                entropies.append(-(probs * log_probs).sum().item())

            traj_policy_reward /= max(len(traj.turns), 1)
            policy_rewards.append(traj_policy_reward)
            csr_hits.append(traj_correct / len(traj.turns))

    return {
        "conversation_success_rate":   float(np.mean(csr_hits)),
        "mean_policy_expected_reward": float(np.mean(policy_rewards)),
        "mean_policy_entropy":         float(np.mean(entropies)),
    }


# =============================================================================
# Single RL training epoch (REINFORCE + SFT + Entropy)
# =============================================================================

def train_rl_epoch(
    model, loader: DataLoader, optimizer, scheduler
) -> Tuple[float, float, float, float]:
    """
    One epoch of combined SFT + REINFORCE + Entropy training.

    Returns (mean_total_loss, mean_sft_loss, mean_rl_loss, mean_entropy_loss)
    """
    model.train()
    total_loss = sft_total = rl_total = entropy_total = 0.0
    n = 0

    # Compute reward baseline as mean of returns in this epoch's batch
    # (running baseline reduces gradient variance)
    all_returns = []

    for batch in loader:
        all_returns.extend(batch["G_t"].tolist())
    baseline = float(np.mean(all_returns)) if all_returns else 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label_id"].to(DEVICE)
        returns        = batch["G_t"].to(DEVICE)

        outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
        logits   = outputs.logits                          # (B, 8)

        # ── Supervised cross-entropy loss (SFT signal preserved) ──────────
        L_sft = F.cross_entropy(logits, labels)

        # ── REINFORCE policy gradient loss ─────────────────────────────────
        # Advantage = G_t - baseline  (variance reduction)
        log_probs        = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        advantages       = returns - baseline
        L_rl             = -(advantages * action_log_probs).mean()

        # ── SAFEGUARD: Entropy regularisation (prevents policy collapse) ───
        # Without this, RL can make the model over-confident on one label.
        # β·H(π) added as bonus (negative loss) to keep distribution spread.
        probs    = F.softmax(logits, dim=-1)
        entropy  = -(probs * log_probs).sum(dim=-1).mean()
        L_entropy = -RL_ENTROPY_BETA * entropy             # negative = maximise

        # ── Combined loss ──────────────────────────────────────────────────
        L = L_sft + RL_LAMBDA * L_rl + L_entropy

        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            RL_MAX_GRAD_NORM
        )
        optimizer.step()
        scheduler.step()

        total_loss    += L.item()
        sft_total     += L_sft.item()
        rl_total      += L_rl.item()
        entropy_total += entropy.item()
        n             += 1

    return (
        total_loss    / max(n, 1),
        sft_total     / max(n, 1),
        rl_total      / max(n, 1),
        entropy_total / max(n, 1),
    )


# =============================================================================
# Main fine-tuning pipeline
# =============================================================================

def run_rl_finetuning(
    sft_model_dir:  str,
    data_dir:       str,
    output_dir:     str,
    results_dir:    str,
    n_trajectories: int = RL_N_TRAJECTORIES,
) -> dict:
    """
    End-to-end RL fine-tuning pipeline.

    1. Load SFT checkpoint.
    2. Apply layer freezing (SAFEGUARD 1).
    3. Generate synthetic trajectories.
    4. Evaluate SFT baseline (per-turn + conversation metrics).
    5. REINFORCE fine-tuning with combined SFT + RL + Entropy loss.
    6. Evaluate RL model on same metrics.
    7. Save best checkpoint and results JSON.
    """
    random.seed(RL_SEED)
    np.random.seed(RL_SEED)
    torch.manual_seed(RL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RL_SEED)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    log.info("Device: %s", DEVICE)
    log.info("Loading SFT checkpoint: %s", sft_model_dir)

    # ── 1. Load model ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(sft_model_dir)
    model.to(DEVICE)

    # ── 2. SAFEGUARD: Apply selective layer freezing ───────────────────────
    if RL_FREEZE_TRANSFORMER:
        frozen, trainable = apply_layer_freezing(model)
        log.info(
            "Layer freezing applied — Frozen: %s params | Trainable: %s params",
            f"{frozen:,}", f"{trainable:,}"
        )
    else:
        trainable = sum(p.numel() for p in model.parameters())
        log.warning(
            "RL_FREEZE_TRANSFORMER=False — training all %s params. "
            "Higher risk of catastrophic forgetting.", f"{trainable:,}"
        )

    # ── 3. Find training data ──────────────────────────────────────────────
    data_path = Path(data_dir)
    for candidate in [
        data_path / "v3_train_full_50k.csv",
        data_path / "v2_train_augmented.csv",
    ] + list(data_path.glob("*train*.csv")):
        if candidate.exists():
            train_csv = candidate
            break
    else:
        raise FileNotFoundError(f"No training CSV found in {data_dir}")
    log.info("Training data: %s", train_csv)

    test_csv = data_path / "v2_test_augmented.csv"
    if not test_csv.exists():
        test_csv = data_path / "v2_val_augmented.csv"
    log.info("Test data: %s", test_csv)

    df_train = pd.read_csv(train_csv)
    log.info("Loaded %d training rows.", len(df_train))

    # ── 4. Generate trajectories ───────────────────────────────────────────
    log.info("Generating %d synthetic trajectories…", n_trajectories)
    all_trajs = generate_trajectories(df_train, tokenizer, n_trajectories)

    random.shuffle(all_trajs)
    split       = int(0.8 * len(all_trajs))
    train_trajs = all_trajs[:split]
    val_trajs   = all_trajs[split:]
    log.info("%d train / %d val trajectories.", len(train_trajs), len(val_trajs))

    # ── 5. SFT Baseline evaluation ─────────────────────────────────────────
    log.info("\n%s\nSFT BASELINE EVALUATION\n%s", "=" * 60, "=" * 60)
    baseline_pt = evaluate_per_turn(model, test_csv, tokenizer)
    baseline_cv = evaluate_conversation_metrics(model, val_trajs, sample_size=500)

    log.info("SFT  Accuracy  : %.4f", baseline_pt["accuracy"])
    log.info("SFT  Macro-F1  : %.4f", baseline_pt["macro_f1"])
    log.info("SFT  CSR       : %.4f", baseline_cv["conversation_success_rate"])
    log.info("SFT  E[reward] : %.4f", baseline_cv["mean_policy_expected_reward"])
    log.info("SFT  Entropy   : %.4f", baseline_cv["mean_policy_entropy"])

    # ── 6. Build DataLoaders ───────────────────────────────────────────────
    train_ds     = TrajectoryDataset(train_trajs)
    val_ds       = TrajectoryDataset(val_trajs)
    train_loader = DataLoader(train_ds, batch_size=RL_BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=RL_BATCH_SIZE, shuffle=False, num_workers=0)
    log.info("TrajectoryDataset: %d train / %d val samples.", len(train_ds), len(val_ds))

    # ── 7. Optimiser + scheduler (only trainable params) ──────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.AdamW(trainable_params, lr=RL_LR, weight_decay=0.01)
    total_steps  = len(train_loader) * RL_EPOCHS
    warmup_steps = int(total_steps * RL_WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── 8. RL training loop ────────────────────────────────────────────────
    best_reward = baseline_cv["mean_policy_expected_reward"]
    history     = {
        "epoch": [], "train_loss": [], "sft_loss": [],
        "rl_loss": [], "entropy": [],
        "val_accuracy": [], "val_macro_f1": [],
        "val_csr": [], "val_mean_reward": [], "val_entropy": [],
    }

    for epoch in range(1, RL_EPOCHS + 1):
        log.info("\n%s\nRL EPOCH %d / %d\n%s", "=" * 60, epoch, RL_EPOCHS, "=" * 60)

        t_loss, s_loss, r_loss, ent = train_rl_epoch(
            model, train_loader, optimizer, scheduler
        )
        log.info(
            "Train — total: %.4f  sft: %.4f  rl: %.4f  entropy: %.4f",
            t_loss, s_loss, r_loss, ent
        )

        pt_m = evaluate_per_turn(model, test_csv, tokenizer)
        cv_m = evaluate_conversation_metrics(model, val_trajs, sample_size=500)

        log.info(
            "Val — acc: %.4f  macro-F1: %.4f  CSR: %.4f  E[reward]: %.4f  entropy: %.4f",
            pt_m["accuracy"], pt_m["macro_f1"],
            cv_m["conversation_success_rate"],
            cv_m["mean_policy_expected_reward"],
            cv_m["mean_policy_entropy"],
        )

        # ── Safety check: alert if per-turn accuracy drops more than 0.5% ─
        acc_drop = baseline_pt["accuracy"] - pt_m["accuracy"]
        if acc_drop > 0.005:
            log.warning(
                "⚠ Per-turn accuracy dropped by %.2f%% (%.4f → %.4f). "
                "Consider reducing RL_LAMBDA or enabling RL_FREEZE_TRANSFORMER.",
                acc_drop * 100, baseline_pt["accuracy"], pt_m["accuracy"]
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(t_loss)
        history["sft_loss"].append(s_loss)
        history["rl_loss"].append(r_loss)
        history["entropy"].append(ent)
        history["val_accuracy"].append(pt_m["accuracy"])
        history["val_macro_f1"].append(pt_m["macro_f1"])
        history["val_csr"].append(cv_m["conversation_success_rate"])
        history["val_mean_reward"].append(cv_m["mean_policy_expected_reward"])
        history["val_entropy"].append(cv_m["mean_policy_entropy"])

        # Save best model by conversation-level expected reward
        if cv_m["mean_policy_expected_reward"] > best_reward:
            best_reward = cv_m["mean_policy_expected_reward"]
            best_dir    = Path(output_dir) / "best_rl_model"
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            log.info(
                "✓ Saved new best RL model (E[reward]=%.4f) → %s",
                best_reward, best_dir
            )

    # ── 9. Final evaluation ────────────────────────────────────────────────
    log.info("\n%s\nFINAL RL MODEL EVALUATION\n%s", "=" * 60, "=" * 60)
    final_pt = evaluate_per_turn(model, test_csv, tokenizer)
    final_cv = evaluate_conversation_metrics(model, val_trajs, sample_size=1000)

    # ── 10. Comparison table ───────────────────────────────────────────────
    _SEP = "=" * 65
    print(f"\n{_SEP}")
    print("  RL FINE-TUNING RESULTS SUMMARY")
    print(_SEP)
    print(f"  {'Metric':<40}  {'SFT':>8}  {'RL':>8}  {'Δ':>8}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")

    def _row(label, base, rl_val, pct=False):
        delta = rl_val - base
        b_str = f"{base*100:.2f}%"   if pct else f"{base:.4f}"
        r_str = f"{rl_val*100:.2f}%" if pct else f"{rl_val:.4f}"
        d_str = f"{delta*100:+.2f}pp" if pct else f"{delta:+.4f}"
        print(f"  {label:<40}  {b_str:>8}  {r_str:>8}  {d_str:>8}")

    _row("Per-turn Accuracy",           baseline_pt["accuracy"],  final_pt["accuracy"],  pct=True)
    _row("Per-turn Macro-F1",           baseline_pt["macro_f1"],  final_pt["macro_f1"])
    _row("Conversation Success Rate",   baseline_cv["conversation_success_rate"],
                                        final_cv["conversation_success_rate"],    pct=True)
    _row("Mean Policy Expected Reward", baseline_cv["mean_policy_expected_reward"],
                                        final_cv["mean_policy_expected_reward"])
    _row("Policy Entropy",              baseline_cv["mean_policy_entropy"],
                                        final_cv["mean_policy_entropy"])
    print(_SEP + "\n")
    print("Per-class report (RL model):")
    print(final_pt["report"])

    # ── 11. Save results JSON ──────────────────────────────────────────────
    results = {
        "baseline": {
            "accuracy":                   baseline_pt["accuracy"],
            "macro_f1":                   baseline_pt["macro_f1"],
            "weighted_f1":                baseline_pt["weighted_f1"],
            "conversation_success_rate":  baseline_cv["conversation_success_rate"],
            "mean_policy_expected_reward":baseline_cv["mean_policy_expected_reward"],
            "mean_policy_entropy":        baseline_cv["mean_policy_entropy"],
        },
        "rl_finetuned": {
            "accuracy":                   final_pt["accuracy"],
            "macro_f1":                   final_pt["macro_f1"],
            "weighted_f1":                final_pt["weighted_f1"],
            "conversation_success_rate":  final_cv["conversation_success_rate"],
            "mean_policy_expected_reward":final_cv["mean_policy_expected_reward"],
            "mean_policy_entropy":        final_cv["mean_policy_entropy"],
        },
        "training_history": history,
        "rl_config": {
            "lambda":               RL_LAMBDA,
            "gamma":                RL_GAMMA,
            "lr":                   RL_LR,
            "epochs":               RL_EPOCHS,
            "batch_size":           RL_BATCH_SIZE,
            "n_trajectories":       n_trajectories,
            "reward_clip":          RL_REWARD_CLIP,
            "freeze_transformer":   RL_FREEZE_TRANSFORMER,
            "entropy_beta":         RL_ENTROPY_BETA,
        },
    }

    results_path = Path(results_dir) / "rl_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", results_path)

    return results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="REINFORCE fine-tuning for DistilBERT CRS classifier."
    )
    parser.add_argument(
        "--sft_model_dir", default=os.path.join(OUTPUT_DIR, "best_model"),
        help="Path to SFT-trained model checkpoint directory.",
    )
    parser.add_argument(
        "--data_dir", default=DATA_DIR,
        help="Directory containing training and test CSV files.",
    )
    parser.add_argument(
        "--output_dir", default=RL_OUTPUT_DIR,
        help="Directory to save the RL fine-tuned model.",
    )
    parser.add_argument(
        "--results_dir", default=RL_RESULTS_DIR,
        help="Directory to save evaluation results JSON.",
    )
    parser.add_argument(
        "--n_trajectories", type=int, default=RL_N_TRAJECTORIES,
        help="Number of synthetic trajectories to generate.",
    )
    args = parser.parse_args()

    run_rl_finetuning(
        sft_model_dir=args.sft_model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        n_trajectories=args.n_trajectories,
    )
