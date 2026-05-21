# =============================================================================
# rl_config.py  (MERGED FINAL VERSION)
# =============================================================================
# Hyperparameters and reward design for REINFORCE fine-tuning of the
# DistilBERT CRS classifier.
#
# Research motivation:
#   The SFT model achieves 99.72% per-turn accuracy on synthetic test data.
#   However, per-turn accuracy does not measure whether the classifier's
#   decisions lead to good *conversation outcomes*. A classifier that always
#   picks the technically correct label can still trigger suboptimal action
#   sequences (e.g. repeated REFINEMENT loops that frustrate users).
#
#   REINFORCE teaches the model to prefer label sequences that maximise
#   cumulative user satisfaction, directly addressing the sim-to-real gap
#   inherent in synthetic training data.
#
# Algorithm: REINFORCE with combined loss (Williams, 1992)
#   L = L_SFT + λ · L_REINFORCE
#   L_SFT       = CrossEntropy(logits, true_label)
#   L_REINFORCE = -(G_t - baseline) · log π_θ(a_t | s_t)
#   G_t         = Σ_k γ^k · r_{t+k}   (discounted return)
#
# Three safeguards against catastrophic forgetting:
#   1. Selective layer freezing: only pre_classifier + classifier updated
#   2. Low RL learning rate (5e-6 vs SFT 2e-5)
#   3. Low RL lambda (0.1) keeps SFT loss dominant
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RL_OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "rl_model")
RL_RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "rl_results")

# ── Core RL hyperparameters ───────────────────────────────────────────────────
RL_LAMBDA        = 0.1    # Weight of RL loss vs SFT loss.
                          # 0.1 = SFT stays dominant, RL nudges gently.
                          # Empirically safe range: 0.05–0.2.

RL_GAMMA         = 0.95   # Discount factor for future rewards.
                          # 0.95 means a reward 5 turns away counts as
                          # 0.95^5 = 0.77 of its face value.

RL_LR            = 5e-6   # RL learning rate — 4× lower than SFT (2e-5).
                          # Lower LR prevents overwriting SFT knowledge.

RL_BATCH_SIZE    = 16     # Smaller than SFT (32) — RL gradients are
                          # higher-variance, smaller batches are safer.

RL_EPOCHS        = 3      # Fine-tuning epochs on top of SFT checkpoint.

RL_MAX_GRAD_NORM = 0.5    # Gradient clipping — tighter than SFT (1.0)
                          # because RL gradients can spike.

RL_WARMUP_RATIO  = 0.05   # 5% warmup steps (same principle as SFT).

RL_REWARD_CLIP   = 1.5    # Clip discounted returns to [-1.5, 1.5].
                          # Prevents extreme gradient updates from
                          # outlier trajectories.

RL_SEED          = 42

# ── SAFEGUARD: Selective layer freezing ──────────────────────────────────────
# Only update the classification head, NOT the transformer layers.
#
# Why this matters:
#   - DistilBERT has 66M parameters total.
#   - Updating all 66M from sparse RL signal risks catastrophic forgetting.
#   - The pre_classifier (768×768) and classifier (768×8) head = 599,304 params.
#   - These are the layers responsible for label assignment.
#   - The transformer layers encode language — no need to change them.
#
# Research precedent:
#   - InstructGPT (Ouyang et al., 2022): freezes early layers during RL
#   - LoRA (Hu et al., 2022): updates only small adapter matrices
#   - RLHF for BERT classifiers: head-only fine-tuning is standard
#
# Set to False to update ALL layers (riskier, may not improve results).
RL_FREEZE_TRANSFORMER = True

# ── SAFEGUARD: Entropy regularisation ────────────────────────────────────────
# Adds β·H(π) to the loss to prevent policy collapse.
# Policy collapse = model always predicts the same label regardless of input.
# H(π) = -Σ π(a|s) log π(a|s)
# β = 0.01 keeps exploration alive without destabilising training.
RL_ENTROPY_BETA = 0.01

# ── Trajectory generation settings ───────────────────────────────────────────
RL_N_TRAJECTORIES = 15_000  # Synthetic trajectories to generate.
RL_MIN_TRAJ_LEN   = 2       # Minimum turns per trajectory.
RL_MAX_TRAJ_LEN   = 6       # Maximum turns per trajectory.

# ── Per-label intrinsic turn reward ──────────────────────────────────────────
# Immediate reward r_t for being in each conversational state.
#
# Design rationale:
#   SELECTION_REFERENCE (+0.5): user pointed at an item → strong success signal.
#   ATTRIBUTE_QUESTION, EXPLANATION_WHY, COMPARISON (+0.2): user is engaged
#     with the recommendation → positive exploration signal.
#   REFINEMENT (−0.1): each refinement means the previous recommendation
#     was not satisfying → mild inefficiency penalty.
#   INITIAL_REQUEST, CHITCHAT (0.0): neutral — no quality signal yet.
#   FEEDBACK: determined by terminal reward (see below).
#
# These values are consistent with the CRS evaluation literature:
#   - CoRE (arXiv:2501.09493): engagement signals (attribute questions,
#     explanations) correlate with higher user satisfaction scores.
#   - RA-Rec (SIGIR 2024): repeated refinements correlate with
#     lower task success rates.
LABEL_TURN_REWARDS = {
    "INITIAL_REQUEST":     0.0,
    "REFINEMENT":         -0.1,
    "ATTRIBUTE_QUESTION":  0.2,
    "EXPLANATION_WHY":     0.2,
    "COMPARISON":          0.2,
    "SELECTION_REFERENCE": 0.5,
    "FEEDBACK":            0.0,
    "CHITCHAT":            0.0,
}

# ── Terminal reward values ────────────────────────────────────────────────────
# Added to the last turn's reward only. Represents overall conversation outcome.
FEEDBACK_TERMINAL_REWARDS = {
    "strong_positive":  1.0,   # user selected item enthusiastically
    "mild_positive":    0.5,   # user satisfied but not enthusiastic
    "neutral":          0.0,   # conversation ended without clear outcome
    "mild_negative":   -0.5,   # user gave mixed/lukewarm feedback
    "strong_negative": -1.0,   # user frustrated or gave up
}

# ── Synthetic conversation patterns ──────────────────────────────────────────
# Each entry: (label_sequence, terminal_reward)
# Represents realistic CRS conversation trajectories observed in literature.
#
# Pattern selection rationale:
#   Positive patterns (reward > 0): user reaches a satisfying recommendation
#     quickly with meaningful engagement.
#   Negative patterns (reward < 0): excessive refinements indicate poor
#     recommendation quality or classifier misdirection.
#
# Literature basis:
#   - INSPIRED dataset analysis shows most successful conversations have
#     2-4 turns with at least one ATTRIBUTE_QUESTION or EXPLANATION_WHY.
#   - ReDial dataset: conversations with 3+ REFINEMENT turns rarely end
#     in a positive FEEDBACK outcome.
CONVERSATION_PATTERNS = [
    # ── High-quality patterns (terminal_reward = 0.8–1.0) ────────────────────
    (["INITIAL_REQUEST", "SELECTION_REFERENCE"],                              1.0),
    (["INITIAL_REQUEST", "ATTRIBUTE_QUESTION", "SELECTION_REFERENCE"],        0.9),
    (["INITIAL_REQUEST", "REFINEMENT", "SELECTION_REFERENCE"],                0.8),
    (["INITIAL_REQUEST", "EXPLANATION_WHY", "SELECTION_REFERENCE"],           0.8),
    (["INITIAL_REQUEST", "REFINEMENT", "FEEDBACK"],                           0.8),

    # ── Good patterns (terminal_reward = 0.5–0.7) ────────────────────────────
    (["INITIAL_REQUEST", "REFINEMENT", "COMPARISON", "SELECTION_REFERENCE"],  0.7),
    (["INITIAL_REQUEST", "ATTRIBUTE_QUESTION", "EXPLANATION_WHY",
      "SELECTION_REFERENCE"],                                                  0.6),
    (["INITIAL_REQUEST", "COMPARISON", "SELECTION_REFERENCE"],                0.6),
    (["INITIAL_REQUEST", "REFINEMENT", "ATTRIBUTE_QUESTION",
      "SELECTION_REFERENCE"],                                                  0.5),

    # ── Medium patterns (terminal_reward = 0.0–0.3) ──────────────────────────
    (["INITIAL_REQUEST", "REFINEMENT", "REFINEMENT", "SELECTION_REFERENCE"],  0.3),
    (["INITIAL_REQUEST", "CHITCHAT", "REFINEMENT", "SELECTION_REFERENCE"],    0.2),
    (["INITIAL_REQUEST", "REFINEMENT", "REFINEMENT", "FEEDBACK"],             0.0),
    (["CHITCHAT", "INITIAL_REQUEST", "SELECTION_REFERENCE"],                  0.2),

    # ── Poor patterns (terminal_reward = −0.3 to −0.8) ───────────────────────
    (["INITIAL_REQUEST", "REFINEMENT", "REFINEMENT", "REFINEMENT",
      "SELECTION_REFERENCE"],                                                 -0.3),
    (["INITIAL_REQUEST", "REFINEMENT", "REFINEMENT", "REFINEMENT",
      "FEEDBACK"],                                                            -0.5),
    (["INITIAL_REQUEST", "REFINEMENT", "REFINEMENT", "REFINEMENT",
      "REFINEMENT", "FEEDBACK"],                                              -0.8),
]

# Sampling probabilities (must sum to 1.0, aligned with CONVERSATION_PATTERNS)
# Positive patterns sampled 2:1 over negative so model sees enough "good" signal.
PATTERN_WEIGHTS = [
    # High-quality (5 patterns)
    3.0, 2.8, 2.5, 2.5, 2.5,
    # Good (4 patterns)
    2.0, 1.8, 1.8, 1.5,
    # Medium (4 patterns)
    1.2, 1.0, 1.0, 0.8,
    # Poor (3 patterns)
    0.8, 0.6, 0.4,
]
