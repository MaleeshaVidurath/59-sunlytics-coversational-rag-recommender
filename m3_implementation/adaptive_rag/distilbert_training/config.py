# =============================================================================
# config.py
# =============================================================================
# ALL hyperparameters, paths, and constants live here.
# Never hardcode these values inside training or evaluation scripts.
# If you need to experiment (e.g. change batch size or learning rate),
# you change ONE number here and every other script automatically uses it.
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Set BASE_DIR to the folder that contains your data files.
# os.path.dirname(__file__) means "the folder this config.py file lives in".
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_FILE = os.path.join(DATA_DIR, "v2_train_augmented.csv")
VAL_FILE   = os.path.join(DATA_DIR, "v2_val_augmented.csv")
TEST_FILE  = os.path.join(DATA_DIR, "v2_test_augmented.csv")

# Where the best model checkpoint gets saved during training
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "best_model")

# Where evaluation plots and reports are saved
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# ── Model ─────────────────────────────────────────────────────────────────────
# We use distilbert-base-uncased because:
#   - It is 40% smaller than BERT-base (66M vs 110M parameters)
#   - It is 60% faster at inference
#   - It retains 97% of BERT's language understanding on GLUE benchmarks
#   - The MBA-RAG paper (which your research cites) validates DistilBERT
#     as a viable retrieval strategy selector
PRETRAINED_MODEL = "distilbert-base-uncased"

# ── Classification labels ──────────────────────────────────────────────────────
# This must match EXACTLY the label integers in your CSV files.
# The order matters: index 0 = INITIAL_REQUEST, index 1 = REFINEMENT, etc.
LABEL_NAMES = [
    "INITIAL_REQUEST",      # 0 → FULL retrieval
    "REFINEMENT",           # 1 → FULL retrieval
    "ATTRIBUTE_QUESTION",   # 2 → PARTIAL retrieval
    "EXPLANATION_WHY",      # 3 → PARTIAL retrieval
    "COMPARISON",           # 4 → PARTIAL retrieval
    "SELECTION_REFERENCE",  # 5 → PARTIAL retrieval
    "FEEDBACK",             # 6 → NO retrieval
    "CHITCHAT",             # 7 → NO retrieval
]
NUM_LABELS = len(LABEL_NAMES)   # 8

# Maps each label integer to its retrieval strategy (used during inference)
RETRIEVAL_STRATEGY_MAP = {
    0: "FULL",
    1: "FULL",
    2: "PARTIAL",
    3: "PARTIAL",
    4: "PARTIAL",
    5: "PARTIAL",
    6: "NO",
    7: "NO",
}

# ── Tokenizer settings ────────────────────────────────────────────────────────
# MAX_LEN: maximum number of tokens per input sequence.
# Your dataset's longest input is ~246 approximate tokens (chars/4).
# 256 gives a safe margin while being much smaller than DistilBERT's 512 limit.
# Smaller MAX_LEN = faster training and less GPU memory.
MAX_LEN = 256

# ── Training hyperparameters ──────────────────────────────────────────────────
# BATCH_SIZE: how many samples are processed together in one forward pass.
#   - 32 is a standard choice for sequence classification fine-tuning.
#   - If you get CUDA out-of-memory errors, reduce to 16.
BATCH_SIZE     = 32

# LEARNING_RATE: how fast the model's weights are updated each step.
#   - 2e-5 is the standard recommendation from the original BERT paper
#     for fine-tuning on classification tasks.
#   - Too high (e.g. 1e-4) → unstable training, loss spikes.
#   - Too low (e.g. 1e-6) → model barely learns from your data.
LEARNING_RATE  = 2e-5

# NUM_EPOCHS: how many complete passes through the training data.
#   - 5 epochs is standard for DistilBERT fine-tuning on ~7K samples.
#   - Early stopping will stop training automatically if val_loss
#     stops improving, so 5 is a safe upper bound.
NUM_EPOCHS     = 5

# WEIGHT_DECAY: L2 regularisation to prevent overfitting.
#   - 0.01 is the standard value from the BERT fine-tuning paper.
WEIGHT_DECAY   = 0.01

# WARMUP_RATIO: fraction of total training steps used for learning rate warmup.
#   - During warmup, the LR increases gradually from 0 to LEARNING_RATE.
#   - This prevents the model from making large destructive weight updates
#     at the very start of training when gradients can be noisy.
#   - 0.1 means 10% of total steps are warmup steps.
WARMUP_RATIO   = 0.1

# EARLY_STOPPING_PATIENCE: how many epochs without improvement before stopping.
#   - If validation macro-F1 does not improve for 3 consecutive epochs, stop.
EARLY_STOPPING_PATIENCE = 3

# ── Reproducibility ───────────────────────────────────────────────────────────
# Setting a fixed seed ensures you get the same results every time you run.
# This is important for a final year project where you need reproducible results.
SEED = 42
