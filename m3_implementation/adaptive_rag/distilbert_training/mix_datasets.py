"""
mix_datasets.py
───────────────
Creates v5_train_mixed.csv by combining:
  - real_data_simmc.csv  (SIMMC 2.1 real fashion dialogues)
  - v4_train_midSession.csv (synthetic data — only for labels missing in SIMMC)

Balancing rule (TARGET_PER_LABEL = 3000):
  - Label has >= 3000 SIMMC rows  -> randomly sample 3000 from SIMMC only
  - Label has 0 < n < 3000 SIMMC  -> use all SIMMC + top up from synthetic
  - Label has 0 SIMMC rows        -> take 3000 from synthetic only

This maximises real data usage while keeping all 8 labels balanced at 3000 rows.

Output: data/v5_train_mixed.csv

Usage:
    python mix_datasets.py
"""

import os
import pandas as pd

_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_SIMMC_CSV      = os.path.join(_BASE_DIR, "data", "real_data_simmc.csv")
_SYNTHETIC_CSV  = os.path.join(_BASE_DIR, "data", "v4_train_midSession.csv")
_OUTPUT_CSV     = os.path.join(_BASE_DIR, "data", "v5_train_mixed.csv")

TARGET_PER_LABEL = 3000
RANDOM_SEED      = 42

LABEL_NAMES = [
    "INITIAL_REQUEST",
    "REFINEMENT",
    "ATTRIBUTE_QUESTION",
    "EXPLANATION_WHY",
    "COMPARISON",
    "SELECTION_REFERENCE",
    "FEEDBACK",
    "CHITCHAT",
]


def main():
    print("=" * 60)
    print("Dataset Mixing: SIMMC real + Synthetic")
    print(f"Target per label : {TARGET_PER_LABEL}")
    print("=" * 60)

    # ── Load source datasets ──────────────────────────────────────────────────
    print("\nLoading source datasets...")
    simmc     = pd.read_csv(_SIMMC_CSV)
    synthetic = pd.read_csv(_SYNTHETIC_CSV)
    print(f"  SIMMC rows     : {len(simmc)}")
    print(f"  Synthetic rows : {len(synthetic)}")

    # ── Mix per label ─────────────────────────────────────────────────────────
    print("\nMixing per label:")
    mixed_parts = []

    for label in LABEL_NAMES:
        simmc_rows     = simmc[simmc["label_name"] == label]
        synthetic_rows = synthetic[synthetic["label_name"] == label]

        n_simmc     = len(simmc_rows)
        n_synthetic = len(synthetic_rows)

        if n_simmc >= TARGET_PER_LABEL:
            # Enough real data — sample down, no synthetic needed
            chosen = simmc_rows.sample(TARGET_PER_LABEL, random_state=RANDOM_SEED)
            source_note = f"SIMMC {TARGET_PER_LABEL} (capped from {n_simmc})"

        elif n_simmc > 0:
            # Some real data — use all + top up from synthetic
            needed      = TARGET_PER_LABEL - n_simmc
            top_up      = synthetic_rows.sample(
                min(needed, n_synthetic), random_state=RANDOM_SEED
            )
            chosen      = pd.concat([simmc_rows, top_up], ignore_index=True)
            source_note = f"SIMMC {n_simmc} + Synthetic {len(top_up)}"

        else:
            # No real data — synthetic only
            chosen      = synthetic_rows.sample(
                min(TARGET_PER_LABEL, n_synthetic), random_state=RANDOM_SEED
            )
            source_note = f"Synthetic {len(chosen)} (no SIMMC data)"

        mixed_parts.append(chosen)
        print(f"  {label:<25}  {len(chosen):>5} rows  |  {source_note}")

    # ── Combine and shuffle ───────────────────────────────────────────────────
    combined = pd.concat(mixed_parts, ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ── Keep only the training columns ───────────────────────────────────────
    keep_cols = [
        "input_text", "current_message", "conversation_history_json",
        "label", "label_name", "retrieval_strategy", "exchanges",
    ]
    combined = combined[[c for c in keep_cols if c in combined.columns]]

    # ── Save ──────────────────────────────────────────────────────────────────
    combined.to_csv(_OUTPUT_CSV, index=False)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final dataset summary")
    print("=" * 60)
    print(f"\nTotal rows : {len(combined)}")
    print("\nLabel distribution:")
    dist = combined["label_name"].value_counts()
    for label in LABEL_NAMES:
        count = dist.get(label, 0)
        bar   = "#" * (count // 75)
        print(f"  {label:<25} {count:>5}  {bar}")

    real_count      = len(simmc[simmc["label_name"].isin(LABEL_NAMES)])
    real_in_output  = sum(
        min(len(simmc[simmc["label_name"] == l]), TARGET_PER_LABEL)
        for l in LABEL_NAMES
    )
    synthetic_count = len(combined) - real_in_output

    print(f"\nReal data rows    : {real_in_output} ({real_in_output/len(combined)*100:.1f}%)")
    print(f"Synthetic rows    : {synthetic_count} ({synthetic_count/len(combined)*100:.1f}%)")
    print(f"\nSaved: {_OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
