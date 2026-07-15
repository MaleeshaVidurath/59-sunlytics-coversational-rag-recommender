# m3_implementation/test_result/hallucination_result/expanded_eval/make_4000_subset.py
#
# Reduces the expanded sets to a 4,000-row evaluation suite (user decision:
# 4,000 rows are sufficient; also roughly halves compute time).
#
# RULES (documented for the write-up):
#   - ALL 526 clean cases are kept — they anchor precision and the human
#     audit; only corrupted variants are down-sampled.
#   - Corrupted rows are sampled EVENLY PER CORRUPTION TYPE (seeded, 42) so
#     the per-type recall breakdowns keep balanced support.
#   - Targets: standard 2,600 rows (526 clean + 2,074 corrupted),
#              hard 1,400 rows → total 4,000.
#   - The full sets are preserved as *_full.jsonl next to the subsets.
#
# Run:  python test_result/hallucination_result/expanded_eval/make_4000_subset.py

import json
import os
import random
import shutil
from collections import defaultdict

_DIR = os.path.dirname(os.path.abspath(__file__))
STD  = os.path.join(_DIR, "labeled_test_set_expanded.jsonl")
HARD = os.path.join(_DIR, "hard_set", "labeled_hard_set.jsonl")

SEED = 42
TARGET_STD_CORRUPTED = 2074   # + 526 clean = 2600
TARGET_HARD          = 1400   # corrupted only → grand total 4000


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _sample_per_type(rows, target, rng):
    """Even per-corruption-type sampling to `target` total."""
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["corruption"]["type"]].append(r)
    types = sorted(by_type)
    quota, extra = divmod(target, len(types))
    picked = []
    for i, t in enumerate(types):
        k = quota + (1 if i < extra else 0)
        pool = by_type[t]
        picked.extend(pool if len(pool) <= k else rng.sample(pool, k))
    return picked


def _shrink(path, keep_clean: bool, target_corrupted: int, rng):
    rows = _load(path)
    full_path = path.replace(".jsonl", "_full.jsonl")
    if not os.path.exists(full_path):
        shutil.copy2(path, full_path)

    clean = [r for r in rows if r["label"] == "clean"] if keep_clean else []
    corrupted = [r for r in rows if r["label"] == "hallucinated"]
    picked = _sample_per_type(corrupted, target_corrupted, rng)
    subset = clean + picked
    _save(path, subset)

    from collections import Counter
    by = Counter(r["corruption"]["type"] for r in picked)
    print(f"{os.path.basename(path)}: {len(rows)} → {len(subset)} rows "
          f"({len(clean)} clean + {len(picked)} corrupted)")
    for t, n in sorted(by.items()):
        print(f"   {t:22s} {n}")
    return len(subset)


def main():
    rng = random.Random(SEED)
    n_std  = _shrink(STD,  keep_clean=True,  target_corrupted=TARGET_STD_CORRUPTED, rng=rng)
    n_hard = _shrink(HARD, keep_clean=False, target_corrupted=TARGET_HARD, rng=rng)
    print(f"\nTOTAL: {n_std + n_hard} rows (full sets preserved as *_full.jsonl)")


if __name__ == "__main__":
    main()
