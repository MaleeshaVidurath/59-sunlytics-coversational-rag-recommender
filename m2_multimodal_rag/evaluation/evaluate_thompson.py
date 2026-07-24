"""
Thompson Sampling Bandit + MMR — Adaptive Lambda Demo
Shows how lambda adapts based on session rejection and retention signals.

Run from project root:
    python Accuracy/evaluate_thompson.py
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np.random.seed(42)

from m2_multimodal_rag.diversity_bandit import diversity_bandit

FIXED_LAMBDA = 0.70   # standard MMR baseline

SEP  = "=" * 65
THIN = "-" * 65

# ── Session scenarios ─────────────────────────────────────────────────────────
# Each scenario represents a conversational session state
SCENARIOS = [
    (0, 2, "Session start — user kept 2 items (relevance focus)"),
    (1, 1, "1 rejection, 1 retention — balanced session"),
    (3, 0, "3 rejections — user dissatisfied, needs diversity"),
    (5, 0, "5 rejections — strong diversity signal"),
    (8, 0, "8 rejections — maximum diversity push"),
    (2, 3, "2 rejections + 3 retained — relevance wins"),
]

print(f"\n{SEP}")
print("  NOVELTY 3 — Thompson Sampling Bandit + MMR")
print("  Adaptive λ vs Fixed λ=0.70 across session scenarios")
print(SEP)
print(f"\n  MMR formula: λ × relevance(i) − (1−λ) × max_sim(i, selected)")
print(f"  λ = 1.0 → pure relevance   |   λ = 0.0 → pure diversity\n")

print(f"  {'Scenario':<45} {'Fixed λ':>8} {'TS λ':>8} {'Behaviour'}")
print(f"  {THIN}")

for reject, retain, label in SCENARIOS:
    ts_lambda    = diversity_bandit.sample_lambda(reject, retain)
    diff         = ts_lambda - FIXED_LAMBDA
    behaviour    = "more relevance" if diff > 0.03 else ("more diversity" if diff < -0.03 else "balanced")
    short_label  = label[:43] + ".." if len(label) > 43 else label
    print(f"  {short_label:<45} {FIXED_LAMBDA:>8.3f} {ts_lambda:>8.3f}   {behaviour}")

print(f"\n{SEP}")
print("  POSTERIOR DETAIL — how Beta(α, β) shifts with rejections")
print(SEP)
print(f"\n  Prior: Beta(7.0, 3.0) → E[λ] = 0.700  (matches fixed λ baseline)\n")
print(f"  {'Rejections':>12} {'Retained':>10} {'Alpha':>8} {'Beta':>8} {'E[λ]':>8} {'Direction'}")
print(f"  {THIN}")

for reject, retain, _ in SCENARIOS:
    alpha = 7.0 + retain * 0.5
    beta  = 3.0 + reject * 0.8
    e_lam = np.clip(alpha / (alpha + beta), 0.50, 0.90)
    direction = "↑ relevance" if e_lam > 0.72 else ("↓ diversity" if e_lam < 0.68 else "→ balanced")
    print(f"  {reject:>12} {retain:>10} {alpha:>8.1f} {beta:>8.1f} {e_lam:>8.3f}   {direction}")

print(f"\n{SEP}")
print("  KEY FINDING")
print(THIN)
print("  Fixed MMR always uses λ=0.700 regardless of user behaviour.")
print("  Thompson Sampling starts at λ=0.700 (same as fixed baseline)")
print("  but adapts: rejections push λ DOWN (more diversity),")
print("  retentions push λ UP (more relevance) — automatically,")
print("  without any explicit user rating or preference input.")
print(f"{SEP}\n")
