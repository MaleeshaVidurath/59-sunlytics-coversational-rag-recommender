# m3_implementation/test_result/contradiction_result_v2/make_v2_figures.py
#
# Figures for the v2 cross-turn consistency evaluation.
#
# Reads sample599/results_v2_eval.json (the like-for-like slice — v1's own
# stratified sample, seed 123) so every bar on a shared axis comes from the same
# 599 cases. v1's baseline rows are read verbatim from its results file; nothing
# in ../contradiction_result/ is written to.
#
# Palette and chrome are copied from the v1/hallucination figure scripts so the
# two chapters sit side by side without a visual seam.
#
# Run:  python test_result/contradiction_result_v2/make_v2_figures.py

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DIR        = os.path.dirname(os.path.abspath(__file__))
V2_RESULTS  = os.path.join(_DIR, "sample599", "results_v2_eval.json")
V1_RESULTS  = os.path.join(_DIR, "..", "contradiction_result", "results_contra_eval.json")
FIGDIR      = os.path.join(_DIR, "figures")

# ── Palette & chrome (identical to the v1 / hallucination chapters) ──────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

C_V2_DET  = "#2a78d6"   # v2 detected  — same blue "ours" carried in v1
C_V2_REP  = "#8fbdf0"   # v2 reported  — lighter tint of the same blue
C_V1      = "#0b4f9e"   # v1 ours      — darker tint, reads as the same family
C_STRING  = "#9aa0a6"
C_HIST    = "#1baf7a"
C_UTTR    = "#7b5cd6"
C_JUDGE   = "#eda100"
C_DEFER   = "#eda100"   # the deferred share

SYSTEM_LABELS = {
    "v2_detected":   "v2 detected",
    "v2_reported":   "v2 reported",
    "ours":          "v1 ours (graph+NLI)",
    "string_only":   "v1 string-only (−NLI)",
    "history_nli":   "v1 History-NLI",
    "uttr_pair_nli": "v1 Utterance-pair NLI",
    "llm_judge":     "v1 LLM judge",
}
SYSTEM_COLORS = {
    "v2_detected": C_V2_DET, "v2_reported": C_V2_REP, "ours": C_V1,
    "string_only": C_STRING, "history_nli": C_HIST,
    "uttr_pair_nli": C_UTTR, "llm_judge": C_JUDGE,
}
# "v2_reported" is deliberately absent. Every series on these axes answers
# "did the system find the mismatch?"; reporting answers "did it rewrite the
# text?", which on this test set is ~1% by design. Drawn beside detection bars
# it reads as failure instead of as the deliberate hand-off it is, so it is
# shown on its own terms in figV2 and kept in full in results_v2_eval.json.
SYSTEM_ORDER = ["v2_detected", "ours", "string_only",
                "history_nli", "uttr_pair_nli", "llm_judge"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE,
    "font.size": 10,
})


def _style_axes(ax, ymax=1.0, ylabel=None):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_ylim(0, ymax)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)


def _bar_labels(ax, bars, fmt="{:.2f}", fontsize=7.5):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", xytext=(0, 2),
                    textcoords="offset points", fontsize=fontsize, color=INK)


def _title(fig, title, subtitle):
    fig.text(0.06, 0.98, title, fontsize=13.5, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.06, 0.905, subtitle, fontsize=9.5, color=INK_2,
             ha="left", va="top")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {name}")


# ── Data assembly ────────────────────────────────────────────────────────────

def load() -> tuple[dict, dict, int]:
    with open(V2_RESULTS, encoding="utf-8") as f:
        v2 = json.load(f)
    with open(V1_RESULTS, encoding="utf-8") as f:
        v1 = json.load(f)

    systems = {
        "v2_detected": v2["v2_detected"],
        "v2_reported": v2["v2_reported"],
    }
    for name in ("ours", "string_only", "history_nli", "uttr_pair_nli", "llm_judge"):
        if v1.get(name, {}).get("metrics"):
            systems[name] = v1[name]
    return systems, v2, v2["n_cases"]


def _present(systems):
    return [s for s in SYSTEM_ORDER if s in systems]


# ── Figure 1: detection accuracy, v2 beside v1 and the baselines ─────────────

def fig_detection(systems, n):
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    labels  = ["Precision", "Recall", "F1", "Balanced acc."]
    present = _present(systems)

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    x = np.arange(len(metrics))
    w = 0.8 / len(present)

    for i, name in enumerate(present):
        m = systems[name]["metrics"]
        vals = [m.get(k, 0) for k in metrics]
        bars = ax.bar(x + (i - (len(present) - 1) / 2) * w, vals, width=w * 0.92,
                      color=SYSTEM_COLORS[name], label=SYSTEM_LABELS[name])
        if name in ("v2_detected", "ours"):
            _bar_labels(ax, bars, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    _style_axes(ax, ymax=1.12)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.02))
    _title(fig, "Cross-turn consistency: v2 versus the evaluated v1 detector",
           f"Same labelled test set, same stratified sample ({n} cases, seed 123), "
           f"same metric code. v2 uses no LLM call.")
    fig.subplots_adjust(top=0.80)
    _save(fig, "figV1_detection_v2_vs_v1.png")


# ── Figure 2: ownership — the duplication v1 was counting ────────────────────

def fig_ownership(v2):
    own = v2["ownership"]
    reported = own["reported_by_v2"]
    deferred = own["deferred_to_guard"]

    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.barh([0], [deferred], color=C_DEFER, height=0.5,
            label=f"deferred to the hallucination guard ({deferred})")
    ax.barh([0], [reported], left=[deferred], color=C_V2_DET, height=0.5,
            label=f"reported by v2 itself ({reported})")

    total = deferred + reported
    ax.set_xlim(0, total * 1.02)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_xlabel("contradictions detected by v2", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=2)

    ax.annotate(f"{own['share_deferred']*100:.1f}%",
                (deferred / 2, 0), ha="center", va="center",
                fontsize=15, fontweight="bold", color="#5a3d00")

    _title(fig, "Most of what v2 detects, it deliberately does not report",
           "Same-turn mismatches belong to the hallucination guard. That share is "
           "the double-counting\nv1's recall contained — one failure reported by "
           "two components.")
    fig.subplots_adjust(top=0.72, bottom=0.34)
    _save(fig, "figV2_ownership_split.png")


# ── Figure 3: recall by turn distance ────────────────────────────────────────

def fig_distance(systems, n):
    order = ["0", "1", "2", "3+"]
    present = [s for s in _present(systems)
               if systems[s].get("recall_by_distance")]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(order))
    for name in present:
        rbd = systems[name]["recall_by_distance"]
        vals = [rbd.get(k, {}).get("recall", np.nan) for k in order]
        ax.plot(x, vals, color=SYSTEM_COLORS[name], linewidth=2.2,
                marker="o", markersize=6, markerfacecolor=SYSTEM_COLORS[name],
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=SYSTEM_LABELS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(["same turn\n(d=0)", "d=1", "d=2", "d≥3"], fontsize=9.5)
    _style_axes(ax, ymax=1.08, ylabel="recall")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower left")
    _title(fig, "Detection does not decay with turn distance",
           f"How many turns back the contradicted value was established "
           f"({n}-case sample).")
    fig.subplots_adjust(top=0.84)
    _save(fig, "figV3_recall_by_distance.png")


# ── Figure 4: recall by corruption type ──────────────────────────────────────

def fig_corruption(systems, n):
    present = [s for s in _present(systems)
               if systems[s].get("recall_by_corruption")]
    types = sorted({t for s in present
                    for t in systems[s]["recall_by_corruption"]})

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x = np.arange(len(types))
    w = 0.8 / len(present)

    for i, name in enumerate(present):
        rbc = systems[name]["recall_by_corruption"]
        vals = [rbc.get(t, {}).get("recall", 0) for t in types]
        bars = ax.bar(x + (i - (len(present) - 1) / 2) * w, vals, width=w * 0.92,
                      color=SYSTEM_COLORS[name], label=SYSTEM_LABELS[name])
        if name == "v2_detected":
            _bar_labels(ax, bars, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=9)
    _style_axes(ax, ymax=1.12, ylabel="recall")
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.03))
    _title(fig, "Detection recall by corruption type",
           f"v2 reads values deterministically — no LLM extraction "
           f"({n}-case sample).")
    fig.subplots_adjust(top=0.80)
    _save(fig, "figV4_recall_by_corruption.png")


# ── Figure 5: false alarms on the negative class ─────────────────────────────

def fig_false_alarms(systems):
    present = _present(systems)
    fa0 = systems[present[0]]["false_alarms"]
    clean_n, hard_n = fa0["clean"]["total"], fa0["hard_negative"]["total"]

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    x = np.arange(len(present))
    w = 0.36
    clean = [systems[s]["false_alarms"]["clean"]["fp"] for s in present]
    hard  = [systems[s]["false_alarms"]["hard_negative"]["fp"] for s in present]

    b1 = ax.bar(x - w / 2, clean, width=w, color=BASELINE,
                label=f"clean (n={clean_n})")
    b2 = ax.bar(x + w / 2, hard, width=w, color=C_JUDGE,
                label=f"hard negatives (n={hard_n})")
    _bar_labels(ax, b1, fmt="{:.0f}")
    _bar_labels(ax, b2, fmt="{:.0f}")

    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in present],
                       fontsize=8.5, rotation=18, ha="right")
    _style_axes(ax, ymax=max(clean + hard) * 1.25 + 1,
                ylabel="false alarms (count)")
    ax.legend(frameon=False, fontsize=9)
    _title(fig, "False alarms on the negative class",
           "Hard negatives are benign subtype paraphrases (\"Dress\" → \"maxi "
           "dress\") that must not be flagged.")
    fig.subplots_adjust(top=0.84)
    _save(fig, "figV5_false_alarms.png")


def main():
    if not os.path.exists(V2_RESULTS):
        print(f"missing {V2_RESULTS}\n"
              "run:  python test_result/contradiction_result_v2/run_v2_eval.py "
              "--sample 599 --out-dir test_result/contradiction_result_v2/sample599")
        return

    systems, v2, n = load()
    print(f"Building figures from the {n}-case like-for-like sample...")
    fig_detection(systems, n)
    fig_ownership(v2)
    fig_distance(systems, n)
    fig_corruption(systems, n)
    fig_false_alarms(systems)
    print(f"\nFigures in {os.path.relpath(FIGDIR)}")


if __name__ == "__main__":
    main()
