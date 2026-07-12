# m3_implementation/test_result/contradiction_result/make_contra_figures.py
#
# Generates the contradiction-detector dissertation figures into figures/.
# Mirrors the hallucination chapter's figure style (fig1-fig5) for a
# consistent thesis look: same light surface, chrome, value labels, palette
# family. Re-run whenever results_contra_eval.json / results_correction_eval.json
# change.
#
# Figures:
#   figC1_detection_comparison  P/R/F1/BalAcc, 5 systems (grouped bars)
#   figC2_false_alarms          false alarms on clean + hard-negative cases
#   figC3_recall_by_corruption  recall per corruption type x system
#   figC4_recall_by_distance    SIGNATURE: recall vs turn distance (line plot)
#   figC5_correction_onoff      ON/OFF user-facing contradiction rate (Exp B)
#
# Run:  python test_result/contradiction_result/make_contra_figures.py

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DIR       = os.path.dirname(os.path.abspath(__file__))
FIGDIR     = os.path.join(_DIR, "figures")
DET_RESULTS = os.path.join(_DIR, "results_contra_eval.json")
COR_RESULTS = os.path.join(_DIR, "results_correction_eval.json")

# ── Palette & chrome (identical to hallucination chapter) ────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

# Five entities: our detector (blue) + string-only ablation (grey) +
# two NLI baselines (aqua, purple) + LLM judge (yellow). CVD-aware, distinct.
C_OURS    = "#2a78d6"
C_STRING  = "#9aa0a6"
C_HIST    = "#1baf7a"
C_UTTR    = "#7b5cd6"
C_JUDGE   = "#eda100"

SYSTEM_LABELS = {
    "ours":          "Ours (graph+NLI)",
    "string_only":   "String-only (−NLI)",
    "history_nli":   "History-NLI",
    "uttr_pair_nli": "Utterance-pair NLI",
    "llm_judge":     "LLM judge",
}
SYSTEM_COLORS = {
    "ours": C_OURS, "string_only": C_STRING, "history_nli": C_HIST,
    "uttr_pair_nli": C_UTTR, "llm_judge": C_JUDGE,
}
SYSTEM_ORDER = ["ours", "string_only", "history_nli", "uttr_pair_nli", "llm_judge"]

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
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)


def _bar_labels(ax, bars, fmt="{:.2f}", fontsize=7.5):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", xytext=(0, 2),
                    textcoords="offset points", fontsize=fontsize, color=INK)


def _title(fig, title, subtitle):
    fig.text(0.06, 0.965, title, fontsize=12.5, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.06, 0.905, subtitle, fontsize=9.5, color=INK_2,
             ha="left", va="top")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {name}")


def _present_systems(S):
    return [s for s in SYSTEM_ORDER if s in S and "metrics" in S[s]]


# ── figC1: detection comparison ──────────────────────────────────────────────

def figC1_detection_comparison(S):
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    metric_labels = ["Precision", "Recall", "F1", "Balanced\naccuracy"]
    systems = _present_systems(S)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(metrics))
    n = len(systems)
    w = 0.8 / n
    for i, sysname in enumerate(systems):
        vals = [S[sysname]["metrics"][m] for m in metrics]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, width=w * 0.92,
                      color=SYSTEM_COLORS[sysname], label=SYSTEM_LABELS[sysname])
        _bar_labels(ax, bars)
    _style_axes(ax, ymax=1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, metric_labels)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=8.5, ncol=5, borderaxespad=0, columnspacing=1.0,
              handlelength=1.1, handleheight=1.0)
    n_cases = S.get("n_cases", "?")
    n_pos = S.get("n_contradiction", "?")
    _title(fig, "Cross-turn contradiction detection — ours vs baselines",
           f"{n_cases} labeled cases ({n_pos} contradictions) · positive class = contradiction")
    fig.subplots_adjust(top=0.82)
    _save(fig, "figC1_detection_comparison.png")


# ── figC2: false alarms (clean + hard negatives) ─────────────────────────────

def figC2_false_alarms(S):
    systems = _present_systems(S)
    clean_fp = [S[s]["false_alarms"]["clean"]["fp"] for s in systems]
    hard_fp  = [S[s]["false_alarms"]["hard_negative"]["fp"] for s in systems]
    clean_n  = S[systems[0]]["false_alarms"]["clean"]["total"]
    hard_n   = S[systems[0]]["false_alarms"]["hard_negative"]["total"]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(systems))
    w = 0.38
    b1 = ax.bar(x - w / 2, clean_fp, width=w, color=BASELINE, label=f"clean (n={clean_n})")
    b2 = ax.bar(x + w / 2, hard_fp, width=w, color=C_JUDGE, label=f"hard negatives (n={hard_n})")
    _bar_labels(ax, b1, fmt="{:.0f}", fontsize=8.5)
    _bar_labels(ax, b2, fmt="{:.0f}", fontsize=8.5)
    ymax = max(max(clean_fp + hard_fp), 1) * 1.25
    _style_axes(ax, ymax=ymax, ylabel="false alarms (count)")
    ax.set_xticks(x, [SYSTEM_LABELS[s] for s in systems], fontsize=8.5, rotation=12)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _title(fig, "False alarms on correct + benign-variation responses",
           "Negatives wrongly flagged as contradictions (lower is better) · the NLI gate's value")
    fig.subplots_adjust(top=0.82)
    _save(fig, "figC2_false_alarms.png")


# ── figC3: recall by corruption type ─────────────────────────────────────────

def figC3_recall_by_corruption(S):
    ctypes = ["colour_drift", "price_drift", "name_drift", "type_drift", "cross_item_swap"]
    clabels = ["Colour", "Price", "Name", "Type", "Cross-item"]
    systems = _present_systems(S)

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(ctypes))
    n = len(systems)
    w = 0.8 / n
    for i, sysname in enumerate(systems):
        rbc = S[sysname]["recall_by_corruption"]
        vals = [rbc.get(c, {}).get("recall", 0.0) for c in ctypes]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, width=w * 0.92,
                      color=SYSTEM_COLORS[sysname], label=SYSTEM_LABELS[sysname])
        _bar_labels(ax, bars, fmt="{:.2f}", fontsize=6.8)
    _style_axes(ax, ymax=1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, clabels)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=8.5, ncol=5, borderaxespad=0, columnspacing=1.0,
              handlelength=1.1, handleheight=1.0)
    _title(fig, "Detection rate by corruption type",
           "Recall on contradiction cases · cross-item swaps expose presence-checking baselines")
    fig.subplots_adjust(top=0.82)
    _save(fig, "figC3_recall_by_corruption.png")


# ── figC4: recall vs turn distance (SIGNATURE) ───────────────────────────────

def figC4_recall_by_distance(S):
    dbuckets = ["0", "1", "2", "3+"]
    dlabels  = ["0\n(same turn)", "1", "2", "3+"]
    systems = _present_systems(S)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(dbuckets))
    for sysname in systems:
        rbd = S[sysname]["recall_by_distance"]
        vals = [rbd.get(d, {}).get("recall", np.nan) for d in dbuckets]
        ax.plot(x, vals, color=SYSTEM_COLORS[sysname], linewidth=2.2,
                marker="o", markersize=6, markerfacecolor=SYSTEM_COLORS[sysname],
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=SYSTEM_LABELS[sysname])
    _style_axes(ax, ymax=1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.3, len(dbuckets) - 0.7)
    ax.set_xticks(x, dlabels)
    ax.set_xlabel("turn distance — turns since the product's truth was established",
                  fontsize=9, color=INK_2)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=8.5, ncol=5, borderaxespad=0, columnspacing=1.0,
              handlelength=1.4)
    _title(fig, "Recall across turn distance",
           "Our detector retains product truth over long sessions · all systems "
           "are fed the same session facts, so this does not isolate a graph-only case")
    fig.subplots_adjust(top=0.80)
    _save(fig, "figC4_recall_by_distance.png")


# ── figC5: correction ON/OFF (Experiment B) ──────────────────────────────────

def figC5_correction_onoff(C):
    off_rate = C["off"]["user_facing_contradiction_rate"]
    on_rate  = C["on"]["user_facing_contradiction_rate"]
    p_fix    = C["on"]["p_fix_given_detect"]
    det_rate = C["on"]["detection_rate"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ["Detector OFF", "Detector ON"]
    vals = [off_rate, on_rate]
    colors = [BASELINE, C_OURS]
    bars = ax.bar(labels, vals, width=0.5, color=colors)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v*100:.1f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=11,
                    fontweight="bold", color=INK)
    _style_axes(ax, ymax=1.14, ylabel="user-facing contradiction rate")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    sub = (f"Detection rate {det_rate*100:.1f}% · P(correct fix | detected) "
           f"{p_fix*100:.1f}% · n={C['n_contradiction']} contradictions")
    _title(fig, "Automatic correction removes user-facing contradictions",
           sub)
    fig.subplots_adjust(top=0.82)
    _save(fig, "figC5_correction_onoff.png")


def main():
    print("Writing figures to figures/:")
    if os.path.exists(DET_RESULTS):
        with open(DET_RESULTS, encoding="utf-8") as f:
            S = json.load(f)
        figC1_detection_comparison(S)
        figC2_false_alarms(S)
        figC3_recall_by_corruption(S)
        figC4_recall_by_distance(S)
    else:
        print(f"  (skipping detection figures — {os.path.basename(DET_RESULTS)} not found)")

    if os.path.exists(COR_RESULTS):
        with open(COR_RESULTS, encoding="utf-8") as f:
            C = json.load(f)
        figC5_correction_onoff(C)
    else:
        print(f"  (skipping correction figure — {os.path.basename(COR_RESULTS)} not found)")
    print("Done.")


if __name__ == "__main__":
    main()
