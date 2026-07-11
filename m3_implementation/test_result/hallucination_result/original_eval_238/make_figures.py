# m3_implementation/test_result/hallucination_result/make_figures.py
#
# Generates the dissertation figures from results_summary.json into figures/.
# Re-run after build_summary.py whenever results change.
#
# Palette: validated categorical trio (CVD-safe, fixed entity assignment)
#   our checker = blue #2a78d6 · naive NLI = aqua #1baf7a · LLM judge = yellow #eda100
# Version progression uses a single-hue ordinal blue ramp (light -> dark).
# Sub-3:1 contrast on aqua/yellow is relieved by direct value labels on every bar.
#
# Run:  python test_result/hallucination_result/make_figures.py

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGDIR  = os.path.join(_DIR, "figures")
SUMMARY = os.path.join(_DIR, "results_summary.json")

# ── Palette & chrome (light surface) ─────────────────────────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

C_OURS    = "#2a78d6"   # categorical slot 1 — our checker (all versions = v3 unless ramp)
C_NAIVE   = "#1baf7a"   # slot 2 — naive NLI baseline
C_JUDGE   = "#eda100"   # slot 3 — LLM judge baseline
RAMP_V    = ["#86b6ef", "#2a78d6", "#104281"]   # ordinal: v1 -> v2 -> v3

SYSTEM_LABELS = {"ours": "Our checker (v3)", "naive_nli": "Naive NLI", "llm_judge": "LLM judge"}
SYSTEM_COLORS = {"ours": C_OURS, "naive_nli": C_NAIVE, "llm_judge": C_JUDGE}

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


def _bar_labels(ax, bars, fmt="{:.3f}", fontsize=8.5):
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


# ── Figures ──────────────────────────────────────────────────────────────────

def fig1_detection_comparison(S):
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    metric_labels = ["Precision", "Recall", "F1", "Balanced\naccuracy"]
    systems = ["ours", "naive_nli", "llm_judge"]
    values = {
        "ours":      S["checker_versions"]["v3"]["metrics"],
        "naive_nli": S["baselines"]["naive_nli"]["metrics"],
        "llm_judge": S["baselines"]["llm_judge"]["metrics"],
    }

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(metrics))
    w = 0.24
    for i, sysname in enumerate(systems):
        vals = [values[sysname][m] for m in metrics]
        bars = ax.bar(x + (i - 1) * (w + 0.02), vals, width=w,
                      color=SYSTEM_COLORS[sysname], label=SYSTEM_LABELS[sysname])
        _bar_labels(ax, bars)
    _style_axes(ax, ymax=1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, metric_labels)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=9, ncol=3, borderaxespad=0, columnspacing=1.4,
              handlelength=1.2, handleheight=1.0)
    _title(fig, "Hallucination detection accuracy — our checker vs baselines",
           "238 labeled cases (205 synthetic hallucinations, 33 clean) · positive class = hallucinated")
    fig.subplots_adjust(top=0.82)
    _save(fig, "fig1_detection_comparison.png")


def fig2_false_alarms(S):
    systems = ["ours", "naive_nli", "llm_judge"]
    fps = [
        S["checker_versions"]["v3"]["metrics"]["fp"],
        S["baselines"]["naive_nli"]["metrics"]["fp"],
        S["baselines"]["llm_judge"]["metrics"]["fp"],
    ]
    n_clean = S["meta"]["n_clean"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(systems))
    bars = ax.bar(x, fps, width=0.5,
                  color=[SYSTEM_COLORS[s] for s in systems])
    for b, v in zip(bars, fps):
        ax.annotate(f"{v} / {n_clean}",
                    (b.get_x() + b.get_width() / 2, max(v, 0)),
                    ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=10,
                    fontweight="bold", color=INK)
    ax.axhline(n_clean, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"all {n_clean} clean responses", (2.35, n_clean),
                ha="right", va="bottom", fontsize=8.5, color=MUTED)
    _style_axes(ax, ymax=n_clean * 1.14, ylabel="false alarms")
    ax.set_xticks(x, [SYSTEM_LABELS[s] for s in systems])
    _title(fig, "False alarms on correct responses",
           "Correct responses wrongly flagged as hallucination (lower is better)")
    fig.subplots_adjust(top=0.80)
    _save(fig, "fig2_false_alarms.png")


def fig3_version_progression(S):
    metrics = ["recall", "f1", "balanced_accuracy"]
    metric_labels = ["Recall", "F1", "Balanced accuracy"]
    versions = ["v1", "v2", "v3"]
    vlabels = ["v1  original", "v2  two-sided gates", "v3  + response-level"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(metrics))
    w = 0.24
    for i, v in enumerate(versions):
        vals = [S["checker_versions"][v]["metrics"][m] for m in metrics]
        bars = ax.bar(x + (i - 1) * (w + 0.02), vals, width=w,
                      color=RAMP_V[i], label=vlabels[i])
        _bar_labels(ax, bars)
    _style_axes(ax, ymax=1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, metric_labels)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=9, ncol=3, borderaxespad=0, columnspacing=1.4,
              handlelength=1.2, handleheight=1.0)
    _title(fig, "Evaluation-driven refinement of the checker (v1 → v3)",
           "Same 238-case test set at each step · precision stays 1.000 in v1 and v3 (0.994 in v2)")
    fig.subplots_adjust(top=0.82)
    _save(fig, "fig3_version_progression.png")


def fig4_recall_by_corruption(S):
    ctypes = ["colour_swap", "price_change", "name_swap", "cross_item_swap"]
    clabels = ["Colour swap", "Price change", "Name swap", "Cross-item swap"]
    systems = ["ours", "naive_nli", "llm_judge"]
    values = {
        "ours":      S["checker_versions"]["v3"]["recall_by_corruption"],
        "naive_nli": S["baselines"]["naive_nli"]["recall_by_corruption"],
        "llm_judge": S["baselines"]["llm_judge"]["recall_by_corruption"],
    }

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(ctypes))
    w = 0.24
    for i, sysname in enumerate(systems):
        vals = [values[sysname][c]["recall"] for c in ctypes]
        bars = ax.bar(x + (i - 1) * (w + 0.02), vals, width=w,
                      color=SYSTEM_COLORS[sysname], label=SYSTEM_LABELS[sysname])
        _bar_labels(ax, bars, fmt="{:.2f}")
    _style_axes(ax, ymax=1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, clabels)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=9, ncol=3, borderaxespad=0, columnspacing=1.4,
              handlelength=1.2, handleheight=1.0)
    _title(fig, "Detection rate by corruption type",
           "Share of injected errors detected, per corruption type (recall on hallucinated cases)")
    fig.subplots_adjust(top=0.82)
    _save(fig, "fig4_recall_by_corruption.png")


def fig5_threshold_sweep(S):
    sweep = S["threshold_sweep_v3"]
    th  = [r["threshold"] for r in sweep]
    ser = [("precision", "Precision", C_OURS),
           ("recall", "Recall", C_NAIVE),
           ("f1", "F1", C_JUDGE)]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    # artifact region: containment flags carry synthetic score 1.0, so
    # thresholds >= 1.0 exclude them and no longer reflect real behaviour
    ax.axvspan(1.0, max(th), color=GRID, alpha=0.45, zorder=0)
    ax.annotate("artifact region — containment flags\n(synthetic score 1.0) excluded when t ≥ 1.0",
                (1.12, 0.30), fontsize=8.5, color=MUTED, ha="left")
    ax.axvline(0.65, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("operating\nthreshold 0.65", (0.65, 1.045), fontsize=8.5,
                color=INK_2, ha="center", va="bottom")

    for key, label, color in ser:
        vals = [r[key] for r in sweep]
        ax.plot(th, vals, color=color, linewidth=2, marker="o",
                markersize=5.5, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        ax.annotate(label, (th[-1], vals[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=color,
                    va="center")

    _style_axes(ax, ymax=1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(0, max(th) * 1.12)
    ax.set_xlabel("NLI contradiction threshold (raw logits)", fontsize=9, color=INK_2)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    _title(fig, "Threshold sensitivity (checker v3)",
           "Metrics recomputed offline from stored NLI scores · valid NLI region is t < 1.0")
    fig.subplots_adjust(top=0.82)
    _save(fig, "fig5_threshold_sweep.png")


def main():
    with open(SUMMARY, encoding="utf-8") as f:
        S = json.load(f)
    print("Writing figures to figures/:")
    fig1_detection_comparison(S)
    fig2_false_alarms(S)
    fig3_version_progression(S)
    fig4_recall_by_corruption(S)
    fig5_threshold_sweep(S)
    print("Done.")


if __name__ == "__main__":
    main()
