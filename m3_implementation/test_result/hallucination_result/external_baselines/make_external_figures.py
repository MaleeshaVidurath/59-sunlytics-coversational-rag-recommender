# m3_implementation/test_result/hallucination_result/external_baselines/make_external_figures.py
#
# Figures for the off-the-shelf baselines experiment.
#   fig9  — metric comparison: our checker vs three unmodified external tools
#   fig10 — cross-item swap recall: the lock-map advantage in one chart
#
# Same validated palette/chrome as ../make_figures.py (categorical slots 1-4
# in fixed order: ours=blue, HHEM=aqua, SummaC=yellow, LettuceDetect=green).
#
# Run:  python test_result/hallucination_result/external_baselines/make_external_figures.py

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGDIR  = os.path.join(_DIR, "figures")
EXT     = os.path.join(_DIR, "results_external_baselines.json")
SUMMARY = os.path.join(_DIR, "..", "results_summary.json")

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
COLORS = {"ours": "#2a78d6", "hhem": "#1baf7a",
          "summac": "#eda100", "lettuce": "#008300"}
LABELS = {"ours": "Our checker (v3)", "hhem": "Vectara HHEM-2.1",
          "summac": "SummaC-Conv", "lettuce": "LettuceDetect"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "font.size": 10,
})


def _style(ax, ymax=1.0, ylabel=None):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)


def _title(fig, title, subtitle):
    fig.text(0.06, 0.955, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.885, subtitle, fontsize=9.5, color=INK_2, va="top")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name), dpi=200,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {name}")


def load_all():
    ext = json.load(open(EXT, encoding="utf-8"))
    summ = json.load(open(SUMMARY, encoding="utf-8"))
    data = {"ours": {
        "metrics": summ["checker_versions"]["v3"]["metrics"],
        "recall_by_corruption": summ["checker_versions"]["v3"]["recall_by_corruption"],
    }}
    for tool in ("hhem", "summac", "lettuce"):
        if "metrics" in ext.get(tool, {}):
            data[tool] = ext[tool]
    return data


def fig9_comparison(data):
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    mlabels = ["Precision", "Recall", "F1", "Balanced\naccuracy"]
    systems = [s for s in ("ours", "hhem", "summac", "lettuce") if s in data]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(metrics))
    w = 0.19
    for i, s in enumerate(systems):
        vals = [data[s]["metrics"][m] for m in metrics]
        bars = ax.bar(x + (i - 1.5) * (w + 0.015), vals, width=w,
                      color=COLORS[s], label=LABELS[s])
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", xytext=(0, 2),
                        textcoords="offset points", fontsize=7.5, color=INK)
    _style(ax, ymax=1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, mlabels)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=8.5, ncol=4, borderaxespad=0, columnspacing=1.1,
              handlelength=1.1, handleheight=1.0)
    _title(fig, "Our checker vs unmodified off-the-shelf detectors",
           "Same 238-case test set · external tools run as released, evidence "
           "serialized to text · thresholds at tool defaults (0.5)")
    fig.subplots_adjust(top=0.80)
    _save(fig, "fig9_external_comparison.png")


def fig10_cross_item(data):
    systems = [s for s in ("ours", "summac", "hhem", "lettuce") if s in data]
    vals, ns = [], []
    for s in systems:
        r = data[s]["recall_by_corruption"]["cross_item_swap"]
        vals.append(r["recall"])
        ns.append((r["detected"], r["total"]))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(range(len(systems)), vals, width=0.5,
                  color=[COLORS[s] for s in systems])
    for b, v, (d, t) in zip(bars, vals, ns):
        ax.annotate(f"{100*v:.1f}%  ({d}/{t})",
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=9.5,
                    fontweight="bold", color=INK)
    _style(ax, ymax=1.14, ylabel="cross-item swaps detected")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(range(len(systems)), [LABELS[s] for s in systems])
    _title(fig, "Cross-item swap detection — the lock-map advantage",
           "Values swapped between items: every value IS in the context, only the "
           "item association is wrong.\nGeneral-purpose tools verify presence, "
           "not association — the item→sentence lock map verifies both.")
    fig.subplots_adjust(top=0.76)
    _save(fig, "fig10_cross_item_recall.png")


def main():
    data = load_all()
    print("Writing external-baseline figures to figures/:")
    fig9_comparison(data)
    fig10_cross_item(data)
    print("Done.")


if __name__ == "__main__":
    main()
