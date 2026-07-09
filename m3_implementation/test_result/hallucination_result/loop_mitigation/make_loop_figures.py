# m3_implementation/test_result/hallucination_result/loop_mitigation/make_loop_figures.py
#
# Figures for the loop-mitigation experiment (results_loop_eval.json → figures/).
# Same validated palette and chrome as ../make_figures.py.
#
# Run:  python test_result/hallucination_result/loop_mitigation/make_loop_figures.py

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGDIR  = os.path.join(_DIR, "figures")
RESULTS = os.path.join(_DIR, "results_loop_eval.json")

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C_LOOP = "#2a78d6"       # the system with the loop — same blue as "ours"
C_OFF  = "#898781"       # loop absent — neutral gray, not a series

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "font.size": 10,
})


def _style(ax, ymax, ylabel=None):
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


def fig_loop_overall(R):
    n = R["n_cases"]
    on = R["loop_on"]
    off_pct = 100.0
    on_pct = 100 * on["residual_hallucination_rate"]
    missed = n - on["detected_attempt1"]
    failed = on["wrong_shipped"] - missed

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bars = ax.bar([0, 1], [off_pct, on_pct], width=0.48, color=[C_OFF, C_LOOP])
    for b, v, extra in zip(bars, [off_pct, on_pct],
                           [f"{n} / {n}", f"{on['wrong_shipped']} / {n}"]):
        ax.annotate(f"{v:.1f}%\n({extra})", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=10,
                    fontweight="bold", color=INK)
    ax.annotate(f"= {missed} undetected lies\n+ {failed} imperfect regenerations",
                (1, on_pct + 14), ha="center", fontsize=8.5, color=INK_2)
    _style(ax, ymax=118, ylabel="hallucinated responses reaching the user (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([0, 1], ["Loop OFF\n(response ships as generated)",
                           "Loop ON\n(detect → reject → regenerate)"])
    _title(fig, "Effect of the detect–reject–regenerate loop",
           f"{n} induced-hallucination cases · final outputs graded by an "
           "independent\nvalue-verification referee (not the checker)")
    fig.subplots_adjust(top=0.78)
    _save(fig, "fig6_loop_mitigation.png")


def fig_loop_by_type(R):
    by = R["by_corruption_type"]
    labels = {"colour_swap": "Colour swap", "price_change": "Price change",
              "name_swap": "Name swap", "cross_item_swap": "Cross-item swap"}
    ctypes = ["colour_swap", "price_change", "name_swap", "cross_item_swap"]
    vals = [100 * by[c]["residual_rate"] for c in ctypes]
    ns   = [(by[c]["still_wrong_with_loop"], by[c]["n"]) for c in ctypes]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(range(len(ctypes)), vals, width=0.5, color=C_LOOP)
    for b, v, (w, n) in zip(bars, vals, ns):
        ax.annotate(f"{v:.1f}%  ({w}/{n})",
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=9, color=INK)
    _style(ax, ymax=max(vals) * 1.5 + 4, ylabel="still wrong with loop ON (%)")
    ax.set_xticks(range(len(ctypes)), [labels[c] for c in ctypes])
    _title(fig, "Residual hallucinations by corruption type (loop ON)",
           "Loop OFF is 100% for every type by construction")
    fig.subplots_adjust(top=0.80)
    _save(fig, "fig7_loop_residual_by_type.png")


def fig_attempts_outcome(R):
    """How many attempts each case needed, split by final outcome."""
    recs = R["records"]
    C_OK, C_WRONG = "#2a78d6", "#e34948"   # categorical slots 1 and 6

    counts_ok, counts_wrong = [], []
    for a in (1, 2, 3):
        bucket = [r for r in recs if r["attempts"] == a]
        ok = sum(1 for r in bucket if r["shipped_correct"])
        counts_ok.append(ok)
        counts_wrong.append(len(bucket) - ok)

    xlabels = ["Attempt 1\n(checker missed —\nno retry happened)",
               "Attempt 2\n(one regeneration,\nstrictness 1)",
               "Attempt 3\n(two regenerations,\nstrictness 2)"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(3)
    b1 = ax.bar(x, counts_ok, width=0.5, color=C_OK, label="final response correct")
    b2 = ax.bar(x, counts_wrong, width=0.5, bottom=counts_ok, color=C_WRONG,
                label="still wrong", edgecolor=SURFACE, linewidth=1.5)
    for i in range(3):
        total = counts_ok[i] + counts_wrong[i]
        ax.annotate(f"{total} cases", (i, total), ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=INK)
        ax.annotate(f"{counts_ok[i]} correct · {counts_wrong[i]} wrong",
                    (i, total), ha="center", va="bottom", xytext=(0, 18),
                    textcoords="offset points", fontsize=8.5, color=INK_2)
    _style(ax, ymax=max(counts_ok[i] + counts_wrong[i] for i in range(3)) * 1.28,
           ylabel="cases")
    ax.set_xticks(x, xlabels)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _title(fig, "Attempts needed per induced hallucination (loop ON)",
           "205 cases · 94% of detected lies fixed by ONE regeneration · "
           "strictest mode rescued 10 of the last 11")
    fig.subplots_adjust(top=0.80)
    _save(fig, "fig8_attempts_outcome.png")


def main():
    with open(RESULTS, encoding="utf-8") as f:
        R = json.load(f)
    print("Writing loop figures to figures/:")
    fig_loop_overall(R)
    fig_loop_by_type(R)
    fig_attempts_outcome(R)
    print("Done.")


if __name__ == "__main__":
    main()
