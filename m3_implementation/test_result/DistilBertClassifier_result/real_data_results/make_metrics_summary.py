# m3_implementation/test_result/DistilBertClassifier_result/real_data_results/make_metrics_summary.py
#
# Summary figure for the DistilBERT intent classifier.
# Source: test-set classification report on real_test_simmc.csv (7,309 samples).
#
# CONTENT — deliberately minimal: overall accuracy plus the per-class F1. The
# precision / recall / support columns live in the full report; repeating them
# here only crowds the figure.
#
# FORM — a dot plot, not bars. Every class scores between 0.921 and 1.000. Bars
# must start at zero, so on a 0-1 axis all eight would look identical; truncating
# a BAR axis to 0.90 would exaggerate differences through false area. A dot
# encodes POSITION, not length, so a zoomed axis is honest.
#
# COLOUR — encodes a performance band, not identity. Validated with the dataviz
# palette checker (light mode, surface #fcfcfb): lightness band PASS, chroma
# floor PASS, CVD separation PASS (worst adjacent dE 23.1 protan), normal-vision
# floor PASS. The contrast WARN is discharged by direct labels: every dot carries
# its numeric value, so nothing is encoded by colour alone.
#
# Run:  python make_metrics_summary.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(_DIR, "metrics_summary.png")

SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"

C_HIGH = "#1baf7a"   # F1 >= 0.99
C_MID  = "#2a78d6"   # 0.95 <= F1 < 0.99
C_LOW  = "#eda100"   # F1 <  0.95

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": BASELINE,
    "font.size": 14,
})

ACCURACY  = 0.9748
N_SAMPLES = 7309

# class, F1
ROWS = [
    ("INITIAL_REQUEST",     0.9733),
    ("REFINEMENT",          0.9542),
    ("ATTRIBUTE_QUESTION",  0.9342),
    ("EXPLANATION_WHY",     1.0000),
    ("COMPARISON",          0.9213),
    ("SELECTION_REFERENCE", 0.9987),
    ("FEEDBACK",            0.9993),
    ("CHITCHAT",            1.0000),
]


def band(f1: float):
    if f1 >= 0.99:
        return C_HIGH
    if f1 >= 0.95:
        return C_MID
    return C_LOW


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    fig.subplots_adjust(left=0.24, right=0.90, top=0.70, bottom=0.13)

    # ── title + headline accuracy ────────────────────────────────────────────
    fig.text(0.035, 0.975, "DistilBERT Intent Classifier Test Set Results",
             fontsize=27, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.035, 0.912,
             f"real_test_simmc.csv  ·  {N_SAMPLES:,} samples  ·  8 intent classes",
             fontsize=16.5, color=INK, ha="left", va="top")

    fig.text(0.035, 0.845, "A C C U R A C Y", fontsize=14.5, color=INK_2,
             ha="left", va="top", fontweight="bold")
    fig.text(0.035, 0.815, f"{ACCURACY*100:.2f}%", fontsize=52,
             fontweight="bold", color=C_MID, ha="left", va="top")

    # legend for the colour bands
    for i, (colour, label) in enumerate([(C_HIGH, "F1 ≥ 0.99"),
                                         (C_MID,  "0.95 – 0.99"),
                                         (C_LOW,  "below 0.95")]):
        x = 0.60 + i * 0.135
        fig.text(x, 0.828, "●", fontsize=20, color=colour, ha="left", va="top")
        fig.text(x + 0.022, 0.822, label, fontsize=14, color=INK_2,
                 ha="left", va="top")

    # ── per-class F1 ─────────────────────────────────────────────────────────
    names = [r[0] for r in ROWS]
    f1s   = [r[1] for r in ROWS]
    y     = list(range(len(ROWS)))[::-1]

    for yi, f1 in zip(y, f1s):
        c = band(f1)
        ax.plot([0.90, f1], [yi, yi], color=c, linewidth=3.2, alpha=0.30,
                solid_capstyle="round", zorder=1)
        ax.plot(f1, yi, "o", markersize=19, color=c,
                markeredgecolor=SURFACE, markeredgewidth=3, zorder=3)
        ax.annotate(f"{f1:.4f}", (f1, yi), xytext=(19, 0),
                    textcoords="offset points", fontsize=19,
                    color=c, va="center", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=17, color=INK, fontweight="bold")
    # A touch past 1.0 so dots sitting exactly on 1.0000 are not clipped by the
    # axis edge — four of the eight classes land there.
    ax.set_xlim(0.90, 1.006)
    ax.set_ylim(-0.75, len(ROWS) - 0.25)
    ax.set_xticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    ax.tick_params(axis="x", labelsize=16, colors=INK)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("per-class F1", fontsize=17, color=INK, labelpad=12,
                  fontweight="bold")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.35)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
