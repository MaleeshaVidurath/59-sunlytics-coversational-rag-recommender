# m3_implementation/test_result/hallucination_result/threshold_calibration/make_sweep_figures.py
#
# Two figures for the threshold calibration experiment. Palette and chrome are
# copied from ../original_eval_238/make_figures.py so these sit beside the
# reported figures without a visual seam.
#
#   figT1  the full sweep, -5.0 .. +5.0, with the plateau shaded and both
#          candidate operating points marked
#   figT2  the distribution of contradiction logits, split by whether the check
#          also beat entailment — shows WHY the threshold is inert
#
# Run:  python test_result/hallucination_result/threshold_calibration/make_sweep_figures.py

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGDIR  = os.path.join(_DIR, "figures")
RESULTS = os.path.join(_DIR, "results_threshold_sweep.json")
SCORES  = os.path.join(_DIR, "per_check_scores.jsonl")

# ── Palette & chrome — identical to the reported figures ────────────────────
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"

C_OURS   = "#2a78d6"
C_NAIVE  = "#1baf7a"
C_JUDGE  = "#eda100"
C_ALERT  = "#c0392b"

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


def _title(fig, title, subtitle):
    fig.text(0.06, 0.965, title, fontsize=13, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.06, 0.905, subtitle, fontsize=9.5, color=INK_2,
             ha="left", va="top")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {name}")


# ── figT1 — the sweep ────────────────────────────────────────────────────────

def fig_sweep(res: dict) -> None:
    rows = res["sweep"]
    th   = [r["threshold"] for r in rows]
    pl   = res["plateau_balanced_accuracy"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(top=0.78)

    # plateau band
    ax.axvspan(pl["from"], pl["to"], color=C_OURS, alpha=0.07, zorder=0)
    ax.annotate(f"measured plateau  {pl['from']} to {pl['to']}\n"
                f"identical on every metric",
                (pl["from"] + 0.15, 0.13), fontsize=8.5, color=INK_2, ha="left")

    # artifact region
    ax.axvspan(1.0, max(th), color=GRID, alpha=0.55, zorder=0)
    ax.annotate("t ≥ 1.0 partly an artifact:\ncontainment flags score 1.0",
                (1.15, 0.72), fontsize=8.5, color=MUTED, ha="left")

    for key, colour, label in (("precision", C_NAIVE, "Precision"),
                               ("recall", C_OURS, "Recall"),
                               ("f1", C_JUDGE, "F1"),
                               ("balanced_accuracy", INK_2, "Balanced accuracy")):
        ax.plot(th, [r[key] for r in rows], marker="o", markersize=3.2,
                linewidth=1.7, color=colour, label=label)

    for x, colour, label, dy in ((0.65, BASELINE, "0.65  config default\n(evaluated & reported)", 0.30),
                                 (0.70, C_ALERT, "0.70  deployed (.env)", 0.44)):
        ax.axvline(x, color=colour, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(label, (x, dy), fontsize=8, color=colour, ha="right",
                    rotation=90, va="bottom")

    _style_axes(ax, ymax=1.06, ylabel="score")
    ax.set_xlabel("contradiction threshold (raw cross-encoder logit)",
                  fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower left",
              bbox_to_anchor=(0.0, -0.30))

    _title(fig, "figT1  Threshold sensitivity across the full logit range",
           "238 cases (205 hallucinated / 33 clean). Precision stays 1.000 at every "
           "threshold tested, including −5.0 —\nso the absolute bar never "
           "admits a false positive. 0.65 and 0.70 are indistinguishable.")
    _save(fig, "figT1_threshold_sweep.png")


# ── figT2 — why the threshold is inert ───────────────────────────────────────

def fig_logits(res: dict) -> None:
    beats, loses = [], []
    with open(SCORES, encoding="utf-8") as f:
        for line in f:
            for ch in json.loads(line)["checks"]:
                (beats if ch["contradiction"] > ch["entailment"]
                 else loses).append(ch["contradiction"])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(top=0.78)

    bins = 60
    ax.hist(loses, bins=bins, color=BASELINE, alpha=0.85,
            label=f"entailment wins — never flagged  (n={len(loses)})")
    ax.hist(beats, bins=bins, color=C_OURS, alpha=0.85,
            label=f"contradiction beats entailment  (n={len(beats)})")

    for x, colour, label in ((0.65, BASELINE, "0.65"), (0.70, C_ALERT, "0.70")):
        ax.axvline(x, color=colour, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(label, (x, ax.get_ylim()[1] * 0.92), fontsize=8,
                    color=colour, ha="right", rotation=90, va="top")

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_ylabel("checks", fontsize=9, color=INK_2)
    ax.set_xlabel("contradiction logit", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    gap_lo = max((v for v in beats if v < 1.0), default=None)
    _title(fig, "figT2  Why the threshold is inert",
           "Every check the checker ran. The two populations separate on the "
           "relative test, not the absolute one —\nalmost nothing sits near "
           "the 0.65–0.70 line, so moving it changes no decisions.")
    _save(fig, "figT2_logit_distribution.png")

    print(f"    checks that beat entailment: {len(beats)}")
    print(f"    of those, logit < 1.0: "
          f"{sum(1 for v in beats if v < 1.0)}  (highest such: {gap_lo})")
    print(f"    of those, logit in [0.65, 0.70]: "
          f"{sum(1 for v in beats if 0.65 <= v <= 0.70)}")


# ── figT3 — the derivation: threshold scored as the SOLE rule ────────────────

def fig_derivation() -> None:
    path = os.path.join(_DIR, "results_threshold_derivation.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    roc = [r for r in d["rule_a_threshold_alone"]["roc"]
           if -1.5 <= r["threshold"] <= 2.5]
    th = [r["threshold"] for r in roc]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(top=0.78)

    # three regimes
    ax.axvspan(-1.5, 0.0, color=C_ALERT, alpha=0.07, zorder=0)
    ax.axvspan(0.80, 1.0, color=C_NAIVE, alpha=0.13, zorder=0)
    ax.axvspan(1.0, 2.5, color=C_ALERT, alpha=0.07, zorder=0)

    ax.annotate("t < 0\nspecificity\ncollapses", (-1.35, 0.30), fontsize=8.5,
                color=C_ALERT, ha="left")
    ax.annotate("OPTIMUM\n0.80 – 0.95", (0.83, 0.13), fontsize=8.5,
                color="#12694a", ha="left", fontweight="bold")
    ax.annotate("t ≥ 1.0\nrecall collapses\n(containment flags = 1.0)",
                (1.08, 0.30), fontsize=8.5, color=C_ALERT, ha="left")

    for key, colour, label in (("precision", C_NAIVE, "Precision"),
                               ("recall", C_OURS, "Recall"),
                               ("specificity", C_JUDGE, "Specificity"),
                               ("youden_j", INK_2, "Youden's J")):
        ax.plot(th, [r[key] for r in roc], marker="o", markersize=2.6,
                linewidth=1.7, color=colour, label=label)

    ax.axvline(0.80, color="#12694a", linewidth=1.4, linestyle=(0, (4, 3)))
    ax.annotate("0.80  derived", (0.78, 0.50), fontsize=8, color="#12694a",
                ha="right", rotation=90, va="bottom", fontweight="bold")
    ax.axvline(0.70, color=BASELINE, linewidth=1.1, linestyle=(0, (2, 3)))
    ax.annotate("0.70  previous", (0.68, 0.50), fontsize=8, color=MUTED,
                ha="right", rotation=90, va="bottom")

    _style_axes(ax, ymax=1.06, ylabel="score")
    ax.set_xlabel("contradiction threshold (raw logit)", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower left",
              bbox_to_anchor=(0.0, -0.30))

    _title(fig, "Deriving the threshold — scored as the sole decision rule",
           "The relative test is ablated so the threshold actually decides. Youden's J and F1 "
           "both peak on [0.80, 0.95];\n0.80 is taken as the lower edge, farthest from the "
           "recall cliff at 1.0.")
    _save(fig, "fig_threshold_derivation.png")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="also rebuild the two diagnostic figures (figT1 invariance, "
                         "figT2 logit gap). They are not part of the final result and "
                         "are not shipped by default.")
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    print("Building figures...")
    fig_derivation()                       # the final result
    if args.all:
        with open(RESULTS, encoding="utf-8") as f:
            res = json.load(f)
        fig_sweep(res)
        fig_logits(res)
    print(f"\nFigures -> {FIGDIR}")


if __name__ == "__main__":
    main()
