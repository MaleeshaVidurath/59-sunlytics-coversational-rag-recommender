# m3_implementation/test_result/hallucination_result/expanded_eval/make_expanded_figures.py
#
# Figures for the EXPANDED evaluation (4,000-row suite). Auto-includes every
# system whose results file exists — re-run after each stage lands and the
# figures grow. Adds 95% CI error bars (new vs the original figure set).
#
#   figE1  standard set — metric comparison with CI error bars
#   figE2  standard set — false alarms on the 526 clean cases
#   figE3  hard set — detection rate per hard-corruption type
#   figE4  the joint view: hard-set recall vs clean false-alarm rate
#          (guards against the "always-beeping smoke detector" misreading)
#   figE5  original 238-row set vs expanded 2,600-row set (our checker)
#
# Fixed entity colours across ALL expanded figures (validated palette order):
#   ours=blue, naive=aqua, judge=yellow, lettuce=green, hhem=violet, summac=red
#
# Run:  python test_result/hallucination_result/expanded_eval/make_expanded_figures.py

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DIR   = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_DIR, "figures")

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
COLORS = {"ours": "#2a78d6", "naive_nli": "#1baf7a", "llm_judge": "#eda100",
          "lettuce": "#008300", "hhem": "#4a3aa7", "summac": "#e34948"}
LABELS = {"ours": "Our checker", "naive_nli": "Naive NLI",
          "llm_judge": "LLM judge", "lettuce": "LettuceDetect",
          "hhem": "HHEM-2.1", "summac": "SummaC-Conv"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
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


def _legend_top(ax, ncol):
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=8.5, ncol=ncol, borderaxespad=0, columnspacing=1.1,
              handlelength=1.1, handleheight=1.0)


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name), dpi=200,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {name}")


def _jload(fname):
    p = os.path.join(_DIR, fname)
    if os.path.exists(p) and os.path.getsize(p) > 10:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get(block):
    return block if isinstance(block, dict) and "metrics" in block else None


def collect(setname):
    """Returns {system: results-block} for one test set, from local canonical
    files first, Colab files as fallback."""
    out = {}
    if setname == "standard":
        out["ours"] = _get(_jload("results_standard_ours.json").get("ours")) \
            or _get(_jload("colab_results_standard_ours_naive.json").get("ours"))
        out["naive_nli"] = _get(_jload("colab_results_standard_ours_naive.json").get("naive_nli"))
        out["llm_judge"] = _get(_jload("results_standard_judge800.json").get("llm_judge"))
        out["lettuce"]   = _get(_jload("colab_results_external_standard.json").get("lettuce"))
        out["hhem"]      = _get(_jload("results_external_standard_hhem.json").get("hhem"))
        out["summac"]    = _get(_jload("results_external_standard_summac.json").get("summac"))
    else:
        out["ours"] = _get(_jload("results_hard_ours.json").get("ours")) \
            or _get(_jload("colab_results_hard_ours_naive.json").get("ours"))
        out["naive_nli"] = _get(_jload("colab_results_hard_ours_naive.json").get("naive_nli"))
        out["llm_judge"] = _get(_jload("results_hard_judge600.json").get("llm_judge"))
        out["lettuce"]   = _get(_jload("colab_results_external_hard.json").get("lettuce"))
        out["hhem"]      = _get(_jload("results_external_hard_hhem.json").get("hhem"))
        out["summac"]    = _get(_jload("results_external_hard_summac.json").get("summac"))
    return {k: v for k, v in out.items() if v}


def figE1_standard(std):
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    mlabels = ["Precision", "Recall", "F1", "Balanced\naccuracy"]
    ci_keys = {"precision": "precision_ci95", "recall": "recall_ci95",
               "f1": "f1_ci95", "balanced_accuracy": "balanced_accuracy_ci95"}
    systems = list(std)
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    x = np.arange(len(metrics))
    w = 0.8 / max(len(systems), 1)
    for i, s in enumerate(systems):
        m = std[s]["metrics"]
        vals = [m[k] for k in metrics]
        errs_lo, errs_hi = [], []
        for k in metrics:
            ci = m.get(ci_keys[k]) or [m[k], m[k]]
            errs_lo.append(max(0, m[k] - ci[0]))
            errs_hi.append(max(0, ci[1] - m[k]))
        bars = ax.bar(x + (i - (len(systems)-1)/2) * w, vals, width=w*0.92,
                      color=COLORS[s], label=LABELS[s],
                      yerr=[errs_lo, errs_hi], capsize=2,
                      error_kw={"ecolor": INK_2, "elinewidth": 0.9})
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}",
                        (b.get_x() + b.get_width()/2, 0.02),
                        ha="center", va="bottom", fontsize=7,
                        color=SURFACE, fontweight="bold")
    _style(ax, ymax=1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, mlabels)
    _legend_top(ax, ncol=min(len(systems), 6))
    _title(fig, "Expanded standard set — detection accuracy with 95% CIs",
           "2,600 rows (526 clean incl. real user chats + 2,074 corrupted) · "
           "error bars: Wilson/bootstrap 95% CIs")
    fig.subplots_adjust(top=0.80)
    _save(fig, "figE1_standard_comparison.png")


def figE2_false_alarms(std):
    systems = list(std)
    n_clean = std["ours"]["metrics"]["fp"] + std["ours"]["metrics"]["tn"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    fps = [std[s]["metrics"]["fp"] for s in systems]
    bars = ax.bar(range(len(systems)), fps, width=0.5,
                  color=[COLORS[s] for s in systems])
    for b, v in zip(bars, fps):
        ax.annotate(f"{v} / {n_clean}", (b.get_x()+b.get_width()/2, v),
                    ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=9.5,
                    fontweight="bold", color=INK)
    _style(ax, ymax=max(fps + [10]) * 1.2, ylabel="false alarms")
    ax.set_xticks(range(len(systems)), [LABELS[s] for s in systems])
    _title(fig, "False alarms on clean responses (expanded standard set)",
           f"{n_clean} clean cases, including real user conversations · lower is better")
    fig.subplots_adjust(top=0.80)
    _save(fig, "figE2_false_alarms.png")


def figE3_hard_by_type(hard):
    ctypes = ["paraphrase_colour", "paraphrase_price", "fabricated_attribute"]
    clabels = ["Paraphrased\ncolour", "Paraphrased\nprice", "Fabricated\nattribute*"]
    systems = list(hard)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(ctypes))
    w = 0.8 / max(len(systems), 1)
    for i, s in enumerate(systems):
        rb = hard[s].get("recall_by_corruption", {})
        vals = [rb.get(c, {}).get("recall", 0.0) for c in ctypes]
        bars = ax.bar(x + (i - (len(systems)-1)/2) * w, vals, width=w*0.92,
                      color=COLORS[s], label=LABELS[s])
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}",
                        (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", xytext=(0, 2),
                        textcoords="offset points", fontsize=7.5, color=INK)
    _style(ax, ymax=1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, clabels)
    _legend_top(ax, ncol=min(len(systems), 6))
    _title(fig, "Hard adversarial set — detection per corruption family",
           "1,400 corrupted-only rows · *unsupported claims are ignored BY DESIGN "
           "by the contradiction-only checker\n· interpret jointly with figE4 — "
           "trigger-happy detectors score high here while failing clean cases")
    fig.subplots_adjust(top=0.78)
    _save(fig, "figE3_hard_by_type.png")


def figE4_joint(std, hard):
    systems = [s for s in std if s in hard]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s in systems:
        m_std = std[s]["metrics"]
        fp_rate = m_std["fp"] / (m_std["fp"] + m_std["tn"]) if (m_std["fp"] + m_std["tn"]) else 0
        hard_recall = hard[s]["metrics"]["recall"]
        ax.scatter(fp_rate, hard_recall, s=140, color=COLORS[s], zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(LABELS[s], (fp_rate, hard_recall), xytext=(8, -4),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.annotate("ideal corner", (0.015, 1.02), fontsize=8.5, color=MUTED)
    _style(ax, ymax=1.12, ylabel="hard-set recall (adversarial lies caught)")
    ax.set_xlim(-0.04, 1.0)
    ax.set_xlabel("false-alarm rate on clean responses (standard set)",
                  fontsize=9, color=INK_2)
    _title(fig, "The joint view — adversarial recall vs clean-case false alarms",
           "High hard-set recall is trivial for detectors that flag everything; "
           "the top-left corner is what matters")
    fig.subplots_adjust(top=0.82)
    _save(fig, "figE4_joint_view.png")


def figE5_old_vs_new(std):
    # original 238-row results (frozen, from the main evaluation)
    orig = _jload(os.path.join("..", "original_eval_238", "results_summary.json"))
    if not orig:
        print("  figE5 skipped (results_summary.json not found)")
        return
    o = orig["checker_versions"]["v3"]["metrics"]
    n = std["ours"]["metrics"]
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    mlabels = ["Precision", "Recall", "F1", "Balanced accuracy"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(metrics))
    b1 = ax.bar(x - 0.17, [o[k] for k in metrics], width=0.32,
                color="#86b6ef", label="original set (238 rows, 33 clean)")
    b2 = ax.bar(x + 0.17, [n[k] for k in metrics], width=0.32,
                color="#2a78d6", label="expanded set (2,600 rows, 526 clean)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}",
                        (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", xytext=(0, 2),
                        textcoords="offset points", fontsize=8, color=INK)
    _style(ax, ymax=1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, mlabels)
    _legend_top(ax, ncol=2)
    _title(fig, "Our checker: original vs expanded test set",
           "More and messier data (incl. real chats) yields more conservative, "
           "credible numbers — precision holds at 1.000 throughout")
    fig.subplots_adjust(top=0.80)
    _save(fig, "figE5_original_vs_expanded.png")


def main():
    std = collect("standard")
    hard = collect("hard")
    print(f"systems with standard results: {list(std)}")
    print(f"systems with hard results:     {list(hard)}")
    print("Writing figures to figures/:")
    if std:
        figE1_standard(std)
        figE2_false_alarms(std)
        figE5_old_vs_new(std)
    if hard:
        figE3_hard_by_type(hard)
    if std and hard:
        figE4_joint(std, hard)
    print("Done — re-run after new results files land to extend the figures.")


if __name__ == "__main__":
    main()
