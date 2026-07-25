"""
Report figure generator — one PNG (300 dpi) + PDF per novelty, built from the
CSVs in evaluation/results/. Figures are skipped gracefully when their input
CSV doesn't exist yet, so this can be re-run after each experiment.

Usage (from repo root):
    python -m m2_multimodal_rag.evaluation.make_figures

Outputs → evaluation/results/figures/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
FIGS = RESULTS / "figures"

# Consistent style
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})
BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"


def _save(fig, name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}.png (+.pdf)")


# ────────────────────────────────────────────────────────────────────

def fig_n1_retrieval():
    path = RESULTS / "retrieval_results.csv"
    if not path.exists():
        return print("  [skip] N1 — retrieval_results.csv missing")
    df = pd.read_csv(path)
    df = df[~df["config"].astype(str).str.startswith("delta")]
    labels = {"A_stock_single": "Stock CLIP",
              "B_finetuned_single": "Fine-tuned CLIP",
              "C_finetuned_ensemble": "Fine-tuned +\nLLM ensemble"}
    df["label"] = df["config"].map(labels).fillna(df["config"])
    metrics = ["R@1", "R@5", "R@10"]
    x = range(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (m, c) in enumerate(zip(metrics, (BLUE, ORANGE, GREEN))):
        ax.bar([xi + (i - 1) * width for xi in x], df[m] * 100, width,
               label=m, color=c)
        for xi, v in zip(x, df[m] * 100):
            ax.text(xi + (i - 1) * width, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(df["label"])
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"N1 — Known-item retrieval (n={int(df['n'].iloc[0])} held-out queries)")
    ax.legend(frameon=False)
    _save(fig, "n1_retrieval")


def fig_n2_cf():
    path = RESULTS / "cf_results.csv"
    if not path.exists():
        return print("  [skip] N2 — cf_results.csv missing")
    df = pd.read_csv(path)
    df = df[~df["config"].astype(str).str.startswith("delta")]
    labels = {"popularity": "Popularity", "rule_based": "Rule-based", "ncf": "Two-Tower NCF"}
    df["label"] = df["config"].map(labels).fillna(df["config"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = range(len(df))
    width = 0.35
    axes[0].bar([xi - width / 2 for xi in x], df["hit5"] * 100, width, label="Hit@5", color=BLUE)
    axes[0].bar([xi + width / 2 for xi in x], df["hit10"] * 100, width, label="Hit@10", color=ORANGE)
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(df["label"])
    axes[0].set_ylabel("Hit rate (%)"); axes[0].legend(frameon=False)
    axes[0].set_title(f"All held-out purchases (n={int(df['n'].iloc[0])})")
    axes[1].bar([xi - width / 2 for xi in x], df["hit10"] * 100, width,
                label="All items", color=GREY)
    axes[1].bar([xi + width / 2 for xi in x], df["cold_hit10"] * 100, width,
                label="Cold-start only", color=GREEN)
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(df["label"])
    axes[1].set_ylabel("Hit@10 (%)"); axes[1].legend(frameon=False)
    axes[1].set_title(f"Cold-start slice (n={int(df['n_cold'].iloc[0])})")
    fig.suptitle("N2 — Two-Tower NCF vs baselines (100 negatives per case)", y=1.02)
    _save(fig, "n2_cf")


def fig_n3_diversity():
    path = RESULTS / "diversity_results.csv"
    if not path.exists():
        return print("  [skip] N3 — diversity_results.csv missing")
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = {"fixed_0.5": GREY, "fixed_0.7": BLUE, "fixed_0.9": ORANGE, "adaptive": GREEN}
    for cfg, grp in df.groupby("config"):
        grp = grp.sort_values("rejections")
        axes[0].plot(grp["rejections"], grp["mean_lambda"], marker="o", ms=4,
                     label=cfg, color=colors.get(cfg, GREY))
        axes[1].plot(grp["rejections"], grp["ild_mean"], marker="o", ms=4,
                     label=cfg, color=colors.get(cfg, GREY))
    axes[0].set_xlabel("Cumulative rejections"); axes[0].set_ylabel("Mean λ used")
    axes[0].set_title("λ adaptation (Thompson posterior)")
    axes[1].set_xlabel("Cumulative rejections"); axes[1].set_ylabel("Intra-list diversity")
    axes[1].set_title("Selection diversity (CLIP space)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("N3 — Thompson Sampling bandit vs fixed-λ MMR", y=1.02)
    _save(fig, "n3_diversity")


def fig_n4_guard():
    path = RESULTS / "guard_results.csv"
    if not path.exists():
        return print("  [skip] N4 — guard_results.csv missing")
    df = pd.read_csv(path)
    cases = pd.read_csv(RESULTS / "guard_case_results.csv")
    labels = {"l1": "L1 only", "l12": "L1+L2", "full": "Full guard\n(+L3 visual)"}
    df["label"] = df["config"].map(labels).fillna(df["config"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = range(len(df))
    width = 0.27
    for i, (m, c) in enumerate(zip(("precision", "recall", "f1"), (BLUE, ORANGE, GREEN))):
        axes[0].bar([xi + (i - 1) * width for xi in x], df[m], width, label=m.capitalize(), color=c)
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(df["label"])
    axes[0].set_ylim(0, 1.05); axes[0].legend(frameon=False)
    axes[0].set_title(f"Detection quality (n={int(df['n'].iloc[0])} cases)")
    # Visual-corruption recall — the Layer-3 story
    axes[1].bar(df["label"], df["visual_recall"], color=RED, width=0.5)
    for xi, v in zip(range(len(df)), df["visual_recall"]):
        axes[1].text(xi, v + 0.02, f"{v:.2f}", ha="center")
    n_vis = int((cases["corruption"] == "visual_claim").sum())
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Recall on visual-only corruptions (n={n_vis})")
    fig.suptitle("N4 — Hallucination guard ablation", y=1.02)
    _save(fig, "n4_guard")


def fig_n5_kansei():
    path = RESULTS / "kansei_results.csv"
    if not path.exists():
        return print("  [skip] N5 — kansei_results.csv missing")
    kv = dict(pd.read_csv(path).values)
    wins_on, wins_off, ties = int(kv["kb_on_wins"]), int(kv["kb_off_wins"]), int(kv["ties"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(["Blind LLM judge"], [wins_on], color=GREEN, label=f"KB on wins ({wins_on})")
    ax.barh(["Blind LLM judge"], [ties], left=[wins_on], color=GREY, label=f"Ties ({ties})")
    ax.barh(["Blind LLM judge"], [wins_off], left=[wins_on + ties], color=RED,
            label=f"KB off wins ({wins_off})")
    ax.set_xlabel("Queries")
    ax.set_title("N5 — Kansei KB: blind paired preference "
                 f"(win rate {float(kv['kb_on_win_rate_decided']):.0%}, "
                 f"CI [{float(kv['ci_low']):.0%}, {float(kv['ci_high']):.0%}])")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _save(fig, "n5_kansei")


def main():
    print("Generating report figures →", FIGS)
    fig_n1_retrieval()
    fig_n2_cf()
    fig_n3_diversity()
    fig_n4_guard()
    fig_n5_kansei()
    print("Done.")


if __name__ == "__main__":
    main()
