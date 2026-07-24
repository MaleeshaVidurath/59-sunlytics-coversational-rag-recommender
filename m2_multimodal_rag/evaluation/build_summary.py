"""
Aggregates all evaluation CSVs into one summary table for the report.

Usage (from repo root):
    python -m m2_multimodal_rag.evaluation.build_summary

Output: evaluation/results/SUMMARY.md (+ summary.csv)
"""

import csv
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    rows = []

    # N1 — retrieval
    p = RESULTS / "retrieval_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        base = df[df["config"] == "A_stock_single"]
        fine = df[df["config"] == "B_finetuned_single"]
        ens = df[df["config"] == "C_finetuned_ensemble"]
        delta = df[df["config"].astype(str).str.startswith("delta_finetuning")]
        if len(base) and len(fine):
            ci = (f"[{delta['ci_low'].iloc[0]*100:+.1f}, {delta['ci_high'].iloc[0]*100:+.1f}]"
                  if len(delta) else "—")
            rows.append({
                "novelty": "N1 CLIP fine-tuning",
                "metric": "Recall@10",
                "baseline": f"{base['R@10'].iloc[0]*100:.1f}% (stock CLIP)",
                "novelty_value": f"{fine['R@10'].iloc[0]*100:.1f}%",
                "delta": f"+{(fine['R@10'].iloc[0]-base['R@10'].iloc[0])*100:.1f} pp",
                "ci95": ci, "n": int(base["n"].iloc[0]),
            })
        if len(ens) and len(fine):
            rows.append({
                "novelty": "N1 multi-vector ensemble",
                "metric": "Recall@10",
                "baseline": f"{fine['R@10'].iloc[0]*100:.1f}% (single vector)",
                "novelty_value": f"{ens['R@10'].iloc[0]*100:.1f}%",
                "delta": f"{(ens['R@10'].iloc[0]-fine['R@10'].iloc[0])*100:+.1f} pp",
                "ci95": "—", "n": int(ens["n"].iloc[0]),
            })

    # N2 — CF
    p = RESULTS / "cf_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        pop = df[df["config"] == "popularity"]
        ncf = df[df["config"] == "ncf"]
        rule = df[df["config"] == "rule_based"]
        if len(ncf) and len(rule):
            rows.append({
                "novelty": "N2 NCF (cold-start)",
                "metric": "Cold-start Hit@10",
                "baseline": f"{rule['cold_hit10'].iloc[0]*100:.1f}% (rule) / "
                            f"{pop['cold_hit10'].iloc[0]*100:.1f}% (popularity)",
                "novelty_value": f"{ncf['cold_hit10'].iloc[0]*100:.1f}%",
                "delta": f"+{(ncf['cold_hit10'].iloc[0]-rule['cold_hit10'].iloc[0])*100:.1f} pp vs rule",
                "ci95": "—", "n": int(ncf["n_cold"].iloc[0]),
            })
            rows.append({
                "novelty": "N2 NCF (overall)",
                "metric": "Hit@10",
                "baseline": f"{rule['hit10'].iloc[0]*100:.1f}% (rule)",
                "novelty_value": f"{ncf['hit10'].iloc[0]*100:.1f}%",
                "delta": f"{(ncf['hit10'].iloc[0]-rule['hit10'].iloc[0])*100:+.1f} pp "
                         "(additive boost in prod, not standalone ranker)",
                "ci95": "—", "n": int(ncf["n"].iloc[0]),
            })

    # N3 — diversity
    p = RESULTS / "diversity_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        a0 = df[(df["config"] == "adaptive") & (df["rejections"] == 0)]
        a10 = df[(df["config"] == "adaptive") & (df["rejections"] == 10)]
        if len(a0) and len(a10):
            rows.append({
                "novelty": "N3 Thompson bandit",
                "metric": "λ adaptation (0→10 rejections)",
                "baseline": "fixed λ (no adaptation)",
                "novelty_value": f"λ {a0['mean_lambda'].iloc[0]:.2f} → {a10['mean_lambda'].iloc[0]:.2f}",
                "delta": f"ILD {a0['ild_mean'].iloc[0]:.3f} → {a10['ild_mean'].iloc[0]:.3f}",
                "ci95": "—", "n": int(a0["n_points"].iloc[0]),
            })

    # N4 — guard
    p = RESULTS / "guard_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        l1 = df[df["config"] == "l1"]
        full = df[df["config"] == "full"]
        if len(l1) and len(full):
            rows.append({
                "novelty": "N4 hallucination guard",
                "metric": "Detection F1",
                "baseline": f"{l1['f1'].iloc[0]:.3f} (L1 only)",
                "novelty_value": f"{full['f1'].iloc[0]:.3f} (full 3-layer)",
                "delta": f"visual-corruption recall {l1['visual_recall'].iloc[0]:.2f} → "
                         f"{full['visual_recall'].iloc[0]:.2f}",
                "ci95": "—", "n": int(full["n"].iloc[0]),
            })

    # N5 — kansei
    p = RESULTS / "kansei_results.csv"
    if p.exists():
        kv = dict(pd.read_csv(p).values)
        rows.append({
            "novelty": "N5 Kansei KB",
            "metric": "Blind LLM-judge win rate",
            "baseline": "KB off",
            "novelty_value": f"{float(kv['kb_on_win_rate_decided'])*100:.0f}% KB-on wins (decided)",
            "delta": f"{int(kv['kb_on_wins'])}W/{int(kv['kb_off_wins'])}L/{int(kv['ties'])}T",
            "ci95": f"[{float(kv['ci_low'])*100:.0f}%, {float(kv['ci_high'])*100:.0f}%]",
            "n": int(kv["n_queries"]),
        })

    if not rows:
        raise SystemExit("No results CSVs found yet.")

    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# M2 Novelty Evaluation — Summary", "",
             "| Novelty | Metric | Baseline | With novelty | Δ | 95% CI | n |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['novelty']} | {r['metric']} | {r['baseline']} | "
                     f"{r['novelty_value']} | {r['delta']} | {r['ci95']} | {r['n']} |")
    lines += ["", "Figures: `results/figures/` — generated by `make_figures.py`.",
              "Per-experiment detail: the individual `*_results.csv` files."]
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {RESULTS / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
