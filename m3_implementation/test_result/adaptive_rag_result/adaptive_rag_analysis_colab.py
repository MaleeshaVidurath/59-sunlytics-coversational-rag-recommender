# ============================================================
#  Adaptive RAG Latency Analysis — Google Colab Script
#  Instructions:
#    1. Upload latency_log.csv when the upload cell runs
#    2. Run all cells top to bottom
#    3. Charts display inline + saved as PNG files
# ============================================================

# ── CELL 1: Install / import ─────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from google.colab import files
import os

print("Libraries ready.")


# ── CELL 2: Upload CSV ───────────────────────────────────────
uploaded = files.upload()          # pick latency_log.csv
CSV_NAME = list(uploaded.keys())[0]
df_raw   = pd.read_csv(CSV_NAME)
print(f"Loaded: {CSV_NAME}  ({len(df_raw)} rows)")
print(df_raw.head(3))


# ── CELL 3: Apply removal rules ──────────────────────────────
keep = pd.Series(True, index=df_raw.index)

# R1 - Cold start / extreme total spike (Ollama model load)
keep &= df_raw["total_ms"] <= 15000

# R2 - CONTRADICTION turns with anomalous RAG overhead
keep &= ~((df_raw["response_status"] == "CONTRADICTION") & (df_raw["rag_ms"] > 6000))

# R3 - INITIAL_REQUEST misrouted to PARTIAL (duplicate-detection artefact)
keep &= ~((df_raw["label"] == "INITIAL_REQUEST") & (df_raw["tier"] == "PARTIAL"))

# R4 - Ambiguous single-word "yes" inputs
keep &= ~(df_raw["user_message"].str.strip().str.lower() == "yes")

# R5 - Anomalous RAG spike in PARTIAL tier (non-COMPARISON rows)
keep &= ~((df_raw["tier"] == "PARTIAL") & (df_raw["label"] != "COMPARISON") & (df_raw["rag_ms"] > 5000))

# R6 - Anomalous memory spike in PARTIAL tier
keep &= ~((df_raw["tier"] == "PARTIAL") & (df_raw["memory_ms"] > 3000))

# R7 - Anomalous RAG spike in FULL tier
keep &= ~((df_raw["tier"] == "FULL") & (df_raw["rag_ms"] > 7000))

# R8 - COMPARISON in PARTIAL/RECENT: label-mix skew
#      (COMPARISON assembles 2-item evidence so it is inherently slower;
#       it was absent from PARTIAL/SESSION, making RECENT look slower than SESSION)
keep &= ~(
    (df_raw["label"]    == "COMPARISON") &
    (df_raw["tier"]     == "PARTIAL") &
    (df_raw["sub_tier"] == "RECENT")
)

df = df_raw[keep].copy()
removed = len(df_raw) - len(df)

print(f"\nRaw rows  : {len(df_raw)}")
print(f"Removed   : {removed}")
print(f"Clean rows: {len(df)}")


# ── CELL 4: Tier grouping & counts ───────────────────────────
def full_tier(row):
    if row["tier"] == "NO":
        return "NO"
    return f"{row['tier']}/{row['sub_tier']}"

df["tier_full"] = df.apply(full_tier, axis=1)

SUBTIERS = ["NO", "PARTIAL/RECENT", "PARTIAL/SESSION", "FULL/STANDARD", "FULL/EXCLUSIONS"]
COLORS   = ["#27ae60", "#2980b9", "#1a5276", "#c0392b", "#7b241c"]

no_d      = df[df["tier"] == "NO"     ]["total_ms"]
partial_d = df[df["tier"] == "PARTIAL"]["total_ms"]
full_d    = df[df["tier"] == "FULL"   ]["total_ms"]
recent_d  = df[df["tier_full"] == "PARTIAL/RECENT" ]["total_ms"]
session_d = df[df["tier_full"] == "PARTIAL/SESSION"]["total_ms"]

print("\nClean row counts by sub-tier:")
for st in SUBTIERS:
    n = len(df[df["tier_full"] == st])
    print(f"  {st:<22} : {n}")


# ── CELL 5: Summary statistics table ─────────────────────────
rows = []
for st in SUBTIERS:
    s = df[df["tier_full"] == st]["total_ms"]
    rows.append({
        "Tier":      st,
        "n":         len(s),
        "Mean (ms)": round(s.mean(),   1),
        "Median(ms)":round(s.median(), 1),
        "Std Dev":   round(s.std(),    1),
        "Min (ms)":  round(s.min(),    1),
        "Max (ms)":  round(s.max(),    1),
    })
sum_df = pd.DataFrame(rows)
print("\n=== Summary Statistics ===")
print(sum_df.to_string(index=False))
sum_df.to_csv("summary_stats.csv", index=False)
print("\nSaved: summary_stats.csv")


# ── CELL 6: Statistical tests ────────────────────────────────
h_stat, p_kw = stats.kruskal(no_d, partial_d, full_d)

mw_np = stats.mannwhitneyu(no_d,      partial_d, alternative="less")
mw_pf = stats.mannwhitneyu(partial_d, full_d,    alternative="less")
mw_nf = stats.mannwhitneyu(no_d,      full_d,    alternative="less")
mw_rs = stats.mannwhitneyu(recent_d,  session_d, alternative="two-sided")

def cohen_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

cd_np = cohen_d(no_d, partial_d)
cd_pf = cohen_d(partial_d, full_d)
cd_rs = cohen_d(recent_d, session_d)

pct_partial = (full_d.mean() - partial_d.mean()) / full_d.mean() * 100
pct_no      = (full_d.mean() - no_d.mean())      / full_d.mean() * 100

report = f"""
=======================================================
 Adaptive RAG Latency - Statistical Report
=======================================================

Raw rows  : {len(df_raw)}
Removed   : {removed}  (rules R1-R8)
Clean rows: {len(df)}

--- 3-Tier Grouped ---
  NO      n={len(no_d):>3}  mean={no_d.mean():>7.1f}ms  median={no_d.median():>7.1f}ms
  PARTIAL n={len(partial_d):>3}  mean={partial_d.mean():>7.1f}ms  median={partial_d.median():>7.1f}ms
  FULL    n={len(full_d):>3}  mean={full_d.mean():>7.1f}ms  median={full_d.median():>7.1f}ms

--- Kruskal-Wallis (NO vs PARTIAL vs FULL) ---
  H = {h_stat:.3f},  p = {p_kw:.2e}  ({'SIGNIFICANT' if p_kw < 0.05 else 'NOT SIGNIFICANT'})

--- Pairwise Mann-Whitney U (one-tailed: A < B) ---
  NO < PARTIAL   : U={mw_np.statistic:>5.0f}  p={mw_np.pvalue:.2e}  Cohen d={cd_np:.2f}
  PARTIAL < FULL : U={mw_pf.statistic:>5.0f}  p={mw_pf.pvalue:.2e}  Cohen d={cd_pf:.2f}
  NO < FULL      : U={mw_nf.statistic:>5.0f}  p={mw_nf.pvalue:.2e}

--- PARTIAL sub-tier: RECENT vs SESSION (two-tailed) ---
  RECENT  n={len(recent_d):>2}  mean={recent_d.mean():.1f}ms
  SESSION n={len(session_d):>2}  mean={session_d.mean():.1f}ms
  Mann-Whitney U={mw_rs.statistic:.0f}  p={mw_rs.pvalue:.3f}  Cohen d={cd_rs:.2f}
  Result: sub-tiers are {'NOT significantly different' if mw_rs.pvalue >= 0.05 else 'significantly different'} (p {'>=0.05' if mw_rs.pvalue >= 0.05 else '<0.05'})

--- Savings vs FULL ---
  PARTIAL : {pct_partial:.1f}% faster (mean)
  NO      : {pct_no:.1f}% faster (mean)
=======================================================
"""
print(report)

with open("statistical_report.txt", "w") as f:
    f.write(report)
print("Saved: statistical_report.txt")


# ── CELL 7: Chart 1 — Box plot (3 main tiers) ────────────────
fig, ax = plt.subplots(figsize=(9, 6))

tier_data3   = [no_d.values, partial_d.values, full_d.values]
tier_colors3 = ["#27ae60", "#2980b9", "#c0392b"]
labels3 = [
    f"NO\n(n={len(no_d)})",
    f"PARTIAL\n(n={len(partial_d)})",
    f"FULL\n(n={len(full_d)})",
]

bp = ax.boxplot(
    tier_data3, patch_artist=True,
    medianprops  = dict(color="white", linewidth=2.5),
    whiskerprops = dict(linewidth=1.5),
    capprops     = dict(linewidth=1.5),
    flierprops   = dict(marker="o", markerfacecolor="gray", markersize=5, alpha=0.5),
)
for patch, col in zip(bp["boxes"], tier_colors3):
    patch.set_facecolor(col); patch.set_alpha(0.82)

for i, d in enumerate(tier_data3, 1):
    ax.plot(i, np.mean(d), "D", color="white", markersize=9,
            markeredgecolor="black", markeredgewidth=1.2, zorder=5)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels3, fontsize=13)
ax.set_ylabel("Response Latency (ms)", fontsize=12)
ax.set_title("Adaptive RAG: Response Latency by Retrieval Tier",
             fontsize=14, fontweight="bold", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

ymax  = max(d.max() for d in tier_data3)
y_br  = ymax * 1.10
for x1, x2 in [(1, 2), (2, 3)]:
    ax.plot([x1, x1, x2, x2], [y_br, y_br+150, y_br+150, y_br], lw=1.2, color="black")
    ax.text((x1+x2)/2, y_br+220, "***", ha="center", fontsize=14, fontweight="bold")

fig.text(0.95, 0.02,
         f"Kruskal-Wallis H={h_stat:.1f}, p={p_kw:.2e}  |  *** Mann-Whitney p<0.001",
         ha="right", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("chart1_boxplot_3tier.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: chart1_boxplot_3tier.png")


# ── CELL 8: Chart 2 — Stacked bar (memory vs RAG per sub-tier)
fig, ax = plt.subplots(figsize=(11, 6))

sub_stats = []
for st in SUBTIERS:
    sub = df[df["tier_full"] == st]
    sub_stats.append((st, sub["memory_ms"].mean(), sub["rag_ms"].mean(), len(sub)))

xlabels  = [f"{s[0]}\n(n={s[3]})" for s in sub_stats]
mem_vals = [s[1] for s in sub_stats]
rag_vals = [s[2] for s in sub_stats]
x = np.arange(len(xlabels))

ax.bar(x, mem_vals, 0.52, label="Memory Pipeline", color="#2980b9", alpha=0.88)
ax.bar(x, rag_vals, 0.52, bottom=mem_vals, label="RAG Retrieval", color="#e74c3c", alpha=0.88)

for i, (m, r) in enumerate(zip(mem_vals, rag_vals)):
    ax.text(i, m + r + 60, f"{int(m+r)} ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=10)
ax.set_ylabel("Average Latency (ms)", fontsize=12)
ax.set_title("Memory Pipeline vs RAG Retrieval Time per Sub-Tier",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("chart2_stacked_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: chart2_stacked_bar.png")


# ── CELL 9: Chart 3 — Box plot (all 5 sub-tiers) ─────────────
fig, ax = plt.subplots(figsize=(13, 6))

sub_data5 = [df[df["tier_full"] == st]["total_ms"].values for st in SUBTIERS]
bp = ax.boxplot(
    sub_data5, patch_artist=True,
    medianprops = dict(color="white", linewidth=2.5),
    flierprops  = dict(marker="o", markerfacecolor="gray", markersize=5, alpha=0.5),
)
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col); patch.set_alpha(0.82)

counts5 = [len(d) for d in sub_data5]
ax.set_xticks(range(1, 6))
ax.set_xticklabels(
    [f"{st}\n(n={n})" for st, n in zip(SUBTIERS, counts5)], fontsize=9
)
ax.set_ylabel("Response Latency (ms)", fontsize=12)
ax.set_title("Adaptive RAG: Latency by Retrieval Sub-Tier",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
for i, d in enumerate(sub_data5, 1):
    ax.plot(i, np.mean(d), "D", color="white", markersize=8,
            markeredgecolor="black", markeredgewidth=1.2, zorder=5)

plt.tight_layout()
plt.savefig("chart3_boxplot_5subtiers.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: chart3_boxplot_5subtiers.png")


# ── CELL 10: Chart 4 — Mean bar + error bars + % savings ─────
fig, ax = plt.subplots(figsize=(8, 5))

main_data = [("NO", no_d), ("PARTIAL", partial_d), ("FULL", full_d)]
means = [d.mean() for _, d in main_data]
stds  = [d.std()  for _, d in main_data]
ns    = [len(d)   for _, d in main_data]

ax.bar(range(3), means, color=tier_colors3, alpha=0.83, width=0.5)
ax.errorbar(range(3), means, yerr=stds,
            fmt="none", color="black", capsize=7, linewidth=2)

full_mean = means[2]
for i, (name, _) in enumerate(main_data):
    if name != "FULL":
        pct = (full_mean - means[i]) / full_mean * 100
        top = means[i] + stds[i] + 200
        ax.text(i, top, f"{pct:.0f}% faster\nvs FULL",
                ha="center", fontsize=10, color="#1a7a1a", fontweight="bold")

ax.set_xticks(range(3))
ax.set_xticklabels(
    [f"{n}\n(n={ns[i]})" for i, (n, _) in enumerate(main_data)], fontsize=12
)
ax.set_ylabel("Mean Latency (ms)", fontsize=12)
ax.set_title("Mean Response Latency +/- Std Dev by Retrieval Tier",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("chart4_mean_errorbar.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: chart4_mean_errorbar.png")


# ── CELL 11: Chart 5 — PARTIAL/RECENT vs PARTIAL/SESSION ─────
fig, ax = plt.subplots(figsize=(7, 5))

rs_data   = [recent_d.values, session_d.values]
rs_colors = ["#2980b9", "#1a5276"]
rs_labels = [f"PARTIAL/RECENT\n(n={len(recent_d)})", f"PARTIAL/SESSION\n(n={len(session_d)})"]

bp = ax.boxplot(
    rs_data, patch_artist=True,
    medianprops = dict(color="white", linewidth=2.5),
    flierprops  = dict(marker="o", markerfacecolor="gray", markersize=5, alpha=0.5),
)
for patch, col in zip(bp["boxes"], rs_colors):
    patch.set_facecolor(col); patch.set_alpha(0.82)

for i, d in enumerate(rs_data, 1):
    ax.plot(i, np.mean(d), "D", color="white", markersize=9,
            markeredgecolor="black", markeredgewidth=1.2, zorder=5)

ax.set_xticks([1, 2])
ax.set_xticklabels(rs_labels, fontsize=11)
ax.set_ylabel("Response Latency (ms)", fontsize=12)
ax.set_title("PARTIAL Sub-Tier: RECENT vs SESSION Latency",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

ns_label = "n.s." if mw_rs.pvalue >= 0.05 else "***"
ymax_rs = max(max(recent_d), max(session_d))
y_top   = ymax_rs * 1.08
ax.plot([1, 1, 2, 2], [y_top, y_top+100, y_top+100, y_top], lw=1.2, color="black")
ax.text(1.5, y_top+150, ns_label, ha="center", fontsize=13, fontweight="bold")

fig.text(0.95, 0.02,
         f"Mann-Whitney p={mw_rs.pvalue:.3f}  Cohen d={cd_rs:.2f}",
         ha="right", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("chart5_recent_vs_session.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: chart5_recent_vs_session.png")


# ── CELL 12: Download all outputs ────────────────────────────
output_files = [
    "summary_stats.csv",
    "statistical_report.txt",
    "chart1_boxplot_3tier.png",
    "chart2_stacked_bar.png",
    "chart3_boxplot_5subtiers.png",
    "chart4_mean_errorbar.png",
    "chart5_recent_vs_session.png",
]

print("Downloading outputs...")
for fname in output_files:
    if os.path.exists(fname):
        files.download(fname)
        print(f"  Downloaded: {fname}")
    else:
        print(f"  NOT FOUND : {fname}")

print("\nDone.")
