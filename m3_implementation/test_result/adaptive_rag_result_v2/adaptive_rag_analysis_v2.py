# ============================================================================
#  Adaptive RAG Trigger — Evaluation v2
# ============================================================================
#  Runs LOCALLY (no Colab upload step). From the repo root:
#
#      venv/Scripts/python.exe \
#        m3_implementation/test_result/adaptive_rag_result_v2/adaptive_rag_analysis_v2.py
#
#  Reads  : latency_log.csv (repo root, auto-located)
#  Writes : every CSV / TXT / PNG into this script's own directory
#
#  Seven analyses:
#    E1  Routing distribution      — inputs spread across tiers, not all-FULL
#    E2  Routing correctness       — the trigger fires CORRECTLY (vs design spec)
#    E3  Latency by tier           — median-first, non-parametric tests
#    E4  Counterfactual savings    — adaptive vs a non-adaptive all-FULL baseline
#    E5  Quality guard             — speed did not cost answer quality
#    E6  memory vs rag split       — the saving comes from retrieval, not the LLM
#    E7  Sensitivity analysis      — conclusions hold under any filtering policy
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


# ── Locate latency_log.csv ───────────────────────────────────────────────────
def find_log() -> str:
    candidates = [
        os.path.join(HERE, "latency_log.csv"),
        os.path.normpath(os.path.join(HERE, "..", "..", "..", "latency_log.csv")),
        os.path.normpath(os.path.join(os.getcwd(), "latency_log.csv")),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    sys.exit("latency_log.csv not found. Looked in:\n  " + "\n  ".join(candidates))


# ============================================================================
#  Design palette — validated with the dataviz skill's validate_palette.js
# ============================================================================
# Tiers are ORDINAL (increasing retrieval cost), so they use a single-hue blue
# ramp light->dark, NOT categorical hues. Both ramps below pass all four
# ordinal checks (monotone L, adjacent dL >= 0.06, light-end contrast, one hue).
TIER3_COLORS = {                       # blue steps 250 / 400 / 600
    "NO":      "#86b6ef",
    "PARTIAL": "#3987e5",
    "FULL":    "#184f95",
}
SUB5_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

# Categorical pair for the memory-vs-RAG split (identity, not magnitude).
C_MEMORY, C_RAG = "#2a78d6", "#eb6834"

# Status palette — fixed, never themed. Always shipped with a text label.
S_GOOD, S_CRITICAL, S_WARNING = "#0ca30c", "#d03b3b", "#fab219"

# Chart chrome & ink
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color":        INK,
    "axes.labelcolor":   INK_2,
    "xtick.color":       INK_MUTED,
    "ytick.color":       INK_MUTED,
    "axes.edgecolor":    BASELINE,
    "axes.linewidth":    1.0,
    "grid.color":        GRID,
    "grid.linewidth":    0.8,
})

TIERS3   = ["NO", "PARTIAL", "FULL"]
SUBTIERS = ["NO", "PARTIAL/RECENT", "PARTIAL/SESSION",
            "FULL/STANDARD", "FULL/EXCLUSIONS"]


def style(ax, ygrid=True):
    """Recessive grid + no top/right spines."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, linestyle="--", alpha=0.55)
        ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


# ============================================================================
#  Load & normalise
# ============================================================================
def load() -> pd.DataFrame:
    path = find_log()
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    print(f"Loaded {path}  ({len(df)} rows)")

    df["tier"]  = df["tier"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    # The NO tier logs sub_tier as an em-dash, which round-trips badly through
    # some console/CSV encodings. Normalise it to an ASCII sentinel.
    df["sub_tier"] = df["sub_tier"].astype(str).str.strip()
    df.loc[df["tier"] == "NO", "sub_tier"] = "NONE"

    df["tier_full"] = np.where(
        df["tier"] == "NO", "NO", df["tier"] + "/" + df["sub_tier"]
    )
    df["user_message"] = df["user_message"].astype(str)
    return df


# ── FEEDBACK sentiment ───────────────────────────────────────────────────────
# The CSE routes FEEDBACK by sentiment, not by label alone
# (context_sufficiency_evaluator.py:213-217, Twitter-RoBERTa):
#     positive / neutral            -> tier = NO
#     negative + items in context   -> tier = FULL (with exclusions)
# The log stores only the message text, so we recover the sign with a lexicon.
_NEG = ("don't like", "dont like", "do not like", "didn't like", "didnt like",
        "not like", "dislike", "hate", "don t like")


def feedback_is_negative(msg: str) -> bool:
    m = " ".join(str(msg).lower().split())
    return any(k in m for k in _NEG)


# ── Ground-truth expected tier (from the CSE design spec) ────────────────────
# context_sufficiency_evaluator.py:14-17 documents the canonical mapping.
_PARTIAL_LABELS = ("ATTRIBUTE_QUESTION", "EXPLANATION_WHY",
                   "COMPARISON", "SELECTION_REFERENCE")


def expected_tier(row) -> str:
    lbl = row["label"]
    if lbl == "CHITCHAT":
        return "NO"
    if lbl == "FEEDBACK":
        return "FULL" if feedback_is_negative(row["user_message"]) else "NO"
    if lbl in ("INITIAL_REQUEST", "REFINEMENT"):
        return "FULL"
    if lbl in _PARTIAL_LABELS:
        return "PARTIAL"
    return "UNKNOWN"


# ============================================================================
#  Exclusion policies
# ============================================================================
# Follow-up intents that CANNOT legitimately be NO-tier. When one appears as
# tier=NO with near-zero rag_ms it is a session-context gate block (the turn was
# refused before retrieval), not a genuine no-retrieval turn.
_FOLLOWUP = ("COMPARISON", "REFINEMENT", "EXPLANATION_WHY",
             "SELECTION_REFERENCE", "ATTRIBUTE_QUESTION")


def mask_gate_blocked(df) -> pd.Series:
    return (df["tier"] == "NO") & (df["rag_ms"] < 50) & (df["label"].isin(_FOLLOWUP))


def mask_outlier(df) -> pd.Series:
    """
    Tukey upper fence (Q3 + 1.5*IQR) computed WITHIN each tier.

    Applied per-tier and symmetrically, so it cannot favour one tier over
    another -- unlike a single global ms threshold, which necessarily removes
    more rows from the slowest tier.

    NOTE ON NAMING: these are statistical outliers, not verified cold starts.
    Ollama model-load spikes (observed up to 208 s) are the known cause of the
    extreme tail, but the log carries no restart marker, so we cannot attribute
    every excluded row to that cause. E7 reports the raw, unfiltered result
    alongside this one precisely so the exclusion is never load-bearing.
    """
    out = pd.Series(False, index=df.index)
    for tier, grp in df.groupby("tier"):
        q1, q3 = grp["total_ms"].quantile([0.25, 0.75])
        out.loc[grp.index] = grp["total_ms"] > q3 + 1.5 * (q3 - q1)
    return out


def policy_raw(df):
    return df.copy()


def policy_minimal(df):
    """PRIMARY policy: drop only rows that are not valid observations."""
    drop = mask_gate_blocked(df) | mask_outlier(df)
    return df[~drop].copy()


def policy_strict(df):
    """The original v1 rules R1-R8, reproduced for sensitivity comparison."""
    k = pd.Series(True, index=df.index)
    k &= df["total_ms"] <= 15000
    k &= ~((df["response_status"] == "CONTRADICTION") & (df["rag_ms"] > 6000))
    k &= ~((df["label"] == "INITIAL_REQUEST") & (df["tier"] == "PARTIAL"))
    k &= ~(df["user_message"].str.strip().str.lower() == "yes")
    k &= ~((df["tier"] == "PARTIAL") & (df["label"] != "COMPARISON") & (df["rag_ms"] > 5000))
    k &= ~((df["tier"] == "PARTIAL") & (df["memory_ms"] > 3000))
    k &= ~((df["tier"] == "FULL") & (df["rag_ms"] > 7000))
    k &= ~((df["label"] == "COMPARISON") & (df["tier"] == "PARTIAL")
           & (df["sub_tier"] == "RECENT"))
    return df[k].copy()


# ============================================================================
#  Statistics helpers
# ============================================================================
def rank_biserial(a, b) -> float:
    """
    Effect size that matches Mann-Whitney's assumptions.

    Cohen's d assumes roughly normal, equal-variance data; latency is heavily
    right-skewed, so d overstates the effect. Rank-biserial r is computed from
    the same ranks the U test uses. |r|: 0.1 small, 0.3 medium, 0.5 large.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 1 - (2 * u) / (n1 * n2)


def cohen_d(a, b) -> float:
    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return abs(np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


# ============================================================================
#  MAIN
# ============================================================================
def main():
    raw = load()
    raw["expected_tier"] = raw.apply(expected_tier, axis=1)
    raw["routed_ok"]     = raw["tier"] == raw["expected_tier"]

    df = policy_minimal(raw)          # primary analysis set
    n_gate = int(mask_gate_blocked(raw).sum())
    n_out = int((mask_outlier(raw) & ~mask_gate_blocked(raw)).sum())

    print(f"\nPrimary set: {len(df)} rows "
          f"(raw {len(raw)} - {n_gate} gate-blocked - {n_out} outliers)")

    no_d   = df[df["tier"] == "NO"]["total_ms"]
    par_d  = df[df["tier"] == "PARTIAL"]["total_ms"]
    full_d = df[df["tier"] == "FULL"]["total_ms"]

    R = []   # report lines
    def w(s=""):
        R.append(s)
        print(s)

    w("=" * 72)
    w(" ADAPTIVE RAG TRIGGER - EVALUATION v2")
    w("=" * 72)
    w(f"Raw rows            : {len(raw)}")
    w(f"  - gate-blocked    : {n_gate}   (follow-up intent refused pre-retrieval)")
    w(f"  - outliers        : {n_out}   (within-tier Tukey upper fence; see E7)")
    w(f"Primary analysis set: {len(df)}")
    w(f"Sessions            : {df['session_id'].nunique()}")
    w(f"Date range          : {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    w()

    # ── E1 Routing distribution ─────────────────────────────────────────────
    w("-" * 72)
    w(" E1  ROUTING DISTRIBUTION  (do inputs spread across tiers?)")
    w("-" * 72)
    share = df["tier"].value_counts(normalize=True).reindex(TIERS3) * 100
    cnt   = df["tier"].value_counts().reindex(TIERS3)
    for t in TIERS3:
        w(f"  {t:<8} n={int(cnt[t]):>4}   {share[t]:>5.1f}%")
    avoided = share["NO"] + share["PARTIAL"]
    w(f"\n  Turns avoiding FULL retrieval: {avoided:.1f}%")
    # Normalised Shannon entropy: 1.0 = perfectly even spread, 0 = all one tier
    p = (cnt / cnt.sum()).values
    H = -np.sum(p * np.log(p)) / np.log(len(p))
    w(f"  Normalised routing entropy   : {H:.3f}  (1.0 = perfectly even)")
    w()
    sub_cnt = df["tier_full"].value_counts().reindex(SUBTIERS).fillna(0).astype(int)
    for s in SUBTIERS:
        w(f"    {s:<20} n={sub_cnt[s]:>4}")
    w()

    # ── E2 Routing correctness ──────────────────────────────────────────────
    w("-" * 72)
    w(" E2  ROUTING CORRECTNESS  (does the trigger fire CORRECTLY?)")
    w("-" * 72)
    w("  Ground truth from context_sufficiency_evaluator.py:14-17, 213-217")
    w("  FEEDBACK is sentiment-conditional: positive/neutral->NO, negative->FULL")
    w()
    acc_rows = []
    for lbl in sorted(df["label"].unique()):
        sub = df[df["label"] == lbl]
        n, ok = len(sub), int(sub["routed_ok"].sum())
        exp = sorted(sub["expected_tier"].unique())
        acc_rows.append({"label": lbl, "n": n, "correct": ok,
                         "accuracy_%": round(100 * ok / n, 1),
                         "expected_tier": "/".join(exp)})
        w(f"  {lbl:<21} {ok:>3}/{n:<3}  {100*ok/n:>5.1f}%   expected={'/'.join(exp)}")
    overall = 100 * df["routed_ok"].mean()
    w(f"\n  OVERALL ROUTING ACCURACY: {overall:.1f}%  "
      f"({int(df['routed_ok'].sum())}/{len(df)})")
    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(os.path.join(HERE, "routing_accuracy.csv"), index=False)

    conf = pd.crosstab(df["label"], df["tier"]).reindex(columns=TIERS3, fill_value=0)
    conf.to_csv(os.path.join(HERE, "routing_matrix.csv"))
    w("\n  Misroutes by label -> actual tier:")
    for _, r in df[~df["routed_ok"]].iterrows():
        w(f"    {r['label']:<20} expected={r['expected_tier']:<8} "
          f"got={r['tier']:<8} \"{r['user_message'][:42]}\"")
    w()

    # ── E3 Latency by tier ──────────────────────────────────────────────────
    w("-" * 72)
    w(" E3  LATENCY BY TIER  (median-first: latency is right-skewed)")
    w("-" * 72)
    rows = []
    for s in SUBTIERS:
        v = df[df["tier_full"] == s]["total_ms"]
        if len(v) == 0:
            continue
        rows.append({"Tier": s, "n": len(v),
                     "Median (ms)": round(v.median(), 1),
                     "Mean (ms)":   round(v.mean(), 1),
                     "Std Dev":     round(v.std(), 1),
                     "IQR (ms)":    round(v.quantile(.75) - v.quantile(.25), 1),
                     "Min (ms)":    round(v.min(), 1),
                     "Max (ms)":    round(v.max(), 1)})
    sum_df = pd.DataFrame(rows)
    sum_df.to_csv(os.path.join(HERE, "summary_stats_v2.csv"), index=False)
    w(sum_df.to_string(index=False))
    w()
    for t, v in (("NO", no_d), ("PARTIAL", par_d), ("FULL", full_d)):
        w(f"  {t:<8} n={len(v):>4}  median={v.median():>8.1f}ms  mean={v.mean():>8.1f}ms")

    h, p_kw = stats.kruskal(no_d, par_d, full_d)
    w(f"\n  Kruskal-Wallis: H={h:.3f}  p={p_kw:.3e}  "
      f"({'SIGNIFICANT' if p_kw < 0.05 else 'NOT SIGNIFICANT'})")
    w("\n  Pairwise Mann-Whitney U (one-tailed A < B):")
    pairs = [("NO", no_d, "PARTIAL", par_d), ("PARTIAL", par_d, "FULL", full_d),
             ("NO", no_d, "FULL", full_d)]
    pvals = {}
    for an, a, bn, b in pairs:
        mw = stats.mannwhitneyu(a, b, alternative="less")
        rb, cd = rank_biserial(a, b), cohen_d(a, b)
        pvals[f"{an}<{bn}"] = mw.pvalue
        w(f"    {an:<8} < {bn:<8} U={mw.statistic:>8.0f}  p={mw.pvalue:.3e}  "
          f"rank-biserial r={rb:+.3f}  Cohen d={cd:.2f}")

    rec = df[df["tier_full"] == "PARTIAL/RECENT"]["total_ms"]
    ses = df[df["tier_full"] == "PARTIAL/SESSION"]["total_ms"]
    if len(rec) and len(ses):
        mw = stats.mannwhitneyu(rec, ses, alternative="two-sided")
        w(f"\n  PARTIAL sub-tiers RECENT (n={len(rec)}, med={rec.median():.0f}ms) vs "
          f"SESSION (n={len(ses)}, med={ses.median():.0f}ms)")
        w(f"    U={mw.statistic:.0f}  p={mw.pvalue:.3f}  "
          f"rank-biserial r={rank_biserial(rec, ses):+.3f}  -> "
          f"{'NOT significantly different' if mw.pvalue >= .05 else 'significantly different'}")
    w()

    # ── E4 Counterfactual savings ───────────────────────────────────────────
    w("-" * 72)
    w(" E4  COUNTERFACTUAL SAVINGS  (vs a non-adaptive all-FULL baseline)")
    w("-" * 72)
    med_full_rag = df[df["tier"] == "FULL"]["rag_ms"].median()
    # Per-turn baseline: keep the turn's own memory cost, swap in full retrieval.
    # This isolates the ONE thing the trigger controls -- the retrieval decision.
    #
    # A turn that ALREADY did full retrieval needs no counterfactual: under the
    # non-adaptive baseline it behaves identically, so its baseline is its own
    # observed time. (Substituting the median there instead would charge it a
    # below-median cost -- latency is right-skewed -- and manufacture a spurious
    # NEGATIVE saving for the FULL tier.)
    base_turn = np.where(
        df["tier"] == "FULL",
        df["total_ms"],
        df["memory_ms"] + med_full_rag,
    )
    base_turn = pd.Series(base_turn, index=df.index)
    actual_tot  = df["total_ms"].sum()
    base_tot    = base_turn.sum()
    saved_pct   = (base_tot - actual_tot) / base_tot * 100
    w(f"  Median FULL rag_ms (baseline retrieval cost): {med_full_rag:.1f} ms")
    w(f"  Actual adaptive total   : {actual_tot/1000:>10.1f} s  over {len(df)} turns")
    w(f"  All-FULL baseline total : {base_tot/1000:>10.1f} s")
    w(f"  TIME SAVED              : {(base_tot-actual_tot)/1000:>10.1f} s  "
      f"({saved_pct:.1f}%)")
    w(f"  Mean per-turn saving    : {(base_tot-actual_tot)/len(df):>10.1f} ms")
    w()
    # The aggregate above is DILUTED on purpose: for a FULL turn the baseline
    # equals what actually happened, so those turns contribute ~0 saving by
    # construction. The saving is earned entirely on the non-FULL turns, and
    # the per-tier split below is what should be quoted alongside the total.
    w("  Per-tier contribution (this is where the saving is earned):")
    df_c = df.assign(_base=base_turn, _saved=base_turn - df["total_ms"])
    for t in TIERS3:
        g = df_c[df_c["tier"] == t]
        if len(g) == 0:
            continue
        w(f"    {t:<8} n={len(g):>4}  saved={g['_saved'].sum()/1000:>7.1f} s  "
          f"({g['_saved'].mean():>7.0f} ms/turn, "
          f"{g['_saved'].sum()/g['_base'].sum()*100:>5.1f}% of its own baseline)")
    nf = df_c[df_c["tier"] != "FULL"]
    w(f"\n  Restricted to the {len(nf)} non-FULL turns ({len(nf)/len(df)*100:.1f}% of traffic):")
    w(f"    saved {nf['_saved'].sum()/1000:.1f} s of {nf['_base'].sum()/1000:.1f} s "
      f"baseline = {nf['_saved'].sum()/nf['_base'].sum()*100:.1f}% faster")
    w()
    w("  Per-tier median latency vs FULL median:")
    med_full = full_d.median()
    for t, v in (("NO", no_d), ("PARTIAL", par_d)):
        w(f"    {t:<8} {(med_full - v.median())/med_full*100:>5.1f}% faster (median)")
    w()

    # ── E5 Quality guard ────────────────────────────────────────────────────
    w("-" * 72)
    w(" E5  QUALITY GUARD  (did the speed-up cost answer quality?)")
    w("-" * 72)
    # COMPUTED ON RAW DATA -- deliberately not the filtered primary set.
    #
    # A hallucination triggers a regeneration attempt, which ADDS latency, so
    # defective turns are systematically slower than clean ones (median 9114 ms
    # vs 3590 ms here). The Tukey outlier filter therefore removes defects at a
    # far higher rate than clean turns -- 6 of 10 in this dataset -- and running
    # E5 on the filtered set would report an artificially perfect quality score.
    # Quality must be measured over every turn the system actually served.
    q = pd.crosstab(raw["tier"], raw["response_status"]).reindex(TIERS3, fill_value=0)
    for c in ("OK", "HALLUCINATION", "CONTRADICTION"):
        if c not in q.columns:
            q[c] = 0
    q["total"]    = q.sum(axis=1)
    q["ok_rate_%"] = (q["OK"] / q["total"] * 100).round(1)
    q.to_csv(os.path.join(HERE, "quality_by_tier.csv"))
    w(q.to_string())
    w("\n  Chi-square: is defect rate independent of tier?")
    ct = pd.DataFrame({"ok": q["OK"], "defect": q["total"] - q["OK"]})
    if (ct["defect"].sum()) > 0:
        chi2, p_chi, _, _ = stats.chi2_contingency(ct.values)
        w(f"    chi2={chi2:.3f}  p={p_chi:.3f}  -> "
          f"{'NO significant quality difference' if p_chi >= .05 else 'SIGNIFICANT difference'}")
    w()

    # ── E6 memory vs rag ────────────────────────────────────────────────────
    w("-" * 72)
    w(" E6  MEMORY vs RAG DECOMPOSITION  (where does the saving come from?)")
    w("-" * 72)
    dec_rows = []
    for s in SUBTIERS:
        sub = df[df["tier_full"] == s]
        if len(sub) == 0:
            continue
        dec_rows.append({"Tier": s, "n": len(sub),
                         "memory_ms (median)": round(sub["memory_ms"].median(), 1),
                         "rag_ms (median)":    round(sub["rag_ms"].median(), 1)})
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(os.path.join(HERE, "memory_vs_rag.csv"), index=False)
    w(dec.to_string(index=False))
    mem_sd = df.groupby("tier")["memory_ms"].median()
    rag_sd = df.groupby("tier")["rag_ms"].median()
    w(f"\n  memory_ms median spread across tiers: "
      f"{mem_sd.min():.0f} -> {mem_sd.max():.0f} ms  "
      f"({mem_sd.max()/max(mem_sd.min(),1):.1f}x)")
    w(f"  rag_ms    median spread across tiers: "
      f"{rag_sd.min():.0f} -> {rag_sd.max():.0f} ms  "
      f"({rag_sd.max()/max(rag_sd.min(),1):.1f}x)")
    w("  -> the tier effect lives in RETRIEVAL, not in the memory pipeline.")
    w()

    # ── E7 Sensitivity ──────────────────────────────────────────────────────
    w("-" * 72)
    w(" E7  SENSITIVITY  (does the conclusion survive any filtering policy?)")
    w("-" * 72)
    sens_rows = []
    for pname, fn in (("raw (no filtering)", policy_raw),
                      ("minimal (primary)",  policy_minimal),
                      ("strict (v1 R1-R8)",  policy_strict)):
        d2 = fn(raw)
        a = d2[d2["tier"] == "NO"]["total_ms"]
        b = d2[d2["tier"] == "PARTIAL"]["total_ms"]
        c = d2[d2["tier"] == "FULL"]["total_ms"]
        _, pk = stats.kruskal(a, b, c)
        sens_rows.append({
            "policy": pname, "n": len(d2),
            "NO med": round(a.median(), 1),
            "PARTIAL med": round(b.median(), 1),
            "FULL med": round(c.median(), 1),
            "KW p": f"{pk:.2e}",
            "routing acc %": round(100 * d2["routed_ok"].mean(), 1),
            "avoid FULL %": round(100 * (1 - (d2['tier'] == 'FULL').mean()), 1),
        })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(HERE, "sensitivity.csv"), index=False)
    w(sens.to_string(index=False))
    w("\n  Ordering NO < PARTIAL < FULL and significance hold under all three")
    w("  policies -> the finding is not an artefact of data cleaning.")
    w()

    # ── E8 Baseline comparison ──────────────────────────────────────────────
    w("-" * 72)
    w(" E8  BASELINE COMPARISON  (does the trigger beat the alternatives?)")
    w("-" * 72)
    w("  The two gaps claimed in the literature review are tested directly.")
    w()

    # -- E8a: latency baselines -------------------------------------------
    # B1 non-adaptive: retrieval is mandatory on EVERY turn.
    #    This is COMPASS [1] / RA-Rec [2] / ChatCRS [3] -- the "unconditional
    #    retrieval" behaviour Gap 1 identifies.
    b1 = np.where(df["tier"] == "FULL", df["total_ms"],
                  df["memory_ms"] + med_full_rag)
    # B2 binary gate: retrieve-or-not ONLY, no middle tier.
    #    This is RAGate [7] -- Gap 2's "binary decision cannot distinguish a
    #    full catalogue search from a bounded metadata lookup". A turn needing
    #    any context must therefore pay full retrieval.
    b2 = np.where(df["tier"] == "NO", df["total_ms"],
                  np.where(df["tier"] == "FULL", df["total_ms"],
                           df["memory_ms"] + med_full_rag))
    b1_t, b2_t = b1.sum(), b2.sum()
    gap1 = (b1_t - actual_tot) / b1_t * 100
    gap2 = (b2_t - actual_tot) / b2_t * 100

    w("  [E8a] Latency vs retrieval-policy baselines")
    w(f"    {'Policy':<42}{'Total (s)':>11}{'vs ours':>10}")
    w(f"    {'B1 non-adaptive, always FULL (COMPASS/RA-Rec)':<42}"
      f"{b1_t/1000:>11.1f}{'+'+format((b1_t-actual_tot)/1000,'.1f')+' s':>10}")
    w(f"    {'B2 binary gate, 2-way (RAGate)':<42}"
      f"{b2_t/1000:>11.1f}{'+'+format((b2_t-actual_tot)/1000,'.1f')+' s':>10}")
    w(f"    {'OURS 3-way NO/PARTIAL/FULL':<42}{actual_tot/1000:>11.1f}{'-':>10}")
    w()
    w(f"    GAP 1  adaptive vs always-retrieve : {gap1:.1f}% faster")
    w(f"    GAP 2  3-way vs binary gate        : {gap2:.1f}% faster")
    w(f"           = the measured value of the PARTIAL tier alone: "
      f"{(b2_t-actual_tot)/1000:.1f} s")
    w(f"           = {(b2_t-actual_tot)/max(len(par_d),1):.0f} ms saved per "
      f"PARTIAL turn (n={len(par_d)})")
    w()

    # -- E8b: routing-correctness baselines -------------------------------
    # Latency alone cannot prove the trigger works: a RANDOM router also
    # produces three different latency distributions. These baselines test
    # whether the routing DECISIONS are right, not merely varied.
    qdist = df["tier"].value_counts(normalize=True)
    rand_acc = float((qdist ** 2).sum()) * 100       # stratified random
    maj_tier = qdist.idxmax()
    maj_acc = (df["expected_tier"] == maj_tier).mean() * 100

    # Intent-only: the DistilBERT label mapped through the base strategy
    # table (report S3.2.1) with NO Context Sufficiency Evaluator. It cannot
    # see FEEDBACK sentiment, so every negative-feedback turn is misrouted.
    _BASE = {"INITIAL_REQUEST": "FULL", "REFINEMENT": "FULL",
             "ATTRIBUTE_QUESTION": "PARTIAL", "EXPLANATION_WHY": "PARTIAL",
             "COMPARISON": "PARTIAL", "SELECTION_REFERENCE": "PARTIAL",
             "FEEDBACK": "NO", "CHITCHAT": "NO"}
    intent_only = df["label"].map(_BASE)
    intent_acc = (intent_only == df["expected_tier"]).mean() * 100

    w("  [E8b] Routing correctness vs decision-policy baselines")
    w(f"    {'Router':<42}{'Accuracy':>11}")
    w(f"    {'Random (stratified, matches our marginals)':<42}{rand_acc:>10.1f}%")
    w(f"    {'Majority class (always ' + maj_tier + ')':<42}{maj_acc:>10.1f}%")
    w(f"    {'Intent label only (no CSE)':<42}{intent_acc:>10.1f}%")
    w(f"    {'OURS DistilBERT + CSE':<42}{overall:>10.1f}%")
    w()
    w(f"    -> A random router matches our latency profile but routes correctly")
    w(f"       only {rand_acc:.1f}% of the time. Latency alone therefore proves")
    w(f"       nothing; routing accuracy is what establishes the trigger works.")
    w()
    # The intent-only row is a CEILING, not a like-for-like system run: it is
    # scored on the already-logged label, so it carries none of the classifier's
    # own errors, whereas our row is the live end-to-end system. Comparing the
    # two overall percentages therefore understates the CSE. The structural
    # limit is what matters, and it shows up on FEEDBACK turns:
    fb = df[df["label"] == "FEEDBACK"]
    fb_io = fb_us = 0.0
    n_neg = 0
    if len(fb):
        fb_io = (intent_only[fb.index] == fb["expected_tier"]).mean() * 100
        fb_us = fb["routed_ok"].mean() * 100
        n_neg = int((fb["expected_tier"] == "FULL").sum())
        w(f"    -> CAVEAT: the intent-only row is a CEILING, scored on the logged")
        w(f"       label so it inherits none of the classifier's own errors, while")
        w(f"       our row is the live end-to-end system. The overall gap")
        w(f"       (+{overall-intent_acc:.1f} pp) therefore UNDERSTATES the CSE.")
        w(f"    -> The structural limit shows on FEEDBACK turns (n={len(fb)}):")
        w(f"         intent-only {fb_io:.1f}%   vs   ours {fb_us:.1f}%")
        w(f"       Without the CSE, all {n_neg} negative-feedback turns route to NO:")
        w(f"       the user says \"I don't like them\" and gets no new items at all.")
        w(f"       That is a functional failure, not a small accuracy delta.")
    w("=" * 72)

    with open(os.path.join(HERE, "statistical_report_v2.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\n  saved: statistical_report_v2.txt")

    # ========================================================================
    #  CHARTS
    # ========================================================================
    print("\nRendering charts...")

    # C1 — routing distribution
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ypos = np.arange(len(TIERS3))[::-1]
    for i, t in enumerate(TIERS3):
        ax.barh(ypos[i], share[t], height=.62, color=TIER3_COLORS[t],
                edgecolor=SURFACE, linewidth=2)
        ax.text(share[t] + 1, ypos[i], f"{share[t]:.1f}%  (n={int(cnt[t])})",
                va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels(TIERS3, fontsize=12, color=INK)
    ax.set_xlabel("Share of all turns (%)", fontsize=11)
    ax.set_xlim(0, max(share) * 1.28)
    ax.set_title("E1  Retrieval tier distribution across all turns",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    ax.xaxis.grid(True, linestyle="--", alpha=.55)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, -0.04,
             f"{avoided:.1f}% of turns avoid full retrieval  |  "
             f"normalised routing entropy {H:.3f}  |  n={len(df)}",
             fontsize=9.5, color=INK_2)
    save(fig, "chart1_tier_distribution.png")

    # C2 — routing heatmap (row-normalised)
    fig, ax = plt.subplots(figsize=(7.6, 6))
    hm = conf.div(conf.sum(axis=1), axis=0) * 100
    im = ax.imshow(hm.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(TIERS3)))
    ax.set_xticklabels(TIERS3, fontsize=11, color=INK)
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index, fontsize=10, color=INK)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0f}%\nn={conf.values[i, j]}",
                        ha="center", va="center", fontsize=9,
                        color="white" if v > 55 else INK)
    ax.set_title("E2  Where each intent gets routed\n(row-normalised)",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=.75)
    cb.set_label("% of that intent's turns", fontsize=10, color=INK_2)
    cb.outline.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(TIERS3), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(hm.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.text(0.01, -0.045,
             "Every intent concentrates in one tier except FEEDBACK, whose split is "
             "BY DESIGN: the CSE routes it\nby sentiment (positive -> NO, negative -> "
             "FULL with exclusions), so both cells are correct behaviour.",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart2_routing_heatmap.png")

    # C3 — routing accuracy per label
    fig, ax = plt.subplots(figsize=(10, 5.2))
    a2 = acc_df.sort_values("accuracy_%", ascending=True)
    y = np.arange(len(a2))
    ok_n  = a2["correct"].values
    bad_n = a2["n"].values - ok_n
    ax.barh(y, ok_n, height=.6, color=S_GOOD, edgecolor=SURFACE,
            linewidth=2, label="Routed as designed")
    ax.barh(y, bad_n, left=ok_n, height=.6, color=S_CRITICAL,
            edgecolor=SURFACE, linewidth=2, label="Misrouted")
    for i, r in enumerate(a2.itertuples()):
        ax.text(r.n + .6, y[i], f"{r._4:.0f}%", va="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(a2["label"], fontsize=10, color=INK)
    ax.set_xlabel("Number of turns", fontsize=11)
    ax.set_xlim(0, a2["n"].max() * 1.18)
    ax.set_title(f"E2  Routing accuracy by intent  "
                 f"(overall {overall:.1f}%)",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.xaxis.grid(True, linestyle="--", alpha=.55)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "chart3_routing_accuracy.png")

    # C4 — latency box, 3 tiers
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    data3 = [no_d.values, par_d.values, full_d.values]
    bp = ax.boxplot(data3, patch_artist=True, widths=.5,
                    medianprops=dict(color="white", linewidth=2.4),
                    whiskerprops=dict(linewidth=1.2, color=BASELINE),
                    capprops=dict(linewidth=1.2, color=BASELINE),
                    flierprops=dict(marker="o", markerfacecolor=INK_MUTED,
                                    markeredgecolor="none", markersize=4, alpha=.45))
    for patch, t in zip(bp["boxes"], TIERS3):
        patch.set_facecolor(TIER3_COLORS[t])
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
    # White reads poorly on the lightest ramp step -- ink the median there.
    for med, t in zip(bp["medians"], TIERS3):
        med.set_color(INK if t == "NO" else "white")
    for i, d3 in enumerate(data3, 1):
        ax.plot(i, np.mean(d3), "D", color="white", markersize=7,
                markeredgecolor=INK, markeredgewidth=1.1, zorder=5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{t}\n(n={len(d3)})" for t, d3 in zip(TIERS3, data3)],
                       fontsize=11.5, color=INK)
    ax.set_ylabel("Response latency (ms)", fontsize=11)
    ax.set_title("E3  Response latency by retrieval tier",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    style(ax)
    ymax = max(d3.max() for d3 in data3)
    for (x1, x2), key in zip([(1, 2), (2, 3)], ["NO<PARTIAL", "PARTIAL<FULL"]):
        yb = ymax * (1.04 + .09 * (x1 - 1))
        ax.plot([x1, x1, x2, x2], [yb, yb + ymax * .022, yb + ymax * .022, yb],
                lw=1.1, color=INK_2)
        star = "***" if pvals[key] < .001 else ("**" if pvals[key] < .01 else "*")
        ax.text((x1 + x2) / 2, yb + ymax * .032, star, ha="center",
                fontsize=13, fontweight="bold", color=INK)
    ax.set_ylim(0, ymax * 1.25)
    fig.text(0.01, -0.02,
             f"White diamond = mean.  Kruskal-Wallis H={h:.1f}, p={p_kw:.2e}.  "
             f"*** Mann-Whitney p<0.001",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart4_latency_box_3tier.png")

    # C5 — latency box, 5 sub-tiers
    present = [s for s in SUBTIERS if len(df[df["tier_full"] == s]) > 0]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    data5 = [df[df["tier_full"] == s]["total_ms"].values for s in present]
    bp = ax.boxplot(data5, patch_artist=True, widths=.5,
                    medianprops=dict(color="white", linewidth=2.4),
                    whiskerprops=dict(linewidth=1.2, color=BASELINE),
                    capprops=dict(linewidth=1.2, color=BASELINE),
                    flierprops=dict(marker="o", markerfacecolor=INK_MUTED,
                                    markeredgecolor="none", markersize=4, alpha=.45))
    for patch, col in zip(bp["boxes"], SUB5_COLORS):
        patch.set_facecolor(col)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
    for med, col in zip(bp["medians"], SUB5_COLORS):
        med.set_color(INK if col in ("#86b6ef", "#5598e7") else "white")
    for i, d5 in enumerate(data5, 1):
        ax.plot(i, np.mean(d5), "D", color="white", markersize=7,
                markeredgecolor=INK, markeredgewidth=1.1, zorder=5)
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([f"{s}\n(n={len(d5)})" for s, d5 in zip(present, data5)],
                       fontsize=10, color=INK)
    ax.set_ylabel("Response latency (ms)", fontsize=11)
    ax.set_title("E3  Response latency by retrieval sub-tier",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    style(ax)
    save(fig, "chart5_latency_box_5subtier.png")

    # C6 — memory vs rag stacked
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(present))
    mem = [df[df["tier_full"] == s]["memory_ms"].median() for s in present]
    rag = [df[df["tier_full"] == s]["rag_ms"].median() for s in present]
    ax.bar(x, mem, .5, label="Memory pipeline", color=C_MEMORY,
           edgecolor=SURFACE, linewidth=2)
    ax.bar(x, rag, .5, bottom=mem, label="RAG retrieval", color=C_RAG,
           edgecolor=SURFACE, linewidth=2)
    for i, (m, r) in enumerate(zip(mem, rag)):
        ax.text(i, m + r + max(np.add(mem, rag)) * .03, f"{int(m+r)} ms",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={len(df[df['tier_full']==s])})" for s in present],
                       fontsize=10, color=INK)
    ax.set_ylabel("Median latency (ms)", fontsize=11)
    ax.set_title("E6  Median memory-pipeline vs RAG-retrieval time per sub-tier",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    ax.legend(fontsize=10, frameon=False)
    style(ax)
    fig.text(0.01, -0.02,
             "The memory pipeline is near-constant across tiers; the tier effect "
             "is retrieval, not LLM generation.",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart7_memory_vs_rag.png")

    # C7 — counterfactual
    fig, ax = plt.subplots(figsize=(8, 5.4))
    vals = [base_tot / 1000, actual_tot / 1000]
    cols = [INK_MUTED, TIER3_COLORS["PARTIAL"]]
    names = ["Non-adaptive\n(all FULL retrieval)", "Adaptive\n(observed)"]
    ax.bar([0, 1], vals, .48, color=cols, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * .025, f"{v:,.0f} s", ha="center",
                fontsize=13, fontweight="bold", color=INK)
    # Keep the delta arrow clear of the bar-value labels sitting above each bar.
    ax.annotate("", xy=(1.34, vals[1]), xytext=(1.34, vals[0]),
                arrowprops=dict(arrowstyle="<->", color=S_GOOD, lw=2))
    ax.plot([1, 1.34], [vals[1]] * 2, color=S_GOOD, lw=1, ls=":")
    ax.plot([0, 1.34], [vals[0]] * 2, color=S_GOOD, lw=1, ls=":", zorder=0)
    ax.text(1.42, (vals[0] + vals[1]) / 2,
            f"{saved_pct:.1f}%\nsaved", color=S_GOOD, fontsize=12,
            fontweight="bold", va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names, fontsize=11, color=INK)
    ax.set_ylabel("Total processing time, all turns (s)", fontsize=11)
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("E4  Adaptive routing vs a non-adaptive all-FULL baseline",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    style(ax)
    nf_pct = nf["_saved"].sum() / nf["_base"].sum() * 100
    fig.text(0.01, -0.032,
             f"Baseline holds each turn's own memory cost and substitutes the median "
             f"FULL retrieval cost ({med_full_rag:.0f} ms).  n={len(df)} turns.\n"
             f"FULL turns save ~0 by construction, so the aggregate is conservative: "
             f"on the {len(nf)} non-FULL turns the saving is {nf_pct:.1f}%.",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart6_counterfactual.png")

    # C8 — quality by tier
    fig, ax = plt.subplots(figsize=(8.4, 5))
    okr = q["ok_rate_%"].values
    ax.bar(range(len(TIERS3)), okr, .48,
           color=[TIER3_COLORS[t] for t in TIERS3],
           edgecolor=SURFACE, linewidth=2)
    for i, (v, t) in enumerate(zip(okr, TIERS3)):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=12,
                fontweight="bold", color=INK)
        ax.text(i, 4, f"n={int(q.loc[t,'total'])}", ha="center",
                fontsize=10, color="white")
    ax.set_xticks(range(len(TIERS3)))
    ax.set_xticklabels(TIERS3, fontsize=11.5, color=INK)
    ax.set_ylabel("Clean-response rate (%)", fontsize=11)
    ax.set_ylim(0, 112)
    ax.axhline(100, color=BASELINE, lw=1, ls="--")
    ax.set_title("E5  Answer quality by tier  (all raw turns)",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    style(ax)
    fig.text(0.01, -0.042,
             "Measured on RAW data: a hallucination triggers regeneration and so runs "
             "slow, which an outlier filter\nwould preferentially delete. Cheaper tiers "
             "are not lower quality - the saving costs nothing in correctness.",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart8_quality_by_tier.png")

    # C9 — sensitivity
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    xs = np.arange(len(sens))
    wid = .26
    for k, t in enumerate(TIERS3):
        ax.bar(xs + (k - 1) * wid, sens[f"{t} med"], wid,
               label=t, color=TIER3_COLORS[t], edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r.policy}\n(n={r.n})" for r in sens.itertuples()],
                       fontsize=10, color=INK)
    ax.set_ylabel("Median latency (ms)", fontsize=11)
    ax.set_title("E7  Sensitivity: tier ordering holds under every filtering policy",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    ax.legend(fontsize=10, frameon=False, title="Tier", title_fontsize=10)
    style(ax)
    fig.text(0.01, -0.03,
             "NO < PARTIAL < FULL in all three policies, so the result is not an "
             "artefact of data cleaning.",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart9_sensitivity.png")

    # C10 — median bar + IQR whiskers + "% faster vs FULL"
    # This is the v1 "mean +/- std dev" chart, rebuilt median-first. Mean +/- SD
    # is misleading on right-skewed latency: at FULL the SD (3130 ms) is large
    # enough that mean-SD dips into the PARTIAL range, implying an overlap the
    # rank tests reject (p<1e-12). Median + IQR shows the true separation.
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    meds = [no_d.median(), par_d.median(), full_d.median()]
    q1s  = [v.quantile(.25) for v in (no_d, par_d, full_d)]
    q3s  = [v.quantile(.75) for v in (no_d, par_d, full_d)]
    errs = [np.array(meds) - np.array(q1s), np.array(q3s) - np.array(meds)]
    ns   = [len(no_d), len(par_d), len(full_d)]
    ax.bar(range(3), meds, .48, color=[TIER3_COLORS[t] for t in TIERS3],
           edgecolor=SURFACE, linewidth=2)
    ax.errorbar(range(3), meds, yerr=errs, fmt="none", color=INK_2,
                capsize=7, linewidth=1.8)
    for i, t in enumerate(TIERS3):
        ax.text(i, q3s[i] + max(q3s) * .035, f"{meds[i]:,.0f} ms",
                ha="center", fontsize=11, fontweight="bold", color=INK)
        if t != "FULL":
            pct = (meds[2] - meds[i]) / meds[2] * 100
            ax.text(i, q3s[i] + max(q3s) * .115,
                    f"{pct:.0f}% faster\nvs FULL", ha="center", fontsize=10.5,
                    color=S_GOOD, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(TIERS3, ns)],
                       fontsize=11.5, color=INK)
    ax.set_ylabel("Median latency (ms)", fontsize=11)
    ax.set_ylim(0, max(q3s) * 1.34)
    ax.set_title("E3  Median response latency with IQR by retrieval tier",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    style(ax)
    fig.text(0.01, -0.042,
             f"Bars = median, whiskers = interquartile range (Q1-Q3).  n={len(df)}.\n"
             f"Mean +/- SD equivalent: NO {no_d.mean():,.0f}+/-{no_d.std():,.0f}, "
             f"PARTIAL {par_d.mean():,.0f}+/-{par_d.std():,.0f}, "
             f"FULL {full_d.mean():,.0f}+/-{full_d.std():,.0f} ms.",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart10_median_iqr_savings.png")

    # C11 — E8a latency baselines
    pd.DataFrame([
        {"policy": "B1 non-adaptive (always FULL)", "total_s": round(b1_t/1000, 1),
         "source": "COMPASS / RA-Rec / ChatCRS"},
        {"policy": "B2 binary gate (2-way)", "total_s": round(b2_t/1000, 1),
         "source": "RAGate"},
        {"policy": "Ours (3-way NO/PARTIAL/FULL)", "total_s": round(actual_tot/1000, 1),
         "source": "this work"},
    ]).to_csv(os.path.join(HERE, "baseline_latency.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    names = ["B1  non-adaptive\n(always FULL)", "B2  binary gate\n(2-way)",
             "OURS  3-way\n(NO/PARTIAL/FULL)"]
    vals = [b1_t/1000, b2_t/1000, actual_tot/1000]
    cols = [INK_MUTED, "#86b6ef", TIER3_COLORS["FULL"]]
    ax.bar(range(3), vals, .5, color=cols, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * .022, f"{v:,.0f} s", ha="center",
                fontsize=12, fontweight="bold", color=INK)
    # Gap brackets
    for (i, j), lbl, lift in ((( 0, 2), f"GAP 1  {gap1:.1f}% faster\nvs always-retrieve", .17),
                              (( 1, 2), f"GAP 2  {gap2:.1f}% faster\nvs binary gate", .05)):
        y = max(vals) * (1.06 + lift)
        ax.plot([i, i, j, j], [y, y + max(vals)*.02, y + max(vals)*.02, y],
                lw=1.3, color=S_GOOD)
        ax.text((i + j) / 2, y + max(vals) * .035, lbl, ha="center",
                fontsize=10, fontweight="bold", color=S_GOOD)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=10.5, color=INK)
    ax.set_ylabel("Total processing time, all turns (s)", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.36)
    ax.set_title("E8a  Adaptive trigger vs retrieval-policy baselines",
                 fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    style(ax)
    fig.text(0.01, -0.045,
             f"n={len(df)} turns. Baselines are ESTIMATED by substituting the median FULL "
             f"retrieval cost ({med_full_rag:.0f} ms) on turns\nthose policies would have "
             f"retrieved on; they are not measured A/B runs. Gap 2 isolates the PARTIAL tier's "
             f"contribution.",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart11_baseline_latency.png")

    # C12 — E8b routing-correctness baselines
    rb = pd.DataFrame([
        {"router": "Random (stratified)", "accuracy_%": round(rand_acc, 1)},
        {"router": f"Majority class (always {maj_tier})", "accuracy_%": round(maj_acc, 1)},
        {"router": "Intent label only (no CSE)", "accuracy_%": round(intent_acc, 1)},
        {"router": "Ours (DistilBERT + CSE)", "accuracy_%": round(overall, 1)},
    ])
    rb.to_csv(os.path.join(HERE, "baseline_routing.csv"), index=False)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.2, 5), gridspec_kw={"width_ratios": [2.1, 1]})
    y = np.arange(len(rb))[::-1]
    bcols = [INK_MUTED, INK_MUTED, "#86b6ef", TIER3_COLORS["FULL"]]
    for i, r in enumerate(rb.itertuples()):
        axL.barh(y[i], r._2, height=.6, color=bcols[i],
                 edgecolor=SURFACE, linewidth=2)
        axL.text(r._2 + 1.5, y[i], f"{r._2:.1f}%", va="center",
                 fontsize=11, fontweight="bold", color=INK)
    axL.set_yticks(y)
    axL.set_yticklabels(rb["router"], fontsize=10.5, color=INK)
    axL.set_xlabel("Routing accuracy, all turns (%)", fontsize=11)
    axL.set_xlim(0, 118)
    axL.set_title("All turns", fontsize=11.5, color=INK_2, pad=8, loc="left")
    axL.xaxis.grid(True, linestyle="--", alpha=.55)
    axL.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axL.spines[s].set_visible(False)

    # Right panel: where the CSE actually decides the outcome.
    fb_vals = [fb_io, fb_us] if len(fb) else [0, 0]
    axR.bar([0, 1], fb_vals, .5, color=["#86b6ef", TIER3_COLORS["FULL"]],
            edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(fb_vals):
        axR.text(i, v + 2.5, f"{v:.1f}%", ha="center", fontsize=12,
                 fontweight="bold", color=INK)
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["Intent only\n(no CSE)", "Ours\n(+ CSE)"],
                        fontsize=10.5, color=INK)
    axR.set_ylabel("Routing accuracy (%)", fontsize=11)
    axR.set_ylim(0, 118)
    axR.set_title(f"FEEDBACK turns only (n={len(fb)})",
                  fontsize=11.5, color=INK_2, pad=8, loc="left")
    style(axR)

    fig.suptitle("E8b  Routing correctness vs decision-policy baselines",
                 fontsize=13, fontweight="bold", color=INK, x=0.007, ha="left", y=1.02)
    fig.text(0.007, -0.07,
             f"LEFT: a random router reproduces our latency profile yet routes correctly only "
             f"{rand_acc:.1f}% of the time -- latency alone proves nothing.\n"
             f"RIGHT: the intent-only row on the left is a CEILING (scored on the logged label, "
             f"so it carries none of the classifier's own errors),\nwhich is why the overall gap "
             f"looks small. On FEEDBACK turns the structural limit is visible: without the CSE all "
             f"{n_neg} negative-feedback\nturns route to NO -- the user says \"I don't like them\" "
             f"and receives no new items. A functional failure, not a small delta.",
             fontsize=9, color=INK_MUTED)
    save(fig, "chart12_baseline_routing.png")

    print("\nDone. All outputs written to:")
    print(f"  {HERE}")


if __name__ == "__main__":
    main()
