# m3_implementation/test_result/adaptive_rag_result_v2/backfill_has_context.py
#
# Adds `has_context` / `context_source` to rows of latency_log.csv that were
# written before the logger recorded them.
#
# WHY THE COLUMN EXISTS
#   E2 scores the chosen tier against a table keyed on the intent label alone.
#   That table is state-blind, and the correct tier is not:
#
#       "tell me more about the first one"
#           items in context -> PARTIAL correct
#           context empty    -> PARTIAL IMPOSSIBLE; FULL is the right recovery,
#                               but the label-only table still says PARTIAL and
#                               records a misroute.
#
#   Three of the current misroutes are exactly this case.
#
# HOW OLD ROWS ARE RECOVERED
#   The true value (was the dialogue-state context window non-empty?) was not
#   logged, so it is reconstructed from the log itself:
#
#       has_context = (any EARLIER turn in the same session ran FULL or PARTIAL)
#
#   Retrieval is the only way items enter the context window, so a session with
#   no prior retrieval cannot have had anything to reuse.
#
#   THIS IS A RECONSTRUCTION, NOT A MEASUREMENT. It is marked
#   context_source="derived" so it can never be silently mixed with rows the
#   logger observed directly ("logged"). Two known ways it can be wrong:
#     - a prior retrieval that returned zero items still counts as context
#     - context evicted from the window by age still counts
#   Both make it OPTIMISTIC: it over-reports context, so it under-counts the
#   "PARTIAL was impossible" cases. The effect on E2 is therefore conservative.
#
# Idempotent: rows already carrying a context_source are left untouched.
#
# Run:  python test_result/adaptive_rag_result_v2/backfill_has_context.py [--apply]

import argparse
import os
import shutil

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "latency_log.csv"))


def derive(df: pd.DataFrame) -> pd.Series:
    """True when an earlier turn in the same session performed retrieval."""
    d = df.sort_values(["session_id", "timestamp"])
    retrieved = d["tier"].isin(["FULL", "PARTIAL"])
    prior = (retrieved.groupby(d["session_id"])
                      .apply(lambda s: s.shift(1, fill_value=False).cumsum())
                      .reset_index(level=0, drop=True))
    return (prior > 0).reindex(df.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the file (default is a dry run)")
    ap.add_argument("--log", default=LOG)
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    print(f"{args.log}\n  {len(df)} rows, {len(df.columns)} columns")

    for col in ("has_context", "context_source"):
        if col not in df.columns:
            df[col] = pd.NA
            print(f"  added column: {col}")

    missing = df["context_source"].isna() | (df["context_source"].astype(str).str.strip() == "")
    print(f"  rows needing backfill: {int(missing.sum())}")
    print(f"  rows already logged  : {int((~missing).sum())}")

    if missing.any():
        d = derive(df)
        df.loc[missing, "has_context"] = d[missing]
        df.loc[missing, "context_source"] = "derived"

    filled = df[df["context_source"] == "derived"]
    if len(filled):
        n_true = int(filled["has_context"].astype(bool).sum())
        print(f"\n  derived: has_context=True {n_true}  "
              f"False {len(filled) - n_true}")
        print("\n  by tier (derived rows):")
        print(filled.groupby(["tier", filled["has_context"].astype(bool)])
                    .size().to_string().replace("\n", "\n    "))

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply")
        return

    backup = args.log + ".bak"
    shutil.copy2(args.log, backup)
    df.to_csv(args.log, index=False)
    print(f"\n  backup -> {backup}")
    print(f"  written -> {args.log}  ({len(df.columns)} columns)")


if __name__ == "__main__":
    main()
