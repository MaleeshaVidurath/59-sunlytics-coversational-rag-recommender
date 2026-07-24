"""
N5 evaluation — Kansei Psychology Knowledge Base.

Replaces the preliminary attempt's circular metric (KB-boost vs zero) with:
  1. paired KB-on vs KB-off retrieval runs over 30 emotional queries
     (full catalog_search pipeline; explanation generation stubbed out so
     only ranking is measured)
  2. a BLIND LLM judge (Groq): for each query it sees both top-3 sets as
     "Set A"/"Set B" (order randomised, config hidden, metadata only) and
     picks which better matches the emotional style → win rate + 95% CI
  3. the automated style-alignment score (% of top items whose colour/type
     is in the KB's preferred set for that style) — reported WITH the
     circularity caveat; the judge and the user study are primary evidence

Usage (from repo root, GROQ_API_KEY needed):
    python -m m2_multimodal_rag.evaluation.eval_kansei --config kb_on
    python -m m2_multimodal_rag.evaluation.eval_kansei --config kb_off
    python -m m2_multimodal_rag.evaluation.eval_kansei --config judge

Outputs: evaluation/results/kansei_items_{on,off}.csv, kansei_results.csv
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

SEED = 42
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

STYLE_QUERIES = {
    "elegant":      ["an elegant dress for a formal evening", "something elegant for a gala",
                     "an elegant blouse for a dinner party", "elegant trousers for an event",
                     "an elegant outfit top for the opera"],
    "casual":       ["a casual top for the weekend", "something casual for everyday wear",
                     "casual trousers for relaxing", "a casual sweater for home",
                     "a casual t-shirt for errands"],
    "sporty":       ["a sporty top for the gym", "something sporty for running",
                     "sporty shorts for training", "a sporty jacket for outdoor exercise",
                     "sporty leggings for a workout"],
    "romantic":     ["a romantic dress for a date night", "something romantic for valentine's dinner",
                     "a romantic blouse for an anniversary", "a romantic skirt for a picnic date",
                     "a romantic top for a candlelit dinner"],
    "professional": ["a professional blazer for the office", "something professional for a job interview",
                     "professional trousers for work", "a professional shirt for a client meeting",
                     "a professional dress for a conference"],
    "bold":         ["a bold statement top for a party", "something bold for a night out",
                     "a bold dress that stands out", "a bold jacket to draw attention",
                     "a bold outfit piece for a festival"],
}


def _run_config(kb_on: bool):
    # The ablation flag must be set BEFORE m2_handlers is imported.
    os.environ["M2_ABLATE_KB"] = "0" if kb_on else "1"
    os.environ["M2_ABLATE_GUARD"] = "none"   # ranking-only comparison

    from m2_multimodal_rag import m2_handlers
    # Stub out per-item explanation generation — N5 concerns ranking, not
    # explanations; this removes ~2 LLM+guard calls per query.
    m2_handlers.generator_loop.generate_faithful_explanation = \
        lambda article_id, force_hallucination_test=False, kb_fact="": ("", {})

    rows = []
    for style, queries in STYLE_QUERIES.items():
        for q in queries:
            retrieval_input = {
                "action": "catalog_search", "retrieval_strategy": "FULL",
                "user_message": q, "items_in_context": {}, "exclude_ids": [],
                "payload": {"filters": {}, "preference_boosts": [], "penalties": {},
                            "soft_constraints": {"style": style},
                            "purchase_history_hints": {}, "num_items": 3},
            }
            result = m2_handlers.handle_catalog_search(retrieval_input, {})
            for rank, item in enumerate(result.get("items", []), 1):
                rows.append({"style": style, "query": q, "rank": rank,
                             "article_id": item["article_id"],
                             "prod_name": item["prod_name"],
                             "colour": item["colour_group_name"],
                             "type": item["product_type_name"],
                             "appearance": item["graphical_appearance_name"]})
            print(f"  [{style}] '{q[:40]}' → "
                  f"{[i['prod_name'] for i in result.get('items', [])]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"kansei_items_{'on' if kb_on else 'off'}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out}")


def _judge():
    import pandas as pd
    import numpy as np
    from m2_multimodal_rag.llm_generator import llm_generator
    if not llm_generator.is_available:
        raise SystemExit("GROQ_API_KEY required for the judge stage.")

    on = pd.read_csv(RESULTS_DIR / "kansei_items_on.csv")
    off = pd.read_csv(RESULTS_DIR / "kansei_items_off.csv")
    random.seed(SEED)

    def fmt(items: "pd.DataFrame") -> str:
        return "; ".join(f"{r['prod_name']} ({r['colour']} {r['type']}, {r['appearance']})"
                         for _, r in items.iterrows())

    wins_on = wins_off = ties = 0
    records = []
    queries = on[["style", "query"]].drop_duplicates()
    for _, row in queries.iterrows():
        s, q = row["style"], row["query"]
        set_on = on[(on["style"] == s) & (on["query"] == q)].sort_values("rank")
        set_off = off[(off["style"] == s) & (off["query"] == q)].sort_values("rank")
        if set_on.empty or set_off.empty:
            continue
        flip = random.random() < 0.5
        a, b = (set_off, set_on) if flip else (set_on, set_off)
        prompt = (
            f"A fashion shopper asked for: \"{q}\" (desired style: {s}).\n"
            f"Two recommendation sets:\nSet A: {fmt(a)}\nSet B: {fmt(b)}\n"
            f"Which set better matches the '{s}' style the shopper asked for? "
            f"Reply with exactly one word: A, B, or TIE."
        )
        raw = llm_generator._call_llm(prompt, max_tokens=5, temperature=0.0)
        if raw is None:
            # API failure (e.g. daily quota) — must NOT be recorded as a tie.
            api_fails = sum(1 for r in records if r["winner"] == "api_fail") + 1
            records.append({"style": s, "query": q, "winner": "api_fail"})
            print(f"  [{s}] '{q[:40]}' → API FAIL")
            if api_fails >= 3:
                raise SystemExit(
                    "\nABORT: 3 judge calls failed (Groq quota likely exhausted). "
                    "The kansei_items_{on,off}.csv passes are saved — rerun ONLY "
                    "`--config judge` after the daily quota resets (~5k tokens).")
            continue
        verdict = raw.strip().upper()
        winner = "tie"
        if verdict.startswith("A"):
            winner = "kb_off" if flip else "kb_on"
        elif verdict.startswith("B"):
            winner = "kb_on" if flip else "kb_off"
        wins_on += winner == "kb_on"
        wins_off += winner == "kb_off"
        ties += winner == "tie"
        records.append({"style": s, "query": q, "winner": winner})
        print(f"  [{s}] '{q[:40]}' → {winner}")

    n = wins_on + wins_off + ties
    decided = wins_on + wins_off
    win_rate = wins_on / decided if decided else float("nan")
    # Wilson 95% CI for the KB-on win rate among decided comparisons
    if decided:
        z, p = 1.96, win_rate
        denom = 1 + z * z / decided
        centre = (p + z * z / (2 * decided)) / denom
        half = z * ((p * (1 - p) / decided + z * z / (4 * decided ** 2)) ** 0.5) / denom
        ci = (max(0.0, centre - half), min(1.0, centre + half))
    else:
        ci = (float("nan"), float("nan"))

    print(f"\nBlind LLM judge over {n} queries: KB-on wins {wins_on}, "
          f"KB-off wins {wins_off}, ties {ties}")
    print(f"KB-on win rate (decided): {win_rate:.1%}  95% CI [{ci[0]:.1%}, {ci[1]:.1%}]")

    with open(RESULTS_DIR / "kansei_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_queries", n])
        w.writerow(["kb_on_wins", wins_on])
        w.writerow(["kb_off_wins", wins_off])
        w.writerow(["ties", ties])
        w.writerow(["kb_on_win_rate_decided", round(win_rate, 4)])
        w.writerow(["ci_low", round(ci[0], 4)])
        w.writerow(["ci_high", round(ci[1], 4)])
    with open(RESULTS_DIR / "kansei_judgements.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'kansei_results.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["kb_on", "kb_off", "judge"], required=True)
    args = ap.parse_args()
    if args.config == "judge":
        _judge()
    else:
        _run_config(kb_on=args.config == "kb_on")


if __name__ == "__main__":
    main()
