# m3_implementation/test_result/hallucination_result/expanded_eval/mongo_harvest.py
#
# Harvests REAL chat history from MongoDB into evaluation cases.
#
# The live system stores every turn (sessions.turns[]) and, for
# recommendation turns, the items shown (recommendations collection, linked
# via recommendation_id). Those items ARE the evidence for catalog responses,
# so genuine (evidence, response) pairs can be reconstructed for every
# historical recommendation turn — adding real human conversations to the
# dataset alongside the scripted driver runs.
#
# WHAT IS / ISN'T HARVESTABLE
#   catalog-style turns (recommendation attached)  → harvestable (this script)
#   attribute/compare/explanation turns            → NOT harvestable: their
#       evidence bundle (extracted_facts, item_a/item_b, ...) is never stored
#
# PIPELINE
#   1. join turns × recommendations, map ItemInContext fields → checker fields
#   2. quality filter: ≥1 item, and the response actually references the
#      items (item name or £ present) — drops refusals/fallbacks
#   3. de-duplicate by normalized response text (also removes overlap with
#      scripted-driver captures, which live in the same DB)
#   4. run checker v3 OFFLINE on each pair → presumed-clean vs flagged split
#      (same convention as live capture; clean still needs the human audit)
#
# Output rows use the exact captured_cases.jsonl schema, with
# "source": "mongodb" added.
#
# Run:  python test_result/hallucination_result/expanded_eval/mongo_harvest.py

import asyncio
import contextlib
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dotenv import load_dotenv
load_dotenv()

_DIR = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(_DIR, "mongo_cases.jsonl")

_CACHED_SUFFIX = "Would you like to see new recommendations for this?"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _map_item(it: dict) -> dict:
    price = it.get("price")
    if isinstance(price, (int, float)):
        price = f"£{price:.2f}"
    return {
        "article_id":           str(it.get("article_id", "")),
        "name":                 it.get("prod_name") or "",
        "type":                 it.get("product_type_name") or "",
        "colour":               it.get("colour_group_name") or "",
        "price":                price,
        "pattern":              it.get("graphical_appearance_name") or None,
        "index_group":          it.get("index_group_name") or None,
        "section":              it.get("section_name") or None,
        "material_description": it.get("detail_desc") or None,
    }


async def harvest() -> list[dict]:
    from memory.db.mongo import (
        connect_to_mongodb, close_mongodb_connection, get_db,
        get_collection_name,
    )
    await connect_to_mongodb()
    db = get_db()
    recs_coll = get_collection_name("recommendations", "m3")

    # recommendation_id → items
    rec_items: dict[str, list] = {}
    async for rec in db[recs_coll].find({}, {"recommendation_id": 1, "items": 1}):
        rec_items[rec.get("recommendation_id", "")] = rec.get("items", [])

    rows, seen = [], set()
    n_bot, n_joined, n_filtered = 0, 0, 0

    async for sess in db.sessions.find({}, {"turns": 1, "session_id": 1}):
        turns = sess.get("turns", [])
        last_user_msg = ""
        for t in turns:
            role = t.get("role")
            if role == "user":
                last_user_msg = t.get("content", "")
                continue
            if role not in ("bot", "assistant"):
                continue
            n_bot += 1
            rec_id = t.get("recommendation_id")
            if not rec_id or rec_id not in rec_items:
                continue
            items = [_map_item(i) for i in rec_items[rec_id] if i]
            if not items:
                continue
            n_joined += 1

            response = (t.get("content") or "").strip()
            if response.endswith(_CACHED_SUFFIX):
                response = response[: -len(_CACHED_SUFFIX)].strip()
            if len(response) < 30:
                continue

            # quality filter: response must actually reference the items
            resp_norm = _norm(response)
            references = ("£" in response) or any(
                _norm(i["name"]) in resp_norm for i in items if i["name"])
            if not references:
                n_filtered += 1
                continue

            # de-duplicate (also removes scripted-driver overlap)
            key = resp_norm[:400]
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "captured_at":   str(t.get("timestamp", "")),
                "session_id":    sess.get("session_id", ""),
                "user_message":  last_user_msg,
                "action":        "catalog_search",
                "attempt":       1,
                "evidence":      {"action": "catalog_search", "items": items},
                "response_text": response,
                "source":        "mongodb",
                "checker":       {},   # filled offline below
            })

    await close_mongodb_connection()
    print(f"bot turns seen: {n_bot} · joined with items: {n_joined} · "
          f"dropped (no item reference): {n_filtered} · unique rows: {len(rows)}")
    return rows


def classify_offline(rows: list[dict]) -> None:
    """Runs checker v3 on each harvested pair to split presumed-clean vs
    flagged — the same convention live capture uses."""
    from text_rag.core.hallucination_checker import HallucinationChecker
    checker = HallucinationChecker()
    t0 = time.time()
    for i, row in enumerate(rows):
        with contextlib.redirect_stdout(io.StringIO()):
            result = checker.check(row["response_text"], row["evidence"])
        row["checker"] = {
            "passed":              result["passed"],
            "n_checked":           result["n_checked"],
            "n_flagged":           result["n_flagged"],
            "hallucination_score": result["hallucination_score"],
            "contradicted_fields": result["contradicted_fields"],
        }
        if (i + 1) % 50 == 0:
            print(f"  checker {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")


def main():
    rows = asyncio.run(harvest())
    print("Running checker v3 offline over harvested pairs...")
    classify_offline(rows)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    n_pass = sum(1 for r in rows if r["checker"].get("passed"))
    print(f"\nWritten {os.path.basename(OUT)}: {len(rows)} rows "
          f"({n_pass} checker-passed / {len(rows) - n_pass} flagged)")


if __name__ == "__main__":
    main()
