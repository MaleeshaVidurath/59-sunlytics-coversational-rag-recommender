# m3_implementation/demo_revision.py
#
# Live demo helper for catalogue-revision detection.
#
# Changing a product in PostgreSQL mid-conversation is what triggers the
# cross-turn consistency layer's headline behaviour: the reply is brought up to
# date and every earlier message that quoted the old value gets an in-chat
# correction note. Doing that by hand means digging an article_id out of the
# logs and writing SQL, which is not something to be doing in front of an
# audience. This script removes both steps — products are addressed by the same
# NAME the chat shows, so it can be copied straight out of the conversation.
#
#   python demo_revision.py
#       Lists the newest chat session's products with their current values.
#
#   python demo_revision.py "Chablis skirt"
#       Shows one product without changing anything.
#
#   python demo_revision.py "Chablis skirt" colour=Red
#       Changes one field.
#
#   python demo_revision.py "Chablis skirt" colour=Red price=29.99 type=Dress
#       Changes several at once — each becomes its own correction note, which
#       is the clearest way to show the per-(product, attribute) grouping.
#
#   python demo_revision.py --revert
#       Puts every value this script changed back.
#
# An article_id works anywhere a name does.
#
# FIELDS — colour | price | type | name
#   These four are exactly what the assertion ledger tracks, so they are the
#   only ones a revision notice can be raised for. Editing anything else
#   (detail_desc, section, …) changes the database but produces no notice.

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

from memory.db.mongo import (
    connect_to_mongodb, close_mongodb_connection, get_db, get_collection_name,
)
from memory.core.assertion_ledger import AssertionLedger
from text_rag.db.postgres_client import get_pool, close_pool

BACKUP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      ".demo_revision_backup.json")

# friendly name → (column, numeric?)
FIELDS = {
    "colour": ("colour_group_name", False),
    "price":  ("avg_price",         True),
    "type":   ("product_type_name", False),
    "name":   ("prod_name",         False),
}

SELECT = ("SELECT article_id, prod_name, product_type_name, "
          "colour_group_name, avg_price FROM articles WHERE article_id = $1")


def _fmt(row, field) -> str:
    column, numeric = FIELDS[field]
    value = row[column]
    return f"£{float(value):.2f}" if numeric else str(value)


# ── backup, so --revert always works ─────────────────────────────────────────

def _load_backup() -> dict:
    try:
        with open(BACKUP, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_backup(data: dict) -> None:
    with open(BACKUP, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── the newest session's memory ──────────────────────────────────────────────

async def _session_products() -> tuple:
    """
    (session_id, [article_ids]) for the most recently updated chat.
    Returns ("", []) if MongoDB is unreachable or no session has a ledger —
    name lookup then simply falls back to the whole catalogue.
    """
    try:
        db = get_db()
    except Exception:
        return "", []

    newest = None
    for prefix in ("m3", "m2", "m1"):
        try:
            coll = get_collection_name("session_graphs", prefix)
            doc = await db[coll].find_one(sort=[("updated_at", -1)])
        except Exception:
            doc = None
        if doc and (newest is None
                    or doc.get("updated_at", "") > newest[0].get("updated_at", "")):
            newest = (doc, prefix)

    if not newest:
        return "", []

    doc, prefix = newest
    session_id = doc.get("session_id", "")
    ledger = await AssertionLedger.load(session_id, prefix)
    return session_id, [p["article_id"] for p in ledger.known_products()]


# ── resolving a product ──────────────────────────────────────────────────────

async def _resolve(target: str, session_ids: list):
    """
    Turns a product name (or an article_id) into a database row.

    Names are not unique in the catalogue — the same garment exists in several
    colours under one name — so when a name matches more than one article the
    ones the current conversation actually knows about are preferred. That is
    almost always the intent: you copied the name out of the chat. If it is
    still ambiguous the candidates are listed rather than one being guessed.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if target.strip().isdigit():
            row = await conn.fetchrow(SELECT, int(target.strip()))
            if not row:
                print(f"No article with id {target}")
            return row

        rows = await conn.fetch(
            "SELECT article_id, prod_name, product_type_name, "
            "colour_group_name, avg_price FROM articles "
            "WHERE lower(prod_name) = lower($1) ORDER BY article_id",
            target.strip(),
        )

    if not rows:
        print(f'No product named "{target}".')
        print("Run with no arguments to list the products this chat knows.")
        return None

    if len(rows) == 1:
        return rows[0]

    known = [r for r in rows if str(r["article_id"]) in set(session_ids)]
    if len(known) == 1:
        return known[0]

    candidates = known or rows
    print(f'"{target}" matches {len(candidates)} articles — '
          f"use the article_id instead:\n")
    for r in candidates:
        mark = "  ← in this chat" if str(r["article_id"]) in set(session_ids) else ""
        print(f"  {r['article_id']:<12} {r['colour_group_name']:<14} "
              f"£{float(r['avg_price']):>8.2f}{mark}")
    return None


# ── commands ─────────────────────────────────────────────────────────────────

async def cmd_list() -> None:
    session_id, ids = await _session_products()
    if not session_id:
        print("No chat session has a memory graph yet — send a message in the "
              "app first.")
        return

    numeric = [a for a in ids if str(a).isdigit()]
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = [await conn.fetchrow(SELECT, int(a)) for a in numeric]
    rows = [r for r in rows if r]

    print(f"\nNewest session: {session_id}  —  {len(rows)} product(s) in memory\n")
    if not rows:
        print("None of them are in the articles table.")
        return

    print(f"  {'name':<26} {'type':<14} {'colour':<14} {'price':>9}")
    print(f"  {'-'*26} {'-'*14} {'-'*14} {'-'*9}")
    for r in rows:
        print(f"  {(r['prod_name'] or '')[:26]:<26} "
              f"{(r['product_type_name'] or '')[:14]:<14} "
              f"{(r['colour_group_name'] or '')[:14]:<14} "
              f"£{float(r['avg_price']):>8.2f}")

    pick = rows[0]["prod_name"]
    print(f"\nChange one, then ask about it in the SAME chat:\n")
    print(f'  python demo_revision.py "{pick}" colour=Red')
    print(f'  python demo_revision.py "{pick}" colour=Red price=29.99')
    print(f'\n  then type:  "tell me more about {pick}"')
    print("\nUndo everything with:  python demo_revision.py --revert")


async def cmd_show(row) -> None:
    print(f"\n  {row['prod_name']}   (article_id {row['article_id']})")
    for field in FIELDS:
        print(f"    {field:<7}: {_fmt(row, field)}")
    print(f'\n  python demo_revision.py "{row["prod_name"]}" colour=Red price=29.99')


async def cmd_set(row, changes: dict) -> None:
    article_id = int(row["article_id"])
    original_name = row["prod_name"]

    backup = _load_backup()
    entry = backup.setdefault(str(article_id), {})

    pool = await get_pool()
    async with pool.acquire() as conn:
        for field, new_value in changes.items():
            column, numeric = FIELDS[field]
            # Record the FIRST original only, so repeated changes during a demo
            # still revert to the true starting value.
            entry.setdefault(field,
                             float(row[column]) if numeric else str(row[column]))
            await conn.execute(
                f"UPDATE articles SET {column} = $1 WHERE article_id = $2",
                float(new_value) if numeric else str(new_value), article_id,
            )
        after = await conn.fetchrow(SELECT, article_id)

    _save_backup(backup)

    print(f"\n  {original_name}   (article_id {article_id})")
    for field in changes:
        print(f"    {field:<7}: {_fmt(row, field)}  →  {_fmt(after, field)}")

    if "name" in changes:
        print(f"\n  NOTE: the name changed. The chat still knows this product as "
              f'"{original_name}" —\n  ask using that name, not the new one.')

    print(f'\nNow ask in the SAME chat:  "tell me more about {original_name}"')
    print("Then look for:")
    print(f"  [LEDGER] REVISION {article_id}.<field>: 'old' → 'new'")
    print("  and the amber note under the earlier message that quoted the old "
          "value.")


async def cmd_revert() -> None:
    backup = _load_backup()
    if not backup:
        print("Nothing to revert — this script has not changed anything.")
        return

    pool = await get_pool()
    async with pool.acquire() as conn:
        for article_id, fields in backup.items():
            for field, original in fields.items():
                column, numeric = FIELDS[field]
                await conn.execute(
                    f"UPDATE articles SET {column} = $1 WHERE article_id = $2",
                    float(original) if numeric else str(original),
                    int(article_id),
                )
                shown = f"£{float(original):.2f}" if numeric else original
                print(f"  restored {article_id}.{field} → {shown}")

    os.remove(BACKUP)
    print("\nAll values restored.")


# ── argument parsing ─────────────────────────────────────────────────────────

def _parse_changes(args: list):
    """
    Accepts `field=value` pairs, and also the two-word form `field value` so the
    shorter command shown in earlier instructions keeps working.
    Returns (changes, error_message).
    """
    if not args:
        return {}, None

    if len(args) == 2 and "=" not in args[0]:
        args = [f"{args[0]}={args[1]}"]

    changes = {}
    for arg in args:
        if "=" not in arg:
            return {}, (f"expected field=value, got {arg!r}\n"
                        f"fields: {', '.join(FIELDS)}")
        field, value = arg.split("=", 1)
        field = field.strip().lower()
        if field not in FIELDS:
            return {}, (f"unknown field {field!r}\n"
                        f"fields: {', '.join(FIELDS)}")
        if not value.strip():
            return {}, f"no value given for {field!r}"
        changes[field] = value.strip()
    return changes, None


async def main(argv: list) -> None:
    mongo_up = False
    try:
        await connect_to_mongodb()
        mongo_up = True
    except Exception as e:
        print(f"(MongoDB unavailable — name lookup will not prefer this "
              f"chat's products: {e})")

    try:
        if argv and argv[0] == "--revert":
            await cmd_revert()
            return
        if not argv:
            await cmd_list()
            return

        changes, error = _parse_changes(argv[1:])
        if error:
            print(error)
            return

        _, session_ids = await _session_products()
        row = await _resolve(argv[0], session_ids)
        if row is None:
            return

        if changes:
            await cmd_set(row, changes)
        else:
            await cmd_show(row)
    finally:
        # Both pools must close inside the loop that opened them.
        await close_pool()
        if mongo_up:
            await close_mongodb_connection()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
