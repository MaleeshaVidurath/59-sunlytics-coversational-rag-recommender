"""
Seeds one login account per H&M research persona, plus an admin account.

    python m3_implementation/scripts/seed_accounts.py --force

Each of the 250 customers in sample_customers.csv gets an account (user001 …
user250) linked to the persona document that already holds that customer's
purchase history, so recommendations stay personalised after sign-in.

Passwords are generated here and written to credentials/seeded_accounts.csv.
That file is the ONLY plaintext copy — MongoDB stores Argon2id hashes, which
cannot be reversed. Lose the file and the only recovery is re-seeding.

The credentials directory is gitignored. Never commit it.
"""
import argparse
import asyncio
import csv
import os
import secrets
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory.db.mongo import connect_to_mongodb, close_mongodb_connection, get_db  # noqa: E402
from api.security.models import AccountDocument, ensure_indexes                    # noqa: E402
from api.security.passwords import hash_password                                   # noqa: E402

REPO_ROOT       = os.path.dirname(ROOT)
CUSTOMERS_CSV   = os.path.join(REPO_ROOT, "shared", "main_data_set", "sample_customers.csv")
CREDENTIALS_DIR = os.path.join(REPO_ROOT, "credentials")
CREDENTIALS_CSV = os.path.join(CREDENTIALS_DIR, "seeded_accounts.csv")
CREDENTIALS_MD  = os.path.join(CREDENTIALS_DIR, "README.md")

# Deliberately excludes 0/O, 1/l/I — these passwords get read off a screen and
# retyped, and an ambiguous glyph turns into a support problem, not security.
_ALPHABET = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_PASSWORD_LENGTH = 16   # ~92 bits of entropy over this alphabet

ADMIN_USERNAME = "admin"


def generate_password() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(_PASSWORD_LENGTH))


def load_customers() -> list[dict]:
    with open(CUSTOMERS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def wipe_seeded(db) -> tuple[int, int]:
    """
    Removes previously seeded accounts and their refresh tokens.

    Scoped to is_seeded=True so a real self-registered account is never touched.
    Refresh tokens go too: rotating a password must not leave a live session
    that outlives the credential it was issued against.
    """
    seeded_ids = [d["account_id"] async for d in
                  db.accounts.find({"is_seeded": True}, {"account_id": 1})]
    tokens = 0
    if seeded_ids:
        res = await db.refresh_tokens.delete_many({"account_id": {"$in": seeded_ids}})
        tokens = res.deleted_count
    res = await db.accounts.delete_many({"is_seeded": True})
    return res.deleted_count, tokens


async def main(force: bool) -> int:
    await connect_to_mongodb()
    db = get_db()
    await ensure_indexes()

    existing = await db.accounts.count_documents({"is_seeded": True})
    if existing and not force:
        print(f"\n{existing} seeded account(s) already exist.")
        print("Re-seeding ROTATES every password and invalidates existing sessions.")
        print("Re-run with --force if that is what you want.\n")
        await close_mongodb_connection()
        return 1

    if existing:
        accounts_removed, tokens_removed = await wipe_seeded(db)
        print(f"Removed {accounts_removed} seeded account(s) "
              f"and {tokens_removed} refresh token(s).")

    customers = load_customers()
    print(f"Loaded {len(customers)} customers from sample_customers.csv")

    rows, unlinked = [], 0

    for index, customer in enumerate(customers, start=1):
        customer_id = customer["customer_id"]

        # Look the persona up by customer_id rather than constructing the
        # user_id from a naming convention: the convention is an implementation
        # detail, and a mismatch here would silently cost the account its
        # purchase history.
        persona = await db.users.find_one({"customer_id": customer_id}, {"user_id": 1})
        user_id = persona.get("user_id") if persona else None
        if not user_id:
            unlinked += 1

        username = f"user{index:03d}"
        password = generate_password()

        account = AccountDocument(
            username=username,
            password_hash=hash_password(password),
            roles=["user"],
            user_id=user_id,
            linked_customer_id=customer_id,
            is_seeded=True,
        )
        await db.accounts.insert_one(account.model_dump(mode="json"))

        rows.append({
            "username":           username,
            "password":           password,
            "role":               "user",
            "customer_id":        customer_id,
            "user_id":            user_id or "",
            "age":                customer.get("age", ""),
            "club_member_status": customer.get("club_member_status", ""),
        })

        if index % 50 == 0:
            print(f"  seeded {index}/{len(customers)}…")

    # Admin account: gives require_role("admin") something real to gate on.
    admin_password = generate_password()
    admin = AccountDocument(
        username=ADMIN_USERNAME,
        password_hash=hash_password(admin_password),
        roles=["admin", "user"],
        is_seeded=True,
    )
    admin.user_id = admin.account_id
    await db.accounts.insert_one(admin.model_dump(mode="json"))
    rows.append({
        "username": ADMIN_USERNAME, "password": admin_password, "role": "admin",
        "customer_id": "", "user_id": admin.user_id, "age": "", "club_member_status": "",
    })

    write_credentials(rows)

    print(f"\nSeeded {len(rows)} accounts ({len(customers)} personas + 1 admin).")
    if unlinked:
        print(f"WARNING: {unlinked} account(s) had no matching persona document "
              f"and will run cold-start with no purchase history.")
    print(f"Credentials written to: {CREDENTIALS_CSV}")
    print("This is the only plaintext copy. It is gitignored — keep it that way.")

    await close_mongodb_connection()
    return 0


def write_credentials(rows: list[dict]) -> None:
    os.makedirs(CREDENTIALS_DIR, exist_ok=True)

    with open(CREDENTIALS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "username", "password", "role", "customer_id", "user_id",
            "age", "club_member_status",
        ])
        writer.writeheader()
        writer.writerows(rows)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(CREDENTIALS_MD, "w", encoding="utf-8") as f:
        f.write(f"""# Seeded account credentials

Generated {generated} by `m3_implementation/scripts/seed_accounts.py`.

**{len(rows)} accounts** — {len(rows) - 1} research personas plus one admin.

## Do not commit this directory

`credentials/` is in `.gitignore`. These are plaintext passwords for every
account in the system. A `git add -f`, a zipped copy of the project folder, or
a screen share of this file all leak the whole set at once.

## Using it

Open `seeded_accounts.csv` and pick a row. Sort or filter by `age` or
`club_member_status` to choose a persona with the shopping history you want,
then sign in with that `username` and `password`.

| Column | Meaning |
|---|---|
| `username` | `user001`–`user{len(rows) - 1:03d}`, or `admin` |
| `password` | 16 random characters; copy-paste rather than retype |
| `role` | `user`, or `admin` for the one admin account |
| `customer_id` | The H&M customer this persona represents |
| `user_id` | The persona document holding the purchase history |
| `age`, `club_member_status` | From the source dataset, to help you pick |

## These passwords cannot be recovered

MongoDB stores Argon2id hashes only. Nothing can turn a hash back into a
password. If this file is lost, re-run the seed script — which **rotates every
password** and signs out every existing session:

```bash
python m3_implementation/scripts/seed_accounts.py --force
```

Self-registered accounts are never touched by the script; it only removes and
recreates accounts marked `is_seeded`.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="rotate passwords for accounts that already exist")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.force)))
