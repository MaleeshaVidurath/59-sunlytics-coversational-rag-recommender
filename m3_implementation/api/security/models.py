"""
Account model and its MongoDB access layer.

Deliberately a separate collection from `users`, and a separate module from
memory/models/schemas.py:

  accounts — who is signing in (credentials, roles, lockout state)
  users    — the shopper persona the recommender reasons about (purchase
             history, preferences, style profile)

Keeping them apart means a credential never sits next to behavioural data, and
one account can point at a persona without owning its lifecycle.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from memory.db.mongo import get_db

from .config import LOGIN_LOCKOUT_MINUTES, LOGIN_MAX_ATTEMPTS


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_account_id() -> str:
    import secrets
    return f"acct_{secrets.token_hex(8)}"


class AccountDocument(BaseModel):
    account_id: str = Field(default_factory=_new_account_id)

    # Stored lowercase; the login lookup lowercases too, so usernames are
    # case-insensitive and "Admin" cannot be registered alongside "admin".
    username: str
    password_hash: str

    roles: list[str] = Field(default_factory=lambda: ["user"])

    # The shopper persona this account drives. None for a self-registered
    # account, which then starts cold with no purchase history.
    user_id: Optional[str] = None
    linked_customer_id: Optional[str] = None

    # Brute-force state
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

    created_at: datetime = Field(default_factory=_now)
    last_login_at: Optional[datetime] = None

    # True for the 250 seeded research personas — lets the UI and any cleanup
    # script tell them apart from real registrations.
    is_seeded: bool = False

    class Config:
        extra = "allow"

    @field_validator("locked_until", "created_at", "last_login_at", mode="after")
    @classmethod
    def _assume_utc(cls, v):
        """
        MongoDB stores BSON dates without a zone and returns them naive, so a
        value read back would raise TypeError when compared against an aware
        "now". Everything written here is UTC, so tagging it on read is correct
        rather than a guess.
        """
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def is_locked(self) -> bool:
        return bool(self.locked_until and self.locked_until > _now())

    def public_dict(self) -> dict:
        """
        The only shape an account is ever allowed to leave the server in.

        Building this explicitly rather than excluding fields means a column
        added later is private by default instead of accidentally exposed.
        """
        return {
            "account_id": self.account_id,
            "username":   self.username,
            "roles":      self.roles,
            "user_id":    self.user_id,
            "is_seeded":  self.is_seeded,
        }


# ── Collection access ─────────────────────────────────────────────────────────

def _accounts():
    return get_db().accounts


def _refresh_tokens():
    return get_db().refresh_tokens


async def ensure_indexes() -> None:
    """
    Unique username, and a TTL index that lets MongoDB delete expired refresh
    tokens on its own rather than growing the collection forever.
    """
    await _accounts().create_index("username", unique=True)
    await _accounts().create_index("account_id", unique=True)
    await _refresh_tokens().create_index("token_hash", unique=True)
    await _refresh_tokens().create_index("family_id")
    await _refresh_tokens().create_index("expires_at", expireAfterSeconds=0)


def _strip_id(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


async def get_account_by_username(username: str) -> Optional[AccountDocument]:
    doc = await _accounts().find_one({"username": (username or "").strip().lower()})
    return AccountDocument.model_validate(_strip_id(doc)) if doc else None


async def get_account_by_id(account_id: str) -> Optional[AccountDocument]:
    doc = await _accounts().find_one({"account_id": account_id})
    return AccountDocument.model_validate(_strip_id(doc)) if doc else None


async def create_account(account: AccountDocument) -> AccountDocument:
    account.username = account.username.strip().lower()
    await _accounts().insert_one(account.model_dump(mode="json"))
    return account


async def record_failed_login(username: str) -> None:
    """
    Counts a failed attempt and locks the account once the threshold is hit.

    Runs for real accounts only — there is nothing to lock for a username that
    does not exist, and creating a record would leak which usernames are taken.
    """
    username = (username or "").strip().lower()
    doc = await _accounts().find_one_and_update(
        {"username": username},
        {"$inc": {"failed_attempts": 1}},
        return_document=True,
    )
    if doc and doc.get("failed_attempts", 0) >= LOGIN_MAX_ATTEMPTS:
        await _accounts().update_one(
            {"username": username},
            {"$set": {"locked_until": _now() + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)}},
        )


async def record_successful_login(account_id: str, *, new_hash: str | None = None) -> None:
    """Clears brute-force state and, if the hash was outdated, upgrades it."""
    update = {
        "$set": {"last_login_at": _now(), "failed_attempts": 0, "locked_until": None},
    }
    if new_hash:
        update["$set"]["password_hash"] = new_hash
    await _accounts().update_one({"account_id": account_id}, update)


# ── Refresh token store ───────────────────────────────────────────────────────

async def store_refresh_token(*, token_hash: str, account_id: str, family_id: str,
                              expires_at: datetime) -> None:
    await _refresh_tokens().insert_one({
        "token_hash": token_hash,
        "account_id": account_id,
        "family_id":  family_id,
        "expires_at": expires_at,
        "created_at": _now(),
        "used_at":    None,
        "revoked":    False,
    })


async def find_refresh_token(token_hash: str) -> Optional[dict]:
    doc = await _refresh_tokens().find_one({"token_hash": token_hash})
    return _strip_id(doc) if doc else None


async def mark_refresh_token_used(token_hash: str) -> None:
    await _refresh_tokens().update_one(
        {"token_hash": token_hash}, {"$set": {"used_at": _now()}},
    )


async def revoke_family(family_id: str) -> int:
    """
    Kills every token in a chain.

    Called both on sign-out and on reuse detection: if a token that was already
    rotated away comes back, either it was stolen or the real user is replaying
    an old one, and neither case should be allowed to continue.
    """
    res = await _refresh_tokens().update_many(
        {"family_id": family_id, "revoked": False}, {"$set": {"revoked": True}},
    )
    return res.modified_count


async def revoke_all_for_account(account_id: str) -> int:
    res = await _refresh_tokens().update_many(
        {"account_id": account_id, "revoked": False}, {"$set": {"revoked": True}},
    )
    return res.modified_count
