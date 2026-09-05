"""
Authentication endpoints.

  POST /api/auth/register  create an account
  POST /api/auth/login     exchange credentials for cookies
  POST /api/auth/refresh   rotate the refresh token, mint a new access token
  POST /api/auth/logout    revoke the token family and clear cookies
  GET  /api/auth/me        who am I (used by the frontend to bootstrap)
  GET  /api/auth/customers persona catalogue — now requires authentication

Tokens are delivered exclusively as httpOnly cookies. No token is ever returned
in a response body, so a cross-site script cannot read one.
"""
import csv
import os
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from memory.core.user_manager import UserManager
from memory.db.mongo import get_db

from ..security.config import (
    LOGIN_RATE_LIMIT_PER_IP, LOGIN_RATE_WINDOW_SEC,
    REFRESH_COOKIE_NAME, REGISTER_RATE_LIMIT, REGISTER_RATE_WINDOW_SEC,
)
from ..security.cookies import clear_auth_cookies, set_auth_cookies
from ..security.dependencies import get_current_account, verify_csrf
from ..security.models import (
    AccountDocument, create_account, find_refresh_token, get_account_by_id,
    get_account_by_username, mark_refresh_token_used, record_failed_login,
    record_successful_login, revoke_family, store_refresh_token,
)
from ..security.passwords import (
    PasswordPolicyError, hash_password, needs_rehash, validate_password, verify_password,
)
from ..security.rate_limit import (
    check_rate_limit, client_ip, ip_key, login_key, reset_rate_limit,
)
from ..security.tokens import (
    create_access_token, create_csrf_token, create_refresh_token,
    hash_refresh_token, new_token_family,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

CUSTOMERS_CSV = os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'shared', 'main_data_set', 'sample_customers.csv'
)

USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")

# One message for every credential failure. Distinguishing "no such user" from
# "wrong password" tells an attacker which usernames are worth attacking.
_BAD_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid username or password.",
)

# Distinct from _BAD_CREDENTIALS: this one tells the client its session is over
# and re-authentication is required, rather than that a credential was wrong.
_UNAUTHENTICATED_REFRESH = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Session expired. Please sign in again.",
)


# ── customer catalogue ────────────────────────────────────────────────────────

def _load_customers():
    customers = []
    try:
        with open(CUSTOMERS_CSV, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                customers.append({
                    "customer_id":          row['customer_id'],
                    "short_id":             row['customer_id'][:12] + "...",
                    "club_member_status":   row.get('club_member_status', ''),
                    "fashion_news_frequency": row.get('fashion_news_frequency', ''),
                    "age":                  row.get('age', ''),
                    "active":               row.get('Active', '') == '1.0',
                })
    except Exception as e:
        print(f"[Auth] Could not load customers CSV: {e}")
    return customers


_customers_cache = None


def get_customers_list():
    global _customers_cache
    if _customers_cache is None:
        _customers_cache = _load_customers()
    return _customers_cache


# ── request models ────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


# ── helpers ───────────────────────────────────────────────────────────────────

async def _issue_session(response: Response, account: AccountDocument,
                         *, family_id: str | None = None) -> dict:
    """
    Mints a token pair and attaches the cookies.

    A fresh login starts a new token family; a refresh continues the existing
    one so reuse detection can still see the whole chain.
    """
    access = create_access_token(
        account_id=account.account_id,
        user_id=account.user_id or account.account_id,
        username=account.username,
        roles=account.roles,
    )
    refresh_plain, refresh_hash, expires_at = create_refresh_token()
    family = family_id or new_token_family()

    await store_refresh_token(
        token_hash=refresh_hash, account_id=account.account_id,
        family_id=family, expires_at=expires_at,
    )

    set_auth_cookies(
        response,
        access_token=access,
        refresh_token=refresh_plain,
        csrf_token=create_csrf_token(),
    )
    return account.public_dict()


async def _profile_for(account: AccountDocument) -> dict:
    """
    Account plus the persona details the chat UI shows.

    A self-registered account has no persona, so the purchase summary is empty
    — a cold-start user the recommender ranks semantically until preferences
    build up through conversation.
    """
    payload = account.public_dict()
    payload["customer_id"] = account.linked_customer_id
    payload["purchase_summary"] = {}
    payload["age"] = None
    payload["club_member_status"] = ""

    if not account.linked_customer_id:
        return payload

    customer = next(
        (c for c in get_customers_list() if c["customer_id"] == account.linked_customer_id),
        None,
    )
    if customer:
        payload["age"] = customer.get("age") or None
        payload["club_member_status"] = customer.get("club_member_status", "")
        payload["fashion_news_frequency"] = customer.get("fashion_news_frequency", "")

    if account.user_id:
        doc = await get_db().users.find_one(
            {"user_id": account.user_id}, {"purchase_history": 1},
        )
        ph = (doc or {}).get("purchase_history", {}) or {}
        payload["purchase_summary"] = {
            "total_purchases": ph.get("total_purchases", 0),
            "dominant_colour": ph.get("dominant_colour", ""),
            "dominant_type":   ph.get("dominant_product_type", ""),
            "budget_tier":     ph.get("price_stats", {}).get("budget_tier", ""),
            "inferred_gender": ph.get("inferred_gender", ""),
        }
    return payload


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, request: Request, response: Response):
    """
    Creates a cold-start account: no linked persona, no purchase history.

    There is no email verification — that needs a mail provider this project
    does not have. Documented as a known gap rather than faked.
    """
    ip = client_ip(request)
    allowed, retry_after = await check_rate_limit(
        ip_key("register", ip),
        limit=REGISTER_RATE_LIMIT, window_seconds=REGISTER_RATE_WINDOW_SEC,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many accounts created from this address. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    username = req.username.strip().lower()
    if not USERNAME_PATTERN.match(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be 3-32 characters: letters, digits, dot, dash or underscore.",
        )

    try:
        validate_password(req.password, username=username)
    except PasswordPolicyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if await get_account_by_username(username) is not None:
        # Registration inevitably reveals that a username is taken — there is no
        # way around it without email confirmation. Rate limiting above is what
        # stops this being a usable enumeration oracle.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That username is already taken.",
        )

    account = AccountDocument(
        username=username,
        password_hash=hash_password(req.password),
        roles=["user"],
    )
    # A persona document keyed on the account gives the memory pipeline
    # somewhere to accumulate preferences from the very first turn.
    account.user_id = account.account_id
    await create_account(account)

    await _issue_session(response, account)
    return await _profile_for(account)


@router.post("/login")
async def login(req: LoginRequest, request: Request, response: Response):
    ip = client_ip(request)
    rl_key = login_key(ip, req.username)
    allowed, retry_after = await check_rate_limit(
        rl_key, limit=LOGIN_RATE_LIMIT_PER_IP, window_seconds=LOGIN_RATE_WINDOW_SEC,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many sign-in attempts. Please wait and try again.",
            headers={"Retry-After": str(retry_after)},
        )

    username = req.username.strip().lower()
    account = await get_account_by_username(username)

    # Runs even when the account is missing: verify_password hashes against a
    # dummy so the response takes the same time either way.
    password_ok = verify_password(req.password, account.password_hash if account else None)

    if account is None or not password_ok:
        if account is not None:
            await record_failed_login(username)
        raise _BAD_CREDENTIALS

    if account.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Too many failed attempts. This account is temporarily locked.",
        )

    # Correct password on an outdated hash — upgrade it while we have the
    # plaintext, which is the only moment it is possible.
    upgraded = hash_password(req.password) if needs_rehash(account.password_hash) else None
    await record_successful_login(account.account_id, new_hash=upgraded)
    await reset_rate_limit(rl_key)

    await _issue_session(response, account)
    return await _profile_for(account)


@router.post("/refresh")
async def refresh(request: Request, response: Response):
    """
    Rotates the refresh token and issues a new access token.

    Rotation with reuse detection: each refresh retires the presented token. If
    a retired token is presented again, someone holds a copy they should not, so
    the entire family is revoked and the user must sign in again.
    """
    presented = request.cookies.get(REFRESH_COOKIE_NAME)
    if not presented:
        raise _UNAUTHENTICATED_REFRESH

    record = await find_refresh_token(hash_refresh_token(presented))
    if record is None:
        clear_auth_cookies(response)
        raise _UNAUTHENTICATED_REFRESH

    if record.get("revoked"):
        clear_auth_cookies(response)
        raise _UNAUTHENTICATED_REFRESH

    if record.get("used_at") is not None:
        # Replay of an already-rotated token. Either it was stolen, or a stale
        # client is retrying; both warrant killing the chain.
        revoked = await revoke_family(record["family_id"])
        print(f"[AUTH] Refresh token reuse detected — revoked {revoked} token(s) "
              f"in family {record['family_id']}")
        clear_auth_cookies(response)
        raise _UNAUTHENTICATED_REFRESH

    expires_at = record.get("expires_at")
    if isinstance(expires_at, datetime):
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            clear_auth_cookies(response)
            raise _UNAUTHENTICATED_REFRESH

    account = await get_account_by_id(record["account_id"])
    if account is None or account.is_locked:
        clear_auth_cookies(response)
        raise _UNAUTHENTICATED_REFRESH

    await mark_refresh_token_used(record["token_hash"])
    await _issue_session(response, account, family_id=record["family_id"])
    return await _profile_for(account)


@router.post("/logout", dependencies=[Depends(verify_csrf)])
async def logout(request: Request, response: Response):
    """
    Signs out.

    Revokes server-side before clearing cookies, so a refresh token already
    copied off the machine stops working too. Always returns 200 — signing out
    should never fail in a way the user has to think about.
    """
    presented = request.cookies.get(REFRESH_COOKIE_NAME)
    if presented:
        record = await find_refresh_token(hash_refresh_token(presented))
        if record:
            await revoke_family(record["family_id"])

    clear_auth_cookies(response)
    return {"detail": "Signed out."}


@router.get("/me")
async def me(account: AccountDocument = Depends(get_current_account)):
    """Who the caller is. The frontend calls this on load to restore session."""
    return await _profile_for(account)


@router.get("/customers")
async def list_customers(account: AccountDocument = Depends(get_current_account)):
    """
    The persona catalogue.

    Now behind authentication: it exposes 250 records including ages, and
    combined with the seeded user_id scheme it previously allowed anyone to
    enumerate every account in the system.
    """
    customers = get_customers_list()
    return {"customers": customers, "total": len(customers)}

