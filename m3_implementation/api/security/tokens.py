"""
Token issue and verification.

Two different mechanisms, chosen for different jobs:

  Access token  — a signed JWT, 15 minutes, verified without touching the
                  database so it costs nothing on every request.
  Refresh token — an opaque 256-bit random value, 14 days, stored *hashed*
                  server-side. Opaque means it cannot be forged; server-side
                  means it can be revoked the instant something looks wrong.

Storing only a SHA-256 of the refresh token means a database leak does not hand
the attacker usable tokens. SHA-256 rather than Argon2 is correct here: the
value is already 256 bits of entropy, so there is nothing to brute-force, and
refresh happens often enough that a slow hash would hurt.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from .config import (
    ACCESS_TOKEN_TTL_MINUTES, JWT_ALGORITHM, JWT_AUDIENCE, JWT_ISSUER,
    JWT_SECRET, REFRESH_TOKEN_TTL_DAYS,
)


class TokenError(Exception):
    """Any failure to produce a trustworthy identity from a token."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Access token ──────────────────────────────────────────────────────────────

def create_access_token(*, account_id: str, user_id: str, username: str, roles: list[str]) -> str:
    """
    Mints a short-lived access JWT.

    Claims stay minimal: enough to authorise a request without a database read,
    and nothing sensitive. A JWT is signed, not encrypted — anyone holding it
    can read the payload.
    """
    now = _now()
    payload = {
        "sub":      account_id,
        "user_id":  user_id,
        "username": username,
        "roles":    roles,
        "iat":      now,
        "nbf":      now,
        "exp":      now + timedelta(minutes=ACCESS_TOKEN_TTL_MINUTES),
        "iss":      JWT_ISSUER,
        "aud":      JWT_AUDIENCE,
        "jti":      secrets.token_urlsafe(16),
        "typ":      "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Verifies an access token and returns its claims.

    `algorithms` is pinned to a single value on purpose: accepting a list the
    caller controls is how the classic "alg: none" and RS256->HS256 confusion
    attacks get in.
    """
    try:
        claims = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise TokenError("expired") from e
    except jwt.InvalidTokenError as e:
        raise TokenError("invalid") from e

    if claims.get("typ") != "access":
        # A refresh token must never be accepted as an access token.
        raise TokenError("wrong token type")
    return claims


# ── Refresh token ─────────────────────────────────────────────────────────────

def create_refresh_token() -> tuple[str, str, datetime]:
    """
    Returns (plaintext, sha256_hex, expires_at).

    The plaintext is handed to the browser once and never stored; only the hash
    is persisted.
    """
    plaintext = secrets.token_urlsafe(32)          # 256 bits
    return plaintext, hash_refresh_token(plaintext), _now() + timedelta(days=REFRESH_TOKEN_TTL_DAYS)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token_family() -> str:
    """
    Identifies one chain of rotated refresh tokens (one browser, one login).

    Reuse detection works at family granularity: replaying a retired token means
    someone holds a copy they should not, so the entire family is revoked rather
    than just that token.
    """
    return secrets.token_urlsafe(16)


# ── CSRF ──────────────────────────────────────────────────────────────────────

def create_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_tokens_match(cookie_value: str | None, header_value: str | None) -> bool:
    """Constant-time comparison; both halves must be present."""
    if not cookie_value or not header_value:
        return False
    return secrets.compare_digest(cookie_value, header_value)
