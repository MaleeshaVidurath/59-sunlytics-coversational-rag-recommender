"""
Authentication configuration.

Every secret is read from the environment. Nothing here has a usable default:
a missing JWT secret raises at import time rather than silently falling back to
a shared constant that would let anyone forge a token.
"""
import os
import secrets

from dotenv import load_dotenv

load_dotenv()


def _require_secret(name: str, min_bytes: int = 32) -> str:
    """
    Reads a secret from the environment, refusing weak or absent values.

    In development a missing secret is generated per-process so the app still
    starts, but that invalidates every existing token on restart — which is the
    correct, noisy failure mode. In production it is a hard error.
    """
    value = os.getenv(name, "")
    if value and len(value) >= min_bytes:
        return value

    if IS_PRODUCTION:
        raise RuntimeError(
            f"{name} is missing or shorter than {min_bytes} characters. "
            f"Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )

    generated = secrets.token_urlsafe(48)
    print(
        f"[AUTH] WARNING: {name} not set — generated an ephemeral one. "
        f"All sessions will be invalidated on restart. Add {name} to .env."
    )
    return generated


ENVIRONMENT   = os.getenv("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"

# ── Tokens ────────────────────────────────────────────────────────────────────
JWT_SECRET    = _require_secret("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_ISSUER    = "sunlytics-m3"
JWT_AUDIENCE  = "sunlytics-web"

# Short access lifetime bounds the damage from a stolen access token; the
# refresh token is what keeps the user signed in.
ACCESS_TOKEN_TTL_MINUTES = int(os.getenv("ACCESS_TOKEN_TTL_MINUTES", "15"))
REFRESH_TOKEN_TTL_DAYS   = int(os.getenv("REFRESH_TOKEN_TTL_DAYS", "14"))

# ── Cookies ───────────────────────────────────────────────────────────────────
ACCESS_COOKIE_NAME  = "sunlytics_access"
REFRESH_COOKIE_NAME = "sunlytics_refresh"
CSRF_COOKIE_NAME    = "sunlytics_csrf"
CSRF_HEADER_NAME    = "X-CSRF-Token"

# The refresh cookie is scoped to its own endpoint, so it is not attached to
# every ordinary request and cannot leak through unrelated handlers.
REFRESH_COOKIE_PATH = "/api/auth/refresh"

# Ports are not part of a "site", so localhost:5173 -> localhost:8000 is
# same-site and Lax works in development. A cross-domain deployment needs
# SameSite=None, which browsers only accept together with Secure.
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax").lower()
COOKIE_SECURE   = os.getenv("COOKIE_SECURE", "true").lower() == "true"
COOKIE_DOMAIN   = os.getenv("COOKIE_DOMAIN") or None

# ── Password policy (NIST SP 800-63B) ────────────────────────────────────────
# Length is the control that matters. Composition rules ("must contain a
# symbol") push users toward predictable substitutions, so they are not applied.
PASSWORD_MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "12"))
PASSWORD_MAX_LENGTH = 128   # bounds the work an attacker can force us to hash

# ── Brute-force controls ──────────────────────────────────────────────────────
LOGIN_MAX_ATTEMPTS      = int(os.getenv("LOGIN_MAX_ATTEMPTS", "8"))
LOGIN_LOCKOUT_MINUTES   = int(os.getenv("LOGIN_LOCKOUT_MINUTES", "15"))
LOGIN_RATE_LIMIT_PER_IP = int(os.getenv("LOGIN_RATE_LIMIT_PER_IP", "20"))
LOGIN_RATE_WINDOW_SEC   = int(os.getenv("LOGIN_RATE_WINDOW_SEC", "300"))
REGISTER_RATE_LIMIT     = int(os.getenv("REGISTER_RATE_LIMIT", "5"))
REGISTER_RATE_WINDOW_SEC = int(os.getenv("REGISTER_RATE_WINDOW_SEC", "3600"))

# Only enable behind a reverse proxy that overwrites X-Forwarded-For. A client
# can set the header itself, so trusting it without a proxy in front hands
# attackers a trivial bypass for every IP-based limit.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"


def cookie_kwargs(max_age: int, path: str = "/") -> dict:
    """Shared flags for every auth cookie. httpOnly is never optional here."""
    kwargs = {
        "httponly": True,
        "secure":   COOKIE_SECURE,
        "samesite": COOKIE_SAMESITE,
        "max_age":  max_age,
        "path":     path,
    }
    if COOKIE_DOMAIN:
        kwargs["domain"] = COOKIE_DOMAIN
    return kwargs
