"""
FastAPI dependencies that turn a cookie into a verified identity.

This module is the only place a request's identity is established. Handlers
must take `user_id` from here and never from the request body or query string —
a client-supplied user id is not a credential, and trusting one was the original
vulnerability this work replaces.
"""
from fastapi import Depends, HTTPException, Request, status

from .config import ACCESS_COOKIE_NAME, CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from .models import AccountDocument, get_account_by_id
from .tokens import TokenError, csrf_tokens_match, decode_access_token

# Signals to the client that refreshing is worth attempting. The frontend
# interceptor keys off this to call /api/auth/refresh exactly once.
_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Cookie"},
)


async def get_current_account(request: Request) -> AccountDocument:
    """
    Resolves the signed-in account, or raises 401.

    Every failure returns the same generic 401: telling a caller whether a token
    was absent, malformed, or expired-but-valid hands them information about the
    system they do not need.
    """
    token = request.cookies.get(ACCESS_COOKIE_NAME)
    if not token:
        raise _UNAUTHENTICATED

    try:
        claims = decode_access_token(token)
    except TokenError:
        raise _UNAUTHENTICATED

    account = await get_account_by_id(claims.get("sub", ""))
    if account is None:
        # Token verified but the account is gone — a deleted account must not
        # keep working until its token happens to expire.
        raise _UNAUTHENTICATED

    if account.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is temporarily locked.",
        )
    return account


async def get_current_user_id(account: AccountDocument = Depends(get_current_account)) -> str:
    """
    The persona id for the signed-in account.

    A self-registered account has no linked persona yet, so its own account_id
    doubles as the user_id — that keeps every downstream memory and session
    query keyed on a single stable value.
    """
    return account.user_id or account.account_id


def require_role(*roles: str):
    """
    Dependency factory gating an endpoint on a role.

    Usage:  @router.get("/admin", dependencies=[Depends(require_role("admin"))])
    """
    async def _check(account: AccountDocument = Depends(get_current_account)) -> AccountDocument:
        if not any(r in account.roles for r in roles):
            # 403, not 404: the caller is authenticated, they simply lack the
            # permission. Hiding that would make the API harder to reason about
            # without making it meaningfully safer.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to do that.",
            )
        return account
    return _check


async def verify_csrf(request: Request) -> None:
    """
    Double-submit CSRF check for state-changing requests.

    SameSite=Lax already blocks most cross-site POSTs. This is the second layer,
    because SameSite is a browser-side control and not every client honours it
    identically. Safe methods are exempt — they must not change state anyway.
    """
    if request.method in ("GET", "HEAD", "OPTIONS", "TRACE"):
        return

    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_HEADER_NAME)

    if not csrf_tokens_match(cookie_token, header_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing or invalid.",
        )
