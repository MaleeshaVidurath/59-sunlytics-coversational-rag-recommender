"""
Cookie helpers.

Auth cookies are set in exactly one place so no handler can accidentally issue
one without httpOnly, or forget to clear a matching cookie on sign-out.
"""
from fastapi import Response

from .config import (
    ACCESS_COOKIE_NAME, ACCESS_TOKEN_TTL_MINUTES, COOKIE_DOMAIN, COOKIE_SAMESITE,
    COOKIE_SECURE, CSRF_COOKIE_NAME, REFRESH_COOKIE_NAME, REFRESH_COOKIE_PATH,
    REFRESH_TOKEN_TTL_DAYS, cookie_kwargs,
)

_ACCESS_MAX_AGE  = ACCESS_TOKEN_TTL_MINUTES * 60
_REFRESH_MAX_AGE = REFRESH_TOKEN_TTL_DAYS * 24 * 60 * 60


def set_auth_cookies(response: Response, *, access_token: str,
                     refresh_token: str, csrf_token: str) -> None:
    response.set_cookie(ACCESS_COOKIE_NAME, access_token,
                        **cookie_kwargs(_ACCESS_MAX_AGE))

    # Scoped to the refresh endpoint, so it is not attached to ordinary requests.
    response.set_cookie(REFRESH_COOKIE_NAME, refresh_token,
                        **cookie_kwargs(_REFRESH_MAX_AGE, path=REFRESH_COOKIE_PATH))

    # The CSRF cookie is the one cookie the frontend MUST be able to read, so it
    # can echo the value back in a header. That is the double-submit pattern:
    # readable is fine because an attacker on another origin still cannot read
    # it, and a forged request cannot set the matching header.
    csrf_kwargs = cookie_kwargs(_REFRESH_MAX_AGE)
    csrf_kwargs["httponly"] = False
    response.set_cookie(CSRF_COOKIE_NAME, csrf_token, **csrf_kwargs)


def clear_auth_cookies(response: Response) -> None:
    """
    Deletes every auth cookie.

    Path and domain must match what set_cookie used or the browser keeps the
    old cookie and the user stays signed in.
    """
    common = {"secure": COOKIE_SECURE, "samesite": COOKIE_SAMESITE}
    if COOKIE_DOMAIN:
        common["domain"] = COOKIE_DOMAIN

    response.delete_cookie(ACCESS_COOKIE_NAME, path="/", httponly=True, **common)
    response.delete_cookie(REFRESH_COOKIE_NAME, path=REFRESH_COOKIE_PATH, httponly=True, **common)
    response.delete_cookie(CSRF_COOKIE_NAME, path="/", httponly=False, **common)
