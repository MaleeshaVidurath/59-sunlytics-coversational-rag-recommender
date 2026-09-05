"""
Redis-backed fixed-window rate limiting.

Guards the endpoints where an attacker gets unlimited free guesses: login and
register. Account lockout alone is not enough — it protects one account at a
time, while a password-spraying attack tries one common password against many
usernames and never trips a single account's counter.

Fixed window rather than sliding: it is one INCR plus one EXPIRE, and the worst
case (2x the limit across a window boundary) does not matter at these
thresholds.
"""
from memory.db.redis_client import get_redis


async def check_rate_limit(key: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
    """
    Counts a hit against `key`.

    Returns (allowed, retry_after_seconds). Fails **open** if Redis is
    unavailable: the alternative is locking every user out of the application
    because the cache is down, and the account lockout in models.py still
    applies as a second layer.
    """
    try:
        redis = get_redis()
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window_seconds)
        if count > limit:
            ttl = await redis.ttl(key)
            return False, max(ttl, 1)
        return True, 0
    except Exception as e:
        print(f"[RATE-LIMIT] Redis unavailable, allowing request: {e}")
        return True, 0


async def reset_rate_limit(key: str) -> None:
    """Clears a counter — called after a successful login so one bad typo
    followed by a correct password does not consume the budget."""
    try:
        await get_redis().delete(key)
    except Exception:
        pass


def login_key(ip: str, username: str) -> str:
    """
    Keyed on IP *and* username so the two attack shapes are both covered:
    many passwords against one account, and one password against many accounts.
    """
    return f"ratelimit:login:{ip}:{(username or '').strip().lower()}"


def ip_key(scope: str, ip: str) -> str:
    return f"ratelimit:{scope}:{ip}"


def client_ip(request) -> str:
    """
    Best-effort client address.

    X-Forwarded-For is only trusted when a proxy is expected, because a client
    can set it freely — trusting it unconditionally would let an attacker
    rotate the header and bypass every IP-based limit.
    """
    from .config import TRUST_PROXY_HEADERS

    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
