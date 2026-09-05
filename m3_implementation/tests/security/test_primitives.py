import sys, os, time
ROOT = "d:/L4S2/research_Final/ResearchProjectImplementation/59-sunlytics-coversational-rag-recommender/m3_implementation"
sys.path.insert(0, ROOT)
os.environ.setdefault("JWT_SECRET", "x" * 48)

fails = 0
def check(label, cond):
    global fails
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond: fails += 1

from api.security.passwords import (
    hash_password, verify_password, validate_password, needs_rehash, PasswordPolicyError,
)
from api.security import tokens as T
from api.security.tokens import TokenError

print("── password hashing ──")
h = hash_password("correct horse battery staple")
check("hash is argon2id", h.startswith("$argon2id$"))
check("salt is embedded (two hashes differ)", hash_password("same") != hash_password("same"))
check("correct password verifies", verify_password("correct horse battery staple", h) is True)
check("wrong password rejected", verify_password("wrong", h) is False)
check("plaintext absent from hash", "correct horse" not in h)
check("no rehash needed at current params", needs_rehash(h) is False)

print("\n── user enumeration defence ──")
check("verify against None returns False", verify_password("anything", None) is False)
t0 = time.perf_counter(); verify_password("x" * 20, h);    real = time.perf_counter() - t0
t0 = time.perf_counter(); verify_password("x" * 20, None); dummy = time.perf_counter() - t0
ratio = max(real, dummy) / max(min(real, dummy), 1e-9)
check(f"unknown-user timing within 2x of real ({real*1000:.0f}ms vs {dummy*1000:.0f}ms)", ratio < 2.0)

print("\n── password policy ──")
def rejects(pw, **kw):
    try: validate_password(pw, **kw); return False
    except PasswordPolicyError: return True
check("rejects short",            rejects("short"))
check("rejects empty",            rejects("   "))
check("rejects common password",  rejects("password123"))
check("rejects username inside",  rejects("alice-supersecret", username="alice"))
check("rejects over 128 chars",   rejects("a" * 129))
check("accepts a 12+ passphrase", not rejects("purple-monkey-dishwasher"))

print("\n── access tokens ──")
tok = T.create_access_token(account_id="acct_1", user_id="u1", username="alice", roles=["user"])
claims = T.decode_access_token(tok)
check("round-trips subject",  claims["sub"] == "acct_1")
check("carries user_id",      claims["user_id"] == "u1")
check("carries roles",        claims["roles"] == ["user"])
check("typed as access",      claims["typ"] == "access")
check("has jti/iss/aud",      all(k in claims for k in ("jti", "iss", "aud")))

import jwt as pyjwt
from api.security.config import JWT_SECRET
def rejected(t):
    try: T.decode_access_token(t); return False
    except TokenError: return True

check("rejects tampered signature", rejected(tok[:-4] + "AAAA"))
check("rejects wrong secret",       rejected(pyjwt.encode({"sub":"x"}, "other-secret-value-here", algorithm="HS256")))
check("rejects alg=none",           rejected(pyjwt.encode({"sub":"x"}, key="", algorithm="none")))
from datetime import datetime, timedelta, timezone
past = datetime.now(timezone.utc) - timedelta(hours=1)
expired = pyjwt.encode({"sub":"x","iat":past,"exp":past,"iss":"sunlytics-m3","aud":"sunlytics-web","typ":"access"}, JWT_SECRET, algorithm="HS256")
check("rejects expired",            rejected(expired))
wrong_aud = pyjwt.encode({"sub":"x","iat":past,"exp":datetime.now(timezone.utc)+timedelta(hours=1),"iss":"sunlytics-m3","aud":"someone-else","typ":"access"}, JWT_SECRET, algorithm="HS256")
check("rejects wrong audience",     rejected(wrong_aud))
refresh_shaped = pyjwt.encode({"sub":"x","iat":past,"exp":datetime.now(timezone.utc)+timedelta(hours=1),"iss":"sunlytics-m3","aud":"sunlytics-web","typ":"refresh"}, JWT_SECRET, algorithm="HS256")
check("refresh type not accepted as access", rejected(refresh_shaped))

print("\n── refresh tokens ──")
plain, hashed, exp = T.create_refresh_token()
check("plaintext is 256-bit-ish",   len(plain) >= 40)
check("hash is sha256 hex",         len(hashed) == 64 and all(c in "0123456789abcdef" for c in hashed))
check("hash is deterministic",      T.hash_refresh_token(plain) == hashed)
check("plaintext not recoverable",  plain not in hashed)
check("two tokens differ",          T.create_refresh_token()[0] != plain)
# timedelta.days truncates, so compare seconds rather than whole days.
_ttl_h = (exp - datetime.now(timezone.utc)).total_seconds() / 3600
check(f"expiry ~14 days out ({_ttl_h:.1f}h)", 13.9 * 24 < _ttl_h <= 14 * 24)

print("\n── csrf ──")
c = T.create_csrf_token()
check("matches itself",       T.csrf_tokens_match(c, c) is True)
check("rejects mismatch",     T.csrf_tokens_match(c, T.create_csrf_token()) is False)
check("rejects missing half", T.csrf_tokens_match(c, None) is False and T.csrf_tokens_match(None, c) is False)

print("\n── cookie flags ──")
from api.security.config import cookie_kwargs
k = cookie_kwargs(900)
check("httpOnly always on", k["httponly"] is True)
check("samesite set",       k["samesite"] == "lax")
check("secure set",         k["secure"] is True)

print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
