"""Registration, sign-in, refresh rotation, CSRF, lockout and rate limiting,
exercised against the real auth router."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # m3_implementation/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from fakes import Client, install_fakes, new_loop      # noqa: E402

_db, _redis = install_fakes()
_loop = new_loop()

from fastapi import FastAPI
import api.security.models as sec_models
from api.routers import auth as auth_router
auth_router.get_db = lambda: _db

app = FastAPI()
app.include_router(auth_router.router)

client = Client(app, _loop)

fails = 0


def check(label, cond):
    global fails
    print("  " + ("ok   " if cond else "FAIL ") + label)
    if not cond:
        fails += 1


_loop.run_until_complete(sec_models.ensure_indexes())
GOOD = "correct-horse-battery-staple"

print("-- registration --")
r = client.post("/api/auth/register", json={"username": "alice", "password": GOOD})
check("creates account (201)", r.status_code == 201)
check("no token or password echoed in body",
      "token" not in r.text.lower() and "password" not in r.text.lower())
check("sets access cookie", "sunlytics_access" in r.cookies)
check("sets refresh cookie", "sunlytics_refresh" in r.cookies)
check("sets csrf cookie", "sunlytics_csrf" in r.cookies)

raw = r.headers.get_list("set-cookie")
acc = next(c for c in raw if c.startswith("sunlytics_access"))
ref = next(c for c in raw if c.startswith("sunlytics_refresh"))
csrf = next(c for c in raw if c.startswith("sunlytics_csrf"))
check("access cookie is httpOnly", "httponly" in acc.lower())
check("refresh cookie is httpOnly", "httponly" in ref.lower())
check("refresh cookie is path-scoped", "path=/api/auth/refresh" in ref.lower())
check("csrf cookie readable by JS", "httponly" not in csrf.lower())
check("cookies carry SameSite", "samesite=lax" in acc.lower())
check("cold start: no linked persona", r.json()["customer_id"] is None)
check("default role is user", r.json()["roles"] == ["user"])

print("\n-- registration validation --")
check("rejects short password",
      client.post("/api/auth/register", json={"username": "bob", "password": "short"}).status_code == 400)
check("rejects common password",
      client.post("/api/auth/register", json={"username": "bob", "password": "password123456"}).status_code == 400)
check("rejects malformed username",
      client.post("/api/auth/register", json={"username": "a b", "password": GOOD}).status_code == 400)
check("rejects duplicate username",
      client.post("/api/auth/register", json={"username": "alice", "password": GOOD}).status_code == 409)

print("\n-- /me and protected routes --")
check("authenticated /me works", client.get("/api/auth/me").status_code == 200)
check("username echoed back", client.get("/api/auth/me").json()["username"] == "alice")
bare = Client(app, _loop)
check("anonymous /me is 401", bare.get("/api/auth/me").status_code == 401)
check("anonymous /customers is 401", bare.get("/api/auth/customers").status_code == 401)
check("authenticated /customers works", client.get("/api/auth/customers").status_code == 200)

print("\n-- login --")
c2 = Client(app, _loop)
r_unknown = c2.post("/api/auth/login", json={"username": "nobody", "password": "wrong-one-here"})
r_wrong = c2.post("/api/auth/login", json={"username": "alice", "password": "wrong-one-here"})
check("wrong password rejected", r_wrong.status_code == 401)
check("unknown user rejected", r_unknown.status_code == 401)
check("identical message for both (no enumeration)",
      r_unknown.json()["detail"] == r_wrong.json()["detail"] == "Invalid username or password.")
c3 = Client(app, _loop)
ok = c3.post("/api/auth/login", json={"username": "alice", "password": GOOD})
check("correct password succeeds", ok.status_code == 200)
check("issues fresh cookies", "sunlytics_access" in ok.cookies)

print("\n-- forged and stale tokens --")
c4 = Client(app, _loop)
c4.cookies.set("sunlytics_access", "not.a.jwt")
check("garbage token rejected", c4.get("/api/auth/me").status_code == 401)

import jwt as pyjwt
from api.security.config import JWT_SECRET
from datetime import datetime, timedelta, timezone
n = datetime.now(timezone.utc)
forged = pyjwt.encode(
    {"sub": "acct_evil", "user_id": "u", "username": "evil", "roles": ["admin"],
     "iat": n, "exp": n + timedelta(hours=1), "iss": "sunlytics-m3",
     "aud": "sunlytics-web", "typ": "access"},
    "attacker-secret-attacker-secret-xx", algorithm="HS256")
c5 = Client(app, _loop)
c5.cookies.set("sunlytics_access", forged)
check("token signed with attacker key rejected", c5.get("/api/auth/me").status_code == 401)

ghost = pyjwt.encode(
    {"sub": "acct_does_not_exist", "user_id": "u", "username": "ghost", "roles": ["user"],
     "iat": n, "exp": n + timedelta(hours=1), "iss": "sunlytics-m3",
     "aud": "sunlytics-web", "typ": "access"},
    JWT_SECRET, algorithm="HS256")
c6 = Client(app, _loop)
c6.cookies.set("sunlytics_access", ghost)
check("valid token for deleted account rejected", c6.get("/api/auth/me").status_code == 401)

print("\n-- refresh rotation and reuse detection --")
c7 = Client(app, _loop)
c7.post("/api/auth/login", json={"username": "alice", "password": GOOD})
old_refresh = c7.cookies.get("sunlytics_refresh")
r1 = c7.post("/api/auth/refresh")
check("refresh succeeds", r1.status_code == 200)
new_refresh = c7.cookies.get("sunlytics_refresh")
check("refresh token was rotated", new_refresh != old_refresh)

c8 = Client(app, _loop)
c8.cookies.set("sunlytics_refresh", old_refresh)
check("replaying a retired token is rejected", c8.post("/api/auth/refresh").status_code == 401)
check("reuse revoked the whole family", c7.post("/api/auth/refresh").status_code == 401)

print("\n-- logout and CSRF --")
c9 = Client(app, _loop)
c9.post("/api/auth/login", json={"username": "alice", "password": GOOD})
csrf_val = c9.cookies.get("sunlytics_csrf")
check("logout without CSRF header is 403",
      c9.post("/api/auth/logout", csrf=False).status_code == 403)
check("logout with wrong CSRF header is 403",
      c9.post("/api/auth/logout", csrf=False,
              headers={"X-CSRF-Token": "wrong"}).status_code == 403)
check("logout with correct CSRF header succeeds",
      c9.post("/api/auth/logout").status_code == 200)

print("\n-- registration rate limit --")
# Six register attempts were already made above, against a limit of 5.
check("6th registration from same IP is 429",
      Client(app, _loop).post("/api/auth/register",
                           json={"username": "dave", "password": GOOD}).status_code == 429)

print("\n-- account lockout --")
# Clear the counter so the lockout test is not blocked by the limiter
# it just proved works.
_redis.data.clear()
reg = Client(app, _loop).post("/api/auth/register", json={"username": "carol", "password": GOOD})
check("carol registered for lockout test", reg.status_code == 201)
_redis.data.clear()
for _ in range(9):
    Client(app, _loop).post("/api/auth/login", json={"username": "carol", "password": "nope-nope-nope"})
    _redis.data.clear()   # isolate lockout from the login rate limiter
final = Client(app, _loop).post("/api/auth/login", json={"username": "carol", "password": GOOD})
check("correct password refused while locked out (403)", final.status_code == 403)

print("\nALL PASS" if fails == 0 else "\n" + str(fails) + " FAILURE(S)")
sys.exit(1 if fails else 0)
