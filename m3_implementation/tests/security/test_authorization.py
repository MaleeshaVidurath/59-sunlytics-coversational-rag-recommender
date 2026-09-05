"""Authorization on sessions and chat.

The headline case replays the original vulnerability: derive a user_id from the
publicly listed customers and pass it as a query parameter. It must now fail."""
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
from api.routers import sessions as sessions_router
auth_router.get_db = lambda: _db
sessions_router.get_db = lambda: _db
sessions_router.get_collection_name = lambda n: n

app = FastAPI()
app.include_router(auth_router.router)
app.include_router(sessions_router.router)

fails = 0


def check(label, cond):
    global fails
    print("  " + ("ok   " if cond else "FAIL ") + label)
    if not cond:
        fails += 1


_loop.run_until_complete(sec_models.ensure_indexes())
GOOD = "correct-horse-battery-staple"

# ── two accounts, and a private session belonging to the victim ──────────────
victim = Client(app, _loop)
victim.post("/api/auth/register", json={"username": "victim", "password": GOOD})
victim_id = victim.get("/api/auth/me").json()["user_id"]

_loop.run_until_complete(_db.sessions.insert_one({
    "session_id": "sess_victim_private",
    "user_id": victim_id,
    "title": "Victim private chat",
    "selected_model": "m3",
    "messages": [{"role": "user", "content": "SECRET-VICTIM-CONTENT"}],
}))

attacker = Client(app, _loop)
attacker.post("/api/auth/register", json={"username": "attacker", "password": GOOD})
attacker_id = attacker.get("/api/auth/me").json()["user_id"]

print("-- setup --")
check("two distinct accounts created", victim_id != attacker_id)

print("\n-- anonymous access is refused --")
anon = Client(app, _loop)
check("GET  /api/sessions            -> 401", anon.get("/api/sessions").status_code == 401)
check("GET  /api/sessions/{id}       -> 401", anon.get("/api/sessions/sess_victim_private").status_code == 401)
check("POST /api/sessions/new        -> 401", anon.post("/api/sessions/new").status_code == 401)
check("DEL  /api/sessions/{id}       -> 401", anon.delete("/api/sessions/sess_victim_private").status_code == 401)

print("\n-- the ORIGINAL attack, replayed --")
# Before phase 3 this returned the victim's data: user_id was just a query param.
r = anon.get("/api/sessions?user_id=" + victim_id)
check("query-param user_id no longer authenticates -> 401", r.status_code == 401)

r = attacker.get("/api/sessions?user_id=" + victim_id)
check("authenticated attacker passing victim user_id is ignored", r.status_code == 200)
body = r.text
check("  ...victim session NOT in the response", "sess_victim_private" not in body)
check("  ...victim content NOT leaked", "SECRET-VICTIM-CONTENT" not in body)

r = attacker.get("/api/sessions/sess_victim_private?user_id=" + victim_id)
check("attacker cannot read victim transcript (404)", r.status_code == 404)

r = attacker.delete("/api/sessions/sess_victim_private?user_id=" + victim_id)
check("attacker cannot delete victim session (404)", r.status_code == 404)
still_there = _loop.run_until_complete(_db.sessions.find_one({"session_id": "sess_victim_private"}))
check("  ...victim session still exists", still_there is not None)

print("\n-- the owner still has full access --")
r = victim.get("/api/sessions")
check("victim lists own sessions", r.status_code == 200 and "sess_victim_private" in r.text)
check("victim reads own transcript",
      victim.get("/api/sessions/sess_victim_private").status_code == 200)
check("victim can start a new session", victim.post("/api/sessions/new").status_code == 200)

print("\n-- CSRF on state-changing session routes --")
# Same authenticated cookies, header deliberately withheld.
check("POST without CSRF header -> 403",
      victim.post("/api/sessions/new", csrf=False).status_code == 403)
check("GET without CSRF header still works (safe method)",
      victim.get("/api/sessions").status_code == 200)

print("\n-- chat request model no longer accepts identity --")
from api.routers.chat import ChatRequest
fields = set(ChatRequest.model_fields)
check("user_id removed from ChatRequest", "user_id" not in fields)
check("customer_id removed from ChatRequest", "customer_id" not in fields)
check("message still present", "message" in fields)
check("selected_model still present", "selected_model" in fields)

print("\n-- victim finally deletes their own session --")
d = victim.delete("/api/sessions/sess_victim_private")
check("owner delete succeeds", d.status_code == 200)
gone = _loop.run_until_complete(_db.sessions.find_one({"session_id": "sess_victim_private"}))
check("  ...session actually removed", gone is None)

print("\nALL PASS" if fails == 0 else "\n" + str(fails) + " FAILURE(S)")
sys.exit(1 if fails else 0)
