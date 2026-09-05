# Authentication & Authorization

How sign-in works in Sunlytics, what it defends against, and — in section 8 —
what it deliberately does **not** cover.

---

## 1. What this replaced

The previous system had no authentication. `POST /api/auth/login` accepted a
`customer_id` and returned a profile; no secret was ever checked. There were no
auth dependencies anywhere in the API.

It also had a **horizontal privilege escalation affecting all 250 accounts**:

1. `GET /api/auth/customers` publicly served every `customer_id` — no auth.
2. Seeded `user_id`s were `user_hist_` + the first 8 hex chars of the
   `customer_id` (verified: `user_hist_01dd9605` ← `01dd96059a1175…`).
3. `user_id` was the only thing gating session read and delete, and it arrived
   as a query parameter.

So anyone who could reach the API could derive any user's id and read or
**delete** their entire chat history. Only localhost-binding contained it.

That chain is now closed, and the closure is regression-tested — see
`tests/security/test_authorization.py`, which replays the attack and asserts it
fails.

---

## 2. Tokens

| | Type | Lifetime | Storage |
|---|---|---|---|
| **Access** | Signed JWT (HS256) | 15 min | `httpOnly` cookie |
| **Refresh** | Opaque 256-bit random | 14 days | `httpOnly` cookie, `Path=/api/auth/refresh` |
| **CSRF** | Random token | 14 days | Readable cookie (by design) |

**Why the split.** A JWT access token verifies with no database round-trip, so
it costs nothing per request. An opaque refresh token cannot be forged and — 
because it is stored server-side — can be revoked instantly. JWTs alone cannot
be revoked before they expire.

**Refresh tokens are stored as SHA-256 only.** A database leak yields no usable
tokens. SHA-256 rather than Argon2 is correct here: the value is already 256
bits of entropy, so there is nothing to brute-force.

**Rotation with reuse detection.** Every refresh retires the presented token and
issues a new one in the same *family*. If a retired token is presented again,
someone holds a copy they should not, so the **entire family is revoked** and
the user must sign in again.

> Worth knowing: with Redis already in the stack, opaque server-side sessions
> would have been simpler than JWTs and easier to revoke. JWT was a requirement;
> the hybrid above recovers most of what a session store would have given.

---

## 3. Passwords

**Argon2id** (`argon2-cffi`, RFC 9106 low-memory profile: 64 MiB, t=3, p=4).
Memory-hard, so GPU and ASIC attacks gain far less than against bcrypt or
PBKDF2. Parameters are recorded inside each hash, so raising them later does not
invalidate existing passwords — they rehash on the next successful login.

**Policy** follows NIST SP 800-63B: minimum 12 characters, a 128-character
ceiling (an unbounded password is a cheap way to make the server burn CPU), and
a common-password blocklist. **No composition rules** — "must contain a symbol"
pushes users toward predictable substitutions.

The blocklist also strips trailing digits and punctuation before matching, so
`password123456` and `Passw0rd!!!!!` are rejected while a genuine passphrase
like `my-password-notebook-blue` is not.

---

## 4. Defences, and how each is verified

| Threat | Defence | Test |
|---|---|---|
| Credential stuffing | Rate limit (20/5min per IP+username) + lockout (8 attempts → 15 min) | `test_auth_flow.py` |
| Password spraying | Rate limit keyed on IP **and** username | `test_auth_flow.py` |
| User enumeration | Identical message for unknown user and wrong password; Argon2 verify against a dummy hash equalises timing (58ms vs 61ms measured) | `test_primitives.py`, `test_auth_flow.py` |
| Token forgery | Pinned algorithm; rejects `alg: none`, wrong key, wrong audience, expired, refresh-as-access | `test_primitives.py`, `test_auth_flow.py` |
| Stolen refresh token | Rotation + family revocation on reuse | `test_auth_flow.py` |
| Deleted account with a live token | Account re-checked on every request | `test_auth_flow.py` |
| CSRF | `SameSite=Lax` + double-submit token, constant-time compared | `test_auth_flow.py`, `test_authorization.py` |
| XSS stealing the session | Tokens in `httpOnly` cookies; no token in JS or Redux | `frontend/tests/auth-client.test.jsx` |
| Reading another user's chats | `user_id` derived from the token, never accepted from the client | `test_authorization.py` |
| Account enumeration via `/customers` | Endpoint now requires authentication | `test_authorization.py` |

---

## 5. Authorization

`api/security/dependencies.py` is the only place identity is established.

- `get_current_account` — verifies the cookie, loads the account, rejects locked
  accounts. Every failure returns the same generic 401.
- `get_current_user_id` — the persona id for downstream queries.
- `require_role("admin")` — role gating; the seeded `admin` account exercises it.
- `verify_csrf` — double-submit check on unsafe methods.

**Dependency order matters.** Routers declare
`dependencies=[Depends(auth), Depends(verify_csrf)]` in that order. Reversed, an
unauthenticated POST answers 403 (CSRF) instead of 401, and the frontend cannot
distinguish "your session expired, refresh it" from "you may not do that".

**`user_id` and `customer_id` were removed from `ChatRequest` entirely.** They
identify whose data the pipeline reads and writes; accepting them from a request
body was the vulnerability.

---

## 6. Frontend

- **No `localStorage`.** The session lives only in `httpOnly` cookies. On load,
  `GET /api/auth/me` re-establishes who you are.
- `credentials: "include"` on every request; the CSRF token is read from its
  cookie and echoed in `X-CSRF-Token`.
- **Single-flight 401 → refresh → replay.** A 15-minute expiry is invisible.
  Concurrent 401s share one refresh — important, because two parallel refreshes
  would trip reuse detection and log the user out for being busy.
- When refresh itself fails, the store is told the session is gone and the UI
  returns to sign-in with an explanation.

---

## 7. Seeded accounts

`python m3_implementation/scripts/seed_accounts.py --force` creates
`user001`–`user250` (each linked to the persona holding that customer's purchase
history) plus one `admin`, and writes credentials to `credentials/`.

- **`credentials/` is gitignored. Never commit it.**
- It is the **only** plaintext copy — the database stores Argon2id hashes.
- Re-running **rotates every password** and invalidates sessions. It refuses to
  run without `--force` when seeded accounts already exist.
- Self-registered accounts are never touched: the wipe is scoped to
  `is_seeded: true`.

---

## 8. What this does NOT cover

Stated plainly, because "secure" without a boundary is meaningless.

| Gap | Why | Risk |
|---|---|---|
| **No TLS** | Needs a reverse proxy or hosting platform | Cookies are `Secure`-flagged, but on a plain-HTTP non-localhost host browsers drop them and auth silently fails. Do not deploy without HTTPS. |
| **No email verification or password reset** | Needs a mail provider | A forgotten password cannot be recovered. Faking this would be worse than omitting it. |
| **No MFA** | Out of scope | Password is the only factor. |
| **No breached-password check** | Needs the HIBP k-anonymity API | The local blocklist catches trivial cases only. |
| **Secrets in `.env`** | No secret manager here | `JWT_SECRET` sits on disk. Rotating it invalidates all sessions. |
| **Plaintext credentials on disk** | You asked for a readable list | `credentials/` is gitignored, but a `git add -f`, a zipped project folder, or a screen share leaks all 251 at once. |
| **Registration reveals taken usernames** | Unavoidable without email confirmation | Rate limiting (5/hour/IP) stops it being a usable oracle. |
| **`/api/rl/feedback` still takes `user_id`** | It is a research signal collector, not a data-access route | A caller could attribute feedback to another user. No data is readable this way. Worth fixing if it ever informs anything user-visible. |
| **Rate limiting fails open** | Redis down should not lock everyone out | If Redis is unavailable, only account lockout applies. |

---

## 9. Running the tests

```bash
# Backend — no databases needed, uses in-memory fakes
python m3_implementation/tests/security/run_all.py

# Frontend auth client
cd frontend && npm run test:auth
```

The backend suites run in separate processes on purpose: they patch module
globals at import time, so sharing an interpreter would let one suite's fakes
leak into another's.

---

## 10. Configuration

Set in `.env`:

```bash
ENVIRONMENT=development          # "production" makes a weak JWT_SECRET fatal
JWT_SECRET=<48+ random bytes>    # python -c "import secrets; print(secrets.token_urlsafe(48))"
ACCESS_TOKEN_TTL_MINUTES=15
REFRESH_TOKEN_TTL_DAYS=14
COOKIE_SECURE=true               # false ONLY for local plain-HTTP testing
COOKIE_SAMESITE=lax              # "none" (+ Secure) if the API is cross-site
PASSWORD_MIN_LENGTH=12
LOGIN_MAX_ATTEMPTS=8
LOGIN_LOCKOUT_MINUTES=15
TRUST_PROXY_HEADERS=false        # true ONLY behind a proxy that overwrites X-Forwarded-For
```

In development a missing `JWT_SECRET` generates an ephemeral one and warns —
sessions then break on every restart, which is the intended noisy failure. In
production it raises at startup instead.

`TRUST_PROXY_HEADERS=true` without a proxy in front lets a client forge
`X-Forwarded-For` and bypass every IP-based limit.
