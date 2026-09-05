/**
 * Frontend auth-client tests.
 *
 * Drives the real store and services against a mock fetch that emulates the
 * server's cookie behaviour: httpOnly access/refresh cookies the page cannot
 * read, a readable CSRF cookie, 401 on an expired access token, and refresh
 * rotation that requires an actual refresh cookie.
 */
import { store } from "../src/store";
import { bootstrapSession, login, register, logout } from "../src/store/slices/authSlice";
import { fetchSessions } from "../src/store/slices/sessionsSlice";
import { sendChatMessage } from "../src/store/slices/chatSlice";

let fails = 0;
const check = (l, c) => { console.log("  " + (c ? "ok   " : "FAIL ") + l); if (!c) fails++; };

// ── fake browser cookie jar ───────────────────────────────────────────────────
const jar = new Map();
globalThis.document = {
  get cookie() {
    // Exactly what a page can see: httpOnly cookies are invisible.
    return [...jar.entries()]
      .filter(([, v]) => !v.httpOnly)
      .map(([k, v]) => `${k}=${v.value}`).join("; ");
  },
};

const setSession = () => {
  jar.set("sunlytics_access",  { value: "access-" + Math.random(), httpOnly: true });
  jar.set("sunlytics_refresh", { value: "refresh-" + Math.random(), httpOnly: true });
  jar.set("sunlytics_csrf",    { value: "csrf-token-abc", httpOnly: false });
};

// ── mock server ───────────────────────────────────────────────────────────────
const calls = [];
let accessValid = false;
let refreshCount = 0;
let failRefresh = false;

const PROFILE = { account_id: "acct_1", username: "user001", roles: ["user"],
                  user_id: "user_hist_abc", customer_id: "abc123", purchase_summary: {} };

globalThis.fetch = async (url, opts = {}) => {
  const path = String(url).replace("http://localhost:8000", "");
  calls.push({ path, method: opts.method || "GET",
               credentials: opts.credentials,
               csrf: opts.headers?.["X-CSRF-Token"] ?? null,
               body: opts.body ? JSON.parse(opts.body) : null });

  const json = (obj, status = 200) =>
    ({ ok: status < 400, status, text: async () => JSON.stringify(obj) });

  if (path === "/api/auth/login" || path === "/api/auth/register") {
    setSession(); accessValid = true;
    return json(PROFILE, path.endsWith("register") ? 201 : 200);
  }
  if (path === "/api/auth/refresh") {
    refreshCount++;
    // A signed-out visitor holds no refresh cookie, so the server has nothing
    // to rotate and answers 401. Modelling this matters: without it the test
    // would "restore" a session that never existed.
    if (failRefresh || !jar.has("sunlytics_refresh")) {
      return json({ detail: "Session expired. Please sign in again." }, 401);
    }
    setSession(); accessValid = true;
    return json(PROFILE);
  }
  if (path === "/api/auth/logout") { jar.clear(); accessValid = false; return json({ detail: "Signed out." }); }

  if (!accessValid) return json({ detail: "Not authenticated" }, 401);

  if (path === "/api/auth/me")   return json(PROFILE);
  if (path === "/api/sessions")  return json({ sessions: [{ session_id: "s1", title: "Chat", selected_model: "m3" }] });
  if (path === "/api/chat")      return json({ response_text: "hi", session_id: "s1", revisions: [] });
  return json({}, 404);
};

const last = p => [...calls].reverse().find(c => c.path === p);

(async () => {
  console.log("-- bootstrap with no cookies at all --");
  await store.dispatch(bootstrapSession());
  let s = store.getState().auth;
  check("status becomes anonymous", s.status === "anonymous");
  check("no user", s.user === null);
  check("an ordinary signed-out 401 shows no error banner", s.error === null);

  console.log("\n-- sign in --");
  await store.dispatch(login({ username: "user001", password: "jPv6k2nrivcnjXUv" }));
  s = store.getState().auth;
  check("status becomes authenticated", s.status === "authenticated");
  check("profile stored", s.user.username === "user001");
  const loginCall = last("/api/auth/login");
  check("sends credentials: include", loginCall.credentials === "include");
  check("password sent in the body, not a query string", loginCall.body.password === "jPv6k2nrivcnjXUv");

  console.log("\n-- no token is readable by the page --");
  check("access cookie invisible to JS",  !document.cookie.includes("sunlytics_access"));
  check("refresh cookie invisible to JS", !document.cookie.includes("sunlytics_refresh"));
  check("csrf cookie IS readable (double-submit)", document.cookie.includes("sunlytics_csrf"));
  check("no token anywhere in redux state",
        !JSON.stringify(store.getState()).match(/access-0|refresh-0/));

  console.log("\n-- csrf header on state-changing requests --");
  await store.dispatch(sendChatMessage({ text: "I need 4 shirts", model: "m3" }));
  const chatCall = last("/api/chat");
  check("POST /api/chat carries X-CSRF-Token", chatCall.csrf === "csrf-token-abc");
  check("POST /api/chat carries cookies", chatCall.credentials === "include");
  check("identity NOT in the chat payload",
        !("user_id" in chatCall.body) && !("customer_id" in chatCall.body));
  check("message still sent", chatCall.body.message === "I need 4 shirts");

  console.log("\n-- sessions call sends no user_id --");
  await store.dispatch(fetchSessions());
  check("GET /api/sessions has no query string", last("/api/sessions").path === "/api/sessions");

  console.log("\n-- expired access token is refreshed transparently --");
  accessValid = false;
  refreshCount = 0;
  calls.length = 0;
  await store.dispatch(fetchSessions());
  s = store.getState();
  check("refresh attempted exactly once", refreshCount === 1);
  check("original request replayed",
        calls.filter(c => c.path === "/api/sessions").length === 2);
  check("sessions loaded despite the expiry", s.sessions.items.length === 1);
  check("user never saw a sign-out", s.auth.status === "authenticated");

  console.log("\n-- concurrent 401s share ONE refresh --");
  accessValid = false;
  refreshCount = 0;
  await Promise.all([
    store.dispatch(fetchSessions()),
    store.dispatch(fetchSessions()),
    store.dispatch(fetchSessions()),
  ]);
  check(`three parallel 401s triggered ${refreshCount} refresh (expect 1)`, refreshCount === 1);

  console.log("\n-- refresh itself fails: session is over --");
  accessValid = false; failRefresh = true;
  await store.dispatch(fetchSessions());
  s = store.getState();
  check("status drops to anonymous", s.auth.status === "anonymous");
  check("user cleared", s.auth.user === null);
  check("explains why", /session expired/i.test(s.auth.error || ""));
  check("sidebar cleared on expiry", s.sessions.items.length === 0);

  console.log("\n-- register --");
  failRefresh = false;
  await store.dispatch(register({ username: "newbie", password: "correct-horse-battery" }));
  s = store.getState().auth;
  check("registration signs the user in", s.status === "authenticated");

  console.log("\n-- sign out --");
  await store.dispatch(logout());
  s = store.getState();
  check("status anonymous", s.auth.status === "anonymous");
  check("cookies gone", document.cookie === "");
  check("model choice cleared", s.model.selected === null);
  check("chat transcript cleared", s.chat.messages.length === 0);
  check("sessions cleared", s.sessions.items.length === 0);

  console.log(fails === 0 ? "\nALL PASS" : `\n${fails} FAILURE(S)`);
  process.exit(fails ? 1 : 0);
})();
