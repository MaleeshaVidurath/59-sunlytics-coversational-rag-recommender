import { ApiError, ErrorKind, toApiError } from "./ApiError";

// Shared HTTP plumbing for every backend call.

export const BASE = "http://localhost:8000";
// M2 serves product photos at /api/images/{article_id}
export const M2_IMAGE_BASE = "http://localhost:8001";

/** A chat turn runs an LLM plus retrieval, so the deadline is generous. */
export const DEFAULT_TIMEOUT_MS = 30000;

const CSRF_COOKIE = "sunlytics_csrf";
const CSRF_HEADER = "X-CSRF-Token";
const REFRESH_PATH = "/api/auth/refresh";

const sleep = ms => new Promise(r => setTimeout(r, ms));

/**
 * Reads the CSRF token the server set.
 *
 * This is the one auth cookie that is deliberately NOT httpOnly: the
 * double-submit pattern needs the page to echo it back in a header, which a
 * cross-origin attacker cannot do even though the browser would attach the
 * cookies automatically.
 */
function csrfToken() {
  const match = document.cookie.match(new RegExp(`(?:^|;\\s*)${CSRF_COOKIE}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

/**
 * Reads a response body without ever throwing.
 *
 * An error response is frequently HTML (a proxy page, a stack trace) rather than
 * JSON, so parsing is best-effort: the text is kept for diagnostics either way.
 */
async function readBody(res) {
  const text = await res.text().catch(() => "");
  if (!text) return null;
  try { return JSON.parse(text); }
  catch { return text; }
}

/** FastAPI reports errors as {detail: "..."}; prefer that over a generic string. */
function detailFrom(body) {
  if (body && typeof body === "object" && typeof body.detail === "string") return body.detail;
  return null;
}

// ── session refresh ───────────────────────────────────────────────────────────

/**
 * In-flight refresh, shared across callers.
 *
 * Several requests can 401 at the same moment when a 15-minute access token
 * expires. Without this, each would fire its own refresh; because refresh
 * tokens rotate and reuse is treated as theft, the second one would land on an
 * already-rotated token and the server would revoke the whole family — logging
 * the user out precisely because the page was busy.
 */
let refreshInFlight = null;

/** Notified when the session is definitively gone, so the store can react. */
let onSessionExpired = () => {};

export function setSessionExpiredHandler(fn) {
  onSessionExpired = typeof fn === "function" ? fn : () => {};
}

async function refreshSession() {
  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      try {
        const res = await fetch(`${BASE}${REFRESH_PATH}`, {
          method: "POST",
          credentials: "include",
          headers: csrfToken() ? { [CSRF_HEADER]: csrfToken() } : {},
        });
        return res.ok;
      } catch {
        return false;
      } finally {
        // Cleared synchronously. Callers that already grabbed this promise keep
        // awaiting the same object, so single-flight still holds — while a
        // deferred clear could leave an already-resolved `false` in place, and
        // the next 401 would "fail" a refresh it never actually attempted.
        refreshInFlight = null;
      }
    })();
  }
  return refreshInFlight;
}

// ── requests ──────────────────────────────────────────────────────────────────

async function rawRequest(path, { method, body, timeout, signal }) {
  const url = `${BASE}${path}`;
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(), timeout);
  const onExternalAbort = () => controller.abort();
  signal?.addEventListener("abort", onExternalAbort);

  try {
    const headers = {};
    const options = {
      method,
      signal: controller.signal,
      // Auth lives entirely in httpOnly cookies, so every request must carry
      // them. Without this the browser omits cookies cross-origin and every
      // call is anonymous.
      credentials: "include",
    };

    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    // Safe methods are exempt server-side; sending it anyway is harmless.
    const token = csrfToken();
    if (token) headers[CSRF_HEADER] = token;

    options.headers = headers;
    return await fetch(url, options);
  } finally {
    clearTimeout(deadline);
    signal?.removeEventListener("abort", onExternalAbort);
  }
}

/**
 * Performs a request. Always checks the status, always applies a deadline, and
 * converts every failure into an ApiError.
 *
 * On a 401 it attempts exactly one token refresh and replays the request. That
 * is what makes a 15-minute access token invisible to the user.
 *
 * `retries` uses exponential backoff and only fires for retryable failures
 * (network, timeout, 429, 5xx). Leave it at 0 for anything non-idempotent — a
 * retried POST /api/chat would create a second turn.
 */
export async function request(path, {
  method = "GET", body, timeout = DEFAULT_TIMEOUT_MS, retries = 0, signal,
  allowRefresh = true,
} = {}) {
  const url = `${BASE}${path}`;
  let lastError;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      let res = await rawRequest(path, { method, body, timeout, signal });

      if (res.status === 401 && allowRefresh && path !== REFRESH_PATH) {
        const refreshed = await refreshSession();
        if (refreshed) {
          res = await rawRequest(path, { method, body, timeout, signal });
        }
        if (!refreshed || res.status === 401) {
          onSessionExpired();
          throw new ApiError(ErrorKind.HTTP, "Session expired", {
            status: 401, url, body: await readBody(res),
          });
        }
      }

      if (!res.ok) {
        const parsed = await readBody(res);
        throw new ApiError(
          ErrorKind.HTTP,
          detailFrom(parsed) || `${method} ${path} failed with ${res.status}`,
          { status: res.status, url, body: parsed },
        );
      }
      return res;
    } catch (err) {
      lastError = toApiError(err, url);
      if (attempt < retries && lastError.isRetryable) {
        await sleep(300 * 2 ** attempt);   // 300ms, 600ms, 1200ms…
        continue;
      }
      throw lastError;
    }
  }
  throw lastError;
}

/** As `request`, but parses the response body as JSON. */
export async function requestJson(path, options) {
  const res = await request(path, options);
  const parsed = await readBody(res);
  if (parsed !== null && typeof parsed !== "object") {
    throw new ApiError(ErrorKind.PARSE, `${path} did not return JSON`, {
      status: res.status, url: `${BASE}${path}`, body: parsed,
    });
  }
  return parsed ?? {};
}
