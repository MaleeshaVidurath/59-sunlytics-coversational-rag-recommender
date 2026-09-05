import { ApiError, ErrorKind, toApiError } from "./ApiError";

// Shared HTTP plumbing for every backend call.

export const BASE = "http://localhost:8000";
// M2 serves product photos at /api/images/{article_id}
export const M2_IMAGE_BASE = "http://localhost:8001";

/** A chat turn runs an LLM plus retrieval, so the deadline is generous. */
export const DEFAULT_TIMEOUT_MS = 30000;

const sleep = ms => new Promise(r => setTimeout(r, ms));

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

/**
 * Performs one request. Always checks the status, always applies a deadline, and
 * converts every failure into an ApiError.
 *
 * `retries` uses exponential backoff and only fires for retryable failures
 * (network, timeout, 429, 5xx). Leave it at 0 for anything non-idempotent — a
 * retried POST /api/chat would create a second turn.
 */
export async function request(path, {
  method = "GET", body, timeout = DEFAULT_TIMEOUT_MS, retries = 0, signal,
} = {}) {
  const url = `${BASE}${path}`;
  let lastError;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), timeout);

    // Honour a caller's own abort signal alongside our deadline.
    const onExternalAbort = () => controller.abort();
    signal?.addEventListener("abort", onExternalAbort);

    try {
      const options = { method, signal: controller.signal };
      if (body !== undefined) {
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(body);
      }

      const res = await fetch(url, options);

      if (!res.ok) {
        throw new ApiError(
          ErrorKind.HTTP,
          `${method} ${path} failed with ${res.status}`,
          { status: res.status, url, body: await readBody(res) },
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
    } finally {
      clearTimeout(deadline);
      signal?.removeEventListener("abort", onExternalAbort);
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
