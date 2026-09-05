/**
 * One error type for every failure the network layer can produce.
 *
 * Callers should never have to tell a TypeError from fetch apart from a 500 from
 * the API apart from a JSON parse failure — they all arrive here as an ApiError
 * with a `kind`, so a caller can branch on cause without string-matching.
 */

export const ErrorKind = {
  NETWORK: "network",   // could not reach the server at all
  TIMEOUT: "timeout",   // request aborted by our own deadline
  HTTP:    "http",      // server replied, but with a non-2xx status
  PARSE:   "parse",     // server replied 2xx with a body we could not read
};

export class ApiError extends Error {
  constructor(kind, message, { status = null, url = "", body = null, cause = null } = {}) {
    super(message);
    this.name   = "ApiError";
    this.kind   = kind;
    this.status = status;
    this.url    = url;
    this.body   = body;
    this.cause  = cause;
  }

  /**
   * Whether retrying the identical request could plausibly succeed.
   * Deliberately excludes 4xx other than 429: a rejected request stays rejected.
   */
  get isRetryable() {
    if (this.kind === ErrorKind.NETWORK || this.kind === ErrorKind.TIMEOUT) return true;
    if (this.kind !== ErrorKind.HTTP) return false;
    return this.status === 429 || this.status >= 500;
  }

  /**
   * Text safe to put in front of a user: says what happened and what they can do,
   * and never leaks a stack trace, a URL, or a raw server payload.
   */
  get userMessage() {
    switch (this.kind) {
      case ErrorKind.TIMEOUT:
        return "The server took too long to respond. Please try again.";
      case ErrorKind.NETWORK:
        return "Cannot reach the server. Check that the backend is running.";
      case ErrorKind.PARSE:
        return "The server sent a response we could not read.";
      default:
        if (this.status === 401 || this.status === 403) return "Your session is no longer valid. Please sign in again.";
        if (this.status === 404) return "That item no longer exists.";
        if (this.status === 429) return "Too many requests — please wait a moment and try again.";
        if (this.status >= 500)  return "The server ran into a problem. Please try again.";
        if (this.status >= 400)  return "The server rejected that request.";
        return "Something went wrong.";
    }
  }

  /**
   * Plain object for Redux state. Error instances are not serialisable, so
   * thunks must reject with this rather than the error itself.
   */
  toPayload() {
    return { kind: this.kind, status: this.status, message: this.userMessage };
  }
}

/** Normalises anything thrown during a request into an ApiError. */
export function toApiError(err, url) {
  if (err instanceof ApiError) return err;
  if (err?.name === "AbortError") {
    return new ApiError(ErrorKind.TIMEOUT, "Request timed out", { url, cause: err });
  }
  // fetch rejects with a TypeError when the host is unreachable or CORS blocks it.
  return new ApiError(ErrorKind.NETWORK, err?.message || "Network request failed", { url, cause: err });
}
