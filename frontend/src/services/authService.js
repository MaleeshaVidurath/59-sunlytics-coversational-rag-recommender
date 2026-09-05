import { requestJson, request } from "./http";

/**
 * Authentication calls.
 *
 * None of these return a token: the server sets httpOnly cookies that
 * JavaScript cannot read, which is what makes an XSS unable to steal a session.
 * Each returns the signed-in profile instead.
 */

export function register(username, password) {
  return requestJson("/api/auth/register", {
    method: "POST",
    body: { username, password },
    // A 401 here means the credentials were wrong, not that a session lapsed;
    // refreshing would be meaningless.
    allowRefresh: false,
  });
}

export function login(username, password) {
  return requestJson("/api/auth/login", {
    method: "POST",
    body: { username, password },
    allowRefresh: false,
  });
}

/**
 * Restores the session on page load. The browser sends the cookies; a 401 just
 * means nobody is signed in.
 */
export function fetchMe() {
  return requestJson("/api/auth/me", { allowRefresh: true });
}

export async function logout() {
  await request("/api/auth/logout", { method: "POST", allowRefresh: false });
}

/** Persona catalogue — now behind authentication. */
export function getCustomers() {
  return requestJson("/api/auth/customers", { retries: 2 });
}
