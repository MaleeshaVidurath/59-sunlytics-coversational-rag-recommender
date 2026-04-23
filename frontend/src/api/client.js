const BASE = "http://localhost:8000";

export async function getCustomers() {
  const res = await fetch(`${BASE}/api/auth/customers`);
  return res.json();
}

export async function login(customerId) {
  const res = await fetch(`${BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_id: customerId }),
  });
  if (!res.ok) throw new Error("Login failed");
  return res.json();
}

export async function getSessions(userId) {
  const res = await fetch(`${BASE}/api/sessions?user_id=${userId}`);
  return res.json();
}

export async function getSessionHistory(sessionId, userId) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}?user_id=${userId}`);
  return res.json();
}

export async function sendMessage({ userId, customerId, message, sessionId }) {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      customer_id: customerId,
      message,
      session_id: sessionId || null,
    }),
  });
  if (!res.ok) throw new Error("Chat request failed");
  return res.json();
}
