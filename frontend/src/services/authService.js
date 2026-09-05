import { requestJson } from "./http";

/** Read-only and idempotent, so a transient failure is retried. */
export function getCustomers() {
  return requestJson("/api/auth/customers", { retries: 2 });
}

export function login(customerId) {
  return requestJson("/api/auth/login", {
    method: "POST",
    body: { customer_id: customerId },
  });
}
