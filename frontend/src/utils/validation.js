/**
 * Input validation, kept as pure functions so the rules are testable and stated
 * in exactly one place rather than re-derived at each call site.
 *
 * Each validator returns { valid, value?, error? } — the caller decides whether
 * to show the error, disable a control, or both.
 */

/** Long enough for any real shopping request; short enough to bound the prompt. */
export const MAX_MESSAGE_LENGTH = 2000;

/** Point at which the UI starts showing a character counter. */
export const MESSAGE_WARN_AT = MAX_MESSAGE_LENGTH - 200;

export function validateMessage(text) {
  const value = (text ?? "").trim();

  if (!value) {
    return { valid: false, error: "Type a message first." };
  }
  if (value.length > MAX_MESSAGE_LENGTH) {
    return {
      valid: false,
      error: `Message is ${value.length} characters — the limit is ${MAX_MESSAGE_LENGTH}.`,
    };
  }
  return { valid: true, value };
}

export function validateCustomerId(id) {
  const value = (id ?? "").trim();
  if (!value) return { valid: false, error: "Select a customer profile to continue." };
  return { valid: true, value };
}
