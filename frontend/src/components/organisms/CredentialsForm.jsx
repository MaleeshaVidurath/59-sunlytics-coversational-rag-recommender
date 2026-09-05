import { useState } from "react";
import Button from "../atoms/Button";
import ErrorBanner from "../molecules/ErrorBanner";
import { C } from "../../styles/theme";
import { MIN_PASSWORD_LENGTH } from "../../utils/validation";

/**
 * Username + password form, shared by sign-in and registration.
 *
 * Kept as one component because the two screens differ only in labels and
 * whether a confirm field is shown — duplicating it would mean two places to
 * get autocomplete hints and validation wrong.
 */
export default function CredentialsForm({
  mode,               // "login" | "register"
  onSubmit,           // ({ username, password }) => void
  busy,
  error,
  onDismissError,
}) {
  const isRegister = mode === "register";
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm]   = useState("");
  const [touched, setTouched]   = useState(false);

  // Mismatch is checked here rather than server-side: the server never sees the
  // confirm field, and telling the user immediately is the whole point of it.
  const mismatch = isRegister && touched && confirm !== "" && password !== confirm;
  const tooShort = isRegister && touched && password !== "" && password.length < MIN_PASSWORD_LENGTH;

  const canSubmit = username.trim() !== "" && password !== "" && !busy
    && (!isRegister || (password === confirm && password.length >= MIN_PASSWORD_LENGTH));

  const submit = e => {
    e.preventDefault();
    setTouched(true);
    if (!canSubmit) return;
    onSubmit({ username: username.trim(), password });
  };

  const field = {
    width: "100%", background: C.card, border: `1px solid ${C.border}`,
    borderRadius: 8, color: C.text, padding: "12px 14px", fontSize: 13,
    fontFamily: "system-ui,sans-serif", outline: "none", marginBottom: 12,
  };

  return (
    <form onSubmit={submit} style={{ fontFamily: "system-ui,sans-serif" }}>
      <ErrorBanner message={error} onDismiss={onDismissError} />

      <input
        value={username}
        onChange={e => setUsername(e.target.value)}
        placeholder="Username"
        aria-label="Username"
        autoComplete="username"
        autoCapitalize="none"
        autoFocus
        style={field}
      />

      <input
        type="password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        onBlur={() => setTouched(true)}
        placeholder="Password"
        aria-label="Password"
        // Tells password managers whether to offer a saved password or to
        // generate a new one.
        autoComplete={isRegister ? "new-password" : "current-password"}
        style={{ ...field, borderColor: tooShort ? C.flagText : C.border }}
      />

      {isRegister && (
        <input
          type="password"
          value={confirm}
          onChange={e => setConfirm(e.target.value)}
          onBlur={() => setTouched(true)}
          placeholder="Confirm password"
          aria-label="Confirm password"
          autoComplete="new-password"
          style={{ ...field, borderColor: mismatch ? C.flagText : C.border }}
        />
      )}

      {tooShort && (
        <div style={{ color: C.flagText, fontSize: 11, marginBottom: 10, textAlign: "left" }}>
          Password must be at least {MIN_PASSWORD_LENGTH} characters.
        </div>
      )}
      {mismatch && (
        <div style={{ color: C.flagText, fontSize: 11, marginBottom: 10, textAlign: "left" }}>
          Passwords do not match.
        </div>
      )}

      <Button
        type="submit"
        disabled={!canSubmit}
        style={{ width: "100%", background: canSubmit ? C.accent : C.textMuted,
          border: "none", borderRadius: 8, color: "#0f0f0f", padding: "13px",
          fontSize: 14, fontWeight: 700, cursor: canSubmit ? "pointer" : "not-allowed",
          letterSpacing: 1, textTransform: "uppercase", transition: "background 0.2s" }}>
        {busy
          ? (isRegister ? "Creating account…" : "Signing in…")
          : (isRegister ? "Create account" : "Sign in")}
      </Button>
    </form>
  );
}
