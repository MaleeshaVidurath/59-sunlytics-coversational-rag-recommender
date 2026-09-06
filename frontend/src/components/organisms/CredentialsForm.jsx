import { useState } from "react";
import Button from "../atoms/Button";
import ErrorBanner from "../molecules/ErrorBanner";
import { MIN_PASSWORD_LENGTH } from "../../utils/validation";
import styles from "./CredentialsForm.module.css";

/**
 * Username + password form, shared by sign-in and registration.
 *
 * Kept as one component because the two screens differ only in labels and
 * whether a confirm field is shown — duplicating it would mean two places to
 * get autocomplete hints and validation wrong.
 */
export default function CredentialsForm({
  mode,               // "login" | "register"
  onSubmit,
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

  return (
    <form className={styles.form} onSubmit={submit}>
      <ErrorBanner message={error} onDismiss={onDismissError} />

      <input
        className={styles.field}
        value={username}
        onChange={e => setUsername(e.target.value)}
        placeholder="Username"
        aria-label="Username"
        autoComplete="username"
        autoCapitalize="none"
        autoFocus
      />

      <input
        className={`${styles.field} ${tooShort ? styles.invalid : ""}`.trim()}
        type="password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        onBlur={() => setTouched(true)}
        placeholder="Password"
        aria-label="Password"
        // Tells password managers whether to offer a saved password or to
        // generate a new one.
        autoComplete={isRegister ? "new-password" : "current-password"}
      />

      {isRegister && (
        <input
          className={`${styles.field} ${mismatch ? styles.invalid : ""}`.trim()}
          type="password"
          value={confirm}
          onChange={e => setConfirm(e.target.value)}
          onBlur={() => setTouched(true)}
          placeholder="Confirm password"
          aria-label="Confirm password"
          autoComplete="new-password"
        />
      )}

      {tooShort && (
        <div className={styles.hint}>
          Password must be at least {MIN_PASSWORD_LENGTH} characters.
        </div>
      )}
      {mismatch && <div className={styles.hint}>Passwords do not match.</div>}

      <Button type="submit" variant="primary" fullWidth disabled={!canSubmit}>
        {busy
          ? (isRegister ? "Creating account…" : "Signing in…")
          : (isRegister ? "Create account" : "Sign in")}
      </Button>
    </form>
  );
}
