import { useDispatch, useSelector } from "react-redux";
import CenteredTemplate from "../components/templates/CenteredTemplate";
import CredentialsForm from "../components/organisms/CredentialsForm";
import Wordmark from "../components/atoms/Wordmark";
import Button from "../components/atoms/Button";
import { login, selectAuthError, authErrorCleared } from "../store/slices/authSlice";
import styles from "./AuthCard.module.css";

export default function LoginPage({ onShowRegister }) {
  const dispatch = useDispatch();
  const error = useSelector(selectAuthError);
  const busy  = useSelector(s => s.auth.status === "bootstrapping");

  return (
    <CenteredTemplate font="serif">
      <div className={styles.card}>
        <Wordmark size={36} letterSpacing={6} marginBottom={8} />
        <div className={styles.subtitle}>Fashion Intelligence System</div>

        <CredentialsForm
          mode="login"
          busy={busy}
          error={error}
          onDismissError={() => dispatch(authErrorCleared())}
          onSubmit={({ username, password }) => dispatch(login({ username, password }))}
        />

        <div className={styles.switch}>
          No account?{" "}
          <Button variant="link" onClick={onShowRegister}>Create one</Button>
        </div>
      </div>
    </CenteredTemplate>
  );
}
