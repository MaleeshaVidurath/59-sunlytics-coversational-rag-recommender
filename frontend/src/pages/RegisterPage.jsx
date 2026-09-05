import { useDispatch, useSelector } from "react-redux";
import CenteredTemplate from "../components/templates/CenteredTemplate";
import CredentialsForm from "../components/organisms/CredentialsForm";
import Wordmark from "../components/atoms/Wordmark";
import { register, selectAuthError, authErrorCleared } from "../store/slices/authSlice";
import { C } from "../styles/theme";

/**
 * New accounts start cold: no linked H&M persona, so no purchase history.
 * Recommendations are ranked semantically until preferences build up through
 * conversation. Said plainly on the form so the difference is not a surprise.
 */
export default function RegisterPage({ onShowLogin }) {
  const dispatch = useDispatch();
  const error = useSelector(selectAuthError);

  return (
    <CenteredTemplate fontFamily="'Playfair Display',Georgia,serif">
      <div style={{ width:440, maxWidth:"92vw", background:C.sidebar,
        border:`1px solid ${C.border}`, borderRadius:16, padding:"48px 40px", textAlign:"center" }}>
        <Wordmark size={36} letterSpacing={6} color={C.accent} weight={700} marginBottom={8} />
        <div style={{ fontSize:12, letterSpacing:3, color:C.textDim,
          textTransform:"uppercase", marginBottom:24, fontFamily:"system-ui,sans-serif" }}>
          Create an account
        </div>

        <div style={{ fontSize:12, color:C.textMuted, lineHeight:1.6, marginBottom:24,
          fontFamily:"system-ui,sans-serif", textAlign:"left",
          background:C.card, border:`1px solid ${C.border}`, borderRadius:8, padding:"10px 12px" }}>
          A new account starts with no shopping history, so early recommendations
          are based on what you ask for rather than what you have bought before.
        </div>

        <CredentialsForm
          mode="register"
          error={error}
          onDismissError={() => dispatch(authErrorCleared())}
          onSubmit={({ username, password }) => dispatch(register({ username, password }))}
        />

        <div style={{ marginTop:20, fontSize:12, color:C.textDim,
          fontFamily:"system-ui,sans-serif" }}>
          Already have an account?{" "}
          <button onClick={onShowLogin}
            style={{ background:"none", border:"none", color:C.accent,
              cursor:"pointer", fontSize:12, padding:0, textDecoration:"underline" }}>
            Sign in
          </button>
        </div>
      </div>
    </CenteredTemplate>
  );
}
