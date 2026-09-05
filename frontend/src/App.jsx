import { useEffect, useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import ModelSelectPage from "./pages/ModelSelectPage";
import ChatPage from "./pages/ChatPage";
import CenteredTemplate from "./components/templates/CenteredTemplate";
import Wordmark from "./components/atoms/Wordmark";
import { bootstrapSession, selectAuthStatus } from "./store/slices/authSlice";
import { selectModel } from "./store/slices/modelSlice";
import { C } from "./styles/theme";

/**
 * Gate: sign in -> choose a model -> chat.
 *
 * On load the session is unknown, because it lives in httpOnly cookies the page
 * cannot read. `bootstrapSession` asks the server who we are. Rendering the
 * login screen during that check would flash it at users who are in fact signed
 * in, so a neutral splash holds the gate until the answer arrives.
 */
export default function App() {
  const dispatch = useDispatch();
  const status = useSelector(selectAuthStatus);
  const model  = useSelector(selectModel);

  // Local: which auth screen is showing. Nothing outside this gate needs it.
  const [showRegister, setShowRegister] = useState(false);

  useEffect(() => { dispatch(bootstrapSession()); }, [dispatch]);

  if (status === "bootstrapping") {
    return (
      <CenteredTemplate fontFamily="'Playfair Display',Georgia,serif">
        <div style={{ textAlign: "center" }}>
          <Wordmark size={30} letterSpacing={5} color={C.accentDim} weight={700} marginBottom={10} />
          <div style={{ color: C.textMuted, fontSize: 12,
            fontFamily: "system-ui,sans-serif" }}>
            Restoring your session…
          </div>
        </div>
      </CenteredTemplate>
    );
  }

  if (status !== "authenticated") {
    return showRegister
      ? <RegisterPage onShowLogin={() => setShowRegister(false)} />
      : <LoginPage onShowRegister={() => setShowRegister(true)} />;
  }

  if (!model) return <ModelSelectPage />;
  return <ChatPage />;
}
