import { useSelector } from "react-redux";
import LoginPage from "./pages/LoginPage";
import ModelSelectPage from "./pages/ModelSelectPage";
import ChatPage from "./pages/ChatPage";
import { selectUser } from "./store/slices/authSlice";
import { selectModel } from "./store/slices/modelSlice";

/**
 * Three-stage gate: sign in, choose a model, then chat.
 *
 * The signed-in user is restored from localStorage so a refresh does not sign
 * them out; the model choice deliberately is not, so every fresh visit makes
 * that choice explicit.
 */
export default function App() {
  const user  = useSelector(selectUser);
  const model = useSelector(selectModel);

  if (!user)  return <LoginPage />;
  if (!model) return <ModelSelectPage />;
  return <ChatPage />;
}
