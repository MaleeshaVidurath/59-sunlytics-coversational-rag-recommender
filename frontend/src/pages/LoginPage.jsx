import { useState, useEffect } from "react";
import { useDispatch } from "react-redux";
import CenteredTemplate from "../components/templates/CenteredTemplate";
import CustomerPicker from "../components/organisms/CustomerPicker";
import Wordmark from "../components/atoms/Wordmark";
import { getCustomers, login } from "../services/authService";
import { loggedIn } from "../store/slices/authSlice";
import { C } from "../styles/theme";

export default function LoginPage() {
  const dispatch = useDispatch();

  // Form state stays local: nothing outside this screen reads a half-filled
  // login form, and it is discarded the moment sign-in succeeds.
  const [customers, setCustomers] = useState([]);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(true);
  const [logging, setLogging] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getCustomers().then(d => { setCustomers(d.customers||[]); setLoading(false); })
      .catch(() => { setError("Cannot connect to server."); setLoading(false); });
  }, []);

  const handleLogin = async () => {
    if (!selected) return;
    setLogging(true); setError("");
    try { dispatch(loggedIn(await login(selected))); }
    catch { setError("Login failed."); setLogging(false); }
  };

  return (
    <CenteredTemplate fontFamily="'Playfair Display',Georgia,serif">
      <div style={{ width:440, background:C.sidebar,
        border:`1px solid ${C.border}`, borderRadius:16, padding:"48px 40px", textAlign:"center" }}>
        <Wordmark size={36} letterSpacing={6} color={C.accent} weight={700} marginBottom={8} />
        <div style={{ fontSize:12, letterSpacing:3, color:C.textDim,
          textTransform:"uppercase", marginBottom:40, fontFamily:"system-ui,sans-serif" }}>
          Fashion Intelligence System
        </div>
        <div style={{ color:C.textDim, fontSize:13, marginBottom:20, fontFamily:"system-ui,sans-serif" }}>
          Select your customer profile to continue
        </div>
        <CustomerPicker
          customers={customers}
          selected={selected}
          onSelect={setSelected}
          loading={loading}
          logging={logging}
          error={error}
          onSubmit={handleLogin}
        />
      </div>
    </CenteredTemplate>
  );
}
