import { C } from "../../styles/theme";

/**
 * Full-height centred shell shared by the login and model-select screens —
 * the two views that appear before a chat exists.
 */
export default function CenteredTemplate({ children, fontFamily }) {
  return (
    <div style={{ minHeight:"100vh", background:C.bg,
      display:"flex", alignItems:"center", justifyContent:"center",
      fontFamily }}>
      {children}
    </div>
  );
}
