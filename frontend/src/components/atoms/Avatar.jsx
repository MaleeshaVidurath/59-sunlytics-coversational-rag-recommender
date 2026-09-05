import { C } from "../../styles/theme";

/**
 * Square-or-round tile holding a glyph — the assistant "S" mark, the signed-in
 * user's initial, a model's "M3" tag, and the placeholder shown when a product
 * photo fails to load. Defaults to the gold gradient those marks share.
 */
export default function Avatar({
  children,
  size = 32,
  radius = "50%",
  background = `linear-gradient(135deg,${C.accentDim},${C.accent})`,
  fontSize = 14,
  style = {},
}) {
  return (
    <div style={{
      width: size, height: size, borderRadius: radius, flexShrink: 0, background,
      display: "flex", alignItems: "center", justifyContent: "center", fontSize,
      ...style,
    }}>
      {children}
    </div>
  );
}
