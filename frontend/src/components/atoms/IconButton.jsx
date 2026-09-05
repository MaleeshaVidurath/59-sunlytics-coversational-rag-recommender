import { useState } from "react";
import { C } from "../../styles/theme";

/**
 * Borderless glyph button — sidebar toggle, sign out, delete chat.
 *
 * Only covers the transparent icon-button pattern. The composer's send button
 * is a filled 36px square with its own disabled states, so it stays defined
 * where it is used rather than being bent into this shape.
 */
export default function IconButton({
  children, title, onClick,
  color = C.textMuted, hoverColor, fontSize = 18, padding = "2px 6px", borderRadius = 6,
  style = {},
  ...rest
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      title={title}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: "transparent", border: "none", cursor: "pointer",
        color: hovered && hoverColor ? hoverColor : color,
        fontSize, padding, borderRadius,
        // caller overrides last, so positioning/layout can be supplied per use
        ...style,
      }}
      {...rest}
    >
      {children}
    </button>
  );
}
