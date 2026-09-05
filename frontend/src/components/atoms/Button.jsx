import { useState } from "react";

/**
 * Button with a declarative hover state.
 *
 * Replaces the onMouseEnter/onMouseLeave pairs that were mutating
 * `e.currentTarget.style` by hand in eight places. `hoverStyle` is merged over
 * `style` while hovered, so callers describe the hover appearance instead of
 * the transition into and out of it. Hover is suppressed while disabled.
 */
export default function Button({ children, style = {}, hoverStyle, disabled = false, ...rest }) {
  const [hovered, setHovered] = useState(false);
  const applied = hovered && !disabled && hoverStyle ? { ...style, ...hoverStyle } : style;
  return (
    <button
      disabled={disabled}
      style={applied}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      {...rest}
    >
      {children}
    </button>
  );
}
