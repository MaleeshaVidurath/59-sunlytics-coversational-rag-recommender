import styles from "./Badge.module.css";

/**
 * Small pill label.
 *
 * `color` is only used by the intent variant, where the hue is chosen per
 * label at runtime — it arrives as a custom property so the stylesheet still
 * owns the layout.
 */
export default function Badge({
  children,
  variant = "mono",      // mono | intent | flag | contra
  color,
  className = "",
}) {
  const classes = [styles.badge, styles[variant], className].filter(Boolean).join(" ");
  return (
    <span className={classes} style={color ? { "--badge-color": color } : undefined}>
      {children}
    </span>
  );
}
