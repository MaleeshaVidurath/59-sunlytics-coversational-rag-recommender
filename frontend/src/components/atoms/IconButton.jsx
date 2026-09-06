import styles from "./IconButton.module.css";

/**
 * Borderless glyph button — sidebar toggle, sign out, delete chat.
 *
 * Only covers the transparent icon-button pattern. The composer's send button
 * is a filled square with its own disabled states, so it stays defined where it
 * is used rather than being bent into this shape.
 */
export default function IconButton({
  children,
  tone = "danger",       // danger | muted
  small = false,
  floating = false,
  className = "",
  ...rest
}) {
  const classes = [
    styles.iconButton,
    styles[tone],
    small ? styles.small : "",
    floating ? styles.floating : "",
    className,
  ].filter(Boolean).join(" ");

  return <button className={classes} {...rest}>{children}</button>;
}
