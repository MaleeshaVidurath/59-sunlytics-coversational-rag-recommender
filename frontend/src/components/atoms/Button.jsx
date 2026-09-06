import styles from "./Button.module.css";

/**
 * Button.
 *
 * Hover, focus and disabled states live in Button.module.css. The previous
 * version tracked hover in React state and merged style objects, which meant a
 * re-render every time the pointer crossed a button — CSS does it for free, and
 * can also express :focus-visible, which inline styles cannot.
 */
export default function Button({
  children,
  variant = "ghost",     // ghost | chip | primary | success | neutral | link
  fullWidth = false,
  className = "",
  ...rest
}) {
  const classes = [
    styles.button,
    styles[variant],
    fullWidth ? styles.fullWidth : "",
    className,
  ].filter(Boolean).join(" ");

  return <button className={classes} {...rest}>{children}</button>;
}
