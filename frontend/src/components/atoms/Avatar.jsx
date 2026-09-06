import styles from "./Avatar.module.css";

export default function Avatar({
  children,
  size = 32,
  radius,
  fontSize,
  variant,               // undefined (assistant) | user | tag
  className = "",
  style = {},
}) {
  const classes = [styles.avatar, variant ? styles[variant] : "", className]
    .filter(Boolean).join(" ");

  return (
    <div
      className={classes}
      style={{
        "--avatar-size": `${size}px`,
        ...(radius !== undefined && { "--avatar-radius": typeof radius === "number" ? `${radius}px` : radius }),
        ...(fontSize !== undefined && { "--avatar-font-size": `${fontSize}px` }),
        ...style,
      }}
    >
      {children}
    </div>
  );
}
