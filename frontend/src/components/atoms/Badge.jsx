/**
 * Small pill label. Holds only the geometry the fifteen pills in this app
 * share; colour, weight and font are passed per use via `style`.
 */
export default function Badge({ children, style = {}, ...rest }) {
  return (
    <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 12, ...style }} {...rest}>
      {children}
    </span>
  );
}
