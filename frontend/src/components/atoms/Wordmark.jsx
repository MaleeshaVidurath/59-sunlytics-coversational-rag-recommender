/**
 * The SUNLYTICS wordmark. Appears at four sizes — login, model select, sidebar
 * header and the empty-chat state — differing only in scale and colour.
 *
 * `weight` is left unset by default because the empty-chat instance never
 * declared one; callers that had an explicit 700 pass it through.
 */
export default function Wordmark({
  size = 18, letterSpacing = 3, color, weight, marginBottom = 0, style = {},
}) {
  return (
    <div style={{
      fontSize: size, letterSpacing, color, fontWeight: weight, marginBottom,
      fontFamily: "'Playfair Display',Georgia,serif",
      ...style,
    }}>
      SUNLYTICS
    </div>
  );
}
