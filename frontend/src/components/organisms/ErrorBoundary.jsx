import { Component } from "react";
import { C } from "../../styles/theme";

/**
 * Catches render-time errors so one bad message or malformed product payload
 * cannot blank the entire page.
 *
 * This is the one place a class component is still required: there is no hook
 * equivalent of componentDidCatch.
 *
 * Note what it does NOT catch — event handlers, async code, and thunks. Those
 * paths are covered by the ApiError handling in the services and slices; this is
 * strictly the last line of defence for rendering.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Kept as console output on purpose: this project has no error-reporting
    // backend. If one is added, report it here.
    console.error("Render error caught by ErrorBoundary:", error, info?.componentStack);
  }

  handleReset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div style={{ minHeight:"100vh", background:C.bg, color:C.text,
        display:"flex", alignItems:"center", justifyContent:"center", padding:24,
        fontFamily:"system-ui,-apple-system,sans-serif" }}>
        <div style={{ maxWidth:460, background:C.sidebar, border:`1px solid ${C.border}`,
          borderRadius:14, padding:"32px 28px", textAlign:"center" }}>
          <div style={{ fontSize:30, marginBottom:12 }}>⚠</div>
          <div style={{ fontSize:16, fontWeight:600, marginBottom:8 }}>
            Something went wrong displaying this page.
          </div>
          <div style={{ fontSize:13, color:C.textDim, lineHeight:1.6, marginBottom:20 }}>
            The error has been logged to the browser console. You can try rendering
            again, or reload if the problem persists.
          </div>
          <div style={{ display:"flex", gap:10, justifyContent:"center" }}>
            <button onClick={this.handleReset}
              style={{ background:C.accent, border:"none", borderRadius:8, color:"#0f0f0f",
                padding:"10px 20px", fontSize:13, fontWeight:700, cursor:"pointer" }}>
              Try again
            </button>
            <button onClick={() => window.location.reload()}
              style={{ background:"transparent", border:`1px solid ${C.border}`,
                borderRadius:8, color:C.textDim, padding:"10px 20px",
                fontSize:13, cursor:"pointer" }}>
              Reload
            </button>
          </div>
          {import.meta.env?.DEV && (
            <pre style={{ marginTop:18, textAlign:"left", fontSize:11, color:C.textMuted,
              background:C.bg, border:`1px solid ${C.border}`, borderRadius:8,
              padding:10, overflowX:"auto", maxHeight:160 }}>
              {String(error?.stack || error)}
            </pre>
          )}
        </div>
      </div>
    );
  }
}
