import { Component } from "react";
import Button from "../atoms/Button";
import styles from "./ErrorBoundary.module.css";

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
      <div className={styles.screen}>
        <div className={styles.panel}>
          <div className={styles.icon} aria-hidden="true">⚠</div>
          <div className={styles.title}>Something went wrong displaying this page.</div>
          <div className={styles.explanation}>
            The error has been logged to the browser console. You can try rendering
            again, or reload if the problem persists.
          </div>
          <div className={styles.actions}>
            <Button variant="primary" onClick={this.handleReset}>Try again</Button>
            <Button variant="neutral" onClick={() => window.location.reload()}>Reload</Button>
          </div>
          {import.meta.env?.DEV && (
            <pre className={styles.stack}>{String(error?.stack || error)}</pre>
          )}
        </div>
      </div>
    );
  }
}
