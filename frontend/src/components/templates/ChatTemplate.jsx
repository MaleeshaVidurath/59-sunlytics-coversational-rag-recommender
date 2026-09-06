import styles from "./ChatTemplate.module.css";

/**
 * Two-column chat shell: a collapsible sidebar beside a header/body/composer
 * column. Owns only layout — the width transition that animates the sidebar
 * open and closed lives here so the panel's own content stays position-agnostic.
 */
export default function ChatTemplate({ sidebar, sidebarOpen, header, children, composer }) {
  return (
    <div className={styles.layout}>
      <div className={`${styles.sidebar} ${sidebarOpen ? "" : styles.collapsed}`.trim()}>
        {sidebar}
      </div>
      <div className={styles.main}>
        {header}
        {children}
        {composer}
      </div>
    </div>
  );
}
