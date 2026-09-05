import { C } from "../../styles/theme";

/**
 * Two-column chat shell: a collapsible sidebar beside a header/body/composer
 * column. Owns only layout — the width transition that animates the sidebar
 * open and closed lives here so the panel's own content stays position-agnostic.
 */
export default function ChatTemplate({ sidebar, sidebarOpen, header, children, composer }) {
  return (
    <div style={{ height:"100vh", display:"flex", background:C.bg,
      fontFamily:"system-ui,-apple-system,sans-serif", overflow:"hidden" }}>

      <div style={{ width:sidebarOpen?260:0, flexShrink:0, background:C.sidebar,
        borderRight:`1px solid ${C.border}`, display:"flex", flexDirection:"column",
        overflow:"hidden", transition:"width 0.25s ease" }}>
        {sidebar}
      </div>

      <div style={{ flex:1, display:"flex", flexDirection:"column", minWidth:0 }}>
        {header}
        {children}
        {composer}
      </div>

    </div>
  );
}
