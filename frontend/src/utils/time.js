/** Renders a timestamp as a short relative age ("just now", "5m ago"). */
export function timeAgo(ts) {
  if (!ts) return "";
  const s = Math.floor((new Date() - new Date(ts)) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}
