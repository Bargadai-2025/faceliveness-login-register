/**
 * API origin with no trailing slash.
 * Empty string = same-origin (use Vite dev proxy in development).
 */
export function getApiBase() {
  const v = import.meta.env.VITE_API_URL;
  if (v === "" || v === "proxy") return "";
  if (v == null) return "http://localhost:8000";
  const s = String(v).trim();
  if (!s) return "";
  return s.replace(/\/$/, "");
}
