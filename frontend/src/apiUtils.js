/**
 * Safe parsing of FastAPI / backend responses for PoC (no raw stack traces in UI).
 */
import { ERROR_LABELS, resolveSecurityErrorLabel } from "./securityErrorMessages";

export function parseApiDetail(data, fallback = "Verification Failed Retry") {
  if (!data) return fallback;
  if (typeof data.user_message === "string" && data.user_message.trim()) {
    return data.user_message.trim();
  }
  if (typeof data.error === "string" && data.error.trim()) {
    return data.error.trim();
  }
  const d = data.detail;
  if (typeof d === "string" && d.trim()) return d.trim();
  if (Array.isArray(d) && d.length > 0) {
    const first = d[0];
    if (typeof first === "string") return first;
    if (first?.msg) return String(first.msg);
  }
  return fallback;
}

/** Never show raw HTTP bodies or Python exceptions to users. */
export function sanitizeUserMessage(raw, hints = {}) {
  const text = String(raw || "");
  if (
    /traceback|exception|sql|postgres|uvicorn|internal server|500:/i.test(text)
  ) {
    return resolveSecurityErrorLabel("", { ...hints });
  }
  if (text.length > 220) {
    return resolveSecurityErrorLabel(text.slice(0, 220), hints);
  }
  return resolveSecurityErrorLabel(text, hints);
}

export async function parseJsonResponse(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return {
      error: ERROR_LABELS.SERVER_ERROR,
      error_code: "INVALID_JSON",
      retry_allowed: res.status >= 500 || res.status === 429,
    };
  }
}

export function isRetryAllowed(data) {
  return Boolean(data?.retry_allowed);
}

/** True when backend completed registration (handles 200 + success or legacy shape). */
export function isRegistrationSuccess(data, res) {
  if (!res?.ok) return false;
  if (data?.success === true) return true;
  if (data?.error) return false;
  return Boolean(data?.face_label || data?.email || data?.message);
}

export function parseRegisterFailureMessage(res, data) {
  if (res?.status === 404 || data?.detail === "Not Found") {
    return (
      "Register API not found. Start the backend (uvicorn on port 8000) " +
      "or set VITE_API_URL for production."
    );
  }
  if (res?.status >= 500) {
    return parseApiDetail(data, "Server error during registration. Please try again shortly.");
  }
  return parseApiDetail(data, "Registration failed. Please try again.");
}
