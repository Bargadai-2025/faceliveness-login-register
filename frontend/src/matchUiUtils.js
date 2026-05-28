const LAPTOP_DEVICE_RE = /laptop|macbook/i;

/** YOLO often labels the user's own machine during laptop webcam capture — not a replay attack. */
export function isLaptopOnlyDevicesDetected(devices = []) {
  const list = Array.isArray(devices) ? devices : [];
  if (!list.length) return false;
  return list.every((name) => LAPTOP_DEVICE_RE.test(String(name)));
}

/**
 * In-stream or frame response: treat as phone/screen replay device alert (not ambient laptop).
 */
export function isReplayDeviceAlert(data) {
  if (!data?.is_suspicious) return false;
  if (data.laptop_capture_context === true) return false;
  const devices = Array.isArray(data.devices_detected) ? data.devices_detected : [];
  if (isLaptopOnlyDevicesDetected(devices)) return false;

  const deviceNames = devices.join(", ").toLowerCase();
  const deviceDetail = String(data.detail || "").toLowerCase();

  if (
    data.display_attack &&
    (deviceNames ||
      /photograph|digital screen|phone|tablet|replay|electronic device/i.test(deviceDetail))
  ) {
    return true;
  }
  if (deviceNames && !isLaptopOnlyDevicesDetected(devices)) return true;
  if (
    deviceDetail.includes("phone") ||
    deviceDetail.includes("tablet") ||
    deviceDetail.includes("television") ||
    deviceDetail.includes("mobile phone")
  ) {
    return true;
  }
  if (
    deviceDetail.includes("electronic device") &&
    !deviceDetail.includes("laptop") &&
    !deviceDetail.includes("macbook")
  ) {
    return true;
  }
  return false;
}

/** Max wait for POST /match before abort (slow CPU / large DB / dead API). */
export const MATCH_REQUEST_TIMEOUT_MS = 180000;

/** Registration includes face detect + Cloudinary + DB — allow longer than default fetch. */
export const REGISTER_REQUEST_TIMEOUT_MS = 120000;

/**
 * Progress that eases toward ~92% while waiting (avoids fake "stuck at 90%" UX).
 * @param {(n: number | ((prev: number) => number)) => void} setProgress
 * @returns {ReturnType<typeof setInterval>}
 */
export function startIndeterminateMatchProgress(setProgress) {
  const start = Date.now();
  return setInterval(() => {
    const elapsed = (Date.now() - start) / 1000;
    const target = 92 * (1 - Math.exp(-elapsed / 12));
    setProgress((p) => Math.max(p, Math.min(92, target)));
  }, 400);
}

export function matchFetchErrorMessage(err) {
  if (err && err.name === "AbortError") {
    return "Matching timed out. Run the API from the project root: uvicorn main:app --reload --host 127.0.0.1 --port 8000. For faster local matching, set FAST_MATCH=1 in the backend .env file.";
  }
  return "Match request failed. Check that the API is running and try again.";
}

export function registerFetchErrorMessage(err) {
  if (err?.name === "AbortError") {
    return (
      "Registration timed out. If you clicked Register once, wait a moment and check login — " +
      "the account may already exist. Do not register again with the same email."
    );
  }
  if (err instanceof TypeError && /failed to fetch|networkerror|load failed/i.test(String(err.message))) {
    const base = getApiBaseHint();
    return `Cannot reach the registration server. ${base}`;
  }
  return "Registration request failed. Check that the API is running and try again.";
}

function getApiBaseHint() {
  try {
    const v = import.meta.env.VITE_API_URL;
    if (v === "" || v === "proxy") {
      return "Start the backend: cd backend && uvicorn main:app --reload --host 127.0.0.1 --port 8000";
    }
    if (v) return `Check VITE_API_URL (${String(v).trim()}).`;
  } catch {
    /* ignore */
  }
  return "Start the backend or check VITE_API_URL in production.";
}
