/** Max wait for POST /match before abort (slow CPU / large DB / dead API). */
export const MATCH_REQUEST_TIMEOUT_MS = 180000;

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
