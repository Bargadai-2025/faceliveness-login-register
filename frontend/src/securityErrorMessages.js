/**
 * Official 23 security error messages (Sr No 1–23).
 * Only these strings may appear in the SECURITY ALERT card.
 */
export const SECURITY_ERROR_LIST = [
  "Multiple Users Detected",
  "Do not use Photograph or Digital Screen",
  "Camera Access Blocked",
  "No Face Detected",
  "Face Too Close",
  "Move Back Slightly",
  "Move Closer Now",
  "Camera Access Denied",
  "Session Start Failed",
  "Capture Failed Retry",
  "Complete Liveness First",
  "Upload Selfie First",
  "Verification Failed Retry",
  "Match Timed Out",
  "Match Request Failed",
  "Registration Failed Retry",
  "No Match Found",
  "Match Too Ambiguous",
  "User Identity Mismatch",
  "Invalid Image File",
  "Server Error Occurred",
  "Liveness Check Failed",
  "Security Alert Detected",
];

export const ERROR_LABELS = {
  MULTI_PERSON: SECURITY_ERROR_LIST[0],
  DIGITAL_MEDIA: SECURITY_ERROR_LIST[1],
  CAMERA_BLOCKED: SECURITY_ERROR_LIST[2],
  NO_FACE: SECURITY_ERROR_LIST[3],
  FACE_TOO_CLOSE: SECURITY_ERROR_LIST[4],
  MOVE_BACK: SECURITY_ERROR_LIST[5],
  MOVE_CLOSER: SECURITY_ERROR_LIST[6],
  CAMERA_DENIED: SECURITY_ERROR_LIST[7],
  SESSION_FAILED: SECURITY_ERROR_LIST[8],
  CAPTURE_FAILED: SECURITY_ERROR_LIST[9],
  LIVENESS_INCOMPLETE: SECURITY_ERROR_LIST[10],
  UPLOAD_SELFIE: SECURITY_ERROR_LIST[11],
  VERIFICATION_FAILED: SECURITY_ERROR_LIST[12],
  MATCH_TIMEOUT: SECURITY_ERROR_LIST[13],
  MATCH_FAILED: SECURITY_ERROR_LIST[14],
  REGISTRATION_FAILED: SECURITY_ERROR_LIST[15],
  NO_MATCH: SECURITY_ERROR_LIST[16],
  MATCH_AMBIGUOUS: SECURITY_ERROR_LIST[17],
  USER_MISMATCH: SECURITY_ERROR_LIST[18],
  INVALID_IMAGE: SECURITY_ERROR_LIST[19],
  SERVER_ERROR: SECURITY_ERROR_LIST[20],
  LIVENESS_FAILED: SECURITY_ERROR_LIST[21],
  SECURITY_ALERT: SECURITY_ERROR_LIST[22],
};

const ALLOWED_SET = new Set(SECURITY_ERROR_LIST);

const DIGITAL_KEYWORDS = [
  "photograph",
  "photo",
  "digital screen",
  "screen",
  "phone",
  "tablet",
  "laptop",
  "mobile",
  "device",
  "electronic",
  "spoof",
  "replay",
  "presentation-attack",
  "non-live",
  "reflection",
  "weighted spoof",
  "suspicious",
  "matching blocked",
];

export function isDigitalMediaMessage(text = "") {
  const m = String(text).toLowerCase();
  return DIGITAL_KEYWORDS.some((k) => m.includes(k));
}

const RULES = [
  [/camera blocked|blocked.*camera|uncover camera/, ERROR_LABELS.CAMERA_BLOCKED],
  [/multiple (user|people|person)/, ERROR_LABELS.MULTI_PERSON],
  [/no face/, ERROR_LABELS.NO_FACE],
  [/\bface too close\b/, ERROR_LABELS.FACE_TOO_CLOSE],
  [/too far/, ERROR_LABELS.MOVE_BACK],
  [/move back/, ERROR_LABELS.MOVE_BACK],
  [/move slightly closer|move closer/, ERROR_LABELS.MOVE_CLOSER],
  [/camera access denied|access denied/, ERROR_LABELS.CAMERA_DENIED],
  [/session failed/, ERROR_LABELS.SESSION_FAILED],
  [/capture failed/, ERROR_LABELS.CAPTURE_FAILED],
  [
    /liveness verification must|complete liveness|liveness session|expired liveness|start the camera/,
    ERROR_LABELS.LIVENESS_INCOMPLETE,
  ],
  [/upload.*selfie|take a selfie|take a selfie first/, ERROR_LABELS.UPLOAD_SELFIE],
  [/verification failed/, ERROR_LABELS.VERIFICATION_FAILED],
  [/timed out/, ERROR_LABELS.MATCH_TIMEOUT],
  [/match request failed/, ERROR_LABELS.MATCH_FAILED],
  [/registration failed/, ERROR_LABELS.REGISTRATION_FAILED],
  [/no confident match|no match found/, ERROR_LABELS.NO_MATCH],
  [/ambiguous/, ERROR_LABELS.MATCH_AMBIGUOUS],
  [
    /does not match your registered|not your registered face|only your registered account|logged in as|wrong account|face does not match/i,
    ERROR_LABELS.USER_MISMATCH,
  ],
  [/no registered face found|please complete registration first|register first/i, ERROR_LABELS.VERIFICATION_FAILED],
  [/identity mismatch|totally different|different\./, ERROR_LABELS.USER_MISMATCH],
  [/could not read|could not decode|invalid image/, ERROR_LABELS.INVALID_IMAGE],
  [/server error/, ERROR_LABELS.SERVER_ERROR],
  [/liveness check failed|liveness failed/, ERROR_LABELS.LIVENESS_FAILED],
  [/high risk of digital/, ERROR_LABELS.DIGITAL_MEDIA],
];

/**
 * Map raw text to one of the 23 allowed labels, or null if nothing applies.
 */
export function formatSecurityError(message, hints = {}) {
  if (hints.multiPerson) return ERROR_LABELS.MULTI_PERSON;
  if (hints.digitalMedia || isDigitalMediaMessage(message)) {
    return ERROR_LABELS.DIGITAL_MEDIA;
  }

  const m = String(message || "").toLowerCase().trim();
  if (!m) return null;

  for (const [pattern, label] of RULES) {
    if (pattern.test(m)) return label;
  }

  if (isDigitalMediaMessage(m)) return ERROR_LABELS.DIGITAL_MEDIA;

  return null;
}

/**
 * Always returns one of the 23 official labels (never null).
 */
export function resolveSecurityErrorLabel(message, hints = {}) {
  const mapped = formatSecurityError(message, hints);
  if (mapped) return mapped;
  if (hints.userMismatch) return ERROR_LABELS.USER_MISMATCH;
  return ERROR_LABELS.VERIFICATION_FAILED;
}

/** True only for the official 23 messages. */
export function isAllowedSecurityError(value) {
  return Boolean(value && ALLOWED_SET.has(value));
}

/**
 * Resolve display message: must be one of the 23, otherwise null (no UI error).
 */
export function resolveSecurityDisplayError(error, { multiPerson = false } = {}) {
  if (multiPerson) return ERROR_LABELS.MULTI_PERSON;
  if (!error) return null;
  if (isAllowedSecurityError(error)) return error;
  const mapped = formatSecurityError(error);
  if (isAllowedSecurityError(mapped)) return mapped;
  return resolveSecurityErrorLabel(error);
}
