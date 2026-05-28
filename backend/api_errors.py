"""
User-safe API errors — never expose raw exceptions or stack traces to clients.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Stable codes for frontend / audit (not raw Python messages)
USER_MESSAGES: Dict[str, str] = {
    "SESSION_INVALID": "Liveness session expired. Please start the camera flow again.",
    "SESSION_INCOMPLETE": "Liveness verification incomplete. Please complete all challenges.",
    "EMPTY_FRAME": "Camera frame was empty. Please try again.",
    "FACE_NOT_FOUND": "No face detected. Please face the camera in good lighting.",
    "IMAGE_READ_FAILED": "Could not process the image. Please capture again.",
    "MATCH_SESSION_REQUIRED": "Liveness session is required. Please start the camera flow again.",
    "SERVER_ERROR": "Server Error Occurred",
    "RATE_LIMITED": "Too many requests. Please wait a moment and try again.",
    "DEVICE_INVALID": "Invalid device identifier.",
    "LIVENESS_START_FAILED": "Session Start Failed",
}


def user_error(
    code: str,
    *,
    retry_allowed: bool = False,
    http_status: int = 400,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    msg = USER_MESSAGES.get(code, USER_MESSAGES["SERVER_ERROR"])
    out: Dict[str, Any] = {
        "error": msg,
        "error_code": code,
        "user_message": msg,
        "retry_allowed": bool(retry_allowed),
    }
    if extra:
        out.update(extra)
    out["_http_status"] = http_status
    return out


def public_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal fields before JSON response body."""
    d = dict(payload)
    d.pop("_http_status", None)
    return d


def safe_exception_message(exc: BaseException) -> str:
    """Log internally; return only generic user text."""
    return USER_MESSAGES["SERVER_ERROR"]
