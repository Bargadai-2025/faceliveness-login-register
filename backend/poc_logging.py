"""
Structured JSON logging for PoC / pilot (security events, rejects, passes).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOGGER_NAME = "faceliveness"


def _configure_root() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


_logger = _configure_root()


def log_event(
    event: str,
    *,
    level: str = "info",
    session_id: Optional[str] = None,
    device_id: Optional[str] = None,
    verdict: Optional[str] = None,
    composite_risk: Optional[float] = None,
    error_code: Optional[str] = None,
    retry_allowed: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "service": "faceliveness-api",
    }
    if session_id:
        payload["session_id"] = session_id
    if device_id:
        payload["device_id"] = device_id[:64] if device_id else None
    if verdict is not None:
        payload["verdict"] = verdict
    if composite_risk is not None:
        payload["composite_risk"] = round(float(composite_risk), 2)
    if error_code:
        payload["error_code"] = error_code
    if retry_allowed is not None:
        payload["retry_allowed"] = bool(retry_allowed)
    if extra:
        payload["extra"] = extra
    line = json.dumps(payload, default=str)
    lvl = getattr(_logger, level.lower(), _logger.info)
    lvl(line)


def log_security_verdict(
    *,
    phase: str,
    verdict: str,
    session_id: Optional[str] = None,
    device_id: Optional[str] = None,
    composite_risk: Optional[float] = None,
    reason: Optional[str] = None,
    retry_allowed: bool = False,
    risk_factors: Optional[Dict[str, Any]] = None,
) -> None:
    log_event(
        f"security_{phase}_{verdict}",
        level="warning" if verdict == "reject" else "info",
        session_id=session_id,
        device_id=device_id,
        verdict=verdict,
        composite_risk=composite_risk,
        retry_allowed=retry_allowed,
        extra={
            "reason": (reason or "")[:200],
            "risk_factors": risk_factors,
        },
    )
