"""PoC helpers: retry policy, response enrichment."""
from __future__ import annotations

from typing import Any, Dict, Optional

from liveness_risk_engine import RISK_FACTOR_HIGH, RISK_REJECT_THRESHOLD


def count_high_factors(risk_factors: Optional[Dict[str, Any]]) -> int:
    if not risk_factors:
        return 0
    keys = (
        "device_risk",
        "pad_spoof_risk",
        "display_imaging_risk",
        "identity_continuity_risk",
        "session_risk",
    )
    return sum(1 for k in keys if float(risk_factors.get(k) or 0) >= RISK_FACTOR_HIGH)


def match_retry_allowed(
    *,
    security_verdict: str,
    composite_risk: float,
    risk_factors: Optional[Dict[str, Any]] = None,
    identity_assessment: Optional[Dict[str, Any]] = None,
    screen_replay: bool = False,
) -> bool:
    """Allow one or two selfie retries before hard UX failure (banking PoC)."""
    if security_verdict != "reject":
        return False
    if screen_replay:
        return False
    factors = risk_factors or {}
    if factors.get("hard_device_overlap"):
        return False
    ident = float(factors.get("identity_continuity_risk") or 0)
    if ident >= 85.0:
        return False
    if identity_assessment and identity_assessment.get("no_face_in_challenge_crop"):
        return False
    highs = count_high_factors(factors)
    if highs >= 2 and composite_risk >= RISK_REJECT_THRESHOLD:
        return False
    if composite_risk >= RISK_REJECT_THRESHOLD + 12:
        return False
    return True


def enrich_match_security_payload(payload: Dict[str, Any], **ctx) -> Dict[str, Any]:
    verdict = payload.get("security_verdict", "reject")
    composite = float(payload.get("composite_risk") or 0)
    factors = payload.get("risk_factors")
    retry = match_retry_allowed(
        security_verdict=verdict,
        composite_risk=composite,
        risk_factors=factors if isinstance(factors, dict) else None,
        identity_assessment=ctx.get("identity_assessment"),
        screen_replay=bool(ctx.get("screen_replay")),
    )
    payload["retry_allowed"] = retry
    if verdict == "reject" and retry:
        payload["user_message"] = (
            payload.get("error")
            or "Verification needs another try. Face the camera directly in steady lighting."
        )
    return payload
