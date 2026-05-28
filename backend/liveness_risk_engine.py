"""
Multi-factor liveness / anti-spoof risk fusion.

Design goals:
- No single-signal hard reject (except device physically overlapping face).
- Combine device, PAD, display imaging, identity continuity, session history.
- Environmental glare/sunlight/tube light contributes little unless display corroboration exists.
- Output tiered verdict: pass | pass_penalty | reject
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from spoof_scoring import (
    count_display_imaging_signals,
    has_display_attack_corroboration,
)
from device_filter import filter_devices_for_attack, is_phone_tablet_name
from screen_replay_analysis import has_physical_fullframe_signals, has_structural_fullframe_signals


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


RISK_REJECT_THRESHOLD = _f("RISK_REJECT_THRESHOLD", 72.0)
RISK_PENALTY_THRESHOLD = _f("RISK_PENALTY_THRESHOLD", 48.0)
RISK_CONFIRM_FACTORS_MIN = int(_f("RISK_CONFIRM_FACTORS_MIN", 2))
RISK_FACTOR_HIGH = _f("RISK_FACTOR_HIGH", 58.0)

WEIGHT_DEVICE = _f("RISK_WEIGHT_DEVICE", 0.28)
WEIGHT_PAD = _f("RISK_WEIGHT_PAD", 0.26)
WEIGHT_DISPLAY = _f("RISK_WEIGHT_DISPLAY", 0.22)
WEIGHT_IDENTITY = _f("RISK_WEIGHT_IDENTITY", 0.14)
WEIGHT_SESSION = _f("RISK_WEIGHT_SESSION", 0.10)


@dataclass
class RiskFactors:
    device_risk: float = 0.0
    pad_spoof_risk: float = 0.0
    display_imaging_risk: float = 0.0
    identity_continuity_risk: float = 0.0
    session_risk: float = 0.0
    ambient_only_risk: float = 0.0
    hard_device_overlap: bool = False
    factor_notes: List[str] = field(default_factory=list)

    def high_factor_count(self, threshold: float = RISK_FACTOR_HIGH) -> int:
        vals = [
            self.device_risk,
            self.pad_spoof_risk,
            self.display_imaging_risk,
            self.identity_continuity_risk,
            self.session_risk,
        ]
        return sum(1 for v in vals if v >= threshold)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_risk": round(self.device_risk, 2),
            "pad_spoof_risk": round(self.pad_spoof_risk, 2),
            "display_imaging_risk": round(self.display_imaging_risk, 2),
            "identity_continuity_risk": round(self.identity_continuity_risk, 2),
            "session_risk": round(self.session_risk, 2),
            "ambient_only_risk": round(self.ambient_only_risk, 2),
            "hard_device_overlap": self.hard_device_overlap,
            "notes": list(self.factor_notes),
        }


def compute_device_risk(
    device_replay_score: float,
    *,
    hard_overlap: bool = False,
    devices_found: Optional[List[str]] = None,
    match_selfie: bool = False,
    frame_bezel_score: float = 0.0,
) -> Tuple[float, List[str]]:
    notes: List[str] = []
    if hard_overlap:
        return 100.0, ["device_hard_overlap_face"]
    risk = min(100.0, device_replay_score * 100.0)
    if devices_found and risk < 35.0:
        risk = 35.0
        notes.append(f"devices_in_frame:{','.join(devices_found)}")
    if match_selfie and devices_found:
        if device_replay_score >= 0.12:
            risk = max(risk, 52.0)
        if device_replay_score >= 0.22:
            risk = max(risk, 68.0)
        notes.append("match_device_in_frame_boost")
    if match_selfie and frame_bezel_score >= 0.40 and risk < 48.0:
        risk = max(risk, 48.0)
        notes.append(f"match_bezel_boost={frame_bezel_score:.2f}")
    return risk, notes


def compute_pad_risk(
    spoof_detail: Dict[str, Any],
    *,
    match_context: Optional[Dict[str, Any]] = None,
    device_replay_score: float = 0.0,
    devices_found: Optional[List[str]] = None,
) -> Tuple[float, List[str]]:
    """Map PAD 0–100 spoof score to risk; ambient-only capped low."""
    score = float(spoof_detail.get("total_spoof_score", 0.0))
    per_signal = spoof_detail.get("confidence_per_signal") or {}
    triggered = list(spoof_detail.get("triggered_rules") or [])
    device_v = float(per_signal.get("device_replay", device_replay_score))
    display_corr = has_display_attack_corroboration(
        per_signal,
        triggered,
        device_replay_score=device_v,
        devices_found=devices_found,
        match_context=match_context,
    )
    notes: List[str] = []
    if not display_corr and score > 40.0:
        score = min(score, 38.0)
        notes.append("pad_capped_ambient_only")
    return min(100.0, score), notes


def compute_display_imaging_risk(
    spoof_detail: Dict[str, Any],
    *,
    device_replay_score: float = 0.0,
    devices_found: Optional[List[str]] = None,
    match_context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[str]]:
    per_signal = spoof_detail.get("confidence_per_signal") or {}
    triggered = list(spoof_detail.get("triggered_rules") or [])
    notes: List[str] = []

    if not has_display_attack_corroboration(
        per_signal,
        triggered,
        device_replay_score=device_replay_score,
        devices_found=devices_found,
        match_context=match_context,
    ):
        return 0.0, ["no_display_corroboration"]

    hi = float(per_signal.get("high_brightness_screen", 0.0)) * 100.0
    moire = float(per_signal.get("moire", 0.0)) * 100.0
    grid = float(per_signal.get("pixel_grid", 0.0)) * 100.0
    tex = float(per_signal.get("texture_degraded", 0.0)) * 100.0
    border = float(per_signal.get("screen_border", 0.0)) * 100.0

    imaging_count = count_display_imaging_signals(per_signal, threshold=0.30)
    fusion = 0.30 * hi + 0.22 * moire + 0.18 * grid + 0.15 * tex + 0.15 * border
    if imaging_count >= 2:
        fusion += 12.0 * (imaging_count - 1)
    if imaging_count >= 3:
        notes.append(f"display_imaging_count={imaging_count}")

    mc = match_context or {}
    if mc.get("match_selfie"):
        bezel = float(mc.get("frame_bezel", 0.0))
        if bezel >= 0.38:
            fusion = min(100.0, fusion + 12.0 + bezel * 15.0)
            notes.append("match_bezel_imaging_boost")
        ff_risk, ff_notes = compute_fullframe_replay_risk(mc)
        if ff_risk > 0:
            fusion = max(fusion, ff_risk)
            notes.extend(ff_notes)

    return float(min(100.0, fusion)), notes


def compute_session_risk(
    errcount: int = 0,
    stream_risk_ema: float = 0.0,
) -> Tuple[float, List[str]]:
    """Liveness-session suspicion accumulated during stream (not per-frame glare)."""
    notes: List[str] = []
    err_risk = min(100.0, (float(errcount) / 50.0) * 100.0)
    stream_risk = min(100.0, float(stream_risk_ema))
    combined = max(err_risk, stream_risk)
    if err_risk > 20:
        notes.append(f"errcount={errcount}")
    if stream_risk > 20:
        notes.append(f"stream_ema={stream_risk_ema:.1f}")
    return combined, notes


def compute_identity_continuity_risk(assessment: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    From challenge_frame_verification.assess_challenge_continuity() scores (0–1 similarities).
    Higher risk = lower similarity / more drift.
    """
    if assessment.get("skipped"):
        return 0.0, ["no_challenge_snapshots"]

    notes: List[str] = []
    risk_parts: List[float] = []

    min_sim = assessment.get("min_challenge_selfie_sim")
    if min_sim is not None:
        sim = float(min_sim)
        part = max(0.0, min(100.0, (1.0 - (sim - 0.45) / 0.35) * 100.0)) if sim < 0.80 else 0.0
        if sim < 0.52:
            part = min(100.0, part + 40.0)
        risk_parts.append(part)
        notes.append(f"identity_sim={sim:.3f}")

    min_bg = assessment.get("min_bg_sim")
    if min_bg is not None:
        bg = float(min_bg)
        bg_part = max(0.0, min(100.0, (1.0 - (bg - 0.35) / 0.35) * 100.0)) if bg < 0.70 else 0.0
        risk_parts.append(bg_part * 0.55)
        notes.append(f"bg_sim={bg:.3f}")

    ratio_drift = assessment.get("face_ratio_drift")
    if ratio_drift is not None:
        drift = float(ratio_drift)
        if drift > 0.35:
            risk_parts.append(min(100.0, drift * 120.0))
            notes.append(f"ratio_drift={drift:.3f}")

    if assessment.get("device_only_at_capture") and not assessment.get("laptop_capture_context"):
        if not assessment.get("liveness_session_verified"):
            risk_parts.append(75.0)
            notes.append("device_only_at_capture")
        else:
            notes.append("device_only_at_capture_ignored_liveness_ok")

    if assessment.get("no_face_in_challenge_crop"):
        risk_parts.append(90.0)
        notes.append("no_face_in_challenge_crop")

    if not risk_parts:
        return 0.0, notes
    return float(min(100.0, max(risk_parts))), notes


def fuse_composite_risk(factors: RiskFactors) -> float:
    """Weighted fusion — never one factor alone decides outcome."""
    composite = (
        WEIGHT_DEVICE * factors.device_risk
        + WEIGHT_PAD * factors.pad_spoof_risk
        + WEIGHT_DISPLAY * factors.display_imaging_risk
        + WEIGHT_IDENTITY * factors.identity_continuity_risk
        + WEIGHT_SESSION * factors.session_risk
        + 0.05 * factors.ambient_only_risk
    )
    if factors.display_imaging_risk >= RISK_FACTOR_HIGH and factors.device_risk >= 40.0:
        composite = min(100.0, composite + 8.0)
    if factors.pad_spoof_risk >= RISK_FACTOR_HIGH and factors.display_imaging_risk >= 40.0:
        composite = min(100.0, composite + 6.0)
    return float(min(100.0, composite))


_DIGITAL_SCREEN_USER_MESSAGE = (
    "Security Alert: Digital screen or photo replay detected. "
    "Do not use a photograph or digital screen."
)


def compute_fullframe_replay_risk(match_context: Optional[Dict[str, Any]]) -> Tuple[float, List[str]]:
    """Risk from PNG-normalized multi-region full-frame screen analysis."""
    mc = match_context or {}
    notes: List[str] = []
    score = float(mc.get("fullframe_replay_score", 0.0))
    if score <= 0.01:
        return 0.0, notes
    signals = mc.get("fullframe_signals") or []
    liveness_ok = bool(mc.get("liveness_session_verified"))
    has_physical = has_physical_fullframe_signals(signals)
    has_struct = has_structural_fullframe_signals(signals)
    bezel = float(mc.get("frame_bezel", 0.0))
    border = float(mc.get("frame_screen_border", 0.0))
    has_physical = has_physical or bezel >= 0.45 or border >= 0.42

    if liveness_ok and not has_physical:
        return min(22.0, score * 28.0), ["fullframe_artifact_only_liveness_ok"]

    if liveness_ok and not has_struct:
        return min(28.0, score * 40.0), ["fullframe_soft_only_liveness_ok"]

    risk = min(100.0, score * 88.0)
    n_sig = int(mc.get("fullframe_signal_count", 0))
    if has_struct and n_sig >= 2:
        risk = min(100.0, risk + 6.0 * (n_sig - 1))
        notes.append(f"fullframe_struct_signals={n_sig}")
    if mc.get("fullframe_replay_flag") and has_physical:
        risk = max(risk, 68.0)
        notes.append("fullframe_replay_flag")
    return risk, notes


def _is_match_screen_replay(
    factors: RiskFactors,
    *,
    devices_found: Optional[List[str]] = None,
    frame_bezel_score: float = 0.0,
    match_context: Optional[Dict[str, Any]] = None,
    identity_assessment: Optional[Dict[str, Any]] = None,
) -> bool:
    """Phone/screen replay at selfie — multi-cue; trust completed liveness for live webcam."""
    mc = match_context or {}
    if not mc.get("match_selfie"):
        return False
    assessment = identity_assessment or {}
    liveness_ok = bool(mc.get("liveness_session_verified") or assessment.get("liveness_session_verified"))
    replay_devices = filter_devices_for_attack(
        devices_found, hard_overlap=factors.hard_device_overlap
    )
    ff_signals = mc.get("fullframe_signals") or []
    has_physical_ff = has_physical_fullframe_signals(ff_signals)
    has_struct_ff = has_structural_fullframe_signals(ff_signals)
    bezel_ff = float(mc.get("frame_bezel", frame_bezel_score))
    border_ff = float(mc.get("frame_screen_border", 0.0))
    has_physical_ff = has_physical_ff or bezel_ff >= 0.45 or border_ff >= 0.42

    if factors.hard_device_overlap and replay_devices:
        return True

    if liveness_ok and not factors.hard_device_overlap:
        if mc.get("fullframe_replay_flag") and has_physical_ff:
            return True
        if bezel_ff >= 0.45 or border_ff >= 0.42 or frame_bezel_score >= 0.45:
            return True
        phone_devices = [d for d in (replay_devices or []) if is_phone_tablet_name(d)]
        if phone_devices and factors.device_risk >= 62.0:
            return True
        return False

    if mc.get("fullframe_replay_flag") and has_physical_ff:
        return True
    ff = float(mc.get("fullframe_replay_score", 0.0))
    ff_sig = int(mc.get("fullframe_signal_count", 0))
    if has_physical_ff and ff >= 0.55:
        return True
    if has_physical_ff and ff >= 0.48 and ff_sig >= 2:
        return True
    if assessment.get("device_only_at_capture") and not assessment.get("laptop_capture_context"):
        return True
    if frame_bezel_score >= 0.40:
        return True
    if replay_devices and (
        factors.display_imaging_risk >= 38.0
        or factors.pad_spoof_risk >= 55.0
        or factors.device_risk >= 55.0
    ):
        return True
    if frame_bezel_score >= 0.42 and (
        factors.display_imaging_risk >= 28.0 or factors.pad_spoof_risk >= 48.0
    ):
        return True
    if factors.device_risk >= 52.0 and factors.display_imaging_risk >= 28.0:
        return True
    if factors.device_risk >= 48.0 and factors.pad_spoof_risk >= 52.0:
        return True
    refl = str(mc.get("reflection_label") or "")
    if refl in ("phone_screen_reflection", "monitor_glare", "rectangular_source"):
        if factors.display_imaging_risk >= 28.0 or factors.pad_spoof_risk >= 48.0:
            return True
    if float(mc.get("frame_bezel", 0.0)) >= 0.40:
        return True
    return False


def _prefer_digital_over_identity_mismatch(
    factors: RiskFactors,
    *,
    screen_replay: bool,
    devices_found: Optional[List[str]] = None,
    frame_bezel_score: float = 0.0,
    identity_assessment: Optional[Dict[str, Any]] = None,
) -> bool:
    """Screen replay often lowers challenge↔selfie similarity — show digital-screen error, not identity."""
    assessment = identity_assessment or {}
    liveness_ok = bool(assessment.get("liveness_session_verified"))
    replay_devices = filter_devices_for_attack(devices_found, hard_overlap=False)
    if liveness_ok and not replay_devices and frame_bezel_score < 0.42:
        return False
    if screen_replay:
        return True
    if assessment.get("device_only_at_capture") and not assessment.get("laptop_capture_context"):
        if not liveness_ok:
            return True
    if replay_devices and factors.device_risk >= 45.0:
        return True
    if frame_bezel_score >= 0.38:
        return True
    if factors.device_risk >= 48.0:
        return True
    if factors.display_imaging_risk >= 32.0 and factors.pad_spoof_risk >= 45.0:
        return True
    return False


def decide_security_verdict(
    factors: RiskFactors,
    composite_risk: float,
    *,
    screen_replay: bool = False,
    devices_found: Optional[List[str]] = None,
    frame_bezel_score: float = 0.0,
    identity_assessment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tiered verdict — reject only on high composite + multi-factor confirmation,
    or hard device overlap.
    """
    if factors.hard_device_overlap:
        return {
            "verdict": "reject",
            "composite_risk": round(composite_risk, 2),
            "reason": "Electronic device overlapping face",
            "user_message": (
                "Security Alert: A phone or screen is blocking your face. "
                "Remove the device and take a direct selfie."
            ),
            "digital_media": True,
            "user_mismatch": False,
        }

    if screen_replay:
        return {
            "verdict": "reject",
            "composite_risk": round(composite_risk, 2),
            "reason": "Phone or digital screen replay at capture",
            "user_message": _DIGITAL_SCREEN_USER_MESSAGE,
            "digital_media": True,
            "user_mismatch": False,
        }

    high_count = factors.high_factor_count()
    assessment = identity_assessment or {}
    liveness_ok = bool(assessment.get("liveness_session_verified"))
    replay_devices = filter_devices_for_attack(devices_found, hard_overlap=False)
    if composite_risk >= RISK_REJECT_THRESHOLD and high_count >= RISK_CONFIRM_FACTORS_MIN:
        if liveness_ok and not replay_devices and not factors.hard_device_overlap:
            return {
                "verdict": "pass_penalty",
                "composite_risk": round(composite_risk, 2),
                "reason": f"Elevated risk {composite_risk:.0f} — liveness verified, penalty only",
                "user_message": None,
                "digital_media": False,
                "user_mismatch": False,
                "confidence_penalty": round((composite_risk - RISK_PENALTY_THRESHOLD) * 0.002, 4),
            }
        return {
            "verdict": "reject",
            "composite_risk": round(composite_risk, 2),
            "reason": f"Multi-factor risk {composite_risk:.0f} ({high_count} high factors)",
            "user_message": _DIGITAL_SCREEN_USER_MESSAGE,
            "digital_media": True,
            "user_mismatch": False,
        }

    if factors.identity_continuity_risk >= 85.0 and composite_risk >= RISK_PENALTY_THRESHOLD:
        if _prefer_digital_over_identity_mismatch(
            factors,
            screen_replay=screen_replay,
            devices_found=devices_found,
            frame_bezel_score=frame_bezel_score,
            identity_assessment=identity_assessment,
        ):
            return {
                "verdict": "reject",
                "composite_risk": round(composite_risk, 2),
                "reason": "Screen replay (identity continuity artifact)",
                "user_message": _DIGITAL_SCREEN_USER_MESSAGE,
                "digital_media": True,
                "user_mismatch": False,
            }
        return {
            "verdict": "reject",
            "composite_risk": round(composite_risk, 2),
            "reason": "Identity continuity failed",
            "user_message": "User Identity Mismatch",
            "digital_media": False,
            "user_mismatch": True,
        }

    if composite_risk >= RISK_PENALTY_THRESHOLD:
        return {
            "verdict": "pass_penalty",
            "composite_risk": round(composite_risk, 2),
            "reason": f"Elevated risk {composite_risk:.0f} — match allowed with penalty",
            "user_message": None,
            "digital_media": False,
            "user_mismatch": False,
            "confidence_penalty": round((composite_risk - RISK_PENALTY_THRESHOLD) * 0.002, 4),
        }

    return {
        "verdict": "pass",
        "composite_risk": round(composite_risk, 2),
        "reason": "Risk within acceptable range",
        "user_message": None,
        "digital_media": False,
        "user_mismatch": False,
    }


def evaluate_match_security(
    *,
    spoof_detail: Dict[str, Any],
    device_replay_score: float,
    devices_found: Optional[List[str]],
    device_hard_overlap: bool,
    identity_assessment: Dict[str, Any],
    errcount: int = 0,
    stream_risk_ema: float = 0.0,
    match_context: Optional[Dict[str, Any]] = None,
    frame_bezel_score: float = 0.0,
) -> Dict[str, Any]:
    """Full post-selfie multi-factor evaluation."""
    factors = RiskFactors(hard_device_overlap=device_hard_overlap)
    mc = match_context or {}
    match_selfie = bool(mc.get("match_selfie"))

    replay_devices = filter_devices_for_attack(
        devices_found,
        hard_overlap=device_hard_overlap,
        device_replay_score=device_replay_score,
    )
    dr, n1 = compute_device_risk(
        device_replay_score,
        hard_overlap=device_hard_overlap,
        devices_found=replay_devices or None,
        match_selfie=match_selfie,
        frame_bezel_score=frame_bezel_score,
    )
    factors.device_risk = dr
    factors.factor_notes.extend(n1)

    pad, n2 = compute_pad_risk(
        spoof_detail,
        match_context=match_context,
        device_replay_score=device_replay_score,
        devices_found=replay_devices or None,
    )
    factors.pad_spoof_risk = pad
    factors.factor_notes.extend(n2)

    disp, n3 = compute_display_imaging_risk(
        spoof_detail,
        device_replay_score=device_replay_score,
        devices_found=replay_devices or None,
        match_context=match_context,
    )
    factors.display_imaging_risk = disp
    factors.factor_notes.extend(n3)

    ff_only, n_ff = compute_fullframe_replay_risk(match_context)
    if ff_only > factors.display_imaging_risk:
        factors.display_imaging_risk = ff_only
        factors.factor_notes.extend(n_ff)

    ident, n4 = compute_identity_continuity_risk(identity_assessment)
    factors.identity_continuity_risk = ident
    factors.factor_notes.extend(n4)

    sess, n5 = compute_session_risk(errcount, stream_risk_ema)
    factors.session_risk = sess
    factors.factor_notes.extend(n5)

    composite = fuse_composite_risk(factors)
    identity_with_laptop = dict(identity_assessment)
    if match_context.get("laptop_capture_context"):
        identity_with_laptop["laptop_capture_context"] = True

    screen_replay = _is_match_screen_replay(
        factors,
        devices_found=replay_devices,
        frame_bezel_score=frame_bezel_score,
        match_context=match_context,
        identity_assessment=identity_with_laptop,
    )
    decision = decide_security_verdict(
        factors,
        composite,
        screen_replay=screen_replay,
        devices_found=replay_devices,
        frame_bezel_score=frame_bezel_score,
        identity_assessment=identity_with_laptop,
    )
    decision["screen_replay"] = screen_replay
    decision["risk_factors"] = factors.to_dict()
    decision["high_factor_count"] = factors.high_factor_count()
    return decision
