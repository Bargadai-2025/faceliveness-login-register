"""
Post-selfie security checks (Option A): tune existing YOLO + spoof + light heuristics.

Used by POST /match after face detection. All thresholds are env-configurable.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_detection import compute_brightness_histogram, get_face_roi
from spoof_scoring import (
    analyze_passive_spoof_single_frame,
    has_display_attack_corroboration,
    _f,
)
from liveness_risk_engine import evaluate_match_security
from screen_frame_detection import analyze_match_frame_context
from screen_replay_analysis import (
    analyze_fullframe_screen_replay,
    has_physical_fullframe_signals,
    is_fullframe_screen_replay,
)
from device_filter import (
    adjust_device_replay_score,
    filter_devices_for_attack,
    hard_reject_phone_tablet_in_selfie,
    is_laptop_only_devices,
    screen_physical_replay_cues,
)

# Screen-like reflection labels from spoof_scoring.classify_reflection
SCREEN_REFLECTION_LABELS = frozenset({
    "phone_screen_reflection",
    "monitor_glare",
    "rectangular_source",
})


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def load_post_selfie_config() -> Dict[str, Any]:
    return {
        "yolo_conf": _f("MATCH_YOLO_CONF", 0.22),
        "device_near_face_dr": _f("MATCH_DEVICE_NEAR_FACE_DR", 0.18),
        "device_hard_iou": _f("MATCH_DEVICE_HARD_IOU", 0.32),
        "reflection_raw_max": _f("MATCH_REFLECTION_RAW_MAX", 0.48),
        "hard_reject_spoof": _flag("MATCH_HARD_REJECT_SPOOF", True),
        "hard_reject_screen_reflection": _flag("MATCH_HARD_REJECT_SCREEN_REFLECTION", True),
        "min_face_brightness": _f("MATCH_MIN_FACE_BRIGHTNESS", 35.0),
        "max_face_brightness": _f("MATCH_MAX_FACE_BRIGHTNESS", 252.0),
        "min_face_brightness_std": _f("MATCH_MIN_FACE_BRIGHTNESS_STD", 10.0),
        "max_clipped_highlight_ratio": _f("MATCH_MAX_CLIPPED_HIGHLIGHT_RATIO", 0.20),
        "errcount_device_near_face": int(_f("MATCH_ERRCOUNT_DEVICE_NEAR_FACE", 40)),
        "errcount_device_in_frame": int(_f("MATCH_ERRCOUNT_DEVICE_IN_FRAME", 30)),
        "errcount_spoof_fail": int(_f("MATCH_ERRCOUNT_SPOOF_FAIL", 35)),
        "errcount_bad_lighting": int(_f("MATCH_ERRCOUNT_BAD_LIGHTING", 8)),
    }


def _classify_device_display_name(cls_name: str, obj_w: float, obj_h: float, area_pct: float) -> str:
    from frame_processor import DEVICE_CLASSES

    base = DEVICE_CLASSES.get(cls_name, cls_name.title())
    if cls_name == "cell phone":
        ar = obj_w / (obj_h + 1e-6)
        if 0.72 <= ar <= 1.38 and area_pct >= 0.055:
            return "Tablet"
    if cls_name == "laptop":
        return "Laptop / MacBook"
    return base


def scan_yolo_devices(
    img: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    yolo,
    *,
    conf: float,
    hard_iou: float,
) -> Tuple[float, bool, List[str]]:
    """
    Returns (device_replay_risk, hard_overlap, list of human-readable device names found).
    """
    from frame_processor import DEVICE_CLASSES

    fx, fy, fw, fh = face_bbox
    h, w = img.shape[:2]
    max_risk = 0.0
    hard_overlap = False
    found: List[str] = []

    try:
        yolo_results = yolo(img, verbose=False, conf=conf)
        for r in yolo_results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])].lower()
                if cls_name not in DEVICE_CLASSES:
                    continue
                bx = box.xyxy[0].cpu().numpy()
                ix1 = max(bx[0], fx)
                iy1 = max(bx[1], fy)
                ix2 = min(bx[2], fx + fw)
                iy2 = min(bx[3], fy + fh)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                face_area = fw * fh + 1e-6
                iou_face = inter_area / face_area
                obj_w, obj_h = float(box.xywh[0][2]), float(box.xywh[0][3])
                area_pct = (obj_w * obj_h) / (w * h + 1e-6)
                display = _classify_device_display_name(cls_name, obj_w, obj_h, area_pct)
                if display not in found:
                    found.append(display)
                if iou_face > hard_iou:
                    hard_overlap = True
                part = min(1.0, iou_face * 2.1 + min(0.40, area_pct * 3.0))
                cx_norm = float((bx[0] + bx[2]) * 0.5 / (w + 1e-6))
                # Phone held to camera: device on frame edge or large in frame
                if cls_name in ("cell phone", "tv", "monitor", "laptop"):
                    if area_pct >= 0.05:
                        part = max(part, 0.38)
                    if area_pct >= 0.10:
                        part = max(part, 0.52)
                    if cx_norm < 0.24 or cx_norm > 0.76:
                        part = max(part, 0.45)
                    if cls_name == "cell phone" and area_pct >= 0.06:
                        part = max(part, 0.48)
                max_risk = max(max_risk, part)
    except Exception as e:
        print(f"YOLO selfie scan error: {e}")

    return float(min(1.0, max_risk)), hard_overlap, found


def check_face_environment_light(
    img_bgr: np.ndarray,
    pts_68: Optional[List[Dict[str, Any]]],
    cfg: Dict[str, Any],
    *,
    spoof_per_signal: Optional[Dict[str, float]] = None,
    devices_found: Optional[List[str]] = None,
    device_replay_score: float = 0.0,
    match_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Face-ROI lighting checks. Ambient room light (tube, sun, window) should NOT
    penalize unless display-attack corroboration is present.
    """
    issues: List[Dict[str, Any]] = []
    roi = get_face_roi(img_bgr, pts_68) if pts_68 else None
    target = roi if roi is not None and roi.size > 0 else img_bgr

    stats = compute_brightness_histogram(target)
    brightness = stats["brightness"]
    brightness_std = stats["brightness_std"]

    display_attack = has_display_attack_corroboration(
        spoof_per_signal,
        device_replay_score=device_replay_score,
        devices_found=devices_found,
        match_context=match_context,
    )

    # Usability: face too dark to verify
    if brightness < cfg["min_face_brightness"]:
        issues.append({
            "type": "Face Too Dark",
            "detail": (
                f"Mean face brightness {brightness:.0f} (min {cfg['min_face_brightness']:.0f}). "
                "Use better front lighting."
            ),
            "penalty": 0.10,
        })

    # Extreme over-exposure only (not normal bright room / sunlight)
    if brightness > cfg["max_face_brightness"]:
        issues.append({
            "type": "Face Over-Exposed",
            "detail": (
                f"Mean face brightness {brightness:.0f} (max {cfg['max_face_brightness']:.0f}). "
                "Reduce direct light on face."
            ),
            "penalty": 0.08,
        })

    # Screen-like uniformity / glare — only when display attack is corroborated
    if display_attack:
        min_std = float(cfg["min_face_brightness_std"])
        if brightness > 200:
            min_std = max(min_std, 18.0)
        if brightness > 225:
            min_std = max(min_std, 20.0)

        if brightness_std < min_std and brightness > 170:
            issues.append({
                "type": "Uniform Lighting (Screen-Like)",
                "detail": (
                    f"Low face texture variation (std {brightness_std:.1f}) with display indicators. "
                    "Do not use a phone or tablet screen."
                ),
                "penalty": 0.20,
            })

        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        clipped = float(np.mean(gray >= 250))
        clip_max = float(cfg["max_clipped_highlight_ratio"])
        if brightness > 210:
            clip_max = min(clip_max, 0.12)
        if clipped > clip_max:
            issues.append({
                "type": "Specular Screen Glare",
                "detail": (
                    f"Saturated highlights {clipped * 100:.1f}% of face with display cues "
                    f"(max {clip_max * 100:.0f}%)."
                ),
                "penalty": 0.18,
            })

    return issues


def reflection_should_hard_reject(
    spoof_detail: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    device_replay_score: float = 0.0,
    devices_found: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    if not cfg["hard_reject_screen_reflection"]:
        return False, ""

    refl_label = str(spoof_detail.get("reflection_classification") or "")
    per_signal = spoof_detail.get("confidence_per_signal") or {}
    reflection_raw = float(per_signal.get("reflection_raw", 0.0))
    triggered = list(spoof_detail.get("triggered_rules") or [])

    display_corr = has_display_attack_corroboration(
        per_signal,
        triggered,
        device_replay_score=device_replay_score,
        devices_found=devices_found,
    )
    if not display_corr:
        return False, ""

    if refl_label in SCREEN_REFLECTION_LABELS:
        return True, f"Screen-like reflection detected ({refl_label.replace('_', ' ')})"

    if reflection_raw > cfg["reflection_raw_max"]:
        return True, f"Strong screen reflection signal (score {reflection_raw:.2f})"

    if "rectangular_glare" in triggered:
        return True, "Rectangular glare typical of phone, tablet, or monitor screen"

    if "reflection" in triggered and refl_label not in ("natural_skin", "none", "unknown"):
        return True, f"Reflection pattern consistent with digital display ({refl_label})"

    return False, ""


def display_attack_should_hard_reject(
    spoof_detail: Dict[str, Any],
    *,
    device_replay_score: float = 0.0,
    devices_found: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Hard reject when high-brightness screen replay cues co-occur (photo/video on phone)."""
    per_signal = spoof_detail.get("confidence_per_signal") or {}
    triggered = list(spoof_detail.get("triggered_rules") or [])

    hi = float(per_signal.get("high_brightness_screen", 0.0))
    moire = float(per_signal.get("moire", 0.0))
    grid = float(per_signal.get("pixel_grid", 0.0))
    tex = float(per_signal.get("texture_degraded", 0.0))
    flat = float(per_signal.get("flat_plane", 0.0))

    if devices_found and hi >= 0.22:
        names = ", ".join(devices_found)
        return True, f"Electronic device ({names}) with high-brightness display replay"

    if hi >= 0.48 and (moire >= 0.32 or grid >= 0.34 or tex >= 0.38):
        return True, "High-brightness screen replay detected (moiré/texture/grid fusion)"

    if hi >= 0.58 and flat >= 0.42:
        return True, "High-brightness flat emissive surface detected"

    if count_display_imaging_signals(per_signal, threshold=0.40) >= 3:
        return True, "Multiple display imaging signals (screen/photo attack)"

    if device_replay_score >= 0.30 and is_display_imaging_triggered(triggered, per_signal):
        return True, "Device near face with display imaging pattern"

    return False, ""


def is_display_imaging_triggered(
    triggered: List[str],
    per_signal: Dict[str, float],
) -> bool:
    imaging = {"moire", "high_brightness_screen", "pixel_grid", "texture_degraded", "screen_border"}
    if any(t in imaging for t in triggered):
        return True
    return count_display_imaging_signals(per_signal, threshold=0.34) >= 2


def run_post_selfie_security(
    img_raw: np.ndarray,
    face_landmarks_mp: Optional[List[Dict[str, Any]]],
    *,
    errcount: int,
    penalties_breakdown: List[Dict[str, Any]],
    identity_assessment: Optional[Dict[str, Any]] = None,
    stream_risk_ema: float = 0.0,
    liveness_session_verified: bool = False,
) -> Dict[str, Any]:
    """
    Run YOLO, PAD spoof, multi-factor risk fusion (no single-condition reject).
    Returns security_verdict: pass | pass_penalty | reject, composite_risk, risk_factors.
    """
    identity_assessment = identity_assessment or {"skipped": True}
    cfg = load_post_selfie_config()
    from frame_processor import _get_yolo

    device_replay_score = 0.0
    devices_found: List[str] = []
    device_hard = False
    yolo = _get_yolo()

    face_bbox = (0, 0, img_raw.shape[1], img_raw.shape[0])
    if face_landmarks_mp:
        lm_xs = [p["x"] for p in face_landmarks_mp]
        lm_ys = [p["y"] for p in face_landmarks_mp]
        face_bbox = (
            int(min(lm_xs)),
            int(min(lm_ys)),
            int(max(lm_xs) - min(lm_xs)),
            int(max(lm_ys) - min(lm_ys)),
        )

    if yolo:
        dr, device_hard, devices_found = scan_yolo_devices(
            img_raw,
            face_bbox,
            yolo,
            conf=cfg["yolo_conf"],
            hard_iou=cfg["device_hard_iou"],
        )
        device_replay_score = adjust_device_replay_score(
            dr, devices_found, hard_overlap=device_hard
        )
        devices_for_attack = filter_devices_for_attack(
            devices_found, hard_overlap=device_hard, device_replay_score=dr
        )
        if is_laptop_only_devices(devices_found) and not device_hard:
            print(
                f"💻 Laptop capture context (ambient): YOLO saw {devices_found}, "
                "not treating as replay device"
            )
        print(
            f"🔒 YOLO selfie scan: risk={device_replay_score:.3f}, "
            f"hard_overlap={device_hard}, devices={devices_for_attack or devices_found}"
        )

        if device_replay_score > cfg["device_near_face_dr"] and devices_for_attack:
            print(f"⚠️ Device near face: risk={device_replay_score:.3f}")
            errcount += cfg["errcount_device_near_face"]
            penalties_breakdown.append({
                "type": "Electronic Device Near Face",
                "penalty": round(cfg["errcount_device_near_face"] * 0.01, 2),
                "count": 1,
                "detail": f"Device near face (risk={device_replay_score:.2f}): "
                f"{', '.join(devices_for_attack)}",
            })
        elif devices_for_attack:
            errcount += cfg["errcount_device_in_frame"]
            penalties_breakdown.append({
                "type": "Electronic Device In Frame",
                "penalty": round(cfg["errcount_device_in_frame"] * 0.01, 2),
                "count": 1,
                "detail": f"Detected: {', '.join(devices_for_attack)}",
            })

    face_bbox_match = face_bbox
    fullframe_report = analyze_fullframe_screen_replay(
        img_raw, face_bbox=face_bbox_match
    )
    bezel_score = float(fullframe_report.get("bezel_score", 0.0))
    frame_screen_border = float(fullframe_report.get("screen_border_score", 0.0))
    fullframe_replay_score = float(fullframe_report.get("replay_likelihood", 0.0))
    fullframe_replay = is_fullframe_screen_replay(
        fullframe_report, liveness_verified=liveness_session_verified
    )

    if fullframe_replay_score >= 0.28:
        print(
            f"🖥️ Full-frame screen replay: {fullframe_replay_score:.3f} "
            f"signals={fullframe_report.get('signals')}"
        )
    if bezel_score >= 0.35:
        print(f"📱 Frame bezel score: {bezel_score:.3f}")
    if frame_screen_border >= 0.35:
        print(f"🖥️ Frame screen-border score: {frame_screen_border:.3f}")

    devices_for_security = filter_devices_for_attack(
        devices_found, hard_overlap=bool(device_hard), device_replay_score=device_replay_score
    )

    hard_reject, hard_msg = hard_reject_phone_tablet_in_selfie(
        devices_found,
        device_hard=device_hard,
        device_replay_score=device_replay_score,
        bezel_score=bezel_score,
        screen_border_score=frame_screen_border,
        moire=float(fullframe_report.get("global_moire", 0.0)),
        pixel_grid=float(fullframe_report.get("region_moire_max", 0.0)),
        fullframe_signals=list(fullframe_report.get("signals") or []),
        replay_likelihood=fullframe_replay_score,
    )
    if hard_reject:
        print(f"🚫 Hard reject — phone/screen in capture: {hard_msg[:120]}")
        return {
            "error": hard_msg,
            "user_message": hard_msg,
            "spoof_detail": {"total_spoof_score": 100.0, "triggered_rules": ["phone_tablet_in_capture"]},
            "security_penalty_breakdown": penalties_breakdown,
            "capture_live_ok": False,
            "security_verdict": "reject",
            "composite_risk": 100.0,
            "risk_factors": {"device_risk": 100.0},
            "screen_replay": True,
            "digital_media": True,
            "user_mismatch": False,
        }

    extra_signals: Dict[str, float] = {}
    if device_replay_score > 0.01:
        extra_signals["device_replay"] = device_replay_score
    if bezel_score > 0.05:
        extra_signals["frame_bezel"] = bezel_score
    if frame_screen_border > 0.05:
        extra_signals["frame_screen_border"] = frame_screen_border
    if fullframe_replay_score > 0.05:
        extra_signals["fullframe_replay"] = fullframe_replay_score

    spoof_detail = analyze_passive_spoof_single_frame(
        img_raw,
        face_landmarks_mp,
        strict=True,
        extra_signals=extra_signals or None,
        single_frame_mode=True,
    )

    reflection_label = str(spoof_detail.get("reflection_classification") or "")
    per_signal = spoof_detail.get("confidence_per_signal") or {}
    match_context = {
        "match_selfie": True,
        "frame_bezel": bezel_score,
        "frame_screen_border": frame_screen_border,
        "fullframe_replay_score": fullframe_replay_score,
        "fullframe_signal_count": int(fullframe_report.get("signal_count", 0)),
        "fullframe_signals": list(fullframe_report.get("signals") or []),
        "fullframe_replay_flag": fullframe_replay,
        "global_moire": float(fullframe_report.get("global_moire", 0.0)),
        "global_banding": float(fullframe_report.get("global_banding", 0.0)),
        "reflection_label": reflection_label,
        "devices_found_list": list(devices_for_security),
        "laptop_capture_context": is_laptop_only_devices(devices_found),
        "liveness_session_verified": liveness_session_verified,
    }
    light_issues = check_face_environment_light(
        img_raw,
        face_landmarks_mp,
        cfg,
        spoof_per_signal=per_signal,
        devices_found=devices_for_security,
        device_replay_score=device_replay_score,
        match_context=match_context,
    )
    for issue in light_issues:
        print(f"💡 Lighting issue (display-corroborated): {issue['type']} — {issue['detail']}")
        errcount += cfg["errcount_bad_lighting"]
        penalties_breakdown.append({
            "type": issue["type"],
            "penalty": issue["penalty"],
            "count": 1,
            "detail": issue["detail"],
        })

    triggered_rules = spoof_detail.get("triggered_rules", [])
    live_reason = "OK" if spoof_detail.get("is_live") else (
        "Weighted spoof: " + ", ".join(triggered_rules)[:220]
    )

    effective_hard_overlap = bool(device_hard) and bool(devices_for_security)
    identity_merged = dict(identity_assessment or {})
    if match_context.get("laptop_capture_context"):
        identity_merged["laptop_capture_context"] = True
    if liveness_session_verified:
        identity_merged["liveness_session_verified"] = True

    security = evaluate_match_security(
        spoof_detail=spoof_detail,
        device_replay_score=device_replay_score,
        devices_found=devices_for_security,
        device_hard_overlap=effective_hard_overlap,
        identity_assessment=identity_merged,
        errcount=errcount,
        stream_risk_ema=stream_risk_ema,
        match_context=match_context,
        frame_bezel_score=bezel_score,
    )

    verdict = security["verdict"]
    composite_risk = float(security["composite_risk"])
    print(
        f"🔒 Multi-factor security: verdict={verdict}, composite={composite_risk:.1f}, "
        f"factors={security.get('risk_factors')}"
    )

    penalties_breakdown.append({
        "type": "Security Risk Score",
        "penalty": 0.0,
        "count": 1,
        "detail": f"Composite {composite_risk:.0f}/100 — {security.get('reason', '')}",
    })

    if (
        (fullframe_replay or screen_physical_replay_cues(
            bezel_score=bezel_score,
            screen_border_score=frame_screen_border,
            moire=float(fullframe_report.get("global_moire", 0.0)),
            pixel_grid=float(fullframe_report.get("region_moire_max", 0.0)),
            fullframe_signals=list(fullframe_report.get("signals") or []),
            replay_likelihood=fullframe_replay_score,
        ))
        and verdict != "reject"
    ):
        verdict = "reject"
        security = dict(security)
        security["verdict"] = "reject"
        security["screen_replay"] = True
        security["digital_media"] = True
        security["reason"] = (
            f"Full-frame screen replay ({fullframe_replay_score:.2f}): "
            f"{', '.join(fullframe_report.get('signals') or [])}"
        )
        composite_risk = max(composite_risk, float(security.get("composite_risk", 0)))

    if verdict == "reject":
        screen_replay = bool(security.get("screen_replay")) or fullframe_replay
        digital_media = bool(security.get("digital_media", screen_replay)) or fullframe_replay
        if screen_replay or digital_media:
            msg = (
                "Security Alert: Digital screen or photo replay detected. "
                "Do not use a photograph or digital screen."
            )
        else:
            msg = security.get("user_message") or (
                "Security Alert: Verification failed. Do not use a photograph or digital screen."
            )
        return {
            "error": msg,
            "user_message": msg,
            "spoof_detail": spoof_detail,
            "security_penalty_breakdown": penalties_breakdown,
            "capture_live_ok": False,
            "security_verdict": verdict,
            "composite_risk": composite_risk,
            "risk_factors": security.get("risk_factors"),
            "screen_replay": screen_replay,
            "digital_media": digital_media,
            "user_mismatch": bool(security.get("user_mismatch")),
        }

    live_ok = True
    if verdict == "pass_penalty":
        pen = float(security.get("confidence_penalty", 0.0))
        errcount += max(5, int(pen * 1000))
        penalties_breakdown.append({
            "type": "Elevated Security Risk (penalty)",
            "penalty": pen,
            "count": 1,
            "detail": security.get("reason", ""),
        })

    return {
        "errcount": errcount,
        "penalties_breakdown": penalties_breakdown,
        "spoof_detail": spoof_detail,
        "device_replay_score": device_replay_score,
        "live_ok": live_ok,
        "live_reason": live_reason,
        "light_issues": light_issues,
        "security_verdict": verdict,
        "composite_risk": composite_risk,
        "risk_factors": security.get("risk_factors"),
        "confidence_penalty": float(security.get("confidence_penalty", 0.0)),
    }
