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
from spoof_scoring import analyze_passive_spoof_single_frame, _f

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
        "yolo_conf": _f("MATCH_YOLO_CONF", 0.28),
        "device_near_face_dr": _f("MATCH_DEVICE_NEAR_FACE_DR", 0.18),
        "device_hard_iou": _f("MATCH_DEVICE_HARD_IOU", 0.32),
        "reflection_raw_max": _f("MATCH_REFLECTION_RAW_MAX", 0.38),
        "hard_reject_spoof": _flag("MATCH_HARD_REJECT_SPOOF", True),
        "hard_reject_screen_reflection": _flag("MATCH_HARD_REJECT_SCREEN_REFLECTION", True),
        "min_face_brightness": _f("MATCH_MIN_FACE_BRIGHTNESS", 42.0),
        "max_face_brightness": _f("MATCH_MAX_FACE_BRIGHTNESS", 248.0),
        "min_face_brightness_std": _f("MATCH_MIN_FACE_BRIGHTNESS_STD", 14.0),
        "max_clipped_highlight_ratio": _f("MATCH_MAX_CLIPPED_HIGHLIGHT_RATIO", 0.14),
        "errcount_device_near_face": int(_f("MATCH_ERRCOUNT_DEVICE_NEAR_FACE", 40)),
        "errcount_device_in_frame": int(_f("MATCH_ERRCOUNT_DEVICE_IN_FRAME", 30)),
        "errcount_spoof_fail": int(_f("MATCH_ERRCOUNT_SPOOF_FAIL", 35)),
        "errcount_bad_lighting": int(_f("MATCH_ERRCOUNT_BAD_LIGHTING", 15)),
    }


def _classify_device_display_name(cls_name: str, obj_w: float, obj_h: float, area_pct: float) -> str:
    from frame_processor import DEVICE_CLASSES

    base = DEVICE_CLASSES.get(cls_name, cls_name.title())
    if cls_name == "cell phone":
        ar = obj_w / (obj_h + 1e-6)
        # Large near-square phone bbox → likely tablet
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
                max_risk = max(max_risk, part)
    except Exception as e:
        print(f"YOLO selfie scan error: {e}")

    return float(min(1.0, max_risk)), hard_overlap, found


def check_face_environment_light(
    img_bgr: np.ndarray,
    pts_68: Optional[List[Dict[str, Any]]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Returns list of lighting issues (empty if OK).
    Each item: {type, detail, penalty_fraction}
    """
    issues: List[Dict[str, Any]] = []
    roi = get_face_roi(img_bgr, pts_68) if pts_68 else None
    target = roi if roi is not None and roi.size > 0 else img_bgr

    stats = compute_brightness_histogram(target)
    brightness = stats["brightness"]
    brightness_std = stats["brightness_std"]

    if brightness < cfg["min_face_brightness"]:
        issues.append({
            "type": "Face Too Dark",
            "detail": f"Mean brightness {brightness:.0f} (min {cfg['min_face_brightness']:.0f}). Use better front lighting.",
            "penalty": 0.15,
        })
    if brightness > cfg["max_face_brightness"]:
        issues.append({
            "type": "Face Over-Exposed",
            "detail": f"Mean brightness {brightness:.0f} (max {cfg['max_face_brightness']:.0f}). Reduce direct light on face.",
            "penalty": 0.12,
        })
    if brightness_std < cfg["min_face_brightness_std"]:
        issues.append({
            "type": "Uniform Lighting (Screen-Like)",
            "detail": (
                f"Very low brightness variation (std {brightness_std:.1f}). "
                "May indicate a flat emissive screen rather than natural skin."
            ),
            "penalty": 0.18,
        })

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    clipped = float(np.mean(gray >= 250))
    if clipped > cfg["max_clipped_highlight_ratio"]:
        issues.append({
            "type": "Specular Screen Glare",
            "detail": f"Saturated highlights {clipped * 100:.1f}% of face ROI (max {cfg['max_clipped_highlight_ratio'] * 100:.0f}%).",
            "penalty": 0.20,
        })

    return issues


def reflection_should_hard_reject(spoof_detail: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    if not cfg["hard_reject_screen_reflection"]:
        return False, ""

    refl_label = str(spoof_detail.get("reflection_classification") or "")
    per_signal = spoof_detail.get("confidence_per_signal") or {}
    reflection_raw = float(per_signal.get("reflection_raw", 0.0))
    triggered = set(spoof_detail.get("triggered_rules") or [])

    if refl_label in SCREEN_REFLECTION_LABELS:
        return True, f"Screen-like ambient reflection detected ({refl_label.replace('_', ' ')})"

    if reflection_raw > cfg["reflection_raw_max"]:
        return True, f"Strong screen reflection signal (score {reflection_raw:.2f})"

    if "rectangular_glare" in triggered:
        return True, "Rectangular glare typical of phone, tablet, or monitor screen"

    if "reflection" in triggered and refl_label not in ("natural_skin", "none", "unknown"):
        return True, f"Reflection pattern consistent with digital display ({refl_label})"

    return False, ""


def run_post_selfie_security(
    img_raw: np.ndarray,
    face_landmarks_mp: Optional[List[Dict[str, Any]]],
    *,
    errcount: int,
    penalties_breakdown: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run YOLO, environment light, and PAD spoof on the final selfie.
    Returns dict with optional 'error' for hard reject, updated errcount/penalties, spoof_detail.
    """
    cfg = load_post_selfie_config()
    from frame_processor import _get_yolo

    device_replay_score = 0.0
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
        device_replay_score = dr
        print(f"🔒 YOLO selfie scan: risk={dr:.3f}, hard_overlap={device_hard}, devices={devices_found}")

        if device_hard:
            names = ", ".join(devices_found) if devices_found else "electronic device"
            return {
                "error": (
                    f"Security Alert: {names} detected overlapping your face. "
                    "Take a direct selfie without holding a phone, tablet, or laptop in front of the camera."
                ),
                "security_penalty_breakdown": [{
                    "type": "Electronic Device Blocking Face",
                    "penalty": 1.0,
                    "count": 1,
                    "detail": f"YOLO overlap risk={dr:.2f}; detected: {names}",
                }],
            }

        if dr > cfg["device_near_face_dr"]:
            print(f"⚠️ Device near face: risk={dr:.3f}")
            errcount += cfg["errcount_device_near_face"]
            penalties_breakdown.append({
                "type": "Electronic Device Near Face",
                "penalty": round(cfg["errcount_device_near_face"] * 0.01, 2),
                "count": 1,
                "detail": f"Device near face (risk={dr:.2f})" + (
                    f": {', '.join(devices_found)}" if devices_found else ""
                ),
            })
        elif devices_found:
            errcount += cfg["errcount_device_in_frame"]
            penalties_breakdown.append({
                "type": "Electronic Device In Frame",
                "penalty": round(cfg["errcount_device_in_frame"] * 0.01, 2),
                "count": 1,
                "detail": f"Detected: {', '.join(devices_found)}",
            })

    # Environment / ambient light on face ROI
    light_issues = check_face_environment_light(img_raw, face_landmarks_mp, cfg)
    for issue in light_issues:
        print(f"💡 Lighting issue: {issue['type']} — {issue['detail']}")
        errcount += cfg["errcount_bad_lighting"]
        penalties_breakdown.append({
            "type": issue["type"],
            "penalty": issue["penalty"],
            "count": 1,
            "detail": issue["detail"],
        })

    extra_signals = {"device_replay": device_replay_score} if device_replay_score > 0.01 else None
    spoof_detail = analyze_passive_spoof_single_frame(
        img_raw,
        face_landmarks_mp,
        strict=True,
        extra_signals=extra_signals,
        single_frame_mode=True,
    )

    refl_reject, refl_reason = reflection_should_hard_reject(spoof_detail, cfg)
    if refl_reject:
        print(f"🚨 HARD REJECT reflection: {refl_reason}")
        return {
            "error": (
                "Ambient light / screen reflection detected. "
                "Do not photograph your face from a phone, tablet, or laptop screen. "
                f"({refl_reason})"
            ),
            "spoof_detail": spoof_detail,
            "security_penalty_breakdown": penalties_breakdown,
        }

    live_ok = bool(spoof_detail["is_live"])
    triggered_rules = spoof_detail.get("triggered_rules", [])
    live_reason = "OK" if live_ok else ("Weighted spoof: " + ", ".join(triggered_rules)[:220])

    print(
        f"🔒 Spoof selfie: score={spoof_detail['total_spoof_score']:.1f}/"
        f"{spoof_detail.get('reject_threshold', 68)}, live={live_ok}, rules={triggered_rules}"
    )

    if not live_ok:
        if cfg["hard_reject_spoof"]:
            return {
                "error": (
                    "Security Alert: Final capture failed live-face verification. "
                    "Do not use a photo or screen; face the camera directly in good lighting. "
                    + live_reason[:200]
                ),
                "spoof_detail": spoof_detail,
                "security_penalty_breakdown": penalties_breakdown,
            }
        errcount += cfg["errcount_spoof_fail"]
        penalties_breakdown.append({
            "type": "Digital Media / Screen Detected",
            "penalty": round(cfg["errcount_spoof_fail"] * 0.01, 2),
            "count": 1,
            "detail": (
                f"Spoof score {spoof_detail['total_spoof_score']:.0f}/"
                f"{spoof_detail.get('reject_threshold', 68):.0f} — {live_reason[:400]}"
            ),
        })

    return {
        "errcount": errcount,
        "penalties_breakdown": penalties_breakdown,
        "spoof_detail": spoof_detail,
        "device_replay_score": device_replay_score,
        "live_ok": live_ok,
        "live_reason": live_reason,
        "light_issues": light_issues,
    }
