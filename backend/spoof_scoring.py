"""
Weighted multi-signal presentation-attack (PAD) scoring.

Reflection/glare never rejects alone: glare-related contributions are capped and
down-weighted when classified as natural skin / forehead highlights.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import image heuristics from face_detection after its module init (no cycle if face_detection
# only imports this lazily from compute_passive_liveness).
from face_detection import (
    detect_scanline_artifacts,
    detect_peak_saturation,
    detect_specular_highlights,
    check_emissive_uniformity,
    check_screen_edges,
    compute_color_diversity,
    get_face_roi,
    compute_rtc_compression_score,
    compute_sensor_authenticity,
    analyze_eye_reflections,
    detect_display_refresh_rate,
    normalize_for_pad_analysis,
    signal_subpixel_grid,
    signal_rgb_channel_lock,
    analyze_high_brightness_screen_attack,
)


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def load_spoof_weights() -> Dict[str, float]:
    """
    Default weights favor depth / biology / device / challenge over reflection & flicker.
    All env-tunable (see .env.example).
    """
    return {
        "depth_parallax": _f("SPOOF_WEIGHT_DEPTH_PARALLAX", 40.0),
        "biological": _f("SPOOF_WEIGHT_BIOLOGICAL", 40.0),
        "device_replay": _f("SPOOF_WEIGHT_DEVICE_REPLAY", 30.0),
        "challenge": _f("SPOOF_WEIGHT_CHALLENGE", 28.0),
        "texture_degraded": _f("SPOOF_WEIGHT_TEXTURE", 20.0),
        "moire": _f("SPOOF_WEIGHT_MOIRE", 15.0),
        "flicker": _f("SPOOF_WEIGHT_FLICKER", 5.0),
        "reflection": _f("SPOOF_WEIGHT_REFLECTION", 5.0),
        "flat_plane": _f("SPOOF_WEIGHT_FLAT_PLANE", 12.0),
        "screen_border": _f("SPOOF_WEIGHT_SCREEN_BORDER", 15.0),
        "bg_uniform_motion": _f("SPOOF_WEIGHT_BG_MOTION", 18.0),
        "perspective": _f("SPOOF_WEIGHT_PERSPECTIVE", 8.0),
        "rect_glare": _f("SPOOF_WEIGHT_RECT_GLARE", 10.0),
        "video_call_replay": _f("SPOOF_WEIGHT_VIDEO_CALL", 25.0),
        "eye_screen_reflection": _f("SPOOF_WEIGHT_EYE_REFLECTION", 20.0),
        "temporal_stream_integrity": _f("SPOOF_WEIGHT_TEMPORAL_STREAM", 20.0),
        "sensor_authenticity": _f("SPOOF_WEIGHT_SENSOR_AUTH", 15.0),
        "environment_authenticity": _f("SPOOF_WEIGHT_ENV_AUTH", 15.0),
        "high_brightness_screen": _f("SPOOF_WEIGHT_HIGH_BRIGHT_SCREEN", 22.0),
        "pixel_grid": _f("SPOOF_WEIGHT_PIXEL_GRID", 12.0),
    }


def load_thresholds() -> Dict[str, float]:
    return {
        "reject_total": _f("SPOOF_REJECT_THRESHOLD", 76.0),
        "match_total": _f("SPOOF_MATCH_THRESHOLD", 86.0),
        "streaming_smooth": _f("SPOOF_STREAMING_SMOOTH_THRESHOLD", 71.0),
        "ema_alpha": _f("SPOOF_EMA_ALPHA", 0.28),
        "corr_strong_min": float(_i("SPOOF_CORR_STRONG_MIN", 2)),
        "temporal_hits_required": _i("SPOOF_TEMPORAL_HITS_REQUIRED", 3),
        "cooldown_decay_threshold": _f("SPOOF_COOLDOWN_DECAY_THRESHOLD", 40.0),
    }


DISPLAY_IMAGING_KEYS = (
    "moire",
    "high_brightness_screen",
    "pixel_grid",
    "texture_degraded",
    "screen_border",
    "flat_plane",
    "device_replay",
)

WEAK_GLARE_TRIGGERS = frozenset({"reflection", "rectangular_glare", "flicker"})


def count_display_imaging_signals(
    per_signal: Optional[Dict[str, Any]],
    *,
    threshold: float = 0.32,
) -> int:
    if not per_signal:
        return 0
    return sum(
        1
        for k in DISPLAY_IMAGING_KEYS
        if float(per_signal.get(k, 0.0)) >= threshold
    )


def has_display_attack_corroboration(
    per_signal: Optional[Dict[str, Any]],
    triggered: Optional[List[str]] = None,
    *,
    device_replay_score: float = 0.0,
    devices_found: Optional[List[str]] = None,
    match_context: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when multiple independent cues indicate a digital display, not ambient room light."""
    mc = match_context or {}
    if mc.get("laptop_capture_context"):
        devices_found = None
        device_replay_score = min(float(device_replay_score), 0.10)

    if devices_found:
        return True
    if device_replay_score >= 0.28:
        return True

    per_signal = per_signal or {}
    hi = float(per_signal.get("high_brightness_screen", 0.0))
    moire = float(per_signal.get("moire", 0.0))
    grid = float(per_signal.get("pixel_grid", 0.0))
    tex = float(per_signal.get("texture_degraded", 0.0))
    border = float(per_signal.get("screen_border", 0.0))

    # POST /match only — stricter phone-replay rules; liveness stream passes match_context=None
    if mc.get("match_selfie"):
        ff_score = float(mc.get("fullframe_replay_score", 0.0))
        ff_signals = mc.get("fullframe_signals") or []
        liveness_ok = bool(mc.get("liveness_session_verified"))
        from screen_replay_analysis import has_physical_fullframe_signals

        bezel_ff = float(mc.get("frame_bezel", 0.0))
        border_ff = float(mc.get("frame_screen_border", 0.0))
        has_physical_ff = (
            has_physical_fullframe_signals(ff_signals)
            or bezel_ff >= 0.45
            or border_ff >= 0.42
        )

        if mc.get("fullframe_replay_flag") and has_physical_ff:
            return True
        if liveness_ok:
            if devices_found:
                return True
            if device_replay_score >= 0.22:
                return True
            if has_physical_ff and ff_score >= 0.52:
                return True
            # Do not unlock display path from moiré/artifact-only on verified liveness webcam
        elif ff_score >= 0.55 and has_physical_ff:
            return True
        elif ff_score >= 0.48 and int(mc.get("fullframe_signal_count", 0)) >= 2:
            if has_physical_ff:
                return True
        bezel = float(mc.get("frame_bezel", 0.0))
        screen_border = float(mc.get("frame_screen_border", 0.0))
        refl = str(mc.get("reflection_label") or "")
        screen_refl = refl in (
            "phone_screen_reflection",
            "monitor_glare",
            "rectangular_source",
        )
        if bezel >= 0.40:
            return True
        if mc.get("devices_found_list"):
            return True
        if device_replay_score >= 0.16:
            return True
        if screen_border >= 0.38 and (moire >= 0.22 or hi >= 0.30):
            return True
        if screen_refl and (moire >= 0.24 or grid >= 0.22 or border >= 0.28):
            return True
        if bezel >= 0.32 and (moire >= 0.22 or tex >= 0.30):
            return True
        if count_display_imaging_signals(per_signal, threshold=0.28) >= 2:
            if not liveness_ok or has_physical_ff:
                return True
        if hi >= 0.42 and (moire >= 0.26 or grid >= 0.26):
            return True

    if hi >= 0.55 and (moire >= 0.34 or grid >= 0.34):
        return True
    if count_display_imaging_signals(per_signal, threshold=0.36) >= 3:
        return True
    trig = set(triggered or [])
    strong_imaging = trig & {"moire", "high_brightness_screen", "pixel_grid", "screen_border"}
    if strong_imaging and count_display_imaging_signals(per_signal, threshold=0.32) >= 2:
        return True
    return False


def is_display_related_spoof_trigger(triggered: Optional[List[str]]) -> bool:
    """Exclude glare-only and ambient motion triggers (tube light, window, forehead shine)."""
    if not triggered:
        return False
    ambient_only = frozenset({
        "environment_authenticity",
        "perspective",
        "flicker",
        "reflection",
        "rectangular_glare",
    })
    meaningful = [t for t in triggered if t not in ambient_only]
    if not meaningful:
        return False
    for t in meaningful:
        if t in DISPLAY_IMAGING_KEYS or t in (
            "depth_parallax",
            "biological",
            "device_replay",
            "challenge",
            "video_call_replay",
            "eye_screen_reflection",
            "temporal_stream_integrity",
        ):
            return True
    return False


def classify_reflection(
    roi_bgr: np.ndarray,
    pts_68: Optional[List[Dict[str, Any]]],
) -> Tuple[str, float]:
    """
    Differentiate natural skin highlights vs panel glare.
    Returns (label, confidence_in_[0,1] that label applies).
    """
    if roi_bgr is None or roi_bgr.size == 0 or not pts_68:
        return "unknown", 0.0

    h, w = roi_bgr.shape[:2]
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    high = v > 240  # Raised from 235 to reduce false triggers from bright skin
    if not np.any(high):
        return "none", 0.0

    # Map landmark bbox in full image to ROI coords (ROI is face crop)
    xs = [float(p["x"]) for p in pts_68]
    ys = [float(p["y"]) for p in pts_68]
    fx1, fy1, fx2, fy2 = min(xs), min(ys), max(xs), max(ys)
    # ROI in spoof pipeline matches get_face_roi expansion — caller passes same ROI as face_detection
    # Here pts are in FULL image; we approximate forehead band in ROI using relative y from landmarks
    fh = fy2 - fy1 + 1e-6
    forehead_y = (float(pts_68[19]["y"] + pts_68[24]["y"]) / 2 - fy1) / fh  # 0..1 within face

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(high.astype(np.uint8))
    if num_labels <= 1:
        return "none", 0.0

    best_area = 0
    best_cy = 0.0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
        if area > best_area:
            best_area = area
            best_cy = cy / (h + 1e-6)

    area_ratio = best_area / float(h * w + 1e-6)

    roi_mean = float(np.mean(v))

    # Small highlight in upper face = likely skin / forehead shine
    if area_ratio < 0.06 and forehead_y < 0.60 and best_cy < 0.50:
        # Under very high exposure, small highlights can still be screen glare
        if roi_mean < 210:
            return "natural_skin", min(1.0, area_ratio * 12.0)

    # Large bright blob, not confined to forehead
    if area_ratio > 0.10:  # Raised from 0.08
        rect_score = rectangular_glare_strength(roi_bgr)
        if rect_score > 0.60:  # Raised from 0.55
            return "rectangular_source", rect_score
        return "monitor_glare", min(1.0, area_ratio * 6.0)

    # Mid-sized scattered speculars — raised lower bound
    if 0.06 <= area_ratio <= 0.10:
        return "phone_screen_reflection", min(1.0, area_ratio * 8.0)

    # High-brightness scene: require stronger evidence than a small highlight
    rect_score = rectangular_glare_strength(roi_bgr)
    if roi_mean > 215 and area_ratio >= 0.08 and rect_score > 0.50:
        return "phone_screen_reflection", min(1.0, area_ratio * 6.0)

    return "natural_skin", 0.25


def rectangular_glare_strength(roi_bgr: np.ndarray) -> float:
    """Large axis-aligned saturated regions typical of monitor bezels / windows."""
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    v = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    m = (v > 245).astype(np.uint8) * 255
    if not np.any(m):
        return 0.0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    ah, aw = roi_bgr.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / (bh + 1e-6)
        area = bw * bh
        if area < (ah * aw) * 0.02:
            continue
        if 1.4 < ar < 4.5 or 0.25 < ar < 0.71:
            best = max(best, area / (ah * aw + 1e-6))
    return float(min(1.0, best * 6.0))


def signal_moire(gray: np.ndarray) -> float:
    rows, cols = gray.shape
    if rows < 8 or cols < 8:
        return 0.0
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.abs(dft_shift) + 1)
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = int(min(rows, cols) * 0.15)
    mask[crow - r : crow + r, ccol - r : ccol + r] = 0
    high_freq = magnitude * mask
    freq_score = float(np.mean(high_freq[high_freq > 0])) if np.any(high_freq > 0) else 0.0
    max_peak = float(np.max(high_freq)) if high_freq.size else 0.0
    peak_to_mean = max_peak / (freq_score + 1e-6)
    # Normalize to 0–1
    s = min(1.0, max(0.0, (peak_to_mean - 1.4) / 2.2))
    s = max(s, min(1.0, (freq_score / 120.0)))
    return float(s)


def signal_texture_degraded(gray: np.ndarray, roi_gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_roi = cv2.Laplacian(roi_gray, cv2.CV_64F).var() if roi_gray.size else lap
    # Low texture in ROI but moderate globally → print/screen
    roi_n = min(1.0, lap_roi / 420.0)
    glob_n = min(1.0, lap / 420.0)
    if roi_n < 0.22 and glob_n < 0.35:
        return float(min(1.0, (0.28 - roi_n) / 0.28 + (0.35 - glob_n)))
    return float(max(0.0, min(1.0, 0.45 - roi_n)))


def signal_screen_border(img_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> float:
    if check_screen_edges(img_bgr, face_bbox):
        return 1.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    fx, fy, fw, fh = face_bbox
    pad = int(max(fw, fh) * 0.35)
    h, w = gray.shape
    x1, y1 = max(0, fx - pad), max(0, fy - pad)
    x2, y2 = min(w, fx + fw + pad), min(h, fy + fh + pad)
    ring = edges[y1:y2, x1:x2]
    if ring.size == 0:
        return 0.0
    edge_density = float(np.mean(ring > 0))
    return float(min(1.0, max(0.0, (edge_density - 0.12) / 0.35)))


def signal_flicker(roi_luminance_history: List[float]) -> float:
    if len(roi_luminance_history) < 4:
        return 0.0
    arr = np.array(roi_luminance_history[-12:], dtype=np.float64)
    d = np.diff(arr)
    std = float(np.std(d))
    # Screen refresh / auto-exposure banding
    return float(min(1.0, max(0.0, (std - 2.8) / 8.0)))


def signal_bg_uniform_motion(
    last_small: Optional[np.ndarray],
    curr_small: Optional[np.ndarray],
    landmark_centroid_history: List[Tuple[float, float]],
) -> float:
    """Whole frame shifts uniformly while face landmarks barely move → replay / held photo."""
    if last_small is None or curr_small is None or last_small.shape != curr_small.shape:
        return 0.0
    if len(landmark_centroid_history) < 3:
        return 0.0

    diff = cv2.absdiff(last_small, curr_small)
    frame_motion = float(np.mean(diff)) / 255.0

    cx = [p[0] for p in landmark_centroid_history[-5:]]
    cy = [p[1] for p in landmark_centroid_history[-5:]]
    lm_motion = float(np.hypot(max(cx) - min(cx), max(cy) - min(cy)) + 1e-6)

    if frame_motion < 0.004:
        return 0.0
    ratio = lm_motion / (frame_motion * 120.0 + 1e-6)
    # Low landmark motion vs frame motion
    if ratio < 0.35 and frame_motion > 0.012:
        return float(min(1.0, (0.35 - ratio) / 0.35 + (frame_motion - 0.012) * 8.0))
    return 0.0


def signal_flat_plane(
    landmark_centroid_history: List[Tuple[float, float]],
    roi_luminance_history: List[float],
) -> float:
    """Low relative landmark motion vs global brightness swings + emissive-like flatness."""
    strength = 0.0
    if len(landmark_centroid_history) >= 6:
        cx = [p[0] for p in landmark_centroid_history[-8:]]
        cy = [p[1] for p in landmark_centroid_history[-8:]]
        span = float(np.hypot(max(cx) - min(cx), max(cy) - min(cy)))
        if span < 2.5:
            strength = max(strength, min(1.0, (2.5 - span) / 2.5))

    if len(roi_luminance_history) >= 5:
        arr = np.array(roi_luminance_history[-8:], dtype=np.float64)
        if float(np.std(arr)) < 1.2 and float(np.mean(arr)) > 175:
            strength = max(strength, 0.45)

    return float(min(1.0, strength))


def signal_perspective(pts_68: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> float:
    """Extreme affine inconsistency — weak signal."""
    if not pts_68:
        return 0.0
    h, w = frame_shape
    xs = [p["x"] for p in pts_68]
    ys = [p["y"] for p in pts_68]
    fw = max(xs) - min(xs) + 1e-6
    fh = max(ys) - min(ys) + 1e-6
    aspect = fw / fh
    # Printed photo odd aspect in bbox
    if aspect < 0.55 or aspect > 1.35:
        return min(1.0, abs(aspect - 0.72) * 1.2)
    return 0.0


def _gather_raw_signals(
    img_bgr: np.ndarray,
    pts_68: Optional[List[Dict[str, Any]]],
    strict: bool,
    roi_luminance_history: Optional[List[float]],
    last_gray_small: Optional[np.ndarray],
    curr_gray_small: Optional[np.ndarray],
    landmark_centroid_history: Optional[List[Tuple[float, float]]],
) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    roi_gray = gray
    roi_bgr = img_bgr
    roi_hsv = hsv
    face_bbox = (0, 0, gray.shape[1], gray.shape[0])
    if pts_68:
        xs = [p["x"] for p in pts_68]
        ys = [p["y"] for p in pts_68]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        gh, gw = gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(gw, x2), min(gh, y2)
        if x2 > x1 and y2 > y1:
            roi_gray = gray[y1:y2, x1:x2]
            roi_bgr = img_bgr[y1:y2, x1:x2]
            roi_hsv = hsv[y1:y2, x1:x2]
            face_bbox = (x1, y1, x2 - x1, y2 - y1)

    global_sat = hsv[:, :, 1]
    global_val = hsv[:, :, 2]
    global_high_sat = float(np.sum(global_sat > 200) / (global_sat.size + 1e-6))
    global_high_val = float(np.sum(global_val > 230) / (global_val.size + 1e-6))

    norm_gray, norm_roi_gray = normalize_for_pad_analysis(img_bgr, roi_bgr)
    roi_mean_brightness = float(np.mean(roi_gray)) if roi_gray.size else float(np.mean(gray))
    high_bright_scene = roi_mean_brightness > 175 or global_high_val > 0.32

    hi_bright_screen_pre = analyze_high_brightness_screen_attack(roi_bgr, roi_gray, norm_roi_gray)
    moire_raw_pre = max(signal_moire(gray), signal_moire(norm_gray))
    pixel_grid_pre = signal_subpixel_grid(norm_roi_gray) if norm_roi_gray is not None and norm_roi_gray.size else 0.0
    suspected_display = (
        hi_bright_screen_pre >= 0.35
        or moire_raw_pre >= 0.32
        or pixel_grid_pre >= 0.28
    )

    # Dampen glare/reflection in bright scenes only when there is NO display imaging evidence.
    glare_damp = 1.0
    if (global_high_sat > 0.32 or global_high_val > 0.38) and not suspected_display:
        glare_damp = 0.62 if strict else 0.75

    # Amplify moiré/texture/grid when over-exposed (phone 100% + screen replay).
    screen_sens = 1.0
    if high_bright_scene or suspected_display:
        screen_sens = 1.38 if strict else 1.22

    moire_raw = moire_raw_pre
    moire = min(1.0, moire_raw * screen_sens)
    texture = min(
        1.0,
        max(
            signal_texture_degraded(gray, roi_gray),
            signal_texture_degraded(norm_gray, norm_roi_gray),
        )
        * screen_sens,
    )
    scan = max(
        float(detect_scanline_artifacts(img_bgr)),
        float(detect_scanline_artifacts(cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR))),
    )
    peak = float(detect_peak_saturation(roi_bgr))
    glare_ratio = float(detect_specular_highlights(roi_bgr))
    color_div = float(compute_color_diversity(roi_bgr))

    is_emissive, _ = check_emissive_uniformity(roi_bgr)
    emissive_boost = 0.42 if is_emissive else 0.0
    if high_bright_scene and is_emissive:
        emissive_boost = 0.55

    border = min(1.0, signal_screen_border(img_bgr, face_bbox) * screen_sens)

    flicker = signal_flicker(roi_luminance_history or []) * glare_damp

    pixel_grid = min(1.0, signal_subpixel_grid(norm_roi_gray) * screen_sens)
    rgb_lock = min(1.0, signal_rgb_channel_lock(roi_bgr) * (1.15 if high_bright_scene else 1.0))
    hi_bright_screen = hi_bright_screen_pre

    lm_hist = landmark_centroid_history or []
    flat_p = signal_flat_plane(lm_hist, roi_luminance_history or [])
    bg_mot = signal_bg_uniform_motion(last_gray_small, curr_gray_small, lm_hist)

    persp = signal_perspective(pts_68, (img_bgr.shape[0], img_bgr.shape[1])) if pts_68 else 0.0
    rect_g = rectangular_glare_strength(roi_bgr)

    refl_label, refl_conf = classify_reflection(roi_bgr, pts_68)

    # Scanline + peak + moire fusion for single "moire" bucket
    scan_n = max(0.0, min(1.0, (scan - 0.35) / 0.45))
    moire_combined = min(
        1.0,
        0.55 * moire + 0.35 * scan_n + 0.25 * min(1.0, peak_to_moire_helper(gray)),
    )

    # Suppress glare on natural skin only when display imaging is NOT suspected
    glare_for_score = glare_ratio
    if not suspected_display:
        if refl_label == "natural_skin" and refl_conf > 0.15:
            glare_for_score *= 0.10 + 0.3 * (1.0 - refl_conf)
        if refl_label in ("natural_skin", "none"):
            peak *= 0.25
            glare_for_score *= 0.5
        
    rtc_comp = compute_rtc_compression_score(img_bgr, face_bbox)
    sensor_auth = compute_sensor_authenticity(img_bgr)
    eye_refl = analyze_eye_reflections(img_bgr, pts_68) if pts_68 else 0.0
    refresh_rate = detect_display_refresh_rate(img_bgr)

    return {
        "moire": float(min(1.0, moire_combined + emissive_boost * 0.3 + pixel_grid * 0.2)),
        "flat_plane": float(min(1.0, flat_p + (0.22 if is_emissive else 0.0) + hi_bright_screen * 0.15)),
        "screen_border": border,
        "flicker": float(min(1.0, flicker + refresh_rate * 0.5)),
        "reflection_raw": float(min(1.0, (glare_for_score * 2.5 + peak * 1.2) * glare_damp)),
        "texture_degraded": float(min(1.0, texture + (0.15 if color_div < 0.025 else 0.0) + rgb_lock * 0.12)),
        "rect_glare": float(min(1.0, rect_g + 0.25 * peak)),
        "bg_uniform_motion": bg_mot,
        "perspective": persp,
        "video_call_replay": rtc_comp,
        "sensor_authenticity": sensor_auth,
        "eye_screen_reflection": eye_refl,
        "display_refresh_probability": refresh_rate,
        "high_brightness_screen": hi_bright_screen,
        "pixel_grid": pixel_grid,
        "_refl_label": refl_label,
        "_refl_conf": refl_conf,
        "_scan": scan,
        "_peak": peak,
        "_glare": glare_ratio,
        "_roi_mean_brightness": roi_mean_brightness,
    }


def peak_to_moire_helper(gray: np.ndarray) -> float:
    rows, cols = gray.shape
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.abs(dft_shift) + 1)
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = int(min(rows, cols) * 0.15)
    mask[crow - r : crow + r, ccol - r : ccol + r] = 0
    high_freq = magnitude * mask
    freq_score = float(np.mean(high_freq[high_freq > 0])) if np.any(high_freq > 0) else 0.0
    max_peak = float(np.max(high_freq)) if high_freq.size else 0.0
    peak_to_mean = max_peak / (freq_score + 1e-6)
    return float(min(1.0, max(0.0, (peak_to_mean - 1.6) / 2.0)))


def aggregate_weighted_score(
    raw: Dict[str, float],
    weights: Dict[str, float],
    strict: bool,
    extra_signals: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[str], Dict[str, float]]:
    """Returns (total 0–100, triggered rule names, per-signal confidence 0–1)."""
    strict_mul = 1.06 if strict else 1.0
    extra_signals = extra_signals or {}

    per_signal_conf: Dict[str, float] = {}
    refl_label = raw.get("_refl_label", "unknown")

    def wcontrib(key: str, rkey: str) -> float:
        v = float(min(1.0, max(0.0, raw[rkey])))
        per_signal_conf[key] = v
        w = weights.get(key, 0.0)
        return w * v * strict_mul

    total = 0.0
    total += wcontrib("moire", "moire")
    total += wcontrib("flat_plane", "flat_plane")
    total += wcontrib("screen_border", "screen_border")
    total += wcontrib("flicker", "flicker")

    ref_strength = wcontrib("reflection", "reflection_raw")
    rect_part = weights.get("rect_glare", 10.0) * float(min(1.0, max(0.0, raw["rect_glare"]))) * strict_mul
    per_signal_conf["rect_glare"] = float(min(1.0, max(0.0, raw["rect_glare"])))

    glare_bucket = ref_strength + 0.3 * rect_part  # Reduced rect_part influence from 0.4
    max_glare = weights.get("reflection", 5.0) + 2.0  # Reduced cap from +4 to +2
    if refl_label == "natural_skin":
        glare_bucket *= 0.15  # Much heavier suppression (was 0.32)
    elif refl_label == "none":
        glare_bucket *= 0.20  # Also suppress when no reflection detected
    elif refl_label == "monitor_glare":
        glare_bucket = min(glare_bucket * 1.04, max_glare)  # Reduced from 1.08
    elif refl_label == "phone_screen_reflection":
        glare_bucket = min(glare_bucket * 1.02, max_glare)  # Reduced from 1.04

    glare_bucket = min(glare_bucket, max_glare)
    total += glare_bucket

    total += wcontrib("texture_degraded", "texture_degraded")
    total += wcontrib("high_brightness_screen", "high_brightness_screen")
    total += wcontrib("pixel_grid", "pixel_grid")
    total += wcontrib("bg_uniform_motion", "bg_uniform_motion")
    total += wcontrib("perspective", "perspective")

    if raw["rect_glare"] > 0.55:
        extra_sb = weights.get("screen_border", 15.0) * 0.12 * raw["rect_glare"] * strict_mul
        total += extra_sb
        
    # High-priority session / landmark signals (replay_risk 0..1)
    for key in (
        "depth_parallax", "biological", "device_replay", "challenge",
        "video_call_replay", "eye_screen_reflection", "sensor_authenticity",
        "temporal_stream_integrity", "environment_authenticity"
    ):
        v = float(min(1.0, max(0.0, raw.get(key, extra_signals.get(key, 0.0)))))
        per_signal_conf[key] = v
        w = weights.get(key, 0.0)
        total += w * v * strict_mul

    total = float(min(100.0, total))

    triggered: List[str] = []
    for key in (
        "moire",
        "flat_plane",
        "screen_border",
        "flicker",
        "texture_degraded",
        "bg_uniform_motion",
        "perspective",
        "depth_parallax",
        "biological",
        "device_replay",
        "challenge",
        "video_call_replay",
        "eye_screen_reflection",
        "sensor_authenticity",
        "temporal_stream_integrity",
        "environment_authenticity",
        "high_brightness_screen",
        "pixel_grid",
    ):
        if per_signal_conf.get(key, 0) >= 0.42:
            triggered.append(key)
    if per_signal_conf.get("reflection", 0) >= 0.38 and refl_label != "natural_skin":
        triggered.append("reflection")
    if raw["rect_glare"] >= 0.45:
        triggered.append("rectangular_glare")

    return total, triggered, per_signal_conf


def _apply_correlation_gate(
    total: float,
    per_signal: Dict[str, Any],
    extra_signals: Optional[Dict[str, float]],
    triggered: List[str],
    reject_threshold: float,
    min_strong: int = 3,
    single_frame_mode: bool = False,
    refl_label: str = "unknown",
) -> Tuple[float, List[str]]:
    """
    STRENGTHENED correlation gate: one suspicious signal should NEVER reject.
    Only correlated suspicious signals across multiple categories can push score up.

    IMPORTANT: In single_frame_mode (selfie capture at /match), the gate is
    relaxed because we have NO temporal signals — imaging signals (moiré, texture,
    scanlines, emissive uniformity) are our primary tool for detecting screen replay.
    """
    extra = extra_signals or {}
    strong_keys = ("depth_parallax", "biological", "device_replay", "challenge")
    strong_count = 0
    for k in strong_keys:
        v = max(float(per_signal.get(k, 0.0)), float(extra.get(k, 0.0)))
        if v > 0.48:
            strong_count += 1

    device_hi = max(float(per_signal.get("device_replay", 0.0)), float(extra.get("device_replay", 0.0))) >= 0.75

    weak_trigger_names = {
        "moire",
        "flicker",
        "reflection",
        "flat_plane",
        "perspective",
        "texture_degraded",
        "screen_border",
        "bg_uniform_motion",
        "rectangular_glare",
    }
    only_weak_triggers = bool(triggered) and all(t in weak_trigger_names for t in triggered)
    weak_count = sum(1 for t in triggered if t in weak_trigger_names)

    reasons: List[str] = []
    adj = float(total)

    # ── SINGLE-FRAME MODE (selfie capture): lighter gate ──
    if single_frame_mode:
        # In single-frame mode, we DON'T have temporal/depth/biological signals.
        # Imaging signals (moiré, texture, emissive, scanlines) are our primary tools.
        # Only suppress glare/reflection-only scores, let real imaging signals through.

        # Still suppress if ONLY glare/reflection/flicker triggered (those are unreliable alone)
        glare_only_triggers = {"reflection", "rectangular_glare", "flicker"}
        only_glare = bool(triggered) and all(t in glare_only_triggers for t in triggered)
        screen_refl_labels = {"phone_screen_reflection", "monitor_glare", "rectangular_source"}
        if refl_label in screen_refl_labels:
            adj += 18.0
            reasons.append(f"single_frame_screen_reflection_boost ({refl_label})")
        elif only_glare and adj >= reject_threshold - 8.0:
            adj = min(adj, reject_threshold - 6.0)
            reasons.append("single_frame_glare_only_suppressed")

        # If device_replay detected via YOLO, boost confidence
        if device_hi:
            adj = max(adj, reject_threshold + 5.0)  # Force above threshold
            reasons.append("single_frame_device_detected_boost")

        # ── SCREEN EVIDENCE FUSION BOOST ──
        # When multiple screen-indicative imaging signals co-occur, their combination
        # is VERY strong evidence of a digital screen (phone/laptop/tablet/TV).
        # Real faces almost never trigger moiré + texture + flat_plane together.
        # Each co-occurring signal adds 12 points to the score.
        screen_indicative_signals = [
            "moire", "texture_degraded", "flat_plane", "screen_border",
            "bg_uniform_motion", "perspective", "high_brightness_screen", "pixel_grid",
        ]
        screen_count = sum(
            1 for s in screen_indicative_signals
            if float(per_signal.get(s, 0.0)) > 0.28
        )
        if screen_count >= 2:
            screen_boost = screen_count * 14.0
            adj += screen_boost
            reasons.append(
                f"screen_evidence_fusion (+{screen_boost:.0f} from {screen_count} co-occurring screen signals)"
            )

        hi_bright = float(per_signal.get("high_brightness_screen", 0.0))
        if hi_bright >= 0.38:
            hb_boost = 16.0 + hi_bright * 20.0
            adj += hb_boost
            reasons.append(f"high_brightness_screen_attack (+{hb_boost:.1f})")

        pixel_g = float(per_signal.get("pixel_grid", 0.0))
        moire_v = float(per_signal.get("moire", 0.0))
        if pixel_g >= 0.32 and hi_bright >= 0.26:
            adj += 12.0
            reasons.append("pixel_grid_with_high_brightness")
        if hi_bright >= 0.42 and moire_v >= 0.28:
            adj += 10.0
            reasons.append("moire_with_high_brightness_screen")

        # Also check for moderate device_replay (YOLO saw device but not overlapping face)
        device_score = max(float(per_signal.get("device_replay", 0.0)), float(extra.get("device_replay", 0.0)))
        if device_score > 0.15 and screen_count >= 1:
            device_boost = device_score * 20.0
            adj += device_boost
            reasons.append(f"single_frame_device_plus_screen (+{device_boost:.1f})")

        bezel = float(extra.get("frame_bezel", 0.0))
        if bezel >= 0.38:
            b_boost = 18.0 + bezel * 22.0
            adj += b_boost
            reasons.append(f"phone_frame_bezel (+{b_boost:.1f})")
        if bezel >= 0.48 and screen_count >= 1:
            adj += 14.0
            reasons.append("bezel_plus_screen_imaging")

        return max(0.0, adj), reasons

    # ── STREAMING MODE (liveness flow): full correlation gate ──
    if adj >= reject_threshold - 12.0:
        if strong_count < min_strong and not device_hi:
            damp = min(25.0, max(12.0, reject_threshold - adj + 18.0))
            adj = min(adj, reject_threshold - damp)
            reasons.append(f"correlation_gate_damp_weak_strong (strong={strong_count}/{min_strong})")

        if only_weak_triggers and strong_count == 0:
            adj = min(adj, reject_threshold - 22.0)
            reasons.append("correlation_gate_weak_only")

        if strong_count == 1 and not device_hi and weak_count <= 2:
            adj = min(adj, reject_threshold - 10.0)
            reasons.append("correlation_gate_single_strong_insufficient")

    # Suppress glare-only scores unless display imaging corroboration exists
    refl_v = float(per_signal.get("reflection", 0.0))
    rect_v = float(per_signal.get("rect_glare", 0.0))
    env_v = float(per_signal.get("environment_authenticity", 0.0))
    extra = extra_signals or {}
    device_v = max(
        float(per_signal.get("device_replay", 0.0)),
        float(extra.get("device_replay", 0.0)),
    )
    display_corr = has_display_attack_corroboration(
        per_signal, triggered, device_replay_score=device_v
    )
    if (refl_v > 0.4 or rect_v > 0.4) and not display_corr:
        glare_contrib = refl_v * 5.0 + rect_v * 10.0
        adj = max(0.0, adj - glare_contrib * 0.65)
        reasons.append("glare_suppression_ambient_only")

    # Window/tube light motion without device/display cues — ignore environment signal
    if env_v > 0.35 and device_v < 0.12 and not display_corr:
        adj = max(0.0, adj - env_v * 12.0)
        reasons.append("environment_authenticity_ambient_suppressed")

    return max(0.0, adj), reasons


def analyze_passive_spoof_single_frame(
    img_bgr: np.ndarray,
    pts_68: Optional[List[Dict[str, Any]]],
    strict: bool = False,
    roi_luminance_history: Optional[List[float]] = None,
    last_gray_small: Optional[np.ndarray] = None,
    curr_gray_small: Optional[np.ndarray] = None,
    landmark_centroid_history: Optional[List[Tuple[float, float]]] = None,
    extra_signals: Optional[Dict[str, float]] = None,
    single_frame_mode: bool = False,
) -> Dict[str, Any]:
    """
    Analyze a single frame for presentation-attack indicators.

    single_frame_mode=True: Used for selfie capture (/match endpoint).
    The correlation gate is relaxed so imaging signals (moiré, texture, scanlines)
    can detect mobile/laptop screen replay without needing temporal signals.
    """
    weights = load_spoof_weights()
    th = load_thresholds()

    raw = _gather_raw_signals(
        img_bgr,
        pts_68,
        strict,
        roi_luminance_history,
        last_gray_small,
        curr_gray_small,
        landmark_centroid_history,
    )
    total, triggered, per_signal = aggregate_weighted_score(raw, weights, strict, extra_signals)
    refl_label = str(raw.get("_refl_label", "unknown"))

    # In single_frame_mode, use a lower threshold — imaging signals are our only tool,
    # but screen evidence fusion boost compensates for the lower individual weights.
    # Real selfie = 5-15 score, Phone screen = 50+ with fusion boost.
    if single_frame_mode:
        reject_threshold = _f("SPOOF_SINGLE_FRAME_THRESHOLD", 42.0)
    else:
        reject_threshold = th["match_total"] if strict else th["reject_total"]

    min_strong = int(max(1.0, th.get("corr_strong_min", 2.0)))
    total, gate_notes = _apply_correlation_gate(
        total, per_signal, extra_signals, triggered, reject_threshold,
        min_strong=min_strong,
        single_frame_mode=single_frame_mode,
        refl_label=refl_label,
    )

    per_signal_clean = {k: round(float(v), 4) for k, v in per_signal.items() if not str(k).startswith("_")}
    per_signal_clean["reflection_label"] = refl_label

    # Glare / ambient-only triggers must not reject (sunlight, tube light, window)
    non_glare_triggers = [
        t for t in triggered
        if t not in ("reflection", "rectangular_glare", "flicker", "environment_authenticity", "perspective")
    ]
    hi_bright_sig = float(per_signal.get("high_brightness_screen", 0.0))
    extra = extra_signals or {}
    device_v = max(
        float(per_signal.get("device_replay", 0.0)),
        float(extra.get("device_replay", 0.0)),
    )
    display_corr = has_display_attack_corroboration(
        per_signal, triggered, device_replay_score=device_v
    )
    if (
        not display_corr
        and (not non_glare_triggers or hi_bright_sig < 0.45)
        and total >= reject_threshold - 12.0
    ):
        total = min(total, reject_threshold - 15.0)

    is_live = total < reject_threshold
    if not is_live and not display_corr:
        is_live = True

    return {
        "total_spoof_score": round(total, 2),
        "reject_threshold": reject_threshold,
        "is_live": is_live,
        "triggered_rules": triggered,
        "confidence_per_signal": per_signal_clean,
        "reflection_classification": refl_label,
        "weights_used": {k: round(v, 2) for k, v in weights.items()},
        "correlation_gate_notes": gate_notes,
        "extra_signals_used": {k: round(float(v), 4) for k, v in (extra_signals or {}).items()},
        "single_frame_mode": single_frame_mode,
    }


def streaming_temporal_decision(
    session_spoof_scores: List[float],
    current_score: float,
) -> Tuple[bool, float]:
    """
    Never reject on a single spike: blend window mean with current.
    Returns (should_flag, smoothed_score).

    REDESIGNED: heavier smoothing, require more consecutive high frames,
    and use a wider temporal window to prevent false spikes.
    """
    window = _i("SPOOF_TEMPORAL_WINDOW", 8)  # Wider window (was 5)
    min_high = _i("SPOOF_TEMPORAL_MIN_HIGH_FRAMES", 5)  # More frames required (was 3)
    th = load_thresholds()["streaming_smooth"]

    buf = list(session_spoof_scores[-window:]) + [current_score]
    # Heavier mean weighting to smooth out spikes (was 0.55/0.45)
    smoothed = float(0.65 * (sum(buf) / len(buf)) + 0.35 * current_score)

    # Require more frames above threshold before flagging
    high_count = sum(1 for s in buf if s >= (th - 6.0))  # Tighter threshold window (was -8.0)
    should = smoothed >= th and high_count >= min_high and len(buf) >= min_high
    return should, smoothed


def centroid_of_landmarks(pts_68: List[Dict[str, Any]]) -> Tuple[float, float]:
    xs = [float(p["x"]) for p in pts_68]
    ys = [float(p["y"]) for p in pts_68]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def downsample_gray(img_bgr: np.ndarray, size: int = 64) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
