"""
Full-frame digital screen / phone replay analysis for POST /match selfies.

Normalizes via PNG, samples a 3×3 grid at multiple scales, and fuses moiré, blur,
banding, bezel, and lighting-uniformity cues before a single replay verdict.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from screen_frame_detection import (
    analyze_match_frame_context,
    detect_fullframe_screen_border,
    detect_phone_frame_bezels,
)
from spoof_scoring import signal_moire


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MATCH_FULLFRAME_HARD_REJECT = _f("MATCH_FULLFRAME_HARD_REJECT", 0.58)
MATCH_FULLFRAME_SCREEN_REPLAY = _f("MATCH_FULLFRAME_SCREEN_REPLAY", 0.50)
MATCH_FULLFRAME_MIN_SIGNALS = int(_f("MATCH_FULLFRAME_MIN_SIGNALS", 2))

STRUCTURAL_FULLFRAME_SIGNALS = frozenset({
    "phone_bezel",
    "screen_border",
    "scanline_banding",
    "blur_plus_moire",
    "soft_face_on_screen",
})

# Visible phone/monitor frame — reliable replay cue; artifact signals alone false-positive on webcam.
PHYSICAL_FULLFRAME_SIGNALS = frozenset({
    "phone_bezel",
    "screen_border",
})

ARTIFACT_FULLFRAME_SIGNALS = frozenset({
    "scanline_banding",
    "blur_plus_moire",
    "soft_face_on_screen",
})

SOFT_FULLFRAME_SIGNALS = frozenset({
    "moire",
    "multi_region_moire",
    "flat_panel_lighting",
    "flat_regions",
})


def has_structural_fullframe_signals(signals) -> bool:
    return bool(set(signals or []) & STRUCTURAL_FULLFRAME_SIGNALS)


def has_physical_fullframe_signals(signals) -> bool:
    return bool(set(signals or []) & PHYSICAL_FULLFRAME_SIGNALS)


def has_artifact_fullframe_signals(signals) -> bool:
    return bool(set(signals or []) & ARTIFACT_FULLFRAME_SIGNALS)


def normalize_via_png(img_bgr: np.ndarray) -> np.ndarray:
    """Decode/encode PNG cycle — stabilizes compression artifacts for analysis."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return img_bgr
    arr = np.frombuffer(buf, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img_bgr


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    return cv2.resize(
        img,
        (max(8, int(w * scale)), max(8, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _horizontal_banding_score(gray: np.ndarray) -> float:
    """Periodic horizontal rows — LCD/OLED refresh on photographed screens."""
    if gray is None or gray.size == 0 or gray.shape[0] < 16:
        return 0.0
    small = cv2.resize(gray, (min(256, gray.shape[1]), min(128, gray.shape[0])))
    row_means = np.mean(small.astype(np.float32), axis=1)
    if row_means.size < 8:
        return 0.0
    d = np.diff(row_means)
    if d.size < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(d))
    if spec.size < 3:
        return 0.0
    spec[0] = 0.0
    peak = float(np.max(spec))
    mean = float(np.mean(spec) + 1e-6)
    ratio = peak / mean
    return float(min(1.0, max(0.0, (ratio - 2.8) / 4.5)))


def _region_flat_light_score(gray: np.ndarray) -> float:
    """Emissive flat panels: very low local contrast + mid-high brightness."""
    if gray is None or gray.size == 0:
        return 0.0
    mean_b = float(np.mean(gray))
    std_b = float(np.std(gray))
    if std_b > 38.0:
        return 0.0
    flat = max(0.0, (32.0 - std_b) / 32.0)
    bright = min(1.0, max(0.0, (mean_b - 70.0) / 120.0))
    return float(min(1.0, flat * (0.45 + 0.55 * bright)))


def _analyze_patch(gray: np.ndarray) -> Dict[str, float]:
    if gray is None or gray.size == 0:
        return {"moire": 0.0, "blur_low": 0.0, "banding": 0.0, "flat_light": 0.0}
    lap = _laplacian_variance(gray)
    blur_low = float(min(1.0, max(0.0, (180.0 - lap) / 180.0)))
    return {
        "moire": signal_moire(gray),
        "blur_low": blur_low,
        "banding": _horizontal_banding_score(gray),
        "flat_light": _region_flat_light_score(gray),
    }


def _grid_region_scores(img_bgr: np.ndarray, rows: int = 3, cols: int = 3) -> List[Dict[str, float]]:
    h, w = img_bgr.shape[:2]
    out: List[Dict[str, float]] = []
    for ri in range(rows):
        for ci in range(cols):
            y1 = int(h * ri / rows)
            y2 = int(h * (ri + 1) / rows)
            x1 = int(w * ci / cols)
            x2 = int(w * (ci + 1) / cols)
            patch = img_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            out.append(_analyze_patch(gray))
    return out


def _aggregate_regions(regions: List[Dict[str, float]]) -> Dict[str, float]:
    if not regions:
        return {"moire_max": 0.0, "moire_mean": 0.0, "blur_mean": 0.0, "banding_max": 0.0, "flat_mean": 0.0}
    moires = [r["moire"] for r in regions]
    blurs = [r["blur_low"] for r in regions]
    bands = [r["banding"] for r in regions]
    flats = [r["flat_light"] for r in regions]
    return {
        "moire_max": float(max(moires)),
        "moire_mean": float(np.mean(moires)),
        "blur_mean": float(np.mean(blurs)),
        "banding_max": float(max(bands)),
        "flat_mean": float(np.mean(flats)),
        "moire_regions_high": float(sum(1 for m in moires if m >= 0.26) / len(moires)),
        "banding_regions_high": float(sum(1 for b in bands if b >= 0.28) / len(bands)),
    }


def _lighting_uniformity(img_bgr: np.ndarray) -> float:
    """Screens: similar brightness across corners vs center (unlike natural room light)."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    means = []
    for (y1f, y2f, x1f, x2f) in (
        (0.0, 0.33, 0.0, 0.33),
        (0.0, 0.33, 0.67, 1.0),
        (0.67, 1.0, 0.0, 0.33),
        (0.67, 1.0, 0.67, 1.0),
        (0.33, 0.67, 0.33, 0.67),
    ):
        y1, y2 = int(h * y1f), int(h * y2f)
        x1, x2 = int(w * x1f), int(w * x2f)
        patch = gray[y1:y2, x1:x2]
        if patch.size:
            means.append(float(np.mean(patch)))
    if len(means) < 3:
        return 0.0
    spread = float(np.std(means))
    # Very uniform across quadrants → likely flat display
    if spread < 12.0:
        return float(min(1.0, (14.0 - spread) / 14.0))
    return 0.0


def analyze_fullframe_screen_replay(
    img_bgr: np.ndarray,
    *,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Multi-angle full-frame screen replay analysis.
    Returns scores 0..1 and a fused replay_likelihood for match security.
    """
    empty: Dict[str, Any] = {
        "replay_likelihood": 0.0,
        "signal_count": 0,
        "signals": [],
        "bezel_score": 0.0,
        "screen_border_score": 0.0,
    }
    if img_bgr is None or img_bgr.size == 0:
        return empty

    normalized = normalize_via_png(img_bgr)
    work = _resize_max_side(normalized, 720)

    bezel, border = analyze_match_frame_context(work)
    scales: List[Dict[str, float]] = []
    for max_side in (720, 480, 320):
        scaled = _resize_max_side(normalized, max_side)
        regions = _grid_region_scores(scaled, 3, 3)
        agg = _aggregate_regions(regions)
        agg["scale"] = float(max_side)
        scales.append(agg)

    best = scales[0] if scales else {}
    for s in scales[1:]:
        for key in ("moire_max", "moire_mean", "banding_max", "blur_mean", "flat_mean"):
            best[key] = max(float(best.get(key, 0)), float(s.get(key, 0)))

    gray_full = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    global_moire = signal_moire(gray_full)
    global_banding = _horizontal_banding_score(gray_full)
    global_blur = float(min(1.0, max(0.0, (200.0 - _laplacian_variance(gray_full)) / 200.0)))
    light_uniform = _lighting_uniformity(work)

    face_blur = 0.0
    if face_bbox:
        fx, fy, fw, fh = face_bbox
        h, w = work.shape[:2]
        x1, y1 = max(0, fx), max(0, fy)
        x2, y2 = min(w, fx + fw), min(h, fy + fh)
        face_gray = gray_full[y1:y2, x1:x2]
        if face_gray.size > 0:
            face_lap = _laplacian_variance(face_gray)
            frame_lap = _laplacian_variance(gray_full) + 1e-6
            # Face softer than frame — common when face is on a phone screen in photo
            if face_lap < frame_lap * 0.72:
                face_blur = float(min(1.0, (frame_lap * 0.72 - face_lap) / (frame_lap * 0.5 + 1e-6)))

    signals: List[str] = []
    if global_moire >= 0.24 or float(best.get("moire_max", 0)) >= 0.28:
        signals.append("moire")
    if float(best.get("moire_regions_high", 0)) >= 0.44:
        signals.append("multi_region_moire")
    if global_banding >= 0.36 or float(best.get("banding_max", 0)) >= 0.40:
        signals.append("scanline_banding")
    if bezel >= 0.30:
        signals.append("phone_bezel")
    if border >= 0.28:
        signals.append("screen_border")
    if global_blur >= 0.45 and global_moire >= 0.28:
        signals.append("blur_plus_moire")
    if face_blur >= 0.48 and global_moire >= 0.26:
        signals.append("soft_face_on_screen")
    if light_uniform >= 0.42:
        signals.append("flat_panel_lighting")
    if float(best.get("flat_mean", 0)) >= 0.40:
        signals.append("flat_regions")

    replay_likelihood = (
        0.10 * max(global_moire, float(best.get("moire_max", 0)))
        + 0.08 * float(best.get("moire_mean", 0))
        + 0.14 * max(global_banding, float(best.get("banding_max", 0)))
        + 0.18 * bezel
        + 0.14 * border
        + 0.06 * global_blur
        + 0.12 * face_blur
        + 0.04 * light_uniform
    )
    struct_count = len(signals)
    if has_structural_fullframe_signals(signals):
        replay_likelihood += 0.08 * struct_count
    elif struct_count >= 2:
        replay_likelihood += 0.03 * (struct_count - 1)
    replay_likelihood = float(min(1.0, replay_likelihood))

    return {
        "replay_likelihood": replay_likelihood,
        "signal_count": len(signals),
        "signals": signals,
        "bezel_score": bezel,
        "screen_border_score": border,
        "global_moire": round(global_moire, 3),
        "global_banding": round(global_banding, 3),
        "global_blur": round(global_blur, 3),
        "face_blur": round(face_blur, 3),
        "light_uniformity": round(light_uniform, 3),
        "region_moire_max": round(float(best.get("moire_max", 0)), 3),
        "region_moire_mean": round(float(best.get("moire_mean", 0)), 3),
        "scales_analyzed": [int(s.get("scale", 0)) for s in scales],
    }


def is_fullframe_screen_replay(
    report: Dict[str, Any],
    *,
    liveness_verified: bool = False,
) -> bool:
    """Hard screen-replay — physical frame cues required when liveness already passed."""
    score = float(report.get("replay_likelihood", 0))
    n_sig = int(report.get("signal_count", 0))
    signals = set(report.get("signals") or [])
    bezel = float(report.get("bezel_score", 0.0))
    border = float(report.get("screen_border_score", 0.0))
    has_physical = has_physical_fullframe_signals(signals) or bezel >= 0.45 or border >= 0.42
    has_struct = has_structural_fullframe_signals(signals)

    if liveness_verified and not has_physical:
        return False
    if not has_struct:
        return False

    if score >= MATCH_FULLFRAME_HARD_REJECT and has_physical:
        return True
    if (
        score >= MATCH_FULLFRAME_SCREEN_REPLAY
        and n_sig >= MATCH_FULLFRAME_MIN_SIGNALS
        and has_physical
    ):
        return True
    if "phone_bezel" in signals and score >= 0.48:
        return True
    if "screen_border" in signals and score >= 0.50 and border >= 0.35:
        return True
    if "soft_face_on_screen" in signals and score >= 0.58 and has_physical:
        return True
    if "blur_plus_moire" in signals and score >= 0.55 and has_physical:
        return True
    return False
