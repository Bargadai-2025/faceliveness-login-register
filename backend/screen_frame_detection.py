"""
Full-frame cues for phone / monitor replay at selfie capture (POST /match only).
Does not run during liveness stream — avoids window-light false positives.
"""
from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


BEZEL_SIDE_FRAC = _f("MATCH_BEZEL_SIDE_FRAC", 0.075)
BEZEL_DARK_RATIO = _f("MATCH_BEZEL_DARK_RATIO", 0.72)
BEZEL_ABS_DARK = _f("MATCH_BEZEL_ABS_DARK", 88.0)


def detect_phone_frame_bezels(img_bgr: np.ndarray) -> float:
    """
    Score 0..1 when dark vertical bands (phone bezels) flank a brighter center — typical
    photo-of-phone composition. Ignores pure window glare (no symmetric side bands).
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    h, w = img_bgr.shape[:2]
    if w < 80 or h < 80:
        return 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mx = max(4, int(w * BEZEL_SIDE_FRAC))
    my = max(3, int(h * 0.04))

    left = gray[:, :mx]
    right = gray[:, w - mx :]
    center = gray[my : h - my, mx : w - mx]
    if center.size == 0:
        return 0.0

    left_m = float(np.mean(left))
    right_m = float(np.mean(right))
    center_m = float(np.mean(center))

    side_dark_rel = (
        left_m < center_m * BEZEL_DARK_RATIO and right_m < center_m * BEZEL_DARK_RATIO
    )
    side_dark_abs = left_m < BEZEL_ABS_DARK and right_m < BEZEL_ABS_DARK
    if not (side_dark_rel or side_dark_abs):
        return 0.0

    # Uniform dark bands (plastic bezel) vs noisy shadows
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    uniform = left_std < 42.0 and right_std < 42.0

    contrast = max(0.0, (center_m - min(left_m, right_m)) / (center_m + 1e-6))
    score = 0.35
    if side_dark_rel:
        score += 0.25 * min(1.0, contrast / 0.35)
    if side_dark_abs:
        score += 0.20
    if uniform:
        score += 0.15
    if contrast > 0.45:
        score += 0.10

    return float(min(1.0, score))


def detect_fullframe_screen_border(img_bgr: np.ndarray) -> float:
    """
    Rectangular bright region inset in frame (monitor / phone screen area).
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    margin = int(min(h, w) * 0.04)
    border_band = np.zeros_like(edges)
    border_band[:margin, :] = 255
    border_band[h - margin :, :] = 255
    border_band[:, :margin] = 255
    border_band[:, w - margin :] = 255
    inner = edges[margin : h - margin, margin : w - margin]
    if inner.size == 0:
        return 0.0
    inner_density = float(np.mean(inner > 0))
    outer_density = float(np.mean((edges > 0) & (border_band > 0))) if np.any(border_band) else 0.0
    if inner_density < 0.02 or outer_density < 0.01:
        return 0.0
    ratio = inner_density / (outer_density + 1e-6)
    if ratio < 1.2:
        return 0.0
    return float(min(1.0, 0.25 + 0.35 * min(2.5, ratio - 1.0)))


def analyze_match_frame_context(img_bgr: np.ndarray) -> Tuple[float, float]:
    """Returns (bezel_score, screen_border_score) in 0..1."""
    b = detect_phone_frame_bezels(img_bgr)
    s = detect_fullframe_screen_border(img_bgr)
    return b, s
