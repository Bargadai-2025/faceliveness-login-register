"""
In-memory challenge frame snapshots (gesture phase only).

No disk storage — buffers are cleared after POST /match completes or fails.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_detection import get_face_roi


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


CHALLENGE_SELFIE_MIN_SIM = _f("CHALLENGE_SELFIE_MIN_SIM", 0.58)
CHALLENGE_CROSS_MIN_SIM = _f("CHALLENGE_CROSS_MIN_SIM", 0.62)
CHALLENGE_BG_MIN_SIM = _f("CHALLENGE_BG_MIN_SIM", 0.50)
FACE_RATIO_MIN_FACTOR = _f("CHALLENGE_FACE_RATIO_MIN_FACTOR", 0.42)
FACE_RATIO_MAX_FACTOR = _f("CHALLENGE_FACE_RATIO_MAX_FACTOR", 2.35)
FACE_CROP_SIZE = int(_f("CHALLENGE_FACE_CROP_SIZE", 128))


@dataclass
class ChallengeFrameSnapshot:
    gesture_index: int
    gesture_id: str
    face_jpeg: bytes
    bg_signature: np.ndarray
    face_area_ratio: float
    mean_brightness: float
    devices_in_frame: List[str] = field(default_factory=list)


def _face_bbox_from_pts(pts_68: List[Dict[str, Any]], w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [p["x"] for p in pts_68]
    ys = [p["y"] for p in pts_68]
    x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
    x2, y2 = int(min(w, max(xs))), int(min(h, max(ys)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _face_area_ratio(pts_68: List[Dict[str, Any]], w: int, h: int) -> float:
    _, _, fw, fh = _face_bbox_from_pts(pts_68, w, h)
    return float((fw * fh) / (w * h + 1e-6))


def compute_background_signature(
    img_bgr: np.ndarray,
    pts_68: Optional[List[Dict[str, Any]]],
) -> np.ndarray:
    """Hue histogram of non-face pixels (stable under sunlight / tube light shifts)."""
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (160, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = np.ones((120, 160), dtype=np.uint8) * 255
    if pts_68:
        scale_x = 160 / max(w, 1)
        scale_y = 120 / max(h, 1)
        xs = [int(p["x"] * scale_x) for p in pts_68]
        ys = [int(p["y"] * scale_y) for p in pts_68]
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        ax = int((max(xs) - min(xs)) * 0.65) + 8
        ay = int((max(ys) - min(ys)) * 0.85) + 10
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 0, -1)
    hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    hist = hist.flatten().astype(np.float32)
    s = float(hist.sum()) + 1e-6
    return hist / s


def _bg_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def capture_challenge_frame(
    session: Any,
    img_bgr: np.ndarray,
    pts_68: List[Dict[str, Any]],
    *,
    devices_found: Optional[List[str]] = None,
) -> None:
    """Store one in-memory snapshot when a gesture challenge is completed."""
    h, w = img_bgr.shape[:2]
    roi = get_face_roi(img_bgr, pts_68)
    if roi is None or roi.size == 0:
        return

    crop = cv2.resize(roi, (FACE_CROP_SIZE, FACE_CROP_SIZE), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    snap = ChallengeFrameSnapshot(
        gesture_index=int(session.current_gesture_idx),
        gesture_id=str(session.current_gesture or "unknown"),
        face_jpeg=buf.tobytes(),
        bg_signature=compute_background_signature(img_bgr, pts_68),
        face_area_ratio=_face_area_ratio(pts_68, w, h),
        mean_brightness=float(np.mean(gray)),
        devices_in_frame=list(devices_found or []),
    )
    session.challenge_snapshots.append(snap)

    for d in snap.devices_in_frame:
        if d not in session.challenge_devices_seen:
            session.challenge_devices_seen.append(d)


def clear_challenge_snapshots(session: Any) -> None:
    """Drop all temporary buffers from memory."""
    if hasattr(session, "challenge_snapshots"):
        for snap in session.challenge_snapshots:
            snap.face_jpeg = b""
        session.challenge_snapshots.clear()
    if hasattr(session, "challenge_devices_seen"):
        session.challenge_devices_seen.clear()


def assess_challenge_continuity(
    session: Any,
    selfie_bgr: np.ndarray,
    selfie_emb: np.ndarray,
    face_landmarks_mp: Optional[List[Dict[str, Any]]],
    *,
    mtcnn,
    model,
    device: str,
    selfie_devices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Multi-factor identity / environment continuity (no single hard fail except extreme cases).
    Returns score dict for liveness_risk_engine.compute_identity_continuity_risk().
    """
    snaps: List[ChallengeFrameSnapshot] = getattr(session, "challenge_snapshots", []) or []
    if not snaps:
        return {"skipped": True, "reason": "no_challenge_snapshots"}

    from embedding_pipeline import extract_face_embedding

    h, w = selfie_bgr.shape[:2]
    selfie_ratio = _face_area_ratio(
        [{"x": p["x"], "y": p["y"]} for p in face_landmarks_mp],
        w,
        h,
    ) if face_landmarks_mp else 0.12
    selfie_bg = compute_background_signature(selfie_bgr, face_landmarks_mp)

    similarities: List[float] = []
    bg_sims: List[float] = []
    ratios = [s.face_area_ratio for s in snaps]
    decode_failures = 0

    for snap in snaps:
        arr = np.frombuffer(snap.face_jpeg, dtype=np.uint8)
        crop_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if crop_img is None:
            decode_failures += 1
            continue
        fo = extract_face_embedding(crop_img, mtcnn, model, device)
        if not fo["ok"]:
            decode_failures += 1
            continue
        similarities.append(float(np.dot(fo["embedding"], selfie_emb)))
        bg_sims.append(_bg_similarity(snap.bg_signature, selfie_bg))

    if decode_failures >= len(snaps):
        return {"no_face_in_challenge_crop": True, "skipped": False}

    out: Dict[str, Any] = {"skipped": False, "challenge_frames": len(snaps)}
    if similarities:
        out["min_challenge_selfie_sim"] = min(similarities)
        out["avg_challenge_selfie_sim"] = float(np.mean(similarities))
    if bg_sims:
        out["min_bg_sim"] = min(bg_sims)

    med_ratio = float(np.median(ratios)) if ratios else selfie_ratio
    if med_ratio > 0.02:
        out["face_ratio_drift"] = abs(selfie_ratio - med_ratio) / (med_ratio + 1e-6)

    challenge_devices = set(getattr(session, "challenge_devices_seen", []) or [])
    selfie_devs = set(selfie_devices or [])
    out["device_only_at_capture"] = bool(not challenge_devices and selfie_devs)

    return out


def verify_selfie_against_challenge_frames(
    session: Any,
    selfie_bgr: np.ndarray,
    selfie_emb: np.ndarray,
    face_landmarks_mp: Optional[List[Dict[str, Any]]],
    *,
    mtcnn,
    model,
    device: str,
    selfie_devices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Legacy binary wrapper — prefer assess_challenge_continuity + risk engine.
    """
    assessment = assess_challenge_continuity(
        session,
        selfie_bgr,
        selfie_emb,
        face_landmarks_mp,
        mtcnn=mtcnn,
        model=model,
        device=device,
        selfie_devices=selfie_devices,
    )
    if assessment.get("skipped"):
        return {"ok": True, "reason": "no_challenge_snapshots"}
    if assessment.get("no_face_in_challenge_crop"):
        return {
            "ok": False,
            "error": "User Identity Mismatch — could not verify face during liveness challenges.",
        }

    from liveness_risk_engine import compute_identity_continuity_risk, RISK_REJECT_THRESHOLD

    ident_risk, _ = compute_identity_continuity_risk(assessment)
    if ident_risk >= 85.0:
        return {
            "ok": False,
            "error": "User Identity Mismatch",
            "assessment": assessment,
        }
    return {
        "ok": True,
        "identity_risk": ident_risk,
        "assessment": assessment,
    }
