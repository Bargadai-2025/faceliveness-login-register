"""
Canonical face detection + FaceNet embedding pipeline.

All paths (indexing, /match, /register) must use this module so gallery and query
embeddings are comparable. Mismatch between build_db (800px) and /match (1024px)
was a source of local vs production score drift.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Must match value used when indexing bargad/lfw (build_db.py historically used 800).
FACE_DETECT_MAX_SIDE = int(os.getenv("FACE_DETECT_MAX_SIDE", "800"))
MTCNN_IMAGE_SIZE = 160
MTCNN_MARGIN = 20
EMBEDDING_DIM = 512


def create_face_models(device: str):
    mtcnn = MTCNN(
        image_size=MTCNN_IMAGE_SIZE,
        margin=MTCNN_MARGIN,
        device=device,
    )
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, model


def resize_for_detection(img_bgr: np.ndarray, max_side: int = FACE_DETECT_MAX_SIDE) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_side:
        return img_bgr
    scale = max_side / max(h, w)
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)))


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    emb = emb.astype("float32")
    return emb / (np.linalg.norm(emb) + 1e-6)


def detect_face_tensor(
    img_bgr: np.ndarray,
    mtcnn: MTCNN,
    *,
    max_side: int = FACE_DETECT_MAX_SIDE,
    mp_fallback: bool = True,
) -> Tuple[Optional[torch.Tensor], np.ndarray, Optional[list]]:
    """
    Returns (face_tensor, img_detect_bgr, face_landmarks_mp_scaled_to_original).
    Landmarks are scaled to original image coordinates when detection was resized.
    """
    img_detect = resize_for_detection(img_bgr, max_side)
    img_detect_rgb = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_detect_rgb)

    face_landmarks_mp = None
    mp_faces = None
    if mp_fallback:
        from face_detection import detect_faces

        mp_faces = detect_faces(img_detect)
        if mp_faces:
            orig_h, orig_w = img_bgr.shape[:2]
            det_h, det_w = img_detect.shape[:2]
            scale_x = orig_w / det_w
            scale_y = orig_h / det_h
            face_landmarks_mp = [
                {"x": p["x"] * scale_x, "y": p["y"] * scale_y}
                for p in mp_faces[0]["pts_68"]
            ]

    if face is None and mp_fallback and mp_faces:
        pts = mp_faces[0]["pts_68"]
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        w_b, h_b = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w_b * 0.3))
        y1 = max(0, int(y1 - h_b * 0.3))
        x2 = min(img_detect.shape[1], int(x2 + w_b * 0.3))
        y2 = min(img_detect.shape[0], int(y2 + h_b * 0.3))
        crop_rgb = cv2.cvtColor(img_detect[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        face = mtcnn(crop_rgb)

    return face, img_detect, face_landmarks_mp


def embedding_from_face_tensor(
    face: torch.Tensor,
    model: InceptionResnetV1,
    device: str,
) -> np.ndarray:
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face).cpu().numpy()[0]
    if np.isnan(emb).any():
        raise ValueError("Face analysis yielded invalid (NaN) embedding.")
    return normalize_embedding(emb.astype("float32"))


def extract_face_embedding(
    img_bgr: np.ndarray,
    mtcnn: MTCNN,
    model: InceptionResnetV1,
    device: str,
    *,
    max_side: int = FACE_DETECT_MAX_SIDE,
    mp_fallback: bool = True,
) -> Dict[str, Any]:
    face, img_detect, face_landmarks_mp = detect_face_tensor(
        img_bgr, mtcnn, max_side=max_side, mp_fallback=mp_fallback
    )
    if face is None:
        return {
            "ok": False,
            "error": "No face detected.",
            "embedding": None,
            "face_landmarks_mp": face_landmarks_mp,
            "img_detect": img_detect,
        }
    emb = embedding_from_face_tensor(face, model, device)
    return {
        "ok": True,
        "error": None,
        "embedding": emb,
        "face_landmarks_mp": face_landmarks_mp,
        "img_detect": img_detect,
    }


def load_match_thresholds() -> Dict[str, float]:
    """Env-overridable cosine thresholds (after L2 normalization)."""
    def _f(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        return float(raw) if raw else default

    return {
        "bargad": _f("MATCH_MIN_CONFIDENCE_BARGAD", 0.72),
        "lfw": _f("MATCH_MIN_CONFIDENCE_LFW", 0.62),
        "top2_margin": _f("MATCH_MIN_TOP2_MARGIN", 0.05),
    }
