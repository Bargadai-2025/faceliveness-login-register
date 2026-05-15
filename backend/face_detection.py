"""
Backend face detection & analysis using MediaPipe Tasks API (0.10.x+).
Provides landmark extraction, head pose, EAR/MAR, and expression estimation.
"""
import os
import cv2
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# ── MediaPipe Tasks API ──
MP_AVAILABLE = False
_landmarker = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions

    _model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if os.path.exists(_model_path):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.2, # Lowered from 0.3 for better reliability
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.2,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
        MP_AVAILABLE = True
        print("MediaPipe FaceLandmarker initialized (Tasks API)")
    else:
        print(f"Model file not found at {_model_path}")
except Exception as e:
    print(f"MediaPipe initialization failed: {e}")

# ── Background Removal (Image Segmenter Tasks API) ──
_segmenter = None
try:
    _seg_model_path = os.path.join(os.path.dirname(__file__), "selfie_segmenter.tflite")
    if os.path.exists(_seg_model_path):
        from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
        _seg_options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=_seg_model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        _segmenter = ImageSegmenter.create_from_options(_seg_options)
        print("MediaPipe ImageSegmenter initialized (Tasks API)")
    else:
        print(f"Segmenter model not found at {_seg_model_path}")
except Exception as e:
    print(f"ImageSegmenter load failed: {e}")

def remove_background(img_bgr: np.ndarray, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Remove background and replace with bg_color (default white) using Tasks API."""
    if _segmenter is None or img_bgr is None:
        return img_bgr
        
    try:
        # Convert to RGB and MediaPipe Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Segment
        segmentation_result = _segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()
        
        # In selfie segmenter, confidence mask [0, 1]
        mask = category_mask > 0.5
        
        # Robustness check: If the mask is mostly empty (less than 1% of image),
        # then segmentation likely failed. Return original image.
        if np.sum(mask) < (img_bgr.shape[0] * img_bgr.shape[1] * 0.01):
            print("⚠️ Background removal: No person detected in mask, skipping.")
            return img_bgr

        # Ensure mask is 3D (H, W, 3) for broadcasting
        if len(mask.shape) == 2:
            mask_stack = np.stack((mask,) * 3, axis=-1)
        else:
            mask_stack = np.repeat(mask, 3, axis=-1)
        
        # Create background image
        bg_img = np.full(img_bgr.shape, bg_color, dtype=np.uint8)
        
        # Blend
        out = np.where(mask_stack, img_bgr, bg_img)
        return out
    except Exception as e:
        print(f"Background removal error: {e}")
        return img_bgr

# ── 478-pt to 68-pt landmark mapping (MediaPipe Tasks uses 478 points) ──
MP_TO_68 = {
    # Jaw contour (0-16) - Using standard MediaPipe indices for 68-landmark emulation
    0: 234, 1: 93, 2: 132, 3: 58, 4: 172, 5: 136, 6: 150, 7: 149, 8: 152,
    9: 378, 10: 379, 11: 365, 12: 397, 13: 288, 14: 361, 15: 323, 16: 454,
    # Left eyebrow (17-21)
    17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
    # Right eyebrow (22-26)
    22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
    # Nose bridge + tip (27-35)
    27: 168, 28: 6, 29: 197, 30: 195,
    31: 48, 32: 115, 33: 220, 34: 45, 35: 275,
    # Left eye (36-41)
    36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
    # Right eye (42-47)
    42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
    # Outer mouth (48-59)
    48: 61, 49: 39, 50: 37, 51: 0, 52: 267, 53: 269, 54: 291,
    55: 405, 56: 314, 57: 17, 58: 84, 59: 181,
    # Inner mouth (60-67)
    60: 78, 61: 82, 62: 13, 63: 312, 64: 308, 65: 317, 66: 14, 67: 87,
}

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye corner
    (225.0, 170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0), # Right mouth corner
], dtype=np.float64)


def decode_frame(raw_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw JPEG bytes into a BGR numpy array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_faces(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Detect faces and return 68-pt landmarks using MediaPipe Tasks API."""
    if not MP_AVAILABLE or _landmarker is None:
        print("⚠️ MediaPipe not available or landmarker not initialized")
        return []

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _do_detect(rgb_data):
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_data)
            return _landmarker.detect(mp_image)
        except Exception as e:
            print(f"FaceLandmarker detect error: {e}")
            return None

    result = _do_detect(img_rgb)
    
    # --- FALLBACK 1: Try with Contrast Enhancement (CLAHE) if no face found ---
    if not result or not result.face_landmarks:
        # print("🔍 No face found with standard image, trying contrast enhancement...")
        # Apply CLAHE to L channel in LAB space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        result = _do_detect(enhanced_rgb)
        if result and result.face_landmarks:
            print("✅ Face detected after CLAHE enhancement")

    faces = []
    if result and result.face_landmarks:
        # print(f"✅ Detected {len(result.face_landmarks)} faces")
        for face_lm in result.face_landmarks:
            # Convert normalized landmarks to pixel coordinates
            pts_478 = []
            for lm in face_lm:
                pts_478.append({"x": lm.x * w, "y": lm.y * h, "z": lm.z * w})

            # Map 478 -> 68
            pts_68 = []
            for i in range(68):
                mp_idx = MP_TO_68.get(i, 0)
                if mp_idx < len(pts_478):
                    pts_68.append(pts_478[mp_idx])
                else:
                    pts_68.append({"x": 0, "y": 0, "z": 0})

            face_width = abs(pts_68[16]["x"] - pts_68[0]["x"])
            faces.append({
                "pts_478": pts_478,
                "pts_68": pts_68,
                "face_width": face_width,
            })

    return faces


def compute_ear(pts_68: List[Dict], start_idx: int) -> float:
    """Compute Eye Aspect Ratio for an eye starting at start_idx."""
    p = lambda i: pts_68[start_idx + i]
    dist = lambda a, b: math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    v1 = dist(p(1), p(5))
    v2 = dist(p(2), p(4))
    h = dist(p(0), p(3))
    return (v1 + v2) / (2 * h + 1e-6)


def compute_mar(pts_68: List[Dict]) -> float:
    """Compute Mouth Aspect Ratio."""
    dist = lambda a, b: math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    v = dist(pts_68[62], pts_68[66])
    h = dist(pts_68[48], pts_68[54])
    return v / (h + 1e-6)


def compute_head_pose(pts_68: List[Dict], frame_shape: Tuple[int, int]) -> Dict[str, float]:
    """Estimate head pose using solvePnP."""
    h, w = frame_shape
    image_points = np.array([
        (pts_68[30]["x"], pts_68[30]["y"]),
        (pts_68[8]["x"], pts_68[8]["y"]),
        (pts_68[36]["x"], pts_68[36]["y"]),
        (pts_68[45]["x"], pts_68[45]["y"]),
        (pts_68[48]["x"], pts_68[48]["y"]),
        (pts_68[54]["x"], pts_68[54]["y"]),
    ], dtype=np.float64)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix,
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return {"yaw": 0, "pitch": 0, "roll": 0}
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return {"yaw": float(angles[1]), "pitch": float(angles[0]), "roll": float(angles[2])}


def compute_eye_tilt_rad(pts_68: List[Dict]) -> float:
    """Compute the tilt angle between the two eye corners."""
    return math.atan2(
        pts_68[45]["y"] - pts_68[36]["y"],
        pts_68[45]["x"] - pts_68[36]["x"]
    )


def compute_brow_heights(pts_68: List[Dict]) -> Tuple[float, float]:
    """Return (left_brow_y, right_brow_y) averages."""
    return (
        (pts_68[19]["y"] + pts_68[21]["y"]) / 2,
        (pts_68[24]["y"] + pts_68[26]["y"]) / 2,
    )


def compute_lip_pucker_ratio(pts_68: List[Dict]) -> float:
    """Ratio of lip height to width -- high = puckered."""
    dist = lambda a, b: math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    lip_w = dist(pts_68[48], pts_68[54])
    lip_h = dist(pts_68[51], pts_68[57])
    return lip_h / (lip_w + 1e-6)


def estimate_expression(pts_68: List[Dict], baseline: Optional[Dict] = None) -> Dict[str, float]:
    """Estimate smile and surprise scores from landmarks."""
    mar = compute_mar(pts_68)
    face_w = abs(pts_68[16]["x"] - pts_68[0]["x"]) + 1e-6

    mouth_center_y = (pts_68[62]["y"] + pts_68[66]["y"]) / 2
    corner_avg_y = (pts_68[48]["y"] + pts_68[54]["y"]) / 2
    # In a smile, corners move UP (lower Y) relative to the center.
    smile_score = max(0, (mouth_center_y - corner_avg_y)) / (face_w * 0.05 + 1e-6)

    return {
        "mar": mar,
        "smile_score": min(smile_score, 1.0),
        "surprised": min(max(0, mar - 0.3) * 2, 1.0),
    }


def check_black_screen(img_bgr: np.ndarray, threshold: float = 45) -> bool:
    """Return True if the image is mostly black (camera blocked)."""
    small = cv2.resize(img_bgr, (64, 64))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def compute_passive_liveness(img_bgr: np.ndarray, pts_68: Optional[List[Dict]] = None, strict: bool = False) -> Tuple[bool, float, str]:
    """
    Delegates to weighted multi-signal spoof scoring (see spoof_scoring.py).
    Returns (is_live, health_score 0..1, reason).
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, 0.0, "Empty image"
    from spoof_scoring import analyze_passive_spoof_single_frame

    rep = analyze_passive_spoof_single_frame(img_bgr, pts_68, strict=strict)
    health = max(0.0, min(1.0, 1.0 - rep["total_spoof_score"] / 100.0))
    is_live = bool(rep["is_live"])
    if is_live:
        reason = "OK"
    else:
        rules = ", ".join(rep.get("triggered_rules") or [])[:200]
        reason = f"Spoof score {rep['total_spoof_score']:.0f}/{rep.get('reject_threshold', 70):.0f}"
        if rules:
            reason += f" ({rules})"
    return is_live, round(health, 3), reason


def compute_brightness_histogram(img_bgr: np.ndarray) -> Dict[str, float]:
    """Compute brightness stats for light-challenge verification."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return {
        "brightness": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
        "saturation": float(np.mean(hsv[:, :, 1])),
    }


def landmarks_to_serializable(pts_68: List[Dict]) -> List[Dict[str, float]]:
    """Convert landmark list to JSON-safe format."""
    return [{"x": round(p["x"], 1), "y": round(p["y"], 1)} for p in pts_68]


# ── Anti-Screen / Emissive Light Utils ──

def get_face_roi(img: np.ndarray, pts_68: List[Dict]) -> Optional[np.ndarray]:
    """Extract face region of interest based on landmarks."""
    if not pts_68: return None
    xs = [p["x"] for p in pts_68]
    ys = [p["y"] for p in pts_68]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    
    # Expand slightly
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * 0.1))
    y1 = max(0, int(y1 - bh * 0.1))
    x2 = min(w, int(x2 + bw * 0.1))
    y2 = min(h, int(y2 + bh * 0.1))
    
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2]

def compute_color_diversity(roi: np.ndarray, bins: int = 16) -> float:
    """Measure unique colors in ROI via quantization."""
    if roi is None: return 1.0
    # Downsample for speed
    small = cv2.resize(roi, (64, 64))
    # Quantize
    quantized = (small // (256 // bins))
    unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
    # Max possible unique colors with bins^3
    max_colors = bins ** 3
    return unique_colors / 1000.0 # Normalized score

def detect_specular_highlights(roi: np.ndarray) -> float:
    """Detect saturated pixel clusters typical of screen glare."""
    if roi is None: return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    saturated = v_channel > 245
    
    # Connected components to find cluster sizes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(saturated.astype(np.uint8))
    if num_labels <= 1: return 0.0
    
    # Area of the largest cluster (excluding background label 0)
    max_area = np.max(stats[1:, cv2.CC_STAT_AREA]) if len(stats) > 1 else 0
    roi_area = roi.shape[0] * roi.shape[1]
    return max_area / (roi_area + 1e-6)

def detect_peak_saturation(roi: np.ndarray) -> float:
    """Detect 'digital peaks' (R=G=B=255) which are rare on real skin but common on screens."""
    if roi is None: return 0.0
    # Count pixels that are extremely close to pure white (255, 255, 255)
    # Screens show these specular peaks much more intensely than skin.
    pure_white = np.all(roi > 250, axis=-1)
    ratio = np.sum(pure_white) / (roi.shape[0] * roi.shape[1] + 1e-6)
    return float(ratio)

def detect_scanline_artifacts(img_bgr: np.ndarray) -> float:
    """
    Detect periodic horizontal banding (scanlines) typical of screen recordings.
    Computes row-wise intensity variance to find periodic spikes.
    """
    if img_bgr is None: return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculate average intensity per row
    row_means = np.mean(gray, axis=1)
    
    # Detrend using a simple difference
    diff = np.diff(row_means)
    
    # Look for periodic oscillations in the row means (moiré/banding)
    # Screens show high variance in these differences
    var = float(np.var(diff))
    
    # Normalized score: high variance in row differences = scanlines
    return min(var / 15.0, 1.0)

def check_emissive_uniformity(roi: np.ndarray) -> Tuple[bool, float]:
    """
    Measure local variance in patches. 
    Screens show low micro-variance + high mean brightness (uniform emissive panel).
    """
    if roi is None: return False, 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    patch_size = 5
    
    variances = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            variances.append(np.var(patch))
            
    mean_var = np.mean(variances) if variances else 100.0
    mean_brightness = np.mean(gray)
    
    # Screen signature: High brightness + unnatural smoothness (low variance)
    # In daylight, we need to be careful not to trigger this. 
    # Real skin has texture (pores, hair) even in bright light.
    # Digital screens are perfectly uniform.
    is_emissive = mean_brightness > 195 and mean_var < 14.0 # Relaxed from 175/18.0
    return is_emissive, round(float(mean_var), 3)

def check_screen_edges(img: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> bool:
    """
    Detect rectangular contours (bezel edges) near the face.
    """
    if img is None: return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Focus on area around face
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # Look for rectangles
        if len(approx) == 4:
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect_ratio = float(bw) / (bh + 1e-6)
            area_pct = (bw * bh) / (w * h)
            
            # Typical screen aspect ratios [1.2, 2.2] or square-ish [0.8, 1.2] for older phones
            if 0.8 < aspect_ratio < 2.5 and area_pct > 0.08:
                # Check if face is inside or very near this rectangle
                fx, fy, fw, fh = face_bbox
                # If face center is inside the rectangle or rectangle is very close to face
                if x < (fx + fw/2) < (x + bw) and y < (fy + fh/2) < (y + bh):
                    return True
    return False
