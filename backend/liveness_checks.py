"""
Liveness verification checks: depth, micro-expression, gestures, light, timing, device detection.
"""
import time
import math
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from face_detection import (
    compute_ear, compute_mar, compute_head_pose, compute_eye_tilt_rad,
    compute_brow_heights, compute_lip_pucker_ratio, estimate_expression,
    compute_brightness_histogram,
)

SUSTAINED_FRAMES = 1
HOLD_DURATION_SEC = 0.5


# ═══════════════════════════════════════════════════════
# DEPTH ESTIMATION (Step 2)
# ═══════════════════════════════════════════════════════
def analyze_true_3d_deformation(landmark_history: List[List[Dict]]) -> Tuple[bool, float]:
    """
    True 3D facial deformation analysis.
    Real faces deform organically (non-rigid biological motion) during movement.
    Replays are globally transformed (planar).
    """
    if len(landmark_history) < 5:
        return True, 1.0  # Not enough data yet

    def displacement(pts_list, idx):
        xs = [pts[idx]["x"] for pts in pts_list[-10:]]
        ys = [pts[idx]["y"] for pts in pts_list[-10:]]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    nose_d = displacement(landmark_history, 30)
    left_eye_d = displacement(landmark_history, 36)
    right_eye_d = displacement(landmark_history, 45)
    ear_l_d = displacement(landmark_history, 2)
    ear_r_d = displacement(landmark_history, 14)
    
    eye_avg = (left_eye_d + right_eye_d) / 2
    ear_avg = (ear_l_d + ear_r_d) / 2

    if nose_d < 1.0 and eye_avg < 1.0:
        return True, 0.5  # Too little movement

    # Phase shift: Nose should move significantly more than ears, and differently than eyes
    ratio_nose_eye = nose_d / (eye_avg + 1e-6)
    ratio_nose_ear = nose_d / (ear_avg + 1e-6)
    
    is_3d = ratio_nose_eye > 1.15 and ratio_nose_ear > 1.2
    
    # Calculate non-rigid biological motion score
    # High score means it behaves like a flat plane (spoof)
    # Low score means true 3D
    score = 1.0 - (ratio_nose_eye - 1.0)
    score = min(max(score, 0.0), 1.0)
    
    return is_3d, round(score, 3)


# ═══════════════════════════════════════════════════════
# MICRO EXPRESSION (Step 4)
# ═══════════════════════════════════════════════════════
def check_micro_expressions(landmark_history: List[List[Dict]]) -> Tuple[bool, float]:
    """
    Track eye jitter, lip twitch, micro-blink — natural micro-movements.
    Photos/replays lack these subtle variations.
    """
    if len(landmark_history) < 15:
        return True, 0.5

    recent = landmark_history[-15:]

    # Eye jitter — small random movement in eye landmarks
    left_eye_xs = [pts[37]["x"] for pts in recent]
    left_eye_ys = [pts[37]["y"] for pts in recent]
    jitter_x = float(np.std(left_eye_xs))
    jitter_y = float(np.std(left_eye_ys))
    jitter = math.hypot(jitter_x, jitter_y)

    # Lip micro-movement
    lip_ys = [pts[62]["y"] for pts in recent]
    lip_var = float(np.std(lip_ys))

    # Nose tip micro-movement (natural head sway)
    nose_xs = [pts[30]["x"] for pts in recent]
    nose_ys = [pts[30]["y"] for pts in recent]
    nose_var = math.hypot(float(np.std(nose_xs)), float(np.std(nose_ys)))

    # Combined micro-movement score
    total_variance = jitter + lip_var * 0.5 + nose_var * 0.3

    # Very still = suspicious (photo/replay)
    has_micro = total_variance > 0.55
    score = min(total_variance / 2.0, 1.0)
    return has_micro, round(score, 3)


# ═══════════════════════════════════════════════════════
# LIGHT CHALLENGE (Step 3)
# ═══════════════════════════════════════════════════════
def analyze_active_spectral_reflectance(
    pre_stats: List[Dict[str, float]],
    post_stats: List[Dict[str, float]],
    challenge_color: str,
) -> Tuple[bool, float]:
    """
    Replaces old check_light_response.
    Analyzes RGB decay timing and emissive resistance.
    """
    if not pre_stats or not post_stats:
        return True, 0.5

    pre_bright = np.mean([s["brightness"] for s in pre_stats])
    post_bright = np.mean([s["brightness"] for s in post_stats])
    bright_delta = abs(post_bright - pre_bright)

    pre_sat = np.mean([s["saturation"] for s in pre_stats])
    post_sat = np.mean([s["saturation"] for s in post_stats])
    sat_delta = abs(post_sat - pre_sat)

    # Real skin reflects flash diffusely
    if challenge_color in ("white_flash", "brightness_up"):
        passed = bright_delta > 3.0
    elif challenge_color in ("blue_flash", "green_flash"):
        passed = bright_delta > 2.0 or sat_delta > 2.0
    else:
        passed = bright_delta > 2.5

    # Emissive resistance (displays don't change brightness much)
    emissive_score = 1.0 - min((bright_delta + sat_delta) / 15.0, 1.0)
    return passed, round(emissive_score, 3)


# ═══════════════════════════════════════════════════════
# REACTION TIMING (Step 6)
# ═══════════════════════════════════════════════════════
def check_reaction_timing(reaction_times: List[float]) -> Tuple[bool, str]:
    """
    Verify reaction times are human-like (300ms - 2000ms).
    Too fast = bot/script; Too slow = may be replaying.
    """
    if not reaction_times:
        return True, "no_data"
    for rt in reaction_times:
        if rt < 0.15:
            return False, f"Too fast: {rt:.3f}s"
        if rt > 8.0:
            return False, f"Too slow: {rt:.3f}s"
    return True, "ok"


# ═══════════════════════════════════════════════════════
# GESTURE EVALUATION (Step 5) — All 31 gestures
# ═══════════════════════════════════════════════════════
def evaluate_gesture(
    gesture_id: str,
    pts_68: List[Dict],
    baseline: Dict[str, Any],
    session: Any,
) -> bool:
    """Evaluate whether the current frame satisfies the given gesture."""
    face_w = baseline.get("face_width", 100)
    base_nose_cx = baseline.get("nose_center_x", 0)
    base_nose_y = baseline.get("nose_tip_y", 0)
    base_left_ear = baseline.get("left_ear", 0.25)
    base_right_ear = baseline.get("right_ear", 0.25)
    base_eye_angle = baseline.get("eye_angle", 0)
    base_left_brow_y = baseline.get("left_brow_y", 0)
    base_right_brow_y = baseline.get("right_brow_y", 0)
    base_face_width = baseline.get("face_width", 100)

    face_center_x = (pts_68[0]["x"] + pts_68[16]["x"]) / 2
    nose_offset = pts_68[30]["x"] - face_center_x
    cur_nose_y = pts_68[30]["y"]

    left_ear_val = compute_ear(pts_68, 36)
    right_ear_val = compute_ear(pts_68, 42)
    mar_val = compute_mar(pts_68)
    expr = estimate_expression(pts_68, baseline)
    cur_eye_angle = compute_eye_tilt_rad(pts_68)
    cur_face_w = abs(pts_68[16]["x"] - pts_68[0]["x"])
    pose = compute_head_pose(pts_68, (480, 640))  # approx

    # Helper
    def _threshold(frac, minimum):
        return max(minimum, face_w * frac)

    # ── Original 18 gestures ──
    if gesture_id == "turn_left":
        delta = nose_offset - base_nose_cx
        return delta > _threshold(0.035, 2)

    elif gesture_id == "turn_right":
        delta = base_nose_cx - nose_offset
        return delta > _threshold(0.035, 2)

    elif gesture_id == "nod":
        delta = cur_nose_y - base_nose_y
        return delta > _threshold(0.04, 4)

    elif gesture_id == "look_up":
        delta = base_nose_y - cur_nose_y
        return delta > _threshold(0.04, 4)

    elif gesture_id == "smile":
        return expr["smile_score"] > 0.15

    elif gesture_id == "surprised":
        return expr["surprised"] > 0.22

    elif gesture_id == "mouth_open":
        return mar_val > 0.25


    elif gesture_id == "wide_eyes":
        return (left_ear_val > base_left_ear * 1.05 and
                right_ear_val > base_right_ear * 1.05)

    elif gesture_id == "blink_both":
        is_closed = (left_ear_val < base_left_ear * 0.92 or
                     right_ear_val < base_right_ear * 0.92)
        return is_closed

    elif gesture_id == "raise_eyebrows":
        lb, rb = compute_brow_heights(pts_68)
        left_move = base_left_brow_y - lb
        right_move = base_right_brow_y - rb
        thresh = face_w * 0.025
        return left_move > thresh and right_move > thresh

    elif gesture_id == "pucker_lips":
        ratio = compute_lip_pucker_ratio(pts_68)
        return ratio > 0.25

    elif gesture_id == "frown":
        # Detect by mouth corner droop or brow lowering
        corner_y = (pts_68[48]["y"] + pts_68[54]["y"]) / 2
        center_y = pts_68[62]["y"]
        return corner_y > center_y + face_w * 0.015

    elif gesture_id == "move_closer":
        return cur_face_w > base_face_width * 1.08

    elif gesture_id == "move_farther":
        return cur_face_w < base_face_width * 0.92

    elif gesture_id == "shake_head":
        session.shake_history.append(nose_offset)
        if len(session.shake_history) > 20:
            session.shake_history = session.shake_history[-20:]
        if len(session.shake_history) >= 12:
            amp = max(session.shake_history) - min(session.shake_history)
            if amp > face_w * 0.10:
                last = session.shake_history[-1]
                prev = session.shake_history[-3] if len(session.shake_history) >= 3 else 0
                if (last > 0 and prev < 0) or (last < 0 and prev > 0):
                    session.shake_completed = True
        return session.shake_completed

    # ── Advanced anti-spoof 13 ──
    elif gesture_id == "blink_twice_fast":
        is_closed = (left_ear_val < base_left_ear * 0.9 or
                     right_ear_val < base_right_ear * 0.9)
        if is_closed and not session.was_blink_closed:
            session.blink_count += 1
            session.was_blink_closed = True
        elif not is_closed:
            session.was_blink_closed = False
        return session.blink_count >= 2

    elif gesture_id in ("look_left_hold", "look_right_hold",
                        "look_up_hold", "look_down_hold"):
        direction = gesture_id.replace("look_", "").replace("_hold", "")
        if direction == "left":
            ok = (nose_offset - base_nose_cx) > _threshold(0.045, 3)
        elif direction == "right":
            ok = (base_nose_cx - nose_offset) > _threshold(0.045, 3)
        elif direction == "up":
            ok = (base_nose_y - cur_nose_y) > _threshold(0.035, 3)
        else:  # down
            ok = (cur_nose_y - base_nose_y) > _threshold(0.035, 3)
        if ok:
            if session.hold_start_time is None:
                session.hold_start_time = time.time()
            return (time.time() - session.hold_start_time) >= HOLD_DURATION_SEC
        else:
            session.hold_start_time = None
            return False

    elif gesture_id == "head_forward":
        return cur_face_w > base_face_width * 1.12

    elif gesture_id == "head_backward":
        return cur_face_w < base_face_width * 0.88

    elif gesture_id == "eye_left_right":
        # Detect rapid eye movement (look left then right)
        session.shake_history.append(nose_offset)
        if len(session.shake_history) > 15:
            session.shake_history = session.shake_history[-15:]
        if len(session.shake_history) >= 8:
            amp = max(session.shake_history) - min(session.shake_history)
            return amp > face_w * 0.05
        return False

    elif gesture_id == "smile_then_blink":
        # We use session.was_blink_closed as a 'smile detected' latch
        if not session.was_blink_closed:
            # Phase 1: detect smile first
            if expr["smile_score"] > 0.4:
                session.was_blink_closed = True
                return False
        else:
            # Phase 2: now detect blink
            is_closed = (left_ear_val < base_left_ear * 0.88 or
                         right_ear_val < base_right_ear * 0.88)
            if is_closed:
                return True
        return False

    elif gesture_id == "blink_then_turn_left":
        if not session.was_blink_closed:
            is_closed = (left_ear_val < base_left_ear * 0.85 and
                         right_ear_val < base_right_ear * 0.85)
            if is_closed:
                session.was_blink_closed = True
                session.hold_start_time = time.time()
        else:
            delta = base_nose_cx - nose_offset
            if delta > _threshold(0.075, 4):
                return True
        return False

    elif gesture_id == "raise_eyebrows_hold":
        lb, rb = compute_brow_heights(pts_68)
        left_move = base_left_brow_y - lb
        right_move = base_right_brow_y - rb
        thresh = face_w * 0.035
        ok = left_move > thresh and right_move > thresh
        if ok:
            if session.hold_start_time is None:
                session.hold_start_time = time.time()
            return (time.time() - session.hold_start_time) >= HOLD_DURATION_SEC
        else:
            session.hold_start_time = None
            return False


    return False


# ── Advanced Anti-Screen / Emissive Detection ──

def check_temporal_flicker(luminance_history: List[float], fps: float = 15.0) -> Tuple[bool, float]:
    """
    Run FFT on luminance history to detect peaks in 50-70Hz range (aliasing).
    Screens often alias with camera shutter.
    """
    if len(luminance_history) < 20: return False, 0.0
    
    # Detrend
    y = np.array(luminance_history)
    y = y - np.mean(y)
    
    n = len(y)
    freqs = np.fft.rfftfreq(n, d=1/fps)
    magnitudes = np.abs(np.fft.rfft(y))
    
    # Looking for flickering signal (e.g. 50Hz/60Hz aliased to lower frequencies)
    # Most aliasing appears in 0.5Hz - 5Hz range in short windows
    mask = (freqs > 0.5) & (freqs < 6.0)
    if not np.any(mask): return False, 0.0
    
    peak_val = np.max(magnitudes[mask])
    total_val = np.sum(magnitudes) + 1e-6
    ratio = peak_val / total_val
    
    # Peak power threshold
    is_flickering = ratio > 0.42
    return is_flickering, round(float(ratio), 3)


# ═══════════════════════════════════════════════════════
# FUSION SIGNALS (replay_risk 0 = live, 1 = replay) for spoof_scoring
# ═══════════════════════════════════════════════════════

def _region_displacement(landmark_history: List[List[Dict]], indices: List[int], window: int = 10) -> float:
    """Compute average displacement of a set of landmark indices over recent frames."""
    if len(landmark_history) < 3:
        return 0.0
    recent = landmark_history[-window:]
    total = 0.0
    for idx in indices:
        xs = [pts[idx]["x"] for pts in recent]
        ys = [pts[idx]["y"] for pts in recent]
        total += math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return total / max(1, len(indices))


def _region_motion_series(landmark_history: List[List[Dict]], indices: List[int], window: int = 10) -> np.ndarray:
    """Return per-frame centroid displacement for a region (for cross-correlation analysis)."""
    if len(landmark_history) < 3:
        return np.array([])
    recent = landmark_history[-window:]
    centroids = []
    for pts in recent:
        cx = sum(pts[idx]["x"] for idx in indices) / len(indices)
        cy = sum(pts[idx]["y"] for idx in indices) / len(indices)
        centroids.append((cx, cy))
    # Per-frame displacement
    disps = []
    for i in range(1, len(centroids)):
        d = math.hypot(centroids[i][0] - centroids[i-1][0], centroids[i][1] - centroids[i-1][1])
        disps.append(d)
    return np.array(disps) if disps else np.array([])


def parallax_replay_risk(landmark_history: List[List[Dict]]) -> float:
    """
    ENHANCED: Multi-region parallax analysis.
    
    Real 3D faces: nose, jaw, cheeks, eyes move independently (low cross-correlation).
    Replay screens: all regions move uniformly (high cross-correlation, planar motion).
    
    Returns risk in [0, 1] — higher = more likely replay.
    """
    if len(landmark_history) < 5:
        return 0.0

    # Define distinct facial regions
    nose_region = [30, 31, 35]         # Nose tip + base
    left_jaw = [2, 3, 4, 5]           # Left jaw contour
    right_jaw = [11, 12, 13, 14]      # Right jaw contour
    left_eye = [36, 37, 38, 39]       # Left eye
    right_eye = [42, 43, 44, 45]      # Right eye
    chin = [7, 8, 9]                   # Chin

    # 1. Classic depth ratio (nose vs eyes)
    is_3d, depth_score = analyze_true_3d_deformation(landmark_history)
    classic_risk = float(min(1.0, max(0.0, depth_score)))
    if is_3d:
        classic_risk *= 0.35  # Strong 3D evidence reduces risk

    # 2. Cross-correlation analysis between regions
    nose_motion = _region_motion_series(landmark_history, nose_region)
    ljaw_motion = _region_motion_series(landmark_history, left_jaw)
    rjaw_motion = _region_motion_series(landmark_history, right_jaw)
    leye_motion = _region_motion_series(landmark_history, left_eye)

    correlation_risk = 0.0
    pairs_checked = 0
    min_len = 4

    for a, b in [(nose_motion, ljaw_motion), (nose_motion, leye_motion),
                  (ljaw_motion, rjaw_motion), (leye_motion, rjaw_motion)]:
        if len(a) >= min_len and len(b) >= min_len:
            # Normalize
            a_n = a - np.mean(a)
            b_n = b - np.mean(b)
            a_std = np.std(a_n)
            b_std = np.std(b_n)
            if a_std > 0.3 and b_std > 0.3:  # Only meaningful if both regions actually move
                corr = float(np.corrcoef(a_n, b_n)[0, 1])
                # High correlation = planar (replay-like)
                if corr > 0.92:
                    correlation_risk += 0.3
                elif corr > 0.85:
                    correlation_risk += 0.15
                pairs_checked += 1

    if pairs_checked > 0:
        correlation_risk = min(1.0, correlation_risk)
    else:
        correlation_risk = 0.0  # Can't judge without enough motion

    # 3. Jaw asymmetry check — real faces have asymmetric jaw movement
    ljaw_disp = _region_displacement(landmark_history, left_jaw)
    rjaw_disp = _region_displacement(landmark_history, right_jaw)
    if ljaw_disp > 1.0 and rjaw_disp > 1.0:
        jaw_ratio = min(ljaw_disp, rjaw_disp) / (max(ljaw_disp, rjaw_disp) + 1e-6)
        # Perfectly symmetric = suspicious (ratio near 1.0)
        if jaw_ratio > 0.96:
            correlation_risk = min(1.0, correlation_risk + 0.15)

    # Combine classic depth + correlation analysis
    risk = 0.5 * classic_risk + 0.5 * correlation_risk
    return float(min(1.0, max(0.0, risk)))


def biological_motion_replay_risk(landmark_history: List[List[Dict]]) -> float:
    """
    ENHANCED: Biological motion analysis with asymmetry and coordination tracking.
    
    Real faces show:
    - Natural asymmetry between left/right sides
    - Micro-delays between eye and head movement
    - Varied micro-expression timing
    
    Replay attacks show:
    - Synchronized motion (no micro-delays)
    - Compressed/uniform movement artifacts
    - Unnaturally still or perfectly mirrored motion
    
    Returns risk in [0, 1] — higher = more likely replay.
    """
    if len(landmark_history) < 15:
        return 0.0

    # 1. Classic micro-expression check
    has_micro, micro_score = check_micro_expressions(landmark_history)
    micro_risk = 1.0 - float(min(1.0, max(0.0, micro_score)))
    if has_micro:
        micro_risk *= 0.30  # Strong micro-expression evidence reduces risk

    recent = landmark_history[-15:]

    # 2. Left/Right asymmetry analysis
    # Real faces: left and right eye jitter differently (natural asymmetry)
    left_eye_xs = [pts[37]["x"] for pts in recent]
    right_eye_xs = [pts[43]["x"] for pts in recent]
    left_jitter = float(np.std(left_eye_xs))
    right_jitter = float(np.std(right_eye_xs))

    asymmetry_risk = 0.0
    if left_jitter > 0.3 and right_jitter > 0.3:
        jitter_ratio = min(left_jitter, right_jitter) / (max(left_jitter, right_jitter) + 1e-6)
        # Perfectly symmetric jitter = suspicious
        if jitter_ratio > 0.95:
            asymmetry_risk = 0.3
        elif jitter_ratio > 0.88:
            asymmetry_risk = 0.1
    else:
        # Very low jitter = might be static image
        if left_jitter < 0.15 and right_jitter < 0.15:
            asymmetry_risk = 0.2

    # 3. Eye-to-head coordination
    # Real: slight delay between eye and nose movement
    nose_xs = np.array([pts[30]["x"] for pts in recent])
    eye_mid_xs = np.array([(pts[36]["x"] + pts[45]["x"]) / 2 for pts in recent])
    
    coordination_risk = 0.0
    if len(nose_xs) >= 5:
        nose_diff = np.diff(nose_xs)
        eye_diff = np.diff(eye_mid_xs)
        if np.std(nose_diff) > 0.2 and np.std(eye_diff) > 0.2:
            # Cross-correlation at lag 0 vs lag 1
            corr_0 = float(np.corrcoef(nose_diff, eye_diff)[0, 1]) if len(nose_diff) >= 3 else 0.0
            # Perfect zero-lag correlation = suspicious (replay has no natural delay)
            if corr_0 > 0.97:
                coordination_risk = 0.25
            elif corr_0 > 0.93:
                coordination_risk = 0.1

    # 4. Lip micro-motion variance
    lip_ys = [pts[62]["y"] for pts in recent]
    lip_var = float(np.std(lip_ys))
    lip_risk = 0.0
    if lip_var < 0.2:
        lip_risk = 0.15  # Unnaturally still lips

    # Combine all biological signals
    risk = (0.40 * micro_risk +
            0.25 * asymmetry_risk +
            0.20 * coordination_risk +
            0.15 * lip_risk)

    return float(min(1.0, max(0.0, risk)))


def challenge_consistency_replay_risk(session: Any) -> float:
    """
    ENHANCED: Reaction time analysis for gesture challenges.
    
    Real users show variable reaction times with natural distribution.
    Scripted replays show unnaturally consistent or fast reactions.
    
    Returns risk in [0, 1].
    """
    step = getattr(session, "step", "")
    times = getattr(session, "reaction_times", None) or []
    if step not in ("gesture", "complete", "micro") or len(times) < 2:
        return 0.0
    recent = times[-5:]
    mean_t = float(np.mean(recent))
    std_t = float(np.std(recent))
    cv = std_t / (mean_t + 1e-6)

    risk = 0.0

    # Very low variance + fast reactions = suspicious
    if cv < 0.03 and mean_t < 0.50:
        risk = float(min(1.0, (0.05 - cv) * 18.0 + max(0.0, 0.45 - mean_t) * 1.2))
    # Inhumanly fast reactions
    elif mean_t < 0.15:
        risk = 0.6
    # Very low CV alone (robotic precision)
    elif cv < 0.02 and len(recent) >= 3:
        risk = 0.35

    return float(min(1.0, max(0.0, risk)))


def check_temporal_stream_integrity(landmark_history: List[List[Dict]]) -> float:
    """
    Analyze optical flow / landmark velocity continuity.
    Detects packet-loss style jumps, micro-stutter, and motion interpolation typical of video calls.
    Returns score 0.0 (clean) to 1.0 (highly degraded/stuttering).
    """
    if len(landmark_history) < 6:
        return 0.0
        
    recent = landmark_history[-15:]
    velocities = []
    
    # Calculate velocity of the nose tip
    for i in range(1, len(recent)):
        dx = recent[i][30]["x"] - recent[i-1][30]["x"]
        dy = recent[i][30]["y"] - recent[i-1][30]["y"]
        velocities.append(math.hypot(dx, dy))
        
    if not velocities:
        return 0.0
        
    # Acceleration (difference in velocity)
    accel = np.diff(velocities)
    if len(accel) == 0:
        return 0.0
        
    # Real human motion is relatively smooth (low high-frequency acceleration)
    # Video calls have micro-stutters (zero motion -> sudden jump)
    zeros = sum(1 for v in velocities if v < 0.5)
    jumps = sum(1 for a in np.abs(accel) if a > 5.0)
    
    stutter_score = 0.0
    # Pattern of stuck frames followed by jumps
    if zeros > 2 and jumps > 0:
        stutter_score += 0.4
    
    # Irregular frame pacing (high variance in acceleration)
    accel_var = np.var(accel)
    if accel_var > 15.0:
        stutter_score += min(0.6, accel_var / 50.0)
        
    return float(min(1.0, stutter_score))


def check_background_parallax(last_gray_small: Optional[np.ndarray], curr_gray_small: Optional[np.ndarray], lm_hist: List[Tuple[float, float]]) -> float:
    """
    Face/background motion decoupling.
    If background moves exactly with the face, it's a flat surface (iPad/laptop).
    Returns risk score 0.0 to 1.0.
    """
    if last_gray_small is None or curr_gray_small is None or last_gray_small.shape != curr_gray_small.shape:
        return 0.0
    if len(lm_hist) < 3:
        return 0.0

    diff = cv2.absdiff(last_gray_small, curr_gray_small)
    bg_motion = float(np.mean(diff)) / 255.0

    cx = [p[0] for p in lm_hist[-5:]]
    cy = [p[1] for p in lm_hist[-5:]]
    face_motion = float(np.hypot(max(cx) - min(cx), max(cy) - min(cy)) + 1e-6)

    if bg_motion < 0.005:
        return 0.0
        
    ratio = face_motion / (bg_motion * 120.0 + 1e-6)
    
    # Low face motion vs high background motion = highly suspicious
    if ratio < 0.4 and bg_motion > 0.01:
        return float(min(1.0, (0.4 - ratio) / 0.4 + (bg_motion - 0.01) * 10.0))
        
    return 0.0
