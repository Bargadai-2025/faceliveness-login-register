"""
Frame processor — orchestrates the liveness pipeline per frame.
Called by the /liveness/frame endpoint.
"""
import os
import time
import secrets
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from liveness_session import (
    LivenessSession,
    CALIBRATION_FRAMES,
    LIGHT_CHALLENGES,
    GESTURE_COOLDOWN,
    GESTURE_INSTRUCTION_SEC,
)
from face_detection import (
    decode_frame, detect_faces, compute_ear, compute_head_pose,
    compute_eye_tilt_rad, compute_brow_heights, estimate_expression,
    check_black_screen, compute_brightness_histogram, landmarks_to_serializable,
    get_face_roi,
)
from liveness_checks import (
    analyze_true_3d_deformation, check_micro_expressions,
    analyze_active_spectral_reflectance, evaluate_gesture, SUSTAINED_FRAMES,
    build_pose_snapshot,
    parallax_replay_risk,
    biological_motion_replay_risk,
    challenge_consistency_replay_risk,
    check_temporal_stream_integrity,
    check_background_parallax,
)
from spoof_scoring import (
    analyze_passive_spoof_single_frame,
    streaming_temporal_decision,
    centroid_of_landmarks,
    downsample_gray,
    load_thresholds,
)

# ── YOLO device detection ──
_yolo_model = None
_yolo_warmup_done = False
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
LIVENESS_YOLO_CONF = float(os.getenv("LIVENESS_YOLO_CONF", "0.28"))
DEVICE_NEAR_FACE_DR = float(os.getenv("LIVENESS_DEVICE_NEAR_DR", "0.12"))
LIVENESS_FAST_SETUP = os.getenv("LIVENESS_FAST_SETUP", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DEPTH_FRAMES_REQUIRED = int(os.getenv("LIVENESS_DEPTH_FRAMES", "2"))
LIGHT_PRE_FRAMES = int(os.getenv("LIVENESS_LIGHT_PRE_FRAMES", "2"))
LIGHT_POST_FRAMES = int(os.getenv("LIVENESS_LIGHT_POST_FRAMES", "2"))
MICRO_FRAMES_REQUIRED = int(os.getenv("LIVENESS_MICRO_FRAMES", "2"))
NO_FACE_GRACE_FRAMES = int(os.getenv("LIVENESS_NO_FACE_GRACE_FRAMES", "3"))
MULTI_FACE_GRACE_FRAMES = int(os.getenv("LIVENESS_MULTI_FACE_GRACE_FRAMES", "2"))

DEVICE_CLASSES = {
    "cell phone": "Mobile Phone",
    "laptop": "Laptop",
    "tv": "Television/Screen",
    "monitor": "Television/Screen",
}


def _begin_gesture_phase(session: LivenessSession, pts_68, pts_478) -> Dict[str, Any]:
    """Enter gesture challenges (marks hidden setup steps done for session/complete)."""
    session.depth_passed = True
    session.light_passed = True
    session.micro_passed = True
    session.step = "gesture"
    session.gesture_instruction_time = time.time()
    session.is_transitioning = True
    session.gesture_challenge_baseline = None
    session.gesture_sustain_count = 0
    session.shake_history = []
    session.shake_completed = False
    session.hold_start_time = None
    gesture = session.current_gesture or ""
    return {
        "status": "processing",
        "step": "gesture",
        "gesture": session.current_gesture,
        "gesture_idx": session.current_gesture_idx,
        "detail": f"CHALLENGE: {gesture.replace('_', ' ').upper()}" if gesture else "Starting challenges…",
        "progress": session.progress_pct,
        "landmarks": landmarks_to_serializable(pts_68),
        "mesh": landmarks_to_serializable(pts_478),
        "gesture_prep": True,
    }


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            weights_path = YOLO_WEIGHTS
            if not os.path.isabs(weights_path):
                backend_dir = os.path.dirname(os.path.abspath(__file__))
                candidate = os.path.join(backend_dir, weights_path)
                if os.path.isfile(candidate):
                    weights_path = candidate
            _yolo_model = YOLO(weights_path)
            print(f"✅ YOLO loaded for liveness device detection: {weights_path}")
        except Exception as e:
            print(f"⚠️ YOLO load failed (device detection disabled): {e}")
    return _yolo_model


def warmup_yolo() -> bool:
    """Load YOLO and run one dummy inference so the first user frame is fast."""
    global _yolo_warmup_done
    if _yolo_warmup_done:
        return True
    yolo = _get_yolo()
    if yolo is None:
        return False
    try:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        yolo(dummy, verbose=False, conf=LIVENESS_YOLO_CONF)
        _yolo_warmup_done = True
        print("✅ YOLO warm-up inference complete")
        return True
    except Exception as e:
        print(f"⚠️ YOLO warm-up failed: {e}")
        return False


def _should_run_security_scan(session: LivenessSession) -> bool:
    """YOLO + multi-face on every frame with a detected face."""
    return True


def _should_run_full_spoof_pipeline(session: LivenessSession) -> bool:
    """Skip heavy PAD on first 2 calibration frames for faster startup."""
    if session.step != "calibration":
        return True
    return session.calibration_count >= 2


def _classify_yolo_device(cls_name: str, obj_w: float, obj_h: float, area_pct: float) -> str:
    base = DEVICE_CLASSES.get(cls_name, cls_name.title())
    if cls_name == "cell phone":
        ar = obj_w / (obj_h + 1e-6)
        if 0.72 <= ar <= 1.38 and area_pct >= 0.04:
            return "Tablet"
    return base


def _yolo_device_replay_risk(
    img: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    w: int,
    h: int,
    yolo,
) -> Tuple[float, bool, List[str], bool]:
    """
    Face–device overlap → replay_risk in [0,1].
    hard_overlap: very high IoU (phone covering face).
    device_visible_in_frame: device occupies >= 3.5% of frame area.
    """
    fx, fy, fw, fh = face_bbox
    max_risk = 0.0
    hard_overlap = False
    devices_found: List[str] = []
    device_visible_in_frame = False
    try:
        yolo_results = yolo(img, verbose=False, conf=LIVENESS_YOLO_CONF)
        for r in yolo_results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])].lower()
                if cls_name not in DEVICE_CLASSES:
                    continue
                obj_w, obj_h = float(box.xywh[0][2]), float(box.xywh[0][3])
                area_pct = (obj_w * obj_h) / (w * h + 1e-6)
                if area_pct >= 0.035:
                    device_visible_in_frame = True
                display = _classify_yolo_device(cls_name, obj_w, obj_h, area_pct)
                if display not in devices_found:
                    devices_found.append(display)
                bx = box.xyxy[0].cpu().numpy()
                ix1 = max(bx[0], fx)
                iy1 = max(bx[1], fy)
                ix2 = min(bx[2], fx + fw)
                iy2 = min(bx[3], fy + fh)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                face_area = fw * fh + 1e-6
                iou_face = inter_area / face_area
                if iou_face > 0.32:
                    hard_overlap = True
                part = min(1.0, iou_face * 2.1 + min(0.40, area_pct * 3.0))
                max_risk = max(max_risk, part)
    except Exception as e:
        print(f"YOLO device risk error: {e}")
    return float(min(1.0, max_risk)), hard_overlap, devices_found, device_visible_in_frame


def _device_alert_response(
    session: LivenessSession,
    pts_68,
    pts_478,
    detail: str,
    devices_found: List[str],
    dr: float,
    *,
    hard_overlap: bool = False,
) -> Dict[str, Any]:
    return {
        "status": "processing",
        "detail": detail,
        "step": session.step,
        "landmarks": landmarks_to_serializable(pts_68),
        "mesh": landmarks_to_serializable(pts_478),
        "progress": session.progress_pct,
        "is_suspicious": True,
        "multi_person": False,
        "devices_detected": devices_found,
        "device_replay": round(dr, 3),
        "spoof_debug": {"device_replay": round(dr, 3), "hard_overlap": hard_overlap},
    }


def process_frame(session: LivenessSession, frame_bytes: bytes, identity_callback=None) -> Dict[str, Any]:
    """
    Process a single frame through the liveness pipeline.
    Returns status dict for the frontend.
    """
    img = decode_frame(frame_bytes)
    if img is None:
        return {"status": "error", "detail": "Could not decode frame"}

    session.frame_count += 1
    h, w = img.shape[:2]

    # ── BLACK SCREEN CHECK (continuous) ──
    if check_black_screen(img):
        session.black_screen_count += 1
        if session.black_screen_count >= 3:
            return {
                "status": "processing",
                "detail": "Camera blocked — please uncover camera",
                "step": session.step,
                "progress": session.progress_pct,
                "mesh": None
            }
    else:
        session.black_screen_count = 0

    # ── FACE DETECTION ──
    faces = detect_faces(img)

    if len(faces) == 0:
        session.no_face_streak += 1
        session.multi_face_streak = 0
        if session.no_face_streak >= NO_FACE_GRACE_FRAMES:
            return {
                "status": "processing",
                "step": session.step,
                "detail": "No face detected — center your face",
                "progress": session.progress_pct,
                "mesh": None
            }
        return {
            "status": "processing",
            "step": session.step,
            "detail": "Hold steady — tracking face...",
            "progress": session.progress_pct,
            "mesh": None,
        }

    if len(faces) > 1:
        session.multi_face_streak += 1
        session.no_face_streak = 0
        if session.multi_face_streak >= MULTI_FACE_GRACE_FRAMES:
            return {
                "status": "processing",
                "error": True,
                "multi_person": True,
                "detail": "Multiple people detected — please ensure only one person is in view",
                "step": session.step,
                "progress": session.progress_pct,
                "mesh": None,
                "landmarks": None,
            }
        return {
            "status": "processing",
            "step": session.step,
            "detail": "Hold steady — stabilizing face tracking...",
            "progress": session.progress_pct,
            "mesh": None,
            "landmarks": None,
            "multi_person": False,
        }

    session.no_face_streak = 0
    session.multi_face_streak = 0

    face = faces[0]
    pts_68 = face["pts_68"]
    pts_478 = face["pts_478"]
    face_width = face["face_width"]

    session.landmark_history.append(pts_68)
    if len(session.landmark_history) > 30:
        session.landmark_history = session.landmark_history[-30:]

    session.landmark_centroid_history.append(centroid_of_landmarks(pts_68))
    if len(session.landmark_centroid_history) > 40:
        session.landmark_centroid_history = session.landmark_centroid_history[-40:]

    curr_gray_small = downsample_gray(img)

    roi = get_face_roi(img, pts_68)
    face_bbox = (
        int(min([p["x"] for p in pts_68])),
        int(min([p["y"] for p in pts_68])),
        int(face_width),
        int(max([p["y"] for p in pts_68]) - min([p["y"] for p in pts_68])),
    )
    if roi is not None:
        luminance = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
        session.roi_luminance_history.append(luminance)
        if len(session.roi_luminance_history) > 60:
            session.roi_luminance_history.pop(0)

    # ── YOLO every frame; full spoof PAD when not in early calibration ──
    if _should_run_security_scan(session):
        yolo = _get_yolo()
        dr = 0.0
        devices_found: List[str] = []
        if yolo is not None:
            dr, device_hard, devices_found, device_visible = _yolo_device_replay_risk(
                img, face_bbox, w, h, yolo
            )
            if device_hard:
                names = ", ".join(devices_found) if devices_found else "device"
                return _device_alert_response(
                    session,
                    pts_68,
                    pts_478,
                    f"Security Alert: {names} overlapping face — remove from view",
                    devices_found,
                    dr,
                    hard_overlap=True,
                )
            if devices_found and (dr >= DEVICE_NEAR_FACE_DR or device_visible):
                return _device_alert_response(
                    session,
                    pts_68,
                    pts_478,
                    # f"Electronic device detected ({', '.join(devices_found)}) — move away from camera",
                    f"Electronic device detected (Mobile phone) — move away from camera",
                    devices_found,
                    dr,
                )

        if _should_run_full_spoof_pipeline(session):
            extra_signals: Dict[str, float] = {
                "depth_parallax": parallax_replay_risk(session.landmark_history),
                "biological": biological_motion_replay_risk(session.landmark_history),
                "challenge": challenge_consistency_replay_risk(session),
                "temporal_stream_integrity": check_temporal_stream_integrity(session.landmark_history),
                "environment_authenticity": check_background_parallax(
                    session.last_gray_small, curr_gray_small, session.landmark_centroid_history
                ),
                "device_replay": dr,
            }

            spoof_report = analyze_passive_spoof_single_frame(
                img,
                pts_68,
                strict=False,
                roi_luminance_history=session.roi_luminance_history,
                last_gray_small=session.last_gray_small,
                curr_gray_small=curr_gray_small,
                landmark_centroid_history=session.landmark_centroid_history,
                extra_signals=extra_signals,
            )
            current_fused = float(spoof_report["total_spoof_score"])
            th_all = load_thresholds()
            alpha = float(th_all.get("ema_alpha", 0.22))
            session.replay_risk_ema = alpha * current_fused + (1.0 - alpha) * float(session.replay_risk_ema)
            session.fraud_ema_history.append(round(session.replay_risk_ema, 2))
            if len(session.fraud_ema_history) > 30:
                session.fraud_ema_history = session.fraud_ema_history[-30:]

            signal_conf = spoof_report.get("confidence_per_signal", {})
            for sig_key in session.per_signal_history:
                v = float(signal_conf.get(sig_key, extra_signals.get(sig_key, 0.0)))
                session.per_signal_history[sig_key].append(round(v, 3))
                if len(session.per_signal_history[sig_key]) > 20:
                    session.per_signal_history[sig_key] = session.per_signal_history[sig_key][-20:]

            should_flag, smoothed = streaming_temporal_decision(session.spoof_score_history, session.replay_risk_ema)
            session.spoof_score_history.append(session.replay_risk_ema)
            if len(session.spoof_score_history) > 20:
                session.spoof_score_history = session.spoof_score_history[-20:]

            cooldown_threshold = float(th_all.get("cooldown_decay_threshold", 40.0))
            temporal_hits_required = int(th_all.get("temporal_hits_required", 4))

            if should_flag:
                session.spoof_temporal_hits += 1
                session.fraud_cooldown_frames = 0
            else:
                if session.replay_risk_ema < cooldown_threshold:
                    session.fraud_cooldown_frames += 1
                    if session.fraud_cooldown_frames >= 2 and session.spoof_temporal_hits > 0:
                        session.spoof_temporal_hits = max(0, session.spoof_temporal_hits - 1)
                        session.fraud_cooldown_frames = 0
                else:
                    session.fraud_cooldown_frames = 0

            if session.spoof_temporal_hits >= temporal_hits_required:
                session.spoof_temporal_hits = 0
                rejection_reason = (
                    f"Sustained replay risk ({temporal_hits_required}+ frames, EMA={session.replay_risk_ema:.1f})"
                )
                session.fraud_rejection_reasons.append(rejection_reason)

                per_signal_avgs = {}
                for sig_key, sig_hist in session.per_signal_history.items():
                    if sig_hist:
                        per_signal_avgs[sig_key] = round(sum(sig_hist[-8:]) / len(sig_hist[-8:]), 3)

                return {
                    "status": "processing",
                    "detail": "Security Alert: sustained presentation-attack risk (multi-signal + temporal)",
                    "step": session.step,
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "progress": session.progress_pct,
                    "is_suspicious": True,
                    "multi_person": False,
                    "spoof_debug": {
                        "final_replay_risk": round(session.replay_risk_ema, 2),
                        "frame_fused_score": round(current_fused, 2),
                        "smoothed_window": round(smoothed, 2),
                        "temporal_hits_required": temporal_hits_required,
                        "triggered_rules": spoof_report.get("triggered_rules", []),
                        "confidence_per_signal": spoof_report.get("confidence_per_signal", {}),
                        "extra_signals_used": spoof_report.get("extra_signals_used", {}),
                        "correlation_gate_notes": spoof_report.get("correlation_gate_notes", []),
                        "fraud_ema_history": list(session.fraud_ema_history[-16:]),
                        "per_signal_rolling_averages": per_signal_avgs,
                        "rejection_reason": rejection_reason,
                        "fraud_rejection_count": len(session.fraud_rejection_reasons),
                    },
                }

    session.last_gray_small = curr_gray_small

    if face_width < w * 0.08:
        return {
            "status": "processing",
            "step": session.step,
            "detail": "Face detected — Please move slightly closer",
            "progress": session.progress_pct,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
        }

    # ── CALIBRATION PHASE ──
    if session.step == "calibration":
        session.calibration_count += 1
        if session.calibration_count >= CALIBRATION_FRAMES:
            history = session.landmark_history[-CALIBRATION_FRAMES:]
            nose_ys = [pts[30]["y"] for pts in history]
            nose_cxs = []
            face_ws = []
            left_ears = []
            right_ears = []
            eye_angles = []
            for pts in history:
                fc = (pts[0]["x"] + pts[16]["x"]) / 2
                nose_cxs.append(pts[30]["x"] - fc)
                face_ws.append(abs(pts[16]["x"] - pts[0]["x"]))
                left_ears.append(compute_ear(pts, 36))
                right_ears.append(compute_ear(pts, 42))
                eye_angles.append(compute_eye_tilt_rad(pts))

            nose_ys.sort()
            nose_cxs.sort()
            face_ws.sort()
            left_ears.sort()
            right_ears.sort()
            eye_angles.sort()
            mid = len(nose_ys) // 2

            last_pts = history[-1]
            session.baseline = {
                "nose_tip_y": nose_ys[mid],
                "nose_center_x": nose_cxs[mid],
                "face_width": face_ws[mid],
                "left_ear": left_ears[mid],
                "right_ear": right_ears[mid],
                "eye_angle": eye_angles[mid],
                "left_brow_y": (last_pts[19]["y"] + last_pts[21]["y"]) / 2,
                "right_brow_y": (last_pts[24]["y"] + last_pts[26]["y"]) / 2,
            }
            if LIVENESS_FAST_SETUP:
                return _begin_gesture_phase(session, pts_68, pts_478)
            session.step = "depth"
            session.gesture_instruction_time = time.time()
            return {
                "status": "processing",
                "step": "depth",
                "detail": "Calibration complete — analyzing depth",
                "progress": session.progress_pct,
                "landmarks": landmarks_to_serializable(pts_68),
                "mesh": landmarks_to_serializable(pts_478),
            }

        return {
            "status": "processing",
            "step": "calibration",
            "detail": "Scanning face & environment security...",
            "progress": session.progress_pct,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
        }

    # ── DEPTH ESTIMATION PHASE ──
    if session.step == "depth":
        is_3d, depth_score = analyze_true_3d_deformation(session.landmark_history)
        if is_3d:
            session.depth_scores.append(depth_score)
        else:
            session.depth_scores = []

        if len(session.depth_scores) >= DEPTH_FRAMES_REQUIRED:
            avg_depth = sum(session.depth_scores) / len(session.depth_scores)
            if avg_depth < 0.3:
                session.depth_scores = []
                return {
                    "status": "processing",
                    "step": "depth",
                    "detail": "Depth check failed — move head slightly in all directions",
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "progress": session.progress_pct,
                }
            session.depth_passed = True
            session.light_challenge_color = secrets.choice(LIGHT_CHALLENGES)
            session.step = "light_challenge"
            return {
                "status": "processing",
                "step": "light_challenge",
                "detail": "Depth OK — starting light challenge",
                "instruction": session.light_challenge_color,
                "progress": session.progress_pct,
                "landmarks": landmarks_to_serializable(pts_68),
                "mesh": landmarks_to_serializable(pts_478),
            }

        return {
            "status": "processing",
            "step": "depth",
            "detail": "Scanning face & environment security...",
            "progress": session.progress_pct,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
        }

    # ── LIGHT CHALLENGE PHASE ──
    if session.step == "light_challenge":
        stats = compute_brightness_histogram(img)
        if len(session.light_pre_frames) < LIGHT_PRE_FRAMES:
            session.light_pre_frames.append(stats)
            return {
                "status": "processing",
                "step": "light_challenge",
                "detail": "Scanning face & environment security...",
                "instruction": session.light_challenge_color,
                "progress": session.progress_pct,
                "landmarks": landmarks_to_serializable(pts_68),
                "mesh": landmarks_to_serializable(pts_478),
            }

        session.light_post_frames.append(stats)
        if len(session.light_post_frames) >= LIGHT_POST_FRAMES:
            passed, score = analyze_active_spectral_reflectance(
                session.light_pre_frames, session.light_post_frames,
                session.light_challenge_color
            )
            if passed:
                session.light_passed = True
                session.step = "micro"
                return {
                    "status": "processing",
                    "step": "micro",
                    "detail": "Light check done — analyzing micro expressions",
                    "progress": session.progress_pct,
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                }
            session.light_post_frames = []
            return {
                "status": "processing",
                "step": "light_challenge",
                "detail": "Light challenge failed — face the screen and allow camera lighting change",
                "instruction": session.light_challenge_color,
                "progress": session.progress_pct,
                "landmarks": landmarks_to_serializable(pts_68),
                "mesh": landmarks_to_serializable(pts_478),
            }

        return {
            "status": "processing",
            "step": "light_challenge",
            "detail": "Scanning face & environment security...",
            "instruction": session.light_challenge_color,
            "progress": session.progress_pct,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
        }

    # ── MICRO EXPRESSION PHASE ──
    if session.step == "micro":
        has_micro, micro_score = check_micro_expressions(session.landmark_history)
        if has_micro:
            session.micro_variance_scores.append(micro_score)
        if len(session.micro_variance_scores) >= MICRO_FRAMES_REQUIRED:
            avg = sum(session.micro_variance_scores) / len(session.micro_variance_scores)
            if avg < 0.1 or not has_micro:
                session.micro_variance_scores = []
                return {
                    "status": "processing",
                    "step": "micro",
                    "detail": "Micro expression check failed — blink or move naturally, do not hold perfectly still",
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "progress": session.progress_pct,
                }
            session.micro_passed = True
            return _begin_gesture_phase(session, pts_68, pts_478)

        return {
            "status": "processing",
            "step": "micro",
            "detail": "Scanning face & environment security...",
            "progress": session.progress_pct,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
        }

    if session.step == "gesture":
        if session.all_gestures_done:
            session.step = "complete"
            return {
                "status": "processing",
                "step": "complete",
                "detail": "All gestures passed — verifying...",
                "progress": 95,
                "landmarks": landmarks_to_serializable(pts_68),
                "mesh": landmarks_to_serializable(pts_478),
            }
        elif session.is_transitioning:
            elapsed = time.time() - session.gesture_instruction_time
            extra_cooldown = GESTURE_COOLDOWN if session.current_gesture_idx > 0 else 0.0
            wait_sec = GESTURE_INSTRUCTION_SEC + extra_cooldown
            if elapsed < wait_sec:
                gesture = session.current_gesture or ""
                return {
                    "status": "processing",
                    "step": "gesture",
                    "gesture": session.current_gesture,
                    "gesture_idx": session.current_gesture_idx,
                    "detail": (
                        f"CHALLENGE: {gesture.replace('_', ' ').upper()}"
                        if gesture
                        else "Get ready…"
                    ),
                    "progress": session.progress_pct,
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "gesture_prep": True,
                }
            if session.gesture_challenge_baseline is None and session.baseline:
                session.gesture_challenge_baseline = build_pose_snapshot(
                    pts_68, session.baseline
                )
                session.gesture_sustain_count = 0
            session.is_transitioning = False

        gesture = session.current_gesture
        if gesture:
            passed = evaluate_gesture(gesture, pts_68, session.baseline, session)
            if passed:
                session.gesture_sustain_count += 1
                if session.gesture_sustain_count >= SUSTAINED_FRAMES:
                    session.advance_gesture()
                    if session.all_gestures_done:
                        return {
                            "status": "processing",
                            "step": "complete",
                            "detail": "All gestures passed — verifying...",
                            "progress": 95,
                            "landmarks": landmarks_to_serializable(pts_68),
                            "mesh": landmarks_to_serializable(pts_478),
                        }
                    return {
                        "status": "processing",
                        "step": "gesture",
                        "gesture": session.current_gesture,
                        "gesture_idx": session.current_gesture_idx,
                        "detail": "✅ Good! Next gesture...",
                        "progress": session.progress_pct,
                        "landmarks": landmarks_to_serializable(pts_68),
                        "mesh": landmarks_to_serializable(pts_478),
                    }
            else:
                session.gesture_sustain_count = 0

        return {
            "status": "processing",
            "step": "gesture",
            "gesture": session.current_gesture,
            "gesture_idx": session.current_gesture_idx,
            "detail": session.current_gesture
            and f"CHALLENGE: {session.current_gesture.replace('_', ' ').upper()}"
            or "Waiting for gesture...",
            "progress": session.progress_pct,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
        }

    if session.step == "complete":
        return {
            "status": "verified",
            "step": "complete",
            "detail": "Liveness verified — capture selfie",
            "progress": 100,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
        }

    return {"status": "error", "detail": "Unknown step", "mesh": None}
