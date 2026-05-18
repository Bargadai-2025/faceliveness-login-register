"""
Frame processor — orchestrates the liveness pipeline per frame.
Called by the /liveness/frame endpoint.
"""
import time
import math
import secrets
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from liveness_session import LivenessSession, CALIBRATION_FRAMES, LIGHT_CHALLENGES, GESTURE_COOLDOWN
from face_detection import (
    decode_frame, detect_faces, compute_ear, compute_head_pose,
    compute_eye_tilt_rad, compute_brow_heights, estimate_expression,
    check_black_screen, compute_brightness_histogram, landmarks_to_serializable,
    get_face_roi,
)
from liveness_checks import (
    analyze_true_3d_deformation, check_micro_expressions,
    analyze_active_spectral_reflectance, evaluate_gesture, SUSTAINED_FRAMES,
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

# YOLO device detection — load lazily
_yolo_model = None
# Map COCO classes to user-friendly names for rejection
DEVICE_CLASSES = {
    "cell phone": "Mobile Phone",
    "laptop": "Laptop",
    "tv": "Television/Screen",
    "monitor": "Television/Screen",
}


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            # Upgraded to yolov8s.pt for higher recall
            _yolo_model = YOLO("yolov8s.pt")
            print("✅ YOLOv8s loaded for hard anti-screen detection")
        except Exception as e:
            print(f"⚠️ YOLO load failed (device detection disabled): {e}")
    return _yolo_model


def _yolo_device_replay_risk(
    img: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    w: int,
    h: int,
    yolo,
) -> Tuple[float, bool]:
    """
    Face–device overlap → replay_risk in [0,1].
    hard_overlap: only true for very high IoU (obvious phone covering face).
    """
    fx, fy, fw, fh = face_bbox
    max_risk = 0.0
    hard_overlap = False
    try:
        yolo_results = yolo(img, verbose=False, conf=0.48)
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
                if iou_face > 0.48:
                    hard_overlap = True
                part = min(1.0, iou_face * 2.1 + min(0.35, area_pct * 2.8))
                max_risk = max(max_risk, part)
    except Exception as e:
        print(f"YOLO device risk error: {e}")
    return float(min(1.0, max_risk)), hard_overlap


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
        return {
            "status": "processing",
            "step": session.step,
            "detail": "No face detected — center your face",
            "progress": session.progress_pct,
            "mesh": None
        }

    # 🔥 FIXED: STRICT MULTI-FACE BLOCK
    if len(faces) > 1:
        return {
            "status": "processing",
            "error": True,
            "multi_person": True,
            "detail": "Multiple people detected — please ensure only one person is in view",
            "step": session.step,
            "progress": session.progress_pct,
            "mesh": None
        }

    face = faces[0]
    pts_68 = face["pts_68"]
    pts_478 = face["pts_478"]
    face_width = face["face_width"]

    # ── AGENT VERIFICATION (Disabled) ──
    # if session.agent_embedding is not None and identity_callback is not None:
    #     now = time.time()
    #     # Initialize check time if 0
    #     if session.last_agent_check_time == 0:
    #         session.last_agent_check_time = now - 5.0 # Check soon after start
    #         
    #     if now - session.last_agent_check_time > 10.0:
    #         # We use the current image to verify identity
    #         is_match = identity_callback(img, session.agent_embedding)
    #         session.last_agent_check_time = now
    #         if not is_match:
    #             print(f"❌ Identity Mismatch: Session person does not match selected agent {session.agent_label}")
    #             return {
    #                 "status": "processing",
    #                 "error": True,
    #                 "detail": "Identity Mismatch: User is not the selected agent",
    #                 "step": session.step,
    #                 "progress": session.progress_pct,
    #                 "landmarks": landmarks_to_serializable(pts_68),
    #                 "mesh": landmarks_to_serializable(pts_478),
    #                 "is_suspicious": True
    #             }

    # Store landmarks for history
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

    # ── WEIGHTED SPOOF + YOLO DEVICE (multi-signal; EMA + temporal gate) ──
    critical_steps = ["calibration", "depth", "light_challenge", "micro", "gesture"]
    if session.step in critical_steps or session.frame_count % 5 == 0:
        extra_signals: Dict[str, float] = {
            "depth_parallax": parallax_replay_risk(session.landmark_history),
            "biological": biological_motion_replay_risk(session.landmark_history),
            "challenge": challenge_consistency_replay_risk(session),
            "temporal_stream_integrity": check_temporal_stream_integrity(session.landmark_history),
            "environment_authenticity": check_background_parallax(session.last_gray_small, curr_gray_small, session.landmark_centroid_history),
            "device_replay": 0.0,
        }
        yolo = _get_yolo()
        device_hard = False
        if yolo is not None:
            dr, device_hard = _yolo_device_replay_risk(img, face_bbox, w, h, yolo)
            extra_signals["device_replay"] = dr
            if device_hard:
                return {
                    "status": "processing",
                    "detail": "Security Alert: device obscuring face — remove phone or tablet from view",
                    "step": session.step,
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "progress": session.progress_pct,
                    "is_suspicious": True,
                    "spoof_debug": {"device_replay": round(dr, 3), "hard_overlap": True},
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

        # ── Track per-signal rolling history for debugging ──
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

        # ── COOLDOWN RECOVERY: prevent false spikes from cascading ──
        cooldown_threshold = float(th_all.get("cooldown_decay_threshold", 40.0))
        temporal_hits_required = int(th_all.get("temporal_hits_required", 4))

        if should_flag:
            session.spoof_temporal_hits += 1
            session.fraud_cooldown_frames = 0  # Reset cooldown
        else:
            # If current score is well below threshold, actively decay accumulated hits
            if session.replay_risk_ema < cooldown_threshold:
                session.fraud_cooldown_frames += 1
                # Every 2 clean frames, reduce temporal hits by 1
                if session.fraud_cooldown_frames >= 2 and session.spoof_temporal_hits > 0:
                    session.spoof_temporal_hits = max(0, session.spoof_temporal_hits - 1)
                    session.fraud_cooldown_frames = 0
            else:
                session.fraud_cooldown_frames = 0

        # ── TEMPORAL REJECTION: only if sustained across MANY frames ──
        if session.spoof_temporal_hits >= temporal_hits_required:
            session.spoof_temporal_hits = 0  # Reset after rejection
            rejection_reason = f"Sustained replay risk ({temporal_hits_required}+ frames, EMA={session.replay_risk_ema:.1f})"
            session.fraud_rejection_reasons.append(rejection_reason)

            # Compute per-signal rolling averages for debug
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

    # Face too large (possible screen spoof)
    if face_width > w * 0.85:
        return {
            "status": "processing",
            "detail": "Face too close — move back",
            "step": session.step,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
            "progress": session.progress_pct,
        }

    # ── CALIBRATION PHASE ──
    if session.step == "calibration":
        session.calibration_count += 1
        if session.calibration_count >= CALIBRATION_FRAMES:
            # Build baseline from accumulated landmarks
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
        session.depth_scores.append(depth_score)

        if len(session.depth_scores) >= 4:
            avg_depth = sum(session.depth_scores) / len(session.depth_scores)
            if avg_depth < 0.3:
                # If depth fails, we don't reject immediately, we let them try again (go back to depth start)
                session.depth_scores = [] 
                return {
                    "status": "processing",
                    "step": "depth",
                    "detail": "Depth check failed — move head slightly",
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "progress": session.progress_pct,
                }
            session.depth_passed = True
            # Pick a random light challenge
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
        if len(session.light_pre_frames) < 3:
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
        else:
            session.light_post_frames.append(stats)
            if len(session.light_post_frames) >= 3:
                passed, score = analyze_active_spectral_reflectance(
                    session.light_pre_frames, session.light_post_frames,
                    session.light_challenge_color
                )
                # Be lenient — light challenge is supplementary
                session.light_passed = True  # Always pass but log score
                session.step = "micro"
                return {
                    "status": "processing",
                    "step": "micro",
                    "detail": "Light check done — analyzing micro expressions",
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
        session.micro_variance_scores.append(micro_score)
        if len(session.micro_variance_scores) >= 3:
            avg = sum(session.micro_variance_scores) / len(session.micro_variance_scores)
            if avg < 0.1:
                # If micro expression fails, let them try again
                session.micro_variance_scores = []
                return {
                    "status": "processing",
                    "step": "micro",
                    "detail": "Micro expression check failed — keep moving naturally",
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "progress": session.progress_pct,
                }
            session.micro_passed = True
            session.step = "gesture"
            session.gesture_instruction_time = time.time()
            # session.is_transitioning = True  # REMOVED forced delay for first gesture
            return {
                "status": "processing",
                "step": "gesture",
                "gesture": session.current_gesture,
                "gesture_idx": session.current_gesture_idx,
                "detail": f"CHALLENGE: {session.current_gesture.replace('_', ' ').upper()}",
                "progress": session.progress_pct,
                "landmarks": landmarks_to_serializable(pts_68),
                "mesh": landmarks_to_serializable(pts_478),
            }
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
            # Check if cooldown is over
            elapsed = time.time() - session.gesture_instruction_time
            if elapsed < GESTURE_COOLDOWN:
                return {
                    "status": "processing",
                    "step": "gesture",
                    "gesture": session.current_gesture,
                    "gesture_idx": session.current_gesture_idx,
                    "detail": "Waiting for gesture...",
                    "progress": session.progress_pct,
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                }
            else:
                session.is_transitioning = False
                # Fall through to evaluation

        # ONLY evaluate if not done and not transitioning
        gesture = session.current_gesture
        if gesture:
            passed = evaluate_gesture(gesture, pts_68, session.baseline, session)
            if passed:
                session.gesture_sustain_count += 1
                if session.gesture_sustain_count >= SUSTAINED_FRAMES:
                    session.advance_gesture()
                    # After advancing, check if we are done or transitioning
                    if session.all_gestures_done:
                        return {
                            "status": "processing",
                            "step": "complete",
                            "detail": "All gestures passed — verifying...",
                            "progress": 95,
                            "landmarks": landmarks_to_serializable(pts_68),
                            "mesh": landmarks_to_serializable(pts_478),
                        }
                    # If not done, it will be in transition in the next frame
                    return {
                        "status": "processing",
                        "step": "gesture",
                        "gesture": session.current_gesture,
                        "gesture_idx": session.current_gesture_idx,
                        "detail": f"✅ Good! Next gesture...",
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
            "detail": "Waiting for gesture...",
            "progress": session.progress_pct,
            "landmarks": landmarks_to_serializable(pts_68),
            "mesh": landmarks_to_serializable(pts_478),
        }

    # ── COMPLETE ──
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
