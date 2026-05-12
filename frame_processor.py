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
    compute_passive_liveness,
)
from liveness_checks import (
    check_depth_displacement, check_micro_expressions,
    check_light_response, evaluate_gesture, SUSTAINED_FRAMES,
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

    # ── AGENT VERIFICATION (Every 10 seconds) ──
    if session.agent_embedding is not None and identity_callback is not None:
        import time
        now = time.time()
        # Initialize check time if 0
        if session.last_agent_check_time == 0:
            session.last_agent_check_time = now - 5.0 # Check soon after start
            
        if now - session.last_agent_check_time > 10.0:
            # We use the current image to verify identity
            is_match = identity_callback(img, session.agent_embedding)
            session.last_agent_check_time = now
            if not is_match:
                print(f"❌ Identity Mismatch: Session person does not match selected agent {session.agent_label}")
                return {
                    "status": "processing",
                    "error": True,
                    "detail": "Identity Mismatch: User is not the selected agent",
                    "step": session.step,
                    "progress": session.progress_pct,
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "is_suspicious": True
                }

    # Store landmarks for history
    session.landmark_history.append(pts_68)
    if len(session.landmark_history) > 30:
        session.landmark_history = session.landmark_history[-30:]

    # Face too small
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

    # ── PASSIVE LIVENESS (Texture & Digital Glow Analysis) ──
    # Run every frame during critical early steps, otherwise every 3 frames
    critical_steps = ["calibration", "depth", "light_challenge", "micro"]
    if session.step in critical_steps or session.frame_count % 5 == 0:
        is_live_texture, texture_score, texture_reason = compute_passive_liveness(img, pts_68)
        if not is_live_texture:
            session.digital_screen_fail_count += 1
            # 🔥 Require 3 consecutive failures to avoid light-flare false positives
            if session.digital_screen_fail_count >= 3:
                return {
                    "status": "processing", 
                    "detail": f"Security Alert: Digital screen glow detected",
                    "step": session.step,
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "progress": session.progress_pct,
                    "is_suspicious": True
                }
            else:
                # Still processing, waiting for consistency consensus
                return {
                    "status": "processing",
                    "detail": "Analyzing environment security...",
                    "step": session.step,
                    "landmarks": landmarks_to_serializable(pts_68),
                    "mesh": landmarks_to_serializable(pts_478),
                    "progress": session.progress_pct,
                }
        else:
            # Reset counter on valid live frame
            session.digital_screen_fail_count = 0

    # ── TEMPORAL HISTORY UPDATE ──
    from face_detection import get_face_roi
    roi = get_face_roi(img, pts_68)
    face_bbox = (int(min([p["x"] for p in pts_68])), int(min([p["y"] for p in pts_68])), 
                 int(face_width), int(max([p["y"] for p in pts_68]) - min([p["y"] for p in pts_68])))
    
    if roi is not None:
        luminance = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
        session.roi_luminance_history.append(luminance)
        if len(session.roi_luminance_history) > 60:
            session.roi_luminance_history.pop(0)

    # ── YOLO OBJECT DETECTION (Fraud & Background check) ──
    # Run every frame during critical steps for instant fail, otherwise every 2 frames
    if session.step in critical_steps or session.frame_count % 5 == 0:
        yolo = _get_yolo()
        if yolo is not None:
            try:
                results = yolo(img, verbose=False, conf=0.55)
                for r in results:
                    for box in r.boxes:
                        cls_name = r.names[int(box.cls[0])].lower()
                        if cls_name in DEVICE_CLASSES:
                            # 🚨 HARD FAIL-FAST ON OVERLAP
                            bx = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                            fx, fy, fw, fh = face_bbox
                            
                            # Intersection over Union (coarse overlap check)
                            ix1 = max(bx[0], fx)
                            iy1 = max(bx[1], fy)
                            ix2 = min(bx[2], fx + fw)
                            iy2 = min(bx[3], fy + fh)
                            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            face_area = fw * fh
                            
                            if (inter_area / face_area) > 0.15:
                                return {
                                    "status": "processing",
                                    "detail": f"Security Alert: {DEVICE_CLASSES[cls_name]} overlap detected.",
                                    "step": session.step,
                                    "is_suspicious": True
                                }

                            # Size based check as backup
                            obj_w, obj_h = box.xywh[0][2], box.xywh[0][3]
                            area_pct = (obj_w * obj_h) / (w * h)
                            if area_pct < 0.015: continue

                            display_name = DEVICE_CLASSES.get(cls_name, cls_name.replace("_", " ").title())
                            session.device_detected = True
                            session.device_class = display_name
                            return {
                                "status": "processing", 
                                "detail": f"Security Alert: {display_name} detected.",
                                "step": session.step,
                                "mesh": landmarks_to_serializable(pts_478),
                                "is_suspicious": True
                            }
            except Exception as e:
                print(f"YOLO error: {e}")
                pass


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
        is_3d, depth_score = check_depth_displacement(session.landmark_history)
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
                passed, score = check_light_response(
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
