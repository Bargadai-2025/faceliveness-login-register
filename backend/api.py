import glob
import os
import sys

_backend_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_backend_dir)

# Ensure venv site-packages are accessible even if uvicorn was started globally (Windows + Linux layouts).
for _venv_base in (os.path.join(_backend_dir, "venv"), os.path.join(_repo_root, "venv")):
    if not os.path.isdir(_venv_base):
        continue
    _candidates = glob.glob(os.path.join(_venv_base, "Lib", "site-packages"))
    _candidates += glob.glob(os.path.join(_venv_base, "lib", "python*", "site-packages"))
    for venv_path in _candidates:
        if os.path.isdir(venv_path) and venv_path not in sys.path:
            sys.path.insert(0, venv_path)

from dotenv import load_dotenv

# Repo-root .env (typical) then backend/.env overrides.
load_dotenv(os.path.join(_repo_root, ".env"))
load_dotenv(os.path.join(_backend_dir, ".env"))

import cv2
import torch
import numpy as np
import base64
import uuid
import secrets
from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
import uvicorn
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Any, Dict
import cloudinary
import cloudinary.uploader

from liveness_session import session_manager, ALL_GESTURE_IDS, LIGHT_CHALLENGES
from frame_processor import process_frame
from face_detection import remove_background
from spoof_scoring import analyze_passive_spoof_single_frame
from liveness_checks import check_reaction_timing
from database import (
    close_db_pool,
    complete_liveness_session_if_valid,
    create_liveness_session,
    ensure_indexes,
    get_face_embedding_by_label,
    get_liveness_session,
    get_valid_completed_liveness_session,
    init_db_pool,
    insert_auth_log,
    insert_face,
    list_face_labels,
    list_faces_for_matching,
    update_liveness_session_status,
)

app = FastAPI()

# CORS: browsers require Access-Control-Allow-Origin on cross-origin fetch.
# Explicit list + regex covers facematch / any *.bargad.ai HTTPS (Railway + Vercel + local).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://facematch.bargad.ai",
        "https://www.facematch.bargad.ai",
        "https://face-match-test-xgua.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://faceliveness.bargad.ai",
        "https://faceliveliness.bargad.ai",
    ],
    allow_origin_regex=r"^https://([a-z0-9-]+\.)*bargad\.ai$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_TTL_MINUTES = 15
SESSION_ISSUE_MAX_ATTEMPTS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Cloudinary Setup
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

MIN_CONFIDENCE_BARGAD = 0.65
MIN_CONFIDENCE_LFW = 0.55
TOP_K = 50

# Set FAST_MATCH=1 in .env for local/dev: skips YOLO + MediaPipe background removal on POST /match (much faster).
FAST_MATCH = os.getenv("FAST_MATCH", "").strip().lower() in ("1", "true", "yes")

# ════════════════════════════════════════════════════════════
# IDENTITY VERIFICATION CALLBACK
# ════════════════════════════════════════════════════════════

def verify_identity_callback(img_bgr, target_embedding):
    """
    Continuous identity check during liveness flow.
    Compares current frame against the selected agent's stored embedding.
    """
    try:
        # Convert BGR (OpenCV) to RGB (MTCNN expects RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Detect Face & Generate Tensor
        face_tensor = mtcnn(img_rgb)
        if face_tensor is None:
            return False # No face found in this frame
            
        # 2. Extract Embedding
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face_tensor).cpu().numpy()[0].astype("float32")
            
        # 3. Normalize & Compare (Cosine Similarity)
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        score = float(np.dot(emb, target_embedding))
        
        print(f"👤 Agent Verification — Score: {score:.3f} (Agent Match: {score > 0.60})")
        
        # Use 0.60 as a robust threshold for continuous matching (higher = stricter)
        return score > 0.60
    except Exception as e:
        print(f"⚠️ Identity check internal error: {e}")
        return False


@app.on_event("startup")
async def _startup():
    await init_db_pool()
    try:
        await ensure_indexes()
    except Exception as e:
        print(f"PostgreSQL index warning: {e}")


@app.on_event("shutdown")
async def _shutdown():
    await close_db_pool()

# End of session index setup

# ════════════════════════════════════════════════════════════
# AGENT SELECTION & VERIFICATION
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════

@app.get("/agents/list")
async def list_agents():
    """Returns a unique list of agent names registered in the database."""
    try:
        # Get unique labels from PostgreSQL, excluding any "txt" placeholders
        labels = await list_face_labels()
        # Return as a list of objects to match the frontend expectation in LoginPage.jsx
        return [{"label": l} for l in labels]
    except Exception as e:
        print(f"❌ Error fetching agent list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════
# NEW BACKEND-DRIVEN LIVENESS ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.post("/liveness/session/start")
async def start_liveness_session(payload: Dict[str, Any] = Body(...)):
    """Start a new backend-driven liveness session."""
    try:
        device_id = (payload or {}).get("device_id")
        if not device_id or not isinstance(device_id, str) or len(device_id) > 128:
            raise HTTPException(status_code=400, detail="Invalid device_id")

        agent_label = (payload or {}).get("agent_label")
        agent_emb = None
        if agent_label:
            embedding = await get_face_embedding_by_label(agent_label)
            if embedding:
                agent_emb = np.array(embedding, dtype="float32")
                # Normalize
                norm = np.linalg.norm(agent_emb)
                if norm > 0:
                    agent_emb = agent_emb / norm

        sess = session_manager.create_session(device_id, agent_label=agent_label, agent_embedding=agent_emb)
        print(f"🆕 Creating session: {sess.session_id} for device: {device_id} (Agent: {agent_label})")

        # Also log to PostgreSQL for audit
        try:
            await create_liveness_session(
                session_id=sess.session_id,
                device_id=device_id,
                gestures=sess.gestures,
                status="issued",
                expires_at=datetime.utcnow() + timedelta(minutes=SESSION_TTL_MINUTES),
                raw_data={
                    "agent_label": agent_label,
                    "mode": "backend_driven",
                },
            )
        except Exception as db_err:
            print(f"⚠️ PostgreSQL liveness log warning: {db_err}")

        return {
            "session_id": sess.session_id,
            "gestures": sess.gestures,
            "step": "calibration",
            "agent_label": agent_label
        }
    except Exception as e:
        print(f"❌ Error in start_liveness_session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/liveness/frame")
async def liveness_frame(
    session_id: str = Form(...),
    frame: UploadFile = File(...),
):
    """Process a single frame through the backend liveness pipeline."""
    sess = session_manager.get(session_id)
    if sess is None:
        raise HTTPException(status_code=400, detail="Invalid or expired session")

    raw = await frame.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty frame")

    result = process_frame(sess, raw, identity_callback=verify_identity_callback)
    return result


@app.post("/liveness/session/complete")
async def complete_liveness_session(payload: Dict[str, Any] = Body(...)):
    """Validate and complete a backend-driven liveness session."""
    session_id = (payload or {}).get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    sess = session_manager.get(session_id)
    if sess is None:
        # Fallback: check PostgreSQL for legacy or already persisted sessions
        now = datetime.utcnow()
        updated = await complete_liveness_session_if_valid(
            session_id=session_id,
            now=now,
            raw_updates={"completed_at": now.isoformat()},
        )
        if not updated:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired session, or already completed.",
            )
        return {"ok": True, "verified": True, "confidence": 0.95}

    # Validate all checks passed
    if sess.step != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Session not complete. Current step: {sess.step}",
        )


    # Check reaction timing
    timing_ok, timing_reason = check_reaction_timing(sess.reaction_times)

    # Calculate confidence
    confidence = 0.5
    if sess.depth_passed:
        confidence += 0.15
    if sess.micro_passed:
        confidence += 0.15
    if sess.light_passed:
        confidence += 0.1
    if sess.all_gestures_done:
        confidence += 0.1
    if timing_ok:
        confidence += 0.05
    confidence = min(round(confidence, 2), 1.0)

    if sess.device_detected:
        print(f"⚠️ Device detected during session ({sess.device_class}), allowing completion with penalty.")
        # Lower baseline confidence if device was detected during liveness
        confidence = max(0.1, confidence - 0.3)

    # Update PostgreSQL
    now = datetime.utcnow()
    await update_liveness_session_status(
        session_id=session_id,
        status="completed",
        raw_updates={
            "completed_at": now,
            "confidence": confidence,
            "depth_passed": sess.depth_passed,
            "micro_passed": sess.micro_passed,
            "light_passed": sess.light_passed,
            "gestures_completed": sess.current_gesture_idx,
            "timing_ok": timing_ok,
        },
    )

    # Remove from in-memory store (keep in PostgreSQL)
    # Don't remove yet — /match still needs to verify it
    sess.step = "verified"

    return {"ok": True, "verified": True, "confidence": confidence}


# ════════════════════════════════════════════════════════════
# LEGACY ENDPOINTS (kept for backward compatibility)
# ════════════════════════════════════════════════════════════

@app.post("/liveness/session")
async def create_liveness_session_legacy(payload: Dict[str, Any] = Body(...)):
    """Legacy session endpoint — redirects to new start."""
    return await start_liveness_session(payload)


@app.post("/liveness/temporal")
async def liveness_temporal(
    session_id: str = Form(...),
    device_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Short burst of frames: static / periodic replay heuristics."""
    if len(files) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 frames")

    sess_doc = await get_liveness_session(
        session_id=session_id,
        device_id=device_id,
    )
    if not sess_doc:
        raise HTTPException(status_code=400, detail="Invalid session")
    if sess_doc.get("expires_at") and sess_doc["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Expired session")

    lum: List[float] = []
    lap_vars: List[float] = []
    for uf in files:
        raw = await uf.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lum.append(float(np.mean(gray)))
        lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    if len(lum) < 4:
        raise HTTPException(status_code=400, detail="Could not decode frames")

    lum = np.array(lum, dtype=np.float64)
    diffs = np.diff(lum)
    motion = float(np.std(diffs))
    lap_std = float(np.std(lap_vars)) if len(lap_vars) >= 2 else 0.0

    replay_risk = 0.0
    if motion < 0.35:
        replay_risk += 0.45
    if lap_std < 1.5:
        replay_risk += 0.25

    if len(diffs) >= 6:
        z = diffs - np.mean(diffs)
        spec = np.abs(np.fft.rfft(z))
        spec = spec / (np.max(spec) + 1e-6)
        freqs = np.fft.rfftfreq(len(z), d=1.0)
        mask = (freqs > 0.18) & (freqs < 0.45)
        if np.any(mask) and float(np.max(spec[mask])) > 0.55:
            replay_risk += 0.35

    replay_risk = min(1.0, round(replay_risk, 3))
    ok = replay_risk < 0.72
    return {"ok": ok, "replay_risk": replay_risk, "motion_std": round(motion, 4)}


# ── /match: accepts optional geo fields ──
@app.post("/match")
async def match_face(
    file: UploadFile = File(...),
    geo_lat: Optional[str] = Form(None),
    geo_long: Optional[str] = Form(None),
    geo_timestamp: Optional[str] = Form(None),
    liveness_session_id: Optional[str] = Form(None),
    device_id: Optional[str] = Form(None),
    errcount: Optional[int] = Form(0),
    expected_label: Optional[str] = Form(None),
):
    print("errcount : ", errcount)
    penalties_breakdown = []
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # Optional liveness verification
        if liveness_session_id and device_id:
            now = datetime.utcnow()
            # Check in-memory session first (backend-driven)
            mem_sess = session_manager.get(liveness_session_id)
            if mem_sess and mem_sess.step == "verified":
                pass  # Backend-driven session verified
            else:
                # Fallback to PostgreSQL (legacy or already-completed)
                sess = await get_valid_completed_liveness_session(
                    session_id=liveness_session_id,
                    device_id=device_id,
                    now=now,
                )
                if not sess:
                    return {"error": "Invalid or expired liveness session. Please start the camera flow again."}
        else:
            print("⚠️ Match requested without liveness session (optional mode)")

        # 1. READ ORIGINAL IMAGE
        img_raw = cv2.imread(temp_path)
        if img_raw is None:
            return {"error": "Could not read image."}

        # 3. DETECT FACE ON ORIGINAL IMAGE
        # Resize for faster/more reliable detection if huge
        h, w = img_raw.shape[:2]
        img_detect = img_raw.copy()
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img_detect = cv2.resize(img_raw, (int(w * scale), int(h * scale)))
        
        img_detect_rgb = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)
        
        face = None
        face_landmarks_mp = None
        try:
            # Primary detection
            face = mtcnn(img_detect_rgb)
            
            # Use MediaPipe for landmarks (needed for ROI-based liveness)
            from face_detection import detect_faces
            mp_faces = detect_faces(img_detect) # detect_faces expects BGR
            if mp_faces:
                orig_h, orig_w = img_raw.shape[:2]
                det_h, det_w = img_detect.shape[:2]
                scale_x = orig_w / det_w
                scale_y = orig_h / det_h
                
                face_landmarks_mp = []
                for p in mp_faces[0]["pts_68"]:
                    face_landmarks_mp.append({
                        "x": p["x"] * scale_x,
                        "y": p["y"] * scale_y
                    })
                
            if face is None:
                # Fallback: MediaPipe detection
                print("⚠️ MTCNN failed on original, trying MediaPipe fallback...")
                if mp_faces:
                    print("✅ MediaPipe found a face. Attempting guided MTCNN...")
                    pts = mp_faces[0]["pts_68"]
                    xs = [p["x"] for p in pts]
                    ys = [p["y"] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    # Expand bbox
                    w_b, h_b = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - w_b * 0.3))
                    y1 = max(0, int(y1 - h_b * 0.3))
                    x2 = min(img_detect.shape[1], int(x2 + w_b * 0.3))
                    y2 = min(img_detect.shape[0], int(y2 + h_b * 0.3))
                    
                    crop_bgr = img_detect[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    face = mtcnn(crop_rgb)

        except Exception as e:
            print(f"Face detection crash: {e}")
            return {"error": f"Face detection engine error: {str(e)}"}

        if face is None:
            return {"error": "No face detected in the uploaded image. Please ensure your face is clearly visible, well-lit, and facing the camera directly."}

        # 4. HARD ANTI-SCREEN DETECTION (YOLO + face-device overlap + single-frame spoof)
        # SECURITY: These scans ALWAYS run regardless of FAST_MATCH.
        # FAST_MATCH only skips cosmetic operations (background removal), NEVER security scans.
        device_replay_score = 0.0
        from frame_processor import _get_yolo, _yolo_device_replay_risk, DEVICE_CLASSES
        yolo = _get_yolo()
        if yolo:
            try:
                # Calculate face bounding box from MediaPipe landmarks
                face_bbox_match = (0, 0, img_raw.shape[1], img_raw.shape[0])  # fallback
                if face_landmarks_mp:
                    lm_xs = [p["x"] for p in face_landmarks_mp]
                    lm_ys = [p["y"] for p in face_landmarks_mp]
                    face_bbox_match = (
                        int(min(lm_xs)), int(min(lm_ys)),
                        int(max(lm_xs) - min(lm_xs)),
                        int(max(lm_ys) - min(lm_ys)),
                    )

                h_raw, w_raw = img_raw.shape[:2]
                # Use same face-device overlap analysis as the liveness flow
                dr, device_hard = _yolo_device_replay_risk(img_raw, face_bbox_match, w_raw, h_raw, yolo)
                device_replay_score = dr
                print(f"🔒 YOLO device scan: risk={dr:.3f}, hard_overlap={device_hard}")

                if device_hard:
                    # HARD REJECT: device directly overlapping face
                    print(f"🚨 HARD REJECT: Device overlapping face in selfie (IoU overlap, dr={dr:.3f})")
                    return {
                        "error": "Security Alert: Electronic device (phone/tablet/screen) detected overlapping your face. Please take a direct selfie without any screens in front of you.",
                        "security_penalty_breakdown": [{
                            "type": "Electronic Device Blocking Face",
                            "penalty": 1.0,
                            "count": 1,
                            "detail": f"YOLO detected device with high face overlap (risk={dr:.2f})"
                        }],
                    }

                if dr > 0.25:
                    # Device visible near face but not fully overlapping
                    print(f"⚠️ Device near face in selfie: risk={dr:.3f}")
                    errcount += 40
                    penalties_breakdown.append({
                        "type": "Electronic Device Near Face",
                        "penalty": 0.40,
                        "count": 1,
                        "detail": f"Device detected near face area (risk={dr:.2f})"
                    })

                # Also do the general YOLO scan for any devices in the full frame
                yolo_results = yolo(img_raw, verbose=False, conf=0.30)
                for r in yolo_results:
                    for box in r.boxes:
                        cls_name = r.names[int(box.cls[0])].lower()
                        if cls_name in DEVICE_CLASSES:
                            display_name = DEVICE_CLASSES[cls_name]
                            print(f"🚨 YOLO Detect: {display_name} found in selfie frame!")
                            if dr < 0.25:  # Only add if not already penalized above
                                errcount += 30
                                penalties_breakdown.append({
                                    "type": "Electronic Device Detected",
                                    "penalty": 0.30,
                                    "count": 1,
                                    "detail": f"YOLO detected: {display_name}"
                                })
            except Exception as e:
                print(f"YOLO selfie check error: {e}")

        # Run spoof analysis in SINGLE-FRAME MODE — ALWAYS runs (security critical)
        # This lets imaging signals (moiré, texture, scanlines) detect screen replay
        extra_signals_match = {"device_replay": device_replay_score} if device_replay_score > 0.01 else None
        spoof_detail = analyze_passive_spoof_single_frame(
            img_raw,
            face_landmarks_mp,
            strict=True,
            roi_luminance_history=None,
            last_gray_small=None,
            curr_gray_small=None,
            landmark_centroid_history=None,
            extra_signals=extra_signals_match,
            single_frame_mode=True,  # KEY: relaxed gate for selfie capture
        )
        live_ok = bool(spoof_detail["is_live"])
        triggered_rules = spoof_detail.get("triggered_rules", [])
        
        # USER REQUEST: Explicitly block if ambient/reflected light from a screen is captured
        if "reflection" in triggered_rules or "rectangular_glare" in triggered_rules or spoof_detail.get("confidence_per_signal", {}).get("reflection_raw", 0.0) > 0.45:
            print("🚨 HARD REJECT: Ambient light / screen reflection detected.")
            return {
                "error": "Ambient light / screen reflection detected. Please avoid capturing photos of screens or devices."
            }

        live_score = max(0.0, min(1.0, 1.0 - spoof_detail["total_spoof_score"] / 100.0))
        live_reason = "OK" if live_ok else (
            "Weighted spoof: " + ", ".join(triggered_rules)[:220]
        )
        print(f"🔒 Spoof analysis: score={spoof_detail['total_spoof_score']:.1f}/{spoof_detail.get('reject_threshold', 68)}, live={live_ok}, rules={triggered_rules}")
        if not live_ok:
            print(f"🚨 Single-frame spoof FAILED: score={spoof_detail['total_spoof_score']}/{spoof_detail.get('reject_threshold', 68)}")
            errcount += 35
            penalties_breakdown.append({
                "type": "Digital Media / Screen Detected",
                "penalty": 0.35,
                "count": 1,
                "detail": f"Spoof score {spoof_detail['total_spoof_score']:.0f}/{spoof_detail.get('reject_threshold', 68):.0f} — " + live_reason[:400],
            })

        # 5. PREPARE PROCESSED IMAGE FOR UI (with background removal; FAST_MATCH uses a small RGB resize only)
        if FAST_MATCH:
            print("⚡ FAST_MATCH: skipping remove_background on /match")
            h0, w0 = img_raw.shape[:2]
            mx = 640
            if max(h0, w0) > mx:
                s = mx / max(h0, w0)
                small_bgr = cv2.resize(img_raw, (int(w0 * s), int(h0 * s)))
            else:
                small_bgr = img_raw.copy()
            img_processed = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_processed = remove_background(img_raw)
            if max(img_processed.shape[:2]) > 800:
                scale = 800 / max(img_processed.shape[:2])
                img_processed = cv2.resize(img_processed, (int(img_processed.shape[1] * scale), int(img_processed.shape[0] * scale)))

        # 5. GENERATE EMBEDDING
        face = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face).cpu().numpy()[0].astype("float32")
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        if np.isnan(emb).any():
            return {"error": "Face analysis yielded invalid results. Please check lighting and try again."}

        all_docs = await list_faces_for_matching()
        print(f"📦 Loaded {len(all_docs)} faces from PostgreSQL")

        valid_rows = []
        for doc in all_docs:
            if not doc.get("embedding") or len(doc["embedding"]) != 512:
                continue
            db_emb = np.array(doc["embedding"], dtype="float32")
            norm = np.linalg.norm(db_emb)
            if norm == 0:
                continue
            valid_rows.append((doc, db_emb / norm))

        raw_results = []
        if valid_rows:
            matrix = np.stack([r[1] for r in valid_rows], axis=0)
            scores_vec = matrix @ emb
            for i, (doc, _) in enumerate(valid_rows):
                score = float(scores_vec[i])
                d_type = doc.get("doc_type") or "Selfie"
                raw_results.append({
                    "label": doc["label"],
                    "source": doc["source"],
                    "image_url": doc["image_url"],
                    "doc_type": d_type,
                    "score": round(score, 3),
                })

        raw_results.sort(key=lambda x: x["score"], reverse=True)
        raw_results = raw_results[:TOP_K]

        print("\n🔍 Top 5 raw matches:")
        for r in raw_results[:5]:
            print(f"  {r['label']} ({r['source']}) → {r['score']:.3f} [Type: {r.get('doc_type')}]")

        print(f"\n🔍 Matching against {len(raw_results)} candidates...")
        seen = {}
        for r in raw_results:
            # Senior Debug Tip: Normalize labels AND merge across sources for consistent identity tracking
            normalized_label = " ".join(str(r["label"]).split())
            key = normalized_label.lower() 
            
            score = round(float(r["score"]), 3)
            print(f"  - Checking {key}: Score {score}, Type in raw: {r.get('doc_type')}")
            
            threshold = MIN_CONFIDENCE_BARGAD if r["source"] in ["bargad", "frontend_reg"] else MIN_CONFIDENCE_LFW
            if score < threshold:
                continue
            if key not in seen:
                seen[key] = {
                    "label": r["label"],
                    "source": r["source"],
                    "registered_doc_type": r.get("doc_type") or "Selfie",
                    "verification_type": "Selfie",
                    "confidence": score,
                    "matched_image": r["image_url"],
                    "images": [r["image_url"]]
                }
            else:
                if r["image_url"] not in seen[key]["images"]:
                    seen[key]["images"].append(r["image_url"])
                
                # Merging Logic:
                # 1. Update overall confidence/image if this match is stronger
                if score > seen[key]["confidence"]:
                    seen[key]["confidence"] = score
                    seen[key]["matched_image"] = r["image_url"]
                
                # 2. Update registered_doc_type only if the new record has a specific ID (not Selfie/None)
                new_type = r.get("doc_type")
                if new_type and str(new_type).lower() != "selfie":
                    seen[key]["registered_doc_type"] = new_type
                elif not seen[key].get("registered_doc_type"):
                    seen[key]["registered_doc_type"] = "Selfie"

        results = list(seen.values())
        results.sort(key=lambda x: x["confidence"], reverse=True)

        # Apply security penalty from errcount (each point reduces confidence by 1%)
        if errcount > 0 or True: # Always show breakdown if results exist
            base_conf = results[0]["confidence"] if results else 0.0
            # Increase penalty: each errcount point now reduces confidence by 3% (instead of 1%)
            penalty = float(errcount) * 0.03
            
            # Always add base similarity as the first entry
            penalties_breakdown.insert(0, {
                "type": "Base Face Similarity",
                "penalty": 0.0,
                "count": 1,
                "detail": f"Raw similarity score: {int(round(base_conf * 100))}%"
            })

            if errcount > 0:
                print(f"⚠️ Applying security penalty: -{penalty:.3f} (from errcount={errcount})")
                penalties_breakdown.append({
                    "type": "Digital Media Detection",
                    "penalty": round(penalty, 3),
                    "count": errcount // 10,
                    "detail": "Suspicious activity patterns detected during liveness flow."
                })

                for r in results:
                    r["confidence"] = max(0.0, round(r["confidence"] - penalty, 3))
            
            # Removed hard rejection as per user request to show penalty table instead
            # if errcount >= 20:
            #     return {"error": "Security Alert: High risk of digital spoofing detected during session. Matching blocked."}
            
            # Re-sort results
            results.sort(key=lambda x: x["confidence"], reverse=True)

        # Log geo data
        if geo_lat and geo_long:
            await insert_auth_log(
                timestamp=geo_timestamp or datetime.utcnow(),
                geo_lat=geo_lat,
                geo_long=geo_long,
                top_match=results[0]["label"] if results else "no_match",
                match_count=len(results),
                raw_data={"geo_timestamp": geo_timestamp} if geo_timestamp else {},
            )
            print(f"📍 Geo logged: {geo_lat}, {geo_long}")

        if not results:
            return {"error": "No confident match found in the dataset."}

        # Identity Verification Check (Disabled as per user request to allow anyone to match)
        # if expected_label and results:
        #     top_label = results[0]["label"].lower().strip()
        #     target_label = expected_label.lower().strip()
        #     # Handle underscores/spaces mismatch
        #     top_label = top_label.replace("_", " ")
        #     target_label = target_label.replace("_", " ")
        #     
        #     if top_label != target_label:
        #         print(f"❌ Identity Mismatch: Expected '{target_label}', but matched '{top_label}' ({results[0]['confidence']*100:.0f}%)")
        #         return {
        #             "error": f"Identity mismatch. You are matched as {results[0]['label']} ({results[0]['confidence']*100:.0f}%), but you are logged in as {expected_label}. Please use the correct account."
        #         }

        if liveness_session_id:
            await update_liveness_session_status(
                session_id=liveness_session_id,
                status="consumed",
                raw_updates={"consumed_at": datetime.utcnow()},
            )
            # Clean up in-memory session
            session_manager.remove(liveness_session_id)

        # Encode processed image to base64 for frontend display
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
        processed_b64 = base64.b64encode(buffer).decode("utf-8")

        # Encode the ORIGINAL captured selfie (not background-removed) for comparison
        _, raw_buffer = cv2.imencode(".jpg", img_raw)
        captured_b64 = base64.b64encode(raw_buffer).decode("utf-8")

        return {
            "matches": results,
            "processed_image": f"data:image/jpeg;base64,{processed_b64}",
            "captured_image": f"data:image/jpeg;base64,{captured_b64}",
            "security_penalty_breakdown": penalties_breakdown,
            "capture_live_ok": bool(live_ok),
            "capture_live_score": round(float(live_score), 4),
            "capture_live_reason": live_reason,
            "spoof_score": spoof_detail["total_spoof_score"],
            "spoof_reject_threshold": spoof_detail.get("reject_threshold"),
            "triggered_rules": spoof_detail.get("triggered_rules", []),
            "confidence_per_signal": spoof_detail.get("confidence_per_signal", {}),
            "reflection_classification": spoof_detail.get("reflection_classification"),
            "final_replay_risk": spoof_detail["total_spoof_score"],
            "correlation_gate_notes": spoof_detail.get("correlation_gate_notes", []),
            "extra_signals_used": spoof_detail.get("extra_signals_used", {}),
        }

    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/register")
async def register_user(
    file: Optional[UploadFile] = File(None),
    firstName: str = Form(...),
    middleName: Optional[str] = Form(None),
    lastName: str = Form(...),
    docType: str = Form("Aadhar"),
    document: Optional[UploadFile] = File(None),
    liveness_session_id: Optional[str] = Form(None),
    device_id: str = Form(...)
):
    """Register a new user. Crops face from image for a cleaner database profile."""
    
    primary_file = file or document
    if not primary_file:
        return {"error": "No image source provided (Selfie or Document required)"}

    temp_path = f"temp_reg_{primary_file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await primary_file.read())

    doc_path = None
    if document and document != primary_file:
        doc_path = f"temp_doc_{document.filename}"
        with open(doc_path, "wb") as f:
            f.write(await document.read())

    crop_path = f"temp_crop_{primary_file.filename}.jpg"

    try:
        if file and liveness_session_id:
            mem_sess = session_manager.get(liveness_session_id)
            if not mem_sess or mem_sess.step != "verified":
                return {"error": "Security check required for selfie registration."}

        img_raw = cv2.imread(temp_path)
        if img_raw is None:
            return {"error": "Could not read image."}

        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        # Detect and Crop logic
        face_tensor = mtcnn(img_rgb)
        
        # We need the bounding box to crop the original image nicely for the UI
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Expand crop slightly (30%)
            bw, bh = x2 - x1, y2 - y1
            x1, y1 = max(0, int(x1 - bw * 0.3)), max(0, int(y1 - bh * 0.3))
            x2, y2 = min(img_raw.shape[1], int(x2 + bw * 0.3)), min(img_raw.shape[0], int(y2 + bh * 0.3))
            
            face_crop = img_raw[y1:y2, x1:x2]
            cv2.imwrite(crop_path, face_crop)
        else:
            # Fallback to MediaPipe detection if MTCNN fails to find a box
            from face_detection import detect_faces
            mp_faces = detect_faces(img_raw)
            if mp_faces:
                pts = mp_faces[0]["pts_68"]
                xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                bw, bh = x2 - x1, y2 - y1
                x1, y1 = max(0, int(x1 - bw * 0.4)), max(0, int(y1 - bh * 0.4))
                x2, y2 = min(img_raw.shape[1], int(x2 + bw * 0.4)), min(img_raw.shape[0], int(y2 + bh * 0.4))
                face_crop = img_raw[y1:y2, x1:x2]
                cv2.imwrite(crop_path, face_crop)
            else:
                return {"error": "No face detected in photo. Please ensure your face is clearly visible."}

        # Embedding from the cropped/tensorized face
        if face_tensor is None:
            # Re-detect on crop if first pass failed
            crop_rgb = cv2.cvtColor(cv2.imread(crop_path), cv2.COLOR_BGR2RGB)
            face_tensor = mtcnn(crop_rgb)
            if face_tensor is None:
                return {"error": "Face detection failed during processing."}

        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face_tensor).cpu().numpy()[0].astype("float32")
        emb = emb / np.linalg.norm(emb)

        # 4. UPLOAD TO CLOUDINARY
        full_name = f"{firstName}_{middleName}_{lastName}" if middleName else f"{firstName}_{lastName}"
        clean_name = full_name.replace(' ', '_')
        
        # Upload the CROPPED face as the main profile image
        upload = cloudinary.uploader.upload(
            crop_path,
            folder=f"facematch/users/{clean_name}",
            overwrite=True
        )
        image_url = upload["secure_url"]

        # Optional: Upload the full original document as a separate reference
        document_url = None
        if doc_path:
            doc_upload = cloudinary.uploader.upload(
                doc_path,
                folder=f"facematch/docs/{clean_name}",
                overwrite=True
            )
            document_url = doc_upload["secure_url"]
        elif not file and primary_file: # If we registered via document, the primary_file IS the document
            doc_upload = cloudinary.uploader.upload(
                temp_path,
                folder=f"facematch/docs/{clean_name}",
                overwrite=True
            )
            document_url = doc_upload["secure_url"]

        # 5. SAVE TO POSTGRESQL
        # Normalize label to avoid whitespace issues (Senior Dev Best Practice)
        clean_label = " ".join(f"{firstName} {lastName}".split())
        
        await insert_face(
            label=clean_label,
            source="frontend_reg",
            image_url=image_url,
            embedding=emb.tolist(),
        )

        # 6. CONSUME SESSION (ONLY IF PROVIDED)
        if liveness_session_id:
            session_manager.remove(liveness_session_id)
            await update_liveness_session_status(
                session_id=liveness_session_id,
                status="consumed",
                raw_updates={
                    "consumed_at": datetime.utcnow(),
                    "registration": {
                        "first_name": firstName.strip(),
                        "middle_name": middleName.strip() if middleName else None,
                        "last_name": lastName.strip(),
                        "doc_type": str(docType or "Selfie"),
                        "document_url": document_url,
                    },
                },
            )

        return {
            "success": True,
            "message": f"Successfully registered {firstName} {lastName}!",
            "image_url": image_url,
            "document_url": document_url
        }

    except Exception as e:
        print(f"❌ Registration error: {e}")
        return {"error": f"Registration failed: {str(e)}"}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if doc_path and os.path.exists(doc_path):
            os.remove(doc_path)
        if 'crop_path' in locals() and os.path.exists(crop_path):
            os.remove(crop_path)


# ── Passive liveness endpoint ──
@app.post("/liveness")
async def check_liveness(file: UploadFile = File(...)):
    temp_path = f"temp_live_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        img = cv2.imread(temp_path)
        if img is None:
            return {"live": False, "score": 0.0, "reason": "Cannot read image"}

        rep = analyze_passive_spoof_single_frame(img, None, strict=False)
        is_live = bool(rep["is_live"])
        score = max(0.0, min(1.0, 1.0 - rep["total_spoof_score"] / 100.0))
        reason = "OK" if is_live else f"Spoof score {rep['total_spoof_score']}"
        print(f"🧪 Liveness — spoof={rep['total_spoof_score']}, live={is_live}")

        return {
            "live": is_live,
            "score": score,
            "reason": reason,
            "spoof_score": rep["total_spoof_score"],
            "triggered_rules": rep.get("triggered_rules", []),
            "confidence_per_signal": rep.get("confidence_per_signal", {}),
            "reflection_classification": rep.get("reflection_classification"),
        }

    except Exception as e:
        return {"live": False, "score": 0.0, "reason": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
