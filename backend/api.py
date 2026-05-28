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

import asyncio
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
from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Any, Dict
import cloudinary
import cloudinary.uploader

from liveness_session import session_manager, ALL_GESTURE_IDS, LIGHT_CHALLENGES
from frame_processor import process_frame, warmup_yolo
from face_detection import remove_background
from embedding_pipeline import (
    FACE_DETECT_MAX_SIDE,
    create_face_models,
    extract_face_embedding,
    load_match_thresholds,
)
from post_selfie_security import load_post_selfie_config, run_post_selfie_security, scan_yolo_devices
from challenge_frame_verification import assess_challenge_continuity
from frame_processor import _get_yolo
from poc_logging import log_event, log_security_verdict
from api_errors import user_error, safe_exception_message, USER_MESSAGES, public_dict
from rate_limit import RateLimitMiddleware
from poc_helpers import enrich_match_security_payload
from spoof_scoring import analyze_passive_spoof_single_frame
from liveness_checks import check_reaction_timing
from database import (
    close_db_pool,
    complete_liveness_session_if_valid,
    create_liveness_session,
    ensure_indexes,
    fetchrow,
    get_face_embedding_by_label,
    get_liveness_session,
    get_valid_completed_liveness_session,
    init_db_pool,
    insert_auth_log,
    insert_app_user,
    insert_face,
    email_exists,
    is_registered_email,
    resolve_registered_face_label,
    list_face_labels,
    list_faces_for_matching,
    list_faces_for_matching_by_label,
    update_liveness_session_status,
)

app = FastAPI()

REGISTER_REQUIRE_LIVENESS = os.getenv("REGISTER_REQUIRE_LIVENESS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

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

app.add_middleware(RateLimitMiddleware)

SESSION_TTL_MINUTES = 15
SESSION_ISSUE_MAX_ATTEMPTS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn, model = create_face_models(DEVICE)
MATCH_THRESHOLDS = load_match_thresholds()

# Cloudinary Setup
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

MIN_CONFIDENCE_BARGAD = MATCH_THRESHOLDS["bargad"]
MIN_CONFIDENCE_LFW = MATCH_THRESHOLDS["lfw"]
MATCH_MIN_TOP2_MARGIN = MATCH_THRESHOLDS["top2_margin"]
TOP_K = 50

# Local/dev only: skips remove_background on POST /match (YOLO + spoof always run).
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
    from database import count_faces, get_database_dsn

    await init_db_pool()
    try:
        await ensure_indexes()
    except Exception as e:
        print(f"PostgreSQL index warning: {e}")

    dsn = get_database_dsn()
    host_hint = "configured"
    if "@" in dsn:
        host_hint = dsn.split("@", 1)[-1].split("/")[0].split("?")[0]
    try:
        counts = await count_faces()
        face_rows = counts.get("with_512", 0)
    except Exception:
        face_rows = "unavailable"
    pad = load_post_selfie_config()
    from frame_processor import LIVENESS_FAST_SETUP

    print(
        f"🧬 Match config: device={DEVICE}, detect_max_side={FACE_DETECT_MAX_SIDE}, "
        f"FAST_MATCH={FAST_MATCH}, bargad>={MIN_CONFIDENCE_BARGAD}, lfw>={MIN_CONFIDENCE_LFW}, "
        f"top2_margin>={MATCH_MIN_TOP2_MARGIN}, yolo_conf={pad['yolo_conf']}, "
        f"hard_reject_spoof={pad['hard_reject_spoof']}, db_host={host_hint}, faces_512={face_rows}, "
        f"torch={torch.__version__}, register_api=email-v2, liveness_fast_setup={LIVENESS_FAST_SETUP}"
    )
    asyncio.create_task(asyncio.to_thread(warmup_yolo))
    from liveness_session import session_manager as _sm

    redis_on = hasattr(_sm, "_redis_ok") and getattr(_sm, "_redis_ok", False)
    log_event("api_startup", extra={"redis_sessions": redis_on, "device": DEVICE})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        body = detail
    else:
        body = user_error("SERVER_ERROR", http_status=exc.status_code)
        body["error"] = USER_MESSAGES.get("SERVER_ERROR") if exc.status_code >= 500 else str(detail)[:120]
    status = int(body.pop("_http_status", exc.status_code)) if isinstance(body, dict) and "_http_status" in body else exc.status_code
    if isinstance(body, dict) and "error_code" not in body:
        body.setdefault("error_code", "HTTP_ERROR")
        body.setdefault("user_message", body.get("error"))
        body.setdefault("retry_allowed", exc.status_code in (429, 503))
    return JSONResponse(status_code=status, content=body if isinstance(body, dict) else {"error": str(detail)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_event("unhandled_exception", level="error", extra={"path": request.url.path, "reason": str(exc)[:200]})
    body = user_error("SERVER_ERROR", retry_allowed=True, http_status=500)
    status = int(body.pop("_http_status", 500))
    return JSONResponse(status_code=status, content=body)


@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    """Readiness: DB + face models loaded."""
    from liveness_session import session_manager as _sm

    db_ok = False
    try:
        from database import count_faces

        await count_faces()
        db_ok = True
    except Exception as e:
        log_event("health_db_failed", level="warning", extra={"reason": str(e)[:80]})
    models_ok = mtcnn is not None and model is not None
    redis_ok = getattr(_sm, "_redis_ok", False) if hasattr(_sm, "_redis_ok") else False
    ready = db_ok and models_ok
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "database": db_ok,
            "models": models_ok,
            "redis_sessions": redis_ok,
            "fast_match": FAST_MATCH,
        },
    )


@app.get("/health/match-config")
async def health_match_config():
    """Runtime face-match settings (compare production vs local)."""
    from database import count_faces, get_database_dsn

    dsn = get_database_dsn()
    host_hint = "local"
    if "@" in dsn:
        host_hint = dsn.split("@", 1)[-1].split("/")[0].split("?")[0]
    try:
        counts = await count_faces()
    except Exception as e:
        counts = {"error": str(e)}
    return {
        "device": DEVICE,
        "torch_version": torch.__version__,
        "model": "InceptionResnetV1_vggface2",
        "embedding_dim": 512,
        "face_detect_max_side": FACE_DETECT_MAX_SIDE,
        "fast_match": FAST_MATCH,
        "min_confidence_bargad": MIN_CONFIDENCE_BARGAD,
        "min_confidence_lfw": MIN_CONFIDENCE_LFW,
        "min_top2_margin": MATCH_MIN_TOP2_MARGIN,
        "db_host": host_hint,
        "face_counts": counts,
        "post_selfie_security": load_post_selfie_config(),
        "liveness_fast_setup": __import__("frame_processor").LIVENESS_FAST_SETUP,
    }


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


@app.get("/auth/check-email")
async def check_email_registered(email: str):
    """
    Verify email is registered before allowing FaceMatch login.
    face_label is the value stored in faces.label (typically the email).
    """
    email_clean = (email or "").strip().lower()
    if not email_clean or "@" not in email_clean or len(email_clean) < 5:
        return {"ok": False, "error": "Please provide a valid email address."}

    face_label = await resolve_registered_face_label(email_clean)
    if not face_label:
        return {
            "ok": False,
            "error": "This email is not registered. Please register first.",
        }

    return {"ok": True, "email": email_clean, "face_label": face_label}


# ════════════════════════════════════════════════════════════
# NEW BACKEND-DRIVEN LIVENESS ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.post("/liveness/session/start")
async def start_liveness_session(payload: Dict[str, Any] = Body(...)):
    """Start a new backend-driven liveness session."""
    try:
        device_id = (payload or {}).get("device_id")
        if not device_id or not isinstance(device_id, str) or len(device_id) > 128:
            raise HTTPException(status_code=400, detail=user_error("DEVICE_INVALID"))

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

        asyncio.create_task(asyncio.to_thread(warmup_yolo))

        return {
            "session_id": sess.session_id,
            "gestures": sess.gestures,
            "step": "calibration",
            "agent_label": agent_label
        }
    except HTTPException:
        raise
    except Exception as e:
        log_event("liveness_start_failed", level="error", extra={"reason": str(e)[:200]})
        raise HTTPException(status_code=500, detail=user_error("LIVENESS_START_FAILED", retry_allowed=True, http_status=500))


@app.post("/liveness/frame")
async def liveness_frame(
    session_id: str = Form(...),
    frame: UploadFile = File(...),
):
    """Process a single frame through the backend liveness pipeline."""
    sess = session_manager.get(session_id)
    if sess is None:
        raise HTTPException(status_code=400, detail=user_error("SESSION_INVALID", retry_allowed=True))

    raw = await frame.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail=user_error("EMPTY_FRAME", retry_allowed=True))

    result = await asyncio.to_thread(
        process_frame, sess, raw, verify_identity_callback
    )
    if hasattr(session_manager, "touch"):
        session_manager.touch(sess)
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
        raise HTTPException(status_code=400, detail=user_error("SESSION_INCOMPLETE", retry_allowed=True))

    if not sess.all_gestures_done:
        raise HTTPException(status_code=400, detail=user_error("SESSION_INCOMPLETE", retry_allowed=True))

    if not sess.depth_passed or not sess.light_passed or not sess.micro_passed:
        raise HTTPException(status_code=400, detail=user_error("SESSION_INCOMPLETE", retry_allowed=True))

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

    # Don't remove yet — /match still needs challenge snapshots
    sess.step = "verified"
    if hasattr(session_manager, "touch"):
        session_manager.touch(sess, force=True)

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
    liveness_ref_photo: Optional[str] = Form(None),
):
    print("errcount : ", errcount)
    penalties_breakdown = []
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # Strict liveness verification
        if not liveness_session_id or not device_id:
            return public_dict(user_error("MATCH_SESSION_REQUIRED", retry_allowed=False))

        now = datetime.utcnow()
        # Check in-memory session first (backend-driven)
        mem_sess = session_manager.get(liveness_session_id)
        if mem_sess and mem_sess.step in ("complete", "verified"):
            pass  # Backend-driven session verified
        else:
            # Fallback to PostgreSQL (legacy or already-completed)
            sess = await get_valid_completed_liveness_session(
                session_id=liveness_session_id,
                device_id=device_id,
                now=now,
            )
            if not sess:
                return public_dict(user_error("SESSION_INVALID", retry_allowed=False))

        # 1. READ ORIGINAL IMAGE
        img_raw = cv2.imread(temp_path)
        if img_raw is None:
            return public_dict(user_error("IMAGE_READ_FAILED", retry_allowed=True))

        # 3. DETECT FACE (canonical pipeline — same max_side as build_db indexing)
        try:
            face_out = extract_face_embedding(img_raw, mtcnn, model, DEVICE)
        except Exception as e:
            log_event("face_detection_crash", level="error", session_id=liveness_session_id, extra={"reason": str(e)[:200]})
            return public_dict(user_error("SERVER_ERROR", retry_allowed=True, http_status=500))

        if not face_out["ok"]:
            return public_dict(user_error("FACE_NOT_FOUND", retry_allowed=True))

        emb = face_out["embedding"]
        face_landmarks_mp = face_out["face_landmarks_mp"]

        # Challenge frame continuity (in-memory only; fused into risk engine below)
        mem_sess_verify = session_manager.get(liveness_session_id)
        identity_assessment: Dict[str, Any] = {"skipped": True}
        stream_risk_ema = float(getattr(mem_sess_verify, "stream_risk_ema", 0.0) if mem_sess_verify else 0.0)
        selfie_devices: List[str] = []
        if mem_sess_verify and mem_sess_verify.challenge_snapshots:
            yolo = _get_yolo()
            if yolo and face_landmarks_mp:
                lm_xs = [p["x"] for p in face_landmarks_mp]
                lm_ys = [p["y"] for p in face_landmarks_mp]
                face_bbox_match = (
                    int(min(lm_xs)),
                    int(min(lm_ys)),
                    int(max(lm_xs) - min(lm_xs)),
                    int(max(lm_ys) - min(lm_ys)),
                )
                pad_cfg = load_post_selfie_config()
                _, _, selfie_devices = scan_yolo_devices(
                    img_raw,
                    face_bbox_match,
                    yolo,
                    conf=pad_cfg["yolo_conf"],
                    hard_iou=pad_cfg["device_hard_iou"],
                )
            identity_assessment = assess_challenge_continuity(
                mem_sess_verify,
                img_raw,
                emb,
                face_landmarks_mp,
                mtcnn=mtcnn,
                model=model,
                device=DEVICE,
                selfie_devices=selfie_devices,
            )

        # FaceNet comparison with liveness reference photo if provided
        if liveness_ref_photo:
            try:
                # Remove data URL header if present
                if "," in liveness_ref_photo:
                    _, base64_data = liveness_ref_photo.split(",", 1)
                else:
                    base64_data = liveness_ref_photo
                img_data = base64.b64decode(base64_data)
                nparr = np.frombuffer(img_data, np.uint8)
                ref_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if ref_img is None:
                    return {"error": "Failed to decode the liveness reference photo."}

                ref_face_out = extract_face_embedding(ref_img, mtcnn, model, DEVICE)
                if not ref_face_out["ok"]:
                    return {
                        "error": "No face detected in the liveness reference photo. Please keep your face clearly visible and still during the preparation phase."
                    }
                ref_emb = ref_face_out["embedding"]

                similarity_score = float(np.dot(ref_emb, emb))
                print(f"🔄 FaceNet comparison score between liveness photo and capture image: {similarity_score:.4f}")

                # Set threshold for liveness vs capture (0.70 is standard and safe)
                LIVENESS_COMPARE_THRESHOLD = float(os.getenv("LIVENESS_COMPARE_THRESHOLD", "0.70"))
                if similarity_score < LIVENESS_COMPARE_THRESHOLD:
                    return {
                        "error": "Liveness process user and captured image user are totally different."
                    }
            except Exception as e:
                print(f"Error during liveness reference comparison: {e}")
                return {"error": f"Liveness reference verification error: {str(e)}"}

        # 4. POST-SELFIE SECURITY (YOLO devices, ambient light, PAD spoof) — always runs
        sec = run_post_selfie_security(
            img_raw,
            face_landmarks_mp,
            errcount=errcount,
            penalties_breakdown=penalties_breakdown,
            identity_assessment=identity_assessment,
            stream_risk_ema=stream_risk_ema,
        )
        if sec.get("error"):
            out = {
                "error": sec["error"],
                "user_message": sec.get("user_message", sec["error"]),
                "security_verdict": sec.get("security_verdict", "reject"),
                "composite_risk": sec.get("composite_risk"),
                "risk_factors": sec.get("risk_factors"),
                "capture_live_ok": False,
                "screen_replay": bool(sec.get("screen_replay")),
                "digital_media": bool(sec.get("digital_media", sec.get("screen_replay"))),
                "user_mismatch": bool(sec.get("user_mismatch")),
            }
            if sec.get("security_penalty_breakdown"):
                out["security_penalty_breakdown"] = sec["security_penalty_breakdown"]
            if sec.get("spoof_detail"):
                sd = sec["spoof_detail"]
                out["spoof_score"] = sd.get("total_spoof_score")
                out["triggered_rules"] = sd.get("triggered_rules", [])
            enrich_match_security_payload(
                out,
                identity_assessment=identity_assessment,
                screen_replay=bool(sec.get("screen_replay")),
            )
            log_security_verdict(
                phase="match",
                verdict=out.get("security_verdict", "reject"),
                session_id=liveness_session_id,
                device_id=device_id,
                composite_risk=out.get("composite_risk"),
                reason=out.get("error"),
                retry_allowed=out.get("retry_allowed", False),
                risk_factors=out.get("risk_factors"),
            )
            return out

        errcount = sec["errcount"]
        penalties_breakdown = sec["penalties_breakdown"]
        spoof_detail = sec["spoof_detail"]
        live_ok = sec["live_ok"]
        live_reason = sec["live_reason"]
        security_verdict = sec.get("security_verdict", "pass")
        composite_risk = float(sec.get("composite_risk", 0.0))
        risk_factors = sec.get("risk_factors")
        risk_confidence_penalty = float(sec.get("confidence_penalty", 0.0))
        live_score = max(0.0, min(1.0, 1.0 - spoof_detail["total_spoof_score"] / 100.0))

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

        scoped_label = None
        if expected_label and str(expected_label).strip():
            scoped_label = " ".join(str(expected_label).split()).strip()

        if scoped_label:
            all_docs = await list_faces_for_matching_by_label(scoped_label)
            print(
                f"🔐 Scoped 1:1 match for logged-in user '{scoped_label}': "
                f"{len(all_docs)} registered embedding(s)"
            )
            if not all_docs:
                return {
                    "error": (
                        f"No registered face found for {scoped_label}. "
                        "Please complete registration first."
                    ),
                }
        else:
            all_docs = await list_faces_for_matching()
            print(f"📦 Loaded {len(all_docs)} faces from PostgreSQL (open search)")

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

        match_diagnostics = {
            "detect_max_side": FACE_DETECT_MAX_SIDE,
            "fast_match": FAST_MATCH,
            "scoped_match": bool(scoped_label),
            "scoped_label": scoped_label,
            "top_raw_label": raw_results[0]["label"] if raw_results else None,
            "top_raw_score": raw_results[0]["score"] if raw_results else None,
            "second_raw_label": raw_results[1]["label"] if len(raw_results) > 1 else None,
            "second_raw_score": raw_results[1]["score"] if len(raw_results) > 1 else None,
            "top2_margin": (
                round(float(raw_results[0]["score"]) - float(raw_results[1]["score"]), 4)
                if len(raw_results) > 1
                else None
            ),
        }

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

        # Ambiguous top-2: only for open search (not logged-in 1:1 scoped match).
        if not scoped_label and len(raw_results) >= 2:
            top_raw = raw_results[0]
            second_raw = raw_results[1]
            top_key = " ".join(str(top_raw["label"]).split()).lower()
            second_key = " ".join(str(second_raw["label"]).split()).lower()
            margin = match_diagnostics["top2_margin"]
            if top_key != second_key and margin is not None and margin < MATCH_MIN_TOP2_MARGIN:
                skip_ambiguous = False
                if expected_label:
                    target_key = " ".join(str(expected_label).split()).lower()
                    top_threshold = (
                        MIN_CONFIDENCE_BARGAD
                        if top_raw.get("source") in ("bargad", "frontend_reg")
                        else MIN_CONFIDENCE_LFW
                    )
                    if top_key == target_key and float(top_raw["score"]) >= top_threshold:
                        skip_ambiguous = True
                        print(
                            f"✓ Ambiguous check skipped: logged-in user '{expected_label}' "
                            f"is top match ({top_raw['score']:.3f})"
                        )
                if not skip_ambiguous:
                    print(
                        f"⚠️ Ambiguous match: {top_raw['label']}={top_raw['score']:.3f} vs "
                        f"{second_raw['label']}={second_raw['score']:.3f} "
                        f"(margin {margin:.3f} < {MATCH_MIN_TOP2_MARGIN})"
                    )
                    return {
                        "error": "Match ambiguous — multiple identities scored too closely. Please retake the selfie with even lighting and face centered.",
                        "match_diagnostics": match_diagnostics,
                    }

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

            total_penalty = penalty + risk_confidence_penalty
            if total_penalty > 0:
                print(
                    f"⚠️ Applying security penalty: -{total_penalty:.3f} "
                    f"(errcount={errcount}, risk_penalty={risk_confidence_penalty:.3f})"
                )
                if penalty > 0:
                    penalties_breakdown.append({
                        "type": "Liveness Session Risk",
                        "penalty": round(penalty, 3),
                        "count": errcount // 10,
                        "detail": "Accumulated liveness stream signals.",
                    })
                for r in results:
                    r["confidence"] = max(0.0, round(r["confidence"] - total_penalty, 3))
            
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
            if scoped_label:
                return {
                    "error": (
                        "Your selfie does not match your registered face. "
                        "Please retake with even lighting and your face centered."
                    ),
                    "match_diagnostics": match_diagnostics,
                }
            return {
                "error": "No confident match found in the dataset.",
                "match_diagnostics": match_diagnostics,
            }

        if expected_label and results and not scoped_label:
            top_label = " ".join(str(results[0]["label"]).split()).lower()
            target_label = " ".join(str(expected_label).split()).lower()
            if top_label != target_label:
                print(
                    f"❌ Identity Mismatch: Expected '{target_label}', "
                    f"but matched '{top_label}' ({results[0]['confidence'] * 100:.0f}%)"
                )
                return {
                    "error": (
                        f"Identity mismatch: face matched as {results[0]['label']} "
                        f"({results[0]['confidence'] * 100:.0f}%), but you are logged in as "
                        f"{expected_label}. Only your registered account may verify."
                    ),
                    "match_diagnostics": match_diagnostics,
                }

        if liveness_session_id:
            await update_liveness_session_status(
                session_id=liveness_session_id,
                status="consumed",
                raw_updates={"consumed_at": datetime.utcnow()},
            )
        # Encode processed image to base64 for frontend display
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
        processed_b64 = base64.b64encode(buffer).decode("utf-8")

        # Encode the ORIGINAL captured selfie (not background-removed) for comparison
        _, raw_buffer = cv2.imencode(".jpg", img_raw)
        captured_b64 = base64.b64encode(raw_buffer).decode("utf-8")

        log_security_verdict(
            phase="match",
            verdict=security_verdict,
            session_id=liveness_session_id,
            device_id=device_id,
            composite_risk=composite_risk,
            reason="match_ok",
            retry_allowed=False,
            risk_factors=risk_factors if isinstance(risk_factors, dict) else None,
        )

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
            "match_diagnostics": match_diagnostics,
            "security_verdict": security_verdict,
            "composite_risk": composite_risk,
            "risk_factors": risk_factors,
            "challenge_frames_verified": len(
                getattr(mem_sess, "challenge_snapshots", []) or []
            )
            if mem_sess
            else 0,
            "retry_allowed": False,
        }

    except Exception as e:
        log_event("match_unhandled", level="error", session_id=liveness_session_id, extra={"reason": str(e)[:200]})
        return public_dict(user_error("SERVER_ERROR", retry_allowed=True, http_status=500))

    finally:
        if liveness_session_id:
            session_manager.remove(liveness_session_id)
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def register_user(
    file: Optional[UploadFile] = File(None),
    email: str = Form(...),
    device_id: str = Form(...),
    docType: str = Form(default="Selfie"),
    document: Optional[UploadFile] = File(default=None),
    liveness_session_id: Optional[str] = Form(default=None),
):
    """Register a new user (email stored in faces.label). Selfie + liveness required."""
    
    primary_file = file or document
    if not primary_file:
        return {"error": "Please complete liveness and capture a live selfie to register."}

    email_clean = email.strip().lower()
    if "@" not in email_clean or len(email_clean) < 5:
        return {"error": "Please provide a valid email address."}
    if await email_exists(email_clean):
        return {"error": "This email is already registered. Please use a different email."}
    existing_face = await fetchrow(
        "SELECT 1 FROM faces WHERE LOWER(label) = LOWER($1) LIMIT 1",
        email_clean,
    )
    if existing_face:
        return {"error": "This email is already registered. Please use a different email."}

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
        if file and REGISTER_REQUIRE_LIVENESS:
            if not liveness_session_id:
                return {"error": "Security check required for selfie registration. Complete the liveness flow first."}
            mem_sess = session_manager.get(liveness_session_id)
            if not mem_sess or mem_sess.step not in ("verified", "complete", "capture"):
                return {"error": "Security check required for selfie registration. Complete the liveness flow first."}

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

        # 4. UPLOAD TO CLOUDINARY — faces.label stores email; folder name must be path-safe
        clean_label = email_clean
        clean_name = email_clean.replace("@", "_at_").replace(".", "_")
        
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

        # 5. SAVE TO POSTGRESQL — label column = email
        await insert_face(
            label=clean_label,
            source="frontend_reg",
            image_url=image_url,
            embedding=emb.tolist(),
        )

        await insert_app_user(
            email=email_clean,
            full_name=clean_label,
            face_label=clean_label,
            image_url=image_url,
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
                        "email": email_clean,
                        "doc_type": str(docType or "Selfie"),
                        "document_url": document_url,
                    },
                },
            )

        return {
            "success": True,
            "message": f"Successfully registered {email_clean}!",
            "image_url": image_url,
            "document_url": document_url,
            "email": email_clean,
            "face_label": clean_label,
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


@app.post("/register")
async def register_user_route(
    file: Optional[UploadFile] = File(None),
    email: str = Form(...),
    device_id: str = Form(...),
    docType: str = Form(default="Selfie"),
    document: Optional[UploadFile] = File(default=None),
    liveness_session_id: Optional[str] = Form(default=None),
):
    return await register_user(
        file=file,
        email=email,
        device_id=device_id,
        docType=docType,
        document=document,
        liveness_session_id=liveness_session_id,
    )


@app.post("/auth/register")
async def auth_register_route(
    file: Optional[UploadFile] = File(None),
    email: str = Form(...),
    device_id: str = Form(...),
    docType: str = Form(default="Selfie"),
    document: Optional[UploadFile] = File(default=None),
    liveness_session_id: Optional[str] = Form(default=None),
):
    return await register_user(
        file=file,
        email=email,
        device_id=device_id,
        docType=docType,
        document=document,
        liveness_session_id=liveness_session_id,
    )


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
