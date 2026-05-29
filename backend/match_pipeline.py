"""
Sync CPU/GPU pipeline for POST /match — runs inside inference_runtime thread pool.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from api_errors import public_dict, user_error
from challenge_frame_verification import assess_challenge_continuity
from embedding_cache import embedding_cache
from embedding_pipeline import FACE_DETECT_MAX_SIDE, extract_face_embedding
from face_detection import remove_background
from frame_processor import _get_yolo
from liveness_session import session_manager
from poc_helpers import enrich_match_security_payload
from poc_logging import log_event, log_security_verdict
from post_selfie_security import load_post_selfie_config, run_post_selfie_security, scan_yolo_devices


@dataclass
class MatchRuntimeConfig:
    device: str
    fast_match: bool
    min_confidence_bargad: float
    min_confidence_lfw: float
    min_top2_margin: float
    top_k: int = 50


@dataclass
class MatchPipelineInput:
    temp_path: str
    liveness_session_id: Optional[str]
    device_id: Optional[str]
    errcount: int
    expected_label: Optional[str]
    liveness_ref_photo: Optional[str]
    geo_lat: Optional[str]
    geo_long: Optional[str]
    geo_timestamp: Optional[str]
    mtcnn: Any
    model: Any
    runtime: MatchRuntimeConfig


@dataclass
class MatchPipelineOutput:
    response: Dict[str, Any]
    consume_session: bool = False
    auth_log: Optional[Dict[str, Any]] = None


def _finalize_security_reject(
    sec: Dict[str, Any],
    *,
    identity_assessment: Dict[str, Any],
    liveness_session_id: Optional[str],
    device_id: Optional[str],
) -> Dict[str, Any]:
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


def run_full_match_pipeline(inp: MatchPipelineInput) -> MatchPipelineOutput:
    """Heavy /match work — must run under inference_runtime.run_inference_limited."""
    cfg = inp.runtime
    penalties_breakdown: List[Dict[str, Any]] = []
    errcount = int(inp.errcount or 0)
    liveness_session_id = inp.liveness_session_id
    device_id = inp.device_id

    img_raw = cv2.imread(inp.temp_path)
    if img_raw is None:
        return MatchPipelineOutput(response=public_dict(user_error("IMAGE_READ_FAILED", retry_allowed=True)))

    try:
        face_out = extract_face_embedding(img_raw, inp.mtcnn, inp.model, cfg.device)
    except Exception as e:
        log_event(
            "face_detection_crash",
            level="error",
            session_id=liveness_session_id,
            extra={"reason": str(e)[:200]},
        )
        return MatchPipelineOutput(response=public_dict(user_error("SERVER_ERROR", retry_allowed=True, http_status=500)))

    if not face_out["ok"]:
        return MatchPipelineOutput(response=public_dict(user_error("FACE_NOT_FOUND", retry_allowed=True)))

    emb = face_out["embedding"]
    face_landmarks_mp = face_out["face_landmarks_mp"]

    mem_sess_verify = session_manager.get(liveness_session_id) if liveness_session_id else None
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
            mtcnn=inp.mtcnn,
            model=inp.model,
            device=cfg.device,
            selfie_devices=selfie_devices,
        )

    if inp.liveness_ref_photo:
        try:
            if "," in inp.liveness_ref_photo:
                _, base64_data = inp.liveness_ref_photo.split(",", 1)
            else:
                base64_data = inp.liveness_ref_photo
            img_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_data, np.uint8)
            ref_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if ref_img is None:
                return MatchPipelineOutput(response={"error": "Failed to decode the liveness reference photo."})

            ref_face_out = extract_face_embedding(ref_img, inp.mtcnn, inp.model, cfg.device)
            if not ref_face_out["ok"]:
                return MatchPipelineOutput(
                    response={
                        "error": (
                            "No face detected in the liveness reference photo. "
                            "Please keep your face clearly visible and still during the preparation phase."
                        )
                    }
                )
            ref_emb = ref_face_out["embedding"]
            similarity_score = float(np.dot(ref_emb, emb))
            print(
                f"🔄 FaceNet comparison score between liveness photo and capture image: "
                f"{similarity_score:.4f}"
            )
            threshold = float(os.getenv("LIVENESS_COMPARE_THRESHOLD", "0.70"))
            if similarity_score < threshold:
                return MatchPipelineOutput(
                    response={
                        "error": "Liveness process user and captured image user are totally different."
                    }
                )
        except Exception as e:
            print(f"Error during liveness reference comparison: {e}")
            return MatchPipelineOutput(
                response={"error": f"Liveness reference verification error: {str(e)}"}
            )

    liveness_verified = bool(
        mem_sess_verify
        and getattr(mem_sess_verify, "all_gestures_done", False)
        and getattr(mem_sess_verify, "step", "") in ("complete", "capture", "verified")
    )
    if liveness_verified and mem_sess_verify:
        identity_assessment = dict(identity_assessment)
        identity_assessment["liveness_session_verified"] = True

    sec = run_post_selfie_security(
        img_raw,
        face_landmarks_mp,
        errcount=errcount,
        penalties_breakdown=penalties_breakdown,
        identity_assessment=identity_assessment,
        stream_risk_ema=stream_risk_ema,
        liveness_session_verified=liveness_verified,
    )
    if sec.get("error"):
        return MatchPipelineOutput(
            response=_finalize_security_reject(
                sec,
                identity_assessment=identity_assessment,
                liveness_session_id=liveness_session_id,
                device_id=device_id,
            )
        )

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

    if cfg.fast_match:
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
            img_processed = cv2.resize(
                img_processed,
                (int(img_processed.shape[1] * scale), int(img_processed.shape[0] * scale)),
            )

    scoped_label = None
    if inp.expected_label and str(inp.expected_label).strip():
        scoped_label = " ".join(str(inp.expected_label).split()).strip()

    if scoped_label:
        raw_results, _, note = embedding_cache.search_scores(
            emb, scoped_label=scoped_label, top_k=cfg.top_k
        )
        if not raw_results and embedding_cache.loaded:
            print(f"🔐 Scoped cache miss for '{scoped_label}' ({note})")
        print(
            f"🔐 Scoped 1:1 match for logged-in user '{scoped_label}': "
            f"{len(raw_results)} candidate(s) [{note}]"
        )
        if not raw_results:
            return MatchPipelineOutput(
                response={
                    "error": (
                        f"No registered face found for {scoped_label}. "
                        "Please complete registration first."
                    ),
                }
            )
    else:
        raw_results, _, note = embedding_cache.search_scores(emb, top_k=cfg.top_k)
        print(f"📦 Embedding search: {len(raw_results)} top hits [{note}]")

    print("\n🔍 Top 5 raw matches:")
    for r in raw_results[:5]:
        print(f"  {r['label']} ({r['source']}) → {r['score']:.3f} [Type: {r.get('doc_type')}]")

    match_diagnostics = {
        "detect_max_side": FACE_DETECT_MAX_SIDE,
        "fast_match": cfg.fast_match,
        "scoped_match": bool(scoped_label),
        "scoped_label": scoped_label,
        "embedding_cache": embedding_cache.face_count if embedding_cache.loaded else 0,
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
    seen: Dict[str, Dict[str, Any]] = {}
    for r in raw_results:
        normalized_label = " ".join(str(r["label"]).split())
        key = normalized_label.lower()
        score = round(float(r["score"]), 3)
        print(f"  - Checking {key}: Score {score}, Type in raw: {r.get('doc_type')}")

        threshold = (
            cfg.min_confidence_bargad
            if r["source"] in ["bargad", "frontend_reg"]
            else cfg.min_confidence_lfw
        )
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
                "images": [r["image_url"]],
            }
        else:
            if r["image_url"] not in seen[key]["images"]:
                seen[key]["images"].append(r["image_url"])
            if score > seen[key]["confidence"]:
                seen[key]["confidence"] = score
                seen[key]["matched_image"] = r["image_url"]
            new_type = r.get("doc_type")
            if new_type and str(new_type).lower() != "selfie":
                seen[key]["registered_doc_type"] = new_type
            elif not seen[key].get("registered_doc_type"):
                seen[key]["registered_doc_type"] = "Selfie"

    results = list(seen.values())
    results.sort(key=lambda x: x["confidence"], reverse=True)

    if not scoped_label and len(raw_results) >= 2:
        top_raw = raw_results[0]
        second_raw = raw_results[1]
        top_key = " ".join(str(top_raw["label"]).split()).lower()
        second_key = " ".join(str(second_raw["label"]).split()).lower()
        margin = match_diagnostics["top2_margin"]
        if top_key != second_key and margin is not None and margin < cfg.min_top2_margin:
            skip_ambiguous = False
            if inp.expected_label:
                target_key = " ".join(str(inp.expected_label).split()).lower()
                top_threshold = (
                    cfg.min_confidence_bargad
                    if top_raw.get("source") in ("bargad", "frontend_reg")
                    else cfg.min_confidence_lfw
                )
                if top_key == target_key and float(top_raw["score"]) >= top_threshold:
                    skip_ambiguous = True
                    print(
                        f"✓ Ambiguous check skipped: logged-in user '{inp.expected_label}' "
                        f"is top match ({top_raw['score']:.3f})"
                    )
            if not skip_ambiguous:
                print(
                    f"⚠️ Ambiguous match: {top_raw['label']}={top_raw['score']:.3f} vs "
                    f"{second_raw['label']}={second_raw['score']:.3f} "
                    f"(margin {margin:.3f} < {cfg.min_top2_margin})"
                )
                return MatchPipelineOutput(
                    response={
                        "error": (
                            "Match ambiguous — multiple identities scored too closely. "
                            "Please retake the selfie with even lighting and face centered."
                        ),
                        "match_diagnostics": match_diagnostics,
                    }
                )

    if errcount > 0 or True:
        base_conf = results[0]["confidence"] if results else 0.0
        penalty = float(errcount) * 0.03
        penalties_breakdown.insert(
            0,
            {
                "type": "Base Face Similarity",
                "penalty": 0.0,
                "count": 1,
                "detail": f"Raw similarity score: {int(round(base_conf * 100))}%",
            },
        )
        total_penalty = penalty + risk_confidence_penalty
        if total_penalty > 0:
            print(
                f"⚠️ Applying security penalty: -{total_penalty:.3f} "
                f"(errcount={errcount}, risk_penalty={risk_confidence_penalty:.3f})"
            )
            if penalty > 0:
                penalties_breakdown.append(
                    {
                        "type": "Liveness Session Risk",
                        "penalty": round(penalty, 3),
                        "count": errcount // 10,
                        "detail": "Accumulated liveness stream signals.",
                    }
                )
            for r in results:
                r["confidence"] = max(0.0, round(r["confidence"] - total_penalty, 3))
        results.sort(key=lambda x: x["confidence"], reverse=True)

    auth_log = None
    if inp.geo_lat and inp.geo_long:
        auth_log = {
            "timestamp": inp.geo_timestamp or datetime.utcnow(),
            "geo_lat": inp.geo_lat,
            "geo_long": inp.geo_long,
            "top_match": results[0]["label"] if results else "no_match",
            "match_count": len(results),
            "raw_data": {"geo_timestamp": inp.geo_timestamp} if inp.geo_timestamp else {},
        }
        print(f"📍 Geo logged: {inp.geo_lat}, {inp.geo_long}")

    if not results:
        if scoped_label:
            return MatchPipelineOutput(
                response={
                    "error": (
                        "Your selfie does not match your registered face. "
                        "Please retake with even lighting and your face centered."
                    ),
                    "match_diagnostics": match_diagnostics,
                }
            )
        return MatchPipelineOutput(
            response={
                "error": "No confident match found in the dataset.",
                "match_diagnostics": match_diagnostics,
            }
        )

    if inp.expected_label and results and not scoped_label:
        top_label = " ".join(str(results[0]["label"]).split()).lower()
        target_label = " ".join(str(inp.expected_label).split()).lower()
        if top_label != target_label:
            print(
                f"❌ Identity Mismatch: Expected '{target_label}', "
                f"but matched '{top_label}' ({results[0]['confidence'] * 100:.0f}%)"
            )
            return MatchPipelineOutput(
                response={
                    "error": (
                        f"Identity mismatch: face matched as {results[0]['label']} "
                        f"({results[0]['confidence'] * 100:.0f}%), but you are logged in as "
                        f"{inp.expected_label}. Only your registered account may verify."
                    ),
                    "match_diagnostics": match_diagnostics,
                }
            )

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
    processed_b64 = base64.b64encode(buffer).decode("utf-8")
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

    return MatchPipelineOutput(
        response={
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
            "challenge_frames_verified": len(getattr(mem_sess_verify, "challenge_snapshots", []) or [])
            if mem_sess_verify
            else 0,
            "retry_allowed": False,
        },
        consume_session=bool(liveness_session_id),
        auth_log=auth_log,
    )
