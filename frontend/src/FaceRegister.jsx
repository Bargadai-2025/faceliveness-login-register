import React, { useState, useRef, useEffect, useCallback } from "react";
import { getApiBase } from "./apiBase";
import { getCoverSourceRect } from "./cameraDrawUtils";
import {
  MATCH_REQUEST_TIMEOUT_MS,
  REGISTER_REQUEST_TIMEOUT_MS,
  LIVENESS_FRAME_INTERVAL_MS,
  createLivenessFrameLimiter,
  isReplayDeviceAlert,
  matchFetchErrorMessage,
  registerFetchErrorMessage,
  startIndeterminateMatchProgress,
} from "./matchUiUtils";
import { ERROR_LABELS } from "./securityErrorMessages";
import {
  isRegistrationSuccess,
  parseJsonResponse,
  parseRegisterFailureMessage,
} from "./apiUtils";
import "./FaceMatch.css";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import bargadLogo from "./bargad-logo.png";
import bargadBranding from "./bargad-branding (1).svg?url";
import { MapContainer, TileLayer, Marker } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import {
  Target, Layers, Sun, Activity,
  ArrowLeft, ArrowRight, ArrowUp, ArrowDown,
  Smile, Eye, Maximize, UserCheck, AlertOctagon, Info,
  Camera, MapPin, AlertTriangle
} from "lucide-react";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const API_URL = getApiBase();
const DEVICE_KEY = "facematch_device_id";
const FRAME_INTERVAL_MS = LIVENESS_FRAME_INTERVAL_MS;
const PROCESS_W = 640;
const PROCESS_H = 480;
const SHOW_FACE_MESH_OVERLAY = false;

const CHALLENGE_UI = {
  turn_left: { label: "Turn your Head Left", icon: ArrowLeft },
  turn_right: { label: "Turn your Head Right", icon: ArrowRight },
  smile: { label: "Smile", icon: Smile },
  mouth_open: { label: "Open your mouth", icon: Smile },
  move_closer: { label: "Move Closer", icon: Maximize },
  move_farther: { label: "Move Away", icon: Maximize },
  shake_head: { label: "Shake head left & right (NO)", icon: Activity },
  look_up_hold: { label: "Look Up", icon: ArrowUp },
  look_down_hold: { label: "Look Down", icon: ArrowDown },
  // head_forward: { label: "Move head FORWARD", icon: ArrowUp },
  // head_backward: { label: "Move head BACKWARD", icon: ArrowDown },
  // eye_left_right: { label: "Move eyes L to R", icon: Eye },
  // smile_then_blink: { label: "SMILE then BLINK", icon: Smile },
  // blink_then_turn_left: { label: "BLINK then turn LEFT", icon: Eye },
  // raise_eyebrows_hold: { label: "Raise brows & HOLD", icon: Activity },
};

const FACE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20], [20, 21],
  [22, 23], [23, 24], [24, 25], [25, 26],
  [27, 28], [28, 29], [29, 30],
  [31, 32], [32, 33], [33, 34], [34, 35], [31, 35],
  [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
  [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
  [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],
  [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60],
];

let sessionDeviceId = null;
function getOrCreateDeviceId() {
  if (!sessionDeviceId) {
    try {
      const uuid = (typeof crypto !== 'undefined' && crypto.randomUUID)
        ? crypto.randomUUID().slice(0, 8)
        : Math.random().toString(36).substring(2, 10);
      sessionDeviceId = `session_${uuid}`;
    } catch (e) {
      sessionDeviceId = `session_${Date.now()}`;
    }
  }
  return sessionDeviceId;
}

export default function FaceRegister({ userEmail, userAgentLabel, onLogout, onRegistered }) {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);

  const [challengeIndex, setChallengeIndex] = useState(0);
  const [completedChallenges, setCompletedChallenges] = useState([]);
  const [livenessLive, setLivenessLive] = useState(false);
  const [challengeMsg, setChallengeMsg] = useState("");
  const [sessionChallenges, setSessionChallenges] = useState([]);
  const [livenessSessionLoading, setLivenessSessionLoading] = useState(false);
  const [canMatch, setCanMatch] = useState(false);
  const [livenessStep, setLivenessStep] = useState("idle");
  const [lightOverlay, setLightOverlay] = useState(null);

  // Premium UI features
  const [progress, setProgress] = useState(0);
  const [geoData, setGeoData] = useState(null);
  const [geoError, setGeoError] = useState(null);
  const [geoAddress, setGeoAddress] = useState(null);
  const [processedPreview, setProcessedPreview] = useState(null);
  const [hoverCardIndex, setHoverCardIndex] = useState(null);
  const [selectedImg, setSelectedImg] = useState(null);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [rejectionError, setRejectionError] = useState(null);
  const [multiPersonError, setMultiPersonError] = useState(false);
  const [toastStep, setToastStep] = useState(null);
  const [toastVisible, setToastVisible] = useState(false);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [registerMode, setRegisterMode] = useState(false);
  const [firstName, setFirstName] = useState("");
  const [email, setEmail] = useState("");
  const [middleName, setMiddleName] = useState("");
  const [lastName, setLastName] = useState("");
  const [docType, setDocType] = useState("Selfie");
  const [docFile, setDocFile] = useState(null);
  const [registrationSuccess, setRegistrationSuccess] = useState(null);

  const videoRef = useRef();
  const canvasRef = useRef();
  const overlayCanvasRef = useRef();
  const overlayLandmarksRef = useRef(null);
  const overlayMeshRef = useRef(null);
  const inputRef = useRef();
  const frameIntervalRef = useRef(null);
  const livenessSessionIdRef = useRef(null);
  const registrationSessionIdRef = useRef(null);
  const registerAbortRef = useRef(null);
  const livenessCompletedRef = useRef(false);
  const streamingRef = useRef(false);
  const frameLimiterRef = useRef(createLivenessFrameLimiter());
  const profileMenuRef = useRef(null);
  const [photoCaptured, setPhotoCaptured] = useState(false);

  // Click outside profile menu
  useEffect(() => {
    function handleClickOutside(event) {
      if (profileMenuRef.current && !profileMenuRef.current.contains(event.target)) {
        setProfileMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // ── Geo-location capture ──
  const captureGeo = useCallback(() => {
    return new Promise((resolve) => {
      if (!navigator.geolocation) return resolve(null);
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({
          lat: pos.coords.latitude.toFixed(7),
          long: pos.coords.longitude.toFixed(7),
          timestamp: new Date().toISOString(),
        }),
        () => resolve(null),
        { enableHighAccuracy: true, maximumAge: 0, timeout: 12000 }
      );
    });
  }, []);

  const reverseGeocode = useCallback(async (lat, long) => {
    try {
      const res = await fetch(
        `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${long}&localityLanguage=en`
      );
      const data = await res.json();
      const parts = [data.locality, data.principalSubdivision, data.countryName].filter(Boolean);
      return {
        city: data.locality || data.city || "",
        state: data.principalSubdivision || "",
        country: data.countryName || "",
        full: data.localityInfo?.administrative?.map((a) => a.name).filter(Boolean).join(", ") || parts.join(", "),
        short: parts.join(", "),
      };
    } catch { return null; }
  }, []);

  useEffect(() => {
    if (!showCamera || SHOW_FACE_MESH_OVERLAY) return undefined;
    const canvas = overlayCanvasRef.current;
    if (!canvas) return undefined;
    let rafId = 0;
    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        if (canvas.width !== PROCESS_W) canvas.width = PROCESS_W;
        if (canvas.height !== PROCESS_H) canvas.height = PROCESS_H;
        ctx.clearRect(0, 0, PROCESS_W, PROCESS_H);
      }
      rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, [showCamera]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) return;

    video.srcObject = stream;

    const handlePlay = () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = setInterval(streamFrameToBackend, FRAME_INTERVAL_MS);
    };

    video.onloadedmetadata = () => {
      video.play()
        .then(() => {
          console.log("Video playing successfully");
          handlePlay();
        })
        .catch(e => {
          console.error("Video play failed:", e);
        });
    };

    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      video.onloadedmetadata = null;
    };
  }, [stream]);

  async function handleBackendResponse(data) {
    if (!data) return;
    if (SHOW_FACE_MESH_OVERLAY) {
      if (Array.isArray(data.landmarks) && data.landmarks.length >= 68) {
        overlayLandmarksRef.current = data.landmarks.map((p) => ({ x: p.x, y: p.y }));
      } else if (data.mesh === null) {
        overlayLandmarksRef.current = null;
      }
      if (Array.isArray(data.mesh) && data.mesh.length > 0) {
        overlayMeshRef.current = data.mesh.map((p) => ({ x: p.x, y: p.y }));
      } else if (data.mesh === null) {
        overlayMeshRef.current = null;
      }
    }
    if (data.step && data.step !== livenessStep) {
      const prevStep = livenessStep;
      setLivenessStep(data.step);

      // If we moved forward, show toast
      if (data.step !== "idle" && data.step !== "camera") {
        setToastStep(data.step);
        setToastVisible(true);
        if (prevStep !== "idle" && prevStep !== "camera") {
          setCompletedSteps(prev => [...new Set([...prev, prevStep])]);
        }
      }
    }
    // Multi-person during liveness — show restart message and reset challenge pills.
    if (data.multi_person) {
      setMultiPersonError(true);
      const restartMsg =
        data.detail ||
        "Multiple people detected during liveness. Gestures are restarting from Challenge 1 — only one person may complete the challenges.";
      setRejectionError(restartMsg);
      setError(ERROR_LABELS.MULTI_PERSON);
      setChallengeMsg(restartMsg);
      if (data.gesture_reset && data.gesture_idx !== undefined) {
        setChallengeIndex(data.gesture_idx);
        setCompletedChallenges([]);
      }
      return;
    }

    if (data.identity_mismatch) {
      setMultiPersonError(false);
      const restartMsg =
        data.detail ||
        "Different person detected during liveness. Gestures are restarting from Challenge 1 — only the original user may complete the challenges.";
      setRejectionError(restartMsg);
      setError(ERROR_LABELS.USER_MISMATCH);
      setChallengeMsg(restartMsg);
      if (data.gesture_reset && data.gesture_idx !== undefined) {
        setChallengeIndex(data.gesture_idx);
        setCompletedChallenges([]);
      }
      return;
    }

    if (isReplayDeviceAlert(data)) {
      setRejectionError(data.detail || ERROR_LABELS.DIGITAL_MEDIA);
      setError(ERROR_LABELS.DIGITAL_MEDIA);
      setChallengeMsg(data.detail || ERROR_LABELS.DIGITAL_MEDIA);
      stopCamera();
      return;
    }

    if (data.face_out_of_frame) {
      setMultiPersonError(false);
      setRejectionError(null);
      setError(ERROR_LABELS.NO_FACE);
      setChallengeMsg(
        data.detail || "Keep your face fully inside the frame during the challenge",
      );
      return;
    }

    if (data.detail) setChallengeMsg(data.detail);

    if (data.step === "gesture" && data.gesture_idx !== undefined) {
      setChallengeIndex(data.gesture_idx);
      const completed = [];
      for (let i = 0; i < data.gesture_idx; i++) completed.push(true);
      setCompletedChallenges(completed);
    }

    if (data.status === "verified" || data.step === "complete") {
      if (!livenessCompletedRef.current) {
        livenessCompletedRef.current = true;
        await completeSession();
      }
    }

    if (data.status === "rejected" || data.status === "failed") {
      if (data.status === "rejected") {
        const detail = String(data.detail || "");
        const digital =
          data.display_attack === true ||
          /photograph|digital screen|phone|tablet|replay/i.test(detail);
        setRejectionError(
          digital ? ERROR_LABELS.DIGITAL_MEDIA : detail || ERROR_LABELS.SECURITY_ALERT,
        );
        stopCamera();
      } else {
        setError(data.detail || "Liveness check failed");
      }
    } else {
      // If we got a successful frame, check for critical "processing" warnings to show on overlay
      if (data.status === "processing") {
        const d = data.detail || "";
        if (d.includes("blocked") || d.includes("No face")) {
          setError(d);
        } else {
          setError(null);
        }
      }
    }
  }

  async function streamFrameToBackend() {
    if (
      streamingRef.current ||
      frameLimiterRef.current.shouldSkip() ||
      !videoRef.current ||
      videoRef.current.readyState < 2 ||
      !livenessSessionIdRef.current
    ) {
      return;
    }
    streamingRef.current = true;
    try {
      const video = videoRef.current;
      const c = document.createElement("canvas");
      c.width = PROCESS_W;
      c.height = PROCESS_H;
      const ctx2 = c.getContext("2d");
      if (!ctx2) {
        streamingRef.current = false;
        return;
      }
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const { sx, sy, sw, sh } = getCoverSourceRect(vw, vh, PROCESS_W, PROCESS_H);
      ctx2.drawImage(video, sx, sy, sw, sh, 0, 0, PROCESS_W, PROCESS_H);
      const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
      if (!blob) {
        streamingRef.current = false;
        return;
      }
      const fd = new FormData();
      fd.append("session_id", livenessSessionIdRef.current);
      fd.append("frame", blob, "frame.jpg");
      const res = await fetch(`${API_URL}/liveness/frame`, { method: "POST", body: fd });
      const data = await parseJsonResponse(res);
      if (!res.ok) {
        if (res.status === 429 || data.retry_allowed) {
          const pauseMs = frameLimiterRef.current.onRateLimited();
          console.warn(`Liveness frame rate limited — pausing ${pauseMs}ms`);
        }
        return;
      }
      frameLimiterRef.current.onSuccess();
      await handleBackendResponse(data);
    } catch (e) {
      console.warn("Stream error:", e);
    }
    streamingRef.current = false;
  }

  async function completeSession() {
    try {
      const res = await fetch(`${API_URL}/liveness/session/complete`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: livenessSessionIdRef.current }),
      });
      const data = await res.json();
      if (res.ok && data.ok) { setCanMatch(true); setLivenessLive(true); setLivenessStep("capture"); }
    } catch { setError("Verification failed"); }
  }

  function stopCameraStreamOnly() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    setStream(null);
    setShowCamera(false);
    overlayLandmarksRef.current = null;
    overlayMeshRef.current = null;
  }

  async function startCamera() {
    frameLimiterRef.current.reset();
    setError(null);
    setFile(null);
    setPreview(null);
    setPhotoCaptured(false);
    registrationSessionIdRef.current = null;
    setResults([]);
    setGeoData(null);
    setGeoAddress(null);
    setProgress(0);
    setLivenessSessionLoading(true);
    try {
      let sessData;
      try {
        const registerLabel = email.trim().toLowerCase() || userAgentLabel;
        const sessRes = await fetch(`${API_URL}/liveness/session/start`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            device_id: getOrCreateDeviceId(),
            agent_label: registerLabel,
          }),
        });
        if (!sessRes.ok) {
          const errBody = await sessRes.text();
          throw new Error(`Server error ${sessRes.status}: ${errBody}`);
        }
        sessData = await sessRes.json();
      } catch (e) {
        setError(`Session failed: ${e.message}`);
        setLivenessSessionLoading(false);
        return;
      }

      let mediaStream;
      try {
        const constraints = {
          video: {
            facingMode: { ideal: "user" },
            width: { ideal: 640 },
            height: { ideal: 480 }
          }
        };
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (e) {
        setError(`Camera access denied: ${e.message}`);
        setLivenessSessionLoading(false);
        return;
      }

      livenessSessionIdRef.current = sessData.session_id;
      setSessionChallenges(sessData.gestures);
      setStream(mediaStream);
      setShowCamera(true);
      setLivenessStep("camera");
    } catch (e) { setError(`Unexpected error: ${e.message}`); }
    finally { setLivenessSessionLoading(false); }
  }

  function stopCamera() {
    stopCameraStreamOnly();
    frameLimiterRef.current.reset();
    setLivenessLive(false);
    setPhotoCaptured(false);
    livenessSessionIdRef.current = null;
    registrationSessionIdRef.current = null;
    livenessCompletedRef.current = false;
    setChallengeMsg("");
    setLivenessStep("idle");
    setError(null);
    setCanMatch(false);
  }

  const takeSelfie = () => {
    console.log("📸 Capture button clicked");
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) {
      console.warn("Canvas or video ref missing");
      return;
    }

    // 1. Capture the frame first
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) {
        console.error("Failed to create blob from canvas");
        setError("Capture failed: Could not process image.");
        return;
      }
      const f = new File([blob], "selfie.jpg", { type: "image/jpeg" });
      registrationSessionIdRef.current = livenessSessionIdRef.current;

      setFile(f);
      setPreview(URL.createObjectURL(f));
      setPhotoCaptured(true);
      stopCameraStreamOnly();
    }, "image/jpeg", 0.95);
  };

  const handleMatch = async (fileOverride = null, sessionIdOverride = null) => {
    const fileToUse = fileOverride || file;
    const sessionIdToUse = sessionIdOverride || livenessSessionIdRef.current;

    console.log("🔍 Starting Match process", {
      hasFile: !!fileToUse,
      hasSession: !!sessionIdToUse,
      canMatch
    });

    // If we are overriding with a direct file from capture, we bypass the state-based canMatch 
    // because liveness is already verified to reach the capture button.
    const isDirectCapture = !!fileOverride;

    if (!fileToUse) {
      setError("Please capture a photo first");
      return;
    }

    if (!canMatch) {
      setError("Complete liveness flow first");
      console.warn("Match blocked: canMatch is false");
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);
    setProgress(0);
    setGeoError(null);

    // Geo capture
    console.log("📍 Capturing Geo...");
    const geo = await captureGeo();
    setGeoData(geo);
    if (!geo) {
      console.warn("Geo capture failed or denied");
      setGeoError("⚠ Location unavailable.");
    } else {
      reverseGeocode(geo.lat, geo.long).then(addr => {
        console.log("🌍 Geo Address:", addr);
        setGeoAddress(addr);
      });
    }

    const matchAbort = new AbortController();
    const matchTimeoutId = setTimeout(() => matchAbort.abort(), MATCH_REQUEST_TIMEOUT_MS);
    const pInterval = startIndeterminateMatchProgress(setProgress);

    const fd = new FormData();
    fd.append("file", fileToUse);
    fd.append("device_id", getOrCreateDeviceId());
    if (sessionIdToUse) {
      fd.append("liveness_session_id", sessionIdToUse);
    }
    if (geo) {
      fd.append("geo_lat", geo.lat);
      fd.append("geo_long", geo.long);
      fd.append("geo_timestamp", geo.timestamp);
    }

    try {
      console.log(`📤 Sending match request to ${API_URL}/match ...`);
      const res = await fetch(`${API_URL}/match`, { method: "POST", body: fd, signal: matchAbort.signal });
      const data = await res.json();
      clearInterval(pInterval);
      clearTimeout(matchTimeoutId);

      if (data.error) {
        console.error("❌ Match error from backend:", data.error);
        setError(data.error);
        setProgress(0);
      } else {
        console.log("✅ Match successful", data.matches?.length, "results");
        setResults(data.matches || []);
        if (data.processed_image) setProcessedPreview(data.processed_image);
        setProgress(100);
      }
    } catch (err) {
      console.error("❌ Match request failed:", err);
      setError(matchFetchErrorMessage(err));
      setProgress(0);
      clearInterval(pInterval);
      clearTimeout(matchTimeoutId);
    } finally {
      setLoading(false);
    }
  };
  const handleRegister = async () => {
    if (loading) return;

    if (!file) {
      setError("Please complete liveness and capture your live photo first.");
      return;
    }
    if (!email.trim() || !email.includes("@")) {
      setError("Please enter a valid email address.");
      return;
    }

    registerAbortRef.current?.abort();
    const abort = new AbortController();
    registerAbortRef.current = abort;
    const timeoutId = setTimeout(() => abort.abort(), REGISTER_REQUEST_TIMEOUT_MS);

    setLoading(true);
    setError(null);
    setRegistrationSuccess(null);

    const emailNorm = email.trim().toLowerCase();
    const fd = new FormData();
    fd.append("file", file);
    fd.append("email", emailNorm);
    fd.append("docType", "Selfie");
    const sessionId =
      registrationSessionIdRef.current || livenessSessionIdRef.current;
    if (sessionId) {
      fd.append("liveness_session_id", sessionId);
    }
    fd.append("device_id", getOrCreateDeviceId());

    const registerUrl = `${API_URL}/register`;

    try {
      console.log(`📤 Sending registration request to ${registerUrl} ...`);
      const res = await fetch(registerUrl, {
        method: "POST",
        body: fd,
        signal: abort.signal,
      });
      clearTimeout(timeoutId);

      const data = await parseJsonResponse(res);

      if (isRegistrationSuccess(data, res)) {
        const faceLabel = data.face_label || emailNorm;
        toast.success(data.message || `Successfully registered ${emailNorm}!`);
        setRegistrationSuccess(`Successfully registered ${emailNorm}!`);
        setRegisterMode(false);
        setFirstName("");
        setMiddleName("");
        setLastName("");
        setEmail("");
        setFile(null);
        setPreview(null);
        setPhotoCaptured(false);
        registrationSessionIdRef.current = null;
        setCanMatch(false);
        setLivenessLive(false);
        setLivenessStep("idle");
        setLoading(false);
        try {
          onRegistered?.(emailNorm, faceLabel);
        } catch (cbErr) {
          console.warn("onRegistered callback:", cbErr);
        }
        window.setTimeout(() => {
          window.location.href = "/";
        }, 400);
        return;
      }

      const msg = parseRegisterFailureMessage(res, data);
      setError(msg);
      toast.error(msg);
    } catch (err) {
      clearTimeout(timeoutId);
      const msg = registerFetchErrorMessage(err);
      console.error("❌ Registration request failed:", err);
      setError(msg);
      toast.error(msg);
    } finally {
      if (registerAbortRef.current === abort) {
        registerAbortRef.current = null;
      }
      setLoading(false);
    }
  }; 


  const handleStartRegistration = () => {
    setError(null);
    // if (!firstName.trim() || !lastName.trim()) {
    //   setError("Please enter your first and last name.");
    //   return;
    // }
    if (!email.trim() || !email.includes("@")) {
      setError("Please enter a valid email address.");
      return;
    }
    startCamera();
  };

  const handleRetakePhoto = () => {
    setFile(null);
    setPreview(null);
    setPhotoCaptured(false);
    registrationSessionIdRef.current = null;
    livenessCompletedRef.current = false;
    setCanMatch(false);
    startCamera();
  };

  return (
    <div className="fm-page">
      <header className="fm-header-banner">
        <div className="fm-header-left">
          <img src={bargadLogo} alt="Bargad" className="fm-header-logo" />
          <nav className="fm-header-nav">
            <a href="/" className="fm-nav-link" onClick={(e) => { e.preventDefault(); window.history.pushState({}, '', '/'); window.dispatchEvent(new PopStateEvent('popstate')); }}>Match</a>
            <a href="/register" className="fm-nav-link active" onClick={(e) => { e.preventDefault(); window.history.pushState({}, '', '/register'); window.dispatchEvent(new PopStateEvent('popstate')); }}>Register</a>
          </nav>
        </div>
        <div className="fm-header-profile">
          <div className="fm-profile-avatar-wrap">
            <svg className="fm-profile-avatar-svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v1c0 .55.45 1 1 1h14c.55 0 1-.45 1-1v-1c0-2.66-5.33-4-8-4z" /></svg>
          </div>
        </div>
      </header>

      <div className="fm-container">
        <div className="fm-header">
          <div className="fm-logo">⬡</div>
          <div>
            <h1>Face Registration</h1>
            <p>Enter your name and complete a quick liveness check to register.</p>
          </div>
        </div>

        {/* STEP TOASTS (TOP OF LIVENESS - FLOATING) */}
        {toastVisible && toastStep && (
          <div className={`fm-step-toast-floating ${completedSteps.includes(toastStep) ? "fm-toast-success" : "fm-toast-pending"}`}>
            <div className="fm-toast-inner">
              <div className="fm-toast-icon-wrap">
                {completedSteps.includes(toastStep) ? <CheckCircle size={18} /> : <div className="fm-toast-spinner" />}
              </div>
              <div className="fm-toast-info">
                <div className="fm-toast-title">
                  {toastStep === "calibration" && "Face Calibration"}
                  {toastStep === "depth" && "Depth Analysis"}
                  {toastStep === "light_challenge" && "Light Challenge"}
                  {toastStep === "micro" && "Micro Expression"}
                  {toastStep === "verified" && "Verification Complete"}
                </div>
                <div className="fm-toast-status">
                  {completedSteps.includes(toastStep) ? "Step Completed Successfully" : "Processing Security Layer..."}
                </div>
              </div>
            </div>
            <div className="fm-toast-progress-bar">
              <div className={`fm-toast-progress-fill ${completedSteps.includes(toastStep) ? "full" : "animate"}`} />
            </div>
          </div>
        )}

        <div className="fm-registration-simple-wrap">
          {!showCamera && !photoCaptured && (
            <>
              {/* <div className="fm-reg-input-group">
                <label>First name</label>
                <input
                  type="text"
                  placeholder="First name"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  className="fm-reg-input"
                  disabled={loading || livenessSessionLoading}
                  style={{ width: "100%", textAlign: "left" }}
                />
              </div> */}
              {/* <div className="fm-reg-input-group">
                <label>Last name</label>
                <input
                  type="text"
                  placeholder="Last name"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  className="fm-reg-input"
                  disabled={loading || livenessSessionLoading}
                  style={{ width: "100%", textAlign: "left" }}
                />
              </div> */}
              <div className="fm-reg-input-group">
                <label>Email</label>
                <input
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="fm-reg-input"
                  disabled={loading || livenessSessionLoading}
                  style={{ width: "100%", textAlign: "left" }}
                />
              </div>

              {error && !showCamera && (
                <div className="lp-error" style={{ marginTop: 12 }}>
                  {error}
                </div>
              )}

              <button
                className="fm-btn"
                onClick={handleStartRegistration}
                disabled={
                  // !firstName.trim() ||
                  // !lastName.trim() ||
                  !email.trim() ||
                  loading ||
                  livenessSessionLoading
                }
                style={{ width: "100%", marginTop: "10px" }}
              >
                {livenessSessionLoading
                  ? "Starting camera..."
                  : "Start liveness & capture"}
              </button>
            </>
          )}

          {showCamera && (
            <div className="fm-camera-section-simple">
              <div className="fm-camera-outer">
                <div className="fm-camera-container">
                  <video ref={videoRef} autoPlay playsInline muted className="fm-camera-feed" />
                  <canvas ref={overlayCanvasRef} className="fm-mesh-overlay" />
                  <canvas ref={canvasRef} style={{ display: "none" }} />

                  <div className="fm-viewfinder-corner top-left"></div>
                  <div className="fm-viewfinder-corner top-right"></div>
                  <div className="fm-viewfinder-corner bottom-left"></div>
                  <div className="fm-viewfinder-corner bottom-right"></div>
                  <div className="fm-scanline"></div>

                  {error && <div className="fm-camera-error-overlay">{error}</div>}

                  {/* {challengeMsg && !livenessLive && (
                    <div className="fm-challenge-overlay-simple">
                      {livenessStep === "gesture" && sessionChallenges[challengeIndex] ? (
                        <div className="fm-gesture-instruction">
                          {React.createElement(CHALLENGE_UI[sessionChallenges[challengeIndex]]?.icon || Activity, { size: 24, className: "fm-gesture-icon" })}
                          <span>{CHALLENGE_UI[sessionChallenges[challengeIndex]]?.label || challengeMsg}</span>
                        </div>
                      ) : (
                        <span>{challengeMsg}</span>
                      )}
                    </div>
                  )} */}

                  <div className="fm-camera-actions">
                    {/* {livenessLive && !multiPersonError && !photoCaptured && ( */}
                      <button className="fm-capture-btn" onClick={takeSelfie}>
                        <Camera size={18} /> Take Registration Photo
                      </button>
                    {/* )} */}
                  </div>
                </div>
              </div>
              <button className="fm-camera-btn secondary" onClick={stopCamera} style={{ marginTop: "10px" }}>
                ✕ Cancel
              </button>
            </div>
          )}

          {photoCaptured && file && (
            <div className="fm-registration-final-step">
              <div className="fm-preview-circle">
                <img src={preview} alt="Profile Preview" />
              </div>
              <p className="fm-reg-confirm-text">
                Liveness verified — <strong>{email}</strong>
              </p>
              {error && (
                <div className="lp-error" style={{ marginBottom: 12 }}>
                  {error}
                </div>
              )}
              <div className="fm-reg-actions-row">
                <button className="fm-btn" onClick={handleRegister} disabled={loading}>
                  {loading ? "Registering..." : "Complete Registration"}
                </button>
                <button
                  className="fm-btn secondary"
                  onClick={handleRetakePhoto}
                  disabled={loading}
                >
                  Retake Photo
                </button>
              </div>
            </div>
          )}
        </div>

        {registrationSuccess && (
          <div className="fm-status-alert success" style={{ marginTop: "20px" }}>
            <UserCheck size={20} />
            <span>{registrationSuccess}</span>
          </div>
        )}

        {rejectionError && (
          <div className="fm-modal fm-rejection-modal" onClick={() => setRejectionError(null)}>
            <div className="fm-modal-box fm-rejection-box">
              <AlertOctagon size={48} color="#ff0000" />
              <h2>Verification Failed</h2>
              <p>{rejectionError}</p>
              <button className="fm-modal-close" onClick={() => setRejectionError(null)}>✕ Close</button>
            </div>
          </div>
        )}
      </div>

      <div className="fm-footer-branding"><img src={bargadBranding} alt="Bargad" /></div>
      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
      />
    </div>
  );
}
