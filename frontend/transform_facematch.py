"""
Transform FaceMatch.jsx: remove frontend detection, add backend-driven streaming.
Run: python transform_facematch.py
"""
import re

NEW_TOP = r'''import React, { useState, useRef, useEffect, useCallback } from "react";
import "./FaceMatch.css";
import bargadLogo from "./bargad-logo.png";
import bargadBranding from "./bargad-branding (1).svg?url";
import { MapContainer, TileLayer, Marker } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MATCH_REQUEST_TIMEOUT_MS = 30_000;
const DEVICE_KEY = "facematch_device_id";
const FRAME_INTERVAL_MS = 120;

const CHALLENGE_TEXT = {
  turn_left: "\u21a9\ufe0f Turn your head LEFT",
  turn_right: "\u21aa\ufe0f Turn your head RIGHT",
  nod: "\u2195\ufe0f NOD your head down",
  look_up: "\u2b06\ufe0f LOOK slightly up",
  smile: "\ud83d\ude0a  SMILE",
  # surprised: "\ud83d\ude32 Look SURPRISED",
  mouth_open: "\ud83d\ude2e OPEN your mouth wide",
  # wide_eyes: "\ud83d\udc40 OPEN your eyes wide",
  # blink_both: "\ud83d\ude09 BLINK both eyes quickly",
  # raise_eyebrows: "\ud83e\udd28 Raise your EYEBROWS up",
  # pucker_lips: "\ud83d\ude17 PUCKER your lips forward",
  # frown: "\u2639\ufe0f FROWN (sad face)",
  move_closer: "\ud83d\udcf1 Move CLOSER to camera",
  move_farther: "\ud83d\udcf1 Move FARTHER from camera",
  shake_head: "\ud83d\ude45\u200d\u2642\ufe0f Shake head NO left and right",
  # blink_twice_fast: "\ud83d\ude09 BLINK twice quickly",
  look_left_hold: "\u2b05\ufe0f Look LEFT and HOLD 1.5s",
  look_right_hold: "\u27a1\ufe0f Look RIGHT and HOLD 1.5s",
  look_up_hold: "\u2b06\ufe0f Look UP and HOLD 1.5s",
  look_down_hold: "\u2b07\ufe0f Look DOWN and HOLD 1.5s",
  head_forward: "\ud83d\udcf1 Move head FORWARD",
  head_backward: "\ud83d\udcf1 Move head BACKWARD",
  # eye_left_right: "\ud83d\udc40 Move eyes LEFT then RIGHT",
  # smile_then_blink: "\ud83d\ude0a SMILE then BLINK",
  # blink_then_turn_left: "\ud83d\ude09 BLINK then turn LEFT",
  # raise_eyebrows_hold: "\ud83e\udd28 Raise both eyebrows and HOLD 1.5s",
};

function getOrCreateDeviceId() {
  try {
    let id = localStorage.getItem(DEVICE_KEY);
    if (!id) { id = crypto.randomUUID(); localStorage.setItem(DEVICE_KEY, id); }
    return id;
  } catch { return `anon_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`; }
}

function UserAvatarIcon() {
  return (
    <svg className="fm-profile-avatar-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v1c0 .55.45 1 1 1h14c.55 0 1-.45 1-1v-1c0-2.66-5.33-4-8-4z" fill="currentColor"/>
    </svg>
  );
}

export default function FaceMatch({ userEmail, onLogout }) {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [hoverCardIndex, setHoverCardIndex] = useState(null);
  const [selectedImg, setSelectedImg] = useState(null);
  const [progress, setProgress] = useState(0);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);

  // Liveness states (backend-driven)
  const [challengeIndex, setChallengeIndex] = useState(0);
  const [completedChallenges, setCompletedChallenges] = useState([]);
  const [livenessLive, setLivenessLive] = useState(false);
  const [challengeMsg, setChallengeMsg] = useState("");
  const [geoData, setGeoData] = useState(null);
  const [geoError, setGeoError] = useState(null);
  const [geoAddress, setGeoAddress] = useState(null);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef(null);
  const [sessionChallenges, setSessionChallenges] = useState([]);
  const [livenessSessionLoading, setLivenessSessionLoading] = useState(false);
  const [canMatch, setCanMatch] = useState(false);
  const [livenessStep, setLivenessStep] = useState("idle");
  const [lightOverlay, setLightOverlay] = useState(null);
  const [backendLandmarks, setBackendLandmarks] = useState(null);

  // Refs
  const videoRef = useRef();
  const canvasRef = useRef();
  const overlayCanvasRef = useRef();
  const inputRef = useRef();
  const frameIntervalRef = useRef(null);
  const livenessSessionIdRef = useRef(null);
  const livenessCompletedRef = useRef(false);
  const streamingRef = useRef(false);

  // Close profile menu on outside click
  useEffect(() => {
    if (!profileMenuOpen) return;
    const handler = (e) => {
      if (profileMenuRef.current && !profileMenuRef.current.contains(e.target)) setProfileMenuOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [profileMenuOpen]);

  // Draw mesh from backend landmarks
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !backendLandmarks) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const pts = backendLandmarks;
    if (!pts || pts.length < 68) return;
    const connections = [
      [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
      [17,18],[18,19],[19,20],[20,21],[22,23],[23,24],[24,25],[25,26],
      [27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],
      [36,37],[37,38],[38,39],[39,40],[40,41],[41,36],
      [42,43],[43,44],[44,45],[45,46],[46,47],[47,42],
      [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48],
      [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60],
    ];
    ctx.strokeStyle = "rgba(0, 255, 170, 0.65)";
    ctx.lineWidth = 1.5;
    connections.forEach(([a, b]) => {
      if (pts[a] && pts[b]) {
        ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); ctx.stroke();
      }
    });
    pts.forEach((pt) => {
      if (pt) { ctx.beginPath(); ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI); ctx.fillStyle = "rgba(0, 255, 170, 0.9)"; ctx.fill(); }
    });
  }, [backendLandmarks]);

  const captureGeo = useCallback(() => {
    return new Promise((resolve) => {
      if (!navigator.geolocation) return resolve(null);
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({ lat: pos.coords.latitude.toFixed(7), long: pos.coords.longitude.toFixed(7), timestamp: new Date().toISOString() }),
        () => resolve(null),
        { enableHighAccuracy: true, maximumAge: 0, timeout: 12000 }
      );
    });
  }, []);

  const reverseGeocode = useCallback(async (lat, long) => {
    try {
      const res = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${long}&localityLanguage=en`);
      const data = await res.json();
      const parts = [data.locality, data.principalSubdivision, data.countryName].filter(Boolean);
      return {
        city: data.locality || data.city || "", state: data.principalSubdivision || "",
        country: data.countryName || "",
        full: data.localityInfo?.administrative?.map((a) => a.name).filter(Boolean).join(", ") || parts.join(", "),
        short: parts.join(", "),
      };
    } catch { return null; }
  }, []);

  // ── STREAM FRAME TO BACKEND ──
  const streamFrameToBackend = useCallback(async () => {
    if (streamingRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !livenessSessionIdRef.current) return;
    streamingRef.current = true;
    try {
      const c = document.createElement("canvas");
      c.width = video.videoWidth; c.height = video.videoHeight;
      c.getContext("2d").drawImage(video, 0, 0);
      const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.5));
      if (!blob) { streamingRef.current = false; return; }
      const fd = new FormData();
      fd.append("session_id", livenessSessionIdRef.current);
      fd.append("frame", blob, "frame.jpg");
      const res = await fetch(`${API_URL}/liveness/frame`, { method: "POST", body: fd });
      const data = await res.json();
      handleBackendResponse(data);
    } catch (e) {
      console.warn("Frame stream error:", e);
    }
    streamingRef.current = false;
  }, []);

  const handleBackendResponse = useCallback((data) => {
    if (!data) return;
    if (data.landmarks) setBackendLandmarks(data.landmarks);
    if (data.step) setLivenessStep(data.step);
    if (data.detail) setChallengeMsg(data.detail);

    // Light challenge overlay
    if (data.instruction) {
      const colorMap = { white_flash: "white", blue_flash: "blue", green_flash: "green", brightness_up: "white", brightness_down: null };
      setLightOverlay(colorMap[data.instruction] || null);
    } else if (data.step !== "light_challenge") {
      setLightOverlay(null);
    }

    // Update gesture progress
    if (data.step === "gesture" && data.gesture_idx !== undefined) {
      setChallengeIndex(data.gesture_idx);
      // Build completed list
      const completed = [];
      for (let i = 0; i < data.gesture_idx; i++) completed.push(true);
      setCompletedChallenges(completed);
    }

    if (data.status === "verified" || data.step === "complete") {
      // All checks passed — complete session
      if (!livenessCompletedRef.current) {
        livenessCompletedRef.current = true;
        completeSession();
      }
    }

    if (data.status === "rejected") {
      setError(data.detail || "Liveness check failed");
      stopCamera();
    }
  }, []);

  const completeSession = useCallback(async () => {
    if (!livenessSessionIdRef.current) return;
    // Stop streaming
    if (frameIntervalRef.current) { clearInterval(frameIntervalRef.current); frameIntervalRef.current = null; }
    try {
      const res = await fetch(`${API_URL}/liveness/session/complete`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: livenessSessionIdRef.current }),
      });
      const data = await res.json();
      if (res.ok && data.ok) {
        setCanMatch(true);
        setLivenessLive(true);
        setLivenessStep("capture");
        setChallengeMsg("\u2705 Liveness confirmed! Click Capture.");
      } else {
        setError(data.detail || "Liveness verification failed");
        stopCamera();
      }
    } catch {
      setError("Verification request failed");
      stopCamera();
    }
  }, []);

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f); setPreview(URL.createObjectURL(f)); setResults([]); setError(null);
  };
  const handleDrop = (e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); };

  const startCamera = async () => {
    setError(null); setChallengeMsg(""); setLivenessSessionLoading(true);
    const deviceId = getOrCreateDeviceId();

    try {
      const [sessRes, mediaStream] = await Promise.all([
        fetch(`${API_URL}/liveness/session/start`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device_id: deviceId }),
        }).then(async (r) => ({ ok: r.ok, status: r.status, data: await r.json().catch(() => ({})) })),
        navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        }).catch(() => navigator.mediaDevices.getUserMedia({ video: true })),
      ]);

      if (!sessRes.ok) {
        mediaStream?.getTracks().forEach(t => t.stop());
        setError(sessRes.data?.detail || "Could not start liveness session");
        setLivenessSessionLoading(false); return;
      }
      if (!mediaStream) {
        setError("Camera access denied"); setLivenessSessionLoading(false); return;
      }

      const { session_id, gestures } = sessRes.data;
      livenessSessionIdRef.current = session_id;
      livenessCompletedRef.current = false;
      setCanMatch(false);
      setSessionChallenges(gestures);
      setChallengeIndex(0);
      setCompletedChallenges([]);
      setLivenessLive(false);
      setLivenessStep("calibration");
      setBackendLandmarks(null);
      setLightOverlay(null);
      setStream(mediaStream);
      setShowCamera(true);
      setLivenessSessionLoading(false);

      const video = videoRef.current;
      if (!video) { mediaStream.getTracks().forEach(t => t.stop()); setError("Video element not found"); return; }
      video.srcObject = mediaStream; video.muted = true; video.playsInline = true;
      video.onloadedmetadata = () => {
        video.play().catch(() => { setError("Could not play video"); stopCamera(); });
        // Start streaming frames to backend
        if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = setInterval(streamFrameToBackend, FRAME_INTERVAL_MS);
      };
    } catch (err) {
      setError(err?.message || "Could not start camera session");
      setLivenessSessionLoading(false);
    }
  };

  const stopCamera = () => {
    if (frameIntervalRef.current) { clearInterval(frameIntervalRef.current); frameIntervalRef.current = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); setStream(null); }
    const video = videoRef.current;
    if (video) { video.onloadedmetadata = null; video.onerror = null; video.srcObject = null; }
    const canvas = overlayCanvasRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    setShowCamera(false); setLivenessLive(false); setSessionChallenges([]);
    livenessSessionIdRef.current = null; livenessCompletedRef.current = false;
    setCanMatch(false); setBackendLandmarks(null); setLightOverlay(null);
    setLivenessStep("idle"); streamingRef.current = false;
  };

  const takeSelfie = () => {
    const canvas = canvasRef.current; const video = videoRef.current;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      handleFile(new File([blob], "selfie.jpg", { type: "image/jpeg" }));
      if (frameIntervalRef.current) { clearInterval(frameIntervalRef.current); frameIntervalRef.current = null; }
      if (stream) { stream.getTracks().forEach(t => t.stop()); setStream(null); }
      setShowCamera(false);
    }, "image/jpeg");
  };

  const handleMatch = async () => {
    if (!file) return;
    if (!livenessSessionIdRef.current || !livenessCompletedRef.current || !canMatch) {
      setError("Complete the camera liveness check first, then capture a selfie.");
      return;
    }
    setLoading(true); setError(null); setResults([]); setProgress(0); setGeoError(null);
    const geo = await captureGeo(); setGeoData(geo); setGeoAddress(null);
    if (!geo) setGeoError("\u26a0 Location unavailable \u2014 proceeding without geo.");
    else reverseGeocode(geo.lat, geo.long).then(addr => setGeoAddress(addr));

    const interval = setInterval(() => {
      setProgress(p => { if (p >= 90) { clearInterval(interval); return 90; } return p + Math.random() * 12; });
    }, 300);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), MATCH_REQUEST_TIMEOUT_MS);
    const formData = new FormData();
    formData.append("file", file); formData.append("top_k", 10);
    formData.append("device_id", getOrCreateDeviceId());
    if (livenessSessionIdRef.current) formData.append("liveness_session_id", livenessSessionIdRef.current);
    if (geo) { formData.append("geo_lat", geo.lat); formData.append("geo_long", geo.long); formData.append("geo_timestamp", geo.timestamp); }

    try {
      const res = await fetch(`${API_URL}/match`, { method: "POST", body: formData, signal: controller.signal });
      clearTimeout(timeout); const data = await res.json();
      if (data.error) { setError(data.error); setProgress(0); }
      else {
        setResults(data.matches); setProgress(100);
        livenessSessionIdRef.current = null; livenessCompletedRef.current = false;
        setCanMatch(false); setSessionChallenges([]);
      }
    } catch (err) {
      clearTimeout(timeout); setProgress(0);
      setError(err.name === "AbortError" ? "Request timed out." : "Cannot connect to backend.");
    } finally { clearInterval(interval); setLoading(false); }
  };

  const getColor = (s) => s >= 0.9 ? "#24aa4d" : s >= 0.75 ? "#ffbf01" : "#ff0000";
  const getLabel = (s) => s >= 0.9 ? "High" : s >= 0.75 ? "Medium" : "Low";
'''

# Read original file
with open("src/FaceMatch.jsx", "r", encoding="utf-8") as f:
    original = f.read()

# Find the JSX return section (starts at "return (")
# We want to keep everything from the return statement onwards but update it
jsx_start = original.find("  return (\n")
if jsx_start == -1:
    jsx_start = original.find("  return (")

# Extract the JSX portion  
jsx_portion = original[jsx_start:]

# Add light overlay to camera wrapper and update minor things
# Insert light overlay div after the video element
jsx_portion = jsx_portion.replace(
    '<canvas ref={overlayCanvasRef} className="fm-mesh-overlay" />',
    '''<canvas ref={overlayCanvasRef} className="fm-mesh-overlay" />
                {lightOverlay && (
                  <div className={`fm-light-overlay fm-light-overlay--${lightOverlay}`} />
                )}'''
)

# Remove modelsLoaded reference - camera always works now
jsx_portion = jsx_portion.replace(
    '{livenessSessionLoading && (\n',
    '{livenessSessionLoading && ('
).replace(
    'onClick={showCamera ? stopCamera : startCamera}\n',
    'onClick={showCamera ? stopCamera : startCamera}\n'
)

# Build the new file
new_content = NEW_TOP + "\n" + jsx_portion

# Write back
with open("src/FaceMatch.jsx", "w", encoding="utf-8") as f:
    f.write(new_content)

print("✅ FaceMatch.jsx transformed successfully!")
print(f"   Original: {len(original)} chars")
print(f"   New: {len(new_content)} chars")
