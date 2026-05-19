import React, { useState, useRef, useEffect, useCallback } from "react";
import { getApiBase } from "./apiBase";
import { getCoverSourceRect } from "./cameraDrawUtils";
import {
  MATCH_REQUEST_TIMEOUT_MS,
  matchFetchErrorMessage,
  startIndeterminateMatchProgress,
} from "./matchUiUtils";
import "./FaceMatch.css";
import bargadLogo from "./bargad-logo.png";
import bargadBranding from "./bargad-branding (1).svg?url";
import { MapContainer, TileLayer, Marker } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import {
  Target,
  Layers,
  Sun,
  Activity,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowDown,
  Smile,
  Eye,
  Maximize,
  UserCheck,
  AlertOctagon,
  Info,
  Camera,
  MapPin,
  AlertTriangle,
  Search,
} from "lucide-react";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const API_URL = getApiBase();
const DEVICE_KEY = "facematch_device_id";
const FRAME_INTERVAL_MS = 80;
/** Must match JPEG sent to /liveness/frame (same crop as object-fit: cover in a 4:3 box). */
const PROCESS_W = 640;
const PROCESS_H = 480;

const CHALLENGE_UI = {
  turn_left: { label: "Turn your head LEFT", icon: ArrowLeft },
  turn_right: { label: "Turn your head RIGHT", icon: ArrowRight },
  nod: { label: "NOD your head down", icon: ArrowDown },
  look_up: { label: "LOOK slightly up", icon: ArrowUp },
  smile: { label: "SMILE", icon: Smile },
  // surprised: { label: "Look SURPRISED", icon: Smile },
  mouth_open: { label: "OPEN your mouth wide", icon: Smile },
  // wide_eyes: { label: "OPEN eyes wide", icon: Eye },
  // blink_both: { label: "BLINK both eyes", icon: Eye },
  // raise_eyebrows: { label: "Raise your EYEBROWS", icon: Activity },
  // pucker_lips: { label: "PUCKER your lips", icon: Smile },
  // frown: { label: "FROWN (sad face)", icon: Smile },
  move_closer: { label: "Move CLOSER", icon: Maximize },
  move_farther: { label: "Move Away", icon: Maximize },
  shake_head: { label: "Shake head NO", icon: Activity },
  // blink_twice_fast: { label: "BLINK twice fast", icon: Eye },
  look_left_hold: { label: "Look LEFT & HOLD", icon: ArrowLeft },
  look_right_hold: { label: "Look RIGHT & HOLD", icon: ArrowRight },
  look_up_hold: { label: "Look UP & HOLD", icon: ArrowUp },
  look_down_hold: { label: "Look DOWN & HOLD", icon: ArrowDown },
  // head_forward: { label: "Move head FORWARD", icon: ArrowUp },
  // head_backward: { label: "Move head BACKWARD", icon: ArrowDown },
  // eye_left_right: { label: "Move eyes L to R", icon: Eye },
  // smile_then_blink: { label: "SMILE then BLINK", icon: Smile },
  // blink_then_turn_left: { label: "BLINK then turn LEFT", icon: Eye },
  // raise_eyebrows_hold: { label: "Raise brows & HOLD", icon: Activity },
};

// 68-pt landmark segment indices (MediaPipe → 68 mapping on server)
const FACE_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [8, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [12, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [17, 18],
  [18, 19],
  [19, 20],
  [20, 21],
  [22, 23],
  [23, 24],
  [24, 25],
  [25, 26],
  [27, 28],
  [28, 29],
  [29, 30],
  [31, 32],
  [32, 33],
  [33, 34],
  [34, 35],
  [31, 35],
  [36, 37],
  [37, 38],
  [38, 39],
  [39, 40],
  [40, 41],
  [41, 36],
  [42, 43],
  [43, 44],
  [44, 45],
  [45, 46],
  [46, 47],
  [47, 42],
  [48, 49],
  [49, 50],
  [50, 51],
  [51, 52],
  [52, 53],
  [53, 54],
  [54, 55],
  [55, 56],
  [56, 57],
  [57, 58],
  [58, 59],
  [59, 48],
  [60, 61],
  [61, 62],
  [62, 63],
  [63, 64],
  [64, 65],
  [65, 66],
  [66, 67],
  [67, 60],
];

let sessionDeviceId = null;
function getOrCreateDeviceId() {
  if (!sessionDeviceId) {
    try {
      const uuid =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID().slice(0, 8)
          : Math.random().toString(36).substring(2, 10);
      sessionDeviceId = `session_${uuid}`;
    } catch (e) {
      sessionDeviceId = `session_${Date.now()}`;
    }
  }
  return sessionDeviceId;
}

const getColor = (conf) => {
  if (conf > 0.85) return "#00ffaa";
  if (conf > 0.7) return "#00ddff";
  if (conf > 0.5) return "#ffcc00";
  return "#ff4444";
};

const getLabel = (conf) => {
  if (conf > 0.85) return "High Confidence";
  if (conf > 0.7) return "Strong Match";
  if (conf > 0.5) return "Partial Match";
  return "Low Confidence";
};

export default function FaceMatch({ userEmail, userAgentLabel, onLogout }) {
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
  const [errcount, setErrcount] = useState(0);

  // Premium UI features
  const [progress, setProgress] = useState(0);
  const [geoData, setGeoData] = useState(null);
  const [geoError, setGeoError] = useState(null);
  const [geoAddress, setGeoAddress] = useState(null);
  const [processedPreview, setProcessedPreview] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [rejectionError, setRejectionError] = useState(null);
  const [multiPersonError, setMultiPersonError] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [registrationSuccess, setRegistrationSuccess] = useState(null);
  const [showAllResults, setShowAllResults] = useState(false);
  const [toastStep, setToastStep] = useState(null);
  const [toastVisible, setToastVisible] = useState(false);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [toasts, setToasts] = useState([]);
  const [penaltyDetails, setPenaltyDetails] = useState([]);
  /** When backend flags final selfie as non-live (screen/print), show security failure UI even if cosine match is high. */
  const [captureLiveFailure, setCaptureLiveFailure] = useState(null);

  const addToast = useCallback((msg, type = "success") => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, msg, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  }, []);

  const videoRef = useRef();
  const canvasRef = useRef();
  const overlayCanvasRef = useRef();
  const overlayLandmarksRef = useRef(null);
  const overlayMeshRef = useRef(null);
  const inputRef = useRef();
  const frameIntervalRef = useRef(null);
  const livenessSessionIdRef = useRef(null);
  const livenessCompletedRef = useRef(false);
  const streamingRef = useRef(false);
  const profileMenuRef = useRef(null);
  const lastToastTimeRef = useRef(0);

  // Click outside profile menu
  useEffect(() => {
    function handleClickOutside(event) {
      if (
        profileMenuRef.current &&
        !profileMenuRef.current.contains(event.target)
      ) {
        setProfileMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // 20-second registered user check
  useEffect(() => {
    let timer;
    if (showCamera && !livenessLive) {
      timer = setTimeout(() => {
        addToast(
          "Verification taking longer than usual. Please ensure you are a registered user.",
          "warning",
        );
      }, 20000);
    }
    return () => clearTimeout(timer);
  }, [showCamera, livenessLive, addToast]);

  // ── Geo-location capture ──
  const captureGeo = useCallback(() => {
    return new Promise((resolve) => {
      if (!navigator.geolocation) return resolve(null);
      navigator.geolocation.getCurrentPosition(
        (pos) =>
          resolve({
            lat: pos.coords.latitude.toFixed(7),
            long: pos.coords.longitude.toFixed(7),
            timestamp: new Date().toISOString(),
          }),
        () => resolve(null),
        { enableHighAccuracy: true, maximumAge: 0, timeout: 12000 },
      );
    });
  }, []);

  const reverseGeocode = useCallback(async (lat, long) => {
    try {
      const res = await fetch(
        `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${long}&localityLanguage=en`,
      );
      const data = await res.json();
      const parts = [
        data.locality,
        data.principalSubdivision,
        data.countryName,
      ].filter(Boolean);
      return {
        city: data.locality || data.city || "",
        state: data.principalSubdivision || "",
        country: data.countryName || "",
        full:
          data.localityInfo?.administrative
            ?.map((a) => a.name)
            .filter(Boolean)
            .join(", ") || parts.join(", "),
        short: parts.join(", "),
      };
    } catch {
      return null;
    }
  }, []);

  // Face mesh overlay — rAF (no React state per frame) + lighter drawing
  useEffect(() => {
    if (!showCamera) return undefined;
    const canvas = overlayCanvasRef.current;
    if (!canvas) return undefined;
    let rafId = 0;
    const draw = () => {
      const pts = overlayLandmarksRef.current;
      const mesh = overlayMeshRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        rafId = requestAnimationFrame(draw);
        return;
      }
      if (canvas.width !== PROCESS_W) canvas.width = PROCESS_W;
      if (canvas.height !== PROCESS_H) canvas.height = PROCESS_H;
      ctx.clearRect(0, 0, PROCESS_W, PROCESS_H);

      if (mesh && mesh.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = "rgba(0, 255, 170, 0.12)";
        ctx.lineWidth = 0.5;
        for (let i = 0; i < mesh.length; i += 12) {
          const p1 = mesh[i];
          for (let j = i + 1; j < Math.min(i + 24, mesh.length); j += 8) {
            const p2 = mesh[j];
            const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);
            if (dist < 28) {
              ctx.moveTo(p1.x, p1.y);
              ctx.lineTo(p2.x, p2.y);
            }
          }
        }
        ctx.stroke();
        ctx.fillStyle = "rgba(0, 255, 170, 0.35)";
        for (let i = 0; i < mesh.length; i += 3) {
          const p = mesh[i];
          ctx.fillRect(p.x, p.y, 1, 1);
        }
      }

      if (pts && pts.length >= 68) {
        ctx.strokeStyle = "rgba(0, 255, 170, 0.85)";
        ctx.lineWidth = 1.5;
        for (const [a, b] of FACE_CONNECTIONS) {
          if (!pts[a] || !pts[b]) continue;
          ctx.beginPath();
          ctx.moveTo(pts[a].x, pts[a].y);
          ctx.lineTo(pts[b].x, pts[b].y);
          ctx.stroke();
        }
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i];
          ctx.beginPath();
          ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(0, 255, 170, 1)";
          ctx.fill();
          ctx.beginPath();
          ctx.arc(p.x, p.y, 0.9, 0, Math.PI * 2);
          ctx.fillStyle = "#fff";
          ctx.fill();
        }
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
      frameIntervalRef.current = setInterval(
        streamFrameToBackend,
        FRAME_INTERVAL_MS,
      );
    };

    video.onloadedmetadata = () => {
      video
        .play()
        .then(() => {
          console.log("Video playing successfully");
          handlePlay();
        })
        .catch((e) => {
          console.error("Video play failed:", e);
          // Fallback: try playing again on user interaction if needed,
          // but 'muted' should handle most cases.
        });
    };

    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      video.onloadedmetadata = null;
    };
  }, [stream]);

  async function handleBackendResponse(data) {
    if (!data) return;
    // if (data.landmarks) setBackendLandmarks(data.landmarks);
    if (Array.isArray(data.landmarks) && data.landmarks.length >= 68) {
      overlayLandmarksRef.current = data.landmarks.map((p) => ({
        x: p.x,
        y: p.y,
      }));
    } else if (data.mesh === null) {
      overlayLandmarksRef.current = null;
    }
    if (Array.isArray(data.mesh) && data.mesh.length > 0) {
      overlayMeshRef.current = data.mesh.map((p) => ({ x: p.x, y: p.y }));
    } else if (data.mesh === null) {
      overlayMeshRef.current = null;
    }
    // if (data.step && data.step !== livenessStep) {
    //   const prevStep = livenessStep;
    //   setLivenessStep(data.step);

    //   // If we moved forward, show toast
    //   if (data.step !== "idle" && data.step !== "camera") {
    //     setToastStep(data.step);
    //     setToastVisible(true);
    //     if (prevStep !== "idle" && prevStep !== "camera") {
    //       setCompletedSteps(prev => [...new Set([...prev, prevStep])]);

    //       const STEP_MAP = {
    //         calibration: "Face Calibrated",
    //         depth: "Depth Verified",
    //         light_challenge: "Light Check Passed",
    //         micro: "Security Check Passed",
    //         gesture: "Liveness Verified"
    //       };
    //       if (STEP_MAP[prevStep]) addToast(STEP_MAP[prevStep]);
    //     }
    //   }
    // }
    if (data.step && data.step !== livenessStep) {
      // Skip unwanted phases visually
      const hiddenSteps = ["calibration", "depth", "light_challenge", "micro"];

      // Auto jump frontend directly to gesture
      if (hiddenSteps.includes(data.step)) {
        setLivenessStep("gesture");
        return;
      }

      const prevStep = livenessStep;
      setLivenessStep(data.step);

      if (data.step !== "idle" && data.step !== "camera") {
        setToastStep(data.step);
        setToastVisible(true);

        if (prevStep !== "idle" && prevStep !== "camera") {
          setCompletedSteps((prev) => [...new Set([...prev, prevStep])]);

          const STEP_MAP = {
            gesture: "Liveness Verified",
          };

          if (STEP_MAP[prevStep]) addToast(STEP_MAP[prevStep]);
        }
      }
    }
    if (data.detail) setChallengeMsg(data.detail);

    // Multi-person detection handling
    if (data.multi_person) {
      setMultiPersonError(true);
    } else {
      setMultiPersonError(false);
    }

    if (data.step === "gesture" && data.gesture_idx !== undefined) {
      setChallengeIndex(data.gesture_idx);
      const completed = [];
      for (let i = 0; i < data.gesture_idx; i++) completed.push(true);
      setCompletedChallenges(completed);
    }

    if (data.status === "verified" || data.step === "complete") {
      if (!livenessCompletedRef.current) {
        livenessCompletedRef.current = true;
        setCompletedSteps([
          "calibration",
          "depth",
          "light_challenge",
          "micro",
          "gesture",
        ]);
        addToast("Security Check Passed");
        await completeSession();
      }
    }

    if (data.status === "rejected" || data.status === "failed") {
      if (data.status === "rejected") {
        setErrcount((prev) => prev + 10);
        addToast("Security Alert: Electronic device detected.", "error");
        // Silently log rejection without stopping camera or showing modal
        console.warn("Security rejection caught:", data.detail);
      } else {
        setError(data.detail || "Liveness check failed");
      }
    } else if (data.status === "processing") {
      const d = data.detail || "";

      // Increment errcount for suspicious activity (e.g. digital screen, identity mismatch)
      if (data.is_suspicious) {
        // Increment by 10 per event/frame as requested
        setErrcount((prev) => prev + 10);

        // Show small popup at top
        const now = Date.now();
        if (now - lastToastTimeRef.current > 2500) {
          addToast(
            "Security Warning: Potential device or non-live media detected.",
            "warning",
          );
          lastToastTimeRef.current = now;
        }
      }

      if (
        d.includes("blocked") ||
        d.includes("too close") ||
        d.includes("too far") ||
        d.includes("No face")
      ) {
        setError(d);
      } else {
        setError(null);
      }
    }
  }

  async function streamFrameToBackend() {
    if (
      streamingRef.current ||
      !videoRef.current ||
      videoRef.current.readyState < 2 ||
      !livenessSessionIdRef.current
    )
      return;
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
      const { sx, sy, sw, sh } = getCoverSourceRect(
        vw,
        vh,
        PROCESS_W,
        PROCESS_H,
      );
      ctx2.drawImage(video, sx, sy, sw, sh, 0, 0, PROCESS_W, PROCESS_H);
      const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
      if (!blob) {
        streamingRef.current = false;
        return;
      }
      const fd = new FormData();
      fd.append("session_id", livenessSessionIdRef.current);
      fd.append("frame", blob, "frame.jpg");
      const res = await fetch(`${API_URL}/liveness/frame`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      await handleBackendResponse(data);
    } catch (e) {
      console.warn("Stream error:", e);
    }
    streamingRef.current = false;
  }

  async function completeSession() {
    try {
      const res = await fetch(`${API_URL}/liveness/session/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: livenessSessionIdRef.current }),
      });
      const data = await res.json();
      if (res.ok && data.ok) {
        setCanMatch(true);
        setLivenessLive(true);
        setLivenessStep("capture");
      }
    } catch {
      setError("Verification failed");
    }
  }

  async function startCamera() {
    stopCamera();
    setError(null);
    setFile(null);
    setPreview(null);
    setResults([]);
    setPenaltyDetails([]);
    setCaptureLiveFailure(null);
    setLivenessLive(false);
    setCanMatch(false);
    setGeoData(null);
    setGeoAddress(null);
    setProgress(0);
    setErrcount(0);
    setCompletedSteps([]);
    setLivenessSessionLoading(true);
    try {
      let sessData;
      try {
        const sessRes = await fetch(`${API_URL}/liveness/session/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            device_id: getOrCreateDeviceId(),
            agent_label: userAgentLabel,
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
            height: { ideal: 480 },
          },
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
      setChallengeMsg("Waiting for gesture...");
    } catch (e) {
      setError(`Unexpected error: ${e.message}`);
    } finally {
      setLivenessSessionLoading(false);
    }
  }

  function stopCamera() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    setStream(null);
    setShowCamera(false);

    // Only reset liveness if it wasn't completed
    if (!canMatch) {
      setLivenessLive(false);
      livenessSessionIdRef.current = null;
      livenessCompletedRef.current = false;
      setLivenessStep("idle");
    }

    overlayLandmarksRef.current = null;
    overlayMeshRef.current = null;
    setChallengeMsg("");
    setError(null);
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

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          console.error("Failed to create blob from canvas");
          setError("Capture failed: Could not process image.");
          return;
        }
        console.log("✅ Blob created, triggering match...");
        const f = new File([blob], "selfie.jpg", { type: "image/jpeg" });
        const currentSessionId = livenessSessionIdRef.current;

        setFile(f);
        setPreview(URL.createObjectURL(f));

        // 2. Stop camera and streaming immediately
        stopCamera();

        // 3. Trigger match with the saved session ID
        handleMatch(f, currentSessionId);
      },
      "image/jpeg",
      0.95,
    );
  };

  const handleMatch = async (fileOverride = null, sessionIdOverride = null) => {
    const fileToUse = fileOverride || file;
    const sessionIdToUse = sessionIdOverride || livenessSessionIdRef.current;

    console.log("🔍 Starting Match process", {
      hasFile: !!fileToUse,
      hasSession: !!sessionIdToUse,
      canMatch,
    });

    // If we are overriding with a direct file from capture, we bypass the state-based canMatch
    // because liveness is already verified to reach the capture button.
    const isDirectCapture = !!fileOverride;

    if (!fileToUse) {
      setError("Please upload an image or take a selfie first.");
      return;
    }

    // Liveness is recommended but no longer strictly required for uploaded files in the UI
    // It remains mandatory for direct camera capture (which is handled by isDirectCapture being true)
    // Allow matching if session is technically complete, even if not perfectly 'verified'
    const isSessionComplete =
      livenessStep === "complete" || livenessStep === "capture";

    if (!canMatch && isDirectCapture && !isSessionComplete) {
      setError(
        "Liveness verification failed. Please try the camera flow again.",
      );
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);
    setPenaltyDetails([]);
    setCaptureLiveFailure(null);
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
      reverseGeocode(geo.lat, geo.long).then((addr) => {
        console.log("🌍 Geo Address:", addr);
        setGeoAddress(addr);
      });
    }

    const matchAbort = new AbortController();
    const matchTimeoutId = setTimeout(
      () => matchAbort.abort(),
      MATCH_REQUEST_TIMEOUT_MS,
    );
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

    if (errcount > 0) {
      fd.append("errcount", errcount);
    }
    if (userAgentLabel && isDirectCapture) {
      fd.append("expected_label", userAgentLabel);
    }
    try {
      console.log(`📤 Sending match request to ${API_URL}/match ...`);
      console.log("Data: ", fd);
      const res = await fetch(`${API_URL}/match`, {
        method: "POST",
        body: fd,
        signal: matchAbort.signal,
      });
      const data = await res.json();
      clearInterval(pInterval);
      clearTimeout(matchTimeoutId);

      if (data.error) {
        console.error("❌ Match error from backend:", data.error);
        setError(data.error);
        setCaptureLiveFailure(null);
        setProgress(0);
      } else {
        console.log("✅ Match successful", data.matches?.length, "results");
        setResults(data.matches || []);
        setPenaltyDetails(data.security_penalty_breakdown || []);
        if (data.capture_live_ok === false) {
          setCaptureLiveFailure({
            reason:
              data.capture_live_reason ||
              "Final capture did not pass live-face checks (screen, photo, or replay suspected).",
            score:
              typeof data.capture_live_score === "number"
                ? data.capture_live_score
                : null,
          });
        } else {
          setCaptureLiveFailure(null);
        }
        if (data.processed_image) setProcessedPreview(data.processed_image);
        if (data.captured_image) setCapturedImage(data.captured_image);
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
    if (!file || !registerName) {
      setError("Please provide a name and capture a selfie first.");
      return;
    }
    setLoading(true);
    setError(null);
    setRegistrationSuccess(null);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", registerName);
    fd.append("liveness_session_id", currentSessionId);
    fd.append("device_id", getOrCreateDeviceId());
    fd.append("errcount", errcount);

    try {
      console.log(`📤 Sending registration request to ${API_URL}/register ...`);
      const res = await fetch(`${API_URL}/register`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setRegistrationSuccess(`Successfully registered ${registerName}!`);
        setRegisterMode(false);
        setRegisterName("");
        setCanMatch(false);
        setLivenessStep("idle");
      }
    } catch (err) {
      setError("Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReload = () => {
    window.location.reload();
  };

  return (
    <div className="fm-page">
      {/* Toast Notification Pipeline */}
      <div className="fm-toast-pipeline">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`fm-floating-toast ${t.type || "success"}`}
          >
            <div className="fm-toast-icon">
              {t.type === "warning" ? (
                <AlertTriangle size={18} color="#ffbf01" />
              ) : t.type === "error" ? (
                <AlertOctagon size={18} color="#ff4444" />
              ) : (
                <UserCheck size={18} color="#24aa4d" />
              )}
            </div>
            <span>{t.msg}</span>
          </div>
        ))}
      </div>

      <header className="fm-header-banner">
        <div className="fm-header-left">
          <img src={bargadLogo} alt="Bargad" className="fm-header-logo" />
        </div>
        <div className="fm-header-profile" ref={profileMenuRef}>
          <div
            className="fm-profile-avatar-wrap"
            onClick={() => setProfileMenuOpen(!profileMenuOpen)}
          >
            {/* <div className="fm-profile-info">
              <span className="fm-profile-name">{userAgentLabel || "Agent"}</span>
            </div> */}
            <svg
              className="fm-profile-avatar-svg"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v1c0 .55.45 1 1 1h14c.55 0 1-.45 1-1v-1c0-2.66-5.33-4-8-4z" />
            </svg>
          </div>

          {profileMenuOpen && (
            <div className="fm-profile-dropdown">
              <div className="fm-dropdown-header">
                <div className="fm-user-email">
                  {userEmail || "Session active"}
                </div>
                <div className="fm-user-label">
                  {userAgentLabel || "Authorized Agent"}
                </div>
              </div>
              <div className="fm-dropdown-divider" />
              <button className="fm-dropdown-item logout" onClick={onLogout}>
                <svg
                  className="fm-logout-icon"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" />
                </svg>
                Log Out
              </button>
            </div>
          )}
        </div>
      </header>

      <div className={`fm-container ${showCamera ? "fm-container-wide" : ""}`}>
        {showCamera && (
          <div className="cancel-btn-top-liveness-steps">
            <button className="fm-cancel-btn1" onClick={stopCamera}>
              <ArrowLeft /> Cancel
            </button>
          </div>
        )}
        {!showCamera && (
          <div className="fm-header">
            <div className="fm-logo">⬡</div>
            <div>
              <h1>Face Match and Liveness</h1>   
              <p>
                Complete liveness flow, then capture or upload to find matches.
              </p>
            </div>
          </div>
        )}

        <div className={showCamera ? "fm-upload-camera-row" : ""}>
          {!showCamera && (
            <div className="fm-left-col">
              <div
                className={`fm-dropzone ${preview ? "has-preview" : ""} ${dragging ? "dragging" : ""}`}
                onClick={() => inputRef.current.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  const droppedFile = e.dataTransfer.files[0];
                  if (droppedFile && droppedFile.type.startsWith("image/")) {
                    setFile(droppedFile);
                    setPreview(URL.createObjectURL(droppedFile));
                    setError(null);
                    setResults([]);
                  }
                }}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(e) => {
                    const selectedFile = e.target.files[0];
                    if (selectedFile) {
                      setFile(selectedFile);
                      setPreview(URL.createObjectURL(selectedFile));
                      setError(null);
                      setResults([]);
                    }
                  }}
                />
                {preview ? (
                  <img
                    src={preview}
                    alt="Preview"
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                    }}
                  />
                ) : (
                  <div className="fm-drop-content">
                    <p>
                      Drag & drop or <span>click to upload</span>
                    </p>
                  </div>
                )}
                {preview && !loading && (
                  <button
                    className="fm-clear-preview"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                      setPreview(null);
                      setResults([]);
                      setError(null);
                    }}
                  >
                    ✕
                  </button>
                )}
              </div>
              <div className="fm-btn-row">
                <button
                  className="fm-btn"
                  onClick={() => handleMatch()}
                  disabled={loading || results.length > 0}
                  style={{ width: "100%" }}
                >
                  {loading ? <div className="fm-spinner" /> : null}
                  {loading
                    ? "Searching..."
                    : results.length > 0
                      ? "Results Ready"
                      : "Find Matches"}
                </button>
                <button
                  className="fm-camera-btn"
                  // className="fm-btn"
                  // style={{backgroundColor: "#000", border: "1px solid var(--border)", color: "#fff", fontSize: "14px", fontWeight: "600", borderRadius: "16px", cursor: "pointer", transition: "var(--transition)"}}
                  onClick={showCamera ? stopCamera : startCamera}
                >
                  {showCamera ? "✕ Cancel" : "Take Selfie Instead"}
                </button>
              </div>
            </div>
          )}

          {showCamera && (
            <div className="fm-right-col">
              <div className="fm-camera-outer">
                <div className="fm-camera-container">
                  <div className="fm-main-camera-contianer-relative">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="fm-camera-feed"
                    />
                    <canvas
                      ref={overlayCanvasRef}
                      className="fm-mesh-overlay"
                    />
                    <canvas ref={canvasRef} style={{ display: "none" }} />
                    <div className="fm-main-camera-contianer-absolute">
                      {showCamera && (
                        <div className="fm-liveness-overlay">
                          {/* Secure Scan State */}
                          {[
                            "calibration",
                            "depth",
                            "light_challenge",
                            "micro",
                          ].includes(livenessStep) && (
                            <div className="fm-gesture-pill">
                              <div
                                className="fm-gesture-icon-wrap"
                                style={{ background: "#24aa4d" }}
                              >
                                <Activity size={18} />
                              </div>
                              <span className="fm-gesture-text">
                                Secure Scan...
                              </span>
                            </div>
                          )}

                          {/* Active Gesture Pill */}
                          {livenessStep === "gesture" &&
                            sessionChallenges[challengeIndex] && (
                              <div className="fm-gesture-pill-container">
                                <div className="fm-gesture-pill">
                                  <div className="fm-gesture-icon-wrap">
                                    {(() => {
                                      const IconComp =
                                        CHALLENGE_UI[
                                          sessionChallenges[challengeIndex]
                                        ]?.icon || Activity;
                                      return <IconComp size={18} />;
                                    })()}
                                  </div>
                                  <span className="fm-gesture-text">
                                    {
                                      CHALLENGE_UI[
                                        sessionChallenges[challengeIndex]
                                      ]?.label
                                    }
                                  </span>
                                </div>
                                <div className="fm-gesture-dots">
                                  {sessionChallenges.map((_, idx) => (
                                    <div
                                      key={idx}
                                      className={`fm-gesture-dot ${idx === challengeIndex ? "active" : ""} ${idx < challengeIndex ? "completed" : ""}`}
                                    />
                                  ))}
                                </div>
                              </div>
                            )}

                          {/* Verification Complete Card */}
                          {/* {
                            setTimeout(() => {
                              {
                                (livenessLive || livenessStep === "complete" || livenessStep === "capture") && (
                                  <div className="fm-gesture-pill" style={{ background: 'linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%)' }}>
                                    <div className="fm-gesture-icon-wrap" style={{ background: '#4caf50' }}>
                                      <UserCheck size={18} />
                                    </div>
                                    <span className="fm-gesture-text">Verification Complete</span>
                                  </div>
                                )
                              }
                            }, 1200)
                          } */}

                          {/* Helper message */}
                          {/* {challengeMsg && !livenessLive && !error && (
                            <div className="fm-liveness-status-msg" style={{ marginTop: 10, background: 'rgba(0,0,0,0.5)', padding: '4px 12px', borderRadius: 99, fontSize: 12 }}>
                              <span>{challengeMsg}</span>
                            </div>
                          )} */}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* High-tech Viewfinder Corners */}
                  <div className="fm-viewfinder-corner top-left"></div>
                  <div className="fm-viewfinder-corner top-right"></div>
                  <div className="fm-viewfinder-corner bottom-left"></div>
                  <div className="fm-viewfinder-corner bottom-right"></div>

                  {/* Animated Scanline */}
                  <div className="fm-scanline"></div>

                  {error && (
                    <div className="fm-camera-error-overlay">{error}</div>
                  )}

                  {/* Multi-person Toast Alert */}
                  {multiPersonError && (
                    <div className="fm-multi-person-toast">
                      <AlertOctagon size={18} />
                      <span>
                        Multiple people detected — only one person allowed
                      </span>
                    </div>
                  )}

                  <div className="fm-camera-actions">
                    {(livenessLive ||
                      livenessStep === "complete" ||
                      livenessStep === "capture") &&
                      !multiPersonError && (
                        <button className="fm-capture-btn" onClick={takeSelfie}>
                          <Camera size={18} /> Capture Selfie
                        </button>
                      )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Geo card */}
        {geoData && (
          <div className="fm-geo-card">
            <div className="fm-geo-map">
              <MapContainer
                center={[parseFloat(geoData.lat), parseFloat(geoData.long)]}
                zoom={16}
                style={{ width: "120px", height: "110px" }}
                zoomControl={false}
                dragging={false}
                scrollWheelZoom={false}
                doubleClickZoom={false}
                attributionControl={false}
              >
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                <Marker
                  position={[parseFloat(geoData.lat), parseFloat(geoData.long)]}
                />
              </MapContainer>
            </div>
            <div className="fm-geo-details">
              <div className="fm-geo-city">
                <MapPin size={12} style={{ marginRight: 4 }} />{" "}
                {geoAddress ? geoAddress.short : "Fetching..."} 🇮🇳
              </div>
              {geoAddress && geoAddress.full && (
                <div className="fm-geo-full-address">{geoAddress.full}</div>
              )}
              <div className="fm-geo-coords">
                Lat {parseFloat(geoData.lat).toFixed(6)}° &nbsp; Long{" "}
                {parseFloat(geoData.long).toFixed(6)}°
              </div>
              <div className="fm-geo-time">
                {new Date(geoData.timestamp).toLocaleString()}
              </div>
            </div>
          </div>
        )}
        {geoError && <div className="fm-geo-error">{geoError}</div>}

        {loading && (
          <div className="fm-progress-wrapper">
            <div
              className="fm-progress-bar"
              style={{ width: `${progress}%` }}
            />
            <span>Searching Faces... {Math.round(progress)}%</span>
          </div>
        )}

        {loading && preview && (
          <div className="fm-scan-wrapper">
            <img src={preview} alt="Scanning" className="fm-scan-img" />
            <div className="fm-scan-line" />
            <div className="fm-scan-corners">
              <span className="corner tl" />
              <span className="corner tr" />
              <span className="corner bl" />
              <span className="corner br" />
            </div>
            <div className="fm-scan-label">Analyzing Face...</div>
          </div>
        )}

        {error && (
          <div className="fm-error">
            <AlertTriangle size={16} style={{ marginRight: 8 }} /> {error}
          </div>
        )}

        {results.length > 0 && !loading ? (
          !captureLiveFailure &&
          (results[0].confidence * 100).toFixed(0) >= 80 ? (
            <div>
              <div className="fm-verification-summary-bar">
                <div className="fm-verified-badge">
                  <UserCheck size={18} />
                  <span>
                    Identity Verified (Doc:{" "}
                    {results[0]?.registered_doc_type || "Aadhar"})
                  </span>
                </div>
                <div className="fm-match-indicator">
                  Top Match:{" "}
                  <strong>{(results[0].confidence * 100).toFixed(0)}%</strong>
                </div>
              </div>

              {/* Results Summary Section */}
              <div className="fm-results-summary">
                {registrationSuccess && (
                  <div className="fm-status-alert success">
                    <UserCheck size={20} />
                    <span>{registrationSuccess}</span>
                  </div>
                )}

                <div className="fm-results-container">
                  <div className="fm-grid">
                    {results
                      .filter((match) => match.label !== "txt")
                      .slice(0, showAllResults ? undefined : 1)
                      .map((match, i) => (
                        <div className="fm-card result-card" key={i}>
                          <div className="fm-rank">#{i + 1}</div>
                          <div className="image-compare">
                            <div className="image-box">
                              <p>Matched (DB)</p>
                              <img
                                src={
                                  match.matched_image ||
                                  (match.images && match.images[0])
                                }
                                alt="DB"
                              />
                            </div>
                            <div className="image-box">
                              <p>Captured (Live)</p>
                              <img src={capturedImage || preview} alt="Live" />
                            </div>
                          </div>
                          <div className="fm-card-body result-info">
                            <div className="fm-name-row">
                              <h2 className="fm-match-title">
                                {Math.round(match.confidence * 100)}% Match
                              </h2>
                              <span className="doc-badge">
                                VERIFIED (Doc:{" "}
                                {match.registered_doc_type || "Aadhar"})
                              </span>
                            </div>
                            <p className="fm-label-name">
                              {(match.label || "Unknown").replace(/_/g, " ")}
                            </p>
                            <div className="fm-bar-bg">
                              <div
                                className="fm-bar-fill"
                                style={{
                                  width: `${match.confidence * 100}%`,
                                  background: getColor(match.confidence),
                                }}
                              />
                            </div>
                            <div className="fm-score-row">
                              <span
                                className="fm-badge"
                                style={{
                                  background: getColor(match.confidence),
                                }}
                              >
                                {getLabel(match.confidence)}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>

                  {results.filter((m) => m.label !== "txt").length > 1 &&
                    !showAllResults && (
                      <div className="fm-load-more-container">
                        <button
                          className="fm-load-more-btn"
                          onClick={() => setShowAllResults(true)}
                        >
                          Load More Matches
                        </button>
                      </div>
                    )}

                  {penaltyDetails.length > 0 && (
                    <div className="fm-penalty-container">
                      <h3 className="fm-penalty-title">
                        Security Penalty Breakdown
                      </h3>
                      <table className="fm-penalty-table">
                        <thead>
                          <tr>
                            <th>Violation Type</th>
                            <th>Occurrence</th>
                            <th>Reduction</th>
                          </tr>
                        </thead>
                        <tbody>
                          {penaltyDetails.map((p, i) => (
                            <tr key={i}>
                              <td>{p.type}</td>
                              <td>{p.count}x</td>
                              <td className="fm-penalty-red">
                                -{Math.round(p.penalty * 100)}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="fm-error" style={{ marginTop: 20 }}>
                <AlertTriangle size={16} style={{ marginRight: 8 }} />
                {captureLiveFailure ? (
                  <>
                    Security Check Failed: Final capture failed live-face
                    verification
                    {captureLiveFailure.score != null
                      ? ` (PAD score ${captureLiveFailure.score})`
                      : ""}
                    . {captureLiveFailure.reason} Do not point the camera at a
                    screen or printed photo; use your face directly, well lit.
                  </>
                ) : (
                  <>
                    Security Check Failed: Match confidence too low (
                    {(results[0].confidence * 100).toFixed(0)}%). Please ensure
                    you are the registered agent and not using a digital screen.
                  </>
                )}
              </div>

              {penaltyDetails.length > 0 && (
                <div className="fm-penalty-container">
                  <h3 className="fm-penalty-title">
                    Security Penalty Breakdown
                  </h3>
                  <table className="fm-penalty-table">
                    <thead>
                      <tr>
                        <th>Violation Type</th>
                        <th>Occurrence</th>
                        <th>Reduction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {penaltyDetails.map((p, i) => (
                        <tr key={i}>
                          <td>{p.type}</td>
                          <td>{p.count}x</td>
                          <td className="fm-penalty-red">
                            -{Math.round(p.penalty * 100)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="fm-penalty-retry">
                <p>
                  Verification failed due to high security risk or low
                  similarity. Please try again in a well-lit environment.
                </p>
                <button className="fm-btn retry-btn" onClick={handleReload}>
                  <Camera size={18} /> Retry Verification
                </button>
              </div>
            </>
          )
        ) : null}

        {/* Rejection Modal Removed as per user request to show inline table instead */}
      </div>
      <div className="fm-footer-branding">
        <img src={bargadBranding} alt="Bargad" />
      </div>
    </div>
  );
}
