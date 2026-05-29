// import React, { useState, useRef, useEffect, useCallback } from "react";
// import { getApiBase } from "./apiBase";
// import { getCoverSourceRect } from "./cameraDrawUtils";
// import {
//   MATCH_REQUEST_TIMEOUT_MS,
//   matchFetchErrorMessage,
//   startIndeterminateMatchProgress,
// } from "./matchUiUtils";
// import "./FaceMatch.css";
// import bargadLogo from "./bargad-logo.png";
// import bargadBranding from "./bargad-branding (1).svg?url";
// import { MapContainer, TileLayer, Marker } from "react-leaflet";
// import "leaflet/dist/leaflet.css";
// import L from "leaflet";
// import {
//   Target, 
//   Layers,
//   Sun,
//   Activity,
//   ArrowLeft,
//   ArrowRight,
//   ArrowUp,
//   ArrowDown,
//   Smile,
//   Eye,
//   Maximize,
//   UserCheck,
//   AlertOctagon,
//   Info,
//   Camera,
//   MapPin,
//   AlertTriangle,
//   Search,
//   Loader2, Power, CheckCircle, Play,
// } from "lucide-react";

// delete L.Icon.Default.prototype._getIconUrl;
// L.Icon.Default.mergeOptions({
//   iconRetinaUrl:
//     "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
//   iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
//   shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
// });

// const API_URL = getApiBase();
// const DEVICE_KEY = "facematch_device_id";
// const FRAME_INTERVAL_MS = 80;
// /** Short countdown; probe frames run security checks before main stream. */
// const INIT_COUNTDOWN_SEC = 1;
// const CHALLENGE_PREP_DELAY_MS = 1200;
// const PROBE_FRAME_INTERVAL_MS = 350;
// const DEVICE_TOAST_INTERVAL_MS = 1200;
// const OVERLAY_LERP = 0.42;
// /** Must match JPEG sent to /liveness/frame (same crop as object-fit: cover in a 4:3 box). */
// const PROCESS_W = 640;
// const PROCESS_H = 480;

// function lerpPoints(prev, next, t) {
//   if (!next?.length) return null;
//   if (!prev?.length || prev.length !== next.length) return next.map((p) => ({ ...p }));
//   return next.map((p, i) => ({
//     x: prev[i].x + (p.x - prev[i].x) * t,
//     y: prev[i].y + (p.y - prev[i].y) * t,
//   }));
// }

// const CHALLENGE_UI = {
//   turn_left: { label: "Turn your head LEFT", icon: ArrowLeft },
//   turn_right: { label: "Turn your head RIGHT", icon: ArrowRight },
//   nod: { label: "NOD your head down", icon: ArrowDown },
//   look_up: { label: "LOOK slightly up", icon: ArrowUp },
//   smile: { label: "SMILE", icon: Smile },
//   // surprised: { label: "Look SURPRISED", icon: Smile },
//   mouth_open: { label: "OPEN your mouth wide", icon: Smile },
//   // wide_eyes: { label: "OPEN eyes wide", icon: Eye },
//   // blink_both: { label: "BLINK both eyes", icon: Eye },
//   // raise_eyebrows: { label: "Raise your EYEBROWS", icon: Activity },
//   // pucker_lips: { label: "PUCKER your lips", icon: Smile },
//   // frown: { label: "FROWN (sad face)", icon: Smile },
//   move_closer: { label: "Move CLOSER", icon: Maximize },
//   move_farther: { label: "Move Away", icon: Maximize },
//   shake_head: { label: "Shake head NO", icon: Activity },
//   // blink_twice_fast: { label: "BLINK twice fast", icon: Eye },
//   look_left_hold: { label: "Look LEFT & HOLD", icon: ArrowLeft },
//   look_right_hold: { label: "Look RIGHT & HOLD", icon: ArrowRight },
//   look_up_hold: { label: "Look UP & HOLD", icon: ArrowUp },
//   look_down_hold: { label: "Look DOWN & HOLD", icon: ArrowDown },
//   // head_forward: { label: "Move head FORWARD", icon: ArrowUp },
//   // head_backward: { label: "Move head BACKWARD", icon: ArrowDown },
//   // eye_left_right: { label: "Move eyes L to R", icon: Eye },
//   // smile_then_blink: { label: "SMILE then BLINK", icon: Smile },
//   // blink_then_turn_left: { label: "BLINK then turn LEFT", icon: Eye },
//   // raise_eyebrows_hold: { label: "Raise brows & HOLD", icon: Activity },
// };

// // 68-pt landmark segment indices (MediaPipe → 68 mapping on server)
// const FACE_CONNECTIONS = [
//   [0, 1],
//   [1, 2],
//   [2, 3],
//   [3, 4],
//   [4, 5],
//   [5, 6],
//   [6, 7],
//   [7, 8],
//   [8, 9],
//   [9, 10],
//   [10, 11],
//   [11, 12],
//   [12, 13],
//   [13, 14],
//   [14, 15],
//   [15, 16],
//   [17, 18],
//   [18, 19],
//   [19, 20],
//   [20, 21],
//   [22, 23],
//   [23, 24],
//   [24, 25],
//   [25, 26],
//   [27, 28],
//   [28, 29],
//   [29, 30],
//   [31, 32],
//   [32, 33],
//   [33, 34],
//   [34, 35],
//   [31, 35],
//   [36, 37],
//   [37, 38],
//   [38, 39],
//   [39, 40],
//   [40, 41],
//   [41, 36],
//   [42, 43],
//   [43, 44],
//   [44, 45],
//   [45, 46],
//   [46, 47],
//   [47, 42],
//   [48, 49],
//   [49, 50],
//   [50, 51],
//   [51, 52],
//   [52, 53],
//   [53, 54],
//   [54, 55],
//   [55, 56],
//   [56, 57],
//   [57, 58],
//   [58, 59],
//   [59, 48],
//   [60, 61],
//   [61, 62],
//   [62, 63],
//   [63, 64],
//   [64, 65],
//   [65, 66],
//   [66, 67],
//   [67, 60],
// ];

// let sessionDeviceId = null;
// function getOrCreateDeviceId() {
//   if (!sessionDeviceId) {
//     try {
//       const uuid =
//         typeof crypto !== "undefined" && crypto.randomUUID
//           ? crypto.randomUUID().slice(0, 8)
//           : Math.random().toString(36).substring(2, 10);
//       sessionDeviceId = `session_${uuid}`;
//     } catch (e) {
//       sessionDeviceId = `session_${Date.now()}`;
//     }
//   }
//   return sessionDeviceId;
// }

// const getColor = (conf) => {
//   if (conf > 0.85) return "#00ffaa";
//   if (conf > 0.7) return "#00ddff";
//   if (conf > 0.5) return "#ffcc00";
//   return "#ff4444";
// };

// const getLabel = (conf) => {
//   if (conf > 0.85) return "High Confidence";
//   if (conf > 0.7) return "Strong Match";
//   if (conf > 0.5) return "Partial Match";
//   return "Low Confidence";
// };

// export default function FaceMatch({ userEmail, userAgentLabel, onLogout }) {
//   const [preview, setPreview] = useState(null);
//   const [file, setFile] = useState(null);
//   const [results, setResults] = useState([]);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [dragging, setDragging] = useState(false);
//   const [showCamera, setShowCamera] = useState(false);
//   const [stream, setStream] = useState(null);

//   const [challengeIndex, setChallengeIndex] = useState(0);
//   const [completedChallenges, setCompletedChallenges] = useState([]);
//   const [livenessLive, setLivenessLive] = useState(false);
//   const [challengeMsg, setChallengeMsg] = useState("");
//   const [sessionChallenges, setSessionChallenges] = useState([]);
//   const [livenessSessionLoading, setLivenessSessionLoading] = useState(false);
//   const [canMatch, setCanMatch] = useState(false);
//   const [livenessStep, setLivenessStep] = useState("idle");
//   const [lightOverlay, setLightOverlay] = useState(null);
//   const [errcount, setErrcount] = useState(0);

//   // Premium UI features
//   const [progress, setProgress] = useState(0);
//   const [geoData, setGeoData] = useState(null);
//   const [geoError, setGeoError] = useState(null);
//   const [geoAddress, setGeoAddress] = useState(null);
//   const [processedPreview, setProcessedPreview] = useState(null);
//   const [capturedImage, setCapturedImage] = useState(null);
//   const [profileMenuOpen, setProfileMenuOpen] = useState(false);
//   const [rejectionError, setRejectionError] = useState(null);
//   const [multiPersonError, setMultiPersonError] = useState(false);
//   const [gesturePrepActive, setGesturePrepActive] = useState(false);
//   const [currentSessionId, setCurrentSessionId] = useState(null);
//   const [registrationSuccess, setRegistrationSuccess] = useState(null);
//   const [showAllResults, setShowAllResults] = useState(false);
//   const [toastStep, setToastStep] = useState(null);
//   const [toastVisible, setToastVisible] = useState(false);
//   const [completedSteps, setCompletedSteps] = useState([]);
//   const [toasts, setToasts] = useState([]);
//   const [penaltyDetails, setPenaltyDetails] = useState([]);
//   /** When backend flags final selfie as non-live (screen/print), show security failure UI even if cosine match is high. */
//   const [captureLiveFailure, setCaptureLiveFailure] = useState(null);

//   const addToast = useCallback((msg, type = "success") => {
//     const id = Date.now();
//     setToasts((prev) => [...prev, { id, msg, type }]);
//     setTimeout(() => {
//       setToasts((prev) => prev.filter((t) => t.id !== id));
//     }, 3500);
//   }, []);

//   const videoRef = useRef();
//   const canvasRef = useRef();
//   const overlayCanvasRef = useRef();
//   const overlayLandmarksRef = useRef(null);
//   const overlayMeshRef = useRef(null);
//   const overlayDisplayLandmarksRef = useRef(null);
//   const overlayDisplayMeshRef = useRef(null);
//   const inputRef = useRef();
//   const frameIntervalRef = useRef(null);
//   const livenessSessionIdRef = useRef(null);
//   const livenessCompletedRef = useRef(false);
//   const streamingRef = useRef(false);
//   const profileMenuRef = useRef(null);
//   const lastToastTimeRef = useRef(0);
//   const lastDeviceToastRef = useRef(0);
//   const probeIntervalRef = useRef(null);
//   const gesturePrepTimeoutRef = useRef(null);
//   const gesturePrepDoneRef = useRef(false);
//   const initCountdownIntervalRef = useRef(null);

//   const clearCameraTimers = useCallback(() => {
//     if (initCountdownIntervalRef.current) {
//       clearInterval(initCountdownIntervalRef.current);
//       initCountdownIntervalRef.current = null;
//     }
//     if (probeIntervalRef.current) {
//       clearInterval(probeIntervalRef.current);
//       probeIntervalRef.current = null;
//     }
//     if (gesturePrepTimeoutRef.current) {
//       clearTimeout(gesturePrepTimeoutRef.current);
//       gesturePrepTimeoutRef.current = null;
//     }
//   }, []);

//   // Click outside profile menu
//   useEffect(() => {
//     function handleClickOutside(event) {
//       if (
//         profileMenuRef.current &&
//         !profileMenuRef.current.contains(event.target)
//       ) {
//         setProfileMenuOpen(false);
//       }
//     }
//     document.addEventListener("mousedown", handleClickOutside);
//     return () => document.removeEventListener("mousedown", handleClickOutside);
//   }, []);

//   // 20-second registered user check
//   useEffect(() => {
//     let timer;
//     if (showCamera && !livenessLive) {
//       timer = setTimeout(() => {
//         addToast(
//           "Verification taking longer than usual. Please ensure you are a registered user.",
//           "warning",
//         );
//       }, 20000);
//     }
//     return () => clearTimeout(timer);
//   }, [showCamera, livenessLive, addToast]);

//   // ── Geo-location capture ──
//   const captureGeo = useCallback(() => {
//     return new Promise((resolve) => {
//       if (!navigator.geolocation) return resolve(null);
//       navigator.geolocation.getCurrentPosition(
//         (pos) =>
//           resolve({
//             lat: pos.coords.latitude.toFixed(7),
//             long: pos.coords.longitude.toFixed(7),
//             timestamp: new Date().toISOString(),
//           }),
//         () => resolve(null),
//         { enableHighAccuracy: true, maximumAge: 0, timeout: 12000 },
//       );
//     });
//   }, []);

//   const reverseGeocode = useCallback(async (lat, long) => {
//     try {
//       const res = await fetch(
//         `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${long}&localityLanguage=en`,
//       );
//       const data = await res.json();
//       const parts = [
//         data.locality,
//         data.principalSubdivision,
//         data.countryName,
//       ].filter(Boolean);
//       return {
//         city: data.locality || data.city || "",
//         state: data.principalSubdivision || "",
//         country: data.countryName || "",
//         full:
//           data.localityInfo?.administrative
//             ?.map((a) => a.name)
//             .filter(Boolean)
//             .join(", ") || parts.join(", "),
//         short: parts.join(", "),
//       };
//     } catch {
//       return null;
//     }
//   }, []);

//   // Face mesh overlay — rAF (no React state per frame) + lighter drawing
//   useEffect(() => {
//     if (!showCamera) return undefined;
//     const canvas = overlayCanvasRef.current;
//     if (!canvas) return undefined;
//     let rafId = 0;
//     const draw = () => {
//       const targetPts = overlayLandmarksRef.current;
//       const targetMesh = overlayMeshRef.current;
//       if (targetPts) {
//         overlayDisplayLandmarksRef.current = lerpPoints(
//           overlayDisplayLandmarksRef.current,
//           targetPts,
//           OVERLAY_LERP,
//         );
//       } else {
//         overlayDisplayLandmarksRef.current = null;
//       }
//       if (targetMesh?.length) {
//         overlayDisplayMeshRef.current = lerpPoints(
//           overlayDisplayMeshRef.current,
//           targetMesh,
//           OVERLAY_LERP,
//         );
//       } else {
//         overlayDisplayMeshRef.current = null;
//       }
//       const pts = overlayDisplayLandmarksRef.current;
//       const mesh = overlayDisplayMeshRef.current;
//       const ctx = canvas.getContext("2d");
//       if (!ctx) {
//         rafId = requestAnimationFrame(draw);
//         return;
//       }
//       if (canvas.width !== PROCESS_W) canvas.width = PROCESS_W;
//       if (canvas.height !== PROCESS_H) canvas.height = PROCESS_H;
//       ctx.clearRect(0, 0, PROCESS_W, PROCESS_H);

//       if (mesh && mesh.length > 0) {
//         ctx.beginPath();
//         ctx.strokeStyle = "rgba(0, 255, 170, 0.12)";
//         ctx.lineWidth = 0.5;
//         for (let i = 0; i < mesh.length; i += 12) {
//           const p1 = mesh[i];
//           for (let j = i + 1; j < Math.min(i + 24, mesh.length); j += 8) {
//             const p2 = mesh[j];
//             const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);
//             if (dist < 28) {
//               ctx.moveTo(p1.x, p1.y);
//               ctx.lineTo(p2.x, p2.y);
//             }
//           }
//         }
//         ctx.stroke();
//         ctx.fillStyle = "rgba(0, 255, 170, 0.35)";
//         for (let i = 0; i < mesh.length; i += 3) {
//           const p = mesh[i];
//           ctx.fillRect(p.x, p.y, 1, 1);
//         }
//       }

//       if (pts && pts.length >= 68) {
//         ctx.strokeStyle = "rgba(0, 255, 170, 0.85)";
//         ctx.lineWidth = 1.5;
//         for (const [a, b] of FACE_CONNECTIONS) {
//           if (!pts[a] || !pts[b]) continue;
//           ctx.beginPath();
//           ctx.moveTo(pts[a].x, pts[a].y);
//           ctx.lineTo(pts[b].x, pts[b].y);
//           ctx.stroke();
//         }
//         for (let i = 0; i < pts.length; i++) {
//           const p = pts[i];
//           ctx.beginPath();
//           ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
//           ctx.fillStyle = "rgba(0, 255, 170, 1)";
//           ctx.fill();
//           ctx.beginPath();
//           ctx.arc(p.x, p.y, 0.9, 0, Math.PI * 2);
//           ctx.fillStyle = "#fff";
//           ctx.fill();
//         }
//       }
//       rafId = requestAnimationFrame(draw);
//     };
//     rafId = requestAnimationFrame(draw);
//     return () => cancelAnimationFrame(rafId);
//   }, [showCamera]);

//   useEffect(() => {
//     const video = videoRef.current;
//     if (!video || !stream) return;

//     video.srcObject = stream;

//     const handlePlay = () => {
//       clearCameraTimers();
//       gesturePrepDoneRef.current = false;
//       setGesturePrepActive(false);
//       let remaining = INIT_COUNTDOWN_SEC;
//       streamFrameToBackend();
//       probeIntervalRef.current = setInterval(
//         streamFrameToBackend,
//         PROBE_FRAME_INTERVAL_MS,
//       );

//       initCountdownIntervalRef.current = setInterval(() => {
//         remaining -= 1;
//         if (remaining > 0) return;
//         clearInterval(initCountdownIntervalRef.current);
//         initCountdownIntervalRef.current = null;
//         if (probeIntervalRef.current) {
//           clearInterval(probeIntervalRef.current);
//           probeIntervalRef.current = null;
//         }
//         if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
//         streamFrameToBackend();
//         frameIntervalRef.current = setInterval(
//           streamFrameToBackend,
//           FRAME_INTERVAL_MS,
//         );
//       }, 1000);
//     };

//     video.onloadedmetadata = () => {
//       video
//         .play()
//         .then(() => {
//           console.log("Video playing successfully");
//           handlePlay();
//         })
//         .catch((e) => {
//           console.error("Video play failed:", e);
//           // Fallback: try playing again on user interaction if needed,
//           // but 'muted' should handle most cases.
//         });
//     };

//     return () => {
//       if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
//       clearCameraTimers();
//       video.onloadedmetadata = null;
//     };
//   }, [stream, clearCameraTimers]);

//   function startGesturePrepPause() {
//     if (gesturePrepDoneRef.current) return;
//     setGesturePrepActive(true);

//     // Capture liveness reference photo after a short delay to ensure camera is stable and user is looking straight
//     setTimeout(() => {
//       try {
//         const video = videoRef.current;
//         if (video && video.videoWidth > 0 && video.videoHeight > 0) {
//           const tempCanvas = document.createElement("canvas");
//           tempCanvas.width = video.videoWidth;
//           tempCanvas.height = video.videoHeight;
//           const ctx = tempCanvas.getContext("2d");
//           if (ctx) {
//             ctx.drawImage(video, 0, 0);
//             const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.95);
//             sessionStorage.setItem("liveness_ref_photo", dataUrl);
//             console.log("📸 Liveness reference photo captured and saved in sessionStorage.");
//           }
//         }
//       } catch (e) {
//         console.warn("Failed to capture liveness reference photo:", e);
//       }
//     }, 500);

//     if (frameIntervalRef.current) {
//       clearInterval(frameIntervalRef.current);
//       frameIntervalRef.current = null;
//     }
//     if (gesturePrepTimeoutRef.current) clearTimeout(gesturePrepTimeoutRef.current);
//     gesturePrepTimeoutRef.current = setTimeout(() => {
//       gesturePrepDoneRef.current = true;
//       setGesturePrepActive(false);
//       gesturePrepTimeoutRef.current = null;
//       if (livenessSessionIdRef.current) {
//         frameIntervalRef.current = setInterval(
//           streamFrameToBackend,
//           FRAME_INTERVAL_MS,
//         );
//       }
//     }, CHALLENGE_PREP_DELAY_MS);
//   }

//   async function handleBackendResponse(data) {
//     if (!data) return;

//     if (Array.isArray(data.landmarks) && data.landmarks.length >= 68) {
//       overlayLandmarksRef.current = data.landmarks.map((p) => ({
//         x: p.x,
//         y: p.y,
//       }));
//     } else if (data.mesh === null) {
//       overlayLandmarksRef.current = null;
//     }
//     if (Array.isArray(data.mesh) && data.mesh.length > 0) {
//       overlayMeshRef.current = data.mesh.map((p) => ({ x: p.x, y: p.y }));
//     } else if (data.mesh === null) {
//       overlayMeshRef.current = null;
//     }

//     // Security alerts — always process (never skip on hidden backend steps)
//     if (data.multi_person) {
//       setMultiPersonError(true);
//     } else {
//       setMultiPersonError(false);
//     }

//     const deviceNames = Array.isArray(data.devices_detected)
//       ? data.devices_detected.join(", ")
//       : "";
//     const deviceDetail = (data.detail || "").toLowerCase();
//     const isDeviceAlert =
//       data.is_suspicious &&
//       (deviceNames ||
//         deviceDetail.includes("device") ||
//         deviceDetail.includes("phone") ||
//         deviceDetail.includes("tablet") ||
//         deviceDetail.includes("laptop") ||
//         deviceDetail.includes("electronic"));

//     if (isDeviceAlert) {
//       const now = Date.now();
//       if (now - lastDeviceToastRef.current > DEVICE_TOAST_INTERVAL_MS) {
//         lastDeviceToastRef.current = now;
//         addToast(
//           data.detail ||
//           (deviceNames
//             ? `mobile phone detected — move away from camera`
//             : "Electronic device detected near camera"),
//           "warning",
//         );
//       }
//       setErrcount((prev) => prev + 10);
//     }

//     if (data.gesture_prep && !gesturePrepDoneRef.current) {
//       startGesturePrepPause();
//     }

//     const hiddenSteps = ["calibration", "depth", "light_challenge", "micro"];
//     if (data.step && data.step !== livenessStep) {
//       const prevStep = livenessStep;
//       setLivenessStep(data.step);

//       if (data.step === "gesture" && prevStep !== "gesture" && !gesturePrepDoneRef.current) {
//         startGesturePrepPause();
//       }

//       if (
//         data.step !== "idle" &&
//         data.step !== "camera" &&
//         !hiddenSteps.includes(data.step)
//       ) {
//         setToastStep(data.step);
//         setToastVisible(true);

//         if (prevStep !== "idle" && prevStep !== "camera") {
//           setCompletedSteps((prev) => [...new Set([...prev, prevStep])]);
//           if (prevStep === "gesture") addToast("Liveness Verified");
//         }
//       }
//     }

//     if (data.detail) setChallengeMsg(data.detail);

//     if (data.step === "gesture" && data.gesture_idx !== undefined) {
//       setChallengeIndex(data.gesture_idx);
//       const completed = [];
//       for (let i = 0; i < data.gesture_idx; i++) completed.push(true);
//       setCompletedChallenges(completed);
//     }

//     if (data.status === "verified" || data.step === "complete") {
//       if (!livenessCompletedRef.current) {
//         livenessCompletedRef.current = true;
//         setCompletedSteps([
//           "calibration",
//           "depth",
//           "light_challenge",
//           "micro",
//           "gesture",
//         ]);
//         addToast("Security Check Passed");
//         await completeSession();
//       }
//     }

//     if (data.status === "rejected" || data.status === "failed") {
//       if (data.status === "rejected") {
//         setErrcount((prev) => prev + 10);
//         addToast("Security Alert: Electronic device detected.", "error");
//         // Silently log rejection without stopping camera or showing modal
//         console.warn("Security rejection caught:", data.detail);
//       } else {
//         setError(data.detail || "Liveness check failed");
//       }
//     } else if (data.status === "processing") {
//       const d = data.detail || "";

//       // Increment errcount for suspicious activity (e.g. digital screen, identity mismatch)
//       if (data.is_suspicious) {
//         // Increment by 10 per event/frame as requested
//         setErrcount((prev) => prev + 10);

//         // Show small popup at top
//         const now = Date.now();
//         if (
//           now - lastToastTimeRef.current > DEVICE_TOAST_INTERVAL_MS &&
//           !isDeviceAlert
//         ) {
//           addToast(
//             data.detail ||
//             "Security Warning: Potential device or non-live media detected.",
//             "warning",
//           );
//           lastToastTimeRef.current = now;
//         }
//       }

//       if (
//         d.includes("blocked") ||
//         d.includes("too close") ||
//         d.includes("too far") ||
//         d.includes("No face")
//       ) {
//         setError(d);
//       } else {
//         setError(null);
//       }
//     }
//   }

//   async function streamFrameToBackend() {
//     if (
//       streamingRef.current ||
//       !videoRef.current ||
//       videoRef.current.readyState < 2 ||
//       !livenessSessionIdRef.current
//     )
//       return;
//     streamingRef.current = true;
//     try {
//       const video = videoRef.current;
//       const c = document.createElement("canvas");
//       c.width = PROCESS_W;
//       c.height = PROCESS_H;
//       const ctx2 = c.getContext("2d");
//       if (!ctx2) {
//         streamingRef.current = false;
//         return;
//       }
//       const vw = video.videoWidth;
//       const vh = video.videoHeight;
//       const { sx, sy, sw, sh } = getCoverSourceRect(
//         vw,
//         vh,
//         PROCESS_W,
//         PROCESS_H,
//       );
//       ctx2.drawImage(video, sx, sy, sw, sh, 0, 0, PROCESS_W, PROCESS_H);
//       const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
//       if (!blob) {
//         streamingRef.current = false;
//         return;
//       }
//       const fd = new FormData();
//       fd.append("session_id", livenessSessionIdRef.current);
//       fd.append("frame", blob, "frame.jpg");
//       const res = await fetch(`${API_URL}/liveness/frame`, {
//         method: "POST",
//         body: fd,
//       });
//       const data = await res.json();
//       await handleBackendResponse(data);
//     } catch (e) {
//       console.warn("Stream error:", e);
//     }
//     streamingRef.current = false;
//   }

//   async function completeSession() {
//     try {
//       const res = await fetch(`${API_URL}/liveness/session/complete`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ session_id: livenessSessionIdRef.current }),
//       });
//       const data = await res.json();
//       if (res.ok && data.ok) {
//         setCanMatch(true);
//         setLivenessLive(true);
//         setLivenessStep("capture");
//       }
//     } catch {
//       setError("Verification failed");
//     }
//   }

//   async function startCamera() {
//     stopCamera();
//     setError(null);
//     setFile(null);
//     setPreview(null);
//     setResults([]);
//     setPenaltyDetails([]);
//     setCaptureLiveFailure(null);
//     setLivenessLive(false);
//     setCanMatch(false);
//     setGeoData(null);
//     setGeoAddress(null);
//     setProgress(0);
//     setErrcount(0);
//     setCompletedSteps([]);
//     sessionStorage.removeItem("liveness_ref_photo");
//     setLivenessSessionLoading(true);
//     try {
//       let sessData;
//       try {
//         const sessRes = await fetch(`${API_URL}/liveness/session/start`, {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({
//             device_id: getOrCreateDeviceId(),
//             agent_label: userAgentLabel,
//           }),
//         });
//         if (!sessRes.ok) {
//           const errBody = await sessRes.text();
//           throw new Error(`Server error ${sessRes.status}: ${errBody}`);
//         }
//         sessData = await sessRes.json();
//       } catch (e) {
//         setError(`Session failed: ${e.message}`);
//         setLivenessSessionLoading(false);
//         return;
//       }

//       let mediaStream;
//       try {
//         const constraints = {
//           video: {
//             facingMode: { ideal: "user" },
//             width: { ideal: 640 },
//             height: { ideal: 480 },
//           },
//         };
//         mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
//       } catch (e) {
//         setError(`Camera access denied: ${e.message}`);
//         setLivenessSessionLoading(false);
//         return;
//       }

//       livenessSessionIdRef.current = sessData.session_id;
//       setSessionChallenges(sessData.gestures);
//       setStream(mediaStream);
//       setShowCamera(true);
//       setLivenessStep("camera");
//       setChallengeMsg("Waiting for gesture...");
//     } catch (e) {
//       setError(`Unexpected error: ${e.message}`);
//     } finally {
//       setLivenessSessionLoading(false);
//     }
//   }

//   function stopCamera() {
//     if (stream) stream.getTracks().forEach((t) => t.stop());
//     if (frameIntervalRef.current) {
//       clearInterval(frameIntervalRef.current);
//       frameIntervalRef.current = null;
//     }
//     clearCameraTimers();
//     setStream(null);
//     setShowCamera(false);

//     if (!canMatch) {
//       setLivenessLive(false);
//       livenessSessionIdRef.current = null;
//       livenessCompletedRef.current = false;
//       setLivenessStep("idle");
//     }

//     overlayLandmarksRef.current = null;
//     overlayMeshRef.current = null;
//     overlayDisplayLandmarksRef.current = null;
//     overlayDisplayMeshRef.current = null;
//     gesturePrepDoneRef.current = false;
//     setGesturePrepActive(false);
//     setMultiPersonError(false);
//     setChallengeMsg("");
//     setError(null);
//   }

//   const takeSelfie = () => {
//     console.log("📸 Capture button clicked");
//     const canvas = canvasRef.current;
//     const video = videoRef.current;
//     if (!canvas || !video) {
//       console.warn("Canvas or video ref missing");
//       return;
//     }

//     // 1. Capture the frame first
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     canvas.getContext("2d").drawImage(video, 0, 0);

//     canvas.toBlob(
//       (blob) => {
//         if (!blob) {
//           console.error("Failed to create blob from canvas");
//           setError("Capture failed: Could not process image.");
//           return;
//         }
//         console.log("✅ Blob created, triggering match...");
//         const f = new File([blob], "selfie.jpg", { type: "image/jpeg" });
//         const currentSessionId = livenessSessionIdRef.current;

//         setFile(f);
//         setPreview(URL.createObjectURL(f));

//         // 2. Stop camera and streaming immediately
//         stopCamera();

//         // 3. Trigger match with the saved session ID
//         handleMatch(f, currentSessionId);
//       },
//       "image/jpeg",
//       0.95,
//     );
//   };

//   const handleMatch = async (fileOverride = null, sessionIdOverride = null) => {
//     const fileToUse = fileOverride || file;
//     const sessionIdToUse = sessionIdOverride || livenessSessionIdRef.current;

//     console.log("🔍 Starting Match process", {
//       hasFile: !!fileToUse,
//       hasSession: !!sessionIdToUse,
//       canMatch,
//     });

//     // If we are overriding with a direct file from capture, we bypass the state-based canMatch
//     // because liveness is already verified to reach the capture button.
//     const isDirectCapture = !!fileOverride;

//     if (!fileToUse) {
//       setError("Please upload an image or take a selfie first.");
//       return;
//     }

//     // Liveness is recommended but no longer strictly required for uploaded files in the UI
//     // It remains mandatory for direct camera capture (which is handled by isDirectCapture being true)
//     // Allow matching if session is technically complete, even if not perfectly 'verified'
//     const isSessionComplete =
//       livenessStep === "complete" || livenessStep === "capture";

//     if (!canMatch && isDirectCapture && !isSessionComplete) {
//       setError(
//         "Liveness verification failed. Please try the camera flow again.",
//       );
//       return;
//     }

//     setLoading(true);
//     setError(null);
//     setResults([]);
//     setPenaltyDetails([]);
//     setCaptureLiveFailure(null);
//     setProgress(0);
//     setGeoError(null);

//     // Geo capture
//     console.log("📍 Capturing Geo...");
//     const geo = await captureGeo();
//     setGeoData(geo);
//     if (!geo) {
//       console.warn("Geo capture failed or denied");
//       setGeoError("⚠ Location unavailable.");
//     } else {
//       reverseGeocode(geo.lat, geo.long).then((addr) => {
//         console.log("🌍 Geo Address:", addr);
//         setGeoAddress(addr);
//       });
//     }

//     const matchAbort = new AbortController();
//     const matchTimeoutId = setTimeout(
//       () => matchAbort.abort(),
//       MATCH_REQUEST_TIMEOUT_MS,
//     );
//     const pInterval = startIndeterminateMatchProgress(setProgress);

//     const fd = new FormData();
//     fd.append("file", fileToUse);
//     fd.append("device_id", getOrCreateDeviceId());
//     if (sessionIdToUse) {
//       fd.append("liveness_session_id", sessionIdToUse);
//     }
//     if (geo) {
//       fd.append("geo_lat", geo.lat);
//       fd.append("geo_long", geo.long);
//       fd.append("geo_timestamp", geo.timestamp);
//     }

//     if (errcount > 0) {
//       fd.append("errcount", errcount);
//     }
//     if (userAgentLabel && isDirectCapture) {
//       fd.append("expected_label", userAgentLabel);
//     }
//     const livenessRefPhoto = sessionStorage.getItem("liveness_ref_photo");
//     if (livenessRefPhoto && isDirectCapture) {
//       fd.append("liveness_ref_photo", livenessRefPhoto);
//     }
//     try {
//       console.log(`📤 Sending match request to ${API_URL}/match ...`);
//       console.log("Data: ", fd);
//       const res = await fetch(`${API_URL}/match`, {
//         method: "POST",
//         body: fd,
//         signal: matchAbort.signal,
//       });
//       const data = await res.json();
//       clearInterval(pInterval);
//       clearTimeout(matchTimeoutId);

//       if (data.error) {
//         console.error("❌ Match error from backend:", data.error);
//         setError(data.error);
//         setCaptureLiveFailure(null);
//         setProgress(0);
//       } else {
//         console.log("✅ Match successful", data.matches?.length, "results");
//         setResults(data.matches || []);
//         setPenaltyDetails(data.security_penalty_breakdown || []);
//         if (data.capture_live_ok === false) {
//           setCaptureLiveFailure({
//             reason:
//               data.capture_live_reason ||
//               "Final capture did not pass live-face checks (screen, photo, or replay suspected).",
//             score:
//               typeof data.capture_live_score === "number"
//                 ? data.capture_live_score
//                 : null,
//           });
//         } else {
//           setCaptureLiveFailure(null);
//         }
//         if (data.processed_image) setProcessedPreview(data.processed_image);
//         if (data.captured_image) setCapturedImage(data.captured_image);
//         sessionStorage.removeItem("liveness_ref_photo");
//         setProgress(100);
//       }
//     } catch (err) {
//       console.error("❌ Match request failed:", err);
//       setError(matchFetchErrorMessage(err));
//       setProgress(0);
//       clearInterval(pInterval);
//       clearTimeout(matchTimeoutId);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const handleRegister = async () => {
//     if (!file || !registerName) {
//       setError("Please provide a name and capture a selfie first.");
//       return;
//     }
//     setLoading(true);
//     setError(null);
//     setRegistrationSuccess(null);

//     const fd = new FormData();
//     fd.append("file", file);
//     fd.append("name", registerName);
//     fd.append("liveness_session_id", currentSessionId);
//     fd.append("device_id", getOrCreateDeviceId());
//     fd.append("errcount", errcount);

//     try {
//       console.log(`📤 Sending registration request to ${API_URL}/register ...`);
//       const res = await fetch(`${API_URL}/register`, {
//         method: "POST",
//         body: fd,
//       });
//       const data = await res.json();
//       if (data.error) {
//         setError(data.error);
//       } else {
//         setRegistrationSuccess(`Successfully registered ${registerName}!`);
//         setRegisterMode(false);
//         setRegisterName("");
//         setCanMatch(false);
//         setLivenessStep("idle");
//       }
//     } catch (err) {
//       setError("Registration failed. Please try again.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   const handleReload = () => {
//     window.location.reload();
//   };

//   return (
//     <div className="fm-page">
//       {/* Toast Notification Pipeline */}
//       <div className="fm-toast-pipeline">
//         {toasts.map((t) => (
//           <div
//             key={t.id}
//             className={`fm-floating-toast ${t.type || "success"}`}
//           >
//             <div className="fm-toast-icon">
//               {t.type === "warning" ? (
//                 <AlertTriangle size={18} color="#ffbf01" />
//               ) : t.type === "error" ? (
//                 <AlertOctagon size={18} color="#ff4444" />
//               ) : (
//                 <UserCheck size={18} color="#24aa4d" />
//               )}
//             </div>
//             <span>{t.msg}</span>
//           </div>
//         ))}
//       </div>

//       <header className="fm-header-banner">
//         <div className="fm-header-left">
//           <div className="fm-header-text">
//             <span className="fm-demo-text">DEMO</span>
//             <h1>FACE BIOMETRICS</h1>
//             <p>FACE MATCH, LIVELINESS, DEEP FAKE & LOCATION</p>
//           </div>
//         </div>
//         <div className="fm-header-right">
//           <Power className="fm-power-btn" onClick={onLogout} size={28} />
//         </div>
//       </header>

//       <div className={`fm-main-layout ${showCamera ? "fm-camera-active" : ""}`}>
//         <div className="fm-left-col">
//           <div className="fm-camera-outer">
//             <div className="fm-camera-container">
//               <div className="fm-main-camera-contianer-relative">
//                 <video
//                   ref={videoRef}
//                   autoPlay
//                   playsInline
//                   muted
//                   className="fm-camera-feed"
//                 />
//                 <canvas
//                   ref={overlayCanvasRef}
//                   className="fm-mesh-overlay"
//                 />
//                 <canvas ref={canvasRef} style={{ display: "none" }} />

//                 {/* High-tech Viewfinder Corners */}
//                 <div className="fm-viewfinder-corner top-left"></div>
//                 <div className="fm-viewfinder-corner top-right"></div>
//                 <div className="fm-viewfinder-corner bottom-left"></div>
//                 <div className="fm-viewfinder-corner bottom-right"></div>

//                 {!showCamera && !loading && (
//                   <div className="fm-start-overlay">
//                     <button className="fm-start-btn" onClick={startCamera}>
//                       START <Play size={20} fill="currentColor" />
//                     </button>
//                   </div>
//                 )}

//                 {showCamera && (
//                   <div className="fm-scanline"></div>
//                 )}

//                 {showCamera && (
//                   <div className="fm-liveness-overlay">
//                     {/* Secure Scan State */}
//                     {["calibration", "depth", "light_challenge", "micro"].includes(livenessStep) && (
//                       <div className="fm-gesture-pill">
//                         <div className="fm-gesture-icon-wrap" style={{ background: "#24aa4d" }}>
//                           <Activity size={18} />
//                         </div>
//                         <span className="fm-gesture-text">Secure Scan...</span>
//                       </div>
//                     )}

//                     {gesturePrepActive && (
//                       <div className="fm-gesture-pill">
//                         <div className="fm-gesture-icon-wrap" style={{ background: "#16562a" }}>
//                           <Loader2 size={18} className="fm-security-init-spinner" />
//                         </div>
//                         <span className="fm-gesture-text">Preparing challenge…</span>
//                       </div>
//                     )}

//                     {/* Active Gesture Pill */}
//                     {livenessStep === "gesture" && !gesturePrepActive && sessionChallenges[challengeIndex] && (
//                       <div className="fm-gesture-pill-container">
//                         <div className="fm-gesture-pill">
//                           <div className="fm-gesture-icon-wrap">
//                             {(() => {
//                               const IconComp = CHALLENGE_UI[sessionChallenges[challengeIndex]]?.icon || Activity;
//                               return <IconComp size={18} />;
//                             })()}
//                           </div>
//                           <span className="fm-gesture-text">
//                             {CHALLENGE_UI[sessionChallenges[challengeIndex]]?.label}
//                           </span>
//                         </div>
//                       </div>
//                     )}
//                   </div>
//                 )}

//                 {/* Remove the old error overlay here because it goes in map now */}

//                 <div className="fm-camera-actions">
//                   {(livenessLive || livenessStep === "complete" || livenessStep === "capture") && !multiPersonError && (
//                     <button className="fm-capture-btn" onClick={takeSelfie}>
//                       <Camera size={18} /> Capture Selfie
//                     </button>
//                   )}
//                 </div>
//               </div>
//             </div>
//           </div>

//           <div className="fm-challenges-pills">
//             {[1, 2, 3].map(num => {
//               const isActive = challengeIndex + 1 === num && livenessStep === "gesture";
//               const isCompleted = challengeIndex >= num || livenessLive;
//               return (
//                 <div key={num} className={`fm-pill ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
//                   Challenge {num} <CheckCircle size={14} className="fm-pill-icon" />
//                 </div>
//               )
//             })}
//             <div className={`fm-pill ${livenessLive ? 'active' : ''}`}>
//               Capture Selfie <Camera size={14} className="fm-pill-icon" />
//             </div>
//           </div>
//         </div>

//         <div className="fm-right-col">
//           <div className={`fm-geo-card ${error ? 'fm-geo-error-active' : ''}`}>
//             {error && (
//               <div className="fm-map-error-overlay">
//                 <AlertTriangle size={20} /> {error}
//               </div>
//             )}
//             <div className="fm-geo-map">
//               {geoData ? (
//                 <MapContainer
//                   center={[parseFloat(geoData.lat), parseFloat(geoData.long)]}
//                   zoom={16}
//                   style={{ width: "100%", height: "100%" }}
//                   zoomControl={false}
//                   dragging={false}
//                   scrollWheelZoom={false}
//                   doubleClickZoom={false}
//                   attributionControl={false}
//                 >
//                   <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
//                   <Marker position={[parseFloat(geoData.lat), parseFloat(geoData.long)]} />
//                 </MapContainer>
//               ) : (
//                 <div className="fm-map-placeholder">
//                   <MapPin size={24} />
//                   <span>Map Ready</span>
//                 </div>
//               )}
//             </div>
//             <div className="fm-geo-details">
//               <div className="fm-geo-full-address">
//                 {geoAddress ? geoAddress.full : "Fetching location..."}
//               </div>
//               <div className="fm-geo-coords-row">
//                 <div>
//                   <span className="geo-label">Latitude</span> {geoData ? parseFloat(geoData.lat).toFixed(5) : "0.00000"}° N
//                 </div>
//                 <div>
//                   <span className="geo-label">Longitude</span> {geoData ? parseFloat(geoData.long).toFixed(5) : "0.00000"}° E
//                 </div>
//                 <div className="geo-time">
//                   {geoData ? new Date(geoData.timestamp).toLocaleString() : "Date / Time"}
//                 </div>
//               </div>
//             </div>
//           </div>

//           <div className="fm-matches-container">
//             <div className="fm-match-images">
//               <div className="fm-match-box">
//                 <div className="fm-match-label">#1 MATCHED (DB)</div>
//                 {results.length > 0 && results[0].label !== "txt" ? (
//                   <img src={results[0].matched_image || (results[0].images && results[0].images[0])} alt="DB" />
//                 ) : (
//                   <div className="fm-img-placeholder"></div>
//                 )}
//               </div>
//               <div className="fm-match-box">
//                 <div className="fm-match-label">CAPTURED (LIVE)</div>
//                 {capturedImage || preview ? (
//                   <img src={capturedImage || preview} alt="Live" />
//                 ) : (
//                   <div className="fm-img-placeholder"></div>
//                 )}
//               </div>
//             </div>
//             <div className="fm-score-container">
//               <div className="fm-score-header">
//                 <span>FACE MATCH SCORE</span> <Info size={14} className="fm-info-icon" />
//               </div>
//               <div className="fm-slider-track">
//                 <span className="fm-slider-label">Low</span>
//                 <div className="fm-slider-bar">
//                   <div className="fm-slider-fill" style={{ width: (results.length > 0 ? (results[0].confidence * 100) : 0) + '%' }}></div>
//                   <div className="fm-slider-thumb-wrapper" style={{ left: (results.length > 0 ? (results[0].confidence * 100) : 0) + '%' }}>
//                     <div className="fm-slider-thumb">
//                       {results.length > 0 ? Math.round(results[0].confidence * 100) : 0}%
//                     </div>
//                   </div>
//                 </div>
//                 <span className="fm-slider-label">High</span>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>
//       <div className="fm-footer-branding" style={{ bottom: 10, right: 20 }}>
//         <img src={bargadBranding} alt="Bargad" style={{ width: '120px' }} />
//       </div>
//     </div>
//   );
// }

import React, { useState, useRef, useEffect, useCallback } from "react";
import { getApiBase } from "./apiBase";
import { getCoverSourceRect } from "./cameraDrawUtils";
import {
  MATCH_REQUEST_TIMEOUT_MS,
  LIVENESS_FRAME_INTERVAL_MS,
  LIVENESS_PROBE_FRAME_INTERVAL_MS,
  createLivenessFrameLimiter,
  isReplayDeviceAlert,
  matchFetchErrorMessage,
  startIndeterminateMatchProgress,
} from "./matchUiUtils";
import {
  ERROR_LABELS,
  isDigitalMediaMessage,
  resolveSecurityDisplayError,
  resolveSecurityErrorLabel,
} from "./securityErrorMessages";
import {
  isRetryAllowed,
  parseApiDetail,
  parseJsonResponse,
  sanitizeUserMessage,
} from "./apiUtils";
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
  Loader2, Power, CheckCircle, Play,
  RefreshCcw,
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
/** Only block match when liveness accumulated substantial display-attack penalties. */
const MATCH_ERRCOUNT_BLOCK_THRESHOLD = 30;
const FRAME_INTERVAL_MS = LIVENESS_FRAME_INTERVAL_MS;
/** Selfie retries before full flow reset (banking PoC). */
const MATCH_MAX_SELFIE_RETRIES = 2;
/** Short countdown; probe frames run security checks before main stream. */
const INIT_COUNTDOWN_SEC = 1;
const CHALLENGE_PREP_DELAY_MS = 250;
const PROBE_FRAME_INTERVAL_MS = LIVENESS_PROBE_FRAME_INTERVAL_MS;
const DEVICE_TOAST_INTERVAL_MS = 1200;
const OVERLAY_LERP = 1.0;
const OVERLAY_STALE_MS = 1200;
const STREAM_JPEG_QUALITY = 0.72;
/** Must match JPEG sent to /liveness/frame (same crop as object-fit: cover in a 4:3 box). */
const PROCESS_W = 640;
const PROCESS_H = 480;
/** Face mesh / landmark skeleton during liveness (off for cleaner UX). */
const SHOW_FACE_MESH_OVERLAY = false;
/** Draw only an outer face border (no inner skeleton). */
const SHOW_FACE_OUTLINE_OVERLAY = true;
function lerpPoints(prev, next, t) {
  if (!next?.length) return null;
  if (!prev?.length || prev.length !== next.length) return next.map((p) => ({ ...p }));
  return next.map((p, i) => ({
    x: prev[i].x + (p.x - prev[i].x) * t,
    y: prev[i].y + (p.y - prev[i].y) * t,
  }));
}

// "Turn your Head Left"
// "Turn your Head Right"
// "Smile"
// "Open your mouth"
// "Move Closer"
// "Move Away"
// "Shake head left & right (NO)"
// "Look Up"
// "Look Down"

const CHALLENGE_UI = {
  turn_left: { label: "Turn your Head Left", icon: ArrowLeft },
  turn_right: { label: "Turn your Head Right", icon: ArrowRight },
  smile: { label: "Smile", icon: Smile },
  mouth_open: { label: "Open your mouth", icon: Smile },
  move_closer: { label: "Move Closer", icon: Maximize },
  move_farther: { label: "Move Away", icon: Maximize },
  shake_head: { label: "Shake head left & right (NO)", icon: Activity },
  look_up_hold: { label: "Look Up", icon: ArrowUp },
  look_down_hold: { label: "Slightly Look Down", icon: ArrowDown },
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
const FACE_OUTLINE_LOOP = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 0,
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

function SecurityAlertIcon({ size = 56 }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 437.39 437.39"
      width={size}
      height={size}
      fill="#ff0000"
      aria-hidden
    >
      <path d="M238.59,111.82c-4.16-6.94-11.57-11.1-19.89-11.1s-15.73,4.16-19.89,11.1l-98.08,163.31c-4.16,7.4-4.63,15.73-.46,23.13,4.16,7.4,11.57,11.57,20.36,11.57h195.7c8.33,0,16.19-4.16,20.36-11.57,4.16-7.4,4.16-16.19-.46-23.13l-97.62-163.31ZM218.7,282.54c-7.86,0-13.88-6.01-13.88-13.88s6.01-13.88,13.88-13.88,13.88,6.01,13.88,13.88-6.01,13.88-13.88,13.88ZM232.58,217.77c0,7.86-6.01,13.88-13.88,13.88s-13.88-6.01-13.88-13.88v-46.26c0-7.86,6.01-13.88,13.88-13.88s13.88,6.01,13.88,13.88v46.26Z" />
    </svg>
  );
}

const TASK_COMPLETE_SOUND = "/sounds/Task_complete.mp3";

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
  const [gesturePrepActive, setGesturePrepActive] = useState(false);
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
  /** Track if face match has been completed to remove start button */
  const [faceMatchCompleted, setFaceMatchCompleted] = useState(false);

  const [click, setClick] = useState(false);

  const addToast = useCallback((msg, type = "success") => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, msg, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  }, []);

  /** Maps to one of the 23 official security error labels (always visible in alert card). */
  const applySecurityError = useCallback((message, hints = {}) => {
    setError(resolveSecurityErrorLabel(message, hints));
  }, []);

  const resetAfterMatchFailure = useCallback(() => {
    setFaceMatchCompleted(false);
    setPreview(null);
    setFile(null);
    setCapturedImage(null);
    setProcessedPreview(null);
    setResults([]);
    setPenaltyDetails([]);
    setProgress(0);
    setClick(false);
    setCanMatch(false);
    setLivenessLive(false);
    livenessSessionIdRef.current = null;
    livenessCompletedRef.current = false;
    setLivenessStep("idle");
    setCompletedChallenges([]);
    setChallengeIndex(0);
    prevGestureIdxRef.current = 0;
    setShowCamera(false);
    sessionStorage.removeItem("liveness_ref_photo");
    // Keep security `error` visible until user taps RESTART; startCamera clears it.
  }, []);

  const matchRetriesRef = useRef(0);

  const handleMatchFailure = useCallback(
    (message, hints = {}) => {
      const safeMsg = sanitizeUserMessage(message, hints);
      const label = resolveSecurityErrorLabel(safeMsg, hints);
      const canRetry =
        hints.retryAllowed &&
        !hints.userMismatch &&
        matchRetriesRef.current < MATCH_MAX_SELFIE_RETRIES;

      if (canRetry) {
        matchRetriesRef.current += 1;
        setError(ERROR_LABELS.VERIFICATION_FAILED);
        addToast(
          `Please try again (${matchRetriesRef.current}/${MATCH_MAX_SELFIE_RETRIES}). Face the camera in steady lighting.`,
          "error",
        );
        setLoading(false);
        setProgress(0);
        return;
      }

      setError(label);
      const toastText =
        label === ERROR_LABELS.USER_MISMATCH
          ? "This face does not match the logged-in account."
          : safeMsg || label;
      addToast(toastText, "error");
      matchRetriesRef.current = 0;
      resetAfterMatchFailure();
    },
    [addToast, resetAfterMatchFailure],
  );

  const videoRef = useRef();
  const canvasRef = useRef();
  const overlayCanvasRef = useRef();
  const overlayLandmarksRef = useRef(null);
  const overlayMeshRef = useRef(null);
  const overlayDisplayLandmarksRef = useRef(null);
  const overlayDisplayMeshRef = useRef(null);
  const overlayLandmarksUpdatedAtRef = useRef(0);
  const overlayMeshUpdatedAtRef = useRef(0);
  const streamCanvasRef = useRef(null);
  const inputRef = useRef();
  const frameIntervalRef = useRef(null);
  const livenessSessionIdRef = useRef(null);
  const livenessCompletedRef = useRef(false);
  const prevGestureIdxRef = useRef(0);
  const lastSoundChallengeRef = useRef(-1);
  const audioCtxRef = useRef(null);
  const streamingRef = useRef(false);
  const profileMenuRef = useRef(null);
  const lastToastTimeRef = useRef(0);
  const lastDeviceToastRef = useRef(0);
  const probeIntervalRef = useRef(null);
  const gesturePrepTimeoutRef = useRef(null);
  const gesturePrepDoneRef = useRef(false);
  const initCountdownIntervalRef = useRef(null);
  const multiPersonErrorRef = useRef(false);
  const faceOutOfFrameRef = useRef(false);
  const frameLimiterRef = useRef(createLivenessFrameLimiter());

  const clearCameraTimers = useCallback(() => {
    if (initCountdownIntervalRef.current) {
      clearInterval(initCountdownIntervalRef.current);
      initCountdownIntervalRef.current = null;
    }
    if (probeIntervalRef.current) {
      clearInterval(probeIntervalRef.current);
      probeIntervalRef.current = null;
    }
    if (gesturePrepTimeoutRef.current) {
      clearTimeout(gesturePrepTimeoutRef.current);
      gesturePrepTimeoutRef.current = null;
    }
  }, []);

  const getAudioContext = useCallback(() => {
    if (typeof window === "undefined") return null;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioCtx();
    }
    if (audioCtxRef.current.state === "suspended") {
      audioCtxRef.current.resume().catch(() => { });
    }
    return audioCtxRef.current;
  }, []);

  const playSoundFromPath = useCallback(
    (path, volume = 0.8) => {
      if (typeof window === "undefined") return false;
      getAudioContext();
      try {
        const a = new Audio(path);
        a.volume = volume;
        a.preload = "auto";
        const p = a.play();
        if (p && typeof p.catch === "function") p.catch(() => { });
        return true;
      } catch {
        return false;
      }
    },
    [getAudioContext],
  );

  const playTaskCompleteSound = useCallback(() => {
    playSoundFromPath(TASK_COMPLETE_SOUND, 0.85);
  }, [playSoundFromPath]);

  const unlockAudio = useCallback(() => {
    getAudioContext();
    playSoundFromPath(TASK_COMPLETE_SOUND, 0.001);
  }, [getAudioContext, playSoundFromPath]);

  const playSoundsForChallengeComplete = useCallback(
    (completedChallengeIndex) => {
      if (completedChallengeIndex < 0) return;
      if (completedChallengeIndex <= lastSoundChallengeRef.current) return;
      lastSoundChallengeRef.current = completedChallengeIndex;
      playTaskCompleteSound();
    },
    [playTaskCompleteSound],
  );

  useEffect(() => {
    return () => {
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close().catch(() => { });
      }
    };
  }, []);

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
          // "Verification taking longer than usual. Please ensure you are a registered user.",
          "Verification taking longer than usual.",
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

  // Optional overlay: when mesh is off, show only a white outer face outline.
  useEffect(() => {
    if (!showCamera || SHOW_FACE_MESH_OVERLAY) return undefined;
    const canvas = overlayCanvasRef.current;
    if (!canvas) return undefined;
    let rafId = 0;
    const draw = () => {
      const nowMs = performance.now();
      const landmarksFresh =
        nowMs - (overlayLandmarksUpdatedAtRef.current || 0) <= OVERLAY_STALE_MS;
      const targetPts = overlayLandmarksRef.current;
      if (landmarksFresh && targetPts?.length) {
        overlayDisplayLandmarksRef.current = lerpPoints(
          overlayDisplayLandmarksRef.current,
          targetPts,
          OVERLAY_LERP,
        );
      } else {
        overlayDisplayLandmarksRef.current = null;
      }
      const pts = overlayDisplayLandmarksRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        if (canvas.width !== PROCESS_W) canvas.width = PROCESS_W;
        if (canvas.height !== PROCESS_H) canvas.height = PROCESS_H;
        ctx.clearRect(0, 0, PROCESS_W, PROCESS_H);
        if (SHOW_FACE_OUTLINE_OVERLAY && pts?.length >= 27) {
          ctx.beginPath();
          const first = pts[FACE_OUTLINE_LOOP[0]];
          if (first) {
            ctx.moveTo(first.x, first.y);
            for (let i = 1; i < FACE_OUTLINE_LOOP.length; i += 1) {
              const p = pts[FACE_OUTLINE_LOOP[i]];
              if (p) ctx.lineTo(p.x, p.y);
            }
            ctx.closePath();
            ctx.lineWidth = 2.4;
            ctx.strokeStyle = "#24aa4d";
            ctx.shadowColor = "transparent";
            ctx.shadowBlur = 0;
            ctx.stroke();
          }
        }
      }
      rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, [showCamera]);

  useEffect(() => {
    if (!showCamera || !SHOW_FACE_MESH_OVERLAY) return undefined;
    const canvas = overlayCanvasRef.current;
    if (!canvas) return undefined;
    let rafId = 0;
    const draw = () => {
      const nowMs = performance.now();
      const landmarksFresh =
        nowMs - (overlayLandmarksUpdatedAtRef.current || 0) <= OVERLAY_STALE_MS;
      const meshFresh = nowMs - (overlayMeshUpdatedAtRef.current || 0) <= OVERLAY_STALE_MS;
      const targetPts = overlayLandmarksRef.current;
      const targetMesh = overlayMeshRef.current;
      if (landmarksFresh && targetPts) {
        overlayDisplayLandmarksRef.current = lerpPoints(
          overlayDisplayLandmarksRef.current,
          targetPts,
          OVERLAY_LERP,
        );
      } else {
        overlayDisplayLandmarksRef.current = null;
      }
      if (meshFresh && targetMesh?.length) {
        overlayDisplayMeshRef.current = lerpPoints(
          overlayDisplayMeshRef.current,
          targetMesh,
          OVERLAY_LERP,
        );
      } else {
        overlayDisplayMeshRef.current = null;
      }
      const pts = overlayDisplayLandmarksRef.current;
      const mesh = overlayDisplayMeshRef.current;
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
      clearCameraTimers();
      gesturePrepDoneRef.current = false;
      setGesturePrepActive(false);
      let remaining = INIT_COUNTDOWN_SEC;
      streamFrameToBackend();
      probeIntervalRef.current = setInterval(
        streamFrameToBackend,
        PROBE_FRAME_INTERVAL_MS,
      );

      initCountdownIntervalRef.current = setInterval(() => {
        remaining -= 1;
        if (remaining > 0) return;
        clearInterval(initCountdownIntervalRef.current);
        initCountdownIntervalRef.current = null;
        if (probeIntervalRef.current) {
          clearInterval(probeIntervalRef.current);
          probeIntervalRef.current = null;
        }
        if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
        streamFrameToBackend();
        frameIntervalRef.current = setInterval(
          streamFrameToBackend,
          FRAME_INTERVAL_MS,
        );
      }, 1000);
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
      clearCameraTimers();
      video.onloadedmetadata = null;
    };
  }, [stream, clearCameraTimers]);

  function startGesturePrepPause() {
    if (gesturePrepDoneRef.current) return;
    gesturePrepDoneRef.current = true;
    setGesturePrepActive(true);

    // Brief UI only — keep streaming frames so gestures are not delayed
    setTimeout(() => {
      try {
        const video = videoRef.current;
        if (video && video.videoWidth > 0 && video.videoHeight > 0) {
          const tempCanvas = document.createElement("canvas");
          tempCanvas.width = video.videoWidth;
          tempCanvas.height = video.videoHeight;
          const ctx = tempCanvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(video, 0, 0);
            const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.95);
            sessionStorage.setItem("liveness_ref_photo", dataUrl);
          }
        }
      } catch (e) {
        console.warn("Failed to capture liveness reference photo:", e);
      }
    }, 200);

    if (gesturePrepTimeoutRef.current) clearTimeout(gesturePrepTimeoutRef.current);
    gesturePrepTimeoutRef.current = setTimeout(() => {
      setGesturePrepActive(false);
      gesturePrepTimeoutRef.current = null;
    }, CHALLENGE_PREP_DELAY_MS);
  }

  async function handleBackendResponse(data) {
    if (!data) return;
    const nowMs = performance.now();

    if (Array.isArray(data.landmarks) && data.landmarks.length >= 68) {
      overlayLandmarksRef.current = data.landmarks.map((p) => ({
        x: p.x,
        y: p.y,
      }));
      overlayLandmarksUpdatedAtRef.current = nowMs;
    }

    if (SHOW_FACE_MESH_OVERLAY) {
      if (Array.isArray(data.mesh) && data.mesh.length > 0) {
        overlayMeshRef.current = data.mesh.map((p) => ({ x: p.x, y: p.y }));
        overlayMeshUpdatedAtRef.current = nowMs;
      }
    }

    // Multi-person during liveness — show restart message and reset challenge pills.
    if (data.multi_person) {
      multiPersonErrorRef.current = true;
      setMultiPersonError(true);
      const restartMsg =
        data.detail ||
        "Multiple people detected during liveness. Gestures are restarting from Challenge 1 — only one person may complete the challenges.";
      setError(ERROR_LABELS.MULTI_PERSON);
      setChallengeMsg(restartMsg);
      applySecurityError(restartMsg, { multiPerson: true });
      if (data.gesture_reset && data.gesture_idx !== undefined) {
        prevGestureIdxRef.current = data.gesture_idx;
        setChallengeIndex(data.gesture_idx);
        setCompletedChallenges([]);
        lastSoundChallengeRef.current = -1;
      }
      setErrcount((prev) => prev + 5);
      return;
    }

    if (data.identity_mismatch) {
      const restartMsg =
        data.detail ||
        "Different person detected during liveness. Gestures are restarting from Challenge 1 — only the original user may complete the challenges.";
      applySecurityError(restartMsg, { userMismatch: true });
      setChallengeMsg(restartMsg);
      if (data.gesture_reset && data.gesture_idx !== undefined) {
        prevGestureIdxRef.current = data.gesture_idx;
        setChallengeIndex(data.gesture_idx);
        setCompletedChallenges([]);
        lastSoundChallengeRef.current = -1;
      }
      setErrcount((prev) => prev + 5);
      return;
    }

    if (data.face_out_of_frame) {
      faceOutOfFrameRef.current = true;
      setError(ERROR_LABELS.NO_FACE);
      setChallengeMsg(
        data.detail || "Keep your face fully inside the frame during the challenge",
      );
      setErrcount((prev) => prev + 2);
      return;
    }

    if (faceOutOfFrameRef.current) {
      faceOutOfFrameRef.current = false;
      setError((prev) =>
        prev === ERROR_LABELS.NO_FACE ? null : prev,
      );
    }

    if (multiPersonErrorRef.current) {
      multiPersonErrorRef.current = false;
      setMultiPersonError(false);
      setError((prev) =>
        prev === ERROR_LABELS.MULTI_PERSON || prev?.includes("Multiple Users")
          ? null
          : prev,
      );
      if (livenessCompletedRef.current) {
        setCanMatch(true);
        setLivenessLive(true);
      }
    }
    const isDeviceAlert = isReplayDeviceAlert(data);

    if (isDeviceAlert) {
      applySecurityError(data.detail, { digitalMedia: true });
      setErrcount((prev) => prev + 10);
      if (data.step === "gesture" && data.gesture_idx !== undefined) {
        prevGestureIdxRef.current = data.gesture_idx;
        setChallengeIndex(data.gesture_idx);
        const completed = [];
        for (let i = 0; i < data.gesture_idx; i++) completed.push(true);
        setCompletedChallenges(completed);
      }
      return;
    }

    if (data.gesture_prep && !gesturePrepDoneRef.current) {
      startGesturePrepPause();
    }

    const hiddenSteps = ["calibration", "depth", "light_challenge", "micro"];
    if (data.step && data.step !== livenessStep) {
      const prevStep = livenessStep;
      setLivenessStep(data.step);

      if (data.step === "gesture" && prevStep !== "gesture" && !gesturePrepDoneRef.current) {
        startGesturePrepPause();
      }

      if (
        data.step !== "idle" &&
        data.step !== "camera" &&
        !hiddenSteps.includes(data.step)
      ) {
        setToastStep(data.step);
        setToastVisible(true);

        if (prevStep !== "idle" && prevStep !== "camera") {
          setCompletedSteps((prev) => [...new Set([...prev, prevStep])]);
          if (prevStep === "gesture") addToast("Liveness Verified");
        }
      }
    }

    if (data.detail) setChallengeMsg(data.detail);

    if (data.step === "gesture" && data.gesture_idx !== undefined) {
      const prevIdx = prevGestureIdxRef.current;
      if (data.gesture_idx > prevIdx) {
        playSoundsForChallengeComplete(prevIdx);
      } else if (
        data.detail && 
        (data.detail.includes("Good! Next gesture") || data.detail.includes("Good! Next"))
      ) {
        const completedIdx = Math.max(0, (data.gesture_idx ?? 1) - 1);
        playSoundsForChallengeComplete(completedIdx);
      }
      prevGestureIdxRef.current = data.gesture_idx;
      setChallengeIndex(data.gesture_idx);
      const completed = [];
      for (let i = 0; i < data.gesture_idx; i++) completed.push(true);
      setCompletedChallenges(completed);
    }

    if (data.step === "complete" && data.status === "processing") {
      const lastIdx = sessionChallenges.length - 1;
      if (lastIdx >= 0) {
        playSoundsForChallengeComplete(lastIdx);
      }
      setChallengeIndex(sessionChallenges.length);
      setCompletedChallenges(sessionChallenges.map(() => true));
    }

    if (data.status === "verified") {
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
        applySecurityError(data.detail, { digitalMedia: true });
        console.warn("Security rejection caught:", data.detail);
      } else {
        applySecurityError(data.detail || "Liveness check failed");
      }
    } else if (data.status === "processing") {
      const d = data.detail || "";

      if (data.is_suspicious && data.display_attack && !isDeviceAlert) {
        setErrcount((prev) => prev + 3);
        // Do not flash security card mid-challenge — reduces false alarms in bright rooms.
      } else if (data.is_suspicious && !data.display_attack) {
        setErrcount((prev) => prev + 1);
      }

      if (typeof data.stream_risk === "number" && data.stream_risk > 55) {
        setErrcount((prev) => prev + 2);
      }

      if (d.includes("blocked") || d.includes("No face")) {
        applySecurityError(d);
      } else if (!data.is_suspicious) {
        setError(null);
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
    )
      return;
    streamingRef.current = true;
    try {
      const video = videoRef.current;
      const c = streamCanvasRef.current || document.createElement("canvas");
      streamCanvasRef.current = c;
      if (c.width !== PROCESS_W) c.width = PROCESS_W;
      if (c.height !== PROCESS_H) c.height = PROCESS_H;
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
      const blob = await new Promise((r) =>
        c.toBlob(r, "image/jpeg", STREAM_JPEG_QUALITY),
      );
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
      const data = await parseJsonResponse(res);
      if (!res.ok && data.error) {
        const msg = parseApiDetail(data, ERROR_LABELS.LIVENESS_FAILED);
        if (res.status === 429 || data.retry_allowed) {
          const pauseMs = frameLimiterRef.current.onRateLimited();
          console.warn(
            `Liveness frame rate limited — pausing ${pauseMs}ms:`,
            msg,
          );
        } else if (res.status === 400) {
          applySecurityError(msg);
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
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: livenessSessionIdRef.current }),
      });
      const data = await parseJsonResponse(res);
      if (res.ok && data.ok) {
        setCanMatch(true);
        setLivenessLive(true);
        setLivenessStep("capture");
      } else {
        const msg = parseApiDetail(
          data,
          "Liveness verification incomplete. Please restart and complete all challenges.",
        );
        applySecurityError(sanitizeUserMessage(msg));
        addToast(parseApiDetail(data, ERROR_LABELS.LIVENESS_INCOMPLETE), "error");
        livenessCompletedRef.current = false;
      }
    } catch {
      applySecurityError(ERROR_LABELS.VERIFICATION_FAILED);
    }
  }

  async function startCamera() {
    stopCamera();
    unlockAudio();
    frameLimiterRef.current.reset();
    prevGestureIdxRef.current = 0;
    lastSoundChallengeRef.current = -1;
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
    sessionStorage.removeItem("liveness_ref_photo");
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
        const sessJson = await parseJsonResponse(sessRes);
        if (!sessRes.ok) {
          applySecurityError(
            sanitizeUserMessage(parseApiDetail(sessJson, ERROR_LABELS.SESSION_FAILED)),
          );
          setLivenessSessionLoading(false);
          return;
        }
        sessData = sessJson;
      } catch (e) {
        applySecurityError(ERROR_LABELS.SESSION_FAILED);
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
        applySecurityError(`Camera access denied: ${e.message}`);
        setLivenessSessionLoading(false);
        return;
      }

      livenessSessionIdRef.current = sessData.session_id;
      setSessionChallenges(sessData.gestures);
      setStream(mediaStream);
      setShowCamera(true);
      setLivenessStep("camera");
      setChallengeIndex(0);
      setCompletedChallenges([]);
      setLivenessLive(false);
      setCanMatch(false);
      livenessCompletedRef.current = false;
      setFaceMatchCompleted(false);
      setChallengeMsg("Waiting for gesture...");
    } catch (e) {
      applySecurityError(`Unexpected error: ${e.message}`);
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
    clearCameraTimers();
    setStream(null);
    setShowCamera(false);

    if (!canMatch) {
      setLivenessLive(false);
      livenessSessionIdRef.current = null;
      livenessCompletedRef.current = false;
      setLivenessStep("idle");
    }

    overlayLandmarksRef.current = null;
    overlayMeshRef.current = null;
    overlayDisplayLandmarksRef.current = null;
    overlayDisplayMeshRef.current = null;
    gesturePrepDoneRef.current = false;
    setGesturePrepActive(false);
    multiPersonErrorRef.current = false;
    faceOutOfFrameRef.current = false;
    prevGestureIdxRef.current = 0;
    lastSoundChallengeRef.current = -1;
    frameLimiterRef.current.reset();
    setMultiPersonError(false);
    setChallengeMsg("");
  }

  const takeSelfie = () => {
    setClick(!click);
    console.log("📸 Capture button clicked");
    if (multiPersonError) {
      setError(ERROR_LABELS.MULTI_PERSON);
      return;
    }
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
          applySecurityError("Capture failed: Could not process image.");
          return;
        }
        console.log("✅ Blob created, triggering match...");
        playTaskCompleteSound();
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

    if (multiPersonError) {
      setError(ERROR_LABELS.MULTI_PERSON);
      return;
    }

    if (!fileToUse) {
      applySecurityError("Please upload an image or take a selfie first.");
      return;
    }

    // Strict liveness gating: no match is allowed without a verified session
    if (!canMatch && !faceMatchCompleted) {
      applySecurityError(
        "Liveness verification must be fully completed before matching. Please restart the camera flow.",
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
    if (userAgentLabel) {
      fd.append("expected_label", userAgentLabel);
    }
    const livenessRefPhoto = sessionStorage.getItem("liveness_ref_photo");
    if (livenessRefPhoto && isDirectCapture) {
      fd.append("liveness_ref_photo", livenessRefPhoto);
    }
    try {
      console.log(`📤 Sending match request to ${API_URL}/match ...`);
      console.log("Data: ", fd);
      const res = await fetch(`${API_URL}/match`, {
        method: "POST",
        body: fd,
        signal: matchAbort.signal,
      });
      const data = await parseJsonResponse(res);
      clearInterval(pInterval);
      clearTimeout(matchTimeoutId);

      if (data.error) {
        console.error("❌ Match error from backend:", data.error);
        const screenReplay =
          data.screen_replay === true || data.digital_media === true;
        const rawErr = parseApiDetail(data, data.error);
        const errText = screenReplay ? ERROR_LABELS.DIGITAL_MEDIA : rawErr;
        const userMismatch =
          !screenReplay &&
          (data.user_mismatch === true ||
            /user identity mismatch|registered|logged in|identity mismatch|does not match your/i.test(
              String(rawErr),
            ));
        const digitalMedia =
          screenReplay ||
          (data.security_verdict === "reject" &&
            !userMismatch &&
            (data.composite_risk ?? 0) >= 48);
        handleMatchFailure(errText, {
          userMismatch: screenReplay ? false : userMismatch,
          digitalMedia,
          retryAllowed: isRetryAllowed(data),
        });
        setCaptureLiveFailure(null);
        setProgress(0);
      } else if (data.security_verdict === "reject") {
        handleMatchFailure(
          parseApiDetail(
            data,
            "Security Alert: Digital screen or photo replay detected. Do not use a photograph or digital screen.",
          ),
          {
            digitalMedia: true,
            retryAllowed: isRetryAllowed(data),
          },
        );
        setCaptureLiveFailure(null);
        setProgress(0);
      } else if (
        data.capture_live_ok === false &&
        data.security_verdict !== "pass" &&
        data.security_verdict !== "pass_penalty"
      ) {
        console.error("❌ Capture live failed:", data.capture_live_reason);
        handleMatchFailure(
          data.capture_live_reason ||
          "Security Alert: High risk of digital spoofing detected. Matching blocked.",
          { digitalMedia: true },
        );
        setCaptureLiveFailure(null);
        setProgress(0);
      } else {
        console.log("✅ Match successful", data.matches?.length, "results");
        matchRetriesRef.current = 0;
        setResults(data.matches || []);
        setPenaltyDetails(data.security_penalty_breakdown || []);
        setCaptureLiveFailure(null);
        if (data.processed_image) setProcessedPreview(data.processed_image);
        if (data.captured_image) setCapturedImage(data.captured_image);
        sessionStorage.removeItem("liveness_ref_photo");
        setProgress(100);

        // Set face match as completed
        setFaceMatchCompleted(true);
      }
    } catch (err) {
      console.error("❌ Match request failed:", err);
      handleMatchFailure(matchFetchErrorMessage(err), {
        retryAllowed: true,
        networkError: true,
      });
      setProgress(0);
      clearInterval(pInterval);
      clearTimeout(matchTimeoutId);
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async () => {
    if (!file || !registerName) {
      applySecurityError("Please provide a name and capture a selfie first.");
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
        applySecurityError(data.error);
      } else {
        setRegistrationSuccess(`Successfully registered ${registerName}!`);
        setRegisterMode(false);
        setRegisterName("");
        setCanMatch(false);
        setLivenessStep("idle");
      }
    } catch (err) {
      applySecurityError("Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReload = () => {
    setError(null);
    setCaptureLiveFailure(null);
    setCompletedChallenges([]);
    setChallengeIndex(0);
    stopCamera();
    window.location.reload();
  };

  const securityAlertMessage = resolveSecurityDisplayError(error, {
    multiPerson: multiPersonError,
  });

  const isSecureScanPhase =
    showCamera &&
    ["calibration", "depth", "light_challenge", "micro"].includes(livenessStep);

  const setupStatusLabel =
    livenessStep === "calibration"
      ? "Calibrating face…"
      : livenessStep === "depth"
        ? "Checking depth — move head slightly"
        : livenessStep === "light_challenge"
          ? "Light check…"
          : livenessStep === "micro"
            ? "Almost ready…"
            : gesturePrepActive
              ? "Starting challenges…"
              : "We're setting up";

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
          <div className="fm-header-text">
            <span className="fm-demo-text">DEMO</span>
            <h1>FACE BIOMETRICS</h1>
            <p>FACE MATCH, LIVELINESS, DEEP FAKE & LOCATION</p>
          </div>
        </div>
        <div className="fm-header-right">
          <Power className="fm-power-btn" onClick={onLogout} size={28} />
        </div>
      </header>

      <div className={`fm-main-layout ${showCamera ? "fm-camera-active" : ""}`}>
        <div className="fm-layout-inner">
          <div className="fm-columns-row">
            <div className="fm-left-col">
              <div className="fm-camera-outer">
                <div
                  className={`fm-camera-container${isSecureScanPhase ? " fm-secure-scan-active" : ""}`}
                >
                  <div className="fm-main-camera-contianer-relative">
                    {
                      preview ?
                        <div className="fm-main-camera-contianer-relative">
                          <img
                            src={preview}
                            alt="Preview"
                            className="fm-camera-feed"
                          />
                          {loading && (
                            <div className="fm-scanline"></div>
                          )}
                        </div>
                        :
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          muted
                          className="fm-camera-feed"
                        />
                    }
                    <canvas
                      ref={overlayCanvasRef}
                      className="fm-mesh-overlay"
                    />
                    <canvas ref={canvasRef} style={{ display: "none" }} />

                    {/* High-tech Viewfinder Corners */}
                    <div className="fm-viewfinder-corner top-left"></div>
                    <div className="fm-viewfinder-corner top-right"></div>
                    <div className="fm-viewfinder-corner bottom-left"></div>
                    <div className="fm-viewfinder-corner bottom-right"></div>

                    {!showCamera && !loading && !preview && (
                      <div className="fm-start-overlay">
                        <button className="fm-start-btn" onClick={startCamera}>
                          START <Play size={20} fill="currentColor" />
                        </button>
                      </div>
                    )}

                    {!showCamera && !loading && preview && (
                      <div className="fm-start-overlay">
                        <button className="fm-start-btn" onClick={handleReload}>
                          RESTART
                          <RefreshCcw
                            size={20}
                            // fill="currentColor"
                            color="currentColor"
                          />
                          {/* RESTART LIVENESS SESSION <Play size={20} fill="currentColor" /> */}
                        </button>
                      </div>
                    )}

                    {showCamera && (
                      <div className="fm-scanline"></div>
                    )}

                    {isSecureScanPhase && (
                      <div className="fm-secure-scan-glow" aria-hidden="true" />
                    )}

                    {showCamera && (
                      <div className="fm-liveness-overlay">
                        {(isSecureScanPhase || gesturePrepActive) && (
                          <div className="fm-setup-status" role="status" aria-live="polite">
                            <Loader2 size={22} className="fm-setup-status-spinner" aria-hidden="true" />
                            <p className="fm-setup-status-title">{setupStatusLabel}</p>
                          </div>
                        )}

                        {/* Active Gesture Pill */}
                        {livenessStep === "gesture" && !gesturePrepActive && sessionChallenges[challengeIndex] && (
                          <div className="fm-gesture-pill-container">
                            <div className="fm-gesture-pill">
                              <div className="fm-gesture-icon-wrap">
                                {(() => {
                                  const IconComp = CHALLENGE_UI[sessionChallenges[challengeIndex]]?.icon || Activity;
                                  return <IconComp size={18} />;
                                })()}
                              </div>
                              <span className="fm-gesture-text">
                                {CHALLENGE_UI[sessionChallenges[challengeIndex]]?.label ||
                                  challengeMsg}
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}


                    <div className="fm-camera-actions">
                      {showCamera && (livenessLive || livenessStep === "complete" || livenessStep === "capture") && !multiPersonError && !faceMatchCompleted && (
                        <button className="fm-capture-btn" onClick={takeSelfie}>
                          <Camera size={18} /> Capture Selfie
                        </button>
                      )}
                    </div>
                    {/* {error && (
                  <div className="fm-map-error-overlay">
                    <AlertTriangle size={20} /> {error}
                  </div>
                )} */}
                  </div>
                </div>
              </div>

              <div
                className={`fm-challenges-pills${isSecureScanPhase ? " fm-challenges-pills--secure-scan" : ""}`}
              >
                {[1, 2, 3].map((num) => {
                  const isActive =
                    !isSecureScanPhase &&
                    challengeIndex + 1 === num &&
                    livenessStep === "gesture";
                  const isCompleted =
                    !isSecureScanPhase &&
                    !multiPersonError &&
                    challengeIndex >= num &&
                    ["gesture", "complete", "capture"].includes(livenessStep);
                  return (
                    <div
                      key={num}
                      className={`fm-pill${isSecureScanPhase ? " secure-scan" : ""}${isActive ? " active" : ""}${isCompleted ? " completed" : ""}`}
                    >
                      Challenge {num} <CheckCircle size={14} className="fm-pill-icon" />
                    </div>
                  );
                })}
                <div
                  className={`fm-pill${isSecureScanPhase ? " secure-scan" : ""}${!isSecureScanPhase && livenessLive && !multiPersonError ? " active" : ""}${click ? " completed" : ""}`}
                >
                  Capture Selfie <Camera size={14} className="fm-pill-icon" />
                </div>
              </div>
            </div>

            <div className="fm-right-col">
              {
                !loading && (faceMatchCompleted || securityAlertMessage) && (
                  <>
                    {securityAlertMessage ? (
                      <div className="fm-security-alert-card">
                        <SecurityAlertIcon size={56} />
                        <div className="fm-security-alert-text">
                          <span className="fm-alert-kicker">SECURITY ALERT</span>
                          <div className="error-font-fm-map-overlay">
                            {securityAlertMessage}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="fm-geo-card">
                          <div className="fm-geo-map">

                            {geoData ? (
                              <MapContainer
                                center={[parseFloat(geoData.lat), parseFloat(geoData.long)]}
                                zoom={16}
                                style={{ width: "100%", height: "100%" }}
                                zoomControl={false}
                                dragging={false}
                                scrollWheelZoom={false}
                                doubleClickZoom={false}
                                attributionControl={false}
                              >
                                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                                <Marker position={[parseFloat(geoData.lat), parseFloat(geoData.long)]} />
                              </MapContainer>
                            ) : (
                              <div className="fm-map-placeholder">
                                <MapPin size={24} />
                                <span>Map Ready</span>
                              </div>
                            )}
                          </div>
                          <div className="fm-geo-details">
                            <div className="fm-geo-full-address">
                              {geoAddress ? geoAddress.full : "Fetching location..."}
                            </div>
                            <div className="fm-geo-coords-row">
                              <div>
                                <span className="geo-label">Latitude</span> {geoData ? parseFloat(geoData.lat).toFixed(5) : "0.00000"}° N
                              </div>
                              <div>
                                <span className="geo-label">Longitude</span> {geoData ? parseFloat(geoData.long).toFixed(5) : "0.00000"}° E
                              </div>
                              <div className="geo-time">
                                {geoData ? new Date(geoData.timestamp).toLocaleString() : "Date / Time"}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="fm-matches-container">
                          <div className="fm-match-images">
                            <div className="fm-match-box">
                              <div className="fm-match-label">#1 MATCHED (DB)</div>
                              {!loading && results.length > 0 && results[0].label !== "txt" ? (
                                <img src={results[0].matched_image || (results[0].images && results[0].images[0])} alt="DB" />
                              ) : (
                                <div className="fm-img-placeholder"></div>
                              )}
                            </div>
                            <div className="fm-match-box">
                              <div className="fm-match-label">CAPTURED (LIVE)</div>
                              {capturedImage || preview ? (
                                <img src={capturedImage || preview} alt="Live" />
                              ) : (
                                <div className="fm-img-placeholder"></div>
                              )}
                              {loading && (
                                <div className="fm-scanline"></div>
                              )}
                            </div>
                          </div>
                          <div className="fm-score-container">
                            <div className="fm-score-header">
                              <span>FACE MATCH SCORE</span> <Info size={14} className="fm-info-icon" />
                            </div>
                            <div className="fm-slider-track">
                              <span className="fm-slider-label">Low</span>
                              <div className="fm-slider-bar">
                                <div className="fm-slider-fill" style={{ width: (results.length > 0 && !loading ? (results[0].confidence * 100) : 0) + '%' }}></div>
                                <div className="fm-slider-thumb-wrapper" style={{ left: (results.length > 0 && !loading ? (results[0].confidence * 100) : 0) + '%' }}>
                                  <div className="fm-slider-thumb">
                                    {results.length > 0 && !loading ? Math.round(results[0].confidence * 100) : 0}%
                                  </div>
                                </div>
                              </div>
                              <span className="fm-slider-label">High</span>
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </>
                )
              }
            </div>
          </div>
        </div>
      </div>
      <div className="fm-footer-branding">
        <img src={bargadBranding} alt="Bargad" />
      </div>
    </div >
  );
}