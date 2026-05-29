"""
Liveness Session Manager — in-memory session state for backend-driven liveness.
"""
import os
import uuid
import time
import secrets
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

# Must match frontend CHALLENGE_UI (FaceMatch.jsx / FaceRegister.jsx) — 3 picked per session.
ALL_GESTURE_IDS = [
    "turn_left",
    "turn_right",
    "smile",
    "mouth_open",
    "move_closer",
    "move_farther",
    "shake_head",
    "look_up_hold",
    "look_down_hold",
]

LIGHT_CHALLENGES = ["white_flash", "blue_flash", "green_flash", "brightness_up", "brightness_down"]

SESSION_TTL = 300  # 5 minutes max
CALIBRATION_FRAMES = int(os.getenv("LIVENESS_CALIBRATION_FRAMES", "3"))
GESTURE_COUNT_MIN = 3
GESTURE_COUNT_MAX = 3
GESTURE_COOLDOWN = float(os.getenv("LIVENESS_GESTURE_COOLDOWN_SEC", "0.10"))
GESTURE_INSTRUCTION_SEC = float(os.getenv("LIVENESS_GESTURE_INSTRUCTION_SEC", "0.45"))


@dataclass
class LivenessSession:
    session_id: str
    device_id: str
    gestures: List[str]
    created_at: float = field(default_factory=time.time)

    # Pipeline step: calibration -> depth -> light_challenge -> gesture -> complete
    step: str = "calibration"
    calibration_count: int = 0

    # Calibration baseline (populated after CALIBRATION_FRAMES)
    baseline: Optional[Dict[str, Any]] = None

    # Landmark history for depth/micro analysis (rolling buffer)
    landmark_history: List[Any] = field(default_factory=list)
    frame_count: int = 0

    # Depth estimation
    depth_scores: List[float] = field(default_factory=list)
    depth_passed: bool = False

    # Light challenge
    light_challenge_color: Optional[str] = None
    light_pre_frames: List[Any] = field(default_factory=list)
    light_post_frames: List[Any] = field(default_factory=list)
    light_passed: bool = False

    # Micro expression
    micro_variance_scores: List[float] = field(default_factory=list)
    micro_passed: bool = False

    # Gesture tracking
    current_gesture_idx: int = 0
    gesture_results: List[bool] = field(default_factory=list)
    gesture_sustain_count: int = 0
    gesture_instruction_time: Optional[float] = None
    is_transitioning: bool = False
    gesture_challenge_baseline: Optional[Dict[str, Any]] = None
    gesture_turn_peak: float = 0.0
    gesture_pitch_peak: float = 0.0

    # Timing
    reaction_times: List[float] = field(default_factory=list)

    # Device detection
    device_detected: bool = False
    device_class: Optional[str] = None
    replay_device_detected: bool = False

    # Black screen
    black_screen_count: int = 0
    no_face_streak: int = 0
    multi_face_streak: int = 0
    face_out_of_frame_streak: int = 0
    identity_mismatch_streak: int = 0

    # FaceNet embedding of user at calibration — blocks face-swap mid-session
    liveness_identity_embedding: Optional[Any] = None

    # Shake head tracking
    shake_history: List[float] = field(default_factory=list)
    shake_completed: bool = False

    # Blink tracking
    blink_count: int = 0
    was_blink_closed: bool = False

    # Hold gesture tracking
    hold_start_time: Optional[float] = None

    # Hard Anti-Screen / Emissive Light Buffers
    roi_luminance_history: List[float] = field(default_factory=list)
    row_variance_history: List[np.ndarray] = field(default_factory=list)
    last_face_bbox: Optional[Tuple[int, int, int, int]] = None

    # Weighted spoof scoring (temporal + motion)
    spoof_score_history: List[float] = field(default_factory=list)
    landmark_centroid_history: List[Tuple[float, float]] = field(default_factory=list)
    last_gray_small: Optional[np.ndarray] = None
    spoof_temporal_hits: int = 0
    replay_risk_ema: float = 0.0
    fraud_ema_history: List[float] = field(default_factory=list)

    # NEW: Cooldown recovery — prevents false spikes from cascading
    fraud_cooldown_frames: int = 0  # Counts frames since last low-score frame

    # NEW: Per-signal rolling history for debugging and analysis
    per_signal_history: Dict[str, List[float]] = field(default_factory=lambda: {
        "depth_parallax": [],
        "biological": [],
        "device_replay": [],
        "challenge": [],
        "texture_degraded": [],
        "moire": [],
        "reflection": [],
        "flicker": [],
    })

    # NEW: Fraud rejection reasons for audit trail
    fraud_rejection_reasons: List[str] = field(default_factory=list)

    # Agent Verification (New)
    agent_label: Optional[str] = None
    agent_embedding: Optional[np.ndarray] = None
    last_agent_check_time: float = 0.0
    digital_screen_fail_count: int = 0

    # In-memory gesture challenge snapshots (cleared after /match; never persisted)
    challenge_snapshots: List[Any] = field(default_factory=list)
    challenge_devices_seen: List[str] = field(default_factory=list)
    # Rolling stream risk EMA (0–100) from liveness frames; cleared with session
    stream_risk_ema: float = 0.0

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > SESSION_TTL

    @property
    def current_gesture(self) -> Optional[str]:
        if self.current_gesture_idx < len(self.gestures):
            return self.gestures[self.current_gesture_idx]
        return None

    @property
    def all_gestures_done(self) -> bool:
        return self.current_gesture_idx >= len(self.gestures)

    @property
    def progress_pct(self) -> int:
        total_steps = len(self.gestures) + 3  # cal + depth + light + gestures
        done = 0.0
        if self.step == "calibration":
            done += min(self.calibration_count / CALIBRATION_FRAMES, 1.0)
        else:
            done += 1.0
            if self.depth_passed or self.step == "light_challenge":
                done += 1.0
            if self.light_passed or self.step == "micro":
                done += 1.0
            done += self.current_gesture_idx
            
        return min(int(done / total_steps * 100), 100)

    def advance_gesture(self):
        now = time.time()
        if self.gesture_instruction_time:
            rt = now - self.gesture_instruction_time
            self.reaction_times.append(rt)
        self.current_gesture_idx += 1
        self.gesture_sustain_count = 0
        self.shake_history = []
        self.shake_completed = False
        self.blink_count = 0
        self.was_blink_closed = False
        self.hold_start_time = None
        self.gesture_challenge_baseline = None
        self.gesture_turn_peak = 0.0
        self.gesture_pitch_peak = 0.0
        self.face_out_of_frame_streak = 0
        self.gesture_instruction_time = time.time()
        self.is_transitioning = True  # Mark as transitioning
        if self.all_gestures_done:
            self.step = "complete"


class SessionManager:
    """Thread-safe in-memory session store with TTL cleanup."""

    def __init__(self):
        self._sessions: Dict[str, LivenessSession] = {}
        self._lock = threading.Lock()

    def create_session(self, device_id: str, agent_label: Optional[str] = None, agent_embedding: Optional[np.ndarray] = None) -> LivenessSession:
        rng = secrets.SystemRandom()
        ids = list(ALL_GESTURE_IDS)
        rng.shuffle(ids)
        count = rng.randint(GESTURE_COUNT_MIN, GESTURE_COUNT_MAX)
        gestures = ids[:count]

        session_id = str(uuid.uuid4())
        sess = LivenessSession(
            session_id=session_id, 
            device_id=device_id, 
            gestures=gestures,
            agent_label=agent_label,
            agent_embedding=agent_embedding
        )
        with self._lock:
            self._cleanup()
            self._sessions[session_id] = sess
        return sess

    def get(self, session_id: str) -> Optional[LivenessSession]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess and sess.expired:
                del self._sessions[session_id]
                return None
            return sess

    def remove(self, session_id: str):
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            if sess is not None:
                try:
                    from challenge_frame_verification import clear_challenge_snapshots
                    clear_challenge_snapshots(sess)
                except Exception:
                    pass

    def _cleanup(self):
        expired = [k for k, v in self._sessions.items() if v.expired]
        for k in expired:
            del self._sessions[k]


def _build_session_manager():
    """In-memory + optional Redis (see session_store.py)."""
    try:
        from session_store import HybridSessionManager

        return HybridSessionManager()
    except Exception:
        return SessionManager()


# Global singleton (Redis-backed when REDIS_URL is set)
session_manager = _build_session_manager()
