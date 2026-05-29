"""
Filter YOLO device detections for replay attacks vs ambient capture context.

Laptop built-in webcams often frame keyboard/bezel; YOLO labels "laptop" — that is NOT
a phone-on-screen replay attack and must not trigger digital-screen alerts.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

LAPTOP_DISPLAY_NAMES = frozenset({
    "laptop",
    "laptop / macbook",
    "macbook",
})

PHONE_TABLET_KEYWORDS = (
    "phone",
    "cell",
    "mobile",
    "tablet",
    "ipad",
)

# Background monitor / TV in frame — weak YOLO cue on laptop webcam; not phone-on-screen replay.
AMBIENT_TV_MONITOR_KEYWORDS = (
    "television",
    "tv",
    "monitor",
    "screen",
)


def is_phone_tablet_name(name: str) -> bool:
    n = _norm(name)
    return any(k in n for k in PHONE_TABLET_KEYWORDS)


def is_ambient_tv_monitor_name(name: str) -> bool:
    n = _norm(name)
    if is_phone_tablet_name(n) or is_laptop_display_name(n):
        return False
    return any(k in n for k in AMBIENT_TV_MONITOR_KEYWORDS)


def is_ambient_tv_only_devices(devices_found: Optional[List[str]]) -> bool:
    if not devices_found:
        return False
    return all(is_ambient_tv_monitor_name(d) for d in devices_found)


def is_phone_tablet_only_devices(devices_found: Optional[List[str]]) -> bool:
    if not devices_found:
        return False
    return all(is_phone_tablet_name(d) for d in devices_found)


def _norm(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def is_laptop_display_name(name: str) -> bool:
    n = _norm(name)
    return "laptop" in n or "macbook" in n


def is_replay_display_name(name: str) -> bool:
    """Phone / tablet / TV held up to camera — not the user's own laptop chassis."""
    n = _norm(name)
    if is_laptop_display_name(n):
        return False
    return is_phone_tablet_name(n) or is_ambient_tv_monitor_name(n)


REPLAY_DISPLAY_NAMES_KEYWORDS = PHONE_TABLET_KEYWORDS + AMBIENT_TV_MONITOR_KEYWORDS


def is_laptop_only_devices(devices_found: Optional[List[str]]) -> bool:
    if not devices_found:
        return False
    return all(is_laptop_display_name(d) for d in devices_found)


def phones_in_devices(devices_found: Optional[List[str]]) -> List[str]:
    return [d for d in (devices_found or []) if is_phone_tablet_name(d)]


def screen_physical_replay_cues(
    *,
    bezel_score: float = 0.0,
    screen_border_score: float = 0.0,
    moire: float = 0.0,
    pixel_grid: float = 0.0,
    fullframe_signals: Optional[List[str]] = None,
    replay_likelihood: float = 0.0,
) -> bool:
    """
    Physical phone/monitor frame or strong display combo — not ambient room light alone.
    Used at match and during liveness gestures when YOLO misses the phone class.
    """
    signals = set(fullframe_signals or [])
    if "phone_bezel" in signals or "screen_border" in signals:
        return True
    if float(bezel_score) >= 0.25 or float(screen_border_score) >= 0.22:
        return True
    if float(bezel_score) >= 0.18 and (
        float(moire) >= 0.22 or float(screen_border_score) >= 0.15
    ):
        return True
    if float(moire) >= 0.30 and float(pixel_grid) >= 0.22 and float(bezel_score) >= 0.12:
        return True
    if float(replay_likelihood) >= 0.48 and (
        "phone_bezel" in signals
        or "blur_plus_moire" in signals
        or float(bezel_score) >= 0.20
    ):
        return True
    return False


def filter_devices_for_attack(
    devices_found: Optional[List[str]],
    *,
    hard_overlap: bool = False,
    device_replay_score: float = 0.0,
) -> List[str]:
    """
    Devices that should count toward replay / digital-screen logic.
    Empty when only the capture laptop is visible (no face overlap hard block).
    Ambient TV/monitor in background requires high near-face score or hard overlap.
    """
    devices = list(devices_found or [])
    if not devices:
        return []
    if hard_overlap:
        if is_laptop_only_devices(devices):
            return []
        return devices
    if is_laptop_only_devices(devices):
        return []

    phone_tablet = [d for d in devices if is_phone_tablet_name(d)]
    if phone_tablet:
        return phone_tablet

    ambient = [d for d in devices if is_ambient_tv_monitor_name(d)]
    if ambient:
        if hard_overlap or float(device_replay_score) >= 0.58:
            return ambient
        return []

    replay = [d for d in devices if is_replay_display_name(d)]
    if replay:
        return replay
    return [d for d in devices if not is_laptop_display_name(d)]


def adjust_device_replay_score(
    device_replay_score: float,
    devices_found: Optional[List[str]],
    *,
    hard_overlap: bool = False,
) -> float:
    """Dampen device_replay for laptop-only / ambient TV — never dampen phone/tablet."""
    if hard_overlap:
        return float(device_replay_score)
    if phones_in_devices(devices_found):
        return float(device_replay_score)
    if is_laptop_only_devices(devices_found):
        return min(float(device_replay_score), 0.08)
    if is_ambient_tv_only_devices(devices_found) and not hard_overlap:
        return min(float(device_replay_score), 0.16)
    filtered = filter_devices_for_attack(
        devices_found, hard_overlap=False, device_replay_score=device_replay_score
    )
    if not filtered and devices_found:
        return min(float(device_replay_score), 0.12)
    return float(device_replay_score)


def should_raise_liveness_device_alert(
    devices_found: Optional[List[str]],
    device_replay_score: float,
    *,
    device_visible: bool = False,
    hard_overlap: bool = False,
    near_face_threshold: float = 0.18,
) -> Tuple[bool, List[str]]:
    """
    Whether to return an in-stream device security alert to the frontend.
    """
    if hard_overlap:
        return True, list(devices_found or [])
    attack_devices = filter_devices_for_attack(devices_found, hard_overlap=False)
    if not attack_devices:
        return False, []
    dr = adjust_device_replay_score(device_replay_score, devices_found, hard_overlap=False)
    phone_tablet = [d for d in attack_devices if is_phone_tablet_name(d)]
    if phone_tablet and (device_visible or dr >= 0.04):
        return True, phone_tablet
    if dr >= near_face_threshold:
        return True, attack_devices
    if device_visible and not is_laptop_only_devices(devices_found):
        return True, attack_devices
    return False, attack_devices


def hard_reject_phone_tablet_in_selfie(
    devices_found: Optional[List[str]],
    *,
    device_hard: bool = False,
    device_replay_score: float = 0.0,
    bezel_score: float = 0.0,
    screen_border_score: float = 0.0,
    moire: float = 0.0,
    pixel_grid: float = 0.0,
    fullframe_signals: Optional[List[str]] = None,
    replay_likelihood: float = 0.0,
) -> Tuple[bool, str]:
    """
    Hard reject POST /match when a phone/tablet is visible or physical screen frame detected.
    Laptop-only ambient context is ignored; phone/tablet is never dampened.
    """
    _msg = (
        "Security Alert: Digital screen or photo replay detected. "
        "Do not use a photograph or digital screen."
    )
    attack = filter_devices_for_attack(
        devices_found,
        hard_overlap=device_hard,
        device_replay_score=device_replay_score,
    )
    phones = phones_in_devices(devices_found) or [d for d in attack if is_phone_tablet_name(d)]
    if phones:
        names = ", ".join(phones)
        return (
            True,
            f"{_msg} Electronic device in view ({names}). "
            "Remove the phone or tablet and take a direct selfie.",
        )
    if device_hard and attack and not is_laptop_only_devices(attack):
        names = ", ".join(attack)
        return (
            True,
            f"{_msg} Electronic device blocking the face ({names}).",
        )
    if not is_laptop_only_devices(devices_found) and screen_physical_replay_cues(
        bezel_score=bezel_score,
        screen_border_score=screen_border_score,
        moire=moire,
        pixel_grid=pixel_grid,
        fullframe_signals=fullframe_signals,
        replay_likelihood=replay_likelihood,
    ):
        return (
            True,
            f"{_msg} Phone or monitor frame detected — take a direct selfie without a screen.",
        )
    if bezel_score >= 0.36 or screen_border_score >= 0.34:
        return True, _msg
    return False, ""
