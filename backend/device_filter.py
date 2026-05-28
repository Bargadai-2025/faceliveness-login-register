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
    """Dampen device_replay when YOLO only saw the user's laptop or ambient background TV."""
    if hard_overlap:
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
    if dr >= near_face_threshold:
        return True, attack_devices
    if device_visible and not is_laptop_only_devices(devices_found):
        return True, attack_devices
    return False, attack_devices
