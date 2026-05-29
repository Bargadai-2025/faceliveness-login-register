"""
Lightweight in-memory rate limiting for PoC (per client IP).
Set RATE_LIMIT_ENABLED=0 to disable in local dev.
"""
from __future__ import annotations

import os
import time
import threading
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api_errors import user_error


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
RATE_LIMIT_WINDOW_SEC = _env_int("RATE_LIMIT_WINDOW_SEC", 60)
RATE_LIMIT_FRAME_MAX = _env_int("RATE_LIMIT_FRAME_MAX", 180)
RATE_LIMIT_MATCH_MAX = _env_int("RATE_LIMIT_MATCH_MAX", 15)
RATE_LIMIT_START_MAX = _env_int("RATE_LIMIT_START_MAX", 20)


class _SlidingWindow:
    def __init__(self):
        self._buckets: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_sec: float) -> bool:
        now = time.time()
        cutoff = now - window_sec
        with self._lock:
            hits = [t for t in self._buckets[key] if t > cutoff]
            if len(hits) >= limit:
                self._buckets[key] = hits
                return False
            hits.append(now)
            self._buckets[key] = hits
            return True


_window = _SlidingWindow()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    if request.client:
        return request.client.host[:64]
    return "unknown"


def _limit_for_path(path: str) -> Tuple[int, str]:
    if path.endswith("/liveness/frame"):
        return RATE_LIMIT_FRAME_MAX, "frame"
    if path.endswith("/match"):
        return RATE_LIMIT_MATCH_MAX, "match"
    if path.endswith("/liveness/session/start") or path.endswith("/liveness/session"):
        return RATE_LIMIT_START_MAX, "start"
    return 0, ""


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        if not RATE_LIMIT_ENABLED or request.method != "POST":
            return await call_next(request)

        path = request.url.path.rstrip("/")
        limit, _kind = _limit_for_path(path)
        if limit <= 0:
            return await call_next(request)

        ip = client_ip(request)
        key = f"{ip}:{path}"
        if not _window.allow(key, limit, float(RATE_LIMIT_WINDOW_SEC)):
            body = user_error("RATE_LIMITED", retry_allowed=True, http_status=429)
            status = int(body.pop("_http_status", 429))
            return JSONResponse(status_code=status, content=body)

        return await call_next(request)
