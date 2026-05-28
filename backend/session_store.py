"""
Hybrid session store: in-memory (fast) + optional Redis persistence for PoC stability.
Falls back to memory-only when REDIS_URL is unset or Redis is unavailable.
"""
from __future__ import annotations

import os
import pickle
import threading
from typing import Optional

from liveness_session import LivenessSession, SessionManager, SESSION_TTL
from poc_logging import log_event

REDIS_URL = os.getenv("REDIS_URL", "").strip()
SESSION_REDIS_PREFIX = os.getenv("SESSION_REDIS_PREFIX", "liveness:sess:")
# Persist every N frames to reduce Redis load (still hot in memory)
SESSION_REDIS_SYNC_EVERY_N_FRAMES = int(os.getenv("SESSION_REDIS_SYNC_EVERY_N_FRAMES", "15"))


class HybridSessionManager(SessionManager):
    def __init__(self):
        super().__init__()
        self._redis = None
        self._redis_ok = False
        self._frame_sync_counter: dict = {}
        if REDIS_URL:
            try:
                import redis  # type: ignore

                self._redis = redis.from_url(REDIS_URL, decode_responses=False)
                self._redis.ping()
                self._redis_ok = True
                log_event("redis_session_store_ready", extra={"url_set": True})
            except Exception as e:
                log_event(
                    "redis_session_store_unavailable",
                    level="warning",
                    extra={"reason": str(e)[:120]},
                )
                self._redis = None

    def _key(self, session_id: str) -> bytes:
        return f"{SESSION_REDIS_PREFIX}{session_id}".encode("utf-8")

    def _persist(self, sess: LivenessSession, *, force: bool = False) -> None:
        if not self._redis_ok or not self._redis:
            return
        sid = sess.session_id
        if not force:
            n = self._frame_sync_counter.get(sid, 0) + 1
            self._frame_sync_counter[sid] = n
            if n % max(1, SESSION_REDIS_SYNC_EVERY_N_FRAMES) != 0:
                return
        try:
            self._redis.setex(self._key(sid), SESSION_TTL, pickle.dumps(sess, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:
            log_event("redis_session_persist_failed", level="warning", session_id=sid, extra={"reason": str(e)[:80]})

    def _load(self, session_id: str) -> Optional[LivenessSession]:
        if not self._redis_ok or not self._redis:
            return None
        try:
            raw = self._redis.get(self._key(session_id))
            if not raw:
                return None
            sess = pickle.loads(raw)
            if sess.expired:
                self._redis.delete(self._key(session_id))
                return None
            return sess
        except Exception as e:
            log_event("redis_session_load_failed", level="warning", session_id=session_id, extra={"reason": str(e)[:80]})
            return None

    def create_session(self, device_id: str, agent_label=None, agent_embedding=None) -> LivenessSession:
        sess = super().create_session(device_id, agent_label=agent_label, agent_embedding=agent_embedding)
        self._persist(sess, force=True)
        return sess

    def get(self, session_id: str) -> Optional[LivenessSession]:
        sess = super().get(session_id)
        if sess is not None:
            return sess
        restored = self._load(session_id)
        if restored is None:
            return None
        with self._lock:
            self._cleanup()
            self._sessions[session_id] = restored
        return restored

    def touch(self, sess: LivenessSession, *, force: bool = False) -> None:
        """Call after mutating session (e.g. each liveness frame)."""
        self._persist(sess, force=force)

    def remove(self, session_id: str):
        super().remove(session_id)
        self._frame_sync_counter.pop(session_id, None)
        if self._redis_ok and self._redis:
            try:
                self._redis.delete(self._key(session_id))
            except Exception:
                pass


def create_session_manager() -> HybridSessionManager:
    return HybridSessionManager()
