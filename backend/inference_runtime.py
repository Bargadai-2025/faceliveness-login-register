"""
Bounded concurrent ML inference — keeps FastAPI responsive under load.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_semaphore: asyncio.Semaphore | None = None


def _max_concurrent() -> int:
    try:
        n = int(os.getenv("MAX_CONCURRENT_INFERENCE", "3"))
    except (TypeError, ValueError):
        n = 3
    return max(1, min(n, 16))


def get_inference_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_max_concurrent())
    return _semaphore


def max_concurrent_inference() -> int:
    return _max_concurrent()


async def run_inference_limited(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run sync CPU/GPU work in a thread pool with a global concurrency cap."""
    async with get_inference_semaphore():
        return await asyncio.to_thread(fn, *args, **kwargs)
