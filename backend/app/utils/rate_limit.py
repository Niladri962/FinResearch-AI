"""Fixed-window rate limiter backed by the shared cache (Redis or in-process)."""
from __future__ import annotations

import time

from app.utils.cache import Cache
from app.utils.errors import RateLimitError


class RateLimiter:
    def __init__(self, cache: Cache, *, limit_per_minute: int, enabled: bool = True) -> None:
        self._cache = cache
        self._limit = limit_per_minute
        self._enabled = enabled

    def check(self, client_id: str) -> None:
        if not self._enabled:
            return
        window = int(time.time() // 60)
        count = self._cache.incr(f"ratelimit:{client_id}:{window}", ttl_seconds=90)
        if count > self._limit:
            raise RateLimitError("Too many requests. Please retry in a minute.")
