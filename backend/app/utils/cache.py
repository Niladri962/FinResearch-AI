"""Small cache abstraction: Redis when configured, in-process TTL cache otherwise."""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


class Cache(ABC):
    backend: str = "abstract"

    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None: ...

    @abstractmethod
    def incr(self, key: str, ttl_seconds: int) -> int:
        """Atomically increment a counter that expires ``ttl_seconds`` after creation."""

    def healthy(self) -> bool:
        return True


class InMemoryCache(Cache):
    backend = "memory"

    def __init__(self, max_items: int = 5000) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._max_items = max_items

    def _evict(self, now: float) -> None:
        expired = [k for k, (exp, _) in self._data.items() if exp <= now]
        for k in expired:
            self._data.pop(k, None)
        while len(self._data) >= self._max_items:
            self._data.pop(next(iter(self._data)))

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            if item[0] <= time.monotonic():
                self._data.pop(key, None)
                return None
            return item[1]

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._data) >= self._max_items:
                self._evict(now)
            self._data[key] = (now + ttl_seconds, value)

    def incr(self, key: str, ttl_seconds: int) -> int:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] <= now:
                if len(self._data) >= self._max_items:
                    self._evict(now)
                self._data[key] = (now + ttl_seconds, 1)
                return 1
            self._data[key] = (item[0], item[1] + 1)
            return item[1] + 1


class RedisCache(Cache):
    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
        self._client.ping()

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        self._client.set(key, json.dumps(value), ex=ttl_seconds)

    def incr(self, key: str, ttl_seconds: int) -> int:
        pipe = self._client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl_seconds, nx=True)
        return int(pipe.execute()[0])

    def healthy(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False


class ResilientCache(Cache):
    """Wraps a remote cache so an outage degrades to a cache miss, never an error."""

    def __init__(self, primary: Cache) -> None:
        self._primary = primary
        self._fallback = InMemoryCache()
        self.backend = primary.backend

    def get(self, key: str) -> Any | None:
        try:
            return self._primary.get(key)
        except Exception:
            return self._fallback.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        try:
            self._primary.set(key, value, ttl_seconds)
        except Exception:
            self._fallback.set(key, value, ttl_seconds)

    def incr(self, key: str, ttl_seconds: int) -> int:
        try:
            return self._primary.incr(key, ttl_seconds)
        except Exception:
            return self._fallback.incr(key, ttl_seconds)

    def healthy(self) -> bool:
        return self._primary.healthy()


def build_cache(redis_url: str) -> Cache:
    if redis_url:
        try:
            return ResilientCache(RedisCache(redis_url))
        except Exception as exc:  # unreachable Redis must not stop the app
            logger.warning("Redis unavailable, using in-process cache", extra={"error": type(exc).__name__})
    return InMemoryCache()
