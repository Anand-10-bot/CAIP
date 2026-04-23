"""Redis-backed response cache for production deployments.

Drop-in replacement for ``ResponseCache`` that persists cached responses
in Redis.  Supports multi-worker / multi-pod deployments and survives
server restarts.

Usage::

    from caip_responses.cache.redis_cache import RedisResponseCache
    cache = RedisResponseCache(redis_url="redis://localhost:6379/0")

Or let ``AsyncClient`` create it automatically::

    client = AsyncClient(redis_url="redis://localhost:6379/0", ...)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from caip_responses.models.response import Response


class RedisResponseCache:
    """Redis-backed response cache.

    API-compatible with ``ResponseCache`` so it can be used as a
    drop-in replacement.  Each cached response is stored as a JSON
    string keyed by ``caip:cache:{hash}``.

    Redis native TTL handles expiry automatically — no need for manual
    eviction.

    Args:
        redis_url: Redis connection string (e.g. ``redis://localhost:6379/0``).
        key_prefix: Prefix for all Redis keys.
        default_ttl: Default time-to-live in seconds (0 = no expiry).
        max_size: Not enforced by Redis — memory is managed by Redis config.
            Provided for API compat with ``ResponseCache``.
        enabled: Whether caching is active.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        key_prefix: str = "caip:cache:",
        max_size: int = 0,
        default_ttl: int = 3600,
        enabled: bool = True,
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "Redis support requires the 'redis' package. "
                "Install it with: pip install caip-responses-lib[redis]"
            ) from exc

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

    def _key(self, cache_key: str) -> str:
        return f"{self._prefix}{cache_key}"

    def build_key(self, **kwargs: Any) -> str:
        """Build a cache key from request parameters.

        Identical logic to ``ResponseCache.build_key`` — deterministic
        SHA-256 hash over content-affecting request fields.
        """
        key_data = {
            k: v for k, v in sorted(kwargs.items())
            if v is not None
            and k not in (
                "stream", "store", "metadata", "user",
                "background", "include", "provider",
            )
        }
        raw = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> Response | None:
        """Look up a cached response by key."""
        if not self._enabled:
            self._misses += 1
            return None

        raw = self._client.get(self._key(key))
        if raw is None:
            self._misses += 1
            return None

        self._hits += 1
        data = json.loads(raw)
        return Response(**data)

    def set(
        self, key: str, response: Response, ttl: int | None = None
    ) -> None:
        """Store a response in the cache."""
        if not self._enabled:
            return

        effective_ttl = ttl if ttl is not None else self._default_ttl
        value = json.dumps(response.model_dump(), default=str)
        redis_key = self._key(key)

        if effective_ttl > 0:
            self._client.setex(redis_key, effective_ttl, value)
        else:
            self._client.set(redis_key, value)

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        return self._client.delete(self._key(key)) > 0

    def clear(self) -> None:
        """Remove all cached entries (matching our prefix)."""
        cursor = 0
        while True:
            cursor, keys = self._client.scan(
                cursor=cursor, match=f"{self._prefix}*", count=100
            )
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break

    @property
    def size(self) -> int:
        """Approximate count of cached entries."""
        count = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(
                cursor=cursor, match=f"{self._prefix}*", count=100
            )
            count += len(keys)
            if cursor == 0:
                break
        return count

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def stats(self) -> dict[str, int | float]:
        """Cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate_pct": round(self._hits / total * 100, 1) if total > 0 else 0,
            "size": self.size,
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hits = 0
        self._misses = 0

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self._client.ping()
        except Exception:
            return False
