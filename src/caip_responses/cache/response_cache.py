from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from caip_responses.models.response import Response


class ResponseCache:
    """In-memory LRU cache for LLM responses.

    Caches responses based on a hash of the request parameters (model,
    input, instructions, tools, temperature, etc.). Cache hits avoid
    redundant API calls for identical deterministic requests.

    Only non-streaming requests with temperature=0 (or explicitly cached
    requests) should be cached, since non-deterministic requests produce
    different outputs each time.

    Usage:
        cache = ResponseCache(max_size=500, default_ttl=3600)

        # Check for cached response
        key = cache.build_key(model="gpt-4.1", input="Hello", temperature=0)
        cached = cache.get(key)
        if cached:
            return cached

        # ... make API call ...
        cache.set(key, response)
    """

    def __init__(
        self,
        max_size: int = 500,
        default_ttl: int = 3600,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            max_size: Maximum number of cached responses.
            default_ttl: Default time-to-live in seconds (0 = no expiry).
            enabled: Whether caching is active.
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._enabled = enabled
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def build_key(self, **kwargs: Any) -> str:
        """Build a cache key from request parameters.

        Deterministic hash over the request fields. Only fields that
        affect the response content are included.
        """
        # Normalize: sort keys, convert to stable JSON
        key_data = {
            k: v for k, v in sorted(kwargs.items())
            if v is not None and k not in ("stream", "store", "metadata", "user", "background", "include", "provider")
        }
        raw = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> Response | None:
        """Look up a cached response by key.

        Returns None on miss or if the entry has expired.
        """
        if not self._enabled:
            self._misses += 1
            return None

        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if entry.ttl > 0 and (time.monotonic() - entry.created_at) > entry.ttl:
            del self._entries[key]
            self._misses += 1
            return None

        # LRU: move to end
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.response

    def set(self, key: str, response: Response, ttl: int | None = None) -> None:
        """Store a response in the cache.

        Args:
            key: Cache key (from build_key()).
            response: The Response to cache.
            ttl: Time-to-live in seconds. None uses default_ttl.
        """
        if not self._enabled:
            return

        effective_ttl = ttl if ttl is not None else self._default_ttl

        # Evict oldest if at capacity
        if len(self._entries) >= self._max_size and key not in self._entries:
            self._entries.popitem(last=False)

        self._entries[key] = _CacheEntry(
            response=response,
            created_at=time.monotonic(),
            ttl=effective_ttl,
        )
        self._entries.move_to_end(key)

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._entries)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def stats(self) -> dict[str, int]:
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


class _CacheEntry:
    """Internal cache entry."""

    __slots__ = ("response", "created_at", "ttl")

    def __init__(self, response: Response, created_at: float, ttl: int) -> None:
        self.response = response
        self.created_at = created_at
        self.ttl = ttl
