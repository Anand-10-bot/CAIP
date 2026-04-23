"""Redis-backed conversation store for production deployments.

Drop-in replacement for ``ConversationStore`` that persists conversation
histories in Redis.  Supports multi-worker / multi-pod deployments and
survives server restarts.

Usage::

    from caip_responses.store.redis_store import RedisConversationStore
    store = RedisConversationStore(redis_url="redis://localhost:6379/0")

Or let ``AsyncClient`` create it automatically::

    client = AsyncClient(redis_url="redis://localhost:6379/0", ...)
"""

from __future__ import annotations

import json
from typing import Any

from caip_responses.models.response import Response


class RedisConversationStore:
    """Redis-backed conversation store.

    API-compatible with ``ConversationStore`` so it can be used as a
    drop-in replacement.  Each conversation entry is stored as a JSON
    string keyed by ``caip:conv:{response_id}``.

    Args:
        redis_url: Redis connection string (e.g. ``redis://localhost:6379/0``).
        key_prefix: Prefix for all Redis keys.
        ttl: Time-to-live in seconds for each entry (0 = no expiry).
        max_size: Soft limit — entries are evicted via TTL, not LRU.
            Provided for API compat; Redis handles memory management.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        key_prefix: str = "caip:conv:",
        ttl: int = 86400,
        max_size: int = 0,
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
        self._ttl = ttl
        self._max_size = max_size  # kept for API compat

    def _key(self, response_id: str) -> str:
        return f"{self._prefix}{response_id}"

    def save(
        self,
        response: Response,
        input_items: list[dict[str, Any]],
        instructions: str | None = None,
    ) -> None:
        """Save a response and its input context."""
        # Serialize output items
        output_items = []
        for item in response.output:
            if isinstance(item, dict):
                output_items.append(item)
            elif hasattr(item, "model_dump"):
                output_items.append(item.model_dump())
            else:
                output_items.append(dict(item))

        entry = {
            "response_id": response.id,
            "input_items": input_items,
            "output_items": output_items,
            "instructions": instructions,
            "model": response.model,
        }

        key = self._key(response.id)
        value = json.dumps(entry, default=str)

        if self._ttl > 0:
            self._client.setex(key, self._ttl, value)
        else:
            self._client.set(key, value)

    def get_history(
        self, response_id: str
    ) -> tuple[list[dict[str, Any]], str | None] | None:
        """Reconstitute the full conversation history for a response ID."""
        key = self._key(response_id)
        raw = self._client.get(key)
        if raw is None:
            return None

        entry = json.loads(raw)

        history: list[dict[str, Any]] = []
        history.extend(entry["input_items"])
        history.extend(self._output_to_input(entry["output_items"]))

        return history, entry.get("instructions")

    def get_chain(
        self, response_id: str
    ) -> tuple[list[dict[str, Any]], str | None] | None:
        """Walk the full chain (same as get_history — history is built incrementally)."""
        return self.get_history(response_id)

    def has(self, response_id: str) -> bool:
        return self._client.exists(self._key(response_id)) > 0

    def remove(self, response_id: str) -> bool:
        return self._client.delete(self._key(response_id)) > 0

    def clear(self) -> None:
        """Remove all conversation entries (matching our prefix)."""
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
        """Approximate count of stored conversations."""
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

    @staticmethod
    def _output_to_input(
        output_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert response output items to input items for the next turn."""
        items: list[dict[str, Any]] = []
        for item_dict in output_items:
            item_type = item_dict.get("type")

            if item_type == "message":
                content = item_dict.get("content", [])
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    items.append({
                        "role": "assistant",
                        "content": "".join(text_parts),
                    })

            elif item_type == "function_call":
                items.append({
                    "type": "function_call",
                    "call_id": item_dict.get("call_id", ""),
                    "name": item_dict.get("name", ""),
                    "arguments": item_dict.get("arguments", "{}"),
                })

        return items

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self._client.ping()
        except Exception:
            return False
