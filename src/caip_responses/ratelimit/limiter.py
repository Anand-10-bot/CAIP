from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for a provider's rate limit.

    Args:
        requests_per_minute: Maximum requests per minute (0 = unlimited).
        tokens_per_minute: Maximum tokens per minute (0 = unlimited).
        max_retries: Number of retry attempts on rate limit errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    requests_per_minute: int = 0
    tokens_per_minute: int = 0
    max_retries: int = 3
    retry_base_delay: float = 1.0


class _TokenBucket:
    """Token bucket for rate limiting.

    Allows `capacity` tokens. Refills at `rate` tokens per second.
    """

    def __init__(self, capacity: float, rate: float) -> None:
        self._capacity = capacity
        self._rate = rate  # tokens per second
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            Wait time in seconds (0 if tokens were available immediately).
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate how long to wait for enough tokens
            deficit = tokens - self._tokens
            wait_time = deficit / self._rate if self._rate > 0 else 0
            return wait_time

    async def wait_and_acquire(self, tokens: float = 1.0) -> None:
        """Wait until tokens are available, then consume them."""
        while True:
            wait_time = await self.acquire(tokens)
            if wait_time <= 0:
                return
            await asyncio.sleep(wait_time)
            # Re-check after sleeping (another caller might have consumed tokens)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available(self) -> float:
        """Current number of available tokens (without locking)."""
        self._refill()
        return self._tokens


class RateLimiter:
    """Per-provider rate limiter using token bucket algorithm.

    Manages separate request-rate and token-rate buckets for each provider.
    When limits are configured, calls to `acquire()` will wait until
    capacity is available.

    Usage:
        limiter = RateLimiter()
        limiter.configure("anthropic", RateLimitConfig(requests_per_minute=60))

        # Before making a request:
        await limiter.acquire("anthropic")

        # After getting a response (to track token usage):
        limiter.record_tokens("anthropic", input_tokens=100, output_tokens=50)
    """

    def __init__(self) -> None:
        self._request_buckets: dict[str, _TokenBucket] = {}
        self._token_buckets: dict[str, _TokenBucket] = {}
        self._configs: dict[str, RateLimitConfig] = {}

    def configure(self, provider: str, config: RateLimitConfig) -> None:
        """Configure rate limits for a provider.

        Args:
            provider: Provider name (e.g., "anthropic", "gemini").
            config: Rate limit configuration.
        """
        self._configs[provider] = config

        if config.requests_per_minute > 0:
            rpm = config.requests_per_minute
            self._request_buckets[provider] = _TokenBucket(
                capacity=rpm,
                rate=rpm / 60.0,
            )

        if config.tokens_per_minute > 0:
            tpm = config.tokens_per_minute
            self._token_buckets[provider] = _TokenBucket(
                capacity=tpm,
                rate=tpm / 60.0,
            )

    async def acquire(self, provider: str) -> None:
        """Wait until a request can proceed under the rate limit.

        Call this before making a provider API call.
        """
        bucket = self._request_buckets.get(provider)
        if bucket:
            await bucket.wait_and_acquire(1.0)

    async def acquire_tokens(self, provider: str, tokens: int) -> None:
        """Wait until token capacity is available.

        Call this to pre-reserve token capacity (estimated).
        """
        bucket = self._token_buckets.get(provider)
        if bucket and tokens > 0:
            await bucket.wait_and_acquire(float(tokens))

    def record_tokens(
        self, provider: str, *, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """Record actual token usage after a response.

        This is informational — the token bucket already regulated the
        request. This method can be used for monitoring.
        """
        # Token bucket already handles rate; this is a hook for future extensions
        pass

    def get_config(self, provider: str) -> RateLimitConfig | None:
        """Get the rate limit config for a provider."""
        return self._configs.get(provider)

    def is_configured(self, provider: str) -> bool:
        """Check if a provider has rate limits configured."""
        return provider in self._configs

    @property
    def configured_providers(self) -> list[str]:
        """List providers with rate limits configured."""
        return list(self._configs.keys())
