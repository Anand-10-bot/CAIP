from __future__ import annotations

import asyncio
import time

import pytest

from caip_responses.ratelimit.limiter import RateLimitConfig, RateLimiter, _TokenBucket

# ---------------------------------------------------------------------------
# _TokenBucket tests
# ---------------------------------------------------------------------------


class TestTokenBucket:
    @pytest.fixture
    def bucket(self):
        """Bucket with capacity=10, refill rate=10/sec."""
        return _TokenBucket(capacity=10.0, rate=10.0)

    async def test_acquire_immediate(self, bucket):
        wait = await bucket.acquire(1.0)
        assert wait == 0.0

    async def test_acquire_all_tokens(self, bucket):
        for _ in range(10):
            wait = await bucket.acquire(1.0)
            assert wait == 0.0

    async def test_acquire_returns_wait_when_empty(self, bucket):
        # Drain all tokens
        for _ in range(10):
            await bucket.acquire(1.0)
        # Next acquire should return positive wait time
        wait = await bucket.acquire(1.0)
        assert wait > 0

    async def test_wait_and_acquire(self):
        """Small bucket that refills quickly."""
        bucket = _TokenBucket(capacity=2.0, rate=100.0)
        # Drain
        await bucket.wait_and_acquire(2.0)
        # This should wait a tiny bit and succeed
        start = time.monotonic()
        await bucket.wait_and_acquire(1.0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # Should be very fast with rate=100/sec

    async def test_available_property(self, bucket):
        assert bucket.available == pytest.approx(10.0, abs=0.5)
        await bucket.acquire(5.0)
        assert bucket.available == pytest.approx(5.0, abs=0.5)

    async def test_refill_over_time(self):
        bucket = _TokenBucket(capacity=10.0, rate=1000.0)
        await bucket.acquire(10.0)
        # Wait a tiny bit for refill
        await asyncio.sleep(0.02)
        # Should have some tokens back (rate=1000/sec, so ~20 tokens in 0.02s)
        avail = bucket.available
        assert avail > 0


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_default_no_configs(self):
        limiter = RateLimiter()
        assert limiter.configured_providers == []

    def test_configure_provider(self):
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=60)
        limiter.configure("anthropic", config)
        assert limiter.is_configured("anthropic")
        assert limiter.get_config("anthropic") == config

    def test_not_configured(self):
        limiter = RateLimiter()
        assert not limiter.is_configured("openai")
        assert limiter.get_config("openai") is None

    async def test_acquire_unconfigured_no_wait(self):
        """Unconfigured providers should pass through immediately."""
        limiter = RateLimiter()
        start = time.monotonic()
        await limiter.acquire("openai")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    async def test_acquire_configured_allows_burst(self):
        """RPM-configured provider should allow burst up to capacity."""
        limiter = RateLimiter()
        limiter.configure("anthropic", RateLimitConfig(requests_per_minute=120))
        for _ in range(10):
            await limiter.acquire("anthropic")
        # Should complete quickly — well within the 120 RPM burst capacity

    async def test_acquire_tokens_unconfigured_no_wait(self):
        limiter = RateLimiter()
        await limiter.acquire_tokens("openai", 1000)

    async def test_acquire_tokens_configured(self):
        limiter = RateLimiter()
        limiter.configure("gemini", RateLimitConfig(tokens_per_minute=60000))
        await limiter.acquire_tokens("gemini", 100)

    def test_record_tokens_noop(self):
        """record_tokens is currently a hook — should not raise."""
        limiter = RateLimiter()
        limiter.record_tokens("openai", input_tokens=100, output_tokens=50)

    def test_configured_providers_list(self):
        limiter = RateLimiter()
        limiter.configure("a", RateLimitConfig(requests_per_minute=10))
        limiter.configure("b", RateLimitConfig(tokens_per_minute=5000))
        assert sorted(limiter.configured_providers) == ["a", "b"]


class TestRateLimitConfig:
    def test_defaults(self):
        config = RateLimitConfig()
        assert config.requests_per_minute == 0
        assert config.tokens_per_minute == 0
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0

    def test_custom_values(self):
        config = RateLimitConfig(
            requests_per_minute=120,
            tokens_per_minute=100000,
            max_retries=5,
            retry_base_delay=2.0,
        )
        assert config.requests_per_minute == 120
        assert config.tokens_per_minute == 100000
        assert config.max_retries == 5
        assert config.retry_base_delay == 2.0
