"""Phase 4 integration tests — caching, rate limiting, cost tracking, plugins in the client."""
from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from caip_responses.cost.tracker import ModelPricing
from caip_responses.models.common import Usage
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider
from caip_responses.ratelimit.limiter import RateLimitConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockProvider(BaseProvider):
    """Mock provider that returns a canned response."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name
        self.call_count = 0

    async def create_response(self, request: CreateResponseRequest) -> Response:
        self.call_count += 1
        return Response(
            id=f"resp_{self.call_count:03d}",
            model=request.model,
            status="completed",
            output=[
                {
                    "type": "message",
                    "id": "item_001",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
                    "status": "completed",
                }
            ],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        self.call_count += 1
        yield StreamEvent(type="response.created")  # type: ignore[call-arg]
        yield StreamEvent(  # type: ignore[call-arg]
            type="response.completed",
            response={
                "id": f"resp_{self.call_count:03d}",
                "model": request.model,
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    def supports_tool(self, tool_type: str) -> bool:
        return False

    def supports_reasoning(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return self._name


@pytest.fixture
def mock_provider():
    return _MockProvider(name="mock")


@pytest.fixture
async def client(mock_provider):
    """AsyncClient with a mock provider registered."""
    from caip_responses.client.async_client import AsyncClient

    client = AsyncClient(
        default_provider="mock",
        providers={"mock": mock_provider},
        discover_plugins=False,
    )
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# Cost tracking integration
# ---------------------------------------------------------------------------


class TestCostTrackingIntegration:
    async def test_cost_tracked_after_response(self, client, mock_provider):
        client.cost_tracker.set_pricing(
            "mock-model",
            ModelPricing(input_cost_per_million=2.0, output_cost_per_million=8.0),
        )

        await client.responses.create(model="mock-model", input="Hi")

        assert client.cost_tracker.total_requests == 1
        assert client.cost_tracker.total_cost > 0

    async def test_cost_zero_without_pricing(self, client):
        await client.responses.create(model="mock-model", input="Hi")
        assert client.cost_tracker.total_requests == 1
        assert client.cost_tracker.total_cost == 0.0

    async def test_cost_accumulates_across_requests(self, client):
        client.cost_tracker.set_pricing(
            "mock-model",
            ModelPricing(input_cost_per_million=2.0, output_cost_per_million=8.0),
        )
        await client.responses.create(model="mock-model", input="Hi")
        await client.responses.create(model="mock-model", input="Hello")

        assert client.cost_tracker.total_requests == 2
        tokens = client.cost_tracker.total_tokens
        assert tokens["input"] == 20
        assert tokens["output"] == 10


# ---------------------------------------------------------------------------
# Caching integration
# ---------------------------------------------------------------------------


class TestCachingIntegration:
    async def test_cache_hit_avoids_provider_call(self, client, mock_provider):
        # temperature=0 + non-streaming = cacheable
        await client.responses.create(model="mock-model", input="Hi", temperature=0)
        assert mock_provider.call_count == 1

        # Second identical call should hit cache
        resp2 = await client.responses.create(model="mock-model", input="Hi", temperature=0)
        assert mock_provider.call_count == 1  # No additional provider call
        assert resp2.output_text == "Hello!"

    async def test_different_input_no_cache_hit(self, client, mock_provider):
        await client.responses.create(model="mock-model", input="Hi", temperature=0)
        await client.responses.create(model="mock-model", input="Bye", temperature=0)
        assert mock_provider.call_count == 2

    async def test_non_zero_temp_not_cached(self, client, mock_provider):
        await client.responses.create(model="mock-model", input="Hi", temperature=0.7)
        await client.responses.create(model="mock-model", input="Hi", temperature=0.7)
        assert mock_provider.call_count == 2

    async def test_cache_disabled(self, mock_provider):
        from caip_responses.client.async_client import AsyncClient

        client = AsyncClient(
            default_provider="mock",
            providers={"mock": mock_provider},
            enable_cache=False,
            discover_plugins=False,
        )
        await client.responses.create(model="mock-model", input="Hi", temperature=0)
        await client.responses.create(model="mock-model", input="Hi", temperature=0)
        assert mock_provider.call_count == 2
        await client.close()

    async def test_cache_stats(self, client, mock_provider):
        await client.responses.create(model="mock-model", input="Hi", temperature=0)
        await client.responses.create(model="mock-model", input="Hi", temperature=0)
        stats = client.cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1  # first was a miss
        assert stats["size"] == 1


# ---------------------------------------------------------------------------
# Rate limiting integration
# ---------------------------------------------------------------------------


class TestRateLimitingIntegration:
    async def test_rate_limiter_accessible(self, client):
        assert client.rate_limiter is not None
        assert client.rate_limiter.configured_providers == []

    async def test_configure_and_request(self, client, mock_provider):
        client.rate_limiter.configure(
            "mock", RateLimitConfig(requests_per_minute=120)
        )
        # Should complete without rate limit issues
        resp = await client.responses.create(model="mock-model", input="Hi")
        assert resp.output_text == "Hello!"


# ---------------------------------------------------------------------------
# Plugin integration
# ---------------------------------------------------------------------------


class TestPluginIntegration:
    async def test_plugins_accessible(self, client):
        assert client.plugins is not None

    async def test_register_factory_via_plugins(self, client):
        new_provider = _MockProvider(name="custom")
        client.plugins.register_provider("custom", new_provider, prefixes=["custom-"])
        resp = await client.responses.create(model="custom-model", provider="custom", input="Hi")
        assert resp.output_text == "Hello!"

    async def test_discover_plugins_false_skips_discovery(self, mock_provider):
        """When discover_plugins=False, entry point discovery should not run."""
        from caip_responses.client.async_client import AsyncClient

        with patch("caip_responses.plugins.manager.importlib.metadata.entry_points") as mock_ep:
            client = AsyncClient(
                default_provider="mock",
                providers={"mock": mock_provider},
                discover_plugins=False,
            )
            mock_ep.assert_not_called()
            await client.close()


# ---------------------------------------------------------------------------
# Sync client Phase 4 properties
# ---------------------------------------------------------------------------


class TestSyncClientPhase4:
    def test_sync_client_has_phase4_properties(self):
        from caip_responses.client.sync_client import Client

        mock_prov = _MockProvider(name="mock")
        client = Client(
            default_provider="mock",
            providers={"mock": mock_prov},
            discover_plugins=False,
        )
        assert client.rate_limiter is not None
        assert client.cost_tracker is not None
        assert client.cache is not None
        assert client.plugins is not None
        client.close()

    def test_sync_client_cache_params(self):
        from caip_responses.client.sync_client import Client

        mock_prov = _MockProvider(name="mock")
        client = Client(
            default_provider="mock",
            providers={"mock": mock_prov},
            cache_max_size=100,
            cache_ttl=60,
            enable_cache=False,
            discover_plugins=False,
        )
        assert client.cache.enabled is False
        client.close()
