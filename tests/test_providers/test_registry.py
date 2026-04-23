from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from caip_responses.models.errors import ProviderNotFoundError
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider
from caip_responses.providers.registry import ProviderRegistry


class MockProvider(BaseProvider):
    """Minimal provider for testing."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_tool(self, tool_type: str) -> bool:
        return True

    def supports_reasoning(self) -> bool:
        return True

    async def create_response(self, request: CreateResponseRequest) -> Response:
        return Response(id="resp_mock", model=request.model, output=[])

    async def create_response_stream(self, request: CreateResponseRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="response.created")
        yield StreamEvent(type="response.completed")


class TestProviderRegistry:
    def test_register_and_get(self):
        reg = ProviderRegistry()
        provider = MockProvider("openai")
        reg.register("openai", provider)
        assert reg.get("openai") is provider

    def test_get_nonexistent(self):
        reg = ProviderRegistry()
        assert reg.get("openai") is None

    def test_resolve_by_prefix_gpt(self):
        reg = ProviderRegistry()
        reg.register("openai", MockProvider("openai"))
        provider = reg.resolve("gpt-4.1")
        assert provider.provider_name == "openai"

    def test_resolve_by_prefix_claude(self):
        reg = ProviderRegistry()
        reg.register("anthropic", MockProvider("anthropic"))
        provider = reg.resolve("claude-sonnet-4-20250514")
        assert provider.provider_name == "anthropic"

    def test_resolve_by_prefix_gemini(self):
        reg = ProviderRegistry()
        reg.register("gemini", MockProvider("gemini"))
        provider = reg.resolve("gemini-2.0-flash")
        assert provider.provider_name == "gemini"

    def test_resolve_by_prefix_sarvam(self):
        reg = ProviderRegistry()
        reg.register("sarvam", MockProvider("sarvam"))
        provider = reg.resolve("sarvam-30b")
        assert provider.provider_name == "sarvam"

    def test_resolve_by_prefix_o3(self):
        reg = ProviderRegistry()
        reg.register("openai", MockProvider("openai"))
        provider = reg.resolve("o3-mini")
        assert provider.provider_name == "openai"

    def test_resolve_explicit_provider(self):
        reg = ProviderRegistry()
        reg.register("anthropic", MockProvider("anthropic"))
        provider = reg.resolve("my-custom-model", explicit_provider="anthropic")
        assert provider.provider_name == "anthropic"

    def test_resolve_default_provider(self):
        reg = ProviderRegistry(default_provider="openai")
        reg.register("openai", MockProvider("openai"))
        provider = reg.resolve("unknown-model")
        assert provider.provider_name == "openai"

    def test_resolve_not_found(self):
        reg = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError) as exc_info:
            reg.resolve("unknown-model")
        assert "unknown-model" in str(exc_info.value)

    def test_resolve_prefix_no_provider_registered(self):
        reg = ProviderRegistry()
        # gpt- prefix matches but no openai provider registered
        with pytest.raises(ProviderNotFoundError):
            reg.resolve("gpt-4.1")

    def test_add_prefix_mapping(self):
        reg = ProviderRegistry()
        reg.register("custom", MockProvider("custom"))
        reg.add_prefix_mapping("llama-", "custom")
        provider = reg.resolve("llama-3.1")
        assert provider.provider_name == "custom"

    def test_registered_providers(self):
        reg = ProviderRegistry()
        reg.register("openai", MockProvider("openai"))
        reg.register("anthropic", MockProvider("anthropic"))
        assert sorted(reg.registered_providers) == ["anthropic", "openai"]
