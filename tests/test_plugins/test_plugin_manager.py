from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.plugins.manager import PluginManager
from caip_responses.providers.base import BaseProvider
from caip_responses.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProvider(BaseProvider):
    """Minimal concrete provider for testing."""

    async def create_response(self, request: CreateResponseRequest) -> Response:
        return Response(id="resp_fake", model="fake-model")

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="response.completed")  # type: ignore[call-arg]

    def supports_tool(self, tool_type: str) -> bool:
        return False

    def supports_reasoning(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return "fake"


def _fake_factory(**kwargs) -> _FakeProvider:
    return _FakeProvider()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPluginManagerInit:
    def test_empty_state(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)
        assert mgr.available_plugins == []

    def test_registered_providers_delegates(self):
        registry = ProviderRegistry()
        registry.register("openai", _FakeProvider())
        mgr = PluginManager(registry)
        assert "openai" in mgr.registered_providers


class TestRegisterFactory:
    def test_register_and_list(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)
        mgr.register_factory("my_llm", _fake_factory, prefixes=["mymodel-"])
        assert "my_llm" in mgr.available_plugins

    def test_instantiate_creates_provider(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)
        mgr.register_factory("my_llm", _fake_factory, prefixes=["mymodel-"])

        provider = mgr.instantiate("my_llm")
        assert provider is not None
        assert isinstance(provider, _FakeProvider)
        assert "my_llm" in registry.registered_providers

    def test_instantiate_adds_prefix_mapping(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)
        mgr.register_factory("my_llm", _fake_factory, prefixes=["mymodel-"])
        mgr.instantiate("my_llm")

        # The prefix should now resolve
        resolved = registry.resolve("mymodel-test")
        assert isinstance(resolved, _FakeProvider)

    def test_instantiate_unknown_returns_none(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)
        assert mgr.instantiate("nonexistent") is None

    def test_instantiate_factory_error_returns_none(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)

        def bad_factory(**kwargs):
            raise RuntimeError("boom")

        mgr.register_factory("bad", bad_factory)
        result = mgr.instantiate("bad")
        assert result is None


class TestRegisterProvider:
    def test_register_provider_directly(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)
        provider = _FakeProvider()
        mgr.register_provider("custom", provider, prefixes=["custom-"])

        assert "custom" in registry.registered_providers
        resolved = registry.resolve("custom-model")
        assert resolved is provider

    def test_register_provider_no_prefixes(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)
        provider = _FakeProvider()
        mgr.register_provider("custom", provider)

        assert "custom" in registry.registered_providers


class TestDiscoverEntryPoints:
    def test_discover_with_no_plugins(self):
        registry = ProviderRegistry()
        mgr = PluginManager(registry)
        discovered = mgr.discover_entry_points()
        # No real plugins installed, so should return empty
        assert isinstance(discovered, list)

    def test_discover_loads_entry_point(self):
        """Mock the entry_points to simulate a third-party plugin."""
        registry = ProviderRegistry()
        mgr = PluginManager(registry)

        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = _FakeProvider

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("caip_responses.plugins.manager.importlib.metadata.entry_points", return_value=mock_eps):
            discovered = mgr.discover_entry_points()

        assert "test_plugin" in discovered
        assert "test_plugin" in mgr.available_plugins

    def test_discover_handles_load_failure(self):
        """Entry point that fails to load should be skipped."""
        registry = ProviderRegistry()
        mgr = PluginManager(registry)

        mock_ep = MagicMock()
        mock_ep.name = "bad_plugin"
        mock_ep.load.side_effect = ImportError("missing dependency")

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("caip_responses.plugins.manager.importlib.metadata.entry_points", return_value=mock_eps):
            discovered = mgr.discover_entry_points()

        assert "bad_plugin" not in discovered

    def test_discover_handles_entry_points_error(self):
        """If entry_points() itself fails, discovery returns empty."""
        registry = ProviderRegistry()
        mgr = PluginManager(registry)

        with patch(
            "caip_responses.plugins.manager.importlib.metadata.entry_points",
            side_effect=Exception("broken"),
        ):
            discovered = mgr.discover_entry_points()

        assert discovered == []
