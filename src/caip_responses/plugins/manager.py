from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Any

from caip_responses.providers.base import BaseProvider
from caip_responses.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

# Entry point group name — third-party packages declare providers here:
#   [project.entry-points."caip_responses.providers"]
#   my_provider = "my_package.provider:MyProvider"
ENTRY_POINT_GROUP = "caip_responses.providers"

# Type for a factory function: (api_key, **kwargs) -> BaseProvider
ProviderFactory = Callable[..., BaseProvider]


class PluginManager:
    """Discovers and registers custom LLM provider plugins.

    Supports two mechanisms:
    1. **Entry points** — Third-party packages declare providers via
       `[project.entry-points."caip_responses.providers"]` in pyproject.toml.
       These are discovered automatically at client init time.
    2. **Explicit registration** — Users can register provider factories
       or instances directly via `register_factory()` / `register_provider()`.

    Usage (entry points):
        # In third-party package's pyproject.toml:
        [project.entry-points."caip_responses.providers"]
        my_llm = "my_package.provider:MyLLMProvider"

        # Auto-discovered when client starts:
        client = AsyncClient(my_llm_api_key="...")

    Usage (explicit):
        plugin_mgr = client.plugins
        plugin_mgr.register_factory("my_llm", MyLLMProvider, prefixes=["mymodel-"])
        plugin_mgr.register_provider("custom", provider_instance, prefixes=["custom-"])
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry
        self._factories: dict[str, _PluginEntry] = {}

    def discover_entry_points(self) -> list[str]:
        """Discover and load provider plugins from entry points.

        Returns:
            List of discovered plugin names.
        """
        discovered: list[str] = []
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; .select() works on 3.9+
            if hasattr(eps, "select"):
                group_eps = eps.select(group=ENTRY_POINT_GROUP)
            else:
                group_eps = eps.get(ENTRY_POINT_GROUP, [])

            for ep in group_eps:
                try:
                    provider_cls = ep.load()
                    self._factories[ep.name] = _PluginEntry(
                        name=ep.name,
                        factory=provider_cls,
                        source="entry_point",
                    )
                    discovered.append(ep.name)
                    logger.debug("Discovered provider plugin: %s", ep.name)
                except Exception:
                    logger.warning(
                        "Failed to load provider plugin '%s'", ep.name, exc_info=True
                    )
        except Exception:
            logger.debug("Entry point discovery failed", exc_info=True)

        return discovered

    def register_factory(
        self,
        name: str,
        factory: ProviderFactory,
        *,
        prefixes: list[str] | None = None,
    ) -> None:
        """Register a provider factory for later instantiation.

        Args:
            name: Provider name (e.g., "ollama", "together").
            factory: Callable that creates a BaseProvider instance.
                     Typically a class: factory(api_key=..., base_url=...).
            prefixes: Model prefixes that route to this provider
                      (e.g., ["llama-", "mistral-"]).
        """
        self._factories[name] = _PluginEntry(
            name=name,
            factory=factory,
            source="explicit",
            prefixes=prefixes or [],
        )

    def register_provider(
        self,
        name: str,
        provider: BaseProvider,
        *,
        prefixes: list[str] | None = None,
    ) -> None:
        """Register an already-instantiated provider.

        Args:
            name: Provider name.
            provider: A BaseProvider instance.
            prefixes: Model prefixes that route to this provider.
        """
        self._registry.register(name, provider)
        if prefixes:
            for prefix in prefixes:
                self._registry.add_prefix_mapping(prefix, name)

    def instantiate(
        self, name: str, **kwargs: Any
    ) -> BaseProvider | None:
        """Instantiate a previously registered factory.

        Args:
            name: The plugin name.
            **kwargs: Passed to the factory (api_key, base_url, etc.).

        Returns:
            The provider instance, or None if not found.
        """
        entry = self._factories.get(name)
        if entry is None:
            return None

        try:
            provider = entry.factory(**kwargs)
            self._registry.register(name, provider)
            if entry.prefixes:
                for prefix in entry.prefixes:
                    self._registry.add_prefix_mapping(prefix, name)
            return provider
        except Exception:
            logger.warning("Failed to instantiate plugin '%s'", name, exc_info=True)
            return None

    @property
    def available_plugins(self) -> list[str]:
        """List all registered plugin names (factories not yet instantiated)."""
        return list(self._factories.keys())

    @property
    def registered_providers(self) -> list[str]:
        """List all active (instantiated) provider names."""
        return self._registry.registered_providers


class _PluginEntry:
    """Internal entry for a registered plugin."""

    __slots__ = ("name", "factory", "source", "prefixes")

    def __init__(
        self,
        name: str,
        factory: ProviderFactory,
        source: str,
        prefixes: list[str] | None = None,
    ) -> None:
        self.name = name
        self.factory = factory
        self.source = source
        self.prefixes = prefixes or []
