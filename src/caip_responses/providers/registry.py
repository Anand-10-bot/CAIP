from __future__ import annotations

from caip_responses.models.errors import ProviderNotFoundError
from caip_responses.providers.base import BaseProvider


class ProviderRegistry:
    """Maps model names to providers via prefix matching or explicit override.

    Resolution order:
    1. Explicit `provider` parameter in the request
    2. Model prefix matching (configurable)
    3. Default provider (if set)
    """

    DEFAULT_PREFIX_MAP: dict[str, str] = {
        "gpt-": "openai",
        "o1-": "openai",
        "o3-": "openai",
        "o4-": "openai",
        "claude-": "anthropic",
        "gemini-": "gemini",
        "sarvam-": "sarvam",
        # Open-source models served via an OpenAI-compatible endpoint
        # (Ollama, vLLM, LM Studio). Use "ollama/<model>" for anything else.
        "llama": "ollama",
        "mistral": "ollama",
        "mixtral": "ollama",
        "qwen": "ollama",
        "qwq": "ollama",
        "gemma": "ollama",
        "phi": "ollama",
        "deepseek": "ollama",
        "ollama/": "ollama",
    }

    def __init__(self, default_provider: str | None = None) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._prefix_map: dict[str, str] = dict(self.DEFAULT_PREFIX_MAP)
        self._default_provider = default_provider

    def register(self, name: str, provider: BaseProvider) -> None:
        """Register a provider under a given name."""
        self._providers[name] = provider

    def get(self, name: str) -> BaseProvider | None:
        """Get a provider by name, or None if not registered."""
        return self._providers.get(name)

    def resolve(
        self, model: str, explicit_provider: str | None = None
    ) -> BaseProvider:
        """Resolve a model name to a provider instance.

        Args:
            model: The model name (e.g., "gpt-4.1", "claude-sonnet-4-20250514")
            explicit_provider: If given, use this provider name directly.

        Returns:
            The matching BaseProvider instance.

        Raises:
            ProviderNotFoundError: If no provider can be resolved.
        """
        # 1. Explicit provider override
        if explicit_provider:
            provider = self._providers.get(explicit_provider)
            if provider:
                return provider
            raise ProviderNotFoundError(model)

        # 2. Model prefix matching
        for prefix, provider_name in self._prefix_map.items():
            if model.startswith(prefix):
                provider = self._providers.get(provider_name)
                if provider:
                    return provider

        # 3. Default provider
        if self._default_provider:
            provider = self._providers.get(self._default_provider)
            if provider:
                return provider

        raise ProviderNotFoundError(model)

    def add_prefix_mapping(self, prefix: str, provider_name: str) -> None:
        """Add or override a model prefix → provider mapping."""
        self._prefix_map[prefix] = provider_name

    @property
    def registered_providers(self) -> list[str]:
        """List of registered provider names."""
        return list(self._providers.keys())
