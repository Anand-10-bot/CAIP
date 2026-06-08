from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from caip_responses.models.request import CreateResponseRequest
from caip_responses.providers.ollama_provider import OllamaProvider
from caip_responses.providers.openai_compatible import OpenAICompatibleProvider


class TestOllamaProvider:
    """OllamaProvider is a thin OpenAI-compatible subclass — verify its
    identity and that it inherits the full Chat Completions translation."""

    def _make_provider(self) -> OllamaProvider:
        provider = OllamaProvider.__new__(OllamaProvider)
        provider._api_key = None
        provider._base_url = "http://localhost:11434/v1"
        provider._http = MagicMock()
        return provider

    def test_is_openai_compatible(self):
        assert issubclass(OllamaProvider, OpenAICompatibleProvider)

    def test_provider_name(self):
        assert self._make_provider().provider_name == "ollama"

    def test_default_base_url(self):
        assert OllamaProvider.DEFAULT_BASE_URL == "http://localhost:11434/v1"

    def test_supports_function_tool(self):
        provider = self._make_provider()
        assert provider.supports_tool("function") is True
        assert provider.supports_tool("web_search") is False

    def test_no_auth_header_without_key(self):
        provider = self._make_provider()
        headers = provider._build_headers()
        assert "Authorization" not in headers

    def test_auth_header_with_key(self):
        provider = self._make_provider()
        provider._api_key = "secret"
        headers = provider._build_headers()
        assert headers["Authorization"] == "Bearer secret"

    def test_inherits_input_translation(self):
        provider = self._make_provider()
        messages = provider._translate_input("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_inherits_payload_build(self):
        provider = self._make_provider()
        request = CreateResponseRequest(
            model="llama3.1",
            input="Hi",
            instructions="Be brief",
            max_output_tokens=256,
        )
        payload = provider._build_payload(request, stream=False)
        assert payload["model"] == "llama3.1"
        assert payload["max_tokens"] == 256
        assert payload["messages"][0] == {"role": "system", "content": "Be brief"}
        assert payload["messages"][1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_create_response_text(self):
        provider = self._make_provider()
        response_data = {
            "id": "chatcmpl-ollama",
            "created": 1700000000,
            "choices": [{
                "message": {"role": "assistant", "content": "Hi from Llama!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        provider._http.post = AsyncMock(return_value=mock_resp)

        request = CreateResponseRequest(model="llama3.1", input="Hello")
        response = await provider.create_response(request)

        assert response.output_text == "Hi from Llama!"
        assert response.usage.input_tokens == 4

    @pytest.mark.asyncio
    async def test_create_response_wraps_http_error(self):
        import httpx

        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_resp,
        )
        provider._http.post = AsyncMock(return_value=mock_resp)

        from caip_responses.models.errors import ProviderError
        request = CreateResponseRequest(model="llama3.1", input="Hello")

        with pytest.raises(ProviderError) as exc_info:
            await provider.create_response(request)

        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_function_call_response(self):
        provider = self._make_provider()
        data = {
            "id": "chatcmpl-fn",
            "created": 1700000000,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        response = provider._convert_response(data, "qwen2.5")
        assert response.has_function_calls is True
        assert response.function_calls[0].name == "get_weather"
        assert json.loads(response.function_calls[0].arguments) == {"city": "Pune"}


class TestOpenSourcePrefixRouting:
    """Open-source model prefixes resolve to the ollama provider."""

    def _registry_with_ollama(self):
        from caip_responses.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        provider = OllamaProvider.__new__(OllamaProvider)
        provider._api_key = None
        provider._base_url = "http://localhost:11434/v1"
        provider._http = MagicMock()
        registry.register("ollama", provider)
        return registry, provider

    @pytest.mark.parametrize(
        "model",
        [
            "llama3.1",
            "mistral",
            "mixtral:8x7b",
            "qwen2.5",
            "qwq",
            "gemma2",
            "phi3",
            "deepseek-r1",
            "ollama/custom-model",
        ],
    )
    def test_open_source_prefixes_route_to_ollama(self, model):
        registry, provider = self._registry_with_ollama()
        assert registry.resolve(model) is provider

    def test_gemini_not_captured_by_gemma(self):
        """A gemini- model must not be swallowed by the gemma prefix."""
        from caip_responses.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        assert registry._prefix_map["gemini-"] == "gemini"
        assert registry._prefix_map["gemma"] == "ollama"
