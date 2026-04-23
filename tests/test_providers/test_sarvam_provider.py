from __future__ import annotations

import json
from unittest.mock import MagicMock

from caip_responses.providers.sarvam_provider import SarvamProvider


class TestSarvamTranslation:
    """Test the Sarvam provider's translation logic without live API calls."""

    def _make_provider(self) -> SarvamProvider:
        """Create a provider with a mocked HTTP client."""
        provider = SarvamProvider.__new__(SarvamProvider)
        provider._api_key = "test-key"
        provider._base_url = "https://api.sarvam.ai/v1"
        provider._http = MagicMock()
        return provider

    # ------------------------------------------------------------------
    # Input translation
    # ------------------------------------------------------------------

    def test_translate_input_string(self):
        provider = self._make_provider()
        messages = provider._translate_input("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_translate_input_messages(self):
        provider = self._make_provider()
        messages = provider._translate_input([
            {"role": "user", "content": "What's up?"},
            {"role": "assistant", "content": "Not much!"},
        ])
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "What's up?"}
        assert messages[1] == {"role": "assistant", "content": "Not much!"}

    def test_translate_input_with_function_call(self):
        provider = self._make_provider()
        messages = provider._translate_input([
            {"role": "user", "content": "Get weather"},
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "get_weather",
                "arguments": '{"city": "Mumbai"}',
            },
            {
                "type": "function_call_output",
                "call_id": "fc_1",
                "output": '{"temp": 35}',
            },
        ])
        # user, assistant with tool_calls, tool
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "fc_1"

    def test_translate_input_system_role(self):
        provider = self._make_provider()
        messages = provider._translate_input([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_translate_input_developer_role(self):
        provider = self._make_provider()
        messages = provider._translate_input([
            {"role": "developer", "content": "Instructions"},
            {"role": "user", "content": "Hi"},
        ])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_translate_input_content_blocks(self):
        provider = self._make_provider()
        messages = provider._translate_input([
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "World"},
                ],
            }
        ])
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello World"

    # ------------------------------------------------------------------
    # Tool translation
    # ------------------------------------------------------------------

    def test_translate_tools(self):
        provider = self._make_provider()
        tools = provider._translate_tools([
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            {"type": "web_search"},
        ])
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[0]["function"]["parameters"]["type"] == "object"

    def test_translate_tool_choice_auto(self):
        provider = self._make_provider()
        assert provider._translate_tool_choice("auto") == "auto"

    def test_translate_tool_choice_required(self):
        provider = self._make_provider()
        assert provider._translate_tool_choice("required") == "required"

    def test_translate_tool_choice_none(self):
        provider = self._make_provider()
        assert provider._translate_tool_choice("none") == "none"

    def test_translate_tool_choice_specific(self):
        provider = self._make_provider()
        result = provider._translate_tool_choice({"name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    # ------------------------------------------------------------------
    # Build payload
    # ------------------------------------------------------------------

    def test_build_payload_basic(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="sarvam-m",
            input="Hello",
            instructions="Be helpful",
            temperature=0.5,
            max_output_tokens=1024,
        )
        payload = provider._build_payload(request, stream=False)

        assert payload["model"] == "sarvam-m"
        assert payload["stream"] is False
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 1024
        # System message should be prepended
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "Be helpful"
        assert payload["messages"][1] == {"role": "user", "content": "Hello"}

    def test_build_payload_with_reasoning(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="sarvam-m",
            input="Hello",
            reasoning={"effort": "high"},
        )
        payload = provider._build_payload(request, stream=False)

        assert payload["reasoning_effort"] == "high"

    def test_build_payload_with_tools(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="sarvam-m",
            input="Hello",
            tools=[{
                "type": "function",
                "name": "test",
                "description": "Test tool",
                "parameters": {"type": "object", "properties": {}},
            }],
            tool_choice="auto",
        )
        payload = provider._build_payload(request, stream=False)

        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "test"
        assert payload["tool_choice"] == "auto"

    def test_build_payload_streaming_includes_usage(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="sarvam-m",
            input="Hello",
        )
        payload = provider._build_payload(request, stream=True)

        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True

    # ------------------------------------------------------------------
    # Response conversion
    # ------------------------------------------------------------------

    def test_convert_response_text(self):
        provider = self._make_provider()
        data = {
            "id": "chatcmpl-123",
            "created": 1700000000,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello from Sarvam!",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        response = provider._convert_response(data, "sarvam-m")

        assert response.id == "chatcmpl-123"
        assert response.model == "sarvam-m"
        assert response.status == "completed"
        assert response.output_text == "Hello from Sarvam!"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_convert_response_function_call(self):
        provider = self._make_provider()
        data = {
            "id": "chatcmpl-456",
            "created": 1700000000,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Mumbai"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }

        response = provider._convert_response(data, "sarvam-m")

        assert response.has_function_calls is True
        calls = response.function_calls
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].call_id == "call_123"
        assert json.loads(calls[0].arguments) == {"city": "Mumbai"}

    def test_convert_response_with_reasoning(self):
        provider = self._make_provider()
        data = {
            "id": "chatcmpl-789",
            "created": 1700000000,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think step by step...",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20,
                "total_tokens": 35,
            },
        }

        response = provider._convert_response(data, "sarvam-m")

        assert response.output_text == "The answer is 42."
        # Should have reasoning item
        reasoning_items = [
            i for i in response.output
            if isinstance(i, dict) and i.get("type") == "reasoning"
        ]
        assert len(reasoning_items) == 1

    def test_convert_response_empty(self):
        provider = self._make_provider()
        data = {
            "id": "chatcmpl-empty",
            "created": 1700000000,
            "choices": [],
            "usage": None,
        }

        response = provider._convert_response(data, "sarvam-m")

        assert response.output == []
        assert response.usage is None
