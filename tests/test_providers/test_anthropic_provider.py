from __future__ import annotations

import json
from unittest.mock import MagicMock

from caip_responses.providers.anthropic_provider import AnthropicProvider


class TestAnthropicTranslation:
    """Test the Anthropic provider's translation logic without live API calls."""

    def _make_provider(self) -> AnthropicProvider:
        """Create a provider with a mocked client."""
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._client = MagicMock()
        return provider

    def test_translate_input_string(self):
        provider = self._make_provider()
        messages = provider._translate_input("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_translate_input_messages(self):
        provider = self._make_provider()
        messages = provider._translate_input([
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "Let me check."},
        ])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_translate_input_with_function_call_output(self):
        provider = self._make_provider()
        messages = provider._translate_input([
            {"role": "user", "content": "Get weather"},
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
            {
                "type": "function_call_output",
                "call_id": "fc_1",
                "output": '{"temp": 72}',
            },
        ])
        # Should produce: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"][0]["type"] == "tool_use"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"

    def test_translate_input_skips_system(self):
        provider = self._make_provider()
        messages = provider._translate_input([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

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
                    "required": ["city"],
                },
            },
            # Non-function tools should be skipped
            {"type": "web_search"},
            {"type": "mcp", "server_label": "test", "server_url": "http://localhost"},
        ])
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"
        assert "input_schema" in tools[0]
        assert tools[0]["input_schema"]["type"] == "object"

    def test_translate_tool_choice_auto(self):
        provider = self._make_provider()
        assert provider._translate_tool_choice("auto") == {"type": "auto"}

    def test_translate_tool_choice_required(self):
        provider = self._make_provider()
        assert provider._translate_tool_choice("required") == {"type": "any"}

    def test_translate_tool_choice_specific(self):
        provider = self._make_provider()
        result = provider._translate_tool_choice({"name": "get_weather"})
        assert result == {"type": "tool", "name": "get_weather"}

    def test_build_kwargs_basic(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="claude-sonnet-4-20250514",
            input="Hello",
            instructions="Be helpful",
            temperature=0.7,
            max_output_tokens=2048,
        )
        kwargs = provider._build_kwargs(request)

        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["max_tokens"] == 2048
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert kwargs["system"] == "Be helpful"
        assert kwargs["temperature"] == 0.7

    def test_build_kwargs_with_reasoning(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="claude-sonnet-4-20250514",
            input="Hello",
            reasoning={"effort": "high"},
        )
        kwargs = provider._build_kwargs(request)

        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 16384

    def test_build_kwargs_with_tools(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="claude-sonnet-4-20250514",
            input="Hello",
            tools=[{
                "type": "function",
                "name": "test",
                "description": "Test tool",
                "parameters": {"type": "object", "properties": {}},
            }],
            tool_choice="auto",
        )
        kwargs = provider._build_kwargs(request)

        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["name"] == "test"
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_convert_response_text(self):
        provider = self._make_provider()

        # Mock Anthropic message
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hello! I'm Claude."

        mock_message = MagicMock()
        mock_message.id = "msg_123"
        mock_message.content = [mock_block]
        mock_message.stop_reason = "end_turn"
        mock_message.usage = MagicMock()
        mock_message.usage.input_tokens = 10
        mock_message.usage.output_tokens = 5

        response = provider._convert_response(mock_message, "claude-sonnet-4-20250514")

        assert response.id == "msg_123"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.status == "completed"
        assert response.output_text == "Hello! I'm Claude."
        assert response.usage.input_tokens == 10

    def test_convert_response_tool_use(self):
        provider = self._make_provider()

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Let me check the weather."

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "toolu_123"
        mock_tool_block.name = "get_weather"
        mock_tool_block.input = {"city": "SF"}

        mock_message = MagicMock()
        mock_message.id = "msg_456"
        mock_message.content = [mock_text_block, mock_tool_block]
        mock_message.stop_reason = "tool_use"
        mock_message.usage = MagicMock()
        mock_message.usage.input_tokens = 20
        mock_message.usage.output_tokens = 15

        response = provider._convert_response(mock_message, "claude-sonnet-4-20250514")

        assert response.output_text == "Let me check the weather."
        assert response.has_function_calls is True
        calls = response.function_calls
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].call_id == "toolu_123"
        assert json.loads(calls[0].arguments) == {"city": "SF"}

    def test_convert_response_thinking(self):
        provider = self._make_provider()

        mock_thinking = MagicMock()
        mock_thinking.type = "thinking"
        mock_thinking.thinking = "Let me reason about this..."

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "The answer is 42."

        mock_message = MagicMock()
        mock_message.id = "msg_789"
        mock_message.content = [mock_thinking, mock_text]
        mock_message.stop_reason = "end_turn"
        mock_message.usage = MagicMock()
        mock_message.usage.input_tokens = 30
        mock_message.usage.output_tokens = 50

        response = provider._convert_response(mock_message, "claude-sonnet-4-20250514")

        # Should have reasoning item + message item
        reasoning_items = [i for i in response.output if isinstance(i, dict) and i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert response.output_text == "The answer is 42."
