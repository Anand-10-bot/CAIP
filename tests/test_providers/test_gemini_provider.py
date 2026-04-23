from __future__ import annotations

import json
from unittest.mock import MagicMock

from caip_responses.providers.gemini_provider import GeminiProvider


class TestGeminiTranslation:
    """Test the Gemini provider's translation logic without live API calls."""

    def _make_provider(self) -> GeminiProvider:
        """Create a provider with a mocked client."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._client = MagicMock()
        return provider

    # ------------------------------------------------------------------
    # Input translation
    # ------------------------------------------------------------------

    def test_translate_input_string(self):
        provider = self._make_provider()
        contents = provider._translate_input("Hello")
        assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_translate_input_messages(self):
        provider = self._make_provider()
        contents = provider._translate_input([
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "Let me check."},
        ])
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "What's the weather?"}]
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "Let me check."}]

    def test_translate_input_with_function_call_output(self):
        provider = self._make_provider()
        contents = provider._translate_input([
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
        # user message, model function_call, user function_response
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert "function_call" in contents[1]["parts"][0]
        assert contents[1]["parts"][0]["function_call"]["name"] == "get_weather"
        assert contents[2]["role"] == "user"
        assert "function_response" in contents[2]["parts"][0]

    def test_translate_input_skips_system(self):
        provider = self._make_provider()
        contents = provider._translate_input([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ])
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_translate_input_content_blocks(self):
        provider = self._make_provider()
        contents = provider._translate_input([
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": " World"},
                ],
            }
        ])
        assert len(contents) == 1
        assert contents[0]["parts"] == [{"text": "Hello"}, {"text": " World"}]

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
                    "required": ["city"],
                },
            },
            {"type": "web_search"},
            {"type": "mcp", "server_label": "test", "server_url": "http://localhost"},
        ])
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"
        assert tools[0]["description"] == "Get weather"
        assert tools[0]["parameters"]["type"] == "object"

    def test_translate_tool_choice_auto(self):
        provider = self._make_provider()
        assert provider._translate_tool_choice("auto") == "AUTO"

    def test_translate_tool_choice_required(self):
        provider = self._make_provider()
        assert provider._translate_tool_choice("required") == "ANY"

    def test_translate_tool_choice_none(self):
        provider = self._make_provider()
        assert provider._translate_tool_choice("none") == "NONE"

    def test_translate_tool_choice_specific(self):
        provider = self._make_provider()
        result = provider._translate_tool_choice({"name": "get_weather"})
        assert result == "ANY"

    # ------------------------------------------------------------------
    # Build kwargs
    # ------------------------------------------------------------------

    def test_build_kwargs_basic(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="gemini-2.0-flash",
            input="Hello",
            instructions="Be helpful",
            temperature=0.7,
            max_output_tokens=2048,
        )
        kwargs = provider._build_kwargs(request)

        assert kwargs["model"] == "gemini-2.0-flash"
        assert kwargs["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        assert kwargs["config"]["system_instruction"] == "Be helpful"
        assert kwargs["config"]["temperature"] == 0.7
        assert kwargs["config"]["max_output_tokens"] == 2048

    def test_build_kwargs_with_reasoning(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="gemini-2.0-flash-thinking",
            input="Hello",
            reasoning={"effort": "high"},
        )
        kwargs = provider._build_kwargs(request)

        assert "thinking_config" in kwargs["config"]
        assert kwargs["config"]["thinking_config"]["thinking_budget"] == 16384

    def test_build_kwargs_with_tools(self):
        from caip_responses.models.request import CreateResponseRequest

        provider = self._make_provider()
        request = CreateResponseRequest(
            model="gemini-2.0-flash",
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

        assert "tools" in kwargs["config"]
        assert len(kwargs["config"]["tools"]) == 1
        assert kwargs["config"]["tools"][0]["function_declarations"][0]["name"] == "test"
        assert kwargs["config"]["tool_config"]["function_calling_config"]["mode"] == "AUTO"

    # ------------------------------------------------------------------
    # Response conversion
    # ------------------------------------------------------------------

    def test_convert_response_text(self):
        provider = self._make_provider()

        mock_text_part = MagicMock()
        mock_text_part.text = "Hello from Gemini!"
        mock_text_part.function_call = None
        mock_text_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_text_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_result = MagicMock()
        mock_result.candidates = [mock_candidate]
        mock_result.usage_metadata = MagicMock()
        mock_result.usage_metadata.prompt_token_count = 10
        mock_result.usage_metadata.candidates_token_count = 5

        response = provider._convert_response(mock_result, "gemini-2.0-flash")

        assert response.model == "gemini-2.0-flash"
        assert response.status == "completed"
        assert response.output_text == "Hello from Gemini!"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_convert_response_function_call(self):
        provider = self._make_provider()

        mock_text_part = MagicMock()
        mock_text_part.text = "Let me check."
        mock_text_part.function_call = None
        mock_text_part.thought = None

        mock_fc = MagicMock()
        mock_fc.name = "get_weather"
        mock_fc.args = {"city": "SF"}

        mock_fc_part = MagicMock()
        mock_fc_part.text = None
        mock_fc_part.function_call = mock_fc
        mock_fc_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_text_part, mock_fc_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_result = MagicMock()
        mock_result.candidates = [mock_candidate]
        mock_result.usage_metadata = MagicMock()
        mock_result.usage_metadata.prompt_token_count = 15
        mock_result.usage_metadata.candidates_token_count = 10

        response = provider._convert_response(mock_result, "gemini-2.0-flash")

        assert response.output_text == "Let me check."
        assert response.has_function_calls is True
        calls = response.function_calls
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments) == {"city": "SF"}

    def test_convert_response_empty(self):
        provider = self._make_provider()

        mock_result = MagicMock()
        mock_result.candidates = []
        mock_result.usage_metadata = None

        response = provider._convert_response(mock_result, "gemini-2.0-flash")

        assert response.model == "gemini-2.0-flash"
        assert response.output == []
        assert response.usage is None
