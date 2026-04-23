from __future__ import annotations

from caip_responses.models.common import Reasoning
from caip_responses.models.request import CreateResponseRequest, PromptConfig


class TestCreateResponseRequest:
    def test_minimal(self):
        req = CreateResponseRequest(model="gpt-4.1")
        assert req.model == "gpt-4.1"
        assert req.input == ""
        assert req.stream is False
        assert req.tools is None
        assert req.provider is None

    def test_full_params(self):
        req = CreateResponseRequest(
            model="claude-sonnet-4-20250514",
            input="Hello",
            instructions="Be helpful",
            tools=[{"type": "function", "name": "test", "parameters": {}}],
            tool_choice="auto",
            parallel_tool_calls=True,
            stream=True,
            previous_response_id="resp_prev",
            reasoning={"effort": "high"},
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=1024,
            metadata={"key": "value"},
            store=True,
            user="user_123",
            provider="anthropic",
        )
        assert req.model == "claude-sonnet-4-20250514"
        assert req.input == "Hello"
        assert req.stream is True
        assert req.provider == "anthropic"
        assert req.temperature == 0.7

    def test_input_as_list(self):
        req = CreateResponseRequest(
            model="gpt-4.1",
            input=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert isinstance(req.input, list)
        assert len(req.input) == 2

    def test_reasoning_as_model(self):
        req = CreateResponseRequest(
            model="o3",
            reasoning=Reasoning(effort="medium"),
        )
        assert req.reasoning.effort == "medium"

    def test_reasoning_as_dict(self):
        req = CreateResponseRequest(
            model="o3",
            reasoning={"effort": "low"},
        )
        # Pydantic coerces dict → Reasoning model
        assert req.reasoning.effort == "low"

    def test_provider_excluded_from_serialization(self):
        req = CreateResponseRequest(model="gpt-4.1", provider="openai")
        dumped = req.model_dump()
        assert "provider" not in dumped

    def test_prompt_as_dict(self):
        req = CreateResponseRequest(
            model="gpt-5",
            prompt={
                "id": "pmpt_abc123",
                "version": "2",
                "variables": {"customer_name": "Jane Doe", "product": "juice box"},
            },
        )
        # Pydantic coerces the dict to a PromptConfig model
        assert req.prompt.id == "pmpt_abc123"
        assert req.prompt.variables["customer_name"] == "Jane Doe"

    def test_prompt_as_model(self):
        req = CreateResponseRequest(
            model="gpt-5",
            prompt=PromptConfig(
                id="pmpt_abc123",
                version="2",
                variables={"name": "Alice"},
            ),
        )
        assert req.prompt.id == "pmpt_abc123"
        assert req.prompt.version == "2"

    def test_prompt_default_none(self):
        req = CreateResponseRequest(model="gpt-4.1")
        assert req.prompt is None

    def test_tool_choice_allowed_tools(self):
        """allowed_tools tool_choice — restricts callable tools to a subset."""
        req = CreateResponseRequest(
            model="gpt-4.1",
            tools=[
                {"type": "function", "name": "get_weather", "parameters": {}},
                {"type": "function", "name": "search_docs", "parameters": {}},
                {"type": "function", "name": "send_email", "parameters": {}},
            ],
            tool_choice={
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "name": "get_weather"},
                    {"type": "function", "name": "search_docs"},
                ],
            },
        )
        assert req.tool_choice["type"] == "allowed_tools"
        assert req.tool_choice["mode"] == "auto"
        assert len(req.tool_choice["tools"]) == 2

    def test_tool_choice_forced_function(self):
        """Forced function tool_choice — call exactly one specific function."""
        req = CreateResponseRequest(
            model="gpt-4.1",
            tool_choice={"type": "function", "name": "get_weather"},
        )
        assert req.tool_choice["type"] == "function"
        assert req.tool_choice["name"] == "get_weather"

    def test_tool_choice_none(self):
        req = CreateResponseRequest(model="gpt-4.1", tool_choice="none")
        assert req.tool_choice == "none"


class TestPromptConfig:
    def test_basic(self):
        config = PromptConfig(id="pmpt_abc123")
        assert config.id == "pmpt_abc123"
        assert config.version is None
        assert config.variables is None

    def test_full(self):
        config = PromptConfig(
            id="pmpt_abc123",
            version="3",
            variables={
                "topic": "Dragons",
                "reference_pdf": {
                    "type": "input_file",
                    "file_id": "file-abc123",
                },
            },
        )
        assert config.version == "3"
        assert config.variables["topic"] == "Dragons"
        assert config.variables["reference_pdf"]["type"] == "input_file"
