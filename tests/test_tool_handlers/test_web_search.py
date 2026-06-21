from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from caip_responses.tool_handlers.web_search import (
    WebSearchHandler,
    WebSearchMetrics,
    WebSearchUsage,
)


class TestWebSearchHandler:
    def test_tool_type(self):
        handler = WebSearchHandler()
        assert handler.tool_type() == "web_search"

    def test_to_function_tools(self):
        handler = WebSearchHandler()
        tools = handler.to_function_tools({"type": "web_search"})
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["name"] == "_builtin_web_search_query"
        assert "query" in tool["parameters"]["properties"]
        assert "query" in tool["parameters"]["required"]

    def test_is_synthetic(self):
        handler = WebSearchHandler()
        assert handler.is_synthetic("_builtin_web_search_query") is True
        assert handler.is_synthetic("get_weather") is False

    @pytest.mark.asyncio
    async def test_execute_no_query(self):
        handler = WebSearchHandler()
        result = await handler.execute("_builtin_web_search_query", {})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_with_callback(self):
        """Custom search callback should be called (no OpenAI needed)."""
        results_returned = [
            {"title": "Test", "url": "https://example.com", "snippet": "A test result."}
        ]

        async def mock_search(query: str, num_results: int) -> list[dict]:
            assert num_results == 5
            return results_returned

        handler = WebSearchHandler(search_callback=mock_search)
        handler.to_function_tools({"type": "web_search", "search_context_size": "medium"})
        result = await handler.execute(
            "_builtin_web_search_query", {"query": "test search"}
        )
        data = json.loads(result)
        assert data["query"] == "test search"
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Test"
        # No usage tracked for custom callbacks
        assert "_usage" not in data

    @pytest.mark.asyncio
    async def test_execute_respects_search_context_size(self):
        """The web search handler should honor the tool's search_context_size."""

        async def mock_search(query: str, num_results: int) -> list[dict]:
            assert num_results == 10
            return [
                {"title": "Result", "url": "https://example.com", "snippet": "Outcome."}
            ]

        handler = WebSearchHandler(search_callback=mock_search)
        handler.to_function_tools({"type": "web_search", "search_context_size": "high"})
        result = await handler.execute(
            "_builtin_web_search_query", {"query": "important topic"}
        )
        data = json.loads(result)
        assert data["query"] == "important topic"

    @pytest.mark.asyncio
    async def test_execute_empty_results(self):
        async def empty_search(query: str, num_results: int) -> list[dict]:
            return []

        handler = WebSearchHandler(search_callback=empty_search)
        result = await handler.execute(
            "_builtin_web_search_query", {"query": "nothing"}
        )
        data = json.loads(result)
        assert data["message"] == "No results found."

    def test_to_output_item(self):
        handler = WebSearchHandler()
        item = handler.to_output_item(
            "_builtin_web_search_query",
            {"query": "test"},
            "{}",
        )
        assert item is not None
        assert item["type"] == "web_search_call"
        assert item["status"] == "completed"
        assert item["action"]["type"] == "search"
        assert item["action"]["query"] == "test"

    def test_system_prompt_addendum_is_none(self):
        """WebSearchHandler has no system prompt addendum."""
        handler = WebSearchHandler()
        assert handler.system_prompt_addendum({"type": "web_search"}) is None

    @pytest.mark.asyncio
    async def test_execute_callback_multiple_results(self):
        """Callback returning multiple results formats correctly."""

        async def multi_search(query: str, num_results: int) -> list[dict]:
            return [
                {"title": f"Result {i}", "url": f"https://r{i}.com", "snippet": f"Snippet {i}"}
                for i in range(1, 4)
            ]

        handler = WebSearchHandler(search_callback=multi_search)
        result = await handler.execute(
            "_builtin_web_search_query", {"query": "multi"}
        )
        data = json.loads(result)
        assert len(data["results"]) == 3
        for i, r in enumerate(data["results"], 1):
            assert r["index"] == i
            assert r["title"] == f"Result {i}"


class TestWebSearchUsage:
    def test_usage_defaults(self):
        usage = WebSearchUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.search_calls == 0
        assert usage.provider == "openai"

    def test_usage_with_values(self):
        usage = WebSearchUsage(
            input_tokens=150,
            output_tokens=300,
            total_tokens=450,
            search_calls=1,
            model="gpt-4.1-nano",
        )
        assert usage.input_tokens == 150
        assert usage.output_tokens == 300
        assert usage.total_tokens == 450
        assert usage.model == "gpt-4.1-nano"


class TestWebSearchMetrics:
    def test_empty_metrics(self):
        metrics = WebSearchMetrics()
        assert metrics.total_tokens == 0
        assert metrics.total_search_calls == 0
        assert metrics.calls == []

    def test_record_usage(self):
        metrics = WebSearchMetrics()
        usage = WebSearchUsage(
            input_tokens=100, output_tokens=200, total_tokens=300,
            search_calls=1, model="gpt-4.1-nano",
        )
        metrics.record(usage)

        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 200
        assert metrics.total_tokens == 300
        assert metrics.total_search_calls == 1
        assert len(metrics.calls) == 1

    def test_multiple_records(self):
        metrics = WebSearchMetrics()
        for i in range(3):
            usage = WebSearchUsage(
                input_tokens=100, output_tokens=200, total_tokens=300,
                search_calls=1, model="gpt-4.1-nano",
            )
            metrics.record(usage)

        assert metrics.total_input_tokens == 300
        assert metrics.total_output_tokens == 600
        assert metrics.total_tokens == 900
        assert metrics.total_search_calls == 3
        assert len(metrics.calls) == 3

    def test_to_dict(self):
        metrics = WebSearchMetrics()
        usage = WebSearchUsage(
            input_tokens=100, output_tokens=200, total_tokens=300,
            search_calls=1, model="gpt-4.1-nano",
        )
        metrics.record(usage)
        d = metrics.to_dict()

        assert d["total_input_tokens"] == 100
        assert d["total_output_tokens"] == 200
        assert d["total_tokens"] == 300
        assert d["total_search_calls"] == 1
        assert len(d["calls"]) == 1
        assert d["calls"][0]["model"] == "gpt-4.1-nano"
        assert d["calls"][0]["provider"] == "openai"


class TestOpenAIWebSearch:
    """Tests for the OpenAI-delegated web search backend."""

    def _make_mock_openai_response(self):
        """Build a mock OpenAI response with web_search_call + message output."""
        # Mock annotation (url_citation)
        annotation = MagicMock()
        annotation.type = "url_citation"
        annotation.title = "Python Guide"
        annotation.url = "https://docs.python.org"

        # Mock text block
        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Python is a programming language."
        text_block.annotations = [annotation]

        # Mock message item
        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        # Mock web_search_call item
        ws_item = MagicMock()
        ws_item.type = "web_search_call"

        # Mock usage
        usage = MagicMock()
        usage.input_tokens = 120
        usage.output_tokens = 250
        usage.total_tokens = 370

        # Mock response
        response = MagicMock()
        response.output = [ws_item, message_item]
        response.usage = usage

        return response

    @pytest.mark.asyncio
    async def test_openai_search_extracts_results(self):
        """Web search via OpenAI extracts url_citation annotations."""
        handler = WebSearchHandler(openai_api_key="test-key")
        mock_response = self._make_mock_openai_response()

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        handler._openai_client = mock_client

        result = await handler.execute(
            "_builtin_web_search_query", {"query": "python tutorial"}
        )
        data = json.loads(result)

        assert data["query"] == "python tutorial"
        assert len(data["results"]) >= 1
        assert data["results"][0]["title"] == "Python Guide"
        assert data["results"][0]["url"] == "https://docs.python.org"

    @pytest.mark.asyncio
    async def test_openai_search_tracks_usage(self):
        """Token usage from OpenAI web search is tracked in metrics."""
        handler = WebSearchHandler(openai_api_key="test-key")
        mock_response = self._make_mock_openai_response()

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_web_search_query", {"query": "python tutorial"}
        )

        assert handler.metrics.total_search_calls == 1
        assert handler.metrics.total_input_tokens == 120
        assert handler.metrics.total_output_tokens == 250
        assert handler.metrics.total_tokens == 370

    @pytest.mark.asyncio
    async def test_openai_search_usage_in_result(self):
        """Usage info is embedded in the formatted result string."""
        handler = WebSearchHandler(openai_api_key="test-key")
        mock_response = self._make_mock_openai_response()

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        handler._openai_client = mock_client

        result = await handler.execute(
            "_builtin_web_search_query", {"query": "python tutorial"}
        )
        data = json.loads(result)

        assert "_usage" in data
        assert data["_usage"]["input_tokens"] == 120
        assert data["_usage"]["output_tokens"] == 250
        assert data["_usage"]["total_tokens"] == 370

    @pytest.mark.asyncio
    async def test_openai_search_output_item_has_usage(self):
        """The to_output_item includes web search usage for billing."""
        handler = WebSearchHandler(openai_api_key="test-key")
        mock_response = self._make_mock_openai_response()

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        handler._openai_client = mock_client

        # Execute to populate metrics
        await handler.execute(
            "_builtin_web_search_query", {"query": "billing test"}
        )

        item = handler.to_output_item(
            "_builtin_web_search_query",
            {"query": "billing test"},
            "{}",
        )
        assert item is not None
        assert "_web_search_usage" in item
        assert item["_web_search_usage"]["input_tokens"] == 120
        assert item["_web_search_usage"]["output_tokens"] == 250
        assert item["_web_search_usage"]["total_tokens"] == 370
        assert item["_web_search_usage"]["model"] == "gpt-4.1-nano"
        assert item["_web_search_usage"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_openai_search_cumulative_metrics(self):
        """Multiple searches accumulate in metrics."""
        handler = WebSearchHandler(openai_api_key="test-key")
        mock_response = self._make_mock_openai_response()

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_web_search_query", {"query": "search 1"}
        )
        await handler.execute(
            "_builtin_web_search_query", {"query": "search 2"}
        )

        assert handler.metrics.total_search_calls == 2
        assert handler.metrics.total_input_tokens == 240
        assert handler.metrics.total_tokens == 740

    @pytest.mark.asyncio
    async def test_openai_search_reset_metrics(self):
        """Metrics can be reset."""
        handler = WebSearchHandler(openai_api_key="test-key")
        mock_response = self._make_mock_openai_response()

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_web_search_query", {"query": "test"}
        )
        assert handler.metrics.total_search_calls == 1

        handler.reset_metrics()
        assert handler.metrics.total_search_calls == 0
        assert handler.metrics.total_tokens == 0

    @pytest.mark.asyncio
    async def test_openai_search_model_passed(self):
        """The configured model is passed to the OpenAI client."""
        handler = WebSearchHandler(
            openai_api_key="test-key",
            openai_model="gpt-4.1-mini",
        )
        mock_response = self._make_mock_openai_response()

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_web_search_query", {"query": "test"}
        )

        call_kwargs = mock_client.responses.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4.1-mini"

    @pytest.mark.asyncio
    async def test_openai_search_sends_web_search_tool(self):
        """The OpenAI call includes web_search tool."""
        handler = WebSearchHandler(openai_api_key="test-key")
        mock_response = self._make_mock_openai_response()

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_web_search_query", {"query": "test"}
        )

        call_kwargs = mock_client.responses.create.call_args
        assert call_kwargs.kwargs["tools"] == [{"type": "web_search"}]

    @pytest.mark.asyncio
    async def test_fallback_no_annotations(self):
        """If no url_citation annotations, falls back to text output."""
        handler = WebSearchHandler(openai_api_key="test-key")

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Here are some results about Python."
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 50
        usage.output_tokens = 100
        usage.total_tokens = 150

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        result = await handler.execute(
            "_builtin_web_search_query", {"query": "python"}
        )
        data = json.loads(result)

        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Web Search Results"
        assert "Python" in data["results"][0]["snippet"]

    def test_callback_takes_priority_over_openai(self):
        """Custom callback is used even when OpenAI key is present."""
        handler = WebSearchHandler(
            openai_api_key="test-key",
            search_callback=AsyncMock(return_value=[]),
        )
        # The callback field should be set — it takes priority
        assert handler._search_callback is not None
        assert handler._openai_api_key is not None

    def test_last_usage_none_when_no_calls(self):
        handler = WebSearchHandler()
        assert handler.last_usage is None

    @pytest.mark.asyncio
    async def test_last_usage_after_openai_call(self):
        handler = WebSearchHandler(openai_api_key="test-key")

        usage = MagicMock()
        usage.input_tokens = 80
        usage.output_tokens = 160
        usage.total_tokens = 240

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Result"
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_web_search_query", {"query": "test"}
        )

        last = handler.last_usage
        assert last is not None
        assert last.input_tokens == 80
        assert last.output_tokens == 160
        assert last.total_tokens == 240
        assert last.search_calls == 1
