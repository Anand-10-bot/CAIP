from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.utils.id_gen import generate_item_id

# Type for a pluggable search callback:
#   async def my_search(query: str, num_results: int) -> list[dict]
#   Each dict should have: title, url, snippet (all str)
SearchCallback = Callable[[str, int], Awaitable[list[dict[str, str]]]]

# Context size → result count
_CONTEXT_SIZE_MAP: dict[str, int] = {
    "low": 3,
    "medium": 5,
    "high": 10,
}


@dataclass
class WebSearchUsage:
    """Token usage from a web search call routed through OpenAI."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    search_calls: int = 0
    model: str = ""
    provider: str = "openai"


@dataclass
class WebSearchMetrics:
    """Accumulated web search metrics across multiple calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_search_calls: int = 0
    calls: list[WebSearchUsage] = field(default_factory=list)

    def record(self, usage: WebSearchUsage) -> None:
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.total_search_calls += usage.search_calls
        self.calls.append(usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_search_calls": self.total_search_calls,
            "calls": [
                {
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "total_tokens": c.total_tokens,
                    "model": c.model,
                    "provider": c.provider,
                }
                for c in self.calls
            ],
        }


class WebSearchHandler(BuiltinToolHandler):
    """Client-side web search for non-OpenAI providers.

    Converts ``{"type": "web_search"}`` into a function the model can call.
    Executes the search via a pluggable backend and returns formatted results.

    **Primary backend: Azure OpenAI Responses API** — uses the built-in
    ``web_search`` tool that OpenAI already provides server-side.  This
    requires an OpenAI API key + optional base_url (for Azure OpenAI).
    No extra API keys or services needed.

    Supported backends (in priority order):
    1. User-provided ``search_callback`` (fully custom)
    2. Azure OpenAI Responses API ``web_search`` tool (recommended)
    3. httpx-based fallback using DuckDuckGo HTML (no key, best-effort)

    Usage::

        # Recommended: uses your existing Azure OpenAI deployment
        handler = WebSearchHandler(
            openai_api_key="sk-...",
            openai_base_url="https://your-resource.openai.azure.com",
        )
        registry.register(handler)

    Token tracking::

        # After requests, check web search token usage:
        metrics = handler.metrics
        print(metrics.total_tokens)       # total tokens used for web searches
        print(metrics.total_search_calls) # number of search calls made
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str = "gpt-4.1-nano",
        search_callback: SearchCallback | None = None,
    ) -> None:
        self._openai_api_key = openai_api_key
        self._openai_base_url = openai_base_url
        self._openai_model = openai_model
        self._search_callback = search_callback
        self._metrics = WebSearchMetrics()
        self._tool_config: dict[str, Any] | None = None
        # Lazily initialized OpenAI client
        self._openai_client: Any = None

    @property
    def metrics(self) -> WebSearchMetrics:
        """Accumulated web search token usage and call metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._metrics = WebSearchMetrics()

    @property
    def last_usage(self) -> WebSearchUsage | None:
        """Token usage from the most recent web search call."""
        if self._metrics.calls:
            return self._metrics.calls[-1]
        return None

    def tool_type(self) -> str:
        return "web_search"

    def to_function_tools(
        self, tool_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self._tool_config = tool_config
        return [
            {
                "type": "function",
                "name": self._make_fn_name("query"),
                "description": (
                    "Search the web for current information. "
                    "Returns a list of relevant search results with titles, "
                    "URLs, and snippets. Use this when you need up-to-date "
                    "information that may not be in your training data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute.",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        query = arguments.get("query", "")
        if not query:
            return json.dumps({"error": "No query provided"})

        search_size = self._tool_config.get("search_context_size") if self._tool_config else "medium"
        num_results = _CONTEXT_SIZE_MAP.get(search_size, 5)
        results, usage = await self._do_search(query, num_results)

        # Track the usage
        if usage:
            self._metrics.record(usage)

        return self._format_results(query, results, usage)

    def to_output_item(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> dict[str, Any] | None:
        """Produce a web_search_call output item for API parity."""
        item: dict[str, Any] = {
            "type": "web_search_call",
            "id": generate_item_id(),
            "status": "completed",
            "action": {
                "type": "search",
                "query": arguments.get("query", ""),
            },
        }

        # Attach the web search token usage so callers can track billing
        last = self.last_usage
        if last:
            item["_web_search_usage"] = {
                "input_tokens": last.input_tokens,
                "output_tokens": last.output_tokens,
                "total_tokens": last.total_tokens,
                "model": last.model,
                "provider": last.provider,
            }

        return item

    # ------------------------------------------------------------------
    # Search backends
    # ------------------------------------------------------------------

    async def _do_search(
        self, query: str, num_results: int
    ) -> tuple[list[dict[str, str]], WebSearchUsage | None]:
        """Execute search using the best available backend.

        Returns:
            Tuple of (results, usage). Usage is non-None when OpenAI
            backend is used, enabling token tracking for billing.
        """
        if self._search_callback:
            results = await self._search_callback(query, num_results)
            return results, None

        if self._openai_api_key:
            return await self._openai_web_search(query)

        results = await self._duckduckgo_search(query, num_results)
        return results, None

    async def _openai_web_search(
        self, query: str
    ) -> tuple[list[dict[str, str]], WebSearchUsage]:
        """Search using Azure OpenAI's built-in web_search tool.

        Makes a lightweight responses.create() call with web_search enabled.
        The model searches the web server-side and returns results with
        full token usage tracking.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for web search via OpenAI. "
                "Install it with: pip install caip-responses-lib[openai]"
            )

        # Lazily create client (reuse across calls)
        if self._openai_client is None:
            kwargs: dict[str, Any] = {"api_key": self._openai_api_key}
            if self._openai_base_url:
                kwargs["base_url"] = self._openai_base_url
            self._openai_client = AsyncOpenAI(**kwargs)

        response = await self._openai_client.responses.create(
            model=self._openai_model,
            input=f"Search the web for: {query}\n\nReturn only the search results. Do not add any commentary.",
            tools=[{"type": "web_search"}],
            instructions=(
                "You are a web search assistant. Use the web_search tool to find "
                "information, then return the results clearly with titles, URLs, "
                "and brief descriptions. Do not add opinions or extra commentary."
            ),
        )

        # Extract token usage
        usage = WebSearchUsage(
            search_calls=1,
            model=self._openai_model,
            provider="openai",
        )
        if response.usage:
            usage.input_tokens = response.usage.input_tokens
            usage.output_tokens = response.usage.output_tokens
            usage.total_tokens = response.usage.total_tokens

        # Extract search results from the response output
        results = self._extract_results_from_openai_response(response)

        return results, usage

    def _extract_results_from_openai_response(
        self, response: Any
    ) -> list[dict[str, str]]:
        """Extract structured search results from OpenAI response output.

        Looks at web_search_call items for URLs/queries and the text
        output for the formatted results.
        """
        results: list[dict[str, str]] = []
        text_output = ""

        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "web_search_call":
                # The web_search_call itself doesn't contain result text,
                # but we can note that a search happened
                pass

            elif item_type == "message":
                # Extract text from message content
                for block in getattr(item, "content", []):
                    block_type = getattr(block, "type", None)
                    if block_type == "output_text":
                        text_output += getattr(block, "text", "")

                        # Extract URL citations from annotations
                        for annotation in getattr(block, "annotations", []):
                            ann_type = getattr(annotation, "type", None)
                            if ann_type == "url_citation":
                                results.append({
                                    "title": getattr(annotation, "title", ""),
                                    "url": getattr(annotation, "url", ""),
                                    "snippet": text_output[:200] if text_output else "",
                                })

        # If no structured annotations, parse the text as a single result
        if not results and text_output:
            results.append({
                "title": "Web Search Results",
                "url": "",
                "snippet": text_output.strip(),
            })

        return results

    async def _duckduckgo_search(
        self, query: str, num_results: int
    ) -> list[dict[str, str]]:
        """Best-effort search using DuckDuckGo Instant Answer API (no key needed)."""
        import httpx

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"},
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[dict[str, str]] = []

        # Abstract
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append({
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "snippet": abstract,
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:num_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })

        return results[:num_results]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(
        query: str,
        results: list[dict[str, str]],
        usage: WebSearchUsage | None = None,
    ) -> str:
        """Format search results as a readable string for the model."""
        if not results:
            payload: dict[str, Any] = {
                "query": query,
                "results": [],
                "message": "No results found.",
            }
            if usage:
                payload["_usage"] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                }
            return json.dumps(payload)

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append({
                "index": i,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
            })

        payload = {"query": query, "results": formatted}
        if usage:
            payload["_usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
        return json.dumps(payload)
