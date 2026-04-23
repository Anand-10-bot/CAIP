"""File search tool handler.

Delegates file_search calls to Azure OpenAI, which handles vector store
searches natively. The results are extracted and returned to the
non-OpenAI provider.
"""

from __future__ import annotations

import json
from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.tool_handlers.openai_delegator import (
    DelegatedToolMetrics,
    OpenAIDelegatorMixin,
)
from caip_responses.utils.id_gen import generate_item_id


class FileSearchHandler(BuiltinToolHandler, OpenAIDelegatorMixin):
    """Client-side file search tool handler for non-OpenAI providers.

    Delegates to Azure OpenAI's built-in file_search tool which performs
    semantic search over vector stores.

    Usage::

        handler = FileSearchHandler(
            openai_api_key="sk-...",
            openai_base_url="https://your-resource.openai.azure.com",
        )
        registry.register(handler)
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str = "gpt-4.1-nano",
    ) -> None:
        BuiltinToolHandler.__init__(self)
        OpenAIDelegatorMixin.__init__(
            self,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
        )
        self._file_search_config: dict[str, Any] = {}

    @property
    def metrics(self) -> DelegatedToolMetrics:
        return self._delegated_metrics

    def tool_type(self) -> str:
        return "file_search"

    def to_function_tools(
        self, tool_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self._file_search_config = tool_config
        return [
            {
                "type": "function",
                "name": self._make_fn_name("query"),
                "description": (
                    "Search through uploaded files and documents using "
                    "semantic search. Returns relevant passages with "
                    "source file references."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant file content.",
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

        if not self._openai_api_key:
            return json.dumps({
                "error": "file_search tool requires openai_api_key for delegation",
            })

        client = self._get_openai_client()

        # Build the file_search tool config for delegation
        tool_def: dict[str, Any] = {"type": "file_search"}
        if "vector_store_ids" in self._file_search_config:
            tool_def["vector_store_ids"] = self._file_search_config["vector_store_ids"]
        if "max_num_results" in self._file_search_config:
            tool_def["max_num_results"] = self._file_search_config["max_num_results"]
        if "ranking_options" in self._file_search_config:
            tool_def["ranking_options"] = self._file_search_config["ranking_options"]

        response = await client.responses.create(
            model=self._openai_model,
            input=query,
            tools=[tool_def],
            instructions="Search the files and return the relevant content with source references.",
        )

        usage = self._record_usage(response, "file_search")
        text = self._extract_text_from_response(response)

        # Extract file_search_call items for structured results
        search_results = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "file_search_call":
                queries = getattr(item, "queries", [])
                results = getattr(item, "results", [])
                search_results.append({
                    "queries": self._safe_serialize(queries),
                    "results": self._safe_serialize(results),
                })

        result: dict[str, Any] = {"result": text}
        if search_results:
            result["file_search_results"] = search_results
        result["_usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        return json.dumps(result)

    def to_output_item(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> dict[str, Any] | None:
        item: dict[str, Any] = {
            "type": "file_search_call",
            "id": generate_item_id(),
            "status": "completed",
            "queries": [arguments.get("query", "")],
        }

        last = self.last_delegated_usage
        if last:
            item["_delegated_usage"] = {
                "input_tokens": last.input_tokens,
                "output_tokens": last.output_tokens,
                "total_tokens": last.total_tokens,
                "model": last.model,
                "provider": last.provider,
            }
        return item

    @staticmethod
    def _safe_serialize(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (list, tuple)):
            return [FileSearchHandler._safe_serialize(x) for x in obj]
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return {k: FileSearchHandler._safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        return str(obj)
