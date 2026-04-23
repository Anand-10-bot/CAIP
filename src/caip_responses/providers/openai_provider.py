from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from caip_responses.models.common import Usage
from caip_responses.models.errors import ProviderError
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider
from caip_responses.utils.id_gen import (
    generate_item_id,
    unix_timestamp,
)

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]


class OpenAIProvider(BaseProvider):
    """OpenAI Responses API provider — near pass-through.

    Since the unified API mirrors OpenAI's Responses API, this provider
    mostly wraps the OpenAI SDK's responses.create() directly.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install it with: pip install caip-responses-lib[openai]"
            )
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    @property
    def provider_name(self) -> str:
        return "openai"

    def supports_tool(self, tool_type: str) -> bool:
        return tool_type in {
            "function", "web_search", "file_search",
            "code_interpreter", "computer", "computer_use", "mcp",
            "tool_search", "shell", "image_generation",
            "custom_tool", "namespace",
        }

    def supports_reasoning(self) -> bool:
        return True

    async def create_response(
        self, request: CreateResponseRequest
    ) -> Response:
        """Non-streaming call — pass through to OpenAI SDK."""
        try:
            kwargs = self._build_kwargs(request)
            sdk_response = await self._client.responses.create(**kwargs)
            return self._convert_response(sdk_response)
        except Exception as e:
            if "openai" in type(e).__module__.lower():
                raise ProviderError(
                    str(e),
                    provider="openai",
                    raw_error=e,
                ) from e
            raise

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        """Streaming call — wraps OpenAI SDK stream into unified StreamEvents."""
        try:
            kwargs = self._build_kwargs(request)
            kwargs["stream"] = True
            sdk_stream = await self._client.responses.create(**kwargs)
            async for chunk in sdk_stream:
                yield self._convert_stream_event(chunk)
        except Exception as e:
            if hasattr(e, "__module__") and "openai" in getattr(e, "__module__", "").lower():
                raise ProviderError(
                    str(e),
                    provider="openai",
                    raw_error=e,
                ) from e
            raise

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kwargs(self, request: CreateResponseRequest) -> dict[str, Any]:
        """Build kwargs dict for the OpenAI SDK call."""
        kwargs: dict[str, Any] = {"model": request.model}

        if request.input:
            kwargs["input"] = request.input
        if request.instructions is not None:
            kwargs["instructions"] = request.instructions
        if request.tools:
            kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = request.parallel_tool_calls
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.reasoning is not None:
            reasoning = request.reasoning
            if isinstance(reasoning, dict):
                kwargs["reasoning"] = reasoning
            else:
                kwargs["reasoning"] = reasoning.model_dump(exclude_none=True)
        if request.text is not None:
            if isinstance(request.text, dict):
                kwargs["text"] = request.text
            else:
                kwargs["text"] = request.text.model_dump(exclude_none=True)
        if request.previous_response_id is not None:
            kwargs["previous_response_id"] = request.previous_response_id
        if request.store is not None:
            kwargs["store"] = request.store
        if request.metadata is not None:
            kwargs["metadata"] = request.metadata
        if request.truncation is not None:
            kwargs["truncation"] = request.truncation
        if request.user is not None:
            kwargs["user"] = request.user
        if request.include is not None:
            kwargs["include"] = request.include
        if request.prompt is not None:
            if isinstance(request.prompt, dict):
                kwargs["prompt"] = request.prompt
            else:
                kwargs["prompt"] = request.prompt.model_dump(exclude_none=True)
        if request.background is not None:
            kwargs["background"] = request.background

        return kwargs

    def _convert_response(self, sdk_response: Any) -> Response:
        """Convert an OpenAI SDK Response object to our unified Response."""
        output_items = []
        for item in sdk_response.output:
            output_items.append(self._convert_output_item(item))

        usage = None
        if sdk_response.usage:
            usage = Usage(
                input_tokens=sdk_response.usage.input_tokens,
                output_tokens=sdk_response.usage.output_tokens,
                total_tokens=sdk_response.usage.total_tokens,
            )

        return Response(
            id=sdk_response.id,
            model=sdk_response.model,
            created_at=int(sdk_response.created_at) if sdk_response.created_at else unix_timestamp(),
            status=sdk_response.status or "completed",
            output=output_items,
            usage=usage,
            metadata=sdk_response.metadata if hasattr(sdk_response, "metadata") else None,
        )

    def _convert_output_item(self, item: Any) -> dict[str, Any]:
        """Convert an SDK output item to a dict matching our item types."""
        item_type = getattr(item, "type", None)

        if item_type == "message":
            content = []
            for block in item.content:
                block_type = getattr(block, "type", None)
                if block_type == "output_text":
                    content.append({
                        "type": "output_text",
                        "text": block.text,
                        "annotations": [self._serialize(a) for a in getattr(block, "annotations", [])] or [],
                    })
                elif block_type == "refusal":
                    content.append({
                        "type": "refusal",
                        "refusal": block.refusal,
                    })
            return {
                "type": "message",
                "id": getattr(item, "id", generate_item_id()),
                "role": "assistant",
                "content": content,
                "status": getattr(item, "status", "completed"),
            }

        if item_type == "function_call":
            return {
                "type": "function_call",
                "id": getattr(item, "id", generate_item_id()),
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            }

        if item_type == "reasoning":
            return {
                "type": "reasoning",
                "id": getattr(item, "id", generate_item_id()),
                "summary": getattr(item, "summary", None),
            }

        if item_type == "web_search_call":
            result: dict[str, Any] = {
                "type": "web_search_call",
                "id": getattr(item, "id", generate_item_id()),
                "status": getattr(item, "status", "completed"),
            }
            action = getattr(item, "action", None)
            if action is not None:
                result["action"] = self._serialize(action)
            return result

        if item_type == "file_search_call":
            fs_result: dict[str, Any] = {
                "type": "file_search_call",
                "id": getattr(item, "id", generate_item_id()),
                "status": getattr(item, "status", "completed"),
                "queries": self._serialize(getattr(item, "queries", [])),
            }
            results = getattr(item, "results", None)
            if results is not None:
                fs_result["results"] = self._serialize(results)
            return fs_result

        if item_type == "mcp_call":
            return {
                "type": "mcp_call",
                "id": getattr(item, "id", generate_item_id()),
                "name": getattr(item, "name", ""),
                "arguments": getattr(item, "arguments", ""),
                "output": getattr(item, "output", None),
                "error": getattr(item, "error", None),
                "server_label": getattr(item, "server_label", ""),
                "approval_request_id": getattr(item, "approval_request_id", None),
                "status": getattr(item, "status", "completed"),
            }

        if item_type == "mcp_list_tools":
            return {
                "type": "mcp_list_tools",
                "id": getattr(item, "id", generate_item_id()),
                "server_label": getattr(item, "server_label", ""),
                "tools": self._serialize(getattr(item, "tools", [])),
            }

        if item_type == "mcp_approval_request":
            return {
                "type": "mcp_approval_request",
                "id": getattr(item, "id", generate_item_id()),
                "name": getattr(item, "name", ""),
                "arguments": getattr(item, "arguments", ""),
                "server_label": getattr(item, "server_label", ""),
            }

        if item_type == "shell_call":
            sc_result: dict[str, Any] = {
                "type": "shell_call",
                "id": getattr(item, "id", generate_item_id()),
                "status": getattr(item, "status", "completed"),
                "call_id": getattr(item, "call_id", ""),
                "action": self._serialize(getattr(item, "action", {})),
            }
            max_out = getattr(item, "max_output_length", None)
            if max_out is not None:
                sc_result["max_output_length"] = max_out
            return sc_result

        if item_type == "shell_call_output":
            return {
                "type": "shell_call_output",
                "id": getattr(item, "id", ""),
                "call_id": getattr(item, "call_id", ""),
                "output": getattr(item, "output", ""),
                "error": getattr(item, "error", ""),
                "status": getattr(item, "status", "completed"),
            }

        if item_type == "computer_call":
            cc_result: dict[str, Any] = {
                "type": "computer_call",
                "id": getattr(item, "id", generate_item_id()),
                "call_id": getattr(item, "call_id", ""),
                "actions": self._serialize(getattr(item, "actions", [])),
                "status": getattr(item, "status", "completed"),
            }
            pending = getattr(item, "pending_safety_checks", None)
            if pending:
                cc_result["pending_safety_checks"] = self._serialize(pending)
            return cc_result

        if item_type == "image_generation_call":
            ig_result: dict[str, Any] = {
                "type": "image_generation_call",
                "id": getattr(item, "id", generate_item_id()),
                "status": getattr(item, "status", "completed"),
                "result": getattr(item, "result", None),
            }
            revised = getattr(item, "revised_prompt", None)
            if revised is not None:
                ig_result["revised_prompt"] = revised
            return ig_result

        if item_type == "custom_tool_call":
            return {
                "type": "custom_tool_call",
                "id": getattr(item, "id", generate_item_id()),
                "call_id": getattr(item, "call_id", ""),
                "name": getattr(item, "name", ""),
                "input": getattr(item, "input", ""),
                "status": getattr(item, "status", "completed"),
            }

        if item_type == "tool_search_call":
            return {
                "type": "tool_search_call",
                "id": getattr(item, "id", generate_item_id()),
                "call_id": getattr(item, "call_id", None),
                "execution": getattr(item, "execution", None),
                "status": getattr(item, "status", "completed"),
            }

        if item_type == "tool_search_output":
            return {
                "type": "tool_search_output",
                "id": getattr(item, "id", generate_item_id()),
                "call_id": getattr(item, "call_id", None),
                "tools": self._serialize(getattr(item, "tools", [])),
                "output": self._serialize(getattr(item, "output", [])),
            }

        # Fallback: serialize whatever we got
        return self._serialize(item)

    def _convert_stream_event(self, chunk: Any) -> StreamEvent:
        """Convert an OpenAI SDK stream chunk into a unified StreamEvent."""
        event_type = getattr(chunk, "type", "unknown")
        event_data: dict[str, Any] = {"type": event_type}

        # response-level events
        if hasattr(chunk, "response"):
            resp = chunk.response
            if resp is not None:
                event_data["response"] = {
                    "id": getattr(resp, "id", None),
                    "model": getattr(resp, "model", None),
                    "status": getattr(resp, "status", None),
                }

        # output item events
        if hasattr(chunk, "output_index"):
            event_data["output_index"] = chunk.output_index
        if hasattr(chunk, "item"):
            item = chunk.item
            if item is not None:
                event_data["item"] = self._serialize(item)

        # content part events
        if hasattr(chunk, "content_index"):
            event_data["content_index"] = chunk.content_index
        if hasattr(chunk, "part"):
            part = chunk.part
            if part is not None:
                event_data["part"] = self._serialize(part)

        # delta events
        if hasattr(chunk, "delta"):
            event_data["delta"] = chunk.delta

        # sequence number
        if hasattr(chunk, "sequence_number"):
            event_data["sequence_number"] = chunk.sequence_number

        return StreamEvent(**event_data)

    def _serialize(self, obj: Any) -> Any:
        """Best-effort serialization of an SDK object to a dict."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._serialize(x) for x in obj]
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return {
                k: self._serialize(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        return str(obj)
