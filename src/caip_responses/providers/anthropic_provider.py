from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from caip_responses.models.common import Usage
from caip_responses.models.errors import ProviderError
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider
from caip_responses.utils.id_gen import (
    generate_call_id,
    generate_item_id,
    generate_response_id,
    unix_timestamp,
)
from caip_responses.utils.json_schema import schema_to_instruction

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]
    AsyncAnthropic = None  # type: ignore[assignment,misc]

# Reasoning effort → thinking budget_tokens mapping
_EFFORT_TO_BUDGET: dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 16384,
}


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider — translates Responses API to Messages API.

    The caller's code is identical to calling any other provider.
    All translation happens internally.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install it with: pip install caip-responses-lib[anthropic]"
            )
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def supports_tool(self, tool_type: str) -> bool:
        return tool_type in {"function"}

    def supports_reasoning(self) -> bool:
        return True

    async def create_response(
        self, request: CreateResponseRequest
    ) -> Response:
        """Non-streaming response creation via Anthropic Messages API."""
        try:
            kwargs = self._build_kwargs(request)
            message = await self._client.messages.create(**kwargs)
            return self._convert_response(message, request.model)
        except Exception as e:
            if anthropic and isinstance(e, anthropic.APIError):
                raise ProviderError(
                    str(e),
                    provider="anthropic",
                    status_code=getattr(e, "status_code", None),
                    raw_error=e,
                ) from e
            raise

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        """Streaming response creation — translates Anthropic SSE to unified events."""
        try:
            kwargs = self._build_kwargs(request)
            response_id = generate_response_id()
            seq = 0

            # Emit response.created
            yield StreamEvent(
                type="response.created",
                sequence_number=seq,
                response={"id": response_id, "model": request.model, "status": "in_progress"},
            )
            seq += 1

            # Track state for building events
            current_item_index = 0
            current_content_index = 0
            accumulated_text = ""
            accumulated_fn_args = ""
            current_tool_name = ""
            current_tool_id = ""
            input_tokens = 0
            output_tokens = 0

            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", str(event))

                    if event_type == "message_start":
                        msg = getattr(event, "message", None)
                        if msg and hasattr(msg, "usage"):
                            input_tokens = getattr(msg.usage, "input_tokens", 0)
                        yield StreamEvent(
                            type="response.in_progress",
                            sequence_number=seq,
                            response={"id": response_id, "model": request.model, "status": "in_progress"},
                        )
                        seq += 1

                    elif event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        block_type = getattr(block, "type", None) if block else None

                        if block_type == "text":
                            item_id = generate_item_id()
                            yield StreamEvent(
                                type="response.output_item.added",
                                sequence_number=seq,
                                output_index=current_item_index,
                                item={"type": "message", "id": item_id, "role": "assistant"},
                            )
                            seq += 1
                            yield StreamEvent(
                                type="response.content_part.added",
                                sequence_number=seq,
                                output_index=current_item_index,
                                content_index=current_content_index,
                                part={"type": "output_text", "text": ""},
                            )
                            seq += 1
                            accumulated_text = ""

                        elif block_type == "tool_use":
                            current_tool_name = getattr(block, "name", "")
                            current_tool_id = getattr(block, "id", generate_call_id())
                            item_id = generate_item_id()
                            yield StreamEvent(
                                type="response.output_item.added",
                                sequence_number=seq,
                                output_index=current_item_index,
                                item={
                                    "type": "function_call",
                                    "id": item_id,
                                    "call_id": current_tool_id,
                                    "name": current_tool_name,
                                    "arguments": "",
                                },
                            )
                            seq += 1
                            accumulated_fn_args = ""

                        elif block_type == "thinking":
                            item_id = generate_item_id()
                            yield StreamEvent(
                                type="response.output_item.added",
                                sequence_number=seq,
                                output_index=current_item_index,
                                item={"type": "reasoning", "id": item_id},
                            )
                            seq += 1

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", None) if delta else None

                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            accumulated_text += text
                            yield StreamEvent(
                                type="response.output_text.delta",
                                sequence_number=seq,
                                output_index=current_item_index,
                                content_index=current_content_index,
                                delta=text,
                            )
                            seq += 1

                        elif delta_type == "input_json_delta":
                            partial = getattr(delta, "partial_json", "")
                            accumulated_fn_args += partial
                            yield StreamEvent(
                                type="response.function_call_arguments.delta",
                                sequence_number=seq,
                                output_index=current_item_index,
                                delta=partial,
                            )
                            seq += 1

                        elif delta_type == "thinking_delta":
                            text = getattr(delta, "thinking", "")
                            yield StreamEvent(
                                type="response.reasoning_text.delta",
                                sequence_number=seq,
                                output_index=current_item_index,
                                delta=text,
                            )
                            seq += 1

                    elif event_type == "content_block_stop":
                        # Determine what kind of block just ended based on accumulated state
                        if accumulated_fn_args or current_tool_name:
                            yield StreamEvent(
                                type="response.function_call_arguments.done",
                                sequence_number=seq,
                                output_index=current_item_index,
                                delta=accumulated_fn_args,
                            )
                            seq += 1
                            current_tool_name = ""
                            current_tool_id = ""
                            accumulated_fn_args = ""
                        elif accumulated_text:
                            yield StreamEvent(
                                type="response.output_text.done",
                                sequence_number=seq,
                                output_index=current_item_index,
                                content_index=current_content_index,
                                delta=accumulated_text,
                            )
                            seq += 1
                            yield StreamEvent(
                                type="response.content_part.done",
                                sequence_number=seq,
                                output_index=current_item_index,
                                content_index=current_content_index,
                            )
                            seq += 1
                            accumulated_text = ""

                        yield StreamEvent(
                            type="response.output_item.done",
                            sequence_number=seq,
                            output_index=current_item_index,
                        )
                        seq += 1
                        current_item_index += 1
                        current_content_index = 0

                    elif event_type == "message_delta":
                        delta = getattr(event, "delta", None)
                        usage_info = getattr(event, "usage", None)
                        if usage_info:
                            output_tokens = getattr(usage_info, "output_tokens", 0)

                    elif event_type == "message_stop":
                        pass  # handled below

            # Emit response.completed
            yield StreamEvent(
                type="response.completed",
                sequence_number=seq,
                response={
                    "id": response_id,
                    "model": request.model,
                    "status": "completed",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                },
            )

        except Exception as e:
            if anthropic and isinstance(e, anthropic.APIError):
                raise ProviderError(
                    str(e),
                    provider="anthropic",
                    status_code=getattr(e, "status_code", None),
                    raw_error=e,
                ) from e
            raise

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Internal: request translation
    # ------------------------------------------------------------------

    def _build_kwargs(self, request: CreateResponseRequest) -> dict[str, Any]:
        """Translate unified request → Anthropic Messages API kwargs."""
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens or 4096,
        }

        # Translate input → messages
        messages = self._translate_input(request.input)
        kwargs["messages"] = messages

        # System instructions
        system_parts: list[str] = []
        if request.instructions:
            system_parts.append(request.instructions)

        # Structured output: inject schema into system prompt
        if request.text and not isinstance(request.text, dict):
            fmt = request.text.format
            if fmt and hasattr(fmt, "type") and fmt.type == "json_schema":
                schema = getattr(fmt, "schema_", {})
                if schema:
                    system_parts.append(schema_to_instruction(schema))
        elif isinstance(request.text, dict):
            fmt = request.text.get("format", {})
            if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
                schema = fmt.get("schema", {})
                if schema:
                    system_parts.append(schema_to_instruction(schema))

        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        # Tools → Anthropic format
        if request.tools:
            anthropic_tools = self._translate_tools(request.tools)
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

        # Tool choice
        if request.tool_choice is not None:
            kwargs["tool_choice"] = self._translate_tool_choice(request.tool_choice)

        # Temperature
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        # Reasoning / thinking
        if request.reasoning:
            reasoning = request.reasoning
            if isinstance(reasoning, dict):
                effort = reasoning.get("effort")
            else:
                effort = reasoning.effort
            if effort and effort in _EFFORT_TO_BUDGET:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": _EFFORT_TO_BUDGET[effort],
                }

        return kwargs

    def _translate_input(
        self, input_data: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate unified input format → Anthropic messages array."""
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]

        messages: list[dict[str, Any]] = []
        for item in input_data:
            if isinstance(item, dict):
                item_type = item.get("type")
                role = item.get("role")

                # Regular message
                if role and role in ("user", "assistant"):
                    content = item.get("content", "")
                    if isinstance(content, str):
                        messages.append({"role": role, "content": content})
                    elif isinstance(content, list):
                        anthropic_content = []
                        for block in content:
                            if isinstance(block, dict):
                                btype = block.get("type", "")
                                if btype in ("input_text", "output_text", "text"):
                                    anthropic_content.append({
                                        "type": "text",
                                        "text": block.get("text", ""),
                                    })
                                elif btype == "input_image":
                                    anthropic_content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": block.get("image_url", ""),
                                        },
                                    })
                            elif isinstance(block, str):
                                anthropic_content.append({"type": "text", "text": block})
                        messages.append({"role": role, "content": anthropic_content})

                # system/developer → skip (handled via system parameter)
                elif role in ("system", "developer"):
                    continue

                # Function call output → tool_result
                elif item_type == "function_call_output":
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": item.get("call_id", ""),
                            "content": item.get("output", ""),
                        }],
                    })

                # Function call → assistant tool_use block
                elif item_type == "function_call":
                    try:
                        args = json.loads(item.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    messages.append({
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": item.get("call_id", ""),
                            "name": item.get("name", ""),
                            "input": args,
                        }],
                    })

                # Fallback: treat as user message
                elif not item_type and not role:
                    messages.append({"role": "user", "content": str(item)})

        return messages or [{"role": "user", "content": ""}]

    def _translate_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate unified tool definitions → Anthropic tool format."""
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                anthropic_tools.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
                })
            # Skip non-function tools (web_search, mcp, etc.)
            # — they're not natively supported by Anthropic
        return anthropic_tools

    def _translate_tool_choice(
        self, tool_choice: str | dict[str, str]
    ) -> dict[str, Any]:
        """Translate unified tool_choice → Anthropic tool_choice format."""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            if tool_choice == "required":
                return {"type": "any"}
            if tool_choice == "none":
                return {"type": "auto"}  # Anthropic doesn't have "none"
            return {"type": "auto"}
        if isinstance(tool_choice, dict):
            name = tool_choice.get("function", {}).get("name") if isinstance(tool_choice.get("function"), dict) else tool_choice.get("name")
            if name:
                return {"type": "tool", "name": name}
        return {"type": "auto"}

    # ------------------------------------------------------------------
    # Internal: response translation
    # ------------------------------------------------------------------

    def _convert_response(self, message: Any, model: str) -> Response:
        """Convert an Anthropic Message → unified Response."""
        output_items: list[dict[str, Any]] = []
        text_parts: list[dict[str, Any]] = []

        for block in message.content:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text_parts.append({
                    "type": "output_text",
                    "text": block.text,
                    "annotations": [],
                })

            elif block_type == "tool_use":
                # Flush accumulated text first
                if text_parts:
                    output_items.append({
                        "type": "message",
                        "id": generate_item_id(),
                        "role": "assistant",
                        "content": text_parts,
                        "status": "completed",
                    })
                    text_parts = []

                output_items.append({
                    "type": "function_call",
                    "id": generate_item_id(),
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                })

            elif block_type == "thinking":
                output_items.append({
                    "type": "reasoning",
                    "id": generate_item_id(),
                    "summary": [{"type": "text", "text": getattr(block, "thinking", "")}] if hasattr(block, "thinking") else None,
                })

        # Flush remaining text
        if text_parts:
            output_items.append({
                "type": "message",
                "id": generate_item_id(),
                "role": "assistant",
                "content": text_parts,
                "status": "completed",
            })

        # Build usage
        usage = None
        if message.usage:
            usage = Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
            )

        return Response(
            id=getattr(message, "id", generate_response_id()),
            model=model,
            created_at=unix_timestamp(),
            status="completed" if message.stop_reason in ("end_turn", "stop_sequence") else "completed",
            output=output_items,
            usage=usage,
        )
