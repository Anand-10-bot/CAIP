from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

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


class OpenAICompatibleProvider(BaseProvider):
    """Base adapter for any OpenAI-compatible Chat Completions endpoint.

    Many providers — Sarvam, Ollama, vLLM, LM Studio, Together, Groq, and
    most self-hosted open-source servers (Llama, Mistral, Qwen) — expose a
    standard ``/chat/completions`` endpoint. This class implements the full
    Responses ↔ Chat Completions translation once, so a concrete provider
    only needs to set its name, default base URL, and capabilities.

    The caller's code is identical regardless of which model is behind it —
    switching providers is a one-line model name change.

    Subclass and override the class attributes::

        class OllamaProvider(OpenAICompatibleProvider):
            PROVIDER_NAME = "ollama"
            DEFAULT_BASE_URL = "http://localhost:11434/v1"
    """

    #: Unique provider identifier (e.g. "sarvam", "ollama").
    PROVIDER_NAME: str = "openai_compatible"
    #: Default endpoint used when no ``base_url`` is supplied.
    DEFAULT_BASE_URL: str = ""
    #: Whether this endpoint surfaces reasoning/thinking tokens.
    SUPPORTS_REASONING: bool = True

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._build_headers(),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    @property
    def provider_name(self) -> str:
        return self.PROVIDER_NAME

    def supports_tool(self, tool_type: str) -> bool:
        return tool_type in {"function"}

    def supports_reasoning(self) -> bool:
        return self.SUPPORTS_REASONING

    async def create_response(
        self, request: CreateResponseRequest
    ) -> Response:
        """Non-streaming response creation via Chat Completions API."""
        try:
            payload = self._build_payload(request, stream=False)
            resp = await self._http.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return self._convert_response(data, request.model)
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                str(e),
                provider=self.provider_name,
                status_code=e.response.status_code,
                raw_error=e,
            ) from e
        except httpx.HTTPError as e:
            raise ProviderError(
                str(e),
                provider=self.provider_name,
                raw_error=e,
            ) from e

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        """Streaming response creation — translates SSE Chat Completions to unified events."""
        try:
            payload = self._build_payload(request, stream=True)
            response_id = generate_response_id()
            seq = 0

            # Emit response.created
            yield StreamEvent(
                type="response.created",
                sequence_number=seq,
                response={"id": response_id, "model": request.model, "status": "in_progress"},
            )
            seq += 1

            yield StreamEvent(
                type="response.in_progress",
                sequence_number=seq,
                response={"id": response_id, "model": request.model, "status": "in_progress"},
            )
            seq += 1

            current_item_index = 0
            current_content_index = 0
            accumulated_text = ""
            accumulated_fn_args = ""
            current_fn_name = ""
            current_fn_id = ""
            message_item_emitted = False
            fn_item_emitted = False
            input_tokens = 0
            output_tokens = 0

            async with self._http.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract usage if present
                    if "usage" in chunk and chunk["usage"]:
                        input_tokens = chunk["usage"].get("prompt_tokens", 0)
                        output_tokens = chunk["usage"].get("completion_tokens", 0)

                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})

                        # Function call via tool_calls
                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            for tc in tool_calls:
                                fn = tc.get("function", {})
                                tc_id = tc.get("id")

                                # New function call starting
                                if tc_id:
                                    # Close previous function call if any
                                    if fn_item_emitted:
                                        yield StreamEvent(
                                            type="response.function_call_arguments.done",
                                            sequence_number=seq,
                                            output_index=current_item_index,
                                            delta=accumulated_fn_args,
                                        )
                                        seq += 1
                                        yield StreamEvent(
                                            type="response.output_item.done",
                                            sequence_number=seq,
                                            output_index=current_item_index,
                                        )
                                        seq += 1
                                        current_item_index += 1
                                        accumulated_fn_args = ""

                                    current_fn_id = tc_id
                                    current_fn_name = fn.get("name", "")
                                    item_id = generate_item_id()
                                    yield StreamEvent(
                                        type="response.output_item.added",
                                        sequence_number=seq,
                                        output_index=current_item_index,
                                        item={
                                            "type": "function_call",
                                            "id": item_id,
                                            "call_id": current_fn_id,
                                            "name": current_fn_name,
                                            "arguments": "",
                                        },
                                    )
                                    seq += 1
                                    fn_item_emitted = True

                                # Argument delta
                                arg_delta = fn.get("arguments", "")
                                if arg_delta:
                                    accumulated_fn_args += arg_delta
                                    yield StreamEvent(
                                        type="response.function_call_arguments.delta",
                                        sequence_number=seq,
                                        output_index=current_item_index,
                                        delta=arg_delta,
                                    )
                                    seq += 1

                        # Text content
                        content = delta.get("content")
                        if content:
                            if not message_item_emitted:
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
                                message_item_emitted = True

                            accumulated_text += content
                            yield StreamEvent(
                                type="response.output_text.delta",
                                sequence_number=seq,
                                output_index=current_item_index,
                                content_index=current_content_index,
                                delta=content,
                            )
                            seq += 1

                        # Reasoning content (if provider supports it)
                        reasoning_content = delta.get("reasoning_content")
                        if reasoning_content:
                            yield StreamEvent(
                                type="response.reasoning_text.delta",
                                sequence_number=seq,
                                output_index=current_item_index,
                                delta=reasoning_content,
                            )
                            seq += 1

            # Close open function call
            if fn_item_emitted and accumulated_fn_args:
                yield StreamEvent(
                    type="response.function_call_arguments.done",
                    sequence_number=seq,
                    output_index=current_item_index,
                    delta=accumulated_fn_args,
                )
                seq += 1
                yield StreamEvent(
                    type="response.output_item.done",
                    sequence_number=seq,
                    output_index=current_item_index,
                )
                seq += 1
                current_item_index += 1

            # Close text message
            if message_item_emitted:
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
                yield StreamEvent(
                    type="response.output_item.done",
                    sequence_number=seq,
                    output_index=current_item_index,
                )
                seq += 1

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

        except httpx.HTTPStatusError as e:
            raise ProviderError(
                str(e),
                provider=self.provider_name,
                status_code=e.response.status_code,
                raw_error=e,
            ) from e
        except httpx.HTTPError as e:
            raise ProviderError(
                str(e),
                provider=self.provider_name,
                raw_error=e,
            ) from e

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internal: request translation
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_payload(
        self, request: CreateResponseRequest, *, stream: bool
    ) -> dict[str, Any]:
        """Translate unified request → Chat Completions payload."""
        payload: dict[str, Any] = {
            "model": request.model,
            "stream": stream,
        }

        # Translate input → messages
        messages = self._translate_input(request.input)

        # Prepend system message from instructions
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
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

        payload["messages"] = messages

        # Generation params
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        # Reasoning effort
        if request.reasoning:
            reasoning = request.reasoning
            if isinstance(reasoning, dict):
                effort = reasoning.get("effort")
            else:
                effort = reasoning.effort
            if effort:
                payload["reasoning_effort"] = effort

        # Tools → Chat Completions format
        if request.tools:
            cc_tools = self._translate_tools(request.tools)
            if cc_tools:
                payload["tools"] = cc_tools

                # Tool choice
                if request.tool_choice is not None:
                    payload["tool_choice"] = self._translate_tool_choice(
                        request.tool_choice
                    )

        # Include usage in streaming
        if stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

    def _translate_input(
        self, input_data: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate unified input format → Chat Completions messages array."""
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]

        messages: list[dict[str, Any]] = []
        for item in input_data:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            role = item.get("role")

            # Regular message
            if role and role in ("user", "assistant", "system", "developer"):
                msg_role = "system" if role in ("system", "developer") else role
                content = item.get("content", "")
                if isinstance(content, str):
                    messages.append({"role": msg_role, "content": content})
                elif isinstance(content, list):
                    # Flatten content blocks to text
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype in ("input_text", "output_text", "text"):
                                text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    messages.append({"role": msg_role, "content": " ".join(text_parts)})

            # Function call → assistant message with tool_calls
            elif item_type == "function_call":
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id", generate_call_id()),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }],
                })

            # Function call output → tool message
            elif item_type == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                })

        return messages or [{"role": "user", "content": ""}]

    def _translate_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate unified tool definitions → Chat Completions tools format."""
        cc_tools: list[dict[str, Any]] = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                cc_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
            # Skip non-function tools
        return cc_tools

    def _translate_tool_choice(
        self, tool_choice: str | dict[str, str]
    ) -> str | dict[str, Any]:
        """Translate unified tool_choice → Chat Completions format."""
        if isinstance(tool_choice, str):
            if tool_choice in ("auto", "required", "none"):
                return tool_choice
            return "auto"
        if isinstance(tool_choice, dict):
            name = (
                tool_choice.get("function", {}).get("name")
                if isinstance(tool_choice.get("function"), dict)
                else tool_choice.get("name")
            )
            if name:
                return {"type": "function", "function": {"name": name}}
        return "auto"

    # ------------------------------------------------------------------
    # Internal: response translation
    # ------------------------------------------------------------------

    def _convert_response(
        self, data: dict[str, Any], model: str
    ) -> Response:
        """Convert a Chat Completions response → unified Response."""
        output_items: list[dict[str, Any]] = []

        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            role = message.get("role", "assistant")

            # Text content
            content = message.get("content")
            if content:
                output_items.append({
                    "type": "message",
                    "id": generate_item_id(),
                    "role": role,
                    "content": [
                        {"type": "output_text", "text": content, "annotations": []}
                    ],
                    "status": "completed",
                })

            # Tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    output_items.append({
                        "type": "function_call",
                        "id": generate_item_id(),
                        "call_id": tc.get("id", generate_call_id()),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    })

            # Reasoning content (if the model returned it)
            reasoning = message.get("reasoning_content")
            if reasoning:
                output_items.append({
                    "type": "reasoning",
                    "id": generate_item_id(),
                    "summary": [{"type": "text", "text": reasoning}],
                })

        # Build usage
        usage = None
        usage_data = data.get("usage")
        if usage_data:
            inp = usage_data.get("prompt_tokens", 0)
            out = usage_data.get("completion_tokens", 0)
            usage = Usage(
                input_tokens=inp,
                output_tokens=out,
                total_tokens=usage_data.get("total_tokens", inp + out),
            )

        return Response(
            id=data.get("id", generate_response_id()),
            model=model,
            created_at=data.get("created", unix_timestamp()),
            status="completed",
            output=output_items,
            usage=usage,
        )
