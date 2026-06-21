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
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

# Tool-choice mapping: unified → Gemini
_TOOL_CHOICE_MAP: dict[str, str] = {
    "auto": "AUTO",
    "required": "ANY",
    "none": "NONE",
}


class GeminiProvider(BaseProvider):
    """Google Gemini provider — translates Responses API to GenerateContent API.

    Uses the google-genai SDK (unified SDK). The caller's code is identical
    to calling any other provider; all translation happens internally.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        vertexai: bool = False,
        project: str | None = None,
        location: str = "us-central1",
        service_account_path: str | None = None,
        service_account_info: dict[str, Any] | None = None,
    ) -> None:
        """Create a Gemini provider.

        Two auth modes, selected automatically:

        - **Developer API (AI Studio)** — pass ``api_key``. Simplest; uses
          ``generativelanguage.googleapis.com``.
        - **Vertex AI (Google Cloud)** — pass a service account via
          ``service_account_path`` (path to the JSON) or
          ``service_account_info`` (the parsed dict), or set ``vertexai=True``
          to use Application Default Credentials. Requires a GCP project with
          the Vertex AI API enabled and billing active. ``project`` is read
          from the service account JSON if not given; ``location`` defaults to
          ``us-central1``.
        """
        if genai is None:
            raise ImportError(
                "google-genai package is required for GeminiProvider. "
                "Install it with: pip install caip-responses-lib[gemini]"
            )

        use_vertex = vertexai or bool(service_account_path or service_account_info)
        if use_vertex:
            credentials = None
            if service_account_path or service_account_info:
                from google.oauth2 import service_account as _sa

                scopes = ["https://www.googleapis.com/auth/cloud-platform"]
                if service_account_info is not None:
                    credentials = _sa.Credentials.from_service_account_info(
                        service_account_info, scopes=scopes
                    )
                    project = project or service_account_info.get("project_id")
                else:
                    credentials = _sa.Credentials.from_service_account_file(
                        service_account_path, scopes=scopes
                    )
                    if project is None:
                        with open(service_account_path) as f:
                            project = json.load(f).get("project_id")
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=credentials,
            )
        else:
            self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def supports_tool(self, tool_type: str) -> bool:
        # web_search maps natively to Gemini's Google Search grounding tool,
        # so it does NOT need the client-side WebSearchHandler fallback.
        return tool_type in {"function", "web_search"}

    def supports_reasoning(self) -> bool:
        return True

    async def create_response(
        self, request: CreateResponseRequest
    ) -> Response:
        """Non-streaming response creation via Gemini GenerateContent."""
        try:
            kwargs = self._build_kwargs(request)
            result = await self._client.aio.models.generate_content(**kwargs)
            return self._convert_response(result, request.model)
        except Exception as e:
            if _is_genai_error(e):
                raise ProviderError(
                    str(e),
                    provider="gemini",
                    status_code=getattr(e, "code", None),
                    raw_error=e,
                ) from e
            raise

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        """Streaming response creation — translates Gemini SSE to unified events."""
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

            yield StreamEvent(
                type="response.in_progress",
                sequence_number=seq,
                response={"id": response_id, "model": request.model, "status": "in_progress"},
            )
            seq += 1

            current_item_index = 0
            current_content_index = 0
            accumulated_text = ""
            message_item_emitted = False
            input_tokens = 0
            output_tokens = 0
            grounding_search_items: list[dict[str, Any]] = []

            async for chunk in await self._client.aio.models.generate_content_stream(**kwargs):
                # Extract usage from metadata
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    meta = chunk.usage_metadata
                    input_tokens = getattr(meta, "prompt_token_count", 0) or 0
                    output_tokens = getattr(meta, "candidates_token_count", 0) or 0

                if not chunk.candidates:
                    continue

                for candidate in chunk.candidates:
                    # Capture Google Search grounding (usually on the final
                    # chunk, which may carry no content parts of its own).
                    search_items, _ = _extract_grounding(candidate)
                    if search_items:
                        grounding_search_items = search_items

                    if not candidate.content or not candidate.content.parts:
                        continue

                    for part in candidate.content.parts:
                        # Function call part
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            call_id = generate_call_id()
                            item_id = generate_item_id()
                            fn_args = _extract_function_args(fc)

                            yield StreamEvent(
                                type="response.output_item.added",
                                sequence_number=seq,
                                output_index=current_item_index,
                                item={
                                    "type": "function_call",
                                    "id": item_id,
                                    "call_id": call_id,
                                    "name": fc.name,
                                    "arguments": "",
                                },
                            )
                            seq += 1

                            # Gemini sends complete function calls (not streamed deltas)
                            yield StreamEvent(
                                type="response.function_call_arguments.delta",
                                sequence_number=seq,
                                output_index=current_item_index,
                                delta=fn_args,
                            )
                            seq += 1

                            yield StreamEvent(
                                type="response.function_call_arguments.done",
                                sequence_number=seq,
                                output_index=current_item_index,
                                delta=fn_args,
                            )
                            seq += 1

                            yield StreamEvent(
                                type="response.output_item.done",
                                sequence_number=seq,
                                output_index=current_item_index,
                            )
                            seq += 1
                            current_item_index += 1

                        # Thought / reasoning — check before text because
                        # thinking parts also have a .text attribute.
                        elif hasattr(part, "thought") and part.thought:
                            reasoning_text = getattr(part, "text", "") or ""
                            reasoning_id = generate_item_id()
                            yield StreamEvent(
                                type="response.output_item.added",
                                sequence_number=seq,
                                output_index=current_item_index,
                                item={"type": "reasoning", "id": reasoning_id},
                            )
                            seq += 1
                            if reasoning_text:
                                yield StreamEvent(
                                    type="response.reasoning_text.delta",
                                    sequence_number=seq,
                                    output_index=current_item_index,
                                    delta=reasoning_text,
                                )
                                seq += 1
                            yield StreamEvent(
                                type="response.output_item.done",
                                sequence_number=seq,
                                output_index=current_item_index,
                            )
                            seq += 1
                            current_item_index += 1

                        # Text part
                        elif hasattr(part, "text") and part.text is not None:
                            text = part.text

                            # Emit message item + content part on first text
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

                            accumulated_text += text
                            yield StreamEvent(
                                type="response.output_text.delta",
                                sequence_number=seq,
                                output_index=current_item_index,
                                content_index=current_content_index,
                                delta=text,
                            )
                            seq += 1

            # Close the text message if we emitted one
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
                current_item_index += 1

            # Surface Google Search grounding as web_search_call items
            for search_item in grounding_search_items:
                yield StreamEvent(
                    type="response.output_item.added",
                    sequence_number=seq,
                    output_index=current_item_index,
                    item=search_item,
                )
                seq += 1
                yield StreamEvent(
                    type="response.output_item.done",
                    sequence_number=seq,
                    output_index=current_item_index,
                )
                seq += 1
                current_item_index += 1

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
            if _is_genai_error(e):
                raise ProviderError(
                    str(e),
                    provider="gemini",
                    status_code=getattr(e, "code", None),
                    raw_error=e,
                ) from e
            raise

    async def close(self) -> None:
        pass  # google-genai Client has no close method

    # ------------------------------------------------------------------
    # Internal: request translation
    # ------------------------------------------------------------------

    def _build_kwargs(self, request: CreateResponseRequest) -> dict[str, Any]:
        """Translate unified request → Gemini GenerateContent kwargs."""
        kwargs: dict[str, Any] = {"model": request.model}

        # Translate input → contents
        contents = self._translate_input(request.input)
        kwargs["contents"] = contents

        # System instruction
        system_parts: list[str] = []
        if request.instructions:
            system_parts.append(request.instructions)

        # Structured output: inject schema into system prompt (Gemini also
        # supports native response_mime_type, but system prompt is more reliable)
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
            kwargs["config"] = kwargs.get("config", {})
            if isinstance(kwargs["config"], dict):
                kwargs["config"]["system_instruction"] = "\n\n".join(system_parts)

        # Generation config
        config = kwargs.get("config", {})
        if not isinstance(config, dict):
            config = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.max_output_tokens is not None:
            config["max_output_tokens"] = request.max_output_tokens

        # Reasoning / thinking (Gemini 2.0+ supports thinking)
        if request.reasoning:
            reasoning = request.reasoning
            if isinstance(reasoning, dict):
                effort = reasoning.get("effort")
            else:
                effort = reasoning.effort
            if effort:
                config["thinking_config"] = {"thinking_budget": _effort_to_budget(effort)}

        # Tools → Gemini tools list. Function tools become
        # function_declarations; a web_search tool maps to Gemini's native
        # Google Search grounding tool ({"google_search": {}}).
        if request.tools:
            gemini_tools: list[dict[str, Any]] = []

            declarations = self._translate_tools(request.tools)
            if declarations:
                gemini_tools.append({"function_declarations": declarations})

            if any(
                isinstance(t, dict) and t.get("type") == "web_search"
                for t in request.tools
            ):
                gemini_tools.append({"google_search": {}})

            if gemini_tools:
                config["tools"] = gemini_tools
                kwargs["config"] = config

                # Tool choice only applies to function calling, not grounding.
                if declarations and request.tool_choice is not None:
                    tc_mode = self._translate_tool_choice(request.tool_choice)
                    if tc_mode:
                        config["tool_config"] = {
                            "function_calling_config": {"mode": tc_mode}
                        }

        if config:
            kwargs["config"] = config

        return kwargs

    def _translate_input(
        self, input_data: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate unified input format → Gemini contents array."""
        if isinstance(input_data, str):
            return [{"role": "user", "parts": [{"text": input_data}]}]

        contents: list[dict[str, Any]] = []
        # Track call_id → function name so function_call_output can use the
        # correct name (Gemini requires the function name, not the call_id).
        call_id_to_name: dict[str, str] = {}

        for item in input_data:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            role = item.get("role")

            # Regular message
            if role and role in ("user", "assistant"):
                gemini_role = "model" if role == "assistant" else "user"
                content = item.get("content", "")
                if isinstance(content, str):
                    contents.append({
                        "role": gemini_role,
                        "parts": [{"text": content}],
                    })
                elif isinstance(content, list):
                    parts: list[dict[str, Any]] = []
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype in ("input_text", "output_text", "text"):
                                parts.append({"text": block.get("text", "")})
                            elif btype == "input_image":
                                url = block.get("image_url", "")
                                if url:
                                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": url}})
                        elif isinstance(block, str):
                            parts.append({"text": block})
                    if parts:
                        contents.append({"role": gemini_role, "parts": parts})

            # system/developer → skip (handled via system_instruction)
            elif role in ("system", "developer"):
                continue

            # Function call → model turn with function_call part
            elif item_type == "function_call":
                fn_name = item.get("name", "")
                call_id = item.get("call_id", "")
                if call_id and fn_name:
                    call_id_to_name[call_id] = fn_name
                try:
                    args = json.loads(item.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                contents.append({
                    "role": "model",
                    "parts": [{
                        "function_call": {
                            "name": fn_name,
                            "args": args,
                        },
                    }],
                })

            # Function call output → user turn with function_response part
            elif item_type == "function_call_output":
                call_id = item.get("call_id", "")
                fn_name = call_id_to_name.get(call_id, call_id or "function")
                output = item.get("output", "")
                # Gemini requires function_response.response to be a JSON
                # object. Parse the output, but wrap any non-dict result
                # (number, string, bool, list) so it stays a valid object.
                try:
                    parsed = json.loads(output)
                    response_data = parsed if isinstance(parsed, dict) else {"result": parsed}
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": output}
                contents.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": fn_name,
                            "response": response_data,
                        },
                    }],
                })

        return contents or [{"role": "user", "parts": [{"text": ""}]}]

    def _translate_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate unified tool definitions → Gemini function_declarations."""
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                decl: dict[str, Any] = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
                params = tool.get("parameters", {})
                if params:
                    decl["parameters"] = _sanitize_schema(params)
                declarations.append(decl)
            # Skip non-function tools (web_search, mcp, etc.)
        return declarations

    def _translate_tool_choice(
        self, tool_choice: str | dict[str, str]
    ) -> str | None:
        """Translate unified tool_choice → Gemini function_calling_config mode."""
        if isinstance(tool_choice, str):
            return _TOOL_CHOICE_MAP.get(tool_choice, "AUTO")
        if isinstance(tool_choice, dict):
            # Specific function → ANY (Gemini doesn't support picking one function)
            return "ANY"
        return "AUTO"

    # ------------------------------------------------------------------
    # Internal: response translation
    # ------------------------------------------------------------------

    def _convert_response(self, result: Any, model: str) -> Response:
        """Convert a Gemini GenerateContentResponse → unified Response."""
        output_items: list[dict[str, Any]] = []
        text_parts: list[dict[str, Any]] = []
        search_items: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []

        # Extract parts from first candidate
        candidates = getattr(result, "candidates", []) or []
        if candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []

            # Google Search grounding → web_search_call items + citations
            search_items, annotations = _extract_grounding(candidate)

            for part in parts:
                # Function call
                if hasattr(part, "function_call") and part.function_call:
                    # Flush text first
                    if text_parts:
                        output_items.append({
                            "type": "message",
                            "id": generate_item_id(),
                            "role": "assistant",
                            "content": text_parts,
                            "status": "completed",
                        })
                        text_parts = []

                    fc = part.function_call
                    fn_args = _extract_function_args(fc)
                    output_items.append({
                        "type": "function_call",
                        "id": generate_item_id(),
                        "call_id": generate_call_id(),
                        "name": fc.name,
                        "arguments": fn_args,
                    })

                # Thought / reasoning (Gemini 2.0+ thinking) — check before
                # text because thinking parts also have a .text attribute.
                elif hasattr(part, "thought") and part.thought:
                    output_items.append({
                        "type": "reasoning",
                        "id": generate_item_id(),
                        "summary": [{"type": "text", "text": part.text or ""}] if hasattr(part, "text") and part.text else None,
                    })

                # Text
                elif hasattr(part, "text") and part.text is not None:
                    text_parts.append({
                        "type": "output_text",
                        "text": part.text,
                        "annotations": [],
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

        # Attach grounding citations to the answer text, and surface the
        # search itself as web_search_call item(s) ahead of the message —
        # mirroring how OpenAI's native web_search appears in the output.
        if annotations:
            for item in output_items:
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            block["annotations"] = annotations
        if search_items:
            output_items = search_items + output_items

        # Build usage
        usage = None
        usage_meta = getattr(result, "usage_metadata", None)
        if usage_meta:
            inp = getattr(usage_meta, "prompt_token_count", 0) or 0
            out = getattr(usage_meta, "candidates_token_count", 0) or 0
            usage = Usage(
                input_tokens=inp,
                output_tokens=out,
                total_tokens=inp + out,
            )

        return Response(
            id=generate_response_id(),
            model=model,
            created_at=unix_timestamp(),
            status="completed",
            output=output_items,
            usage=usage,
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _extract_grounding(
    candidate: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract Google Search grounding from a Gemini candidate.

    Returns a tuple of:
    - ``web_search_call`` output items (one per search query) for API parity
      with OpenAI's native web_search.
    - ``url_citation`` annotations built from the grounding chunks.
    """
    meta = getattr(candidate, "grounding_metadata", None)
    if not meta:
        return [], []

    search_items: list[dict[str, Any]] = []
    queries = getattr(meta, "web_search_queries", None)
    if isinstance(queries, (list, tuple)):
        for query in queries:
            search_items.append({
                "type": "web_search_call",
                "id": generate_item_id(),
                "status": "completed",
                "action": {"type": "search", "query": query},
            })

    annotations: list[dict[str, Any]] = []
    chunks = getattr(meta, "grounding_chunks", None)
    if isinstance(chunks, (list, tuple)):
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue
            url = getattr(web, "uri", "") or ""
            if not url:
                continue
            annotations.append({
                "type": "url_citation",
                "url": url,
                "title": getattr(web, "title", "") or "",
            })

    return search_items, annotations


def _effort_to_budget(effort: str) -> int:
    """Map reasoning effort → Gemini thinking budget tokens."""
    return {"low": 1024, "medium": 4096, "high": 16384}.get(effort, 4096)


# JSON Schema meta-keywords that Gemini's function_declarations parser
# rejects. Tools sourced from MCP servers (or rich user schemas) often
# include these, so they must be stripped before translation.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "$schema", "$id", "$ref", "$defs", "$comment",
    "definitions", "additionalProperties",
})


def _sanitize_schema(schema: Any) -> Any:
    """Recursively drop schema keywords Gemini doesn't accept.

    Gemini's function parameter schema is a restricted OpenAPI subset; keys
    like ``$schema`` and ``additionalProperties`` raise validation errors in
    the google-genai SDK. This walks nested objects/arrays and removes them.
    """
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema(v)
            for k, v in schema.items()
            if k not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema(v) for v in schema]
    return schema


def _extract_function_args(fc: Any) -> str:
    """Extract function arguments as a JSON string from a Gemini FunctionCall."""
    args = getattr(fc, "args", None)
    if args is None:
        return "{}"
    if isinstance(args, dict):
        return json.dumps(args)
    if isinstance(args, str):
        return args
    # google-genai may return a proto MapComposite — convert to dict
    try:
        return json.dumps(dict(args))
    except (TypeError, ValueError):
        return str(args)


def _is_genai_error(e: Exception) -> bool:
    """Check if an exception is from the google-genai library."""
    module = getattr(type(e), "__module__", "") or ""
    return "google" in module and "genai" in module
