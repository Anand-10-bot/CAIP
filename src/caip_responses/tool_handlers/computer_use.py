"""Computer use tool handler.

Delegates computer/computer_use calls to Azure OpenAI, which handles
computer use (CUA) natively.
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


class ComputerUseHandler(BuiltinToolHandler, OpenAIDelegatorMixin):
    """Client-side computer use tool handler for non-OpenAI providers.

    Delegates to Azure OpenAI's built-in computer_use tool. Handles
    both "computer" and "computer_use" tool types.

    Usage::

        handler = ComputerUseHandler(
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
        openai_model: str = "computer-use-preview",
    ) -> None:
        BuiltinToolHandler.__init__(self)
        OpenAIDelegatorMixin.__init__(
            self,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
        )
        self._computer_config: dict[str, Any] = {}

    @property
    def metrics(self) -> DelegatedToolMetrics:
        return self._delegated_metrics

    def tool_type(self) -> str:
        return "computer_use"

    def to_function_tools(
        self, tool_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self._computer_config = tool_config
        return [
            {
                "type": "function",
                "name": self._make_fn_name("action"),
                "description": (
                    "Perform a computer action (click, type, scroll, "
                    "screenshot, etc.) on a virtual desktop environment."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": (
                                "Describe the computer action to perform "
                                "(e.g., 'click on the submit button', "
                                "'type hello world in the search box')."
                            ),
                        },
                    },
                    "required": ["instruction"],
                },
            },
        ]

    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        instruction = arguments.get("instruction", "")
        if not instruction:
            return json.dumps({"error": "No instruction provided"})

        if not self._openai_api_key:
            return json.dumps({
                "error": "computer_use tool requires openai_api_key for delegation",
            })

        client = self._get_openai_client()

        # Build tool config — use "computer" type for OpenAI
        tool_def: dict[str, Any] = {"type": "computer_use"}
        if "display_width" in self._computer_config:
            tool_def["display_width"] = self._computer_config["display_width"]
        if "display_height" in self._computer_config:
            tool_def["display_height"] = self._computer_config["display_height"]
        if "environment" in self._computer_config:
            tool_def["environment"] = self._computer_config["environment"]

        response = await client.responses.create(
            model=self._openai_model,
            input=instruction,
            tools=[tool_def],
        )

        usage = self._record_usage(response, "computer_use")
        text = self._extract_text_from_response(response)

        # Extract computer_call items
        actions = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "computer_call":
                item_actions = getattr(item, "actions", [])
                if hasattr(item_actions, "__iter__"):
                    for a in item_actions:
                        if hasattr(a, "model_dump"):
                            actions.append(a.model_dump())
                        elif isinstance(a, dict):
                            actions.append(a)

        result: dict[str, Any] = {"result": text}
        if actions:
            result["actions"] = actions
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
            "type": "computer_call",
            "id": generate_item_id(),
            "call_id": generate_item_id(),
            "actions": [],
            "status": "completed",
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

    def system_prompt_addendum(
        self, tool_config: dict[str, Any]
    ) -> str | None:
        return (
            "You have access to a computer desktop environment. You can "
            f"use the '{self._make_fn_name('action')}' function to perform "
            "actions like clicking, typing, scrolling, and taking screenshots."
        )
