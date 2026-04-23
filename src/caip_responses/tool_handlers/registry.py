from __future__ import annotations

from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler


class BuiltinToolRegistry:
    """Registry of client-side built-in tool handlers.

    Maps tool types (``web_search``, ``code_interpreter``, ``shell``, etc.)
    to their handler instances.  Used by the ``AgentLoop`` and client layer
    to intercept tool calls destined for emulated tools.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, BuiltinToolHandler] = {}

    def register(self, handler: BuiltinToolHandler) -> None:
        """Register a handler for its tool type."""
        self._handlers[handler.tool_type()] = handler

    def get(self, tool_type: str) -> BuiltinToolHandler | None:
        """Get the handler for a tool type, or None."""
        return self._handlers.get(tool_type)

    def can_handle(self, tool_type: str) -> bool:
        """Check if a handler exists for the given tool type."""
        return tool_type in self._handlers

    def resolve_for_function(
        self, fn_name: str
    ) -> BuiltinToolHandler | None:
        """Find the handler that owns a synthetic function name."""
        for handler in self._handlers.values():
            if handler.is_synthetic(fn_name):
                return handler
        return None

    def preprocess_tools(
        self,
        tools: list[dict[str, Any]],
        provider_supports: set[str],
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, Any]]]:
        """Preprocess tools: replace unsupported built-in tools with synthetic functions.

        Args:
            tools: Original tool definitions from the user's request.
            provider_supports: Set of tool types the provider natively supports.

        Returns:
            A tuple of:
            - processed_tools: New tools list with synthetic functions replacing
              unsupported built-in tools.
            - addenda: System prompt additions from handlers.
            - tool_configs: Map of tool_type → original tool config for handlers.
        """
        processed: list[dict[str, Any]] = []
        addenda: list[str] = []
        configs: dict[str, dict[str, Any]] = {}

        for tool in tools:
            tool_type = tool.get("type", "function")

            if tool_type in provider_supports:
                # Provider handles this natively
                processed.append(tool)
            elif self.can_handle(tool_type):
                # Replace with synthetic function tools
                handler = self._handlers[tool_type]
                synthetic = handler.to_function_tools(tool)
                processed.extend(synthetic)
                configs[tool_type] = tool

                addendum = handler.system_prompt_addendum(tool)
                if addendum:
                    addenda.append(addendum)
            else:
                # Unsupported and no handler — pass through (provider may skip)
                processed.append(tool)

        return processed, addenda, configs

    @property
    def registered_types(self) -> list[str]:
        """List of tool types that have registered handlers."""
        return list(self._handlers.keys())

    def __bool__(self) -> bool:
        return bool(self._handlers)
