from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

# Type alias for user-provided tool handlers:
#   async def handler(name: str, arguments: dict) -> str
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


class ToolExecutor:
    """Dispatches function calls to user-registered callbacks.

    Users register named handlers; when the model returns a function_call,
    the executor invokes the matching handler and returns the output as a
    function_call_output item ready to be appended to the conversation.
    """

    def __init__(self, handlers: dict[str, ToolHandler] | None = None) -> None:
        self._handlers: dict[str, ToolHandler] = dict(handlers) if handlers else {}
        self._default_handler: ToolHandler | None = None

    def register(self, name: str, handler: ToolHandler) -> None:
        """Register a handler for a specific function name."""
        self._handlers[name] = handler

    def set_default_handler(self, handler: ToolHandler) -> None:
        """Set a catch-all handler for unregistered function names."""
        self._default_handler = handler

    async def execute(
        self, call_id: str, name: str, arguments: str
    ) -> dict[str, Any]:
        """Execute a function call and return a function_call_output item.

        Args:
            call_id: The function call ID to correlate with the output.
            name: Function name requested by the model.
            arguments: JSON string of arguments from the model.

        Returns:
            A dict matching the function_call_output item schema.
        """
        # Parse arguments
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        # Find handler
        handler = self._handlers.get(name, self._default_handler)

        if handler is None:
            output = json.dumps({
                "error": f"No handler registered for function '{name}'"
            })
        else:
            try:
                result = await handler(name, args)
                output = result if isinstance(result, str) else json.dumps(result)
            except Exception as exc:
                output = json.dumps({"error": str(exc)})

        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }

    async def execute_many(
        self, function_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute multiple function calls and return their outputs.

        Args:
            function_calls: List of function_call items (dicts with
                call_id, name, arguments keys).

        Returns:
            List of function_call_output items in the same order.
        """
        outputs: list[dict[str, Any]] = []
        for fc in function_calls:
            result = await self.execute(
                call_id=fc.get("call_id", ""),
                name=fc.get("name", ""),
                arguments=fc.get("arguments", "{}"),
            )
            outputs.append(result)
        return outputs
