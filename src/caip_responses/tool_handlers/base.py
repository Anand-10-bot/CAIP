from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BuiltinToolHandler(ABC):
    """Base class for client-side built-in tool emulation.

    Each handler converts a non-function tool type (web_search,
    code_interpreter, shell, etc.) into function-call definitions that
    any model can invoke via standard function calling.  The handler
    also knows how to execute those calls and return results.

    Lifecycle during a request:
    1. ``to_function_tools()`` — generate synthetic function definitions
    2. Model calls the synthetic function(s) via function calling
    3. ``execute()`` — run the call and return a result string
    4. (Optional) ``to_output_item()`` — convert the result to the
       native output item type for API parity (e.g. web_search_call)
    """

    # Prefix added to synthetic function names to avoid collision with
    # user-defined functions.  Handlers should use ``_make_fn_name()``
    # to generate names.
    _PREFIX = "_builtin_"

    @abstractmethod
    def tool_type(self) -> str:
        """The built-in tool type this handler emulates (e.g. 'web_search')."""

    @abstractmethod
    def to_function_tools(
        self, tool_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert the built-in tool definition into function-call definitions.

        Args:
            tool_config: The original tool dict from the user's request
                         (e.g. ``{"type": "web_search", "search_context_size": "high"}``).

        Returns:
            A list of function-tool dicts that will be injected into the
            tools list sent to the provider.
        """

    @abstractmethod
    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a synthetic function call and return the result.

        Args:
            name: The synthetic function name the model called.
            arguments: Parsed arguments dict from the model.

        Returns:
            Result string to feed back to the model as function_call_output.
        """

    def system_prompt_addendum(
        self, tool_config: dict[str, Any]
    ) -> str | None:
        """Optional extra text to inject into the system prompt.

        Some tools need the model to understand a protocol (e.g. computer
        use).  Return text to prepend to the instructions, or None.
        """
        return None

    def to_output_item(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> dict[str, Any] | None:
        """Convert a completed call to a native output item dict.

        Override this to produce ``web_search_call``, ``shell_call``,
        etc. in the response output for API parity.  Return None to
        keep the raw ``function_call`` item as-is.
        """
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_fn_name(self, suffix: str) -> str:
        """Build a prefixed synthetic function name."""
        return f"{self._PREFIX}{self.tool_type()}_{suffix}"

    def is_synthetic(self, fn_name: str) -> bool:
        """Check if a function name belongs to this handler."""
        return fn_name.startswith(f"{self._PREFIX}{self.tool_type()}_")
