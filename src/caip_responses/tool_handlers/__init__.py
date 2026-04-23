"""Client-side built-in tool handlers for non-OpenAI providers.

When a provider does not natively support a tool type (web_search,
code_interpreter, shell, mcp, file_search, image_generation, computer_use),
these handlers convert the tool into function-call definitions the model
can use, execute the calls (delegating to Azure OpenAI where needed),
and return results — making every tool available with every LLM.
"""

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.tool_handlers.openai_delegator import (
    DelegatedToolMetrics,
    DelegatedToolUsage,
)
from caip_responses.tool_handlers.registry import BuiltinToolRegistry
from caip_responses.tool_handlers.web_search import WebSearchMetrics, WebSearchUsage

__all__ = [
    "BuiltinToolHandler",
    "BuiltinToolRegistry",
    "DelegatedToolMetrics",
    "DelegatedToolUsage",
    "WebSearchMetrics",
    "WebSearchUsage",
]
