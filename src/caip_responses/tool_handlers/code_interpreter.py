from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.utils.id_gen import generate_item_id

# Type for a pluggable code executor:
#   async def my_executor(code: str, language: str) -> dict
#   Return dict should have: stdout, stderr, exit_code
CodeExecutorCallback = Callable[
    [str, str], Awaitable[dict[str, Any]]
]


class CodeInterpreterHandler(BuiltinToolHandler):
    """Client-side code interpreter for non-OpenAI providers.

    Converts ``{"type": "code_interpreter"}`` into a function the model
    can call to write and execute Python code.  Runs code in a subprocess
    with configurable timeout and working directory.

    **Security**: Disabled by default.  Must be explicitly enabled.

    Usage::

        handler = CodeInterpreterHandler(enabled=True, timeout=30)
        registry.register(handler)
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        timeout: int = 30,
        working_dir: str | None = None,
        executor_callback: CodeExecutorCallback | None = None,
    ) -> None:
        self._enabled = enabled
        self._timeout = timeout
        self._working_dir = working_dir
        self._executor_callback = executor_callback

    def tool_type(self) -> str:
        return "code_interpreter"

    def to_function_tools(
        self, tool_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": self._make_fn_name("run"),
                "description": (
                    "Execute Python code in a sandboxed environment. "
                    "The code runs in an isolated subprocess with access to "
                    "common libraries (pandas, numpy, matplotlib, etc. if installed). "
                    "Use print() to return results. "
                    "Files can be saved to the working directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute.",
                        },
                    },
                    "required": ["code"],
                },
            },
        ]

    def system_prompt_addendum(
        self, tool_config: dict[str, Any]
    ) -> str | None:
        return (
            "You have access to a Python code interpreter. "
            "You can write and execute Python code to perform calculations, "
            "data analysis, create visualizations, and more. "
            "Use print() statements to display results."
        )

    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        if not self._enabled:
            return json.dumps({
                "error": "Code interpreter is disabled. "
                "Enable it with CodeInterpreterHandler(enabled=True).",
            })

        code = arguments.get("code", "")
        if not code:
            return json.dumps({"error": "No code provided"})

        if self._executor_callback:
            result = await self._executor_callback(code, "python")
            return json.dumps(result)

        return await self._run_subprocess(code)

    def to_output_item(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> dict[str, Any] | None:
        return {
            "type": "code_interpreter_call",
            "id": generate_item_id(),
            "status": "completed",
            "code": arguments.get("code", ""),
        }

    async def _run_subprocess(self, code: str) -> str:
        """Execute Python code in a subprocess."""
        work_dir = self._working_dir or tempfile.mkdtemp(prefix="caip_code_")
        Path(work_dir).mkdir(parents=True, exist_ok=True)

        try:
            process = await asyncio.create_subprocess_exec(
                "python", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env={},  # Empty env for isolation
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout,
            )

            return json.dumps({
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": process.returncode,
            })

        except TimeoutError:
            process.kill()
            return json.dumps({
                "error": f"Code execution timed out after {self._timeout}s",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            })
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            })
