from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.utils.id_gen import generate_item_id

# Type for a pluggable shell executor:
#   async def my_executor(command: str) -> dict
#   Return dict should have: stdout, stderr, exit_code
ShellExecutorCallback = Callable[
    [str], Awaitable[dict[str, Any]]
]


class ShellHandler(BuiltinToolHandler):
    """Client-side shell tool for non-OpenAI providers.

    Converts ``{"type": "shell"}`` into a function the model can call
    to execute shell commands.  Runs commands in a subprocess with
    configurable timeout and working directory.

    **Security**: Disabled by default.  Must be explicitly enabled.

    Usage::

        handler = ShellHandler(enabled=True, timeout=30)
        registry.register(handler)
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        timeout: int = 30,
        working_dir: str | None = None,
        executor_callback: ShellExecutorCallback | None = None,
    ) -> None:
        self._enabled = enabled
        self._timeout = timeout
        self._working_dir = working_dir
        self._executor_callback = executor_callback

    def tool_type(self) -> str:
        return "shell"

    def to_function_tools(
        self, tool_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": self._make_fn_name("exec"),
                "description": (
                    "Execute a shell command in a terminal environment. "
                    "Returns stdout, stderr, and the exit code. "
                    "Use this for file operations, system commands, "
                    "running scripts, and interacting with CLI tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        },
                    },
                    "required": ["command"],
                },
            },
        ]

    def system_prompt_addendum(
        self, tool_config: dict[str, Any]
    ) -> str | None:
        return (
            "You have access to a shell terminal. "
            "You can execute commands to interact with the filesystem, "
            "run scripts, install packages, and perform system operations. "
            "Commands run non-interactively."
        )

    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        if not self._enabled:
            return json.dumps({
                "error": "Shell tool is disabled. "
                "Enable it with ShellHandler(enabled=True).",
            })

        command = arguments.get("command", "")
        if not command:
            return json.dumps({"error": "No command provided"})

        if self._executor_callback:
            result = await self._executor_callback(command)
            return json.dumps(result)

        return await self._run_subprocess(command)

    def to_output_item(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> dict[str, Any] | None:
        call_item = {
            "type": "shell_call",
            "id": generate_item_id(),
            "status": "completed",
            "call_id": "",
            "action": {
                "type": "exec",
                "command": arguments.get("command", ""),
            },
        }
        return call_item

    async def _run_subprocess(self, command: str) -> str:
        """Execute a shell command in a subprocess."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
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
                "error": f"Command timed out after {self._timeout}s",
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
