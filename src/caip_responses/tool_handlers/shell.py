from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.utils.id_gen import generate_item_id

logger = logging.getLogger(__name__)

# Default set of safe commands allowed when no custom allowlist is provided.
# Callers should provide their own allowlist for production use.
DEFAULT_COMMAND_ALLOWLIST: frozenset[str] = frozenset({
    "cat",
    "date",
    "df",
    "du",
    "echo",
    "env",
    "find",
    "grep",
    "head",
    "hostname",
    "ls",
    "pwd",
    "tail",
    "uname",
    "wc",
    "which",
    "whoami",
})

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
    When enabled, commands are validated against an allowlist before
    execution.  Every command attempt (allowed or denied) is audit-logged.

    Usage::

        handler = ShellHandler(enabled=True, timeout=30)
        registry.register(handler)

        # Custom allowlist:
        handler = ShellHandler(
            enabled=True,
            command_allowlist={"ls", "cat", "grep", "python"},
        )
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        timeout: int = 30,
        working_dir: str | None = None,
        executor_callback: ShellExecutorCallback | None = None,
        command_allowlist: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._enabled = enabled
        self._timeout = timeout
        self._working_dir = working_dir
        self._executor_callback = executor_callback
        self._command_allowlist: frozenset[str] = (
            frozenset(command_allowlist)
            if command_allowlist is not None
            else DEFAULT_COMMAND_ALLOWLIST
        )

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

        # --- Allowlist validation ---
        allowed, base_command = self._is_command_allowed(command)
        timestamp = datetime.now(UTC).isoformat()

        if not allowed:
            logger.warning(
                "Shell command DENIED: command=%r base_command=%r "
                "timestamp=%s allowlist=%s",
                command,
                base_command,
                timestamp,
                sorted(self._command_allowlist),
            )
            return json.dumps({
                "error": (
                    f"Command '{base_command}' is not in the allowed "
                    f"commands list. Allowed commands: "
                    f"{sorted(self._command_allowlist)}"
                ),
            })

        logger.info(
            "Shell command ALLOWED: command=%r base_command=%r timestamp=%s",
            command,
            base_command,
            timestamp,
        )

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

    def _is_command_allowed(self, command: str) -> tuple[bool, str]:
        """Check whether *command* starts with an allowed base command.

        Extracts the first token (the executable name) from the command
        string using ``shlex.split`` and checks it against the allowlist.
        Pipe chains and logical operators (``|``, ``&&``, ``||``, ``;``)
        cause each segment to be validated — ALL segments must be allowed.

        Returns:
            A tuple of (is_allowed, base_command_that_failed_or_first).
        """
        # Split on shell operators to catch chained commands
        # e.g. "ls -la && rm -rf /" -> ["ls -la", "rm -rf /"]
        segments = re.split(r"\s*(?:\|\||&&|[;|])\s*", command)

        first_base = ""
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            try:
                tokens = shlex.split(segment)
            except ValueError:
                # Malformed shell quoting — reject with the raw segment
                return False, segment.split()[0] if segment.split() else segment
            if not tokens:
                continue
            base = tokens[0]
            if not first_base:
                first_base = base
            if base not in self._command_allowlist:
                return False, base

        return True, first_base

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
