from __future__ import annotations

import json

import pytest

from caip_responses.tool_handlers.shell import ShellHandler


class TestShellHandler:
    def test_tool_type(self):
        handler = ShellHandler()
        assert handler.tool_type() == "shell"

    def test_to_function_tools(self):
        handler = ShellHandler()
        tools = handler.to_function_tools({"type": "shell"})
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["name"] == "_builtin_shell_exec"
        assert "command" in tool["parameters"]["properties"]
        assert "command" in tool["parameters"]["required"]

    def test_is_synthetic(self):
        handler = ShellHandler()
        assert handler.is_synthetic("_builtin_shell_exec") is True
        assert handler.is_synthetic("_builtin_code_interpreter_run") is False

    @pytest.mark.asyncio
    async def test_execute_disabled_by_default(self):
        handler = ShellHandler()
        result = await handler.execute("_builtin_shell_exec", {"command": "echo hi"})
        data = json.loads(result)
        assert "error" in data
        assert "disabled" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_no_command(self):
        handler = ShellHandler(enabled=True)
        result = await handler.execute("_builtin_shell_exec", {})
        data = json.loads(result)
        assert "error" in data
        assert "No command" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_with_callback(self):
        """Custom executor callback should be called."""

        async def mock_executor(command: str) -> dict:
            return {"stdout": f"ran: {command}", "stderr": "", "exit_code": 0}

        handler = ShellHandler(enabled=True, executor_callback=mock_executor)
        result = await handler.execute("_builtin_shell_exec", {"command": "ls -la"})
        data = json.loads(result)
        assert data["stdout"] == "ran: ls -la"
        assert data["exit_code"] == 0

    def test_to_output_item(self):
        handler = ShellHandler()
        item = handler.to_output_item(
            "_builtin_shell_exec",
            {"command": "ls -la"},
            "{}",
        )
        assert item is not None
        assert item["type"] == "shell_call"
        assert item["status"] == "completed"
        assert item["action"]["type"] == "exec"
        assert item["action"]["command"] == "ls -la"

    def test_system_prompt_addendum(self):
        handler = ShellHandler()
        addendum = handler.system_prompt_addendum({"type": "shell"})
        assert addendum is not None
        assert "shell" in addendum.lower()

    def test_default_timeout(self):
        handler = ShellHandler()
        assert handler._timeout == 30

    def test_custom_timeout(self):
        handler = ShellHandler(timeout=120)
        assert handler._timeout == 120

    def test_custom_working_dir(self):
        handler = ShellHandler(working_dir="/tmp/test")
        assert handler._working_dir == "/tmp/test"

    @pytest.mark.asyncio
    async def test_execute_real_subprocess(self):
        """Test actual subprocess execution (simple, safe command)."""
        handler = ShellHandler(enabled=True)
        result = await handler.execute("_builtin_shell_exec", {"command": "echo hello"})
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert "hello" in data["stdout"]
