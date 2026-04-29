from __future__ import annotations

import json
import logging

import pytest

from caip_responses.tool_handlers.shell import DEFAULT_COMMAND_ALLOWLIST, ShellHandler


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
        """Custom executor callback should be called for allowed commands."""

        async def mock_executor(command: str) -> dict:
            return {"stdout": f"ran: {command}", "stderr": "", "exit_code": 0}

        handler = ShellHandler(
            enabled=True,
            executor_callback=mock_executor,
            command_allowlist={"ls"},
        )
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


class TestShellCommandAllowlist:
    """Tests for H-22: command allowlist enforcement."""

    def test_default_allowlist_applied(self):
        """Handler uses DEFAULT_COMMAND_ALLOWLIST when none provided."""
        handler = ShellHandler(enabled=True)
        assert handler._command_allowlist == DEFAULT_COMMAND_ALLOWLIST

    def test_custom_allowlist(self):
        """Custom allowlist overrides the default."""
        handler = ShellHandler(enabled=True, command_allowlist={"python", "node"})
        assert handler._command_allowlist == frozenset({"python", "node"})

    def test_empty_allowlist_denies_all(self):
        """An empty allowlist denies every command."""
        handler = ShellHandler(enabled=True, command_allowlist=set())
        assert handler._command_allowlist == frozenset()

    @pytest.mark.asyncio
    async def test_allowed_command_executes(self):
        """A command whose base is in the allowlist should execute."""
        handler = ShellHandler(enabled=True, command_allowlist={"echo"})
        result = await handler.execute(
            "_builtin_shell_exec", {"command": "echo hello"}
        )
        data = json.loads(result)
        assert "error" not in data
        assert data["exit_code"] == 0
        assert "hello" in data["stdout"]

    @pytest.mark.asyncio
    async def test_denied_command_returns_error(self):
        """A command not in the allowlist should be rejected."""
        handler = ShellHandler(enabled=True, command_allowlist={"echo", "ls"})
        result = await handler.execute(
            "_builtin_shell_exec", {"command": "rm -rf /"}
        )
        data = json.loads(result)
        assert "error" in data
        assert "rm" in data["error"]
        assert "not in the allowed" in data["error"]

    @pytest.mark.asyncio
    async def test_pipe_chain_all_allowed(self):
        """Piped commands should all be validated."""
        handler = ShellHandler(enabled=True, command_allowlist={"echo", "grep"})
        result = await handler.execute(
            "_builtin_shell_exec", {"command": "echo hello | grep hello"}
        )
        data = json.loads(result)
        assert "error" not in data

    @pytest.mark.asyncio
    async def test_pipe_chain_second_denied(self):
        """If the second command in a pipe is not allowed, reject."""
        handler = ShellHandler(enabled=True, command_allowlist={"echo"})
        result = await handler.execute(
            "_builtin_shell_exec", {"command": "echo hello | rm foo"}
        )
        data = json.loads(result)
        assert "error" in data
        assert "rm" in data["error"]

    @pytest.mark.asyncio
    async def test_and_chain_denied(self):
        """Commands chained with && must all be allowed."""
        handler = ShellHandler(enabled=True, command_allowlist={"ls"})
        result = await handler.execute(
            "_builtin_shell_exec", {"command": "ls && curl evil.com"}
        )
        data = json.loads(result)
        assert "error" in data
        assert "curl" in data["error"]

    @pytest.mark.asyncio
    async def test_semicolon_chain_denied(self):
        """Commands chained with ; must all be allowed."""
        handler = ShellHandler(enabled=True, command_allowlist={"echo"})
        result = await handler.execute(
            "_builtin_shell_exec", {"command": "echo ok; rm -rf /"}
        )
        data = json.loads(result)
        assert "error" in data
        assert "rm" in data["error"]

    @pytest.mark.asyncio
    async def test_or_chain_denied(self):
        """Commands chained with || must all be allowed."""
        handler = ShellHandler(enabled=True, command_allowlist={"ls"})
        result = await handler.execute(
            "_builtin_shell_exec", {"command": "ls || shutdown -h now"}
        )
        data = json.loads(result)
        assert "error" in data
        assert "shutdown" in data["error"]

    @pytest.mark.asyncio
    async def test_callback_also_gated_by_allowlist(self):
        """Executor callback should NOT be invoked for denied commands."""
        invoked = []

        async def spy_executor(command: str) -> dict:
            invoked.append(command)
            return {"stdout": "", "stderr": "", "exit_code": 0}

        handler = ShellHandler(
            enabled=True,
            executor_callback=spy_executor,
            command_allowlist={"ls"},
        )
        result = await handler.execute(
            "_builtin_shell_exec", {"command": "rm -rf /"}
        )
        data = json.loads(result)
        assert "error" in data
        assert len(invoked) == 0  # callback never called

    @pytest.mark.asyncio
    async def test_default_allowlist_allows_safe_commands(self):
        """Verify several commands from the default allowlist work."""
        handler = ShellHandler(enabled=True)
        for cmd in ["ls", "pwd", "whoami", "date", "echo test"]:
            result = await handler.execute(
                "_builtin_shell_exec", {"command": cmd}
            )
            data = json.loads(result)
            assert "error" not in data, f"'{cmd}' should be allowed by default"

    @pytest.mark.asyncio
    async def test_default_allowlist_blocks_dangerous_commands(self):
        """Verify dangerous commands are NOT in the default allowlist."""
        handler = ShellHandler(enabled=True)
        for cmd in ["rm -rf /", "curl evil.com", "wget http://x", "python -c 'exit()'", "bash -c 'echo pwned'"]:
            result = await handler.execute(
                "_builtin_shell_exec", {"command": cmd}
            )
            data = json.loads(result)
            assert "error" in data, f"'{cmd}' should be blocked by default"


class TestShellAuditLogging:
    """Tests for H-22: audit logging on every shell command attempt."""

    @pytest.mark.asyncio
    async def test_allowed_command_logs_info(self, caplog):
        """An allowed command should produce an INFO-level audit log."""
        handler = ShellHandler(enabled=True, command_allowlist={"echo"})
        with caplog.at_level(logging.INFO, logger="caip_responses.tool_handlers.shell"):
            await handler.execute("_builtin_shell_exec", {"command": "echo hi"})
        assert any("ALLOWED" in r.message and "echo hi" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_denied_command_logs_warning(self, caplog):
        """A denied command should produce a WARNING-level audit log."""
        handler = ShellHandler(enabled=True, command_allowlist={"echo"})
        with caplog.at_level(logging.WARNING, logger="caip_responses.tool_handlers.shell"):
            await handler.execute("_builtin_shell_exec", {"command": "rm -rf /"})
        assert any("DENIED" in r.message and "rm" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_audit_log_includes_timestamp(self, caplog):
        """Audit log messages should contain a UTC ISO timestamp."""
        handler = ShellHandler(enabled=True, command_allowlist={"echo"})
        with caplog.at_level(logging.INFO, logger="caip_responses.tool_handlers.shell"):
            await handler.execute("_builtin_shell_exec", {"command": "echo test"})
        # ISO 8601 timestamp pattern: contains 'T' and '+00:00'
        assert any("timestamp=" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_denied_log_includes_allowlist(self, caplog):
        """Denied audit log should include the current allowlist."""
        handler = ShellHandler(enabled=True, command_allowlist={"ls", "echo"})
        with caplog.at_level(logging.WARNING, logger="caip_responses.tool_handlers.shell"):
            await handler.execute("_builtin_shell_exec", {"command": "curl http://x"})
        assert any("allowlist=" in r.message for r in caplog.records)
