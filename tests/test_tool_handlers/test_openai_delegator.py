"""Tests for the OpenAI delegator mixin and shared metrics."""

from __future__ import annotations

from caip_responses.tool_handlers.openai_delegator import (
    DelegatedToolMetrics,
    DelegatedToolUsage,
)


class TestDelegatedToolUsage:
    def test_defaults(self):
        usage = DelegatedToolUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.provider == "openai"
        assert usage.tool_type == ""

    def test_with_values(self):
        usage = DelegatedToolUsage(
            input_tokens=100, output_tokens=200, total_tokens=300,
            model="gpt-4.1-nano", tool_type="mcp",
        )
        assert usage.input_tokens == 100
        assert usage.model == "gpt-4.1-nano"
        assert usage.tool_type == "mcp"


class TestDelegatedToolMetrics:
    def test_empty(self):
        m = DelegatedToolMetrics()
        assert m.total_tokens == 0
        assert m.total_calls == 0

    def test_record(self):
        m = DelegatedToolMetrics()
        m.record(DelegatedToolUsage(
            input_tokens=50, output_tokens=100, total_tokens=150,
            tool_type="mcp",
        ))
        assert m.total_input_tokens == 50
        assert m.total_output_tokens == 100
        assert m.total_tokens == 150
        assert m.total_calls == 1

    def test_cumulative(self):
        m = DelegatedToolMetrics()
        for _ in range(3):
            m.record(DelegatedToolUsage(
                input_tokens=10, output_tokens=20, total_tokens=30,
                tool_type="file_search",
            ))
        assert m.total_tokens == 90
        assert m.total_calls == 3

    def test_to_dict(self):
        m = DelegatedToolMetrics()
        m.record(DelegatedToolUsage(
            input_tokens=10, output_tokens=20, total_tokens=30,
            model="gpt-4.1-nano", tool_type="mcp",
        ))
        d = m.to_dict()
        assert d["total_calls"] == 1
        assert len(d["calls"]) == 1
        assert d["calls"][0]["tool_type"] == "mcp"
        assert d["calls"][0]["model"] == "gpt-4.1-nano"
