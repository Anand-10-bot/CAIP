from __future__ import annotations

import pytest


@pytest.fixture
def sample_text_response_data() -> dict:
    """Sample response data matching a simple text completion."""
    return {
        "id": "resp_test_001",
        "object": "response",
        "created_at": 1700000000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "item_001",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello! How can I help you?",
                        "annotations": [],
                    }
                ],
                "status": "completed",
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def sample_function_call_response_data() -> dict:
    """Sample response data with a function call."""
    return {
        "id": "resp_test_002",
        "object": "response",
        "created_at": 1700000000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "item_002",
                "call_id": "fc_001",
                "name": "get_weather",
                "arguments": '{"city": "San Francisco"}',
            }
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 15,
            "total_tokens": 35,
        },
    }


@pytest.fixture
def sample_tools() -> list[dict]:
    """Sample tool definitions."""
    return [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
        {
            "type": "web_search",
            "search_context_size": "medium",
        },
        {
            "type": "mcp",
            "server_label": "test-mcp",
            "server_url": "http://localhost:8080",
            "require_approval": "never",
        },
    ]
