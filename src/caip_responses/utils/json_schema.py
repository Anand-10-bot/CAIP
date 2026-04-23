from __future__ import annotations

import json
from typing import Any


def validate_json_against_schema(text: str, schema: dict[str, Any]) -> bool:
    """Best-effort validation of a JSON string against a schema.

    Returns True if the text is valid JSON. Full JSON Schema validation
    requires an external library — this is a lightweight check.
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def schema_to_instruction(schema: dict[str, Any]) -> str:
    """Convert a JSON schema into a natural-language instruction string.

    Used by providers that don't support native structured outputs
    (e.g., Anthropic) — we inject the schema requirement into the
    system prompt instead.
    """
    schema_str = json.dumps(schema, indent=2)
    return (
        "You MUST respond with valid JSON that conforms to this schema:\n"
        f"```json\n{schema_str}\n```\n"
        "Do not include any text outside the JSON object."
    )
