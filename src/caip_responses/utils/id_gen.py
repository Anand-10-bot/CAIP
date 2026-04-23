from __future__ import annotations

import secrets
import time


def generate_id(prefix: str = "resp") -> str:
    """Generate a unique ID with the given prefix (e.g. resp_abc123)."""
    return f"{prefix}_{secrets.token_hex(12)}"


def generate_response_id() -> str:
    return generate_id("resp")


def generate_item_id() -> str:
    return generate_id("item")


def generate_call_id() -> str:
    return generate_id("fc")


def unix_timestamp() -> int:
    return int(time.time())
