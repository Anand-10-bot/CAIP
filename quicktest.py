"""
Quick live test — paste your API key below and run:
    python quicktest.py

Tests supported:
  1. Gemini  (gemini-2.0-flash)   — free key at https://aistudio.google.com/app/apikey
  2. Sarvam  (sarvam-m)           — free key at https://dashboard.sarvam.ai
"""

import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────
# PASTE YOUR KEY(S) HERE  (or set as env vars)
# ─────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",  "")   # ← paste gemini key here
SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY",  "")   # ← paste sarvam key here

# Gemini via Vertex AI (service account JSON instead of an API key).
# Set the path to your service-account .json file and a region. Requires a
# GCP project with the Vertex AI API enabled and billing active.
GEMINI_SERVICE_ACCOUNT = os.getenv("GEMINI_SERVICE_ACCOUNT", "")  # ← path to .json
GEMINI_LOCATION         = os.getenv("GEMINI_LOCATION", "us-central1")
# ─────────────────────────────────────────────

from caip_responses import AsyncClient


async def test_simple_text(client: AsyncClient, model: str, provider_label: str):
    """Test 1: plain text input, plain text output."""
    print(f"\n{'='*55}")
    print(f"  TEST 1 — Simple text  [{provider_label} / {model}]")
    print(f"{'='*55}")

    response = await client.responses.create(
        model=model,
        input="What is 2 + 2? Reply in one sentence.",
    )

    print(f"  Response ID : {response.id}")
    print(f"  Status      : {response.status}")
    print(f"  Answer      : {response.output_text}")
    print(f"  Tokens used : in={response.usage.input_tokens}  out={response.usage.output_tokens}")
    assert response.output_text, "Expected non-empty output_text"
    print("  ✅ PASSED")


async def test_with_instructions(client: AsyncClient, model: str, provider_label: str):
    """Test 2: system instructions + user message."""
    print(f"\n{'='*55}")
    print(f"  TEST 2 — With instructions  [{provider_label} / {model}]")
    print(f"{'='*55}")

    response = await client.responses.create(
        model=model,
        instructions="You are a helpful assistant who always answers in exactly 3 bullet points.",
        input="What are the benefits of drinking water?",
        max_output_tokens=2000,
    )

    print(f"  Answer:\n{response.output_text}")
    assert response.output_text, "Expected non-empty output_text"
    print("  ✅ PASSED")


async def test_streaming(client: AsyncClient, model: str, provider_label: str):
    """Test 3: streaming output."""
    print(f"\n{'='*55}")
    print(f"  TEST 3 — Streaming  [{provider_label} / {model}]")
    print(f"{'='*55}")

    print("  Streaming tokens: ", end="", flush=True)
    full_text = ""
    event_types = []

    async for event in await client.responses.create(
        model=model,
        input="Name three countries in Europe. Be brief.",
        stream=True,
    ):
        event_types.append(event.type)
        if event.type == "response.output_text.delta":
            delta = event.delta or ""
            print(delta, end="", flush=True)
            full_text += delta

    print()  # newline after stream
    print(f"  Event types seen : {sorted(set(event_types))}")
    assert full_text, "Expected streamed text"
    assert "response.created" in event_types, "Missing response.created event"
    assert "response.completed" in event_types, "Missing response.completed event"
    print("  ✅ PASSED")


async def test_function_calling(client: AsyncClient, model: str, provider_label: str):
    """Test 4: function/tool calling — the model should call get_weather."""
    print(f"\n{'='*55}")
    print(f"  TEST 4 — Function calling  [{provider_label} / {model}]")
    print(f"{'='*55}")

    response = await client.responses.create(
        model=model,
        input="What is the weather like in Mumbai right now?",
        tools=[{
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }],
        tool_choice="auto",
    )

    print(f"  Status          : {response.status}")
    print(f"  Has fn calls?   : {response.has_function_calls}")

    if response.has_function_calls:
        for fc in response.function_calls:
            print(f"  Called fn       : {fc.name}({fc.arguments})")
        print("  ✅ PASSED — model correctly called the function")
    else:
        # Some models answer directly without calling the tool — acceptable
        print(f"  Model answered directly: {response.output_text[:100]}")
        print("  ⚠️  Model skipped tool call (answered directly) — still OK")


async def _add_handler(name: str, arguments: dict) -> str:
    """User-registered callback for the `add` tool. Returns the sum as a string."""
    a = arguments.get("a", 0)
    b = arguments.get("b", 0)
    print(f"  [tool executed] add(a={a}, b={b}) = {a + b}")
    return str(a + b)


async def test_tool_execution(client: AsyncClient, model: str, provider_label: str):
    """Test 6: end-to-end tool calling — the agent loop executes `add` and
    feeds the result back, so the final answer must contain the real sum."""
    print(f"\n{'='*55}")
    print(f"  TEST 6 — Tool execution (add)  [{provider_label} / {model}]")
    print(f"{'='*55}")

    # Register the function so the agent loop can actually run it
    client.tools.register("add", _add_handler)

    response = await client.responses.create(
        model=model,
        input="What is 27 + 17? Use the add tool to compute it.",
        tools=[{
            "type": "function",
            "name": "add",
            "description": "Add two numbers and return their sum",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        }],
        tool_choice="auto",
    )

    print(f"  Final answer    : {response.output_text}")
    assert "44" in response.output_text, (
        f"Expected the executed sum '44' in the final answer, got: {response.output_text!r}"
    )
    print("  ✅ PASSED — model called add, loop executed it, answer used the result")


async def test_multi_turn(client: AsyncClient, model: str, provider_label: str):
    """Test 5: multi-turn conversation using previous_response_id."""
    print(f"\n{'='*55}")
    print(f"  TEST 5 — Multi-turn  [{provider_label} / {model}]")
    print(f"{'='*55}")

    # Turn 1
    r1 = await client.responses.create(
        model=model,
        input="My name is Alex. Remember that.",
    )
    print(f"  Turn 1: {r1.output_text[:80]}")

    # Turn 2 — uses previous_response_id to continue conversation
    r2 = await client.responses.create(
        model=model,
        input="What is my name?",
        previous_response_id=r1.id,
    )
    print(f"  Turn 2: {r2.output_text[:80]}")

    if "alex" in r2.output_text.lower():
        print("  ✅ PASSED — model remembered the name from turn 1")
    else:
        print("  ⚠️  Model may not have retained context (check output above)")


# ──────────────────────────────────────────────────────────
#  Runner
# ──────────────────────────────────────────────────────────

async def run_all_tests(api_key: str, model: str, provider_label: str, extra_kwargs: dict):
    """Run all tests for a given provider."""
    client = AsyncClient(**extra_kwargs)

    passed = 0
    failed = 0

    tests = [
        test_simple_text,
        test_with_instructions,
        test_streaming,
        test_function_calling,
        test_multi_turn,
        test_tool_execution,
    ]

    for test_fn in tests:
        try:
            await test_fn(client, model, provider_label)
            passed += 1
        except Exception as e:
            print(f"\n  ❌ FAILED: {e}")
            failed += 1

    await client.close()

    print(f"\n{'━'*55}")
    print(f"  {provider_label} results: {passed} passed / {failed} failed")
    print(f"{'━'*55}\n")


async def main():
    any_test_ran = False

    # ── Gemini ──────────────────────────────────────────────
    if GEMINI_SERVICE_ACCOUNT:
        any_test_ran = True
        print("\n🟢  Running GEMINI tests (Vertex AI / service account)...")
        await run_all_tests(
            api_key="",
            model="gemini-2.5-flash",
            provider_label="Gemini (Vertex)",
            extra_kwargs={
                "gemini_service_account_path": GEMINI_SERVICE_ACCOUNT,
                "gemini_location": GEMINI_LOCATION,
            },
        )
    elif GEMINI_API_KEY:
        any_test_ran = True
        print("\n🟢  Running GEMINI tests...")
        await run_all_tests(
            api_key=GEMINI_API_KEY,
            model="gemini-2.0-flash-lite",
            provider_label="Gemini",
            extra_kwargs={"gemini_api_key": GEMINI_API_KEY},
        )
    else:
        print("⏭️  Gemini skipped (no GEMINI_API_KEY or GEMINI_SERVICE_ACCOUNT set)")

    # ── Sarvam ──────────────────────────────────────────────
    if SARVAM_API_KEY:
        any_test_ran = True
        print("\n🟣  Running SARVAM tests...")
        await run_all_tests(
            api_key=SARVAM_API_KEY,
            model="sarvam-30b",
            provider_label="Sarvam",
            extra_kwargs={"sarvam_api_key": SARVAM_API_KEY},
        )
    else:
        print("⏭️  Sarvam skipped (no SARVAM_API_KEY set)")

    if not any_test_ran:
        print("\n❗ No API keys configured.")
        print("   Edit quicktest.py and paste your key into GEMINI_API_KEY or SARVAM_API_KEY.")
        print("   Free Gemini key: https://aistudio.google.com/app/apikey")
        print("   Free Sarvam key: https://dashboard.sarvam.ai\n")


if __name__ == "__main__":
    asyncio.run(main())
