from __future__ import annotations

from caip_responses.models.items import (
    ComputerCallItem,
    ComputerCallOutputItem,
    CustomToolCallItem,
    FileSearchCallItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    ImageGenerationCallItem,
    InputFileContent,
    InputMessage,
    MCPApprovalRequestItem,
    MCPApprovalResponseItem,
    MCPCallItem,
    MCPListToolsItem,
    MessageOutputItem,
    OutputTextContent,
    ReasoningItem,
    ShellCallItem,
    ShellCallOutputItem,
    ToolSearchCallItem,
    ToolSearchOutputItem,
    WebSearchCallItem,
)


class TestInputMessage:
    def test_user_message_string_content(self):
        msg = InputMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        msg = InputMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"

    def test_developer_message(self):
        msg = InputMessage(role="developer", content="Be helpful")
        assert msg.role == "developer"

    def test_system_message(self):
        msg = InputMessage(role="system", content="You are helpful")
        assert msg.role == "system"


class TestOutputTextContent:
    def test_basic(self):
        content = OutputTextContent(text="Hello")
        assert content.type == "output_text"
        assert content.text == "Hello"
        assert content.annotations == []


class TestMessageOutputItem:
    def test_basic(self):
        item = MessageOutputItem(
            id="item_1",
            content=[OutputTextContent(text="Hello")],
        )
        assert item.type == "message"
        assert item.role == "assistant"
        assert item.status == "completed"
        assert len(item.content) == 1


class TestFunctionCallItem:
    def test_basic(self):
        item = FunctionCallItem(
            id="item_1",
            call_id="fc_1",
            name="get_weather",
            arguments='{"city": "SF"}',
        )
        assert item.type == "function_call"
        assert item.name == "get_weather"
        assert item.call_id == "fc_1"
        assert item.arguments == '{"city": "SF"}'


class TestFunctionCallOutputItem:
    def test_basic(self):
        item = FunctionCallOutputItem(
            call_id="fc_1",
            output='{"temp": 72}',
        )
        assert item.type == "function_call_output"
        assert item.call_id == "fc_1"
        assert item.output == '{"temp": 72}'


class TestReasoningItem:
    def test_basic(self):
        item = ReasoningItem(id="item_1")
        assert item.type == "reasoning"
        assert item.content == []
        assert item.summary == []

    def test_with_summary(self):
        item = ReasoningItem(
            id="item_1",
            summary=[{"type": "text", "text": "I thought about it..."}],
        )
        assert len(item.summary) == 1

    def test_with_content_and_summary(self):
        """Matches OpenAI doc format: reasoning with content=[] and summary=[]."""
        item = ReasoningItem(
            id="rs_6890e972fa7c819ca8bc561526b989170694874912ae0ea6",
            content=[],
            summary=[],
        )
        assert item.type == "reasoning"
        assert item.content == []
        assert item.summary == []

    def test_with_encrypted_content(self):
        item = ReasoningItem(
            id="item_1",
            encrypted_content="encrypted_data_here",
        )
        assert item.encrypted_content == "encrypted_data_here"


class TestWebSearchCallItem:
    def test_basic(self):
        item = WebSearchCallItem(id="item_1")
        assert item.type == "web_search_call"
        assert item.status == "completed"
        assert item.action is None

    def test_search_action(self):
        """Matches exact format from web search docs."""
        item = WebSearchCallItem(
            id="ws_67c9fa0502748190b7dd390736892e100be649c1a5ff9609",
            status="completed",
            action={"type": "search", "queries": ["latest news March 2025"]},
        )
        assert item.action["type"] == "search"
        assert "latest news" in item.action["queries"][0]

    def test_open_page_action(self):
        """Reasoning models can open pages."""
        item = WebSearchCallItem(
            id="ws_abc",
            action={"type": "open_page", "url": "https://example.com/article"},
        )
        assert item.action["type"] == "open_page"
        assert item.action["url"] == "https://example.com/article"

    def test_find_in_page_action(self):
        """Reasoning models can search within a page."""
        item = WebSearchCallItem(
            id="ws_abc",
            action={"type": "find_in_page", "query": "climate data"},
        )
        assert item.action["type"] == "find_in_page"


class TestWebSearchOutputPattern:
    """Validates the full output pattern from web search docs:
    web_search_call + message with url_citation annotations.
    """

    def test_full_web_search_response_output(self):
        from caip_responses.models.response import Response

        response = Response(
            id="resp_ws_test",
            model="gpt-4.1",
            output=[
                {
                    "type": "web_search_call",
                    "id": "ws_67c9fa0502748190b7dd390736892e100be649c1a5ff9609",
                    "status": "completed",
                },
                {
                    "type": "message",
                    "id": "msg_67c9fa077e288190af08fdffda2e34f20be649c1a5ff9609",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "On March 6, 2025, several news...",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "start_index": 2606,
                                    "end_index": 2758,
                                    "url": "https://example.com/article",
                                    "title": "News Article Title",
                                },
                            ],
                        }
                    ],
                    "status": "completed",
                },
            ],
        )
        assert response.output_text == "On March 6, 2025, several news..."
        assert len(response.output) == 2
        assert response.output[0]["type"] == "web_search_call"
        msg = response.output[1]
        assert msg["content"][0]["annotations"][0]["type"] == "url_citation"
        assert msg["content"][0]["annotations"][0]["url"] == "https://example.com/article"


class TestInputFileContent:
    def test_basic(self):
        content = InputFileContent(file_id="file-abc123")
        assert content.type == "input_file"
        assert content.file_id == "file-abc123"

    def test_with_filename(self):
        content = InputFileContent(filename="report.pdf", file_data="base64data")
        assert content.filename == "report.pdf"
        assert content.file_data == "base64data"


class TestInputMessageWithFileContent:
    def test_message_with_file_content(self):
        msg = InputMessage(
            role="user",
            content=[
                {"type": "input_text", "text": "Summarize this document"},
                {"type": "input_file", "file_id": "file-abc123"},
            ],
        )
        assert msg.role == "user"
        assert len(msg.content) == 2


class TestShellCallItem:
    def test_basic(self):
        item = ShellCallItem(id="item_1")
        assert item.type == "shell_call"
        assert item.status == "completed"
        assert item.call_id == ""
        assert item.action == {}
        assert item.max_output_length is None

    def test_with_action(self):
        item = ShellCallItem(
            id="item_1",
            call_id="call_123",
            action={"type": "exec", "command": ["ls", "-la"]},
        )
        assert item.call_id == "call_123"
        assert item.action["type"] == "exec"

    def test_with_max_output_length(self):
        """max_output_length should be forwarded in shell_call_output."""
        item = ShellCallItem(
            id="sc_1",
            call_id="call_max",
            action={"type": "exec", "command": ["cat", "big_file.txt"]},
            max_output_length=4096,
        )
        assert item.max_output_length == 4096

    def test_extra_fields_allowed(self):
        item = ShellCallItem(id="sc_1", working_directory="/mnt/data")
        assert item.working_directory == "/mnt/data"


class TestShellCallOutputItem:
    def test_basic(self):
        item = ShellCallOutputItem()
        assert item.type == "shell_call_output"
        assert item.output == ""
        assert item.error == ""
        assert item.status == "completed"

    def test_with_stdout(self):
        """Successful command execution with stdout."""
        item = ShellCallOutputItem(
            id="sco_1",
            call_id="call_123",
            output="total 42\ndrwxr-xr-x 2 root root 4096 ...",
            status="completed",
        )
        assert item.call_id == "call_123"
        assert "total 42" in item.output

    def test_with_stderr(self):
        """Failed command with stderr."""
        item = ShellCallOutputItem(
            id="sco_2",
            call_id="call_456",
            output="",
            error="bash: command not found: foobar",
            status="completed",
        )
        assert "command not found" in item.error

    def test_timeout_outcome(self):
        """Docs: if command exceeds timeout, return timeout outcome with partial output."""
        item = ShellCallOutputItem(
            id="sco_3",
            call_id="call_timeout",
            output="partial output before timeout...",
            error="Command timed out after 30s",
            status="incomplete",
        )
        assert item.status == "incomplete"
        assert "partial output" in item.output

    def test_extra_fields_allowed(self):
        item = ShellCallOutputItem(
            id="sco_4",
            call_id="call_1",
            output="ok",
            exit_code=0,
        )
        assert item.exit_code == 0


class TestShellResponsePattern:
    """Full shell patterns: hosted and local mode."""

    def test_hosted_shell_call_and_output(self):
        """Hosted mode: shell_call + shell_call_output in same response."""
        from caip_responses.models.response import Response

        response = Response(
            id="resp_shell_hosted",
            model="gpt-5.4",
            output=[
                {
                    "type": "shell_call",
                    "id": "sc_hosted_1",
                    "call_id": "call_h1",
                    "status": "completed",
                    "action": {
                        "type": "exec",
                        "command": ["python", "--version"],
                    },
                },
                {
                    "type": "shell_call_output",
                    "id": "sco_hosted_1",
                    "call_id": "call_h1",
                    "output": "Python 3.11.0",
                    "status": "completed",
                },
                {
                    "type": "message",
                    "id": "msg_shell",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The container is running Python 3.11.0.",
                            "annotations": [],
                        },
                    ],
                    "status": "completed",
                },
            ],
        )
        assert len(response.output) == 3
        assert response.output[0]["type"] == "shell_call"
        assert response.output[1]["type"] == "shell_call_output"
        assert response.output[1]["output"] == "Python 3.11.0"
        assert response.output_text == "The container is running Python 3.11.0."

    def test_local_shell_call_output_as_input(self):
        """Local mode: send shell_call_output back as input."""
        from caip_responses.models.request import CreateResponseRequest

        req = CreateResponseRequest(
            model="gpt-5.4",
            input=[
                {
                    "type": "shell_call_output",
                    "call_id": "call_local_1",
                    "output": "README.md\nsetup.py\nsrc/\n",
                },
            ],
            tools=[{"type": "shell", "environment": {"type": "local"}}],
            previous_response_id="resp_shell_local_1",
        )
        assert req.input[0]["type"] == "shell_call_output"
        assert req.input[0]["call_id"] == "call_local_1"

    def test_multi_turn_container_reference(self):
        """Multi-turn with container_reference and previous_response_id."""
        from caip_responses.models.request import CreateResponseRequest

        req = CreateResponseRequest(
            model="gpt-5.4",
            previous_response_id="resp_2a8e5c9174d63b0f18a4c572de9f64a1b3c76d508e12f9ab47",
            input="Read /mnt/data/top5.csv and report the top candidate.",
            tools=[
                {
                    "type": "shell",
                    "environment": {
                        "type": "container_reference",
                        "container_id": "cntr_f19c2b51e4a06793d82d54a7be0fc9154d3361ab28ce7f6041",
                    },
                },
            ],
        )
        assert req.previous_response_id.startswith("resp_")
        assert req.tools[0]["environment"]["container_id"].startswith("cntr_")


class TestComputerCallItem:
    def test_basic(self):
        item = ComputerCallItem(id="cc_abc123")
        assert item.type == "computer_call"
        assert item.call_id == ""
        assert item.actions == []
        assert item.status == "completed"
        assert item.pending_safety_checks == []

    def test_screenshot_action(self):
        """First turn often returns a screenshot request."""
        item = ComputerCallItem(
            id="cc_1",
            call_id="call_ss_1",
            actions=[{"type": "screenshot"}],
        )
        assert len(item.actions) == 1
        assert item.actions[0]["type"] == "screenshot"

    def test_batched_actions(self):
        """GA integration batches multiple actions in one call."""
        item = ComputerCallItem(
            id="cc_2",
            call_id="call_batch_1",
            actions=[
                {"type": "click", "x": 100, "y": 200, "button": "left"},
                {"type": "type", "text": "hello world"},
                {"type": "keypress", "keys": ["Enter"]},
            ],
        )
        assert len(item.actions) == 3
        assert item.actions[0]["type"] == "click"
        assert item.actions[0]["x"] == 100
        assert item.actions[1]["type"] == "type"
        assert item.actions[1]["text"] == "hello world"
        assert item.actions[2]["type"] == "keypress"

    def test_click_with_modifiers(self):
        """Click with modifier keys (Ctrl+click to open in new tab)."""
        item = ComputerCallItem(
            id="cc_3",
            call_id="call_mod_1",
            actions=[
                {"type": "click", "x": 300, "y": 400, "button": "left", "keys": ["CTRL"]},
            ],
        )
        assert item.actions[0]["keys"] == ["CTRL"]

    def test_drag_action(self):
        item = ComputerCallItem(
            id="cc_4",
            call_id="call_drag_1",
            actions=[
                {
                    "type": "drag",
                    "startX": 100,
                    "startY": 200,
                    "endX": 300,
                    "endY": 400,
                },
            ],
        )
        assert item.actions[0]["type"] == "drag"
        assert item.actions[0]["endX"] == 300

    def test_scroll_action(self):
        item = ComputerCallItem(
            id="cc_5",
            call_id="call_scroll_1",
            actions=[
                {"type": "scroll", "x": 500, "y": 300, "direction": "down", "amount": 3},
            ],
        )
        assert item.actions[0]["direction"] == "down"

    def test_with_pending_safety_checks(self):
        """Safety checks the application should surface to the user."""
        item = ComputerCallItem(
            id="cc_6",
            call_id="call_safe_1",
            actions=[{"type": "click", "x": 100, "y": 200}],
            pending_safety_checks=[
                {"id": "sc_1", "code": "sensitive_action", "message": "About to submit a form"},
            ],
        )
        assert len(item.pending_safety_checks) == 1
        assert item.pending_safety_checks[0]["code"] == "sensitive_action"

    def test_extra_fields_allowed(self):
        item = ComputerCallItem(id="cc_7", call_id="call_1", current_url="https://example.com")
        assert item.current_url == "https://example.com"


class TestComputerCallOutputItem:
    def test_basic(self):
        item = ComputerCallOutputItem(
            call_id="call_ss_1",
            output={
                "type": "input_image",
                "image_url": "data:image/png;base64,iVBORw0KGgo...",
            },
        )
        assert item.type == "computer_call_output"
        assert item.call_id == "call_ss_1"
        assert item.output["type"] == "input_image"
        assert item.acknowledged_safety_checks == []

    def test_with_detail_original(self):
        """Docs recommend detail='original' for best click accuracy."""
        item = ComputerCallOutputItem(
            call_id="call_1",
            output={
                "type": "input_image",
                "image_url": "data:image/png;base64,abc123...",
                "detail": "original",
            },
        )
        assert item.output["detail"] == "original"

    def test_with_acknowledged_safety_checks(self):
        """Echo back safety checks that the user acknowledged."""
        item = ComputerCallOutputItem(
            call_id="call_safe_1",
            output={
                "type": "input_image",
                "image_url": "data:image/png;base64,abc123...",
            },
            acknowledged_safety_checks=[
                {"id": "sc_1", "code": "sensitive_action", "message": "About to submit a form"},
            ],
        )
        assert len(item.acknowledged_safety_checks) == 1
        assert item.acknowledged_safety_checks[0]["id"] == "sc_1"


class TestComputerUseLoopPattern:
    """Full computer use loop: request → computer_call → computer_call_output → repeat."""

    def test_computer_call_response(self):
        """Model returns computer_call with batched actions."""
        from caip_responses.models.response import Response

        response = Response(
            id="resp_cua_1",
            model="gpt-5.4",
            output=[
                {
                    "type": "computer_call",
                    "id": "cc_abc",
                    "call_id": "call_cua_1",
                    "actions": [
                        {"type": "screenshot"},
                    ],
                    "status": "completed",
                },
            ],
        )
        assert len(response.output) == 1
        assert response.output[0]["type"] == "computer_call"
        assert response.output[0]["actions"][0]["type"] == "screenshot"

    def test_computer_call_output_as_input(self):
        """Send screenshot back as computer_call_output in next request."""
        from caip_responses.models.request import CreateResponseRequest

        req = CreateResponseRequest(
            model="gpt-5.4",
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "call_cua_1",
                    "output": {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,iVBORw0KGgo...",
                    },
                },
            ],
            tools=[{"type": "computer", "display_width": 1024, "display_height": 768}],
            previous_response_id="resp_cua_1",
        )
        assert req.input[0]["type"] == "computer_call_output"
        assert req.previous_response_id == "resp_cua_1"

    def test_multi_action_then_final_message(self):
        """After actions complete, model returns final text message."""
        from caip_responses.models.response import Response

        response = Response(
            id="resp_cua_final",
            model="gpt-5.4",
            output=[
                {
                    "type": "message",
                    "id": "msg_cua_final",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I've completed the form submission. The confirmation number is #12345.",
                            "annotations": [],
                        },
                    ],
                    "status": "completed",
                },
            ],
        )
        assert response.output_text == "I've completed the form submission. The confirmation number is #12345."

    def test_migration_preview_to_ga(self):
        """Preview used single action; GA uses batched actions[]."""
        from caip_responses.models.request import CreateResponseRequest

        # GA request
        req = CreateResponseRequest(
            model="gpt-5.4",
            input="Go to example.com and click the login button",
            tools=[{"type": "computer", "display_width": 1024, "display_height": 768}],
        )
        assert req.tools[0]["type"] == "computer"

        # Preview request (deprecated)
        req_preview = CreateResponseRequest(
            model="computer-use-preview",
            input="Go to example.com and click the login button",
            tools=[{"type": "computer_use", "display_width": 1024, "display_height": 768}],
            truncation="auto",
        )
        assert req_preview.tools[0]["type"] == "computer_use"
        assert req_preview.truncation == "auto"


class TestImageGenerationCallItem:
    def test_basic(self):
        item = ImageGenerationCallItem(id="item_1")
        assert item.type == "image_generation_call"
        assert item.status == "completed"
        assert item.result is None
        assert item.revised_prompt is None

    def test_with_base64_result(self):
        """result contains base64-encoded image data."""
        # Simulate a short base64 string (real ones are much longer)
        b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        item = ImageGenerationCallItem(id="ig_abc123", result=b64_data)
        assert item.result == b64_data

    def test_with_revised_prompt(self):
        """Docs example: revised_prompt shows the auto-revised prompt."""
        item = ImageGenerationCallItem(
            id="ig_123",
            status="completed",
            revised_prompt=(
                "A gray tabby cat hugging an otter. The otter is wearing an "
                "orange scarf. Both animals are cute and friendly, depicted "
                "in a warm, heartwarming style."
            ),
            result="base64encodedimagedata...",
        )
        assert item.revised_prompt.startswith("A gray tabby cat")
        assert "orange scarf" in item.revised_prompt
        assert item.result == "base64encodedimagedata..."

    def test_extra_fields_allowed(self):
        """Forward-compat: extra fields pass through."""
        item = ImageGenerationCallItem(
            id="ig_abc",
            result="data",
            partial_image_index=1,
        )
        assert item.partial_image_index == 1


class TestImageGenerationResponsePattern:
    """Full response pattern from image generation docs."""

    def test_image_generation_response(self):
        from caip_responses.models.response import Response

        response = Response(
            id="resp_img_1",
            model="gpt-5.4",
            output=[
                {
                    "type": "image_generation_call",
                    "id": "ig_abc123",
                    "status": "completed",
                    "revised_prompt": (
                        "A gray tabby cat hugging an otter. The otter is wearing "
                        "an orange scarf. Both animals are cute and friendly."
                    ),
                    "result": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
                },
            ],
        )
        assert len(response.output) == 1
        ig = response.output[0]
        assert ig["type"] == "image_generation_call"
        assert ig["revised_prompt"].startswith("A gray tabby cat")
        assert ig["result"].startswith("iVBORw0KGgo")

    def test_multi_turn_image_editing_via_previous_response_id(self):
        """Multi-turn: second call uses previous_response_id for editing."""
        from caip_responses.models.request import CreateResponseRequest

        # First turn
        req1 = CreateResponseRequest(
            model="gpt-5.4",
            input="Generate an image of gray tabby cat hugging an otter",
            tools=[{"type": "image_generation"}],
        )
        assert req1.tools[0]["type"] == "image_generation"

        # Second turn references first response
        req2 = CreateResponseRequest(
            model="gpt-5.4",
            previous_response_id="resp_img_1",
            input="Now make it look realistic",
            tools=[{"type": "image_generation"}],
        )
        assert req2.previous_response_id == "resp_img_1"
        assert req2.tools[0]["type"] == "image_generation"

    def test_multi_turn_image_editing_via_image_id(self):
        """Multi-turn: reference image_generation_call by id in input."""
        from caip_responses.models.request import CreateResponseRequest

        req = CreateResponseRequest(
            model="gpt-5.4",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Now make it look realistic"}],
                },
                {
                    "type": "image_generation_call",
                    "id": "ig_abc123",
                },
            ],
            tools=[{"type": "image_generation"}],
        )
        assert req.input[0]["role"] == "user"
        assert req.input[1]["type"] == "image_generation_call"
        assert req.input[1]["id"] == "ig_abc123"

    def test_force_tool_choice(self):
        """tool_choice can force image generation."""
        from caip_responses.models.request import CreateResponseRequest

        req = CreateResponseRequest(
            model="gpt-5.4",
            input="Draw a cat",
            tools=[{"type": "image_generation"}],
            tool_choice={"type": "image_generation"},
        )
        assert req.tool_choice == {"type": "image_generation"}


class TestFileSearchCallItem:
    def test_basic(self):
        item = FileSearchCallItem(id="fs_abc123")
        assert item.type == "file_search_call"
        assert item.status == "completed"
        assert item.queries == []
        assert item.results is None

    def test_with_queries(self):
        item = FileSearchCallItem(
            id="fs_abc123",
            queries=["annual revenue", "profit margins"],
        )
        assert item.queries == ["annual revenue", "profit margins"]

    def test_with_results(self):
        """Results returned when include=["file_search_call.results"] is set."""
        item = FileSearchCallItem(
            id="fs_68249d22bef4819085d2e1ebd3bc5c5309018e5785e4fb97",
            status="completed",
            queries=["attributes of an ancient brown dragon"],
            results=[
                {
                    "file_id": "file-ybETioLhNjiqUxMh7L1G5",
                    "filename": "5e_monster_manual.pdf",
                    "score": 0.996,
                    "text": "Ancient Brown Dragon\nGargantuan dragon, neutral evil...",
                    "attributes": {},
                },
                {
                    "file_id": "file-ybETioLhNjiqUxMh7L1G5",
                    "filename": "5e_monster_manual.pdf",
                    "score": 0.488,
                    "text": "Dragon descriptions continued...",
                    "attributes": {"category": "monsters"},
                },
            ],
        )
        assert len(item.results) == 2
        assert item.results[0]["filename"] == "5e_monster_manual.pdf"
        assert item.results[0]["score"] == 0.996
        assert item.results[1]["attributes"]["category"] == "monsters"

    def test_extra_fields_allowed(self):
        item = FileSearchCallItem(
            id="fs_abc123",
            ranking_options={"ranker": "default_2024_08_21"},
        )
        assert item.ranking_options == {"ranker": "default_2024_08_21"}


class TestFileSearchResponsePattern:
    """Full response pattern: file_search_call + message with file citations."""

    def test_file_search_then_message(self):
        from caip_responses.models.response import Response

        response = Response(
            id="resp_fs_1",
            model="gpt-4.1",
            output=[
                {
                    "type": "file_search_call",
                    "id": "fs_68249d22bef4819085d2e1ebd3bc5c5309018e5785e4fb97",
                    "status": "completed",
                    "queries": ["attributes of an ancient brown dragon"],
                    "results": [
                        {
                            "file_id": "file-ybETioLhNjiqUxMh7L1G5",
                            "filename": "5e_monster_manual.pdf",
                            "score": 0.996,
                            "text": "Ancient Brown Dragon\nGargantuan dragon, neutral evil...",
                            "attributes": {},
                        },
                    ],
                },
                {
                    "type": "message",
                    "id": "msg_fs_1",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The Ancient Brown Dragon is a gargantuan creature...",
                            "annotations": [
                                {
                                    "type": "file_citation",
                                    "index": 42,
                                    "file_id": "file-ybETioLhNjiqUxMh7L1G5",
                                    "filename": "5e_monster_manual.pdf",
                                },
                            ],
                        },
                    ],
                    "status": "completed",
                },
            ],
        )
        assert response.output_text == "The Ancient Brown Dragon is a gargantuan creature..."
        assert len(response.output) == 2
        assert response.output[0]["type"] == "file_search_call"
        assert response.output[0]["results"][0]["score"] == 0.996
        msg = response.output[1]
        assert msg["content"][0]["annotations"][0]["type"] == "file_citation"
        assert msg["content"][0]["annotations"][0]["filename"] == "5e_monster_manual.pdf"

    def test_file_search_with_metadata_filter_pattern(self):
        """Validates the full flow: filtered file_search tool → response with results."""
        from caip_responses.models.response import Response

        response = Response(
            id="resp_fs_filtered",
            model="gpt-4.1",
            output=[
                {
                    "type": "file_search_call",
                    "id": "fs_filtered_1",
                    "status": "completed",
                    "queries": ["revenue Q4"],
                    "results": [
                        {
                            "file_id": "file-Q4report",
                            "filename": "Q4_2025_report.pdf",
                            "score": 0.89,
                            "text": "Q4 revenue reached $2.1B...",
                            "attributes": {"year": "2025", "quarter": "Q4"},
                        },
                    ],
                },
                {
                    "type": "message",
                    "id": "msg_fs_filtered",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "According to the Q4 report, revenue reached $2.1B.",
                            "annotations": [
                                {
                                    "type": "file_citation",
                                    "index": 0,
                                    "file_id": "file-Q4report",
                                    "filename": "Q4_2025_report.pdf",
                                },
                            ],
                        },
                    ],
                    "status": "completed",
                },
            ],
        )
        assert response.output[0]["results"][0]["attributes"]["quarter"] == "Q4"
        assert response.output[1]["content"][0]["annotations"][0]["file_id"] == "file-Q4report"


class TestCustomToolCallItem:
    def test_basic(self):
        item = CustomToolCallItem(
            id="ctc_abc123",
            call_id="call_xyz",
            name="code_exec",
            input='print("hello world")',
        )
        assert item.type == "custom_tool_call"
        assert item.id == "ctc_abc123"
        assert item.call_id == "call_xyz"
        assert item.name == "code_exec"
        assert item.input == 'print("hello world")'
        assert item.status == "completed"

    def test_from_doc_example(self):
        """Matches the exact output format from OpenAI's custom tool docs."""
        item = CustomToolCallItem(
            id="ctc_6890e975e86c819c9338825b3e1994810694874912ae0ea6",
            call_id="call_aGiFQkRWSWAIsMQ19fKqxUgb",
            name="code_exec",
            input='print("hello world")',
            status="completed",
        )
        assert item.type == "custom_tool_call"
        assert item.name == "code_exec"
        assert item.input == 'print("hello world")'


class TestToolSearchCallItem:
    def test_basic(self):
        item = ToolSearchCallItem(id="tsc_abc123")
        assert item.type == "tool_search_call"
        assert item.status == "completed"
        assert item.call_id is None
        assert item.execution is None

    def test_hosted_mode(self):
        """Hosted tool search: execution=server, call_id=null."""
        item = ToolSearchCallItem(
            id="tsc_abc123",
            execution="server",
            call_id=None,
        )
        assert item.execution == "server"
        assert item.call_id is None

    def test_client_mode(self):
        """Client-executed tool search: execution=client, call_id is set."""
        item = ToolSearchCallItem(
            id="tsc_abc123",
            execution="client",
            call_id="call_xyz",
        )
        assert item.execution == "client"
        assert item.call_id == "call_xyz"

    def test_extra_fields_allowed(self):
        item = ToolSearchCallItem(id="tsc_abc123", queries=["math tools"])
        assert item.queries == ["math tools"]


class TestToolSearchOutputItem:
    def test_basic(self):
        item = ToolSearchOutputItem(id="tso_abc123")
        assert item.type == "tool_search_output"
        assert item.tools == []
        assert item.output == []

    def test_with_tools(self):
        """tools field contains the loaded subset that becomes callable."""
        item = ToolSearchOutputItem(
            id="tso_abc123",
            tools=[
                {
                    "type": "function",
                    "name": "add",
                    "description": "Add two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    },
                },
            ],
        )
        assert len(item.tools) == 1
        assert item.tools[0]["name"] == "add"

    def test_with_output(self):
        item = ToolSearchOutputItem(
            id="tso_abc123",
            output=[
                {"type": "function", "name": "add", "description": "Add two numbers"},
                {"type": "function", "name": "multiply", "description": "Multiply two numbers"},
            ],
        )
        assert len(item.output) == 2
        assert item.output[0]["name"] == "add"

    def test_client_mode_with_call_id(self):
        """Client-executed: call_id echoes the tool_search_call's call_id."""
        item = ToolSearchOutputItem(
            id="tso_abc123",
            call_id="call_xyz",
            tools=[
                {"type": "function", "name": "add", "description": "Add two numbers"},
            ],
        )
        assert item.call_id == "call_xyz"
        assert len(item.tools) == 1


class TestMCPCallItem:
    def test_basic(self):
        item = MCPCallItem(id="mcp_abc")
        assert item.type == "mcp_call"
        assert item.status == "completed"
        assert item.name == ""
        assert item.arguments == ""
        assert item.output is None
        assert item.error is None

    def test_from_doc_example(self):
        """Exact format from MCP docs: dice rolling tool call."""
        item = MCPCallItem(
            id="mcp_68a6102d8948819c9b1490d36d5ffa4a0679e572a900e618",
            name="roll",
            arguments='{"diceRollExpression":"2d4 + 1"}',
            output="4",
            error=None,
            server_label="dmcp",
            approval_request_id=None,
        )
        assert item.name == "roll"
        assert item.arguments == '{"diceRollExpression":"2d4 + 1"}'
        assert item.output == "4"
        assert item.server_label == "dmcp"
        assert item.approval_request_id is None

    def test_connector_call(self):
        """MCP call from a connector (Google Calendar)."""
        item = MCPCallItem(
            id="mcp_68a62ae1c93c81a2b98c29340aa3ed8800e9b63986850588",
            name="search_events",
            arguments='{"time_min":"2025-08-20T00:00:00","time_max":"2025-08-21T00:00:00"}',
            output='{"events": []}',
            server_label="Google_Calendar",
        )
        assert item.name == "search_events"
        assert item.server_label == "Google_Calendar"

    def test_with_error(self):
        item = MCPCallItem(
            id="mcp_err",
            name="bad_tool",
            error="Tool execution failed: timeout",
            server_label="flaky-server",
        )
        assert item.error == "Tool execution failed: timeout"
        assert item.output is None


class TestMCPListToolsItem:
    def test_basic(self):
        item = MCPListToolsItem(id="mcpl_abc", server_label="dmcp")
        assert item.type == "mcp_list_tools"
        assert item.tools == []

    def test_from_doc_example(self):
        """Exact format from MCP docs: tool listing."""
        item = MCPListToolsItem(
            id="mcpl_68a6102a4968819c8177b05584dd627b0679e572a900e618",
            server_label="dmcp",
            tools=[
                {
                    "annotations": None,
                    "description": "Given a string of text describing a dice roll...",
                    "input_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "diceRollExpression": {"type": "string"},
                        },
                        "required": ["diceRollExpression"],
                        "additionalProperties": False,
                    },
                    "name": "roll",
                },
            ],
        )
        assert item.server_label == "dmcp"
        assert len(item.tools) == 1
        assert item.tools[0]["name"] == "roll"
        assert item.tools[0]["input_schema"]["properties"]["diceRollExpression"]["type"] == "string"


class TestMCPApprovalRequestItem:
    def test_basic(self):
        item = MCPApprovalRequestItem(
            id="mcpr_68a619e1d82c8190b50c1ccba7ad18ef0d2d23a86136d339",
            name="roll",
            arguments='{"diceRollExpression":"2d4 + 1"}',
            server_label="dmcp",
        )
        assert item.type == "mcp_approval_request"
        assert item.name == "roll"
        assert item.server_label == "dmcp"


class TestMCPApprovalResponseItem:
    def test_approve(self):
        item = MCPApprovalResponseItem(
            approve=True,
            approval_request_id="mcpr_682d498e3bd4819196a0ce1664f8e77b04ad1e533afccbfa",
        )
        assert item.type == "mcp_approval_response"
        assert item.approve is True
        assert item.approval_request_id == "mcpr_682d498e3bd4819196a0ce1664f8e77b04ad1e533afccbfa"

    def test_deny(self):
        item = MCPApprovalResponseItem(
            approve=False,
            approval_request_id="mcpr_abc123",
        )
        assert item.approve is False


class TestMCPFullResponsePattern:
    """Validates the full MCP output pattern: mcp_list_tools + mcp_call + message."""

    def test_full_mcp_response_output(self):
        from caip_responses.models.response import Response

        response = Response(
            id="resp_mcp_test",
            model="gpt-5",
            output=[
                {
                    "type": "mcp_list_tools",
                    "id": "mcpl_abc",
                    "server_label": "dmcp",
                    "tools": [{"name": "roll", "description": "Roll dice"}],
                },
                {
                    "type": "mcp_call",
                    "id": "mcp_def",
                    "name": "roll",
                    "arguments": '{"diceRollExpression":"2d4 + 1"}',
                    "output": "4",
                    "server_label": "dmcp",
                },
                {
                    "type": "message",
                    "id": "msg_ghi",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "You rolled a 4!", "annotations": []}],
                    "status": "completed",
                },
            ],
        )
        assert len(response.output) == 3
        assert response.output[0]["type"] == "mcp_list_tools"
        assert response.output[1]["type"] == "mcp_call"
        assert response.output[1]["output"] == "4"
        assert response.output_text == "You rolled a 4!"


class TestHostedToolSearchResponsePattern:
    """Validates hosted tool search output: tool_search_call + tool_search_output + function_call."""

    def test_full_hosted_tool_search_flow(self):
        from caip_responses.models.response import Response

        response = Response(
            id="resp_ts_hosted",
            model="gpt-5.4",
            output=[
                {
                    "type": "tool_search_call",
                    "id": "tsc_abc",
                    "execution": "server",
                    "call_id": None,
                    "status": "completed",
                },
                {
                    "type": "tool_search_output",
                    "id": "tso_abc",
                    "call_id": None,
                    "tools": [
                        {
                            "type": "function",
                            "name": "add",
                            "description": "Add two numbers together",
                            "parameters": {
                                "type": "object",
                                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            },
                        },
                    ],
                },
                {
                    "type": "function_call",
                    "id": "fc_abc",
                    "call_id": "call_abc",
                    "name": "add",
                    "arguments": '{"a": 2, "b": 2}',
                },
            ],
        )
        assert len(response.output) == 3
        assert response.output[0]["type"] == "tool_search_call"
        assert response.output[0]["execution"] == "server"
        assert response.output[1]["type"] == "tool_search_output"
        assert response.output[1]["tools"][0]["name"] == "add"
        assert response.output[2]["type"] == "function_call"
        assert response.output[2]["name"] == "add"


class TestClientToolSearchPattern:
    """Validates client-executed tool search flow."""

    def test_client_tool_search_call(self):
        """First turn: model emits tool_search_call with call_id."""
        from caip_responses.models.response import Response

        response = Response(
            id="resp_ts_client_1",
            model="gpt-5.4",
            output=[
                {
                    "type": "tool_search_call",
                    "id": "tsc_client",
                    "execution": "client",
                    "call_id": "call_ts_123",
                    "status": "completed",
                },
            ],
        )
        assert response.output[0]["type"] == "tool_search_call"
        assert response.output[0]["call_id"] == "call_ts_123"
        assert response.output[0]["execution"] == "client"
