from __future__ import annotations

from caip_responses.models.streaming import (
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ReasoningTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseInProgressEvent,
    StreamEvent,
)


class TestStreamEvent:
    def test_basic(self):
        event = StreamEvent(type="response.output_text.delta", delta="Hello")
        assert event.type == "response.output_text.delta"
        assert event.delta == "Hello"

    def test_with_sequence_number(self):
        event = StreamEvent(type="response.created", sequence_number=0)
        assert event.sequence_number == 0

    def test_with_response(self):
        event = StreamEvent(
            type="response.created",
            response={"id": "resp_1", "model": "gpt-4.1", "status": "in_progress"},
        )
        assert event.response["id"] == "resp_1"

    def test_with_output_index(self):
        event = StreamEvent(
            type="response.output_item.added",
            output_index=0,
            item={"type": "message", "id": "item_1"},
        )
        assert event.output_index == 0


class TestConcreteEventTypes:
    def test_response_created(self):
        event = ResponseCreatedEvent()
        assert event.type == "response.created"

    def test_response_completed(self):
        event = ResponseCompletedEvent()
        assert event.type == "response.completed"

    def test_response_failed(self):
        event = ResponseFailedEvent()
        assert event.type == "response.failed"

    def test_response_in_progress(self):
        event = ResponseInProgressEvent()
        assert event.type == "response.in_progress"

    def test_output_text_delta(self):
        event = OutputTextDeltaEvent(delta="word")
        assert event.type == "response.output_text.delta"
        assert event.delta == "word"

    def test_output_text_done(self):
        event = OutputTextDoneEvent(delta="full text")
        assert event.type == "response.output_text.done"

    def test_function_call_arguments_delta(self):
        event = FunctionCallArgumentsDeltaEvent(delta='{"city')
        assert event.type == "response.function_call_arguments.delta"

    def test_function_call_arguments_done(self):
        event = FunctionCallArgumentsDoneEvent(delta='{"city": "SF"}')
        assert event.type == "response.function_call_arguments.done"

    def test_output_item_added(self):
        event = OutputItemAddedEvent(output_index=0)
        assert event.type == "response.output_item.added"

    def test_output_item_done(self):
        event = OutputItemDoneEvent(output_index=0)
        assert event.type == "response.output_item.done"

    def test_content_part_added(self):
        event = ContentPartAddedEvent(content_index=0)
        assert event.type == "response.content_part.added"

    def test_content_part_done(self):
        event = ContentPartDoneEvent(content_index=0)
        assert event.type == "response.content_part.done"

    def test_reasoning_text_delta(self):
        event = ReasoningTextDeltaEvent(delta="thinking...")
        assert event.type == "response.reasoning_text.delta"
