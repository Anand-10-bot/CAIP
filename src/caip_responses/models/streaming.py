from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class StreamEvent(BaseModel):
    """Base streaming event — all providers emit these same event types."""

    model_config = ConfigDict(extra="allow")

    type: str
    sequence_number: int | None = None

    # Present on response-level events
    response: dict[str, Any] | None = None

    # Present on output item events
    output_index: int | None = None
    item: dict[str, Any] | None = None

    # Present on content part events
    content_index: int | None = None
    part: dict[str, Any] | None = None

    # Present on delta events
    delta: str | None = None


# ---------------------------------------------------------------------------
# Concrete event types for type narrowing
# ---------------------------------------------------------------------------

class ResponseCreatedEvent(StreamEvent):
    type: Literal["response.created"] = "response.created"


class ResponseInProgressEvent(StreamEvent):
    type: Literal["response.in_progress"] = "response.in_progress"


class ResponseCompletedEvent(StreamEvent):
    type: Literal["response.completed"] = "response.completed"


class ResponseFailedEvent(StreamEvent):
    type: Literal["response.failed"] = "response.failed"


class ResponseIncompleteEvent(StreamEvent):
    type: Literal["response.incomplete"] = "response.incomplete"


class OutputItemAddedEvent(StreamEvent):
    type: Literal["response.output_item.added"] = "response.output_item.added"


class OutputItemDoneEvent(StreamEvent):
    type: Literal["response.output_item.done"] = "response.output_item.done"


class ContentPartAddedEvent(StreamEvent):
    type: Literal["response.content_part.added"] = "response.content_part.added"


class ContentPartDoneEvent(StreamEvent):
    type: Literal["response.content_part.done"] = "response.content_part.done"


class OutputTextDeltaEvent(StreamEvent):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"


class OutputTextDoneEvent(StreamEvent):
    type: Literal["response.output_text.done"] = "response.output_text.done"


class ReasoningTextDeltaEvent(StreamEvent):
    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"


class ReasoningTextDoneEvent(StreamEvent):
    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"


class FunctionCallArgumentsDeltaEvent(StreamEvent):
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"


class FunctionCallArgumentsDoneEvent(StreamEvent):
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
