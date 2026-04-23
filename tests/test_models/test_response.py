from __future__ import annotations

from caip_responses.models.items import (
    MessageOutputItem,
    OutputTextContent,
)
from caip_responses.models.response import Response


class TestResponse:
    def test_create_minimal(self):
        r = Response(id="resp_1", model="gpt-4.1")
        assert r.id == "resp_1"
        assert r.object == "response"
        assert r.status == "completed"
        assert r.output == []

    def test_output_text_from_dict(self, sample_text_response_data):
        r = Response(**sample_text_response_data)
        assert r.output_text == "Hello! How can I help you?"

    def test_output_text_from_model_objects(self):
        r = Response(
            id="resp_1",
            model="gpt-4.1",
            output=[
                MessageOutputItem(
                    id="item_1",
                    content=[
                        OutputTextContent(text="Hello "),
                        OutputTextContent(text="World"),
                    ],
                )
            ],
        )
        assert r.output_text == "Hello World"

    def test_output_text_empty(self):
        r = Response(id="resp_1", model="gpt-4.1", output=[])
        assert r.output_text == ""

    def test_has_function_calls_true(self, sample_function_call_response_data):
        r = Response(**sample_function_call_response_data)
        assert r.has_function_calls is True

    def test_has_function_calls_false(self, sample_text_response_data):
        r = Response(**sample_text_response_data)
        assert r.has_function_calls is False

    def test_function_calls_extraction(self, sample_function_call_response_data):
        r = Response(**sample_function_call_response_data)
        calls = r.function_calls
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].call_id == "fc_001"

    def test_usage(self, sample_text_response_data):
        r = Response(**sample_text_response_data)
        assert r.usage is not None
        assert r.usage.input_tokens == 10
        assert r.usage.output_tokens == 8
        assert r.usage.total_tokens == 18

    def test_response_allows_extra_fields(self):
        """Response model should allow extra fields for forward compatibility."""
        r = Response(id="resp_1", model="gpt-4.1", some_future_field="value")
        assert r.id == "resp_1"
