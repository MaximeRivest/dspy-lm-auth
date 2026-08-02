import asyncio
import base64
import copy
import json
import time
from types import SimpleNamespace

import dspy
import pytest
from litellm import ResponseAPIUsage, ResponseCompletedEvent, ResponsesAPIResponse

import dspy_lm_auth
import dspy_lm_auth.lm as dspy_lm_auth_lm
from dspy_lm_auth.auth import AuthStorage
from dspy_lm_auth.lm import DEFAULT_CODEX_API_BASE, DEFAULT_CODEX_INSTRUCTIONS


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_fake_jwt(account_id: str = "acct_test") -> str:
    header = _b64url({"alg": "none", "typ": "JWT"})
    payload = _b64url({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}})
    return f"{header}.{payload}.signature"


def make_auth_storage(tmp_path, account_id: str = "acct_test") -> AuthStorage:
    storage = AuthStorage(tmp_path / "auth.json")
    storage.set(
        "openai-codex",
        {
            "type": "oauth",
            "access": make_fake_jwt(account_id),
            "refresh": "refresh-token",
            "expires": int(time.time() * 1000) + 60_000,
            "accountId": account_id,
        },
    )
    return storage


def make_fake_responses_response(text: str = "Hello!"):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage={},
        model="gpt-5.4",
    )


def make_fake_responses_response_dict(text: str = "Hello!"):
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {},
        "model": "gpt-5.4",
    }


class FakeResponsesStream:
    def __init__(self, response):
        self.completed_response = SimpleNamespace(response=response)

    def __iter__(self):
        response = self.completed_response.response
        output = response["output"] if isinstance(response, dict) else response.output
        message = output[0]
        content = message["content"] if isinstance(message, dict) else message.content
        block = content[0]
        text = block["text"] if isinstance(block, dict) else block.text
        return iter([SimpleNamespace(type="response.output_text.delta", delta=text)])


def test_codex_alias_resolves_to_openai_responses_config(tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_codex")

    lm = dspy_lm_auth.LM("codex/gpt-5.4", auth_storage=storage)

    assert lm.original_model_string == "codex/gpt-5.4"
    assert lm.model == "openai/gpt-5.4"
    assert lm.model_type == "responses"
    assert lm.kwargs["api_base"] == DEFAULT_CODEX_API_BASE
    assert lm.kwargs["api_key"] == storage.get_api_key("openai-codex")
    assert lm.kwargs["headers"]["chatgpt-account-id"] == "acct_codex"
    assert lm.kwargs["headers"]["originator"] == "dspy_lm_auth"


def test_auth_provider_can_apply_codex_route_to_openai_model(tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_explicit")

    lm = dspy_lm_auth.LM("openai/gpt-5.4", auth_provider="codex", auth_storage=storage)

    assert lm.model == "openai/gpt-5.4"
    assert lm.model_type == "responses"
    assert lm.kwargs["headers"]["chatgpt-account-id"] == "acct_explicit"


def test_codex_forward_supplies_required_streaming_request(monkeypatch, tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_forward")
    captured = {}

    def fake_responses(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponsesStream(make_fake_responses_response("Hello from Codex"))

    monkeypatch.setattr(dspy_lm_auth_lm.litellm, "responses", fake_responses)

    lm = dspy_lm_auth.LM("codex/gpt-5.4", auth_storage=storage, cache=False)
    output = lm("hello")

    assert output == [{"text": "Hello from Codex"}]
    assert captured["instructions"] == DEFAULT_CODEX_INSTRUCTIONS
    assert captured["store"] is False
    assert captured["stream"] is True
    assert captured["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]


def test_codex_forward_normalizes_dict_response_for_dspy_3(monkeypatch, tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_dict_response")

    def fake_responses(*args, **kwargs):
        return FakeResponsesStream(make_fake_responses_response_dict("Dict response works"))

    monkeypatch.setattr(dspy_lm_auth_lm.litellm, "responses", fake_responses)

    lm = dspy_lm_auth.LM("codex/gpt-5.4", auth_storage=storage, cache=False)
    assert lm("hello") == [{"text": "Dict response works"}]


def test_codex_logging_uses_detached_typed_event_for_fallback_dict():
    raw_response = {
        "id": "resp-logging",
        "created_at": 1,
        "model": "gpt-5.4",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg-logging",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }
    original_event = ResponseCompletedEvent.model_construct(
        type="response.completed",
        response=raw_response,
    )
    observed = []

    class Stream:
        completed_response = original_event

        def _handle_logging_completed_response(self):
            observed.append(self.completed_response)

    stream = Stream()
    dspy_lm_auth_lm._patch_litellm_stream_logging(stream)
    stream._handle_logging_completed_response()

    assert stream.completed_response is original_event
    assert original_event.response is raw_response
    assert len(observed) == 1
    assert isinstance(observed[0], ResponseCompletedEvent)
    assert isinstance(observed[0].response, ResponsesAPIResponse)
    assert observed[0].response.usage.input_tokens == 2
    assert observed[0].response.usage.output_tokens == 3


def test_codex_stream_accepts_only_typed_output_text_deltas():
    response = make_fake_responses_response("answer")

    class MixedStream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {"type": "response.reasoning_summary_text.delta", "delta": "secret"},
                    SimpleNamespace(type="response.output_text.delta", delta="ans"),
                    {"type": "response.tool_call.delta", "delta": "tool"},
                    SimpleNamespace(type="response.output_text.delta", delta="wer"),
                    {"type": "response.completed", "delta": "lifecycle"},
                ]
            )

    result = dspy_lm_auth_lm._consume_codex_response_stream(MixedStream())

    assert result.output[0].content[0].text == "answer"
    assert result._dspy_lm_auth_stream["event_counts"] == {
        "lifecycle": 1,
        "output_text_delta": 2,
        "reasoning": 1,
        "tool": 1,
    }
    assert "secret" not in repr(result._dspy_lm_auth_stream)
    assert dspy_lm_auth.get_stream_metadata(result) == {
        "event_counts": {
            "lifecycle": 1,
            "output_text_delta": 2,
            "reasoning": 1,
            "tool": 1,
        },
        "output_text_chars": 6,
        "output_text_source": "completed_response",
        "completed_output_text": True,
        "stream_output_text_chars": 6,
        "stream_completed_match": True,
    }
    result._dspy_lm_auth_stream["event_counts"] = {"unbounded-provider-key": 1}
    assert dspy_lm_auth.get_stream_metadata(result) is None


def test_codex_stream_supplies_typed_text_when_completed_payload_omits_it():
    response = make_fake_responses_response_dict("")
    response["output"] = [
        {"type": "reasoning", "id": "reason-1", "content": []},
        {
            "type": "message",
            "id": "msg-fallback",
            "content": [{"type": "output_text", "text": ""}],
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": "{}",
        },
    ]
    original = copy.deepcopy(response)

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {"type": "response.reasoning_summary_text.delta", "delta": "private"},
                    {
                        "type": "response.output_text.delta",
                        "delta": "answer",
                        "output_index": 1,
                        "content_index": 0,
                        "item_id": "msg-fallback",
                    },
                ]
            )

    result = dspy_lm_auth_lm._consume_codex_response_stream(Stream())
    lm = object.__new__(dspy_lm_auth.LM)
    processed = dspy.LM._process_response(lm, result)

    assert response == original
    assert processed[0]["text"] == "answer"
    assert processed[0]["tool_calls"][0]["name"] == "lookup"
    assert [item.type for item in result.output] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert result.output[1].content[0].text == "answer"
    assert result._dspy_lm_auth_stream == {
        "event_counts": {"output_text_delta": 1, "reasoning": 1},
        "completed_output_text": False,
        "output_text_chars": 6,
        "output_text_source": "typed_stream",
        "stream_output_text_chars": 6,
        "stream_completed_match": False,
    }
    assert "private" not in repr(result._dspy_lm_auth_stream)


def test_codex_stream_uses_unique_empty_target_when_event_indices_are_absent():
    response = make_fake_responses_response_dict("")

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter([{"type": "response.output_text.delta", "delta": "answer"}])

    result = dspy_lm_auth_lm._consume_codex_response_stream(Stream())

    assert result.output[0].content[0].text == "answer"
    assert result._dspy_lm_auth_stream["output_text_source"] == "typed_stream"


def test_codex_stream_supplies_missing_output_item_when_completion_output_is_empty():
    response = make_fake_responses_response_dict("")
    response["output"] = []
    original = copy.deepcopy(response)

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {
                        "type": "response.output_text.delta",
                        "delta": "answer",
                        "item_id": "msg-streamed",
                    }
                ]
            )

    result = dspy_lm_auth_lm._consume_codex_response_stream(Stream())

    assert response == original
    assert result.output[0].id == "msg-streamed"
    assert result.output[0].content[0].text == "answer"
    assert result._dspy_lm_auth_stream["output_text_source"] == "typed_stream"


def test_codex_stream_supplies_indexed_output_item_at_completion_tail():
    response = make_fake_responses_response_dict("")
    response["output"] = [{"type": "reasoning", "id": "reason-1", "content": []}]
    original = copy.deepcopy(response)

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {
                        "type": "response.output_text.delta",
                        "delta": "answer",
                        "output_index": 1,
                        "content_index": 0,
                        "item_id": "msg-streamed",
                    }
                ]
            )

    result = dspy_lm_auth_lm._consume_codex_response_stream(Stream())

    assert response == original
    assert [item.type for item in result.output] == ["reasoning", "message"]
    assert result.output[1].id == "msg-streamed"
    assert result.output[1].content[0].text == "answer"


def test_codex_stream_normalizes_empty_completion_with_only_reasoning_gap():
    response = make_fake_responses_response_dict("")
    response["output"] = []
    original = copy.deepcopy(response)

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "delta": "private",
                        "output_index": 0,
                    },
                    {
                        "type": "response.output_text.delta",
                        "delta": "answer",
                        "output_index": 1,
                        "content_index": 0,
                        "item_id": "msg-streamed",
                    },
                ]
            )

    result = dspy_lm_auth_lm._consume_codex_response_stream(Stream())

    assert response == original
    assert result.output[0].id == "msg-streamed"
    assert result.output[0].content[0].text == "answer"
    assert result._dspy_lm_auth_stream["event_counts"]["reasoning"] == 1
    assert "private" not in repr(result._dspy_lm_auth_stream)

    class ToolStream(Stream):
        def __iter__(self):
            return iter(
                [
                    {"type": "response.web_search_call.completed"},
                    {
                        "type": "response.output_item.added",
                        "item": {"type": "file_search_call"},
                    },
                    {"type": "response.tool_call.delta", "delta": "tool"},
                    {
                        "type": "response.output_text.delta",
                        "delta": "answer",
                        "output_index": 1,
                        "content_index": 0,
                        "item_id": "msg-streamed",
                    },
                ]
            )

    with pytest.raises(RuntimeError, match="absent from empty completion"):
        dspy_lm_auth_lm._consume_codex_response_stream(ToolStream())

    class UnboundedGapStream(Stream):
        def __init__(self, output_index: int):
            self.output_index = output_index

        def __iter__(self):
            return iter(
                [
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "delta": "private",
                        "output_index": 0,
                    },
                    {
                        "type": "response.output_text.delta",
                        "delta": "answer",
                        "output_index": self.output_index,
                        "content_index": 0,
                        "item_id": "msg-streamed",
                    },
                ]
            )

    for unexplained_index in (2, 65, 999_999):
        with pytest.raises(RuntimeError, match="absent from empty completion"):
            dspy_lm_auth_lm._consume_codex_response_stream(UnboundedGapStream(unexplained_index))


def test_codex_stream_rejects_ambiguous_typed_text_fallback():
    response = make_fake_responses_response_dict("")

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {
                        "type": "response.output_text.delta",
                        "delta": "a",
                        "output_index": 0,
                        "content_index": 0,
                    },
                    {
                        "type": "response.output_text.delta",
                        "delta": "b",
                        "output_index": 1,
                        "content_index": 0,
                    },
                ]
            )

    with pytest.raises(RuntimeError, match="location was ambiguous"):
        dspy_lm_auth_lm._consume_codex_response_stream(Stream())

    class PartialLocationStream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {
                        "type": "response.output_text.delta",
                        "delta": "answer",
                        "output_index": 0,
                    }
                ]
            )

    with pytest.raises(RuntimeError, match="location was ambiguous"):
        dspy_lm_auth_lm._consume_codex_response_stream(PartialLocationStream())

    class ConflictingItemStream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {
                        "type": "response.output_text.delta",
                        "delta": "answer",
                        "item_id": "different-message",
                    }
                ]
            )

    with pytest.raises(RuntimeError, match="identity drifted"):
        dspy_lm_auth_lm._consume_codex_response_stream(ConflictingItemStream())

    response["output"].append(
        {
            "type": "message",
            "id": "second-empty",
            "content": [{"type": "output_text", "text": ""}],
        }
    )

    class UnindexedStream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter([{"type": "response.output_text.delta", "delta": "answer"}])

    with pytest.raises(RuntimeError, match="location was ambiguous"):
        dspy_lm_auth_lm._consume_codex_response_stream(UnindexedStream())


def test_codex_stream_rejects_refusal_error_and_completed_text_drift():
    response = make_fake_responses_response_dict("answer")

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __init__(self, events):
            self.events = events

        def __iter__(self):
            return iter(self.events)

    with pytest.raises(RuntimeError, match="refusal"):
        dspy_lm_auth_lm._consume_codex_response_stream(
            Stream(
                [
                    {"type": "response.output_text.delta", "delta": "answer"},
                    {"type": "response.refusal.delta", "delta": "no"},
                ]
            )
        )
    with pytest.raises(RuntimeError, match="rate limited"):
        dspy_lm_auth_lm._consume_codex_response_stream(
            Stream(
                [
                    {"type": "response.output_text.delta", "delta": "answer"},
                    {
                        "type": "response.failed",
                        "error": {"message": "rate limited"},
                    },
                ]
            )
        )
    drift = dspy_lm_auth_lm._consume_codex_response_stream(
        Stream([{"type": "response.output_text.delta", "delta": "wrong"}])
    )
    assert drift.output[0].content[0].text == "answer"
    assert drift._dspy_lm_auth_stream["stream_completed_match"] is False

    response["output"] = [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]
    with pytest.raises(RuntimeError, match="refusal"):
        dspy_lm_auth_lm._consume_codex_response_stream(Stream([]))


def test_codex_stream_preserves_pydantic_response_and_dspy_contract():
    response = ResponsesAPIResponse(
        id="resp-1",
        created_at=1,
        model="gpt-5.4",
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [],
                    }
                ],
            }
        ],
        status="completed",
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )
    original = response.model_dump()

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {"type": "response.reasoning_summary_text.delta", "delta": "private"},
                    {"type": "response.output_text.delta", "delta": "answer"},
                ]
            )

    result = dspy_lm_auth_lm._consume_codex_response_stream(Stream())
    lm = object.__new__(dspy_lm_auth.LM)

    assert response.model_dump() == original
    assert dspy.LM._process_response(lm, result) == [{"text": "answer"}]
    assert result.model_dump()["output"] == original["output"]
    assert result._dspy_lm_auth_stream == {
        "event_counts": {"output_text_delta": 1, "reasoning": 1},
        "completed_output_text": True,
        "output_text_chars": 6,
        "output_text_source": "completed_response",
        "stream_output_text_chars": 6,
        "stream_completed_match": True,
    }
    assert "private" not in repr(result._dspy_lm_auth_stream)


def test_codex_response_adapter_supports_mapping_function_calls():
    response = {
        "id": "resp-tool",
        "created_at": 1,
        "model": "gpt-5.4",
        "object": "response",
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "output": [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{}",
            }
        ],
    }

    result = dspy_lm_auth_lm._consume_codex_response_stream(response)
    lm = object.__new__(dspy_lm_auth.LM)

    assert dspy.LM._process_response(lm, result) == [
        {
            "tool_calls": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ]
        }
    ]


def test_codex_direct_response_rejects_refusal_incomplete_and_error():
    refusal = make_fake_responses_response_dict("")
    refusal["output"] = [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]
    with pytest.raises(RuntimeError, match="refusal"):
        dspy_lm_auth_lm._consume_codex_response_stream(refusal)

    incomplete = make_fake_responses_response_dict("partial")
    incomplete.update(
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )
    with pytest.raises(RuntimeError, match="max_output_tokens"):
        dspy_lm_auth_lm._consume_codex_response_stream(incomplete)

    failed = make_fake_responses_response_dict("")
    failed.update(status="failed", error={"message": "upstream failed"})
    with pytest.raises(RuntimeError, match="upstream failed"):
        dspy_lm_auth_lm._consume_codex_response_stream(failed)


def test_codex_stream_surfaces_nested_failure_detail():
    response = make_fake_responses_response_dict("")

    class Stream:
        completed_response = SimpleNamespace(response=response)

        def __iter__(self):
            return iter(
                [
                    {
                        "type": "response.incomplete",
                        "response": {"incomplete_details": {"reason": "content_filter"}},
                    }
                ]
            )

    with pytest.raises(RuntimeError, match="content_filter"):
        dspy_lm_auth_lm._consume_codex_response_stream(Stream())


def test_codex_async_stream_has_typed_event_parity():
    response = make_fake_responses_response("answer")

    class AsyncStream:
        completed_response = SimpleNamespace(response=response)

        def __aiter__(self):
            async def events():
                yield {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "private",
                }
                yield {"type": "response.output_text.delta", "delta": "answer"}

            return events()

    result = asyncio.run(dspy_lm_auth_lm._aconsume_codex_response_stream(AsyncStream()))

    assert result.output[0].content[0].text == "answer"
    assert result._dspy_lm_auth_stream["event_counts"] == {
        "output_text_delta": 1,
        "reasoning": 1,
    }


def test_codex_forward_moves_system_messages_into_instructions(monkeypatch, tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_messages")
    captured = {}

    def fake_responses(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponsesStream(make_fake_responses_response("Done"))

    monkeypatch.setattr(dspy_lm_auth_lm.litellm, "responses", fake_responses)

    lm = dspy_lm_auth.LM("codex/gpt-5.4", auth_storage=storage, cache=False)
    output = lm(
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        ]
    )

    assert output == [{"text": "Done"}]
    assert captured["instructions"] == "Be terse."
    assert captured["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "Hi."}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "How are you?"}],
        },
    ]


def test_codex_route_does_not_use_openai_api_key_env(monkeypatch, tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_env")
    captured = {}
    env_sentinel = "ENV_SENTINEL_SHOULD_NOT_BE_USED"

    monkeypatch.setenv("OPENAI_API_KEY", env_sentinel)

    def fake_responses(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponsesStream(make_fake_responses_response("No env leak"))

    monkeypatch.setattr(dspy_lm_auth_lm.litellm, "responses", fake_responses)

    lm = dspy_lm_auth.LM("codex/gpt-5.4", auth_storage=storage, cache=False)
    output = lm("hello")

    assert output == [{"text": "No env leak"}]
    assert captured["api_key"] == storage.get_api_key("openai-codex")
    assert captured["api_key"] != env_sentinel
    assert lm.kwargs["api_key"] == storage.get_api_key("openai-codex")
    assert lm.kwargs["api_key"] != env_sentinel


def test_install_monkeypatches_dspy_lm(tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_install")
    original_lm = dspy.LM

    try:
        dspy_lm_auth.install(auth_storage=storage)
        assert dspy.LM is dspy_lm_auth.LM
        lm = dspy.LM("codex", auth_storage=storage)
        assert lm.model == "openai/gpt-5.4"
        assert dspy.getauthtoken("codex") == storage.get_api_key("openai-codex")
    finally:
        dspy_lm_auth.uninstall()
        assert dspy.LM is original_lm
