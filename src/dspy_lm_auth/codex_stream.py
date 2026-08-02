"""Typed Codex Responses stream adaptation for DSPy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import litellm

_OutputLocation = tuple[int | None, int | None, str | None]


def _event_field(event: Any, name: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


def _plain_value(value: Any) -> Any:
    if isinstance(value, _ResponseObject):
        return value.model_dump()
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain_value(model_dump())
    return value


class _ResponseObject:
    """Small attribute adapter satisfying DSPy's Responses API read contract."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = {str(key): value for key, value in payload.items()}
        for key, value in self._payload.items():
            if key in {"usage", "_hidden_params"}:
                adapted = _usage_mapping(value)
            else:
                adapted = _adapt_response_value(value)
            setattr(self, key, adapted)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {key: _plain_value(getattr(self, key)) for key in self._payload}


def _object_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    model_fields = getattr(type(value), "model_fields", None)
    if isinstance(model_fields, Mapping):
        return {str(name): getattr(value, str(name), None) for name in model_fields}
    raw = getattr(value, "__dict__", None)
    if isinstance(raw, Mapping):
        return {str(key): item for key, item in raw.items() if not str(key).startswith("__")}
    return {}


def _adapt_response_value(value: Any) -> Any:
    if isinstance(value, _ResponseObject):
        return value
    if isinstance(value, Mapping):
        return _ResponseObject(value)
    if isinstance(value, list):
        return [_adapt_response_value(item) for item in value]
    return value


def _usage_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    payload = _object_payload(value)
    return payload if payload else {}


def _response_adapter(response: Any) -> _ResponseObject:
    payload = _object_payload(response)
    if not payload:
        raise RuntimeError("Codex completed response had an unsupported shape")
    for name in ("_hidden_params", "cache_hit"):
        value = getattr(response, name, None)
        if value is not None and name not in payload:
            payload[name] = value
    return _ResponseObject(payload)


def _completed_output(response: Any) -> tuple[str, bool]:
    output = _event_field(response, "output")
    if not isinstance(output, list):
        return "", False
    text_parts: list[str] = []
    refusal_seen = False
    for item in output:
        if _event_field(item, "type") != "message":
            continue
        content = _event_field(item, "content")
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = _event_field(block, "type")
            if block_type == "refusal":
                refusal_seen = True
            elif block_type == "output_text":
                text = _event_field(block, "text")
                if isinstance(text, str):
                    text_parts.append(text)
    return "".join(text_parts), refusal_seen


def _failure_detail(value: Any) -> str | None:
    if value is None:
        return None
    for path in (
        ("error", "message"),
        ("incomplete_details", "reason"),
        ("response", "error", "message"),
        ("response", "incomplete_details", "reason"),
        ("message",),
    ):
        current = value
        for name in path:
            current = _event_field(current, name)
            if current is None:
                break
        if current is not None and str(current).strip():
            return str(current).strip()
    return None


def _event_bucket(event_type: str, event: Any) -> str:
    if event_type == "response.output_text.delta":
        return "output_text_delta"
    if "reasoning" in event_type:
        return "reasoning"
    if "refusal" in event_type:
        return "refusal"
    if "tool" in event_type or "_call" in event_type:
        return "tool"
    if event_type in {"error", "response.error", "response.failed", "response.incomplete"}:
        return "failure"
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = _event_field(event, "item")
        item_type = _event_field(item, "type")
        if not isinstance(item_type, str):
            return "unknown"
        if "reasoning" in item_type:
            return "reasoning"
        if "refusal" in item_type:
            return "refusal"
        if "tool" in item_type or item_type.endswith("_call"):
            return "tool"
        if item_type != "message":
            return "unknown"
    if event_type.startswith("response."):
        return "lifecycle"
    return "unknown"


@dataclass
class _CodexStreamAccumulator:
    text_parts: list[str] = field(default_factory=list)
    output_locations: set[_OutputLocation | None] = field(default_factory=set)
    event_counts: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    refusal_seen: bool = False

    def add(self, event: Any) -> None:
        event_type = _event_field(event, "type")
        event_type = event_type if isinstance(event_type, str) else ""
        bucket = _event_bucket(event_type, event)
        self.event_counts[bucket] = self.event_counts.get(bucket, 0) + 1
        if event_type == "response.output_text.delta":
            delta = _event_field(event, "delta")
            if isinstance(delta, str):
                self.text_parts.append(delta)
                output_index = _event_field(event, "output_index")
                content_index = _event_field(event, "content_index")
                item_id = _event_field(event, "item_id")
                output_index_valid = (
                    isinstance(output_index, int) and not isinstance(output_index, bool) and output_index >= 0
                )
                content_index_valid = (
                    isinstance(content_index, int) and not isinstance(content_index, bool) and content_index >= 0
                )
                item_id_valid = item_id is None or isinstance(item_id, str)
                if output_index_valid and content_index_valid and item_id_valid:
                    self.output_locations.add((output_index, content_index, item_id))
                elif output_index is None and content_index is None and item_id_valid:
                    self.output_locations.add((None, None, item_id))
                else:
                    self.output_locations.add(None)
        elif event_type in {"response.refusal.delta", "response.refusal.done"}:
            self.refusal_seen = True
        elif bucket == "failure":
            self.failures.append(_failure_detail(event) or event_type or "unknown failure")

    def finish(
        self,
    ) -> tuple[str, dict[str, int], set[_OutputLocation | None]]:
        if self.failures:
            raise RuntimeError("Codex response stream ended with error: " + "; ".join(self.failures))
        if self.refusal_seen:
            raise RuntimeError("Codex response stream returned a refusal")
        return (
            "".join(self.text_parts),
            dict(sorted(self.event_counts.items())),
            set(self.output_locations),
        )


def _validate_completed_response(response: Any) -> _ResponseObject:
    adapted = _response_adapter(response)
    status = _event_field(adapted, "status")
    error = _event_field(adapted, "error")
    if error is not None:
        raise RuntimeError("Codex completed response returned an error: " + (_failure_detail(adapted) or "unknown"))
    if status not in {None, "completed"}:
        raise RuntimeError(f"Codex completed response status={status}: " + (_failure_detail(adapted) or "no detail"))
    _, refusal_seen = _completed_output(adapted)
    if refusal_seen:
        raise RuntimeError("Codex completed response returned a refusal")
    return adapted


class _LoggingStreamProxy:
    def __init__(self, stream: Any, completed_response: Any):
        self._stream = stream
        self.completed_response = completed_response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _normalized_response_api_usage(value: Any) -> Any:
    usage_type = getattr(litellm, "ResponseAPIUsage", None)
    if usage_type is None or isinstance(value, usage_type):
        return value
    usage = _usage_mapping(value)
    if not usage:
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens", int(input_tokens or 0) + int(output_tokens or 0))
    return usage_type(
        input_tokens=int(input_tokens or 0),
        input_tokens_details=usage.get("input_tokens_details", usage.get("prompt_tokens_details")),
        output_tokens=int(output_tokens or 0),
        output_tokens_details=usage.get("output_tokens_details", usage.get("completion_tokens_details")),
        total_tokens=int(total_tokens or 0),
    )


def _detached_completed_event_for_logging(event: Any) -> Any:
    response = _event_field(event, "response")
    if not isinstance(response, Mapping):
        return event
    response_type = getattr(litellm, "ResponsesAPIResponse", None)
    if response_type is None:
        raise RuntimeError("LiteLLM does not expose ResponsesAPIResponse")
    payload = {str(key): value for key, value in response.items()}
    payload["usage"] = _normalized_response_api_usage(payload.get("usage"))
    try:
        normalized_response = response_type(**payload)
        return type(event)(
            type=_event_field(event, "type") or "response.completed",
            response=normalized_response,
        )
    except Exception as exc:
        raise RuntimeError("Codex completed response could not be normalized for LiteLLM logging") from exc


def _patch_litellm_stream_logging(response_stream: Any) -> Any:
    original_function = getattr(type(response_stream), "_handle_logging_completed_response", None)
    if not callable(original_function) or getattr(response_stream, "_dspy_lm_auth_logging_patch", False):
        return response_stream

    def _handle_logging_completed_response_with_proxy() -> Any:
        event = getattr(response_stream, "completed_response", None)
        detached_event = _detached_completed_event_for_logging(event)
        if detached_event is event:
            return original_function(response_stream)
        return original_function(_LoggingStreamProxy(response_stream, detached_event))

    response_stream._handle_logging_completed_response = _handle_logging_completed_response_with_proxy
    response_stream._dspy_lm_auth_logging_patch = True
    return response_stream


def _apply_typed_stream_output(
    adapted: _ResponseObject,
    text: str,
    output_locations: set[_OutputLocation | None],
    event_counts: Mapping[str, int],
) -> None:
    raw_output = getattr(adapted, "output", None)
    output = _adapt_response_value(_plain_value(raw_output))
    if not isinstance(output, list):
        raise RuntimeError("Codex typed output stream item was absent from completion")

    if len(output_locations) != 1 or None in output_locations:
        raise RuntimeError("Codex typed output stream location was ambiguous")
    location = next(iter(output_locations))
    assert location is not None
    stream_output_index, stream_content_index, item_id = location

    def append_empty_target() -> tuple[int, int]:
        message: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        }
        if item_id is not None:
            message["id"] = item_id
        output.append(_ResponseObject(message))
        return len(output) - 1, 0

    if stream_output_index is None and stream_content_index is None:
        candidates: list[tuple[int, int]] = []
        for candidate_output_index, candidate_item in enumerate(output):
            if _event_field(candidate_item, "type") != "message":
                continue
            candidate_content = _event_field(candidate_item, "content")
            if not isinstance(candidate_content, list):
                continue
            for candidate_content_index, candidate_block in enumerate(candidate_content):
                if _event_field(candidate_block, "type") == "output_text" and _event_field(candidate_block, "text") in {
                    None,
                    "",
                }:
                    candidates.append((candidate_output_index, candidate_content_index))
        if len(candidates) == 1:
            output_index, content_index = candidates[0]
        elif not candidates and not output:
            output_index, content_index = append_empty_target()
        else:
            raise RuntimeError("Codex typed output stream location was ambiguous")
    elif isinstance(stream_output_index, int) and isinstance(stream_content_index, int):
        missing_nonanswer_items_only = not output and all(
            event_counts.get(bucket, 0) == 0 for bucket in ("failure", "refusal", "tool", "unknown")
        )
        if stream_content_index == 0 and (stream_output_index == len(output) or missing_nonanswer_items_only):
            output_index, content_index = append_empty_target()
        else:
            output_index = stream_output_index
            content_index = stream_content_index
    else:
        raise RuntimeError("Codex typed output stream location was ambiguous")

    if output_index >= len(output):
        if not output:
            raise RuntimeError("Codex typed output stream item was absent from empty completion")
        if output_index > len(output):
            raise RuntimeError("Codex typed output stream item had an index gap")
        raise RuntimeError("Codex typed output stream item was absent from completion")
    item = output[output_index]
    if _event_field(item, "type") != "message":
        raise RuntimeError("Codex typed output stream item was not a message")
    observed_item_id = _event_field(item, "id")
    if item_id is not None and observed_item_id != item_id:
        raise RuntimeError("Codex typed output stream item identity drifted")
    content = _event_field(item, "content")
    if not isinstance(content, list) or content_index >= len(content):
        raise RuntimeError("Codex typed output stream content was absent from completion")
    block = content[content_index]
    if _event_field(block, "type") != "output_text":
        raise RuntimeError("Codex typed output stream content was not output text")
    existing_text = _event_field(block, "text")
    if existing_text not in {None, ""}:
        raise RuntimeError("Codex typed output stream target was not empty")
    if not isinstance(block, _ResponseObject):
        raise RuntimeError("Codex typed output stream content was not detached")
    block.text = text
    adapted.output = output


def _finalize_codex_stream(
    response_stream: Any,
    text: str,
    event_counts: dict[str, int],
    output_locations: set[_OutputLocation | None],
) -> _ResponseObject:
    completed_event = getattr(response_stream, "completed_response", None)
    completed_response = getattr(completed_event, "response", None)
    if completed_response is None:
        raise RuntimeError("Codex response stream ended without a completed response")
    adapted = _validate_completed_response(completed_response)
    completed_text, _ = _completed_output(adapted)
    if completed_text:
        output_text = completed_text
        output_text_source = "completed_response"
    elif text:
        _apply_typed_stream_output(adapted, text, output_locations, event_counts)
        output_text = text
        output_text_source = "typed_stream"
    else:
        output_text = ""
        output_text_source = "none"
    adapted._dspy_lm_auth_stream = {
        "event_counts": event_counts,
        "completed_output_text": bool(completed_text),
        "output_text_chars": len(output_text),
        "output_text_source": output_text_source,
        "stream_output_text_chars": len(text),
        "stream_completed_match": completed_text == text,
    }
    return adapted


def get_stream_metadata(response: Any) -> dict[str, Any] | None:
    """Return bounded typed-stream metadata without completion content."""

    raw = getattr(response, "_dspy_lm_auth_stream", None)
    if not isinstance(raw, Mapping):
        return None
    counts = raw.get("event_counts")
    output_text_chars = raw.get("output_text_chars")
    completed_output_text = raw.get("completed_output_text")
    output_text_source = raw.get("output_text_source")
    stream_output_text_chars = raw.get("stream_output_text_chars")
    stream_completed_match = raw.get("stream_completed_match")
    allowed = {
        "failure",
        "lifecycle",
        "output_text_delta",
        "reasoning",
        "refusal",
        "tool",
        "unknown",
    }
    if (
        not isinstance(counts, Mapping)
        or any(str(key) not in allowed for key in counts)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values())
        or not isinstance(completed_output_text, bool)
        or output_text_source not in {"completed_response", "typed_stream", "none"}
        or not isinstance(output_text_chars, int)
        or isinstance(output_text_chars, bool)
        or output_text_chars < 0
        or not isinstance(stream_output_text_chars, int)
        or isinstance(stream_output_text_chars, bool)
        or stream_output_text_chars < 0
        or not isinstance(stream_completed_match, bool)
    ):
        return None
    return {
        "event_counts": {str(key): counts[key] for key in sorted(counts)},
        "completed_output_text": completed_output_text,
        "output_text_chars": output_text_chars,
        "output_text_source": output_text_source,
        "stream_output_text_chars": stream_output_text_chars,
        "stream_completed_match": stream_completed_match,
    }


def _consume_codex_response_stream(response_stream: Any) -> _ResponseObject:
    if not hasattr(response_stream, "completed_response"):
        return _validate_completed_response(response_stream)
    accumulator = _CodexStreamAccumulator()
    for event in response_stream:
        accumulator.add(event)
    text, event_counts, output_locations = accumulator.finish()
    return _finalize_codex_stream(response_stream, text, event_counts, output_locations)


async def _aconsume_codex_response_stream(response_stream: Any) -> _ResponseObject:
    if not hasattr(response_stream, "completed_response"):
        return _validate_completed_response(response_stream)
    accumulator = _CodexStreamAccumulator()
    async for event in response_stream:
        accumulator.add(event)
    text, event_counts, output_locations = accumulator.finish()
    return _finalize_codex_stream(response_stream, text, event_counts, output_locations)
