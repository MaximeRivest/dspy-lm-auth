from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dspy
import litellm

from dspy_lm_auth.auth import (
    OPENAI_CODEX_PROVIDER,
    AuthStorage,
    extract_chatgpt_account_id,
    get_default_auth_storage,
    getauthtoken,
    normalize_provider_id,
    set_default_auth_storage,
)

DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_ORIGINATOR = "dspy_lm_auth"
DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."

_DSPY_LM = dspy.LM
_ORIGINAL_DSPY_LM = dspy.LM

RouteResolver = Callable[[str, dict[str, Any], AuthStorage], tuple[str, dict[str, Any]]]
_ROUTE_RESOLVERS: dict[str, RouteResolver] = {}


@dataclass(frozen=True, slots=True)
class RouteRegistration:
    aliases: tuple[str, ...]
    resolver: RouteResolver


def _coerce_auth_storage(auth_storage: AuthStorage | str | os.PathLike[str] | None) -> AuthStorage:
    if auth_storage is None:
        return get_default_auth_storage()
    if isinstance(auth_storage, AuthStorage):
        return auth_storage
    return AuthStorage(auth_storage)


def register_model_alias(aliases: str | tuple[str, ...] | list[str], resolver: RouteResolver) -> None:
    if isinstance(aliases, str):
        aliases = (aliases,)
    for alias in aliases:
        _ROUTE_RESOLVERS[alias] = resolver


def unregister_model_alias(alias: str) -> None:
    _ROUTE_RESOLVERS.pop(alias, None)


def codex_headers(
    token: str,
    *,
    account_id: str | None = None,
    originator: str = DEFAULT_CODEX_ORIGINATOR,
    extra_headers: dict[str, Any] | None = None,
) -> dict[str, str]:
    resolved_account_id = account_id or extract_chatgpt_account_id(token)
    headers = {
        "chatgpt-account-id": resolved_account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": originator,
    }
    if extra_headers:
        headers.update({str(key): str(value) for key, value in extra_headers.items()})
    return headers


def _resolve_codex_route(model: str, kwargs: dict[str, Any], auth_storage: AuthStorage) -> tuple[str, dict[str, Any]]:
    if "/" in model:
        _, model_id = model.split("/", 1)
    else:
        model_id = DEFAULT_CODEX_MODEL

    resolved_kwargs = dict(kwargs)
    token = resolved_kwargs.get("api_key") or auth_storage.get_api_key(OPENAI_CODEX_PROVIDER)
    if not token:
        raise ValueError(
            "No OpenAI Codex credential found. Run `dspy_lm_auth.login('openai-codex')`, "
            "reuse Pi's auth.json, or pass `api_key=` explicitly."
        )

    credential = auth_storage.get(OPENAI_CODEX_PROVIDER)
    account_id = resolved_kwargs.pop("chatgpt_account_id", None)
    if account_id is None and isinstance(credential, dict):
        raw_account_id = credential.get("accountId")
        if isinstance(raw_account_id, str) and raw_account_id:
            account_id = raw_account_id

    originator = str(resolved_kwargs.pop("originator", DEFAULT_CODEX_ORIGINATOR))
    headers = codex_headers(
        token,
        account_id=account_id,
        originator=originator,
        extra_headers=resolved_kwargs.get("headers"),
    )

    resolved_kwargs["headers"] = headers
    resolved_kwargs.setdefault("api_key", token)
    resolved_kwargs.setdefault("api_base", DEFAULT_CODEX_API_BASE)
    resolved_kwargs.setdefault("model_type", "responses")
    resolved_kwargs.setdefault("use_developer_role", True)
    return f"openai/{model_id}", resolved_kwargs


def resolve_lm_route(
    model: str,
    *,
    auth_storage: AuthStorage,
    auth_provider: str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved_kwargs = dict(kwargs or {})

    if auth_provider:
        provider = normalize_provider_id(auth_provider)
        resolver = _ROUTE_RESOLVERS.get(provider)
        if resolver is None:
            raise ValueError(f"No DSPy LM auth route registered for auth_provider={auth_provider!r}")
        return resolver(model, resolved_kwargs, auth_storage)

    alias = model.split("/", 1)[0]
    resolver = _ROUTE_RESOLVERS.get(alias)
    if resolver is None and model in _ROUTE_RESOLVERS:
        resolver = _ROUTE_RESOLVERS[model]

    if resolver is None:
        if "api_key" not in resolved_kwargs:
            provider_prefix = model.split("/", 1)[0]
            api_key = auth_storage.get_api_key(provider_prefix)
            if api_key:
                resolved_kwargs["api_key"] = api_key
        return model, resolved_kwargs
    return resolver(model, resolved_kwargs, auth_storage)


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = headers or {}
    return {
        "User-Agent": f"DSPy/{dspy.__version__}",
        **headers,
    }


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                        if isinstance(url, str) and url:
                            parts.append(url)
                elif item_type == "input_image":
                    image_url = item.get("image_url")
                    if isinstance(image_url, str) and image_url:
                        parts.append(image_url)
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _convert_content_item_to_responses_format(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "image_url":
        image_url = item.get("image_url", {})
        if isinstance(image_url, dict):
            image_url = image_url.get("url", "")
        return {
            "type": "input_image",
            "image_url": image_url,
        }
    if item_type in {"text", "input_text"}:
        return {
            "type": "input_text",
            "text": item.get("text", ""),
        }
    if item_type == "file":
        file = item.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }
    return item


def _convert_message_content_to_responses_format(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                blocks.append(_convert_content_item_to_responses_format(item))
            elif item is not None:
                blocks.append({"type": "input_text", "text": str(item)})
        return blocks
    if content is None:
        return []
    return [{"type": "input_text", "text": str(content)}]


def _coerce_response_format(response_format: Any) -> Any:
    if hasattr(response_format, "model_json_schema") and hasattr(response_format, "__name__"):
        return {
            "name": response_format.__name__,
            "type": "json_schema",
            "schema": response_format.model_json_schema(),
        }
    return response_format


def _merge_codex_instructions(explicit_instructions: Any, instruction_messages: list[str]) -> str:
    parts: list[str] = []
    if explicit_instructions is not None:
        explicit_text = str(explicit_instructions).strip()
        if explicit_text:
            parts.append(explicit_text)

    for instruction in instruction_messages:
        cleaned = instruction.strip()
        if cleaned and cleaned not in parts:
            parts.append(cleaned)

    if not parts:
        parts.append(DEFAULT_CODEX_INSTRUCTIONS)
    return "\n\n".join(parts)


def _build_codex_responses_request(request: dict[str, Any]) -> dict[str, Any]:
    request = dict(request)
    messages = request.pop("messages", [])

    instructions_from_messages: list[str] = []
    input_messages: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role in {"system", "developer"}:
            instruction_text = _stringify_message_content(content).strip()
            if instruction_text:
                instructions_from_messages.append(instruction_text)
            continue

        input_messages.append(
            {
                "role": role,
                "content": _convert_message_content_to_responses_format(content),
            }
        )

    request["input"] = input_messages
    request["instructions"] = _merge_codex_instructions(
        request.pop("instructions", None),
        instructions_from_messages,
    )

    if request.get("max_output_tokens") is None:
        max_tokens = request.pop("max_tokens", None)
        if max_tokens is not None:
            request["max_output_tokens"] = max_tokens
    else:
        request.pop("max_tokens", None)

    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        request["reasoning"] = {"effort": effort, "summary": "auto"}

    if "response_format" in request:
        response_format = _coerce_response_format(request.pop("response_format"))
        text = request.pop("text", {}) or {}
        request["text"] = {**text, "format": response_format}

    request["store"] = False
    request["stream"] = True
    return request


def _consume_codex_response_stream(response_stream: Any) -> Any:
    if not hasattr(response_stream, "completed_response"):
        return response_stream

    for _ in response_stream:
        pass

    completed_event = getattr(response_stream, "completed_response", None)
    completed_response = getattr(completed_event, "response", None)
    if completed_response is None:
        raise RuntimeError("Codex response stream ended without a completed response")
    return completed_response


async def _aconsume_codex_response_stream(response_stream: Any) -> Any:
    if not hasattr(response_stream, "completed_response"):
        return response_stream

    async for _ in response_stream:
        pass

    completed_event = getattr(response_stream, "completed_response", None)
    completed_response = getattr(completed_event, "response", None)
    if completed_response is None:
        raise RuntimeError("Codex response stream ended without a completed response")
    return completed_response


def _litellm_codex_responses_completion(
    request: dict[str, Any],
    num_retries: int,
    cache: dict[str, Any] | None = None,
):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _build_codex_responses_request(request)

    response_stream = litellm.responses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )
    return _consume_codex_response_stream(response_stream)


async def _alitellm_codex_responses_completion(
    request: dict[str, Any],
    num_retries: int,
    cache: dict[str, Any] | None = None,
):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _build_codex_responses_request(request)

    response_stream = await litellm.aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )
    return await _aconsume_codex_response_stream(response_stream)


class LM(_DSPY_LM):
    """Drop-in replacement for ``dspy.LM`` with Pi-style auth and model aliases."""

    def __init__(
        self,
        model: str,
        *args,
        auth_storage: AuthStorage | str | os.PathLike[str] | None = None,
        auth_provider: str | None = None,
        **kwargs,
    ):
        storage = _coerce_auth_storage(auth_storage)
        resolved_model, resolved_kwargs = resolve_lm_route(
            model,
            auth_storage=storage,
            auth_provider=auth_provider,
            kwargs=kwargs,
        )
        self.auth_storage = storage
        self.original_model_string = model
        self.auth_provider = auth_provider
        self.resolved_model_string = resolved_model
        requested_route = normalize_provider_id(auth_provider) if auth_provider else normalize_provider_id(model.split("/", 1)[0])
        self._uses_codex_route = (
            requested_route == OPENAI_CODEX_PROVIDER or resolved_kwargs.get("api_base") == DEFAULT_CODEX_API_BASE
        )
        super().__init__(resolved_model, *args, **resolved_kwargs)

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        if not self._uses_codex_route:
            return super().forward(prompt=prompt, messages=messages, **kwargs)

        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        completion = _litellm_codex_responses_completion
        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)
        results = completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )

        self._check_truncation(results)

        if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
            dspy.settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        if not self._uses_codex_route:
            return await super().aforward(prompt=prompt, messages=messages, **kwargs)

        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        completion = _alitellm_codex_responses_completion
        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)
        results = await completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )

        self._check_truncation(results)

        if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
            dspy.settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results


def install(*, auth_storage: AuthStorage | str | os.PathLike[str] | None = None, attach_helpers: bool = True):
    storage = _coerce_auth_storage(auth_storage)
    set_default_auth_storage(storage)

    dspy.LM = LM
    dspy.clients.LM = LM
    if attach_helpers:
        dspy.getauthtoken = getauthtoken
    return LM


def uninstall() -> None:
    dspy.LM = _ORIGINAL_DSPY_LM
    dspy.clients.LM = _ORIGINAL_DSPY_LM
    if hasattr(dspy, "getauthtoken"):
        delattr(dspy, "getauthtoken")


register_model_alias(("codex", "chatgpt", OPENAI_CODEX_PROVIDER), _resolve_codex_route)


__all__ = [
    "DEFAULT_CODEX_API_BASE",
    "DEFAULT_CODEX_INSTRUCTIONS",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_ORIGINATOR",
    "LM",
    "RouteRegistration",
    "codex_headers",
    "install",
    "register_model_alias",
    "resolve_lm_route",
    "uninstall",
    "unregister_model_alias",
]
