import base64
import json
import time
from types import SimpleNamespace

import dspy

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


def make_auth_storage(tmp_path, account_id: str = "acct_test", provider: str = "openai-codex") -> AuthStorage:
    storage = AuthStorage(tmp_path / "auth.json")
    storage.set(
        provider,
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
                content=[SimpleNamespace(text=text)],
            )
        ],
        usage={},
        model="gpt-5.4",
    )


class FakeResponsesStream:
    def __init__(self, response):
        self.completed_response = SimpleNamespace(response=response)

    def __iter__(self):
        return iter(())


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


def test_suffixed_codex_auth_provider_uses_matching_storage_key(tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_second", provider="openai-codex-2")

    lm = dspy_lm_auth.LM("openai/gpt-5.4", auth_provider="openai-codex-2", auth_storage=storage)

    assert lm.model == "openai/gpt-5.4"
    assert lm.model_type == "responses"
    assert lm.kwargs["api_key"] == storage.get_api_key("openai-codex-2")
    assert lm.kwargs["headers"]["chatgpt-account-id"] == "acct_second"


def test_suffixed_codex_model_alias_selects_matching_storage_key(tmp_path):
    storage = make_auth_storage(tmp_path, account_id="acct_alias", provider="openai-codex-work")

    lm = dspy_lm_auth.LM("codex-work/gpt-5.4", auth_storage=storage)

    assert lm.original_model_string == "codex-work/gpt-5.4"
    assert lm.model == "openai/gpt-5.4"
    assert lm.model_type == "responses"
    assert lm.kwargs["api_key"] == storage.get_api_key("openai-codex-work")
    assert lm.kwargs["headers"]["chatgpt-account-id"] == "acct_alias"


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
            "content": [{"type": "output_text", "text": "Hi."}],
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
