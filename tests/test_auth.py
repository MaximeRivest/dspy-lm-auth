import base64
import json
import time

from dspy_lm_auth.auth import (
    AuthStorage,
    extract_chatgpt_account_id,
    get_oauth_provider,
    is_openai_codex_provider,
    normalize_provider_id,
    register_oauth_provider,
    resolve_config_value,
)


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_fake_jwt(account_id: str = "acct_test") -> str:
    header = _b64url({"alg": "none", "typ": "JWT"})
    payload = _b64url({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}})
    return f"{header}.{payload}.signature"


def test_resolve_config_value_prefers_environment_variable(monkeypatch):
    monkeypatch.setenv("DSPY_LM_AUTH_TEST_KEY", "env-value")
    assert resolve_config_value("DSPY_LM_AUTH_TEST_KEY") == "env-value"


def test_extract_chatgpt_account_id_reads_jwt_claim():
    token = make_fake_jwt("acct_123")
    assert extract_chatgpt_account_id(token) == "acct_123"


def test_codex_provider_suffixes_normalize_to_distinct_storage_keys():
    assert normalize_provider_id("codex-2") == "openai-codex-2"
    assert normalize_provider_id("chatgpt-work") == "openai-codex-work"
    assert is_openai_codex_provider("openai-codex-2")
    assert get_oauth_provider("openai-codex-2") is get_oauth_provider("openai-codex")


def test_auth_storage_resolves_suffixed_codex_oauth_credentials(tmp_path):
    auth_path = tmp_path / "auth.json"
    storage = AuthStorage(auth_path)
    storage.set(
        "openai-codex-2",
        {
            "type": "oauth",
            "access": make_fake_jwt("acct_second"),
            "refresh": "refresh-token",
            "expires": int(time.time() * 1000) + 60_000,
            "accountId": "acct_second",
        },
    )

    assert storage.get_api_key("codex-2") == make_fake_jwt("acct_second")


def test_auth_storage_refreshes_suffixed_codex_oauth_credentials(tmp_path):
    auth_path = tmp_path / "auth.json"
    storage = AuthStorage(auth_path)
    refreshed_token = make_fake_jwt("acct_refreshed_second")
    original_provider = get_oauth_provider("openai-codex")

    class TestCodexOAuthProvider:
        id = "openai-codex"
        name = "Test Codex OAuth"

        def login(self, **kwargs):
            raise NotImplementedError

        def refresh_token(self, credentials):
            assert credentials["refresh"] == "refresh-second"
            return {
                "type": "oauth",
                "access": refreshed_token,
                "refresh": "refresh-second-2",
                "expires": int(time.time() * 1000) + 60_000,
                "accountId": "acct_refreshed_second",
            }

        def get_api_key(self, credentials):
            return credentials["access"]

    try:
        register_oauth_provider(TestCodexOAuthProvider())
        storage.set(
            "openai-codex-2",
            {
                "type": "oauth",
                "access": make_fake_jwt("acct_old_second"),
                "refresh": "refresh-second",
                "expires": int(time.time() * 1000) - 1_000,
                "accountId": "acct_old_second",
            },
        )

        token = storage.get_api_key("codex-2")
    finally:
        if original_provider is not None:
            register_oauth_provider(original_provider)

    assert token == refreshed_token
    persisted = json.loads(auth_path.read_text())
    assert persisted["openai-codex-2"]["access"] == refreshed_token
    assert persisted["openai-codex-2"]["refresh"] == "refresh-second-2"
    assert "openai-codex" not in persisted


def test_auth_storage_resolves_api_key_credentials(monkeypatch, tmp_path):
    auth_path = tmp_path / "auth.json"
    storage = AuthStorage(auth_path)
    monkeypatch.setenv("DSPY_LM_AUTH_LITERAL_KEY", "resolved-from-env")

    storage.set("custom-provider", {"type": "api_key", "key": "DSPY_LM_AUTH_LITERAL_KEY"})

    assert storage.get_api_key("custom-provider") == "resolved-from-env"


def test_auth_storage_refreshes_expired_oauth_credentials(tmp_path):
    auth_path = tmp_path / "auth.json"
    storage = AuthStorage(auth_path)

    refreshed_token = make_fake_jwt("acct_refreshed")

    class TestOAuthProvider:
        id = "test-oauth"
        name = "Test OAuth"

        def login(self, **kwargs):
            raise NotImplementedError

        def refresh_token(self, credentials):
            return {
                "type": "oauth",
                "access": refreshed_token,
                "refresh": "refresh-2",
                "expires": int(time.time() * 1000) + 60_000,
            }

        def get_api_key(self, credentials):
            return credentials["access"]

    register_oauth_provider(TestOAuthProvider())

    storage.set(
        "test-oauth",
        {
            "type": "oauth",
            "access": make_fake_jwt("acct_old"),
            "refresh": "refresh-1",
            "expires": int(time.time() * 1000) - 1_000,
        },
    )

    token = storage.get_api_key("test-oauth")

    assert token == refreshed_token
    persisted = json.loads(auth_path.read_text())
    assert persisted["test-oauth"]["access"] == refreshed_token
    assert persisted["test-oauth"]["refresh"] == "refresh-2"
