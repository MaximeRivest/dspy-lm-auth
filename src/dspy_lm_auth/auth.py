from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import subprocess
import threading
import time
import webbrowser
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast
from urllib.parse import parse_qs, urlencode, urlparse

import requests

DEFAULT_PI_AUTH_PATH = Path("~/.pi/agent/auth.json").expanduser()
OPENAI_CODEX_PROVIDER = "openai-codex"
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_CODEX_SCOPE = "openid profile email offline_access"
OPENAI_CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"

ENV_API_KEY_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
    "zai": "ZAI_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
    "huggingface": "HF_TOKEN",
    "kimi-coding": "KIMI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
}

_AUTH_PROVIDER_ALIASES = {
    "codex": OPENAI_CODEX_PROVIDER,
    "chatgpt": OPENAI_CODEX_PROVIDER,
    OPENAI_CODEX_PROVIDER: OPENAI_CODEX_PROVIDER,
}

_COMMAND_RESULT_CACHE: dict[str, str | None] = {}
_COMMAND_RESULT_CACHE_LOCK = threading.Lock()
_DEFAULT_AUTH_STORAGE: AuthStorage | None = None
_OAUTH_PROVIDERS: dict[str, OAuthProvider] = {}


class ApiKeyCredential(TypedDict):
    type: Literal["api_key"]
    key: str


class OAuthCredential(TypedDict, total=False):
    type: Literal["oauth"]
    access: str
    refresh: str
    expires: int
    accountId: str
    projectId: str


Credential = ApiKeyCredential | OAuthCredential


class OAuthProvider(Protocol):
    id: str
    name: str

    def login(self, **kwargs) -> OAuthCredential:
        ...

    def refresh_token(self, credentials: OAuthCredential) -> OAuthCredential:
        ...

    def get_api_key(self, credentials: OAuthCredential) -> str:
        ...


@contextmanager
def _file_lock(handle):
    try:
        import fcntl
    except ImportError:
        yield
        return

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked_json_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}\n", encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    with path.open("r+", encoding="utf-8") as handle:
        with _file_lock(handle):
            yield handle


def normalize_provider_id(provider: str) -> str:
    aliased = _AUTH_PROVIDER_ALIASES.get(provider)
    if aliased is not None:
        return aliased
    for prefix in ("codex-", "chatgpt-"):
        if provider.startswith(prefix) and len(provider) > len(prefix):
            return f"{OPENAI_CODEX_PROVIDER}-{provider[len(prefix):]}"
    return provider


def is_openai_codex_provider(provider: str) -> bool:
    provider = normalize_provider_id(provider)
    return provider == OPENAI_CODEX_PROVIDER or provider.startswith(f"{OPENAI_CODEX_PROVIDER}-")


def clear_command_cache() -> None:
    with _COMMAND_RESULT_CACHE_LOCK:
        _COMMAND_RESULT_CACHE.clear()


def resolve_config_value(config: str) -> str | None:
    if not config:
        return None
    if config.startswith("!"):
        return _execute_command(config)

    env_value = os.getenv(config)
    return env_value or config


def _execute_command(command_config: str) -> str | None:
    with _COMMAND_RESULT_CACHE_LOCK:
        if command_config in _COMMAND_RESULT_CACHE:
            return _COMMAND_RESULT_CACHE[command_config]

    command = command_config[1:]
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result = completed.stdout.strip() or None
    except Exception:
        result = None

    with _COMMAND_RESULT_CACHE_LOCK:
        _COMMAND_RESULT_CACHE[command_config] = result
    return result


class AuthStorage:
    """Credential storage compatible with Pi's auth.json format."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path).expanduser() if path else DEFAULT_PI_AUTH_PATH
        self.runtime_overrides: dict[str, str] = {}
        self.data: dict[str, Credential] = {}
        self.reload()

    def reload(self) -> None:
        with _locked_json_file(self.path) as handle:
            handle.seek(0)
            content = handle.read().strip() or "{}"
        self.data = cast(dict[str, Credential], json.loads(content))

    def get(self, provider: str) -> Credential | None:
        self.reload()
        return self.data.get(normalize_provider_id(provider))

    def set(self, provider: str, credential: Credential) -> None:
        provider = normalize_provider_id(provider)
        with _locked_json_file(self.path) as handle:
            handle.seek(0)
            current = json.loads(handle.read().strip() or "{}")
            current[provider] = credential
            handle.seek(0)
            handle.truncate(0)
            json.dump(current, handle, indent=2)
            handle.write("\n")
        self.data[provider] = credential

    def remove(self, provider: str) -> None:
        provider = normalize_provider_id(provider)
        with _locked_json_file(self.path) as handle:
            handle.seek(0)
            current = json.loads(handle.read().strip() or "{}")
            current.pop(provider, None)
            handle.seek(0)
            handle.truncate(0)
            json.dump(current, handle, indent=2)
            handle.write("\n")
        self.data.pop(provider, None)

    def set_runtime_api_key(self, provider: str, api_key: str) -> None:
        self.runtime_overrides[normalize_provider_id(provider)] = api_key

    def remove_runtime_api_key(self, provider: str) -> None:
        self.runtime_overrides.pop(normalize_provider_id(provider), None)

    def has_auth(self, provider: str) -> bool:
        provider = normalize_provider_id(provider)
        if provider in self.runtime_overrides:
            return True
        if self.get(provider) is not None:
            return True
        env_var = ENV_API_KEY_BY_PROVIDER.get(provider)
        return bool(env_var and os.getenv(env_var))

    def login(self, provider: str, **kwargs) -> OAuthCredential:
        provider = normalize_provider_id(provider)
        oauth_provider = get_oauth_provider(provider)
        if oauth_provider is None:
            raise ValueError(f"Unknown OAuth provider: {provider}")

        credential = oauth_provider.login(**kwargs)
        full_credential: OAuthCredential = {"type": "oauth", **credential}
        self.set(provider, full_credential)
        return full_credential

    def logout(self, provider: str) -> None:
        self.remove(provider)

    def get_api_key(self, provider: str) -> str | None:
        provider = normalize_provider_id(provider)
        runtime_override = self.runtime_overrides.get(provider)
        if runtime_override:
            return runtime_override

        credential = self.get(provider)
        if credential is not None:
            if credential.get("type") == "api_key":
                return resolve_config_value(cast(ApiKeyCredential, credential)["key"])

            if credential.get("type") == "oauth":
                oauth_provider = get_oauth_provider(provider)
                if oauth_provider is None:
                    return None

                oauth_credential = cast(OAuthCredential, credential)
                expires = oauth_credential.get("expires")
                if isinstance(expires, (int, float)) and time.time() * 1000 >= expires:
                    refreshed = self._refresh_oauth_credential(provider, oauth_provider)
                    if refreshed is None:
                        return None
                    return oauth_provider.get_api_key(refreshed)

                return oauth_provider.get_api_key(oauth_credential)

        env_var = ENV_API_KEY_BY_PROVIDER.get(provider)
        if env_var:
            return os.getenv(env_var)
        return None

    def _refresh_oauth_credential(self, provider: str, oauth_provider: OAuthProvider) -> OAuthCredential | None:
        with _locked_json_file(self.path) as handle:
            handle.seek(0)
            current = cast(dict[str, Credential], json.loads(handle.read().strip() or "{}"))
            credential = current.get(provider)
            if not credential or credential.get("type") != "oauth":
                return None

            oauth_credential = cast(OAuthCredential, credential)
            expires = oauth_credential.get("expires")
            if isinstance(expires, (int, float)) and time.time() * 1000 < expires:
                self.data = current
                return oauth_credential

            refreshed = oauth_provider.refresh_token(oauth_credential)
            next_credential: OAuthCredential = {"type": "oauth", **refreshed}
            current[provider] = next_credential
            handle.seek(0)
            handle.truncate(0)
            json.dump(current, handle, indent=2)
            handle.write("\n")
            self.data = current
            return next_credential


@dataclass(frozen=True, slots=True)
class OpenAICodexOAuthProvider:
    id: str = OPENAI_CODEX_PROVIDER
    name: str = "ChatGPT Plus/Pro (Codex Subscription)"

    def login(
        self,
        *,
        open_browser: bool = True,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
        timeout_seconds: float = 60.0,
        originator: str = "dspy_lm_auth",
    ) -> OAuthCredential:
        return login_openai_codex(
            open_browser=open_browser,
            input_fn=input_fn,
            print_fn=print_fn,
            timeout_seconds=timeout_seconds,
            originator=originator,
        )

    def refresh_token(self, credentials: OAuthCredential) -> OAuthCredential:
        return refresh_openai_codex_token(credentials["refresh"])

    def get_api_key(self, credentials: OAuthCredential) -> str:
        return credentials["access"]


def register_oauth_provider(provider: OAuthProvider) -> None:
    _OAUTH_PROVIDERS[provider.id] = provider


def get_oauth_provider(provider: str) -> OAuthProvider | None:
    provider = normalize_provider_id(provider)
    oauth_provider = _OAUTH_PROVIDERS.get(provider)
    if oauth_provider is None and is_openai_codex_provider(provider):
        return _OAUTH_PROVIDERS.get(OPENAI_CODEX_PROVIDER)
    return oauth_provider


def get_default_auth_storage(path: str | os.PathLike[str] | None = None) -> AuthStorage:
    global _DEFAULT_AUTH_STORAGE

    if path is not None:
        _DEFAULT_AUTH_STORAGE = AuthStorage(path)
    if _DEFAULT_AUTH_STORAGE is None:
        _DEFAULT_AUTH_STORAGE = AuthStorage()
    return _DEFAULT_AUTH_STORAGE


def set_default_auth_storage(storage: AuthStorage) -> AuthStorage:
    global _DEFAULT_AUTH_STORAGE
    _DEFAULT_AUTH_STORAGE = storage
    return storage


def getauthtoken(provider: str = OPENAI_CODEX_PROVIDER, *, auth_storage: AuthStorage | None = None) -> str:
    storage = auth_storage or get_default_auth_storage()
    token = storage.get_api_key(provider)
    if not token:
        raise ValueError(
            f"No credential configured for {normalize_provider_id(provider)}. "
            "Use dspy_lm_auth.login(...) or pass api_key explicitly."
        )
    return token


def login(provider: str = OPENAI_CODEX_PROVIDER, *, auth_storage: AuthStorage | None = None, **kwargs) -> OAuthCredential:
    storage = auth_storage or get_default_auth_storage()
    return storage.login(provider, **kwargs)


def logout(provider: str = OPENAI_CODEX_PROVIDER, *, auth_storage: AuthStorage | None = None) -> None:
    storage = auth_storage or get_default_auth_storage()
    storage.logout(provider)


def _base64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token")

    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload + padding)
    return cast(dict[str, Any], json.loads(decoded.decode("utf-8")))


def extract_chatgpt_account_id(token: str) -> str:
    payload = _decode_jwt_payload(token)
    account_id = payload.get(OPENAI_CODEX_JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id:
        raise ValueError("Failed to extract accountId from token")
    return account_id


def generate_pkce_pair() -> tuple[str, str]:
    verifier = _base64url_encode(secrets.token_bytes(32))
    challenge = _base64url_encode(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def build_openai_codex_authorization_url(*, state: str, challenge: str, originator: str = "dspy_lm_auth") -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
            "scope": OPENAI_CODEX_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": originator,
        }
    )
    return f"{OPENAI_CODEX_AUTHORIZE_URL}?{query}"


def parse_authorization_input(raw: str) -> tuple[str | None, str | None]:
    value = raw.strip()
    if not value:
        return None, None

    try:
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            query = parse_qs(parsed.query)
            code = query.get("code", [None])[0]
            state = query.get("state", [None])[0]
            return cast(str | None, code), cast(str | None, state)
    except Exception:
        pass

    if "#" in value:
        code, state = value.split("#", 1)
        return code or None, state or None

    if "code=" in value:
        query = parse_qs(value)
        code = query.get("code", [None])[0]
        state = query.get("state", [None])[0]
        return cast(str | None, code), cast(str | None, state)

    return value, None


class _OpenAICallbackHandler(BaseHTTPRequestHandler):
    server_version = "DSPyLMAuth/1.0"

    def do_GET(self) -> None:
        assert isinstance(self.server, _OAuthCallbackServer)
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        params = parse_qs(parsed.query)
        state = params.get("state", [None])[0]
        code = params.get("code", [None])[0]
        if state != self.server.expected_state:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"State mismatch")
            return
        if not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing authorization code")
            return

        self.server.authorization_code = code
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<!doctype html><html><body><p>Authentication successful. Return to your terminal.</p></body></html>"
        )

    def log_message(self, format: str, *args) -> None:
        return


class _OAuthCallbackServer(HTTPServer):
    authorization_code: str | None
    expected_state: str


@contextmanager
def _start_local_callback_server(expected_state: str):
    server = _OAuthCallbackServer(("127.0.0.1", 1455), _OpenAICallbackHandler)
    server.timeout = 0.2
    server.authorization_code = None
    server.expected_state = expected_state

    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


def exchange_openai_codex_code(code: str, verifier: str, *, redirect_uri: str = OPENAI_CODEX_REDIRECT_URI) -> OAuthCredential:
    response = requests.post(
        OPENAI_CODEX_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = cast(dict[str, Any], response.json())

    access = payload.get("access_token")
    refresh = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access, str) or not isinstance(refresh, str) or not isinstance(expires_in, (int, float)):
        raise RuntimeError("Token exchange response is missing required fields")

    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires": int(time.time() * 1000 + expires_in * 1000),
        "accountId": extract_chatgpt_account_id(access),
    }


def refresh_openai_codex_token(refresh_token: str) -> OAuthCredential:
    response = requests.post(
        OPENAI_CODEX_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OPENAI_CODEX_CLIENT_ID,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = cast(dict[str, Any], response.json())

    access = payload.get("access_token")
    refresh = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access, str) or not isinstance(refresh, str) or not isinstance(expires_in, (int, float)):
        raise RuntimeError("Token refresh response is missing required fields")

    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires": int(time.time() * 1000 + expires_in * 1000),
        "accountId": extract_chatgpt_account_id(access),
    }


def login_openai_codex(
    *,
    open_browser: bool = True,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    timeout_seconds: float = 60.0,
    originator: str = "dspy_lm_auth",
) -> OAuthCredential:
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_hex(16)
    url = build_openai_codex_authorization_url(state=state, challenge=challenge, originator=originator)

    print_fn(f"Open this URL to authenticate with ChatGPT Codex:\n{url}")
    if open_browser:
        webbrowser.open(url)

    with _start_local_callback_server(state) as server:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if server.authorization_code:
                return exchange_openai_codex_code(server.authorization_code, verifier)
            time.sleep(0.1)

    raw_input = input_fn("Paste the authorization code or full redirect URL: ")
    code, returned_state = parse_authorization_input(raw_input)
    if returned_state and returned_state != state:
        raise ValueError("State mismatch")
    if not code:
        raise RuntimeError("Missing authorization code")
    return exchange_openai_codex_code(code, verifier)


register_oauth_provider(OpenAICodexOAuthProvider())


__all__ = [
    "DEFAULT_PI_AUTH_PATH",
    "ENV_API_KEY_BY_PROVIDER",
    "OPENAI_CODEX_PROVIDER",
    "ApiKeyCredential",
    "AuthStorage",
    "Credential",
    "OAuthCredential",
    "OpenAICodexOAuthProvider",
    "build_openai_codex_authorization_url",
    "clear_command_cache",
    "exchange_openai_codex_code",
    "extract_chatgpt_account_id",
    "generate_pkce_pair",
    "get_default_auth_storage",
    "get_oauth_provider",
    "getauthtoken",
    "is_openai_codex_provider",
    "login",
    "login_openai_codex",
    "logout",
    "normalize_provider_id",
    "parse_authorization_input",
    "refresh_openai_codex_token",
    "register_oauth_provider",
    "resolve_config_value",
    "set_default_auth_storage",
]
