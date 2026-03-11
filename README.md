# dspy-lm-auth

Pi-style LM authentication helpers for DSPy.

`dspy-lm-auth` makes it easy to reuse Pi-style credentials with `dspy.LM`, including ChatGPT Codex subscription auth.

## What it does

- reuses Pi credentials from `~/.pi/agent/auth.json`
- resolves provider config values from:
  - literal strings
  - environment variable names
  - `!shell command` lookups
- supports OAuth login and token refresh flows for subscription-backed providers
- patches `dspy.LM` so model aliases and alternate auth routes work out of the box

## Current support

- OpenAI Codex / ChatGPT Plus or Pro subscription

## Install

```bash
pip install dspy-lm-auth
```

Or with `uv`:

```bash
uv pip install dspy-lm-auth
```

## Quick start

```python
import dspy
import dspy_lm_auth

# Optional: patch dspy.LM in place.
dspy_lm_auth.install()

# Reuse Pi's ChatGPT Codex login from ~/.pi/agent/auth.json.
lm = dspy.LM("codex/gpt-5.4")
dspy.configure(lm=lm)

print(lm("hello")[0]["text"])
```

You can also keep the original model string and apply the Codex auth route explicitly:

```python
import dspy_lm_auth

lm = dspy_lm_auth.LM("openai/gpt-5.4", auth_provider="codex")
print(lm("hello")[0]["text"])
```

## Login

If you do not already have credentials stored in Pi's auth file:

```python
import dspy_lm_auth

# Starts the OAuth flow and writes credentials to ~/.pi/agent/auth.json.
dspy_lm_auth.login("codex")
```

## Credential resolution

API key credentials can be stored as:

- a literal value
- an environment variable name
- a shell lookup prefixed with `!`

Examples:

```json
{
  "some-provider": {
    "type": "api_key",
    "key": "OPENAI_API_KEY"
  }
}
```

```json
{
  "some-provider": {
    "type": "api_key",
    "key": "!op read op://Private/openai/api_key --no-newline"
  }
}
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

## Roadmap

The package is structured so more Pi-like providers can be added later, for example:

- Anthropic subscription auth
- GitHub Copilot
- Gemini CLI
- Antigravity
