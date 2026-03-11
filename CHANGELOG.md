# Changelog

## 0.1.3

- rewrite the README as a more polished end-to-end tutorial
- make the laptop path use Ollama with `qwen3.5:0.8b`
- emphasize the local-student plus subscription-reflection GEPA workflow
- keep `JSONAdapter()` as the main tutorial path for a simpler first experience

## 0.1.2

- add README instructions for running a small self-hosted GPU model with vLLM and Qwen 3.5
- update the GEPA example to point at a remote OpenAI-compatible GPU endpoint
- keep the regression test proving the Codex route does not use `OPENAI_API_KEY`

## 0.1.1

- add MIT license
- add README badges for CI, PyPI, and license
- add a small laptop-friendly DSPy stack section using uv + llama-cpp + Baguettotron
- add a self-contained GEPA demo using a local student model and Codex `gpt-5.4` as the reflection model

## 0.1.0

- initial public release
- Pi-style credential storage and resolution for DSPy
- ChatGPT Codex / OpenAI Codex subscription login and refresh helpers
- `dspy.LM` patching with `codex/...` model aliases and `auth_provider="codex"`
- ChatGPT-compatible Codex Responses routing for `gpt-5.4`
