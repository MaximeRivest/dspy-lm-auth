# dspy-lm-auth

[![CI](https://github.com/MaximeRivest/dspy-lm-auth/actions/workflows/ci.yml/badge.svg)](https://github.com/MaximeRivest/dspy-lm-auth/actions/workflows/ci.yml) [![PyPI version](https://img.shields.io/pypi/v/dspy-lm-auth.svg)](https://pypi.org/project/dspy-lm-auth/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MaximeRivest/dspy-lm-auth/blob/main/LICENSE)

Pi-style LM authentication helpers for DSPy.

`dspy-lm-auth` lets DSPy reuse Pi credentials from `~/.pi/agent/auth.json`, including ChatGPT Codex subscription auth.

The nicest way to use it is not as an isolated auth helper, but as the missing piece in a very practical DSPy workflow:

- run a **small model locally** for the bulk of your cheap inference
- use your **existing ChatGPT subscription** as the stronger GEPA reflection model

If you already pay for ChatGPT Plus or Pro, this gives you a pleasant way to explore DSPy without setting up a separate metered OpenAI API workflow just to optimize prompts.

> Local compute is not literally free — your machine still does work — but it is a very good **no-extra-API-bill** workflow for experimentation.

## Current support

- OpenAI Codex / ChatGPT Plus or Pro subscription

## What this guide will show

We will build a tiny French→English translator in DSPy.

The pattern is simple:

1. run `qwen3.5:0.8b` locally with Ollama
2. use that local model as the **student model**
3. use `codex/gpt-5.4` through `dspy-lm-auth` as the **reflection model**
4. let GEPA improve the student program

This README intentionally sticks to `JSONAdapter()`.

That is not because other adapters are uninteresting — quite the opposite. It is because a good tutorial should hold one thing steady at a time. If you want to compare `JSONAdapter`, `XMLAdapter`, and custom templated adapters, that is best treated as a separate benchmark project.

## Install

```bash
uv pip install dspy-lm-auth
```

Or with `pip`:

```bash
pip install dspy-lm-auth
```

## One-time login

If you already use Pi and your credentials are present in `~/.pi/agent/auth.json`, you can skip this step.

Otherwise:

```python
import dspy_lm_auth

dspy_lm_auth.login("codex")
```

That starts the OAuth flow and stores the resulting credentials in Pi's auth file.

## Tutorial: local DSPy + subscription-powered GEPA

### Step 1: run a small local model with Ollama

On Linux, install Ollama with:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

If the server is not already running, start it:

```bash
ollama serve
```

Now pull the model:

```bash
ollama pull qwen3.5:0.8b
```

Sanity check:

```bash
ollama run qwen3.5:0.8b --think=false "Translate French to English and return only the translation: merci beaucoup"
```

### Why `ollama_chat/...` and `think=False`?

For this model family, the cleanest DSPy setup is the native Ollama LiteLLM route:

- use `ollama_chat/qwen3.5:0.8b`
- set `think=False`

That gives a cleaner programming experience than relying on the OpenAI-compatible Ollama endpoint for this particular model.

### Step 2: configure the two models in DSPy

```python
import dspy
import dspy_lm_auth

# Patch dspy.LM so `codex/...` works.
dspy_lm_auth.install()

# Cheap local student model.
student_lm = dspy.LM(
    "ollama_chat/qwen3.5:0.8b",
    api_base="http://127.0.0.1:11434",
    api_key="ollama",  # dummy value; LiteLLM expects one
    model_type="chat",
    think=False,
    temperature=0,
    max_tokens=200,
)

# Stronger reflection model used by GEPA to improve the prompt.
reflection_lm = dspy.LM("codex/gpt-5.4")

# All program inference goes through the local student model.
dspy.configure(lm=student_lm, adapter=dspy.JSONAdapter())
```

At this point you have the whole idea in place:

- **student model** = local, cheap, yours
- **reflection model** = stronger, subscription-backed, already paid for

### Step 3: write a tiny DSPy program

```python
import dspy


class TranslateFrenchToEnglish(dspy.Signature):
    """Translate the French input into short, natural English."""

    french: str = dspy.InputField(desc="French sentence")
    english: str = dspy.OutputField(desc="Natural English translation")


translator = dspy.Predict(TranslateFrenchToEnglish)

print(translator(french="merci beaucoup").english)
print(translator(french="où est la gare ?").english)
```

A tiny local model is often good enough to be useful, but not always good enough to be reliably right in the way you want.

That is where GEPA comes in.

### Step 4: create a tiny training set

```python
pairs = [
    ("bonjour", "hello"),
    ("merci beaucoup", "thank you very much"),
    ("où est la gare ?", "where is the train station?"),
    ("je suis fatigué", "I am tired"),
    ("il fait très chaud aujourd'hui", "it is very hot today"),
    ("je ne comprends pas", "I do not understand"),
    ("pouvez-vous m'aider ?", "can you help me?"),
    ("j'aime apprendre le français", "I like learning French"),
    ("nous arrivons demain matin", "we are arriving tomorrow morning"),
    ("combien ça coûte ?", "how much does it cost?"),
]

examples = [
    dspy.Example(french=fr, english=en).with_inputs("french")
    for fr, en in pairs
]

trainset = examples[:8]
valset = examples[8:]
```

This is intentionally tiny. The point of the tutorial is the workflow, not leaderboard chasing.

### Step 5: define what “good” means

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    guess = pred.english.strip()
    target = gold.english.strip()

    exact = guess.lower() == target.lower()
    score = 1.0 if exact else 0.0

    if exact:
        feedback = (
            "Exact match. Keep translations short, natural, and direct. "
            "Do not add explanations."
        )
    else:
        feedback = (
            f"Expected {target!r} but got {guess!r}. "
            "Prefer direct, idiomatic English. Preserve tense, pronouns, and politeness. "
            "Do not explain the translation or add extra words."
        )

    return dspy.Prediction(score=score, feedback=feedback)
```

The metric is deliberately simple:

- score exact matches as `1.0`
- score everything else as `0.0`
- give GEPA useful textual feedback so it can rewrite the prompt

### Step 6: run GEPA

```python
gepa = dspy.GEPA(
    metric=metric,
    reflection_lm=reflection_lm,
    auto="light",
)

optimized = gepa.compile(translator, trainset=trainset, valset=valset)
```

This is the moment the package earns its keep.

The student model stays local. GEPA uses the stronger subscription model to think about failures and improve the program. That is the whole value proposition in one place.

### Step 7: inspect the optimized program

```python
print("Optimized instruction:\n")
print(optimized.signature.instructions)
print()

print(optimized(french="je ne comprends pas").english)
print(optimized(french="combien ça coûte ?").english)
```

A good way to read the result is:

- the local model is still the one doing inference
- the stronger subscription model helped shape a better instruction
- you did not need a separate metered API setup for the optimizer model

## A complete copy-paste script

If you prefer one coherent script rather than step-by-step fragments, here is the full version:

```python
import dspy
import dspy_lm_auth


dspy_lm_auth.install()

student_lm = dspy.LM(
    "ollama_chat/qwen3.5:0.8b",
    api_base="http://127.0.0.1:11434",
    api_key="ollama",
    model_type="chat",
    think=False,
    temperature=0,
    max_tokens=200,
)

reflection_lm = dspy.LM("codex/gpt-5.4")

dspy.configure(lm=student_lm, adapter=dspy.JSONAdapter())


class TranslateFrenchToEnglish(dspy.Signature):
    """Translate the French input into short, natural English."""

    french: str = dspy.InputField(desc="French sentence")
    english: str = dspy.OutputField(desc="Natural English translation")


translator = dspy.Predict(TranslateFrenchToEnglish)

pairs = [
    ("bonjour", "hello"),
    ("merci beaucoup", "thank you very much"),
    ("où est la gare ?", "where is the train station?"),
    ("je suis fatigué", "I am tired"),
    ("il fait très chaud aujourd'hui", "it is very hot today"),
    ("je ne comprends pas", "I do not understand"),
    ("pouvez-vous m'aider ?", "can you help me?"),
    ("j'aime apprendre le français", "I like learning French"),
    ("nous arrivons demain matin", "we are arriving tomorrow morning"),
    ("combien ça coûte ?", "how much does it cost?"),
]

examples = [
    dspy.Example(french=fr, english=en).with_inputs("french")
    for fr, en in pairs
]

trainset = examples[:8]
valset = examples[8:]

print("Before optimization:")
print(translator(french="où est la gare ?").english)
print(translator(french="je ne comprends pas").english)
print()


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    guess = pred.english.strip()
    target = gold.english.strip()

    exact = guess.lower() == target.lower()
    score = 1.0 if exact else 0.0

    if exact:
        feedback = (
            "Exact match. Keep translations short, natural, and direct. "
            "Do not add explanations."
        )
    else:
        feedback = (
            f"Expected {target!r} but got {guess!r}. "
            "Prefer direct, idiomatic English. Preserve tense, pronouns, and politeness. "
            "Do not explain the translation or add extra words."
        )

    return dspy.Prediction(score=score, feedback=feedback)


gepa = dspy.GEPA(
    metric=metric,
    reflection_lm=reflection_lm,
    auto="light",
)

optimized = gepa.compile(translator, trainset=trainset, valset=valset)

print("Optimized instruction:\n")
print(optimized.signature.instructions)
print()

print("After optimization:")
print(optimized(french="où est la gare ?").english)
print(optimized(french="je ne comprends pas").english)
print(optimized(french="combien ça coûte ?").english)
```

## When you outgrow the laptop: the same idea on a GPU box

The laptop workflow is the easiest place to start.

When you want more speed or more context, keep the exact same mental model and swap only the student model:

- laptop: `Ollama + qwen3.5:0.8b`
- GPU box: `vLLM + Qwen/Qwen3.5-0.8B`

### Minimal GPU setup

SSH into the GPU box:

```bash
ssh YOUR_GPU_BOX
```

Install `uv` and `vllm`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
uv venv ~/.venvs/vllm-qwen35-08b --python 3.12
uv pip install --python ~/.venvs/vllm-qwen35-08b/bin/python vllm
```

Launch the model:

```bash
CUDA_VISIBLE_DEVICES=0 ~/.venvs/vllm-qwen35-08b/bin/vllm serve Qwen/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name local-model \
  --dtype float16 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 2048
```

Then swap the student model definition in DSPy to:

```python
student_lm = dspy.LM(
    "openai/local-model",
    api_base="http://YOUR_GPU_BOX:8000/v1",
    api_key="",
    model_type="chat",
)
```

Everything else in the GEPA workflow stays the same.

## If you only want the auth piece

You can also use `dspy-lm-auth` without the local-model tutorial.

```python
import dspy
import dspy_lm_auth


dspy_lm_auth.install()

lm = dspy.LM("codex/gpt-5.4")
dspy.configure(lm=lm)

print(lm("hello")[0]["text"])
```

Or keep the original provider and select the auth route explicitly:

```python
import dspy_lm_auth

lm = dspy_lm_auth.LM("openai/gpt-5.4", auth_provider="codex")
print(lm("hello")[0]["text"])
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

## License

MIT
