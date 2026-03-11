# dspy-lm-auth

[![CI](https://github.com/MaximeRivest/dspy-lm-auth/actions/workflows/ci.yml/badge.svg)](https://github.com/MaximeRivest/dspy-lm-auth/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/dspy-lm-auth.svg)](https://pypi.org/project/dspy-lm-auth/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MaximeRivest/dspy-lm-auth/blob/main/LICENSE)

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

```py
%pip install dspy-lm-auth
```

```output:exec-1773240219480-pahro
Running: uv pip install dspy-lm-auth
--------------------------------------------------
[2mAudited [1m1 package[0m [2min 10ms[0m
--------------------------------------------------
Note: Restart kernel to use newly installed packages.
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

```output:exec-1773240228078-aqrym
Hello! How can I help?
```

You can also keep the original model string and apply the Codex auth route explicitly:

```python
import dspy_lm_auth

lm = dspy_lm_auth.LM("openai/gpt-5.4", auth_provider="codex")
print(lm("hello")[0]["text"])
```

```output:exec-1773240238429-rx9qs
Hello! How can I help?
```

## A very cheap laptop stack: uv + llama-cpp + baguettotron + dspy

People often ask for a free stack to try DSPy locally on a laptop.

A nice minimal stack is:

- `uv` for env management
- `llama-cpp-python[server]` for OpenAI-compatible local serving
- [Baguettotron-GGUF](https://huggingface.co/P1eIAS/Baguettotron-GGUF) as the local model
- DSPy for the program
- `dspy-lm-auth` for the Codex reflection model when you want a stronger optimizer / teacher model

Start a local OpenAI-compatible server with `llama-cpp`:

```bash
uv venv
source .venv/bin/activate
uv pip install "llama-cpp-python[server]" huggingface-hub dspy dspy-lm-auth
```

```sh
uv run python -m llama_cpp.server \
  --host 127.0.0.1 \
  --port 8000 \
  --hf_model_repo_id P1eIAS/Baguettotron-GGUF \
  --model Baguettotron-BF16.gguf &&
```

If that GGUF is too large for your machine, swap in a smaller GGUF from the same repo.

Then connect to it from DSPy:

```python
import dspy

local_lm = dspy.LM(
    "openai/local-model",
    api_base="http://localhost:8000/v1",
    api_key="",
    model_type="chat",
)

dspy.configure(lm=local_lm, adapter=dspy.JSONAdapter())
```

```output:exec-1773240681407-w56n4
```

## A faster self-hosted GPU stack: uv + vLLM + Qwen 3.5 + DSPy

If you have a desktop, gaming PC, or another Linux box with an NVIDIA GPU, this is the easiest next step up from the laptop `llama-cpp` setup.

The idea is:

- run a small model yourself with `vLLM`
- point DSPy at that OpenAI-compatible endpoint
- keep using your Codex subscription as the stronger GEPA reflection model

### 1. SSH into the GPU box

```bash
ssh YOUR_GPU_BOX
```

### 2. Install `uv` and create a clean vLLM environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
uv venv ~/.venvs/vllm-qwen35-08b --python 3.12
uv pip install --python ~/.venvs/vllm-qwen35-08b/bin/python vllm
```

### 3. Launch Qwen 3.5 on the GPU with vLLM

This starts an OpenAI-compatible server on port `8000` and exposes the model under the name `local-model`.

```bash
CUDA_VISIBLE_DEVICES=0 ~/.venvs/vllm-qwen35-08b/bin/vllm serve Qwen/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name local-model \
  --dtype float16 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 2048
```

Those flags are deliberately conservative, so they are a good starting point on a busy 24 GB card like a 3090.

If the GPU box is mostly free, you can usually increase one or both of:

- `--gpu-memory-utilization` to something like `0.5` or `0.7`
- `--max-model-len` to `4096`

### 4. If the server is on another machine, open the port

If you use `ufw`:

```bash
sudo ufw allow 8000/tcp
```

### 5. Test that the server is alive

From your laptop:

```bash
curl http://YOUR_GPU_BOX:8000/v1/models
```

You should see `local-model` in the response.

### 6. Use it from DSPy

```python
import dspy

student_lm = dspy.LM(
    "openai/local-model",
    api_base="http://YOUR_GPU_BOX:8000/v1",
    api_key="",
    model_type="chat",
)

dspy.configure(lm=student_lm, adapter=dspy.JSONAdapter())
```

### Optional: run it as a background service with systemd

If you do not want to keep a terminal open on the GPU box, you can make it a user service:

```bash
mkdir -p ~/services/vllm-qwen35-08b ~/.config/systemd/user

cat > ~/services/vllm-qwen35-08b/start.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
exec ~/.venvs/vllm-qwen35-08b/bin/vllm serve Qwen/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name local-model \
  --dtype float16 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 2048
SH
chmod +x ~/services/vllm-qwen35-08b/start.sh

cat > ~/.config/systemd/user/vllm-qwen35-08b.service <<'UNIT'
[Unit]
Description=vLLM OpenAI server for Qwen3.5-0.8B
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/services/vllm-qwen35-08b
ExecStart=%h/services/vllm-qwen35-08b/start.sh
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now vllm-qwen35-08b.service
systemctl --user status vllm-qwen35-08b.service --no-pager
```

That gives you a simple self-hosted setup that a non-expert can copy, paste, and run.

## Tiny GEPA demo: optimize a French→English translator

This is a small self-contained demo that:

- uses a self-hosted `vLLM` server running Qwen 3.5 as the **student model**
- uses your ChatGPT Codex subscription via `dspy-lm-auth` as the **GEPA reflection model**
- optimizes a tiny translator on just **10 French→English examples**



```python
%pip install dspy_template_adapter
```

```output:exec-1773243237064-pojcy
Running: uv pip install dspy_template_adapter
--------------------------------------------------
[2mResolved [1m73 packages[0m [2min 200ms[0m
[2mPrepared [1m1 package[0m [2min 30ms[0m
[2mInstalled [1m1 package[0m [2min 2ms[0m
 [32m+[0m [1mdspy-template-adapter[0;2m==0.2.2[0m
--------------------------------------------------
Note: Restart kernel to use newly installed packages.
```


```python
import dspy
import dspy_lm_auth
from dspy_template_adapter import TemplateAdapter, Predict


# Patch dspy.LM so `codex/...` works.
dspy_lm_auth.install()

# Student model: self-hosted vLLM server on your GPU box.
student_lm = dspy.LM(
    "openai/local-model",
    api_base="http://YOUR_GPU_BOX:8000/v1",  # e.g. http://192.168.2.24:8000/v1
    api_key="",
    model_type="chat",
)

# Reflection model: stronger model used by GEPA to improve the prompt.
reflection_lm = dspy.LM("codex/gpt-5.4")

# DSPy program inference uses the cheap local model.
dspy.configure(lm=student_lm, adapter=dspy.JSONAdapter())


class TranslateFrenchToEnglish(dspy.Signature):
    """Translate the French input into short, natural English."""

    french: str = dspy.InputField(desc="French sentence")
    english: str = dspy.OutputField(desc="Natural English translation")


adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "{instruction}"},
        {"role": "user", "content": "{inputs()}"},
    ],
    parse_mode="full_text",
)

translator = Predict(TranslateFrenchToEnglish, adapter=adapter)

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

```output:exec-1773245901342-lh7nu
```


```python
adapter.format(signature=translator.signature, demos=[], inputs={"french": "bonjour et merci"})
```

```output:exec-1773246381356-empoc
Out[39]: 
[{'role': 'system',
  'content': 'Translate the French input into short, natural English.'},
 {'role': 'user', 'content': 'french: bonjour et merci'}]
```

`TemplateAdapter.format(...)` expects a **DSPy signature**, not a `Predict` module. In this example the correct call is `translator.signature`, not `translator`. The template also needs `{instruction}` and `{inputs(...)}` — the earlier `{instruction}}` and `{input(...)}` forms are invalid.


```python
# Requires the self-hosted vLLM server above to be running.
translator(french="ceci est vert")
```

```output:exec-1773245948646-mybkn
Out[36]: 
Prediction(
    english="Okay, that's green."
)
```

This call should work once the self-hosted `vLLM` server is running.


```py

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
    track_stats=True,
)

optimized = gepa.compile(translator, trainset=trainset, valset=valset)

```

```output:exec-1773245960807-ba341
2026/03/11 12:19:20 INFO dspy.teleprompt.gepa.gepa: Running GEPA for approx 388 metric calls of the program. This amounts to 38.80 full evals on the train+val set.
2026/03/11 12:19:20 INFO dspy.teleprompt.gepa.gepa: Using 2 examples for tracking Pareto scores. You can consider using a smaller sample of the valset to allow GEPA to explore more diverse solutions within the same budget. GEPA requires you to provide the smallest valset that is just large enough to match your downstream task distribution, while providing as large trainset as possible.
GEPA Optimization:   0% 0/388 [00:00<?, ?rollouts/s]2026/03/11 12:19:20 INFO dspy.evaluate.evaluate: Average Metric: 0.0 / 2 (0.0%)
2026/03/11 12:19:20 INFO dspy.teleprompt.gepa.gepa: Iteration 0: Base program full valset score: 0.0 over 2 / 2 examples
GEPA Optimization:   1% 2/388 [00:00<00:20, 18.54rollouts/s]2026/03/11 12:19:20 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Selected program 0 score: 0.0
Average Metric: 0.00 / 3 (0.0%): 100% 3/3 [00:00<00:00, 27.62it/s]
2026/03/11 12:19:21 INFO dspy.evaluate.evaluate: Average Metric: 0.0 / 3 (0.0%)
/home/maxime/Projects/dspy-lm-auth/.venv/lib/python3.13/site-packages/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:
  PydanticSerializationUnexpectedValue(Expected `ResponseAPIUsage` - serialized value may not be as expected [field_name='usage', input_value={'completion_tokens': 211..., 'video_tokens': None}}, input_type=dict])
  return self.__pydantic_serializer__.to_python(
2026/03/11 12:19:25 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Proposed new text for self: Translate the provided French text into short, natural English.

Output rules:
- Return only the English translation.
- Do not add explanations, notes, analysis, or extra text.
- Do not restate the French.
- Keep the translation concise and idiomatic.

.
.
[TRUNCATED]
.
.

2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: Val aggregate for new program: 0.5
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: Individual valset scores for new program: {0: 0.0, 1: 1.0}
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: New valset pareto front scores: {0: 0.0, 1: 1.0}
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: Valset pareto front aggregate score: 0.5
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: Updated valset pareto front programs: {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1: {4, 9, 10, 11, 13, 15}}
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: Best valset aggregate score so far: 0.5
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: Best program as per aggregate score on valset: 4
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: Best score on valset: 0.5
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: Linear pareto front program index: 4
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 70: New program candidate index: 15
GEPA Optimization:  86% 332/388 [04:39<00:48,  1.14rollouts/s]2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 71: No merge candidates found
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 71: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 29.59it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 71: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 71: Reflective mutation did not propose a new candidate
GEPA Optimization:  86% 335/388 [04:39<00:40,  1.32rollouts/s]2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 72: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 27.48it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 72: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 72: Reflective mutation did not propose a new candidate
GEPA Optimization:  87% 338/388 [04:39<00:32,  1.56rollouts/s]2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 73: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 756.37it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 73: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 73: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 74: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 605.01it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 74: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 74: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 75: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 681.52it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 75: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 75: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 76: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 580.10it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 76: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 76: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 77: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 825.27it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 77: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 77: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 78: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 649.24it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 78: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 78: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 79: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 692.97it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 79: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 79: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 80: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 651.73it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 80: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 80: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 81: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 696.81it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 81: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 81: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 82: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 809.61it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 82: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 82: Reflective mutation did not propose a new candidate
GEPA Optimization:  95% 368/388 [04:40<00:03,  5.11rollouts/s]2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 83: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 260.18it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 83: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 83: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 84: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 304.98it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 84: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 84: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 85: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 636.88it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 85: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 85: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 86: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 335.72it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 86: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 86: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 87: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 450.45it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 87: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 87: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 88: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 710.50it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 88: All subsample scores perfect. Skipping.
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 88: Reflective mutation did not propose a new candidate
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 89: Selected program 15 score: 0.5
Average Metric: 3.00 / 3 (100.0%): 100% 3/3 [00:00<00:00, 772.62it/s]
2026/03/11 12:24:00 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 89: All subsample scores perfect. Skipping.
Optimized instruction:

Translate the provided French text into English.

Input:
- You will receive one field containing French text, typically labeled `french`.

Output:
- Return only the English translation as plain text.
- Do not include labels, JSON, metadata, quotes, explanations, comments, annotations, reasoning, or alternatives.

Requirements:
- Translate only the French input and nothing else.
- Preserve the original meaning exactly.
- Preserve pronouns exactly.
- Preserve tense, mood, register, and politeness level.
- Keep the English short, natural, direct, and idiomatic.
- Prefer direct idiomatic equivalents over overly literal phrasing.
- Do not add words not supported by the French, except where needed for natural idiomatic English.
- Do not use contractions. For example:
  - `je suis fatigué` → `I am tired`
  - `je ne comprends pas` → `I do not understand`
- Preserve punctuation from the French input.
- Do not add terminal punctuation if it is not present in the French.
- Match capitalization naturally in English; for very short inputs such as greetings or simple questions, prefer lowercase unless capitalization is clearly required.
- For very short greetings, prefer no final period.

Specific preferences:
- For common travel/location usage, translate `gare` as `train station`, not `station`.
- Translate `j'aime` in neutral contexts as `I like`, not `I love`, unless the French clearly conveys stronger emphasis.

Examples:
- `pouvez-vous m'aider ?` → `can you help me?`
- `bonjour` → `hello`
- `merci beaucoup` → `thank you very much`
- `j'aime apprendre le français` → `I like learning French`
- `où est la gare ?` → `where is the train station?`
- `il fait très chaud aujourd'hui` → `it is very hot today`
- `je suis fatigué` → `I am tired`
- `je ne comprends pas` → `I do not understand`

Final rule:
- Return only the translated English text and nothing else.

2026/03/11 12:24:00 INFO dspy.teleprompt.gepa.gepa: Iteration 89: Reflective mutation did not propose a new candidate
GEPA Optimization:  99% 386/388 [04:40<00:01,  1.38rollouts/s]
I do not understand
how much does it cost?
```

 
```python
print("Optimized instruction:\n")
print(optimized.signature.instructions)
print()

print(optimized(french="je ne comprends pas").english)
print(optimized(french="combien ça coûte ?").english)
```

```output:exec-1773246346946-33f3h
Optimized instruction:

Translate the provided French text into English.

Input:
- You will receive one field containing French text, typically labeled `french`.

Output:
- Return only the English translation as plain text.
- Do not include labels, JSON, metadata, quotes, explanations, comments, annotations, reasoning, or alternatives.

Requirements:
- Translate only the French input and nothing else.
- Preserve the original meaning exactly.
- Preserve pronouns exactly.
- Preserve tense, mood, register, and politeness level.
- Keep the English short, natural, direct, and idiomatic.
- Prefer direct idiomatic equivalents over overly literal phrasing.
- Do not add words not supported by the French, except where needed for natural idiomatic English.
- Do not use contractions. For example:
  - `je suis fatigué` → `I am tired`
  - `je ne comprends pas` → `I do not understand`
- Preserve punctuation from the French input.
- Do not add terminal punctuation if it is not present in the French.
- Match capitalization naturally in English; for very short inputs such as greetings or simple questions, prefer lowercase unless capitalization is clearly required.
- For very short greetings, prefer no final period.

Specific preferences:
- For common travel/location usage, translate `gare` as `train station`, not `station`.
- Translate `j'aime` in neutral contexts as `I like`, not `I love`, unless the French clearly conveys stronger emphasis.

Examples:
- `pouvez-vous m'aider ?` → `can you help me?`
- `bonjour` → `hello`
- `merci beaucoup` → `thank you very much`
- `j'aime apprendre le français` → `I like learning French`
- `où est la gare ?` → `where is the train station?`
- `il fait très chaud aujourd'hui` → `it is very hot today`
- `je suis fatigué` → `I am tired`
- `je ne comprends pas` → `I do not understand`

Final rule:
- Return only the translated English text and nothing else.

I do not understand
how much does it cost?
```



Notes:

- `student_lm` is the cheap local model you serve yourself.
- `reflection_lm` is the stronger model GEPA uses to improve the prompt.
- `auto="light"` keeps the demo small enough for a laptop workflow.
- if your local server requires a specific model name, replace `openai/local-model` with the one exposed by your `llama-cpp` server.

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

## License

MIT
