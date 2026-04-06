# Loophole — quick setup

## What you need

| Requirement | Notes |
|-------------|-------|
| **Python** | 3.12 or newer (`requires-python = ">=3.12"` in `pyproject.toml`; `.python-version` pins `3.12`) |
| **uv** | Recommended by upstream: installs locked deps from `uv.lock` (`uv sync`) |
| **API key** | `ANTHROPIC_API_KEY` for Claude; for Venice / other OpenAI-compatible APIs use `OPENAI_API_KEY` (or `VENICE_API_KEY` when calling `api.venice.ai`) |
| **Network** | For `uv sync` / package install and for API calls at run time |

## Install

```bash
cd /path/to/loophole   # e.g. /workspace/projects/loophole
uv sync
```

If `uv` is missing: `pip install uv` (or see [astral.sh/uv](https://github.com/astral-sh/uv)).

## Choose your provider

Loophole supports two LLM backends:

### Option 1: Anthropic (default)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python -m loophole.main
```

### Option 2: Venice AI (OpenAI-compatible HTTP API)

Venice (and most similar providers) only need a **`base_url`** and an **API key** for the OpenAI Chat Completions shape.

Default Venice endpoint: `https://api.venice.ai/api/v1`.

```bash
export OPENAI_API_KEY="..."   # or VENICE_API_KEY for api.venice.ai
export LOOPHOLE_PROVIDER="venice"
uv run python -m loophole.main
```

Optional: override the API root (same pattern as other OpenAI-compatible gateways):

```bash
export LOOPHOLE_BASE_URL="https://api.venice.ai/api/v1"
```

You can also set options in `config.yaml`:

```yaml
model:
  provider: "venice"  # or "anthropic"
  default: "minimax-m25"  # Venice model id
  # base_url: "https://api.venice.ai/api/v1"  # optional; this is the default for Venice
```

## Provider selection priority

1. `LOOPHOLE_PROVIDER` environment variable
2. `model.provider` in `config.yaml`
3. Default: `"anthropic"`

For the OpenAI-compatible base URL (Venice or otherwise):

1. `LOOPHOLE_BASE_URL` environment variable
2. `model.base_url` in `config.yaml`
3. If provider is `venice`: `https://api.venice.ai/api/v1`

## Default models

| Provider | Default model |
|----------|---------------|
| Anthropic | `claude-sonnet-4-20250514` |
| Venice | `llama-3.3-70b` |

Override with `model.default` in `config.yaml`.

## Run

```bash
uv run python -m loophole.main
# or with a principles file:
uv run python -m loophole.main new --domain privacy -p examples/privacy_principles.txt
```

## Optional: tune behavior

Edit `config.yaml` (provider, model name, `base_url`, temperatures, loop limits, `session_dir`).

## Pip-only note

Plain `pip install -e .` can fail because the repo includes a top-level `sessions/` directory that setuptools may treat as a second package. Prefer **`uv sync`** as documented in `README.md`.
