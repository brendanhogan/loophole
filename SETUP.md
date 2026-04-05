# Loophole — quick setup

## What you need

| Requirement | Notes |
|-------------|-------|
| **Python** | 3.12 or newer (`requires-python = ">=3.12"` in `pyproject.toml`; `.python-version` pins `3.12`) |
| **uv** | Recommended by upstream: installs locked deps from `uv.lock` (`uv sync`) |
| **API key** | Either `ANTHROPIC_API_KEY` (for Claude) or `VENICE_API_KEY` (for Venice) depending on which provider you use |
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

### Option 2: Venice AI

```bash
export VENICE_API_KEY="..."  # from https://venice.ai/settings/api
export LOOPHOLE_PROVIDER="venice"
uv run python -m loophole.main
```

You can also set the provider in `config.yaml`:

```yaml
model:
  provider: "venice"  # or "anthropic"
  default: "minimax-m25"  # venice model name
```

## Provider selection priority

1. `LOOPHOLE_PROVIDER` environment variable
2. `model.provider` in `config.yaml`
3. Default: `"anthropic"`

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

Edit `config.yaml` (provider, model name, temperatures, loop limits, `session_dir`).

## Pip-only note

Plain `pip install -e .` can fail because the repo includes a top-level `sessions/` directory that setuptools may treat as a second package. Prefer **`uv sync`** as documented in `README.md`.
