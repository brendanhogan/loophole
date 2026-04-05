# Loophole — quick setup

## What you need

| Requirement | Notes |
|---------------|--------|
| **Python** | 3.12 or newer (`requires-python = ">=3.12"` in `pyproject.toml`; `.python-version` pins `3.12`) |
| **uv** | Recommended by upstream: installs locked deps from `uv.lock` (`uv sync`) |
| **Venice API key** | Runtime: set `VENICE_API_KEY` so the Venice SDK can call the API for all agents (create a key in [Venice API settings](https://venice.ai/settings/api)) |
| **Network** | For `uv sync` / package install and for API calls at run time |

## Install

```bash
cd /path/to/loophole   # e.g. /workspace/projects/loophole
uv sync
export VENICE_API_KEY="..."   # required; from https://venice.ai/settings/api
```

If `uv` is missing: `pip install uv` (or see [astral.sh/uv](https://github.com/astral-sh/uv)).

## Run

```bash
uv run python -m loophole.main
# or with a principles file:
uv run python -m loophole.main new --domain privacy -p examples/privacy_principles.txt
```

## Optional: tune behavior

Edit `config.yaml` (model name, temperatures, loop limits, `session_dir`).

## Pip-only note

Plain `pip install -e .` can fail because the repo includes a top-level `sessions/` directory that setuptools may treat as a second package. Prefer **`uv sync`** as documented in `README.md`.
