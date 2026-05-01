# Magos

LLM inference API proxy built on [mitmproxy](https://mitmproxy.org/). Translates between Anthropic and OpenAI endpoint shapes, applies [Headroom](https://github.com/headroom-ai/headroom) context compression, and (planned) exposes a unified MCP endpoint.

## Features

- **Bidirectional translation**: Anthropic Messages API <-> OpenAI Chat Completions, including streaming
- **Byte-exact passthrough**: forward Anthropic-to-Anthropic without re-shaping
- **Context compression** via `headroom-ai`
- **Token counting** endpoint
- **Observability**: structured logging (`structlog`) and OpenTelemetry tracing
- **Declarative config** via `pydantic-settings` (env vars prefixed `MAGOS_` or `.env`)

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

## Install

```bash
uv sync --extra cpu   # CPU torch
# or
uv sync --extra gpu   # GPU torch (CUDA 13.0)
```

## Run

```bash
uv run python -m magos
```

Configuration is read from environment variables (prefix `MAGOS_`) or a local `.env` file. See `src/magos/config.py` for the full settings schema.

## Develop

```bash
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy                      # type check (strict)
uv run pytest                    # tests
uv run pytest --cov              # tests with coverage
uv run pre-commit run --all-files
```

## Project layout

See [CLAUDE.md](./CLAUDE.md) for the full layout, conventions, and development guidelines.

## License

TBD.
