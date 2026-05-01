# Magos

LLM inference API proxy built on [mitmproxy](https://mitmproxy.org/). Exposes Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses endpoints, applies [Headroom](https://github.com/headroom-ai/headroom) context compression, and (planned) exposes a unified MCP endpoint.

## Features

- **Multi-shape endpoints**: Anthropic Messages, OpenAI Chat Completions, OpenAI Responses (streaming and non-streaming)
- **Cross-provider translation** delegated to [LiteLLM](https://github.com/BerriAI/litellm): Anthropic-shape input can target OpenAI / Azure / Bedrock / Vertex / etc., and vice versa
- **Byte-exact passthrough**: forward same-shape requests verbatim, preserving auth, beta flags, and prompt-cache hashes
- **Token counting** endpoint via the upstream's native count-tokens API
- **Declarative routing** via `magos.yaml` — match on model / header / endpoint / jq expressions, rewrite headers and bodies, dispatch to translate or passthrough
- **Observability**: structured logging (`structlog`) and OpenTelemetry tracing
- **Configuration** via `pydantic-settings` (env vars prefixed `MAGOS_` or `.env`)

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
