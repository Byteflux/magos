# Magos

LLM inference API proxy built on [mitmproxy](https://mitmproxy.org/). Exposes Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses endpoints, applies [Headroom](https://github.com/headroom-ai/headroom) context compression, drives a provider-discovered model registry, and (planned) exposes a unified MCP endpoint.

## Features

- **Multi-shape endpoints**: Anthropic Messages, OpenAI Chat Completions, OpenAI Responses (streaming and non-streaming)
- **Cross-provider translation** delegated to [LiteLLM](https://github.com/BerriAI/litellm): Anthropic-shape input can target OpenAI / Azure / Bedrock / Vertex / etc., and vice versa
- **Byte-exact passthrough**: forward same-shape requests verbatim, preserving auth, beta flags, and prompt-cache hashes
- **Token counting** endpoint via the upstream's native count-tokens API
- **Declarative routing** via `magos.yaml` — match on model / header / endpoint / jq / registry-field expressions, rewrite headers and bodies, dispatch to translate or passthrough
- **Headroom compression**: `compress` rewrite primitive with token and cache-align modes; registry-aware `model_limit` resolution
- **Model registry**: per-provider auto-discovery (OpenAI / Anthropic / OpenRouter / manual), field-precedence merge over operator overrides and LiteLLM's bundled metadata, soft-delete deprecation, atomic `models.json` persistence
- **Auto-routing fallback**: requests no rule matches resolve via exact `<provider>/<raw_id>` lookup against the registry
- **Operator CLI**: `magos models {list, show, refresh, prune, discover}` against a running server (with disk fallback for read paths)
- **Observability**: structured logging (`structlog`), OpenTelemetry tracing, OpenTelemetry metrics with optional Prometheus `/metrics` endpoint
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
mkdir -p ~/.magos
cp magos.example.yaml ~/.magos/magos.yaml      # then edit to taste
uv run magos                         # serve
uv run magos --config /etc/magos.yaml # non-default config
uv run magos models list             # CLI subcommand
```

Config path resolution: `--config` flag > `MAGOS_CONFIG_PATH` env > `~/.magos/magos.yaml`. Other knobs come from environment variables (prefix `MAGOS_`) or a local `.env`. See `src/magos/config.py` for the full settings schema, `docs/routing.md` for the rule grammar, and `docs/registry.md` for the registry.

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
