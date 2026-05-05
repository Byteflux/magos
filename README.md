# Magos

Declarative LLM API routing proxy with provider-discovered model registry and context compression.

Inbound requests (Anthropic Messages, OpenAI Chat Completions, OpenAI Responses) hit a rule engine declared in `magos.yaml`. Rules decide per request: which upstream provider, byte-exact passthrough vs. [LiteLLM](https://github.com/BerriAI/litellm)-translated dispatch, what to rewrite (headers, body, [Headroom](https://github.com/headroom-ai/headroom) context compression). A provider-discovered model registry catches anything explicit rules miss. An optional embedded `mitmproxy` listener handles `HTTPS_PROXY`-style ingress for clients that change behaviour when their `BASE_URL` is overridden, notably Claude Code.

## Features

- **Three endpoint shapes**: Anthropic Messages, OpenAI Chat Completions, OpenAI Responses (POST + retrieve / cancel / list-input-items), all with streaming.
- **Cross-provider translation** delegated to LiteLLM: Anthropic-shape input can target OpenAI / Azure / Bedrock / Vertex / OpenRouter / etc., and vice versa.
- **Byte-exact passthrough**: forward same-shape requests verbatim, preserving auth, beta flags, and prompt-cache hashes.
- **Token counting** via `litellm.acount_tokens`, which auto-picks between local tokenizers and the upstream's native count-tokens endpoint per model.
- **Declarative routing** in `magos.yaml`: match on model / header / endpoint / jq / registry-field expressions, rewrite headers and bodies, dispatch to translate or passthrough.
- **Headroom compression** as a routing rewrite (`compress`) with token and cache-align modes; per-model token-budget resolution via the registry or LiteLLM.
- **Model registry** with per-provider auto-discovery (OpenAI / Anthropic / OpenRouter / Vultr / manual via `noop`), field-precedence merge over operator overrides and LiteLLM's bundled metadata, soft-delete deprecation with grace period, atomic `models.json` persistence.
- **Auto-routing fallback** for unmatched requests via exact `<provider>/<raw_id>` lookup against the registry.
- **Operator CLI**: `magos serve`, `magos models {list, show, refresh, prune, discover}` against a running server with disk fallback for read paths.
- **Optional `HTTPS_PROXY` ingress** via embedded mitmproxy on the same process; useful for clients whose behaviour changes when their base URL is rewritten.
- **Observability**: `structlog` structured logging, OpenTelemetry tracing, OpenTelemetry metrics with optional Prometheus `/metrics` endpoint.
- **Configuration** via `pydantic-settings` (env vars prefixed `MAGOS_`, or a local `.env`) plus `magos.yaml` for routing / registry / ingress blocks.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

## Install

```bash
uv sync --extra cpu   # CPU torch
# or
uv sync --extra gpu   # GPU torch (CUDA)
```

## Run

```bash
mkdir -p ~/.magos
cp magos.example.yaml ~/.magos/magos.yaml      # edit to taste
uv run magos                                   # serve (default subcommand)
uv run magos --config /etc/magos.yaml          # non-default config
uv run magos --port 9000                       # override bind
uv run magos models list                       # CLI subcommand
```

The `magos` script is also installed by `[project.scripts]`, so once the venv is activated, `magos …` works without `uv run`. `python -m magos` is an equivalent invocation.

Container deploys via the bundled `Dockerfile` + `compose.yaml`; see [`docs/deployment.md`](docs/deployment.md) for the GPU/CPU build paths and volume layout.

## Configuration

Resolution order, highest first:

1. CLI flags (`--config`, `--host`, `--port`).
2. Environment variables (`MAGOS_*`, optionally via `.env`).
3. `magos.yaml` blocks: `pre_rewrites` / `rules` (routing), `providers` / `provider_order` / `registry` (the model registry), `ingress` (FastAPI bind + optional mitmproxy proxy).
4. Built-in defaults.

`MAGOS_HOME` (default `~/.magos`) anchors `MAGOS_CONFIG_PATH` and `MAGOS_MODELS_PATH`. Full env-var table in [`docs/cli.md`](docs/cli.md).

## Develop

```bash
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy                      # type check (strict)
uv run pytest                    # tests + 90% coverage gate
uv run pre-commit run --all-files
```

End-to-end tests against real upstream providers gate on `MAGOS_E2E=1`.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | request lifecycle, dispatch matrix, body-dirty contract, env vars, gotchas. |
| [`docs/routing.md`](docs/routing.md) | `magos.yaml` rule grammar, examples, error envelopes. |
| [`docs/registry.md`](docs/registry.md) | registry config, lifecycle, observability, `model_field` matchers. |
| [`docs/cli.md`](docs/cli.md) | operator CLI reference, env-var table, exit codes. |
| [`docs/ingress.md`](docs/ingress.md) | embedded mitmproxy ingress: setup, CA trust, loop hazard. |
| [`docs/headroom.md`](docs/headroom.md) | Headroom integration notes and non-obvious findings. |
| [`docs/deployment.md`](docs/deployment.md) | Docker / compose deployment. |
| [`integrations/opencode/README.md`](integrations/opencode/README.md) | OpenCode plugin that registers magos models. |
| [`CLAUDE.md`](CLAUDE.md) | full project layout and conventions. |

## License

Apache License 2.0; see [`LICENSE`](LICENSE).

Copyright 2026 Matthew Harris and the Magos contributors
