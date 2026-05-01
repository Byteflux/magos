# Magos

LLM inference API proxy built on mitmproxy. Translates between Anthropic and OpenAI endpoint shapes, applies Headroom context compression, and exposes a unified MCP endpoint.

## Goals

- High performance, small compute footprint
- Built on mitmproxy
- Supports both Anthropic and OpenAI endpoint shapes; translates between mixed request/response shapes
- Declarative configuration
- Context compression with Headroom
- Dynamic, customizable routing
- Unified MCP endpoint
- Strong observability

## Stack

- **Python**: 3.12+ (see `.python-version`)
- **Package manager**: `uv`
- **Proxy core**: `mitmproxy`
- **LLM proxy/router**: `litellm`
- **Compression**: `headroom-ai`
- **MCP**: `fastmcp`
- **Validation**: `pydantic`
- **Logging**: `structlog`

## Layout

```
src/magos/
  __main__.py        # entrypoint (`python -m magos`)
  config.py          # MagosSettings (pydantic-settings)
  server.py          # FastAPI app
  proxy.py           # mitmproxy integration
  addon.py           # mitmproxy addon
  passthrough.py     # byte-exact Anthropic forwarding
  tokens.py          # token counting
  obs.py             # logging + tracing setup
  translation/       # Anthropic <-> OpenAI translation
    forward.py       # Anthropic -> OpenAI
    reverse.py       # OpenAI -> Anthropic
    streaming.py     # streaming translator
    _models.py       # shared pydantic models
    _shared.py       # helpers
tests/               # pytest suites (unit, integration, e2e)
scripts/             # fixture-capture utilities
pyproject.toml       # deps + tool config (ruff, mypy, pytest, coverage)
```

## Common commands

```bash
uv sync --extra cpu              # install with CPU torch
uv sync --extra gpu              # install with GPU torch
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy                      # type check
uv run pytest                    # tests
uv run pytest --cov              # tests with coverage
uv run pre-commit run --all-files
```

## Conventions

- **Style**: `ruff` (lint + format), 100-col lines, double quotes, PEP 8.
- **Types**: `mypy --strict` in src/. Tests are exempt from `disallow_untyped_defs`.
- **Tests**: pytest with markers `unit`, `integration`, `e2e`. Target 80% coverage.
- **Logging**: `structlog`, never `print()` in src/.
- **Config**: declarative, parsed via `pydantic` models.
- **Errors**: handle explicitly at boundaries, never silently swallow.
- **Immutability**: `@dataclass(frozen=True)` or `NamedTuple` for value types.

## Status

Active development. Core proxy, translation (Anthropic <-> OpenAI, including streaming), passthrough mode, token counting, and observability are implemented with unit and e2e test coverage (incl. agent-sdk e2e). MCP endpoint and dynamic routing are still to come.
