# Magos

LLM inference API proxy built on mitmproxy. Translates between Anthropic and OpenAI endpoint shapes, applies Headroom context compression, drives a provider-discovered model registry with auto-routing, and (planned) exposes a unified MCP endpoint.

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
  __main__.py        # entrypoint (`magos [serve|models …]`)
  config.py          # MagosSettings (pydantic-settings; env-only overrides)
  config_loader.py   # load_full_config -> MagosConfig (routing + registry + server)
  server_config.py   # MagosServerConfig schema (yaml `server:` block + ingress)
  serve.py           # process orchestrator: uvicorn + (optional) mitmproxy on one loop
  server.py          # FastAPI app + lifespan
  proxy.py           # translate-mode dispatch into litellm SDK call sites
  addon.py           # mitmproxy egress observer addon (host-allowlisted logging)
  passthrough.py     # byte-exact same-shape forwarding
  tokens.py          # async count_tokens via litellm.acount_tokens
  obs.py             # logging + tracing setup
  ingress/           # in-process mitmproxy ingress (HTTPS_PROXY interception)
    addon.py         # MagosIngressAddon: TLS termination + rewrite to FastAPI
    log_bridge.py    # mitmproxy stdlib-logging records -> structlog
    master.py        # build_ingress_master factory (DumpMaster + addons)
  routing/           # declarative rule-based routing
    models.py        # pydantic schemas for magos.yaml (incl. ModelFieldAtom)
    request.py       # RoutedRequest dataclass
    matchers.py      # match-expression evaluator (registry-aware)
    rewrites.py      # pre/post rewrite applicator (registry-aware compress)
    engine.py        # route(req, cfg, registry=...) -> RouteDecision | RouteError
    errors.py        # per-endpoint error envelopes
    loader.py        # YAML -> RoutingConfig with post-load validation
    dispatch.py      # decision -> proxy/passthrough/tokens dispatch
    jq_compat.py     # jq compile + truthy predicate helpers
  registry/          # model registry: discovery, lifecycle, lookup
    models.py        # ModelEntry / RegistryState frozen dataclasses
    schema.py        # pydantic for providers/provider_order/registry blocks
    store.py         # atomic JSON persistence (models.json)
    merge.py         # field precedence: override > discovery > litellm > null
    deprecation.py   # soft-delete state machine
    provider_order.py # tie-break: pin > order > lex-smallest
    refresher.py     # async lifecycle owner: load, boot-discover, refresh
    obs.py           # OTel meters + structlog event helpers
    litellm_lookup.py # bundled-registry fallback wrapper
    discovery/       # adapters
      base.py        # DiscoveryAdapter Protocol + types
      factory.py     # adapter_for(ProviderConfig) -> DiscoveryAdapter
      openai.py
      anthropic.py
      openrouter.py
      vultr.py
      noop.py
  cli/               # operator CLI dispatched from __main__
    models_cmd.py    # magos models {list,show,refresh,prune,discover}
    admin_client.py  # tiny httpx wrapper for /admin/registry endpoints
magos.example.yaml   # routing config to copy and customise
tests/               # pytest suites (unit, integration, e2e)
  fixtures/          # test routing yaml
scripts/             # operator-facing one-shot probes
pyproject.toml       # deps + tool config (ruff, mypy, pytest, coverage)
docs/architecture.md # request lifecycle, lifespan, dispatch matrix, env vars, gotchas
docs/ingress.md      # mitmproxy HTTPS_PROXY ingress: setup, CA trust, gotchas
docs/routing.md      # rule grammar, examples, env vars
docs/registry.md     # registry lifecycle, config, CLI, observability
docs/headroom.md     # Headroom integration notes + non-obvious findings
```

**Start with `docs/architecture.md`** for cross-cutting facts (request
flow, body_dirty contract, passthrough byte-exactness, auth-header
injection, env vars, gotchas) that aren't tied to a single component.

Translation between Anthropic and OpenAI shapes is delegated to LiteLLM's
SDK (``litellm.anthropic_messages`` for ``/v1/messages``,
``litellm.acompletion`` for ``/v1/chat/completions``,
``litellm.aresponses`` for ``/v1/responses``,
``litellm.acount_tokens`` for ``/v1/messages/count_tokens``). Magos owns
routing, header forwarding, byte-exact passthrough, and observability;
LiteLLM owns wire-shape translation across providers.

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
- **Tests**: pytest; markers `unit`, `integration`, `e2e` are declared but only applied in a handful of files. End-to-end tests gate on `MAGOS_E2E=1`. No coverage threshold is enforced today.
- **Logging**: `structlog`, never `print()` in src/.
- **Config**: declarative, parsed via `pydantic` models.
- **Errors**: handle explicitly at boundaries, never silently swallow.
- **Immutability**: `@dataclass(frozen=True)` or `NamedTuple` for value types.

## Status

Active development. Core proxy (Anthropic / OpenAI Chat Completions / OpenAI Responses shapes), byte-exact passthrough, token counting, observability, **declarative rule-based routing** (`magos.yaml`), **Headroom compression** (`compress` rewrite primitive with token and cache-align modes), and a **provider-driven model registry** with auto-routing, soft-delete deprecation, OTel metrics, and an operator CLI (`magos models …`) are implemented with unit + e2e coverage (incl. agent-sdk e2e). Wire-shape translation is delegated to LiteLLM's SDK. MCP endpoint is the only major surface still to come.
