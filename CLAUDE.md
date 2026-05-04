# Magos

Declarative routing proxy for LLM API traffic. Inbound requests
(Anthropic Messages, OpenAI Chat Completions, OpenAI Responses) hit a
rule engine that decides per request: which provider, byte-exact
passthrough vs LiteLLM-translated dispatch, which rewrites apply
(including Headroom context compression). A provider-discovered model
registry catches anything the rules don't match. An optional embedded
mitmproxy listener handles `HTTPS_PROXY`-style ingress for clients that
can't be reconfigured (notably Claude Code).

## Conceptual model

Three layers, in flow order:

- **Ingress** — how requests enter. FastAPI is the default entry point;
  mitmproxy is the optional `HTTPS_PROXY` entry point. Both feed the
  same routing engine.
- **Routing** — the rule engine in `magos.routing`. The product. Reads
  `magos.yaml`, decides per request: provider, mode, rewrites,
  dispatch model id.
- **Egress** — how requests leave. Three paths: byte-exact passthrough,
  wire-shape-translated via LiteLLM, or count-tokens.

## Goals

- High performance, small compute footprint
- Declarative configuration (`magos.yaml`)
- Both Anthropic and OpenAI endpoint shapes; cross-shape translation
- Context compression with Headroom
- Provider-discovered model registry with auto-routing
- Optional `HTTPS_PROXY`-style ingress via embedded mitmproxy
- Unified MCP endpoint (planned)
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
  serve.py           # process orchestrator: uvicorn + (optional) mitmproxy on one loop

  config/            # process + yaml configuration
    settings.py      # MagosSettings (pydantic-settings; env-only overrides)
    schema.py        # MagosServerConfig + IngressConfig (yaml `server:` block)
    loader.py        # load_full_config -> MagosConfig (routing + registry + server)
    paths.py         # magos_home(), resolve_models_path()

  telemetry/         # observability scaffolding
    logging.py       # structlog setup, get_logger
    tracing.py       # OTel + traced decorator
    metrics.py       # Prometheus exporter + OTel meter provider
  ingress/           # how requests enter
    http/            # FastAPI entry (was: server.py split)
      app.py        # create_app, app.state wiring
      lifespan.py   # async context manager (Headroom warmup, refresher start, kompress)
      handlers.py   # 7 endpoint handlers (4 POST + 3 auxiliary)
      run.py        # shared dispatch helper called by every handler
      headers.py    # _BLOCKED_FORWARD_HEADERS + filter
      admin.py      # /admin/registry/* mount
    mitm/            # optional in-process mitmproxy ingress (HTTPS_PROXY interception)
      addon.py      # MagosIngressAddon: TLS termination + rewrite to FastAPI
      master.py     # build_ingress_master factory (DumpMaster + addons)
      log_bridge.py # mitmproxy stdlib-logging records -> structlog

  routing/           # the rule engine (the product)
    schema.py        # pydantic schemas for magos.yaml rules (renamed from models.py)
    request.py       # RoutedRequest dataclass
    matchers.py      # match-expression evaluator (registry-aware)
    engine.py        # route(req, cfg, registry=...) -> RouteDecision | RouteError
    auto_route.py    # registry-driven fallback (extracted from engine.py)
    errors.py        # per-endpoint error envelopes
    loader.py        # YAML -> RoutingConfig with post-load validation
    jq_compat.py     # jq compile + truthy predicate helpers
    rewrites/        # pre/post rewrite primitives (was: rewrites.py)
      headers.py     # SetHeader / AddHeader / RemoveHeader
      model.py       # SetModel
      jq_patch.py    # JqPatch
      compress.py    # Compress + model_limit resolution + sentence_transformers preload

  egress/            # how requests leave
    dispatch.py      # RouteDecision -> execution branch (was: routing/dispatch.py)
    auth.py          # provider-aware API-key + header injection (extracted from dispatch.py)
    passthrough.py   # byte-exact same-shape forwarding (was: top-level)
    tokens.py        # async count_tokens via litellm.acount_tokens (was: top-level)
    observer.py      # mitmproxy egress observer addon (was: addon.py)
    translate/       # LiteLLM SDK marshalling (was: proxy.py)
      payload.py     # _build_payload, header allowlists, canonical fields
      sse.py         # SSE framing helpers
      anthropic.py   # anthropic_messages flows + output_config translation
      openai_chat.py # acompletion flows
      openai_responses.py # aresponses flows

  registry/          # model registry: discovery, lifecycle, lookup
    state.py         # ModelEntry / RegistryState frozen dataclasses (renamed from models.py)
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

Wire-shape translation between Anthropic and OpenAI is delegated to
LiteLLM's SDK (``litellm.anthropic_messages`` for ``/v1/messages``,
``litellm.acompletion`` for ``/v1/chat/completions``,
``litellm.aresponses`` for ``/v1/responses``,
``litellm.acount_tokens`` for ``/v1/messages/count_tokens``). The
calling code lives under ``magos.egress.translate``. Magos owns
routing, header forwarding, byte-exact passthrough
(``magos.egress.passthrough``), and observability; LiteLLM owns
wire-shape translation across providers.

## Library roles

| Library | Role | Magos package |
|---------|------|---------------|
| FastAPI | HTTP-level entry routing | `magos.ingress.http` |
| mitmproxy | optional HTTPS_PROXY ingress (TLS termination) | `magos.ingress.mitm` |
| — | rule-based router (the product) | `magos.routing` |
| LiteLLM | wire-shape translator | `magos.egress.translate` |
| httpx | byte-exact egress forwarder | `magos.egress.passthrough` |

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
