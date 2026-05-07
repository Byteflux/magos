# Magos

Declarative LLM API routing proxy with provider-discovered model
registry and context compression. Inbound requests (Anthropic Messages,
OpenAI Chat Completions, OpenAI Responses) hit a rule engine that
decides per request: which provider, byte-exact passthrough vs
LiteLLM-translated dispatch, which transforms apply (including Headroom
context compression). A provider-discovered model registry catches
anything the rules don't match. An optional embedded mitmproxy
listener handles `HTTPS_PROXY`-style ingress for clients that can't be
reconfigured (notably Claude Code).

## Conceptual model

Three layers, in flow order:

- **Ingress**: how requests enter. FastAPI is the default entry point;
  mitmproxy is the optional `HTTPS_PROXY` entry point. Both feed the
  same routing engine.
- **Routing**: the rule engine in `magos.routing`. The product. Reads
  `magos.yaml`, decides per request: provider, gateway, transforms,
  dispatch model id.
- **Egress**: how requests leave. Three paths: byte-exact passthrough,
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

  api/               # FastAPI ingress for client traffic
    __init__.py     # re-exports build_api
    build.py        # build_api — composition root; wires collaborators into the FastAPI app
    lifespan/       # ordered LifespanComponent runner
      __init__.py   # Protocol + _COMPONENTS list + lifespan asynccontextmanager
      kompress.py   # KompressBackendOverride + KompressPreload
      components.py # MetricsMeter + MagosCompressionWarmup + RegistryRefresher
    handlers.py     # endpoint handlers (4 POST + 3 auxiliary)
    run.py          # FastAPI <-> RequestService adapter (run_endpoint)
    headers.py      # _BLOCKED_FORWARD_HEADERS + forwardable_headers
    models.py       # GET /v1/models (registry-backed, OpenAI/Anthropic shape)
    admin.py        # /admin/registry/* mount

  proxy/             # mitmproxy ingress + egress observer
    __init__.py     # re-exports build_proxy
    build.py        # build_proxy — composition root; wires addons into a mitmproxy DumpMaster
    log_bridge.py   # mitmproxy stdlib-logging records -> structlog
    addons/
      ingress.py    # MagosIngressAddon: TLS termination + rewrite to api
      observer.py   # mitmproxy egress observer addon

  service/           # application service layer (transport-agnostic)
    __init__.py     # public surface (RequestService, RoutedResponse, build_request_service)
    request.py      # RequestService.process: route -> transform -> gateway -> RoutedResponse
    build.py        # build_request_service(cfg, refresher, registry_cfg, *, metrics_enabled)

  routing/           # the rule engine (the product)
    __init__.py     # public surface (RoutingConfig, RouteDecision, RouteError, route, ...)
    transform.py    # Transform ABC (Pipes-and-Filters step)
    request.py      # RoutedRequest dataclass + Endpoint / HttpMethod types
    decision.py     # RouteDecision frozen value (engine output, dispatch input)
    errors.py       # RouteError + per-endpoint envelopes
    loader.py       # YAML -> RoutingConfig with post-load validation
    jq_compat.py    # jq compile + truthy predicate helpers
    schema/          # pydantic schemas for magos.yaml rules
      __init__.py    # public surface + config_uses_compress walker
      base.py        # _Frozen pydantic base (frozen + extra=forbid + populate_by_name)
      grammar.py     # matchers + atoms + combinators + MatchExpr
      rewrites.py    # transform primitives (SetHeader / SetModel / Compress / ...) with apply
      structure.py   # Target + Rule + GuardedTransforms + RoutingConfig
    engine/          # Router ABC + canonical implementations
      __init__.py    # re-exports + free `route()` convenience function
      base.py        # Router ABC
      rule_based.py  # RuleBasedRouter (canonical) + apply_pre_transforms + _route
      auto.py        # AutoRouter: registry-driven fallback
      measured.py    # MeasuredRouter (decorator: OTel counter)
    match/           # match-expression evaluator
      __init__.py    # re-exports `matches`
      evaluator.py   # registry-aware match-expression walker
    rewrites/        # transform application + helper modules
      __init__.py    # apply_transforms (polymorphic loop) + RewriteError
      base.py        # Rewriter(Transform, ABC) marker class
      jq_patch.py    # RewriteError exception (JqPatch.apply lives on the schema class)
      compress/      # compressor support modules used by TokenCompressor
        __init__.py  # public re-exports (_resolve_model_limit, _preload_sentence_transformers)
        model_limit.py   # registry/litellm-driven context-window resolution
        _preload.py      # sentence_transformers native-load preload

  compression/        # headroom TransformPipeline lifecycle + tracker + CCR + engine classes
    __init__.py     # public surface (PipelineConfig, apply, eager_warmup, prebuild_from_routing)
    config.py       # PipelineConfig + fingerprint + pipeline_config_from_compress_options
    build.py        # build_pipeline(config, provider_name) -> TransformPipeline
    registry.py     # PipelineRegistry caches by (fingerprint, provider_name)
    pipeline.py     # apply() + ApplyResult; inflation guard
    warmup.py       # eager_warmup + prebuild_from_routing (per-rule pipeline pre-warm)
    engine/         # Compressor ABC + per-engine implementations
      __init__.py   # re-exports (Compressor, TokenCompressor, CacheCompressor, ResponsesCompressor)
      base.py       # Compressor(Transform, ABC) marker class
      token.py      # TokenCompressor (token-mode chat-shape pipeline)
      cache.py      # CacheCompressor (cache-aligner for chat-shape messages)
      responses.py  # ResponsesCompressor (cache-aligner for /v1/responses instructions)
    tracker/        # per-session prefix-cache tracking
      __init__.py   # public surface (TrackerStore, get_store, derive_session_id, PrefixCacheTracker)
      session_id.py # derive_session_id(headers, body, provider) -> str
      store.py      # TrackerStore: dict[(session_id, provider), PrefixCacheTracker]; TTL evict
    ccr/            # CCR (reversible compression) integration with headroom.ccr
      __init__.py   # public surface (is_ccr_request, wrap_response, wrap_stream, ...)
      continuation.py # closure builder; re-runs translate adapter with substituted messages/tools
      handler.py    # wrap_response (non-streaming) + wrap_stream (streaming) handlers

  dispatch/          # how requests leave
    __init__.py      # CompletionFn Protocol
    errors.py        # DispatchError shared across dispatch branches
    auth.py          # provider-aware API-key + header injection
    passthrough.py   # call_passthrough / stream_passthrough (httpx forwarders)
    tokens.py        # async count_tokens via litellm.acount_tokens
    gateway/         # Gateway ABC + canonical implementations
      __init__.py    # public surface
      base.py        # Gateway ABC + make_on_complete helper
      passthrough.py # PassthroughGateway (byte-exact httpx forward)
      translate.py   # TranslateGateway (LiteLLM SDK + CCR wrap)
      count_tokens.py # CountTokensGateway
      routed.py      # RoutedGateway (composite selector by endpoint + target.gateway)
      measured.py    # MeasuredGateway (decorator: OTel counter + duration histogram)
      tracing.py     # TracingGateway (decorator: OTel span per dispatch)
    usage/           # per-response token-usage logging
      __init__.py    # public surface
      core.py        # Usage dataclass + usage_from_body + log_usage helpers
      accumulator.py # UsageAccumulator (streaming SSE event aggregator, shape-driven)
      tap.py         # tap_stream byte passthrough generator
    translate/       # LiteLLM SDK marshalling
      __init__.py    # TRANSLATE_HANDLERS dispatch table (endpoint -> TranslateAdapter)
      payload.py     # build_payload, header allowlists, canonical fields
      sse.py         # SSE framing helpers
      runner.py      # generic proxy_translate / stream_translate (per-adapter dispatch)
      safe_stream.py # gateway-boundary wrapper: mid-stream exc -> log + per-shape SSE error
      anthropic/     # /v1/messages translate path
        __init__.py  # re-exports ADAPTER + _dispatch_anthropic_messages
        translation.py # output_config / additionalProperties / unknown-field stripping
        dispatch.py  # anthropic_messages vs acompletion routing
        adapter.py   # TranslateAdapter wiring + model rewrite hooks
      openai_chat.py # acompletion flows
      openai_responses.py # aresponses flows

  shapes/            # wire-shape data: per-shape field locations + usage maps
    __init__.py     # public surface (Shape, SHAPES, Usage, shape_for_endpoint, shape_by_name)
    base.py         # Shape dataclass + StreamEvent + CompressionProvider literal
    usage.py        # Usage dataclass (normalised input/output/cache token counts)
    anthropic.py    # /v1/messages spec
    openai_chat.py  # /v1/chat/completions spec
    openai_responses.py # /v1/responses{,/{id}} spec

  config/            # process + yaml configuration
    __init__.py
    settings.py      # MagosSettings (pydantic-settings; env-only overrides) + magos_home()
    schema.py        # MagosIngressConfig + HttpIngressConfig + MitmIngressConfig (yaml `ingress:`)
    loader.py        # load_full_config -> MagosConfig (routing + registry + ingress)

  registry/          # model registry: discovery, lifecycle, lookup
    __init__.py     # package docstring only; callers import submodules directly
    state.py         # ModelEntry / RegistryState frozen dataclasses
    schema.py        # pydantic for providers/provider_order/pins/registry blocks
    store.py         # atomic JSON persistence (models.json)
    merge.py         # field precedence: override > discovery > litellm > null
    deprecation.py   # soft-delete state machine
    provider_order.py # tie-break: pin > order > lex-smallest
    refresher.py     # async lifecycle owner: load, boot-discover, refresh
    telemetry.py     # OTel meters + structlog event helpers
    litellm_lookup.py # bundled-registry fallback wrapper
    pipeline.py     # pure pipeline functions: merge, diff, override-to-fields conversion
    discovery/      # per-provider discovery adapters
      __init__.py   # adapter_for + DiscoveryAdapter Protocol re-exports
      base.py       # DiscoveryAdapter Protocol + JsonListAdapter + types
      factory.py    # adapter_for(ProviderConfig) -> DiscoveryAdapter
      _auth.py      # shared auth-header builders
      _coerce.py    # type-coercion helpers shared across adapters
      openai.py
      anthropic.py
      openrouter.py
      vultr.py
      noop.py

  telemetry/         # observability scaffolding
    __init__.py     # re-exports configure_logging / configure_tracing / get_logger / traced
    logging.py       # structlog setup, get_logger
    tracing.py       # OTel + traced decorator
    metrics.py       # Prometheus exporter + OTel meter provider

  cli/               # operator CLI; entrypoint is magos.cli.app:main
    __init__.py
    app.py           # root Typer app, --home / --config / --models / --version
    serve.py         # `serve` command + bootstrap (logging/tracing config + log event)
    models.py        # `magos models {list,show,refresh,prune,discover}` subapp
    _helpers.py      # shared state-loading + print helpers
    admin_client.py  # tiny httpx wrapper for /admin/registry endpoints

magos.example.yaml   # routing config to copy and customise
tests/               # mirrors src/magos/, see "Test layout" below
scripts/             # operator-facing one-shot probes
integrations/        # third-party tool integrations
  opencode/          # OpenCode plugin: registers magos models via /admin/registry
pyproject.toml       # deps + tool config (ruff, mypy, pytest, coverage)
docs/architecture.md # request lifecycle, lifespan, dispatch matrix, env vars, gotchas
docs/cli.md          # operator CLI: top-level options, subcommands, env-var table
docs/deployment.md   # Docker + compose deployment, GPU/CPU build, volume layout
docs/headroom.md     # Headroom integration notes + non-obvious findings
docs/ingress.md      # mitmproxy HTTPS_PROXY ingress: setup, CA trust, gotchas
docs/registry.md     # registry lifecycle, config, CLI, observability
docs/routing.md      # rule grammar, examples, env vars
```

### Test layout

`tests/` mirrors `src/magos/` directory-for-directory; each test file
exercises the like-named source module and lives in the same relative
position. Drop redundant prefixes (`test_routing_engine.py` → `tests/routing/test_engine.py`).
`__init__.py` files at every level keep test names unique across subtrees.

```
tests/
  conftest.py            # pytest fixtures (loaded automatically)
  fixtures/              # test routing yaml
  api/, proxy/, cli/, config/, registry/, shapes/, telemetry/
  compression/{engine/, tracker/, ccr/}
  dispatch/{gateway/, translate/, usage/}
  routing/{schema/, engine/, rewrites/{compress/}}
  e2e/                   # MAGOS_E2E=1-gated full-stack tests (FastAPI -> dispatch -> real provider, plus agent-sdk)
  test_main_module.py, test_serve.py, test_smoke.py
```

Plain helper functions (request builders, sample payloads, TestClient
factories) live in `_helpers.py` modules at the appropriate scope:
`tests/routing/_helpers.py`, `tests/api/_helpers.py`, etc. Tests
import them via absolute imports (`from tests.routing._helpers import
make_req`); ruff's `flake8-tidy-imports` (`ban-relative-imports = "all"`)
enforces this across both `src/` and `tests/`. `conftest.py` is reserved
for pytest fixtures; do not put plain helpers there.

**Start with `docs/architecture.md`** for cross-cutting facts (request
flow, body_dirty contract, passthrough byte-exactness, auth-header
injection, env vars, gotchas) that aren't tied to a single component.

Wire-shape translation between Anthropic and OpenAI is delegated to
LiteLLM's SDK (`litellm.anthropic_messages` for `/v1/messages`,
`litellm.acompletion` for `/v1/chat/completions`,
`litellm.aresponses` for `/v1/responses`,
`litellm.acount_tokens` for `/v1/messages/count_tokens`). The
calling code lives under `magos.dispatch.translate`. Magos owns
routing, header forwarding, byte-exact passthrough
(`magos.dispatch.passthrough`), and observability; LiteLLM owns
wire-shape translation across providers.

## Library roles

| Library | Role | Magos package |
|---------|------|---------------|
| FastAPI | HTTP-level entry routing | `magos.api` |
| mitmproxy | optional HTTPS_PROXY ingress (TLS termination) | `magos.proxy` |
| (none) | rule-based router (the product) | `magos.routing` |
| (none) | transport-agnostic request orchestrator (route -> transform -> dispatch) | `magos.service` |
| (none) | compression pipeline ownership over `headroom.transforms` (lifecycle, registry, inflation guard) | `magos.compression` |
| (none) | per-session prefix-cache tracker store wrapping `headroom.cache.prefix_tracker` | `magos.compression.tracker` |
| (none) | reversible-compression integration with `headroom.ccr` (request injection + response handling) | `magos.compression.ccr` |
| (none) | wire-shape data: per-shape field locations + usage maps consumed by usage / cache / etc. | `magos.shapes` |
| LiteLLM | wire-shape translator | `magos.dispatch.translate` |
| httpx | byte-exact egress forwarder | `magos.dispatch.passthrough` |

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

### Layout & module shape

- **Direction-of-flow top-level packages**. `api/` and `proxy/` (how
  requests enter), `routing/` (the rule engine, the product), `dispatch/`
  (how they leave). New code goes into one of these, picked by which side of
  the request lifecycle it touches. Cross-cutting infrastructure
  (`telemetry/`, `config/`, `registry/`, `cli/`, `compression/`) gets its
  own peer package; do not bury it under a flow package.
- **Name modules for what they do, not what they are.** `translate`
  (LiteLLM SDK marshalling), `passthrough` (byte-exact forwarding),
  `observer` (mitmproxy log addon), not `proxy.py`, `addon.py`,
  `utils.py`. Re-name when the role changes; a wrong name compounds.
- **Singular vs plural package names.** No formal PEP 8 rule, but a
  soft pattern in stdlib and major libraries: singular when the
  package holds an abstraction and its variants — an ABC plus its
  subclasses; plural when it holds distinct peer entities that share
  structure but not a polymorphic relationship. `shapes/` is plural
  because anthropic / openai-chat / openai-responses are three `Shape`
  *values* (not subclasses), mirroring `sqlalchemy.dialects`,
  `pydantic.types`, `concurrent.futures`. The shorthand:
  type-discrimination → singular; value-discrimination → plural.
- **Small focused files.** Aim for one cohesive concept per module.
  When a single file grows past ~400 LOC and contains multiple variants
  / primitives / endpoint families, split it into a package: per-variant
  files plus a thin `__init__.py` that re-exports the public surface and
  holds the dispatcher. Recent examples: `routing/rewrites/`
  (per-primitive), `dispatch/translate/` (per-endpoint family),
  `api/` (per-handler).
- **No backwards-compat re-exports during reorgs.** Move the symbol and
  update every importer. A two-line `from .new import old` shim is
  technical debt that ages badly.
- **Public dispatcher in `__init__.py`, private implementation in
  siblings.** `dispatch/translate/__init__.py` exposes
  `TRANSLATE_HANDLERS` and dispatches per-endpoint to `anthropic/`,
  `openai_chat.py`, `openai_responses.py`. Callers import from the
  package, not the implementation files. (When dispatch is naturally
  polymorphic — e.g. `routing/rewrites/__init__.py` calls
  `rw.apply(out)` on the Transform schema — the `__init__.py` is the
  thin entrypoint and per-primitive logic lives on the schema class
  itself.)
- **Tests mirror src.** When you split a source module, split its test
  file the same way; when you move source, `git mv` the test alongside.

### Style & types

- **Style**: `ruff` (lint + format), 100-col lines, double quotes, PEP 8.
- **Types**: `mypy --strict` in src/. Tests are exempt from
  `disallow_untyped_defs`.
- **Logging**: `structlog`, never `print()` in src/.
- **Config**: declarative, parsed via `pydantic` models.
- **Errors**: handle explicitly at boundaries, never silently swallow.
- **Immutability**: `@dataclass(frozen=True)` or `NamedTuple` for value
  types.

### Investigation

- **Verify, don't ask.** Before asking the user a question whose answer
  is observable (running state, file contents, log output, command
  exit), use the available tools to determine the answer. Reserve
  questions for genuine intent / preference / out-of-band context the
  tools can't reach. The user expects investigation, not interrogation.
- **E2E testing is mandatory for bug investigation.** When debugging a
  reported failure, write or extend an e2e test that exercises the
  failing path before proposing a diagnosis or fix. The test both
  proves the failure mode is understood and locks in the regression
  guard once it's resolved. Unit-level reasoning is not a substitute.
- **Set `MAGOS_HOME` to the project root for e2e runs.** The project
  ships a `magos.yaml` and `models.json` at the root; pinning
  `MAGOS_HOME=<repo root>` makes the spawned `magos serve` use them
  rather than the operator's `~/.magos/` (which may point at a
  Docker-mounted config or other unrelated state). The e2e fixtures
  in `tests/proxy/test_e2e.py` already do this; mirror that
  pattern in any new subprocess-spawning e2e.

### Tests

- pytest; markers `unit`, `integration`, `e2e` are declared but only
  applied in a handful of files. End-to-end tests gate on
  `MAGOS_E2E=1`. Coverage runs on every `pytest` invocation
  (`addopts` includes `--cov`); `fail_under = 90` in
  `[tool.coverage.report]` gates merges. Use `# pragma: no cover` only
  for code paths that genuinely cannot be exercised in-process (e.g.
  the `python -m magos` entrypoint guard).
- Running a subset (e.g. `pytest tests/cli/test_admin_client.py`)
  triggers the coverage gate against partial data and will fail the
  threshold. For iteration on a single file, append
  `--no-cov` (or `--cov-fail-under=0`) explicitly. Full-suite runs
  hit the gate as intended.
- One test file per source module, in the mirrored directory. Drop
  redundant directory-implied prefixes from filenames.
- Plain helpers in `_helpers.py` at the appropriate scope; pytest
  fixtures in `conftest.py`. Don't mix them.
- Counters / OTel meters are cumulative across the test session.
  Snapshot a baseline at test start and assert on the delta, not the
  absolute value, otherwise the test silently picks up emissions from
  any prior test that happens to run first.

### Comments and docstrings

- **Describe current behaviour, not history.** Comments and docstrings
  are documentation, not changelog. "Was the X seam", "extracted from
  Y", "no longer infers Z", "after the SDK fold-in", "renamed from W"
  are all noise. Rewrite each as "does X" / "is Y" / "infers Z when …".
  The git log carries the "what changed" story; the comment exists for
  someone reading the code today.
- **Keep version pins and compatibility notes.** "LiteLLM 1.82+ yields
  bytes already SSE-framed" stays: it's a real fact a reader needs
  when debugging or upgrading. The test is whether the note still
  helps if you removed the surrounding history: a version pin does, a
  refactor reference doesn't.
- **Don't name internal feature tags** ("the registry batch", "the
  ingress reorg") in comments. They're shorthand only the original
  author understands. Describe the concept (`provider/provider_order/registry`
  yaml blocks) instead.
- **When you change behaviour, update the docstrings/comments around
  it in the same change**: same rule as docs, same reason.

### Documentation

- Docs live in `docs/` (one file per top-level concept) and are
  indexed in the layout block above. **Update docs as part of the
  change that invalidates them**, not in a follow-up PR, since drift is
  load-bearing for new contributors and the agents reading this file.
- When you move or rename a source file, grep `docs/` and `CLAUDE.md`
  for the old path and update every reference. The cost is one
  `grep` + a handful of edits; the cost of skipping it is days of
  someone reading the wrong file.
- When you add a new top-level concept (a CLI subapp, a new ingress
  mode, a new egress translation path, a new env var, a new yaml
  block), decide whether it warrants:
  1. A new doc: only if it's a distinct operator-facing surface
     with its own configuration / lifecycle / failure modes (e.g.
     `cli.md`, `ingress.md`).
  2. A section in an existing doc: the default; cross-link from
     the layout block in this file.
  3. Just a docstring on the source: for internal-only concepts.
- Cross-link freely between docs. Each doc should have a "See also"
  pointer back to `architecture.md` and to peers it depends on.
- Worked examples in docs (yaml, CLI invocations) should be valid as
  written. If you change a config schema or rename a flag, fix the
  examples in the same change.
- Don't write aspirational docs. If a feature is planned-but-not-shipped,
  it doesn't belong in operator-facing docs; tracking goes in the
  Status section of this file or a roadmap issue.
- The Status section at the bottom of this file is the single source
  of truth for "what works today." Update it when a major surface
  lands or moves out of "to come."

## Status

Active development. Core proxy (Anthropic / OpenAI Chat Completions / OpenAI Responses shapes), byte-exact passthrough, token counting, observability, **declarative rule-based routing** (`magos.yaml`), **Headroom compression** (`compress` rewrite primitive with token and cache-align modes), and a **provider-driven model registry** with auto-routing, soft-delete deprecation, OTel metrics, and an operator CLI (`magos models …`) are implemented with unit + e2e coverage (incl. agent-sdk e2e). Wire-shape translation is delegated to LiteLLM's SDK. The internal architectural migration (Phases A–F) is complete: the codebase now uses named `Router`, `Gateway`, and `RequestService` abstractions with optional `MeasuredRouter`, `MeasuredGateway`, and `TracingGateway` decorators composed in `magos.service.build`. MCP endpoint is the only major surface still to come.
