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
    settings.py      # MagosSettings (pydantic-settings; env-only overrides) + magos_home()
    schema.py        # MagosServerConfig + IngressConfig (yaml `server:` block)
    loader.py        # load_full_config -> MagosConfig (routing + registry + server) + resolve_models_path

  telemetry/         # observability scaffolding
    logging.py       # structlog setup, get_logger
    tracing.py       # OTel + traced decorator
    metrics.py       # Prometheus exporter + OTel meter provider
  ingress/           # how requests enter
    http/            # FastAPI entry
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
    schema.py        # pydantic schemas for magos.yaml rules
    request.py       # RoutedRequest dataclass
    matchers.py      # match-expression evaluator (registry-aware)
    engine.py        # route(req, cfg, registry=...) -> RouteDecision | RouteError
    auto_route.py    # registry-driven fallback
    errors.py        # per-endpoint error envelopes
    loader.py        # YAML -> RoutingConfig with post-load validation
    jq_compat.py     # jq compile + truthy predicate helpers
    rewrites/        # pre/post rewrite primitives
      headers.py     # SetHeader / AddHeader / RemoveHeader
      model.py       # SetModel
      jq_patch.py    # JqPatch
      compress.py    # Compress + model_limit resolution + sentence_transformers preload

  egress/            # how requests leave
    dispatch.py      # RouteDecision -> execution branch
    auth.py          # provider-aware API-key + header injection
    passthrough.py   # byte-exact same-shape forwarding
    tokens.py        # async count_tokens via litellm.acount_tokens
    observer.py      # mitmproxy egress observer addon
    translate/       # LiteLLM SDK marshalling
      payload.py     # build_payload, header allowlists, canonical fields
      sse.py         # SSE framing helpers
      anthropic.py   # anthropic_messages flows + output_config translation
      openai_chat.py # acompletion flows
      openai_responses.py # aresponses flows

  registry/          # model registry: discovery, lifecycle, lookup
    state.py         # ModelEntry / RegistryState frozen dataclasses
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

  cli/               # operator CLI; entrypoint is magos.cli.app:main
    app.py           # root Typer app, --config / --version, default-to-serve
    serve.py         # `serve` command + bootstrap (logging/tracing config + log event)
    models.py        # `magos models {list,show,refresh,prune,discover}` subapp
    _helpers.py      # shared state-loading + print helpers (admin_client, load_state, print_list)
    admin_client.py  # tiny httpx wrapper for /admin/registry endpoints
magos.example.yaml   # routing config to copy and customise
tests/               # mirrors src/magos/ — see "Test layout" below
scripts/             # operator-facing one-shot probes
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
  cli/, config/, egress/{translate/}, ingress/{http,mitm}/,
  registry/, routing/{rewrites/}/
  test_serve.py, test_smoke.py, test_telemetry.py,
  test_e2e.py, test_e2e_agent_sdk.py
```

Plain helper functions (request builders, sample payloads, TestClient
factories) live in `_helpers.py` modules at the appropriate scope —
`tests/routing/_helpers.py`, `tests/ingress/http/_helpers.py`, etc. Tests
import them via relative imports (`from ._helpers import make_req`).
`conftest.py` is reserved for pytest fixtures; do not put plain helpers
there.

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

### Layout & module shape

- **Direction-of-flow top-level packages**. `ingress/` (how requests
  enter), `routing/` (the rule engine — the product), `egress/` (how
  they leave). New code goes into one of these, picked by which side of
  the request lifecycle it touches. Cross-cutting infrastructure
  (`telemetry/`, `config/`, `registry/`, `cli/`) gets its own peer
  package; do not bury it under a flow package.
- **Name modules for what they do, not what they are.** `translate`
  (LiteLLM SDK marshalling), `passthrough` (byte-exact forwarding),
  `observer` (mitmproxy log addon) — not `proxy.py`, `addon.py`,
  `utils.py`. Re-name when the role changes; a wrong name compounds.
- **Small focused files.** Aim for one cohesive concept per module.
  When a single file grows past ~400 LOC and contains multiple variants
  / primitives / endpoint families, split it into a package: per-variant
  files plus a thin `__init__.py` that re-exports the public surface and
  holds the dispatcher. Recent examples: `routing/rewrites/`
  (per-primitive), `egress/translate/` (per-endpoint family),
  `ingress/http/` (per-handler).
- **No backwards-compat re-exports during reorgs.** Move the symbol and
  update every importer. A two-line `from .new import old` shim is
  technical debt that ages badly.
- **Public dispatcher in `__init__.py`, private implementation in
  siblings.** `routing/rewrites/__init__.py` exposes `apply_rewrites` +
  `RewriteError` and dispatches to per-primitive applicators in
  `headers.py`, `model.py`, `jq_patch.py`, `compress.py`. Callers
  import from the package, not the implementation files.
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
  absolute value — otherwise the test silently picks up emissions from
  any prior test that happens to run first.

### Comments and docstrings

- **Describe current behaviour, not history.** Comments and docstrings
  are documentation, not changelog. "Was the X seam", "extracted from
  Y", "no longer infers Z", "after the SDK fold-in", "renamed from W"
  — all noise. Rewrite each as "does X" / "is Y" / "infers Z when …".
  The git log carries the "what changed" story; the comment exists for
  someone reading the code today.
- **Keep version pins and compatibility notes.** "LiteLLM 1.82+ yields
  bytes already SSE-framed" stays — it's a real fact a reader needs
  when debugging or upgrading. The test is whether the note still
  helps if you removed the surrounding history: a version pin does, a
  refactor reference doesn't.
- **Don't name internal feature tags** ("the registry batch", "the
  ingress reorg") in comments. They're shorthand only the original
  author understands. Describe the concept (`provider/provider_order/registry`
  yaml blocks) instead.
- **When you change behaviour, update the docstrings/comments around
  it in the same change** — same rule as docs, same reason.

### Documentation

- Docs live in `docs/` (one file per top-level concept) and are
  indexed in the layout block above. **Update docs as part of the
  change that invalidates them**, not in a follow-up PR — drift is
  load-bearing for new contributors and the agents reading this file.
- When you move or rename a source file, grep `docs/` and `CLAUDE.md`
  for the old path and update every reference. The cost is one
  `grep` + a handful of edits; the cost of skipping it is days of
  someone reading the wrong file.
- When you add a new top-level concept (a CLI subapp, a new ingress
  mode, a new egress translation path, a new env var, a new yaml
  block), decide whether it warrants:
  1. A new doc — only if it's a distinct operator-facing surface
     with its own configuration / lifecycle / failure modes (e.g.
     `cli.md`, `ingress.md`).
  2. A section in an existing doc — the default; cross-link from
     the layout block in this file.
  3. Just a docstring on the source — for internal-only concepts.
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

Active development. Core proxy (Anthropic / OpenAI Chat Completions / OpenAI Responses shapes), byte-exact passthrough, token counting, observability, **declarative rule-based routing** (`magos.yaml`), **Headroom compression** (`compress` rewrite primitive with token and cache-align modes), and a **provider-driven model registry** with auto-routing, soft-delete deprecation, OTel metrics, and an operator CLI (`magos models …`) are implemented with unit + e2e coverage (incl. agent-sdk e2e). Wire-shape translation is delegated to LiteLLM's SDK. MCP endpoint is the only major surface still to come.
