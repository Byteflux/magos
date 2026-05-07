# Architecture migration

**Status**: in progress (Phase A active as of 2026-05-07).

**Lifetime**: temporary. Delete this file when all eight phases (tasks #48–#55) are completed and the redesign is fully landed. The cross-cutting facts that should outlive the migration get folded into `docs/architecture.md` and `CLAUDE.md` as the relevant phases land.

---

## Mission

**Restructure magos's internals to mirror the composability of the yaml it executes.**

Today, magos's *external* surface is a declarative composition kit — write rules, compose rewrites, swap providers, all from yaml. The *internal* surface doesn't read that way. Behavior is implemented as procedural module-level functions: `route()` runs the rule engine, `process_routed_request()` chains rewrite-compression-dispatch as a sequence, and both ingress surfaces (FastAPI and mitmproxy) reach into the same shared procedures. There's no `Router` class, no `Gateway` class, no `Compressor` class — and no central place that says "this is how magos is wired."

This work introduces named abstractions for the four cross-cutting concerns — `Router`, `Compressor`, `Gateway`, and `Transform` — each as an ABC with canonical implementations and optional decorators (Measured, Tracing, CCR). A `RequestService` becomes the Application-Service-Layer entry point shared by both surfaces. Construction logic consolidates into one composition root per surface (`build_api`, `build_proxy`, `build_request_service`) — the only places in the codebase that import broadly across packages.

The yaml schema, package layout, and class hierarchy share one vocabulary. `target.gateway: translate` parses to a `Gateway` from `magos.dispatch.gateway.translate`; `compress.engine: token` parses to a `Compressor` from `magos.compression.engine.token`. Adding a new gateway, compressor, or transform is one file plus one line in the composition root — not a search-and-edit across handlers, addons, and dispatchers.

Top-level packages reorient around honest surface naming: `cli/`, `api/`, `proxy/` for the three ingress surfaces; `service/`, `routing/`, `compression/`, `dispatch/`, `shapes/`, `registry/`, `config/`, `telemetry/` for shared infrastructure. The old `ingress/`/`egress/` flow-direction split, `cache/` and `ccr/` top-level packages, and inline construction in lifespan components all dissolve.

## Guiding principles

- **Composition over modules-as-singletons.** Behavior is in classes with constructor-injected collaborators, not in functions that depend on import-order side effects.
- **Patterns from the catalog, not invented.** Every Tier-1 abstraction maps to a published pattern: Service Layer, Strategy, Decorator, Adapter, Visitor, Pipes-and-Filters, Composition Root.
- **Vocabulary alignment.** Yaml field names, package paths, and class names use the same words. A reader who learns the term in one place knows it in the others.
- **One file per concept.** Per-mode gateways, per-engine compressors, per-shape adapters — each is its own file, named for what it is.
- **No backwards-compatibility shims.** Renames are clean cuts; the codebase only ever holds one shape at a time.

## What this work is NOT

- **Not a behavior change.** The product surface is unchanged: same yaml output for the same input, same observability, same dispatch matrix, same byte-exact passthrough.
- **Not a feature delivery.** No new gateways, compressors, or routing primitives. The existing ones get clearer homes.
- **Not a performance rewrite.** Hot paths shouldn't get faster *or* slower — same code, reorganized.
- **Not a config flag day.** The yaml schema renames (`action`→`target`, `mode`→`gateway`/`engine`, `rewrites`+`compress`→`transforms`) require operators to update their config; we own that migration in docs and the example yaml, but we do not ship a translator.

## Pattern map

Each Tier-1 abstraction maps to a recognized published pattern. The table also notes patterns we *considered* and rejected with the reason.

| Magos abstraction | Pattern | Source | Notes |
|---|---|---|---|
| `RequestService` | **Service Layer** | PEAA (Fowler) | Owns the application boundary; coordinates the response per operation; shared by both ingress surfaces (api + proxy). |
| `Router` (ABC) | **Content-Based Router** | EIP (Hohpe & Woolf) | Routes by message contents (matchers evaluate against `RoutedRequest`). |
| `RuleBasedRouter` | Strategy + Specification (DDD) | GoF + DDD | Multiple Router impls (`AutoRouter` fallback, `MeasuredRouter` decorator) selected at construction; matchers as composable Specifications. |
| `Compressor` (ABC) + `TokenCompressor` / `CacheCompressor` / `ResponsesCompressor` | **Strategy** | GoF | Interchangeable algorithms behind one interface; selected by yaml's `compress.engine` field. |
| `Gateway` (ABC) + per-mode impls | **Strategy** + **Gateway** (PEAA, narrow) | GoF + PEAA | The umbrella is Strategy (selection by `target.gateway`); each impl is a Fowler Gateway (encapsulates one external system: httpx, LiteLLM SDK, LiteLLM count). |
| `RoutedGateway` | Composite of Gateways | GoF (Composite-ish) | The selector that picks one Gateway per request based on `decision.target.gateway`. |
| `MeasuredRouter`, `MeasuredGateway`, `TracingGateway`, `CCRGateway` | **Decorator** | GoF | Wrap an inner abstraction with cross-cutting concerns; composed in the `build_*` roots. |
| `TranslateAdapter` per shape | **Adapter** + **Message Translator** | GoF + EIP | Adapts in-shape request to LiteLLM SDK call (Adapter), and translates between Anthropic and OpenAI shapes (Message Translator). |
| `Transform` (unified ABC) + per-rule chain | **Pipes and Filters** | EIP / Cloud | One ordered chain per `RouteDecision`; rewrites and compression are both `Transform` subclasses. |
| Visitor for `MatchExpr` evaluation | **Visitor** | GoF | The match-expression tree is closed (atoms + AND/OR/NOT); the evaluator visits it. |
| `build_api`, `build_proxy`, `build_request_service` | **Composition Root** | Mark Seemann (DI book) | Sole locations that import broadly across packages; assemble the object graph for one surface. |
| Magos as a deployed product | **API Gateway** | Cloud (Microsoft) / Microservices (Richardson) | Routing + offloading (auth, TLS termination via mitmproxy); not a class in our code, but the deployment-level pattern that describes the product. |

### Patterns considered and rejected

- **Fowler `Gateway` for `RequestService`.** Wrong fit: Fowler's `Gateway` wraps an external API (matches our per-mode gateway impls, not the request-lifecycle coordinator). Calling the service `Gateway` would have been "the gateway inside the gateway" — confusing at the product level (magos *is* an API gateway).
- **`Pipeline` as the unified ABC name** (early proposal). Wrong: the steps are heterogeneous (decision-making, transformation, I/O, post-processing) — not the homogeneous "stage in, stage out" chain that `Pipeline` implies. Also: `Pipeline` is already taken (`compression/pipeline.py` wraps `headroom.transforms.TransformPipeline`).
- **`Filter` as the unified ABC name** (later proposal). Wrong: "filter" carries a "select/admit/reject" connotation (ACL filters, log filters, `filter()`) — the opposite of what our chain does. `Transform` is active and accurate.
- **`Protocol` over `ABC`** (initial default). Reconsidered: all implementations are internal (no need to retrofit foreign types), explicit subclassing improves readability ("this *is* a Router"), `@abstractmethod` gives runtime construction-time safety, matches the Java-`interface` mental model the user wanted to borrow from. ABC won.
- **Flattening inner packages** (e.g., `routing/{router.py, rule_based.py, ...}` instead of `routing/engine/{base.py, rule_based.py, ...}`). Rejected — too many files at one package level; grouping helps navigation.
- **`shapes/` → `shape/` rename.** Reconsidered after audit: plural is correct because shapes are *value-discriminated* peer entities (three `Shape` instances), not subclasses. Mirrors `sqlalchemy.dialects` / `pydantic.types`. Convention encoded in `CLAUDE.md`.
- **`action.gateway` field flattened to `set_gateway` rewrite.** Rejected: action fields (`provider`, `gateway`, `base_url`, `auth_*`) are routing metadata (consulted by the dispatcher) not body transformations. Required-ness and "primary meaning" benefit from a structured block.
- **`engine.py` (singular file) for the rule engine** under flattened layout. Rejected because the user objected to the parent/inner agent-noun symmetry (`routing/router/`); chose the inner-package form `routing/engine/` with `base.py` for the ABC.

## Locked target structure

```
src/magos/
  __main__.py
  cli/                              # operator CLI (unchanged)

  api/                              # was ingress/http/
    build.py                        # build_api(cfg) -> FastAPI
    app.py                          # FastAPI factory
    adapter.py                      # FastAPI <-> RoutedRequest/Response
    lifespan/                       # FastAPI lifespan components
    routes/
      llm.py                        # was handlers.py
      models.py
      admin.py
      metrics.py

  proxy/                            # was ingress/mitm/ + egress/observer.py
    build.py                        # build_proxy(cfg) -> ProxyListener
    listener.py                     # mitmproxy DumpMaster wrapper
    addons/
      ingress.py                    # MagosIngressAddon (TLS termination, rewrite to api)
      observer.py                   # was egress/observer.py
    log_bridge.py

  service/                          # NEW — application service layer
    request.py                      # RequestService (Service Layer)
    build.py                        # build_request_service(cfg, refresher)
    # mcp.py                        # future MCP service

  routing/
    engine/
      base.py                       # class Router(ABC)
      rule_based.py                 # RuleBasedRouter (canonical)
      auto.py                       # AutoRouter (registry-driven fallback)
      measured.py                   # MeasuredRouter (decorator)
    match/                          # visitor for MatchExpr evaluation
      base.py                       # MatchVisitor protocol + node ABCs
      atoms.py                      # leaf matchers
      combinators.py                # AND, OR, NOT
      evaluator.py                  # MatcherEvaluator(MatchVisitor)
    rewrite/
      base.py                       # class Rewriter(Transform, ABC)
      headers.py                    # SetHeader, AddHeader, RemoveHeader
      model.py                      # SetModel
      jq_patch.py                   # JqPatch
    schema/                         # yaml pydantic models
    transform.py                    # class Transform(ABC) — single file
    decision.py                     # RouteDecision (carries Target + Sequence[Transform])
    errors.py                       # RouteError + per-endpoint envelopes
    request.py                      # RoutedRequest
    loader.py                       # yaml -> RoutingConfig
    jq_compat.py

  compression/
    engine/
      base.py                       # class Compressor(Transform, ABC)
      token.py                      # TokenCompressor
      cache.py                      # CacheCompressor
      responses.py                  # ResponsesCompressor
    tracker/                        # was top-level magos/cache/
      session_id.py
      store.py                      # TrackerStore
    ccr.py                          # request-side CCR injection
    warmup.py                       # eager_warmup, prebuild_from_routing
    model_limit.py
    errors.py

  dispatch/                         # was egress/
    gateway/
      base.py                       # class Gateway(ABC)
      passthrough.py                # PassthroughGateway
      translate.py                  # TranslateGateway
      count_tokens.py               # CountTokensGateway
      routed.py                     # RoutedGateway (selector by target.gateway)
      ccr.py                        # CCRGateway (response-side decorator)
      measured.py                   # MeasuredGateway (decorator)
      tracing.py                    # TracingGateway (decorator)
    adapter/                        # LiteLLM SDK marshalling
      base.py                       # class TranslateAdapter(ABC)
      anthropic.py
      openai_chat.py
      openai_responses.py
      payload.py                    # build_payload + allowlists
      sse.py                        # SSE framing helpers
    usage/                          # streaming-specific (Usage type lives in shapes/)
      accumulator.py
      tap.py
    auth.py                         # API-key + provider-aware header injection
    errors.py

  shapes/                           # plural — value-discriminated registry
    _base.py                        # class Shape (was ShapeSpec) + StreamEvent
    usage.py                        # Usage dataclass
    anthropic.py                    # ANTHROPIC = Shape(...)
    openai_chat.py
    openai_responses.py

  registry/                         # unchanged (provider-discovered model registry)
  config/                           # unchanged (env + yaml settings)
  telemetry/                        # unchanged (logging, tracing, metrics)
```

## Vocabulary alignment

The yaml field, package path, and class name use one word for one concept.

| Concept | Yaml | Package | Class |
|---|---|---|---|
| The routing decision target | `target` | — | `Target` |
| Which outbound gateway | `target.gateway: translate` | `magos.dispatch.gateway.translate` | `TranslateGateway` |
| Which compression engine | `compress.engine: token` | `magos.compression.engine.token` | `TokenCompressor` |
| Provider identifier | `target.provider: anthropic` | `magos.shapes.anthropic` | `ANTHROPIC` (`Shape` instance) |
| Filter chain | `transforms: [...]` | `magos.routing.transform` | `Transform` |
| Pre-rewrite primitives | `transforms: [{set_header: ...}]` | `magos.routing.rewrite.headers` | `SetHeader` |

## Yaml schema renames

These are operator-visible breaks; no shim is shipped. Each phase's task notes the doc + example-yaml updates that travel with the code change.

| Old | New | Phase |
|---|---|---|
| `action:` block | `target:` block | C2 |
| `action.mode: passthrough \| translate` | `target.gateway: passthrough \| translate` | C2 |
| `compress.mode: token \| cache` | `compress.engine: token \| cache` | C3 |
| `rewrites: [...]` + `compress: {...}` (separate fields) | `transforms: [...]` (unified discriminated-union list) | C3 |

Note: `count_tokens` does **not** become a `target.gateway` value. The `/v1/messages/count_tokens` endpoint selects the count-tokens gateway implicitly; rules don't declare it.

## Class renames

| Old | New | Phase |
|---|---|---|
| `ShapeSpec` (frozen dataclass) | `Shape` (frozen dataclass + extraction methods) | A |
| `Shape` (Literal alias) | (deleted — callers pass `Shape` instances) | A |
| `process_routed_request` (free function) | `RequestService.process` (method) | B |
| `Action` (Pydantic model) | `Target` (Pydantic model) | C2 |
| `RouteDecision.action` | `RouteDecision.target` | C2 |
| `DispatchMode` (Literal) | `GatewayMode` (Literal) | C2 |
| `TokenModeCompressor` | `TokenCompressor` | C3 |
| `CacheModeCompressor` | `CacheCompressor` | C3 |
| `RouteDecision.rewrites` + `RouteDecision.compressor` | `RouteDecision.transforms` (unified) | C3 |

## Phases

Eight phases, each tracked as a task. Each is independently mergeable, leaves the test suite green, and contains no transitional shims.

| # | Task | Title | Risk | Notes |
|---|---|---|---|---|
| A | #48 | shapes refactor — `Shape` class + `Usage` move | Low | Foundation: every later phase depends on `Shape` being a real abstraction. |
| B | #49 | extract `RequestService` (Application Service Layer) | Medium | Cutover from function to method-on-injected-instance, both ingress surfaces. |
| C1 | #50 | `Router` ABC + `RuleBasedRouter` + visitor for matchers | Medium | Largest sub-change is the visitor refactor for `MatchExpr` evaluation. |
| C2 | #51 | `Gateway` ABC + per-mode gateways + rename `Action` → `Target` | Medium | Biggest yaml-schema change; operators must update config. |
| C3 | #52 | `Compressor` ABC + `Transform` unification (Pipes-and-Filters) | High | Most invasive — touches schema, runtime, many tests. |
| D | #53 | top-level package reorganization | High mechanical / low semantic | Atomic rename batch (one commit, no shims). |
| E | #54 | composition roots — `build_api`, `build_proxy`, `build_request_service` | Low | Pure code-motion; lifespan components simplify. |
| F | #55 | cross-cutting decorators — `Measured*`, `Tracing*` | Low | Pure addition + reduction; each decorator is independent. |

Dependency chain: `A → B → C1 → C2 → C3 → D → E → F`. See task descriptions (#48–#55) for the detailed file lists, validation steps, and per-phase doc updates.

## Success criteria

- A new contributor can read `magos/service/build.py` and understand how every other component plugs in.
- Adding a new `Compressor` requires touching exactly two files: the new compressor's source file, and the composition root.
- Every Tier-1 abstraction has a one-line description that maps to a catalog pattern.
- The `magos.example.yaml`, `docs/architecture.md`, and `CLAUDE.md` agree on every name.
- Test coverage stays at or above the current 90% gate throughout the migration.
- This document and tasks #48–#55 are deleted/closed.

## References

- `CLAUDE.md` — singular/plural package naming convention is encoded there. The Layout section will be rewritten in Phase D as the package paths shift.
- `docs/architecture.md` — current architecture; the cross-cutting facts that should outlive this migration get folded back here as phases land.
- Patterns catalog used as the lens: PEAA (Fowler), GoF (Gamma et al.), EIP (Hohpe & Woolf), Microsoft Cloud Design Patterns, microservices.io (Richardson), DDD Reference (Evans).
