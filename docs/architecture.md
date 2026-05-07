# Architecture

Orientation map for engineers landing in magos cold. Covers the
non-obvious cross-cutting facts a fresh contributor would otherwise
have to reconstruct from reading 6+ files. Verified against the source
on the dates the references resolve. Split into focused sub-docs below.

| Topic | Contents |
|---|---|
| [Request flow](architecture/request-flow.md) | Process topology, request lifecycle, the `body_dirty` contract, why passthrough is byte-exact |
| [Startup](architecture/startup.md) | Startup order (`serve_async` / `build_api` / `lifespan`), registry single-writer invariant, `litellm.drop_params=True` global |
| [Headers and auth](architecture/headers-and-auth.md) | Auth-header injection (OAuth detection, override, provider default), multi-stage header forwarding |
| [Translation](architecture/translation.md) | Anthropic-shape cross-provider translation: Anthropic-only field stripping, `effort` → `reasoning_effort`, `additionalProperties` coercion |
| [Environment variables](architecture/env-vars.md) | Config-path resolution order, `MAGOS_HOME` bootstrap semantics, full env-var table |
| [Tests](architecture/testing.md) | Markers, e2e gate, conftest preloads, test app construction, completion mocking |
| [Gotchas](architecture/gotchas.md) | Subtleties worth not forgetting (the cheat sheet) |

## Observability decorators

The router and gateway both support optional Decorator wrapping assembled
in `magos.service.build.build_request_service` (the composition root).

### Router decorators

| Class | Condition | Emits |
|---|---|---|
| `MeasuredRouter` (`magos.routing.engine.measured`) | `metrics_enabled=True` | `magos.router.decisions` OTel counter, labelled by `kind` (ok / error) and `code` |

### Gateway decorators

| Class | Condition | Emits |
|---|---|---|
| `TracingGateway` (`magos.dispatch.gateway.tracing`) | Always wired | `gateway.dispatch` OTel span with `magos.gateway`, `magos.provider`, `magos.endpoint`, `magos.dispatch_model` attributes. No-op until `MAGOS_OTEL_ENABLED=1` configures a real tracer. |
| `MeasuredGateway` (`magos.dispatch.gateway.measured`) | `metrics_enabled=True` | `magos.gateway.dispatches` counter (labelled by `gateway`, `endpoint`, `outcome`) + `magos.gateway.duration_ms` histogram |

Wiring order (innermost → outermost): `RoutedGateway → TracingGateway [→ MeasuredGateway]`.
Both decorators are safe to add or remove independently; neither changes
dispatch behaviour.

See also: [routing](routing.md), [registry](registry.md),
[ingress](ingress.md), [cli](cli.md), [deployment](deployment.md),
[headroom](headroom.md).
