# Architecture

Orientation map for engineers landing in magos cold. Covers the
non-obvious cross-cutting facts a fresh contributor would otherwise
have to reconstruct from reading 6+ files. Verified against the source
on the dates the references resolve. Split into focused sub-docs below.

| Topic | Contents |
|---|---|
| [Request flow](architecture/request-flow.md) | Process topology, request lifecycle, the `body_dirty` contract, why passthrough is byte-exact |
| [Startup](architecture/startup.md) | Startup order (`serve_async` / `create_app` / `_lifespan`), registry single-writer invariant, `litellm.drop_params=True` global |
| [Headers and auth](architecture/headers-and-auth.md) | Auth-header injection (OAuth detection, override, provider default), multi-stage header forwarding |
| [Translation](architecture/translation.md) | Anthropic-shape cross-provider translation: Anthropic-only field stripping, `effort` → `reasoning_effort`, `additionalProperties` coercion |
| [Environment variables](architecture/env-vars.md) | Config-path resolution order, `MAGOS_HOME` bootstrap semantics, full env-var table |
| [Tests](architecture/testing.md) | Markers, e2e gate, conftest preloads, test app construction, completion mocking |
| [Gotchas](architecture/gotchas.md) | Subtleties worth not forgetting (the cheat sheet) |

See also: [routing](routing.md), [registry](registry.md),
[ingress](ingress.md), [cli](cli.md), [deployment](deployment.md),
[headroom](headroom.md).
