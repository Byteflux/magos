# Model registry

Magos discovers, caches, and merges model metadata from multiple
providers into a single in-memory registry. Routing rules can pin
specific models to providers, and unmatched requests fall back to
auto-routing via exact namespaced lookup against the registry.

| Topic | Contents |
|-------|----------|
| [Overview](registry/overview.md) | Why the registry exists and the boot/refresh lifecycle with failure modes. |
| [Config](registry/config.md) | `magos.yaml` grammar for `providers`/`provider_order`/`registry` blocks and the override > discovery > LiteLLM merge. |
| [Auto-routing](registry/auto-routing.md) | Registry fallback after explicit rules, tie-breaking, and LiteLLM provider naming for openai-compatible upstreams. |
| [CLI](registry/cli.md) | `magos models` subcommands and the public `GET /v1/models` listing. |
| [Observability](registry/observability.md) | OTel metrics and structlog events emitted by the refresher. |
