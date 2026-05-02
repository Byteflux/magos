# Model registry

Magos discovers, caches, and merges model metadata from multiple
providers into a single in-memory registry. Routing rules can pin
specific models to providers, and unmatched requests fall back to
auto-routing via exact namespaced lookup against the registry.

## Why

Without the registry, every routing rule had to enumerate models by
regex or literal. Onboarding a new provider meant editing yaml. The
registry inverts that: providers describe themselves over their own
discovery API, magos merges with operator overrides and LiteLLM's
bundled metadata, and routing falls back to the registry when no
explicit rule applies.

## Lifecycle

```
boot
 ├── load models.json from disk      (regenerable cache, no schema versioning)
 ├── for each provider with no entries:
 │     run discovery once with tight timeout (10s, 1 attempt)
 │     populate state, persist
 └── start per-provider background loop
       ├── sleep refresh_interval (default 2h, per-provider override)
       ├── refresh with patient timeout (30s, 3 attempts, exponential backoff)
       ├── apply deprecation state machine
       └── atomic state swap, persist to models.json
```

Failure modes:

- **Boot discovery fails** → that provider boots empty; other providers
  unaffected. The background loop will retry on its normal cadence.
- **Background refresh fails** → prior state preserved (atomic). Failure
  metric increments; logs include the error type. Next tick tries again.
- **Provider drops a model** → the entry is marked `deprecated_at = now`
  and continues serving. If absent for 3 days (configurable), the entry
  is hard-deleted on the next refresh that includes that provider.
- **Model reappears mid-grace** → the deprecation mark is cleared.
- **Corrupt models.json** → file is treated as missing; live discovery
  rebuilds. No schema versioning by design.

## Config grammar

```yaml
provider_order: [openrouter, anthropic, openai]    # tie-break order

providers:
  openrouter:
    api_key_env: OPENROUTER_API_KEY                # env-var only, no inline secrets
    discovery: openrouter                          # optional; inferred from base_url when omitted
    refresh_interval: 4h                           # optional, overrides global default
    litellm_provider: openrouter                   # optional, overrides adapter default
    models:                                        # optional per-model overrides
      "anthropic/claude-sonnet-4-6":
        context_size: 1000000                      # override discovery's value
        litellm_id: "openrouter/anthropic/claude-sonnet-4-6:1m"

  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    discovery: anthropic_models

  manual-only-provider:
    # No discovery: manual-only. Models below are permanent until removed
    # from yaml.
    litellm_provider: openai
    models:
      custom-llama:
        context_size: 32768
        litellm_id: openai/custom-llama

registry:
  refresh_interval: 2h                             # global default
  on_unknown_model: error                          # error (404, default) | passthrough
  models_path: ./models.json                       # relative paths anchor to the yaml file's parent
  deprecation_grace_seconds: 259200                # 3 days
  discovery_timeout_seconds: 30
  discovery_max_attempts: 3
  boot_discovery_timeout_seconds: 10
  boot_discovery_max_attempts: 1
```

`models_path` resolves relative to the config file's parent directory
(absolute paths pass through). The server is the sole writer of
`models.json`; out-of-process readers are fine, but mutations go
through the admin endpoints (or the CLI, which wraps them).

Discovery adapters: `openai_models`, `anthropic_models`, `openrouter`,
`noop`. When `discovery:` is omitted, the adapter is inferred from the
provider's `base_url` host:

| Host substring         | Adapter            |
|------------------------|--------------------|
| `openrouter.ai`        | `openrouter`       |
| `anthropic.com`        | `anthropic_models` |
| anything else with `base_url` | `openai_models` |
| `base_url` unset       | `noop` (manual-only) |

## Field-precedence merge

For each model the registry resolves fields by walking three sources
in order; the first non-null value wins per field:

1. **Override** — `providers.<X>.models.<id>` in `magos.yaml`
2. **Discovery** — what the live adapter returned
3. **LiteLLM** — `litellm.get_model_info(litellm_id)` lookup

The `sources` field on each entry records which layers contributed,
in priority order.

## Auto-routing

Explicit rules win. After the rules loop falls through, magos attempts
`registry.get(<inbound model id>)`:

- Hit → synthesize a translate-mode `Action` for the entry's provider,
  hand the entry's `litellm_id` to the dispatcher.
- Miss + `on_unknown_model: error` → 404.
- Miss + `on_unknown_model: passthrough` → hand the raw model string to
  LiteLLM and let it resolve via its bundled registry (works for names
  like `openai/gpt-4o`).

When multiple providers serve the same logical model, tie-breaking:
explicit pin > `provider_order` > lexicographically smallest provider.

## Matcher language: `model_field`

Routing rules can match on registry fields:

```yaml
rules:
  - name: long-context-only
    match:
      all_of:
        - endpoint: { literal: /v1/messages }
        - model_field:
            field: context_size
            op: gte
            value: 200000
    action: { provider: anthropic, mode: translate }

  - name: vision-routing
    match:
      model_field:
        field: modalities
        op: contains
        value: image
    action: { provider: openrouter, mode: translate }
```

Operators: `eq`, `gt`, `gte`, `lt`, `lte` (numeric/string scalars),
`contains` (membership in tuple fields like `modalities`), `in`
(membership of the field value in a list).

## CLI

```bash
magos models list                     # in-memory state from server
magos models list --from-disk         # bypass server, read models.json
magos models list --format json       # machine-readable

magos models show openrouter/anthropic/claude-sonnet-4-6
magos models show <id> --from-disk

magos models refresh                  # all providers
magos models refresh --provider openrouter

magos models prune                    # sweep past-grace deprecated entries

magos models discover --provider openrouter --dry-run
```

Every subcommand accepts `--config <path>` to point at a non-default
yaml; precedence is `--config` > `MAGOS_CONFIG_PATH` > the
`~/.magos/magos.yaml` default.

`list` and `show` fall back to disk if the server isn't reachable.
`refresh` and `prune` require the server to be running and hit
`POST /admin/registry/{refresh,prune}`.

## Observability

OTel metrics (`magos.registry.*`) emitted by the refresher:

- `refresh.total{provider, status}` — counter (`attempt`, `success`, `failure`)
- `refresh.failures{provider, error_type}` — counter
- `refresh.duration` — histogram (seconds, per provider)
- `models.total{provider}` — observable gauge (active count, includes deprecated)
- `models.added{provider}`, `models.deprecated{provider}`, `models.pruned{provider}` — counters

Set `MAGOS_METRICS_ENABLED=1` to install the OTel Prometheus exporter
at startup and mount the `GET /metrics` endpoint. Without the env var,
the meters bind to OTel's no-op default and `/metrics` is not served.

structlog events:

- `registry.refresh.attempt` — debug, per refresh start
- `registry.refresh.success` — info, includes added/deprecated/pruned counts
- `registry.refresh.failure` — warning, includes error and error_type
- `registry.auto_route` — debug, when auto-routing picks a provider
