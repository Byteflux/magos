# Config

## Config grammar

```yaml
provider_order: [openrouter, anthropic, openai]    # tie-break for bare-id auto-routing

pins:                                              # per-raw-id pin; beats provider_order
  claude-sonnet-4-6: anthropic
  gpt-4o: openai

providers:
  openrouter:
    api_key_env: OPENROUTER_API_KEY                # env-var only, no inline secrets
    discovery: openrouter                          # optional; inferred from base_url host (see below)
    refresh_interval: 4h                           # optional, overrides global default
    litellm_provider: openrouter                   # optional, overrides adapter default
    models:                                        # optional per-model overrides
      "anthropic/claude-sonnet-4-6":
        context_size: 1000000                      # override discovery's value
        litellm_id: "openrouter/anthropic/claude-sonnet-4-6:1m"

  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    discovery: anthropic

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
  models_path: models.json                         # ~ expands, absolute passes through, relative anchors to $MAGOS_HOME
  deprecation_grace_seconds: 259200                # 3 days
  discovery_timeout_seconds: 30
  discovery_max_attempts: 3
  boot_discovery_timeout_seconds: 10
  boot_discovery_max_attempts: 1
```

`models_path` defaults to `$MAGOS_HOME/models.json` (i.e.
`~/.magos/models.json` when `MAGOS_HOME` is unset). `~`-prefixed paths
expand against the operator's home directory; other absolute paths
pass through; relative paths anchor to `$MAGOS_HOME` (the magos data
directory), not the yaml file's parent or CWD.

Operators can override this without editing the yaml by setting
`MAGOS_MODELS_PATH`. Precedence: `MAGOS_MODELS_PATH` env >
`registry.models_path` in yaml > derived default. Same path
semantics apply at every layer (`~`, absolute, or relative-to-
`$MAGOS_HOME`).

The server is the sole writer of `models.json`; out-of-process
readers are fine, but mutations go through the admin endpoints (or
the CLI, which wraps them).

Discovery adapters: `openai`, `anthropic`, `openrouter`, `vultr`, `noop`.
When `discovery:` is omitted, the adapter is inferred from the provider's
`base_url` host:

| Host substring         | Adapter      |
|------------------------|--------------|
| `openrouter.ai`        | `openrouter` |
| `anthropic.com`        | `anthropic`  |
| `vultrinference.com`   | `vultr`      |
| anything else with `base_url` | `openai` |
| `base_url` unset       | `noop` (manual-only) |

## Field-precedence merge

For each model the registry resolves fields by walking three sources
in order; the first non-null value wins per field:

1. **Override**: `providers.<X>.models.<id>` in `magos.yaml`
2. **Discovery**: what the live adapter returned
3. **LiteLLM**: `litellm.get_model_info(litellm_id)` lookup

The `sources` field on each entry records which layers contributed,
in priority order.
