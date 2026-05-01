# Routing

Magos routes every inbound request through a declarative ruleset loaded
from `magos.yaml`. Rules choose:

- **provider**: which upstream serves the request (`anthropic`, `openai`, ...)
- **mode**: `translate` (litellm round-trip) or `passthrough` (byte-exact bytes)
- **base_url**: passthrough target host
- **api_key_env**: env var holding the credential
- **count_tokens_mode**: `local` (litellm estimator) or `passthrough` (provider SDK)

Plus optional **rewrites** that mutate the request body or headers before
or after a rule matches.

No match → `404` with an endpoint-shaped error envelope.

## Setup

```bash
cp magos.example.yaml magos.yaml
# edit magos.yaml
MAGOS_CONFIG_PATH=./magos.yaml python -m magos
```

`MAGOS_CONFIG_PATH` defaults to `./magos.yaml`.

## Pipeline

```
inbound request
  -> pre_rewrites          (global, applied unconditionally, top-to-bottom)
  -> match                 (against rewritten request)
  -> post_rewrites         (per matched rule, top-to-bottom)
  -> dispatch via action
```

Rules are evaluated **first-match-wins**. If you want a fallback, declare
it last.

## YAML grammar

```yaml
pre_rewrites: []          # global rewrites; optional, default empty
rules:                    # required, at least one
  - name: human-readable  # optional; appears in route.matched logs
    match: <expr>
    rewrites: []          # per-rule post-rewrites; optional
    action:
      provider: <string>  # required
      mode: translate | passthrough
      base_url: <url>     # required when mode=passthrough
      api_key_env: <NAME> # optional
      count_tokens_mode: local | passthrough  # default local
```

### Match expressions

Atoms (each is a single-key dict):

| Atom        | Shape                                              | Matches against         |
|-------------|----------------------------------------------------|-------------------------|
| `model`     | `{ model: <matcher> }`                             | `body.model` (string)   |
| `header`    | `{ header: { name: <matcher>, value: <matcher> } }`| any inbound header pair |
| `endpoint`  | `{ endpoint: <matcher> }`                          | `/v1/messages` etc.     |
| `jq`        | `{ jq: "<expr>" }`                                 | parsed body (truthy)    |

`<matcher>` is exactly one of:

- `{ literal: "x" }` — exact equality, case-sensitive
- `{ glob: "x*" }` — fnmatch, case-sensitive
- `{ regex: "^x" }` — `re.fullmatch`, no implicit flags

Combinators:

- `{ all_of: [<expr>, ...] }` — every child must match
- `{ any_of: [<expr>, ...] }` — at least one child must match
- `{ not: <expr> }` — child must not match

A bare atom at the top of `match` is shorthand for a single-atom expression.

### Rewrite ops

Each is a single-key dict applied in list order:

| Op             | Shape                                     | Effect                                 |
|----------------|-------------------------------------------|----------------------------------------|
| `set_model`    | `{ set_model: "x" }`                      | replace `body.model`; flips body_dirty |
| `set_header`   | `{ set_header: { name: ..., value: ... }}`| insert or overwrite (case-insensitive) |
| `add_header`   | `{ add_header: { name: ..., value: ... }}`| insert only if absent                  |
| `remove_header`| `{ remove_header: "name" }`               | drop if present                        |
| `jq_patch`     | `{ jq_patch: "<expr>" }`                  | result replaces body; must be a JSON object |

`jq_patch` and `set_model` mark the request body as dirty. Under
`mode: passthrough`, a dirty body forces re-serialisation, breaking
prompt-cache byte-exactness; the loader logs a warning per offending
rule at startup.

### API-key handling

`api_key_env` is the name of an environment variable, **not** the key
itself.

- **translate mode**: the dispatcher reads `os.environ[api_key_env]` and
  passes it to litellm via the `api_key=` kwarg. Lets one provider use
  multiple keys (e.g. tier-routing) by declaring separate rules with
  different env vars.
- **passthrough mode**: when the inbound request has neither
  `Authorization` nor `x-api-key`, the dispatcher injects
  `x-api-key: <env value>` into the forwarded headers. Headers do not
  participate in the prompt-cache hash, so this is safe.

A missing or empty env var produces `503 dispatch_error` at request
time, not config-load time (env state can change between deploys).

## Errors

- `404` — no rule matched. Body: per-endpoint error envelope echoing
  the inbound `model` and a `magos.yaml` hint.
- `503` — a rule matched but dispatch failed (jq_patch result not an
  object, missing api_key_env). Body: `route configuration error: ...`.
- `502` — upstream returned an error or the connection failed.
  Untouched by routing; the existing handler in `server.py` wraps it.

Endpoint-shaped envelopes:

| Endpoint                         | Shape    |
|----------------------------------|----------|
| `/v1/messages`                   | Anthropic|
| `/v1/messages/count_tokens`      | Anthropic|
| `/v1/chat/completions`           | OpenAI   |

## Validation at config load

Loader rejects (raises `RoutingConfigError`):

- regex / glob / jq programs that fail to compile
- `count_tokens_mode: passthrough` for a provider not in
  `tokens.PASSTHROUGH_DISPATCH` (currently only `anthropic`)
- `mode: passthrough` rules without `base_url`

Loader warns (structlog `routing.passthrough_body_touch`):

- a `mode: passthrough` rule combined with a body-touching rewrite
  (`set_model` or `jq_patch`) — re-serialisation breaks byte-exact
  cache hits.

## Migration from the implicit-prefix era

These knobs no longer exist; the loader logs `config.removed_env_var` at
startup for any that remain in the environment:

- `MAGOS_ANTHROPIC_PASSTHROUGH_ENABLED`
- `MAGOS_ANTHROPIC_UPSTREAM_URL`
- `MAGOS_COUNT_TOKENS_PASSTHROUGH_PROVIDERS`

The shipped `magos.example.yaml` reproduces their behaviour. Copy it,
edit, set `MAGOS_CONFIG_PATH`.

## Examples

### Alias normalisation before match

```yaml
pre_rewrites:
  - jq_patch: 'if .model == "sonnet"
                 then .model = "claude-haiku-4-5-20251001"
                 else . end'

rules:
  - match: { model: { literal: "claude-haiku-4-5-20251001" } }
    action:
      provider: anthropic
      mode: passthrough
      base_url: https://api.anthropic.com
      api_key_env: ANTHROPIC_API_KEY
```

### Header-driven tier routing

```yaml
rules:
  - name: cheap-tier
    match:
      all_of:
        - model: { glob: "gpt-*" }
        - header:
            name: { literal: x-magos-tier }
            value: { literal: cheap }
    rewrites:
      - set_model: gpt-4o-mini
    action:
      provider: openai
      mode: translate
      api_key_env: OPENAI_API_KEY_TIER_CHEAP

  - name: default
    match: { model: { glob: "gpt-*" } }
    action:
      provider: openai
      mode: translate
      api_key_env: OPENAI_API_KEY
```

### Reject streaming for a specific model

```yaml
rules:
  - name: claude-no-stream
    match:
      all_of:
        - model: { literal: "claude-haiku-4-5-20251001" }
        - not: { jq: ".stream == true" }
    action:
      provider: anthropic
      mode: passthrough
      base_url: https://api.anthropic.com
      api_key_env: ANTHROPIC_API_KEY
  # Streaming claude requests fall through and 404.
```

## Logging

Per-request structlog events:

- `route.matched` — `rule`, `endpoint`, `model`, `mode`
- `route.unmatched` — `endpoint`, `model`, `message`
- `route.dispatch_error` — `rule`, `endpoint`, `error`

Per-startup events:

- `routing.passthrough_body_touch` — body-rewrite + passthrough warning
- `config.removed_env_var` — stale env var still set
