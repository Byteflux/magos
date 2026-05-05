# Examples

## Alias normalisation before match

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

## Header-driven tier routing

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

## OpenAI Responses passthrough to a self-hosted upstream

A passthrough rule forwards raw bytes (preserving `previous_response_id`
chaining and any built-in tool declarations like `web_search` /
`file_search`) to a same-shape upstream:

```yaml
rules:
  - name: responses-self-hosted
    match:
      endpoint: { literal: /v1/responses }
    action:
      provider: openai
      mode: passthrough
      base_url: https://my-openai-compat.internal
      api_key_env: SELF_HOSTED_API_KEY
```

Translate-mode rules go through `litellm.aresponses`, which handles
provider-specific bridging (e.g. an OpenAI Responses request can be
served by a non-OpenAI provider supported by litellm).

## Auxiliary Responses endpoints (retrieve / cancel / list input items)

The Responses API is stateful: clients chain follow-ups with
`previous_response_id` and may want to retrieve, cancel, or inspect a
prior response. These endpoints have no litellm equivalent, so they must
be routed via `mode: passthrough`:

```yaml
rules:
  - name: openai-responses-aux
    match:
      any_of:
        - endpoint: { literal: "/v1/responses/{id}" }
        - endpoint: { literal: "/v1/responses/{id}/input_items" }
    action:
      provider: openai
      mode: passthrough
      base_url: https://api.openai.com
      api_key_env: OPENAI_API_KEY
```

Match expressions see the templated path; the dispatcher forwards the
concrete inbound path (e.g. `/v1/responses/resp_abc`) and HTTP method
(GET for retrieve / list, DELETE for cancel) verbatim. Pointing a
`mode: translate` rule at one of these endpoints produces a `503
dispatch_error` because the dispatcher cannot translate non-POST
traffic.

## Reject streaming for a specific model

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
