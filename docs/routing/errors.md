# Errors

- `404`: no rule matched. Body: per-endpoint error envelope echoing
  the inbound `model` and a `magos.yaml` hint.
- `503`: a rule matched but dispatch failed (jq_patch result not an
  object, missing api_key_env). Body: `route configuration error: ...`.
- `502`: upstream returned an error or the connection failed.
  Untouched by routing; the existing handler in `ingress/http/` wraps it.

Endpoint-shaped envelopes:

| Endpoint                              | Shape    |
|---------------------------------------|----------|
| `/v1/messages`                        | Anthropic|
| `/v1/messages/count_tokens`           | Anthropic|
| `/v1/chat/completions`                | OpenAI   |
| `/v1/responses`                       | OpenAI   |
| `/v1/responses/{id}`                  | OpenAI   |
| `/v1/responses/{id}/input_items`      | OpenAI   |

## Validation at config load

Loader rejects (raises `RoutingConfigError`):

- regex / glob / jq programs that fail to compile
- `gateway: passthrough` rules without `base_url`

Loader debug-logs (structlog `routing.passthrough_body_touch`):

- a `gateway: passthrough` rule combined with a body-touching rewrite
  (`set_model` or `jq_patch`); re-serialisation breaks byte-exact
  cache hits.
