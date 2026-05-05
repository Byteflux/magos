# Routing

Magos routes every inbound request through a declarative ruleset loaded
from `magos.yaml`. The supported endpoints are:

| Endpoint                                  | Shape / purpose                                                                              |
|-------------------------------------------|----------------------------------------------------------------------------------------------|
| `POST /v1/messages`                       | Anthropic Messages                                                                           |
| `POST /v1/messages/count_tokens`          | Anthropic count_tokens                                                                       |
| `POST /v1/chat/completions`               | OpenAI Chat Completions                                                                      |
| `POST /v1/responses`                      | OpenAI Responses (translate via `litellm.aresponses`, or passthrough)                        |
| `GET /v1/responses/{id}`                  | Retrieve a stored response (passthrough-only)                                                |
| `DELETE /v1/responses/{id}`               | Cancel an in-flight background response (passthrough-only)                                   |
| `GET /v1/responses/{id}/input_items`      | List the input items used to produce a response (passthrough-only)                           |

Auxiliary `/v1/responses/{id}*` endpoints have no litellm equivalent, so a
matching rule must use `mode: passthrough`. Match expressions see the
templated path (e.g. `/v1/responses/{id}`) so rules stay stable across
response IDs; the dispatcher forwards the concrete inbound path so the
upstream sees the real id.

Rules choose:

- **provider**: which upstream serves the request (`anthropic`, `openai`, ...)
- **mode**: `translate` (litellm round-trip) or `passthrough` (byte-exact bytes)
- **base_url**: passthrough target host
- **api_key_env**: env var holding the credential

Plus optional **rewrites** that mutate the request body or headers before
or after a rule matches.

No match → `404` with an endpoint-shaped error envelope, **unless** the
model registry is configured: an unmatched request falls through to
exact `<provider>/<raw_id>` lookup against the registry. Explicit rules
always win; the registry only catches what the rules miss. See
[auto-routing](./registry/auto-routing.md) for details.

| Topic | Contents |
|-------|----------|
| [pipeline](./routing/pipeline.md) | Setup, config-path resolution, and the request lifecycle through pre-rewrites, match, post-rewrites, and dispatch. |
| [grammar](./routing/grammar.md) | YAML schema: match expressions, rewrite ops, `set_model` registry resolution, and `compress` (Headroom) configuration. |
| [api-keys](./routing/api-keys.md) | How `api_key_env` is consumed in translate vs passthrough mode and provider-block inheritance. |
| [errors](./routing/errors.md) | HTTP status codes, endpoint-shaped error envelopes, and config-load validation. |
| [examples](./routing/examples.md) | Worked configs: alias normalisation, tier routing, Responses passthrough, auxiliary endpoints, streaming rejection. |
| [logging](./routing/logging.md) | Per-request and per-startup structlog event names and fields. |
