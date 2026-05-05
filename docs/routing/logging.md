# Logging

Per-request structlog events:

- `route.matched`: `rule`, `endpoint`, `model`, `mode`
- `route.unmatched`: `endpoint`, `model`, `message`
- `route.dispatch_error`: `rule`, `endpoint`, `error`

Per-startup events:

- `routing.passthrough_body_touch`: body-rewrite + passthrough warning
