# Logging

Per-request structlog events:

- `route.matched` (INFO): `rule`, `endpoint`, `model`, `gateway`
- `route.unmatched` (INFO): `endpoint`, `model`, `message`
- `route.dispatch_error` (WARN): `rule`, `endpoint`, `error` -- emitted once
  at the dispatch catch site; `_render_route_error` no longer re-emits.

Per-startup events:

- `routing.passthrough_body_touch` (DEBUG): body-rewrite + passthrough warning
