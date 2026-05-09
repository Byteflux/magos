# Logging

Per-request log fields are split between two layers:

- **Bound via `structlog.contextvars`** in `service.request.process` after
  the route resolves: `rule`, `gateway`, `endpoint`, `model`. Plus
  `request_id` bound earlier in `api.run.run_endpoint`. Every per-request
  log line picks these up automatically through `merge_contextvars`, so
  individual call sites can stay terse.
- **Explicit kwargs** on each event for the data unique to that step
  (e.g. `error`, `dispatch_model`, `tokens_before`/`tokens_after`).

Per-request structlog events:

- `route.matched` (DEBUG): no explicit fields; routing context comes from
  `contextvars`. Demoted from INFO to keep the per-request INFO line count
  at one (`egress.usage`) for successful requests.
- `route.unmatched` (INFO): `message`
- `route.dispatch_error` (WARN): `error` -- emitted once at the dispatch
  catch site; `_render_route_error` no longer re-emits.

Per-startup events:

- `routing.passthrough_body_touch` (DEBUG): body-rewrite + passthrough warning
