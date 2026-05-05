# Request flow

## Process topology

Magos runs as a **single Python process**. By default it listens only
on the FastAPI port and clients hit it directly. When
`ingress.mitm.enabled` is true in `magos.yaml`, an embedded
`mitmproxy` listener runs alongside FastAPI on the same asyncio loop
(`magos.serve`) so a client pointed at `HTTPS_PROXY=...` sees TLS
interception too. See `docs/ingress.md` for the operator guide.

```
                       ┌─────────────────── single magos process ──────────────────┐
client (direct) ──────▶│ FastAPI (uvicorn) :6246                                   │──▶ provider API
                       │   ingress.http → routing.engine → egress.dispatch         │
                       │                                                           │
client (HTTPS_PROXY)──▶│ mitmproxy DumpMaster :6247  (optional)                    │
                       │   ├── ingress.mitm.addon (TLS terminate + rewrite to :6246)│
                       │   ├── egress.observer    (egress logging)                 │
                       │   └── ingress.mitm.log_bridge (mitmproxy log → structlog) │
                       └───────────────────────────────────────────────────────────┘
```

Three roles for the mitmproxy machinery:

1. **In-process ingress (when configured)**: `magos.ingress.mitm`
   rewrites incoming intercepted requests to the FastAPI loopback.
   Same process, same asyncio loop; `magos.serve.serve_async` gathers
   both as named tasks and shuts both down on first-task-done.
2. **In-process egress observer**: `magos.egress.observer` is loaded
   by the in-process master alongside the ingress addon, logging
   outbound LLM provider traffic when magos's own outbound transits
   mitmproxy (which it doesn't by default; see `docs/ingress.md`
   "Loop hazard"). Can also be run standalone via
   `mitmdump -s src/magos/egress/observer.py --listen-port 8080` if
   the operator prefers an out-of-process observer.
3. **Transitive runtime dependency**: `mitmproxy.http` is imported
   directly by `tests/ingress/mitm/test_addon.py`, and transitively by
   `tests/egress/test_observer.py` (via `magos.egress.observer`),
   regardless of whether ingress is enabled. That import triggers the
   Windows pyarrow load-order
   workaround in `tests/conftest.py`. See [headroom/pipeline.md](../headroom/pipeline.md)
   "CacheAligner".

When ingress is **disabled**, both `magos.ingress.mitm` and
`magos.egress.observer` are dormant. Routing-layer bugs are still

never in either; routing always lives in `magos.routing` regardless
of how the request entered.

## Request lifecycle

Per request, the FastAPI app does this:

1. **Inbound parsing** (`ingress/http/run.py`). Body parsed to dict
   (or kept as `raw_body` bytes); inbound headers filtered through
   `_BLOCKED_FORWARD_HEADERS` in `ingress/http/headers.py`, which drops
   hop-by-hop (RFC 7230) plus content-shaping headers
   (`content-length`, `content-encoding`, `host`, etc.). The filtered
   headers are lowercased into a dict.
2. **Construct `RoutedRequest`** (`routing/request.py`). Frozen
   dataclass carrying: endpoint kind, **templated path** (e.g.
   `/v1/responses/{id}`), **actual path** (e.g.
   `/v1/responses/resp_abc123`), `body` (mutable dict view),
   `raw_body` (bytes), `body_dirty=False`, lowercased headers.
3. **Route** (`routing/engine.py:route`). Applies pre-rewrites, walks
   `rules` top-to-bottom, returns the first match's `RouteDecision` or
   a `RouteError`. If no rule matches, **auto-routing** (in
   `routing/auto_route.py`) consults the registry: exact
   `<provider>/<raw_id>` lookup, falling back per
   `on_unknown_model: error|passthrough`. **Explicit rules always win
   over the registry**; the registry only catches misses.
4. **Dispatch** (`egress/dispatch.py`). Decision enum drives one of
   nine branches:

| Mode          | Endpoint                       | Streaming | Implementation                       |
|---------------|--------------------------------|-----------|--------------------------------------|
| `count_tokens`| `/v1/messages/count_tokens`    | n/a       | `egress.tokens.count_tokens` (litellm) |
| `passthrough` | any of the six (incl. auxiliary GET/DELETE) | non-stream | `egress.passthrough.call_passthrough` |
| `passthrough` | any of the six (incl. auxiliary GET/DELETE) | stream     | `egress.passthrough.stream_passthrough` |
| `translate`   | `/v1/messages`                 | non-stream| `proxy_translate` + `anthropic.ADAPTER`         |
| `translate`   | `/v1/messages`                 | stream    | `stream_translate` + `anthropic.ADAPTER`        |
| `translate`   | `/v1/chat/completions`         | non-stream| `proxy_translate` + `openai_chat.ADAPTER`       |
| `translate`   | `/v1/chat/completions`         | stream    | `stream_translate` + `openai_chat.ADAPTER`      |
| `translate`   | `/v1/responses`                | non-stream| `proxy_translate` + `openai_responses.ADAPTER` |
| `translate`   | `/v1/responses`                | stream    | `stream_translate` + `openai_responses.ADAPTER` |

All six translate cells delegate to the generic `proxy_translate` /
`stream_translate` runners in `egress/translate/runner.py`. The
per-shape `TranslateAdapter` constant (`ADAPTER` in `anthropic.py`,
`openai_chat.py`, `openai_responses.py`) supplies the LiteLLM SDK
callable, SSE framer, and payload coercion for that shape. The
adapters are wired into `TRANSLATE_HANDLERS` in
`egress/translate/__init__.py`, keyed by endpoint.

The full endpoint set (`routing/request.py`): `/v1/messages`,
`/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/responses`,
`/v1/responses/{id}`, `/v1/responses/{id}/input_items`. **Translate
mode requires POST** (enforced in `egress/dispatch.py`); auxiliary
GET/DELETE endpoints must use `mode: passthrough`.

`GET /v1/models` (`ingress/http/models.py`) sits beside the routed
endpoints but skips the rule engine entirely: it lists registry
entries (`app.state.refresher.state.entries`, deprecated entries
omitted, sorted by `namespaced_id`) in OpenAI shape by default, or
Anthropic shape when the request carries `anthropic-version` or
`x-api-key`. Returns an empty list when the registry feature is
dormant.

5. **Response**. Translate mode lets LiteLLM regenerate
   `content-type` / `content-length` / `content-encoding`. Passthrough
   forwards response bytes verbatim.

   `egress.usage` log: every successful response (translate or
   passthrough, streaming or not, all three shapes) emits a single
   `egress.usage` event with normalised `input` / `output` /
   `cache_read` / `cache_write` token counts. Field mapping lives in
   `egress/usage.py`; streaming paths use a byte-level SSE tap
   (`tap_stream`) that forwards bytes verbatim while accumulating the
   terminal-event usage block. ``cache_write`` is Anthropic-only;
   OpenAI shapes always report 0.

## The `body_dirty` contract

`RoutedRequest.body_dirty` (`routing/request.py`) is a single bool
that decides whether passthrough re-serialises JSON or sends bytes
verbatim:

```python
# egress/dispatch.py
body_bytes = req.raw_body if not req.body_dirty else json.dumps(dict(req.body)).encode()
```

**Any rewrite primitive that mutates `body` MUST set
`body_dirty=True`.** Today: `SetModel` (`routing/rewrites/model.py`),
`JqPatch` (`routing/rewrites/jq_patch.py`), `Compress`
(`routing/rewrites/compress/`, both token+cache modes) all do this.
Header-only rewrites (`routing/rewrites/headers.py`) leave it
untouched.

Why this matters: Anthropic prompt-cache hashes are computed over the
**exact bytes** of the prefix up to a `cache_control` breakpoint. Any
JSON whitespace shift, key reordering, or float-formatting change
silently invalidates the cache. The `raw_body` short-circuit preserves
byte-exactness when no rewrite touched the body. If you add a new
body-mutating rewrite op and forget the flag, passthrough sends the
*pre-rewrite* bytes: a silent correctness bug, not a crash.

## Passthrough is byte-exact on purpose

`magos.egress.passthrough` is a deliberate non-LiteLLM path. Two
correctness reasons:

1. **Anthropic prompt cache.** Hash stability requires byte-identical
   prefix. LiteLLM normalises (re-serialises) bodies; passthrough must
   not.
2. **Anthropic OAuth tokens.** `sk-ant-oat...` (Claude-Code style) need
   `Authorization: Bearer <token>` plus
   `anthropic-beta: oauth-2025-04-20`. LiteLLM's auth handling is
   keyed off `x-api-key`-style flows and would not preserve this
   shape.

**Don't** add JSON normalisation, header reordering, or "cleanup" to
`egress/passthrough.py`. Don't route it through LiteLLM "for
consistency".
