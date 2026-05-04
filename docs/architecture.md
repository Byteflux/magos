# Architecture

Orientation map for engineers landing in magos cold. Covers the
non-obvious cross-cutting facts a fresh contributor would otherwise
have to reconstruct from reading 6+ files. Verified against the source
on the dates the references resolve.

For component-specific deep-dives:
- Routing rule grammar → `docs/routing.md`
- Registry lifecycle → `docs/registry.md`
- Headroom integration → `docs/headroom.md`

## Process topology

Magos runs as a **single Python process**. By default it listens only
on the FastAPI port and clients hit it directly. When
`server.ingress.enabled` is true in `magos.yaml`, an embedded
`mitmproxy` listener runs alongside FastAPI on the same asyncio loop
(`magos.serve`) so a client pointed at `HTTPS_PROXY=...` sees TLS
interception too — see `docs/ingress.md` for the operator guide.

```
                       ┌─────────────────── single magos process ──────────────────┐
client (direct) ──────▶│ FastAPI (uvicorn) :8000                                   │──▶ provider API
                       │   ingress.http → routing.engine → egress.dispatch         │
                       │                                                           │
client (HTTPS_PROXY)──▶│ mitmproxy DumpMaster :8080  (optional)                    │
                       │   ├── ingress.mitm.addon (TLS terminate + rewrite to :8000)│
                       │   ├── egress.observer    (egress logging)                 │
                       │   └── ingress.mitm.log_bridge (mitmproxy log → structlog) │
                       └───────────────────────────────────────────────────────────┘
```

Three roles for the mitmproxy machinery:

1. **In-process ingress (when configured)**: `magos.ingress.mitm`
   rewrites incoming intercepted requests to the FastAPI loopback.
   Same process, same asyncio loop — `magos.serve.serve_async` gathers
   both as named tasks and shuts both down on first-task-done.
2. **In-process egress observer**: `magos.egress.observer` is loaded
   by the in-process master alongside the ingress addon, logging
   outbound LLM provider traffic when magos's own outbound transits
   mitmproxy (which it doesn't by default — see `docs/ingress.md`
   "Loop hazard"). Can also be run standalone via
   `mitmdump -s -m magos.egress.observer` if the operator prefers an
   out-of-process observer.
3. **Transitive runtime dependency**: `mitmproxy.http` is imported
   by `tests/ingress/mitm/test_addon.py` and `tests/egress/test_observer.py`
   (and therefore the test suite) regardless of whether ingress is
   enabled. That import triggers the Windows pyarrow load-order
   workaround in `tests/conftest.py` — see `docs/headroom.md`
   "CacheAligner".

When ingress is **disabled**, both `magos.ingress.mitm` and
`magos.egress.observer` are dormant. Routing-layer bugs are still
never in either; routing always lives in `magos.routing` regardless
of how the request entered.

## Request lifecycle

Per request, the FastAPI app does this:

1. **Inbound parsing** (`ingress/http/run.py`). Body parsed to dict
   (or kept as `raw_body` bytes); inbound headers filtered through
   `_BLOCKED_FORWARD_HEADERS` in `ingress/http/headers.py` — drops
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
   eight branches:

| Mode          | Endpoint                       | Streaming | Implementation                       |
|---------------|--------------------------------|-----------|--------------------------------------|
| `count_tokens`| `/v1/messages/count_tokens`    | n/a       | `egress.tokens.count_tokens` (litellm) |
| `passthrough` | any of the six (incl. auxiliary GET/DELETE) | non-stream | `egress.passthrough.call_passthrough` |
| `passthrough` | any of the six (incl. auxiliary GET/DELETE) | stream     | `egress.passthrough.stream_passthrough` |
| `translate`   | `/v1/messages`                 | non-stream| `egress.translate.anthropic.proxy`   |
| `translate`   | `/v1/messages`                 | stream    | `egress.translate.anthropic.stream`  |
| `translate`   | `/v1/chat/completions`         | non-stream| `egress.translate.openai_chat.proxy` |
| `translate`   | `/v1/chat/completions`         | stream    | `egress.translate.openai_chat.stream`|
| `translate`   | `/v1/responses`                | both      | `egress.translate.openai_responses`  |

The full endpoint set (`routing/request.py`): `/v1/messages`,
`/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/responses`,
`/v1/responses/{id}`, `/v1/responses/{id}/input_items`. **Translate
mode requires POST** (enforced in `egress/dispatch.py`); auxiliary
GET/DELETE endpoints must use `mode: passthrough`.

5. **Response**. Translate mode lets LiteLLM regenerate
   `content-type` / `content-length` / `content-encoding`. Passthrough
   forwards response bytes verbatim.

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
(`routing/rewrites/compress.py`, both token+cache modes) all do this.
Header-only rewrites (`routing/rewrites/headers.py`) leave it
untouched.

Why this matters: Anthropic prompt-cache hashes are computed over the
**exact bytes** of the prefix up to a `cache_control` breakpoint. Any
JSON whitespace shift, key reordering, or float-formatting change
silently invalidates the cache. The `raw_body` short-circuit preserves
byte-exactness when no rewrite touched the body. If you add a new
body-mutating rewrite op and forget the flag, passthrough sends the
*pre-rewrite* bytes — a silent correctness bug, not a crash.

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

## Auth-header injection

`magos.egress.auth` (extracted from `egress/dispatch.py`) injects an
outbound auth header iff:

- `action.mode == "passthrough"` AND
- The inbound request lacks both `Authorization` and `x-api-key`, AND
- The matched rule's `action.api_key_env` resolves to a non-empty env
  var.

(Translate mode reads `action.api_key_env` separately and hands it to
LiteLLM as the `api_key` kwarg; it does not write headers.)

**Resolution order for the header shape (highest first):**

1. **OAuth detection** — value starts with `sk-ant-oat` →
   `Authorization: Bearer <token>` + `anthropic-beta: oauth-2025-04-20`
   (overrides everything; api.anthropic.com 401s on `x-api-key` for
   that credential class).
2. **Per-rule `action.auth_header` override** — explicit
   `x-api-key` or `bearer` value on the rule.
3. **Provider default** — `provider: anthropic` → `x-api-key`,
   everything else → `Authorization: Bearer`.

OAuth-token detection lives in `egress/auth.py` and the registry-side
discovery counterpart in `registry/discovery/anthropic.py`. If you
change one, change both or the registry's discovery call will fail
against an OAuth-only account.

## Startup order

Two phases — `create_app()` (synchronous, builds the FastAPI app) and
`_lifespan()` (async, runs once when uvicorn starts the app).

**`magos.serve.serve_async()` — top-level orchestrator:**

1. Resolve config path: `--config` flag → `MAGOS_CONFIG_PATH` →
   `$MAGOS_HOME/magos.yaml` (default `~/.magos/magos.yaml`).
   `load_full_config()` parses routing + registry + server blocks.
2. Resolve FastAPI bind via `resolve_bind(settings, server_cfg)`:
   `MAGOS_HOST` env > `server.host` yaml > schema default
   (`127.0.0.1`). Same for port.
3. Build the FastAPI app via `create_app(routing=..., registry=...)`
   and the uvicorn `Server`.
4. Start the FastAPI task; wait on `Server.started` (poll, no event
   surface) so the lifespan completes before any ingress accepts.
5. If `server.ingress.enabled` and `intercept_hosts` non-empty:
   install the structlog bridge, build the `DumpMaster` via
   `build_ingress_master`, start the mitm task.
6. `asyncio.wait(..., FIRST_COMPLETED)`: when one task ends, signal
   the other to shut down (uvicorn `should_exit`, mitm `shutdown()`),
   then surface any exception.

**`magos.ingress.http.app.create_app()` — sync, builds app object:**

1. Stash on `app.state`: `routing` (RoutingConfig), `registry_config`
   (RegistryYaml), `refresher` (Refresher | None — None when
   `providers:` is empty).
2. Mount `/metrics` endpoint via `telemetry.metrics` if
   `MAGOS_METRICS_ENABLED=1`.
3. Mount `/admin/registry/*` endpoints (`ingress/http/admin.py`) if a
   Refresher exists.
4. Register the four POST handlers + three auxiliary GET/DELETE
   handlers (`ingress/http/handlers.py`) for `/v1/responses/{id}*`.

**`_lifespan()` — async, runs at startup:**

1. **Kompress backend monkey-patch** — only if
   `MAGOS_KOMPRESS_BACKEND=pytorch`. Replaces
   `headroom.transforms.kompress_compressor._is_onnx_available` with a
   False-stub. See `docs/headroom.md` "Forcing the Kompress backend".
2. **OTel MeterProvider configuration** — only if
   `MAGOS_METRICS_ENABLED=1`. Wires the Prometheus exporter into the
   global meter provider; the `/metrics` endpoint mounted in
   `create_app` reads from this.
3. **Headroom pipeline warmup** — only if any rule uses `compress`.
   Builds `TransformPipeline` (lazy thread-locked singleton inside
   Headroom).
4. **Kompress preload background task** — only inside the compress
   branch above, AND only if `MAGOS_KOMPRESS_PRELOAD=1` (the default).
   Async via `asyncio.to_thread`, doesn't block startup; cancelled on
   shutdown.
5. **Refresher startup** (`registry/refresher.py`) — only if a
   Refresher was constructed. Loads `models.json`, kicks off
   per-provider boot-discovery tasks, schedules periodic refresh.

Shutdown reverses 4 + 5: cancel preload task, stop Refresher.

## Registry single-writer invariant

**Only `registry/refresher.py` writes `models.json`.** Reads happen
from anywhere (CLI, admin endpoints, routing engine). The store
(`registry/store.py`) does atomic temp-file + rename, but cross-process
write races are not guarded. If you need to mutate the registry from a
new code path, route through the Refresher (e.g. add a method that
schedules a refresh tick) — don't call `store.write()` directly.

CLI commands that look like writes (`magos models refresh`,
`magos models prune`) hit `/admin/registry/*` HTTP endpoints, which
call into the running Refresher. They don't touch `models.json`
directly.

## `litellm.drop_params = True` is process-global

Set once at module import in `egress/translate/payload.py`. LiteLLM
silently drops any parameter the destination provider doesn't accept
(e.g. `reasoning_effort` against a non-reasoning model). This is
**not per-rule** and not toggleable.

When debugging "param X isn't reaching provider Y": this is the first
suspect. Confirm by checking LiteLLM's per-provider supported-params
list, not by reading magos's dispatch code.

## Header forwarding is three-level

| Stage                         | What's blocked                                           | Why                                              |
|-------------------------------|----------------------------------------------------------|--------------------------------------------------|
| Ingress inbound (`ingress/http/headers.py`) | RFC 7230 hop-by-hop + `host` / `content-length` / `content-encoding` / `accept-encoding` | Don't propagate transport-layer junk             |
| Pre-LiteLLM body shape (`egress/translate/payload.py`) | `content-type` / `content-length` / `content-encoding` / `accept-encoding` | LiteLLM regenerates these; overriding causes "unexpected keyword argument" errors at the SDK boundary |
| Pre-LiteLLM auth (`egress/translate/payload.py`) | `authorization` / `x-api-key` — **only when** the rule's `api_key` was resolved | Stops the inbound bearer from leaking into `extra_headers` and overriding the operator-chosen upstream key |
| Pre-passthrough               | nothing additional                                       | Byte-exact forwarding (cache hashes)             |

If a header you expect to see at the provider isn't arriving, check
all three stages.

## Anthropic-shape cross-provider translation

`/v1/messages` against a non-Anthropic upstream (e.g. an OpenAI-shaped
provider mapped via routing) goes through
`litellm.anthropic_messages`. LiteLLM accepts Anthropic-shape *in* and
emits Anthropic-shape *out* regardless of upstream provider, but two
preprocessing steps happen in `egress/translate/anthropic.py` first:

- **Anthropic-only fields stripped** for non-Anthropic upstreams
  (`_strip_anthropic_extras`): `context_management` and similar fields
  LiteLLM passes through as `**kwargs` and that the upstream provider
  doesn't understand.
- **`output_config.effort` → `reasoning_effort`** translation. Anthropic
  uses `output_config.effort` (`low|medium|high|xhigh|max`); OpenAI
  uses `reasoning_effort` (`low|medium|high`). Magos clamps
  `xhigh`/`max` → `high`.

If you add a new Anthropic-only field downstream, mirror it in the
strip list.

## Environment variables

Resolution order (highest first) for the routing config path:

1. `--config <path>` CLI flag
2. `MAGOS_CONFIG_PATH` env var
3. `$MAGOS_HOME/magos.yaml` (default `~/.magos/magos.yaml`)

`MAGOS_HOME` is a **bootstrap-only env var**: it has no settings field
on `MagosSettings`. It anchors defaults for `MAGOS_CONFIG_PATH` and
`models.json`, and is the resolution base for relative registry paths
(not CWD, not the yaml file's parent). The resolution helpers live in
`config/settings.py` (`magos_home()`) and `config/loader.py`
(`resolve_models_path`).

| Variable                     | Default       | Purpose                                                |
|------------------------------|---------------|--------------------------------------------------------|
| `MAGOS_HOME`                 | `~/.magos`    | Data dir; anchors config and registry paths           |
| `MAGOS_CONFIG_PATH`          | `$MAGOS_HOME/magos.yaml` | Routing config YAML                       |
| `MAGOS_HOST`                 | (unset)       | Override `server.host` from yaml; yaml default is `127.0.0.1` |
| `MAGOS_PORT`                 | (unset)       | Override `server.port` from yaml; yaml default is `8000` |
| `MAGOS_LOG_LEVEL`            | `INFO`        | structlog level                                        |
| `MAGOS_LOG_JSON`             | `0`           | `1` flips renderer to JSON                             |
| `MAGOS_LOG_COLOR`            | auto-TTY      | `0`/`1` overrides TTY autodetect                       |
| `MAGOS_OTEL_ENABLED`         | `0`           | `1` ships OTel spans                                   |
| `MAGOS_OTEL_ENDPOINT`        | unset         | OTLP endpoint when OTel enabled                        |
| `MAGOS_KOMPRESS_BACKEND`     | `auto`        | `pytorch` forces PyTorch path (CUDA/MPS/CPU)           |
| `MAGOS_KOMPRESS_PRELOAD`     | `1`           | Preload Kompress weights at startup (only fires when a `compress` rule exists). Set to `0` for lazy on-demand load |
| `MAGOS_ACCESS_LOG`           | `1`           | `0` silences uvicorn access log                        |
| `MAGOS_METRICS_ENABLED`      | `0`           | `1` exposes Prometheus `/metrics`                      |
| `MAGOS_MODELS_PATH`          | `$MAGOS_HOME/models.json` | Override registry persistence path         |

Removed env vars (warn on startup, now in YAML —
`config/settings.py`):
`MAGOS_ANTHROPIC_PASSTHROUGH_ENABLED`,
`MAGOS_ANTHROPIC_UPSTREAM_URL`,
`MAGOS_COUNT_TOKENS_PASSTHROUGH_PROVIDERS`.

## Tests

- **Markers**: `unit`, `integration`, `e2e` are declared (and
  enforced via `--strict-markers` in `pyproject.toml`), but only ~8 of
  ~33 test files apply them. Selecting via `-m unit` runs a strict
  subset, not "all unit tests" — most tests are unmarked. Run all with
  `uv run pytest`; default config does not skip e2e by marker, but…
- **E2E gate**: most e2e tests require `MAGOS_E2E=1` and skip by
  default (provider creds, network).
- **E2E config**: when `MAGOS_E2E=1`, e2e tests load the shipped
  `magos.example.yaml` (operator-grade routing). Unit/integration tests
  use `tests/fixtures/magos.test.yaml`.
- **`tests/conftest.py`** force-imports `sentence_transformers` at
  session start to dodge a Windows pyarrow native-load-order bug
  triggered transitively by `mitmproxy.http`. **Don't remove this**;
  it looks like dead code, isn't. See `docs/headroom.md` "CacheAligner"
  for the full bisection.
- **Test app construction**: tests call
  `create_app(routing=..., registry=...)` to inject config without a
  YAML round-trip. `create_app` accepts both kwargs
  (`ingress/http/app.py`). The
  `app.state.{routing,refresher,registry_config}` slots are designed
  for direct replacement too (per `ingress/http/app.py`'s docstring),
  but no current test exercises that path.
- **Completion mocking**: tests use FastAPI's `dependency_overrides`
  against all four DI seams:
  `get_completion`, `get_anthropic_messages_completion`,
  `get_responses_completion`, `get_count_tokens_completion`. The
  shared TestClient factory lives in `tests/ingress/http/_helpers.py`;
  per-endpoint files (`test_messages.py`, `test_chat_completions.py`,
  `test_count_tokens.py`, `test_responses.py`) wire each completion
  override.

## Subtleties worth not forgetting

- **Routing rules always beat the registry.** Auto-routing is a
  fallback for unmatched requests, not a parallel layer.
- **`models.json` has one writer (Refresher).** Don't add direct writes.
- **`body_dirty` is mandatory for body-mutating rewrites.** Forgetting
  it sends pre-rewrite bytes through passthrough.
- **Passthrough is byte-exact for cache + OAuth reasons.** No
  normalisation. No LiteLLM round-trip.
- **`litellm.drop_params=True` is global.** Suspect this first when a
  param vanishes.
- **mitmproxy is opt-in for ingress.** When `server.ingress.enabled`
  is false (the default), mitmproxy is completely dormant — no
  listener, no addon hooks running. When enabled, it terminates TLS
  for allowlisted hosts and rewrites to FastAPI loopback; routing
  itself still happens in FastAPI.
- **`sentence_transformers` preload in conftest is load-bearing**
  (Windows-only crash, but the preload is unconditional so CI/Linux
  pays nothing).
- **Headroom `_is_onnx_available` is monkey-patched at startup** when
  `MAGOS_KOMPRESS_BACKEND=pytorch`. Looks weird, is intentional.
- **Anthropic OAuth (`sk-ant-oat`) auth shape lives in two places** —
  `egress/auth.py` (proxy-side injection) and
  `registry/discovery/anthropic.py` (discovery). Keep them in sync.
- **Header blocking is three-level** — ingress inbound, pre-LiteLLM
  body shape, and pre-LiteLLM auth (conditional on rule-resolved
  `api_key`). All three must be checked when a header isn't reaching
  the provider.
