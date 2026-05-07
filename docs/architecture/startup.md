# Startup

## Startup order

Two phases: `build_api()` (synchronous, builds the FastAPI app) and
`lifespan()` (async, runs once when uvicorn starts the app). The
lifespan is a thin runner over an ordered list of `LifespanComponent`
objects (`api/lifespan/`); each component implements
`start()` / `stop()` and is run / unwound via an `AsyncExitStack`.

**`magos.serve.serve_async()`, top-level orchestrator:**

1. Resolve config path: `--config` flag → `MAGOS_CONFIG_PATH` →
   `$MAGOS_HOME/magos.yaml` (default `~/.magos/magos.yaml`).
   `load_full_config()` parses routing + registry + ingress blocks.
2. Resolve FastAPI bind via `resolve_bind(settings, http_cfg)`:
   `MAGOS_HOST` env > `ingress.http.host` yaml > schema default
   (`127.0.0.1`). Same for port.
3. Build the FastAPI app via `build_api(routing=..., registry=...)`
   and the uvicorn `Server`.
4. Start the FastAPI task; wait on `Server.started` (poll, no event
   surface) so the lifespan completes before any ingress accepts.
5. If `ingress.mitm.enabled` and `intercept_hosts` non-empty:
   install the structlog bridge, build the `DumpMaster` via
   `build_proxy`, start the mitm task.
6. `asyncio.wait(..., FIRST_COMPLETED)`: when one task ends, signal
   the other to shut down (uvicorn `should_exit`, mitm `shutdown()`),
   then surface any exception.

**`magos.api.build.build_api()`, sync, builds app object:**

1. Stash on `app.state`: `routing` (RoutingConfig), `registry_config`
   (RegistryYaml), `refresher` (Refresher | None, None when
   `providers:` is empty).
2. Mount `/metrics` endpoint via `telemetry.metrics` if
   `MAGOS_METRICS_ENABLED=1`.
3. Mount `/admin/registry/*` endpoints (`api/admin.py`) if a
   Refresher exists.
4. Register the four POST handlers + three auxiliary GET/DELETE
   handlers (`api/handlers.py`) for `/v1/responses/{id}*`.
5. Register the `GET /v1/models` registry-backed endpoint
   (`api/models.py`).

**`lifespan()`, async, runs at startup.** Each step is a separate
`LifespanComponent`; they are started in order and stopped in reverse
order via `AsyncExitStack`:

1. **`KompressBackendOverride`**: only if
   `MAGOS_KOMPRESS_BACKEND=pytorch`. Replaces
   `headroom.transforms.kompress_compressor._is_onnx_available` with a
   False-stub. See [headroom/backend.md](../headroom/backend.md).
2. **`MetricsMeter`**: only if `MAGOS_METRICS_ENABLED=1`. Wires the
   Prometheus exporter into the global meter provider; the `/metrics`
   endpoint mounted in `build_api` reads from this.
3. **`MagosCompressionWarmup`**: only if any rule uses `compress`.
   Calls `magos.compression.prebuild_from_routing(cfg)`, which builds
   a `TransformPipeline` per `(PipelineConfig, provider)` pair and
   then `eager_warmup`s every unique transform (loads Kompress,
   Magika, tree-sitter parsers, etc.).
4. **`KompressPreload`**: independent of step 3. Fires only if any
   rule uses `compress` AND `MAGOS_KOMPRESS_PRELOAD=1` (the default).
   Schedules `_preload_kompress_model` as a background task via
   `asyncio.create_task` (runs in `asyncio.to_thread`); doesn't block
   startup. Cancelled on shutdown.
5. **`RegistryRefresher`**: only if a Refresher was constructed.
   Loads `models.json`, kicks off per-provider boot-discovery tasks,
   schedules periodic refresh.

Shutdown unwinds 5 → 4: stop Refresher, cancel preload task. Steps
1-3 have no shutdown work.

## Registry single-writer invariant

**Only `registry/refresher.py` writes `models.json`.** Reads happen
from anywhere (CLI, admin endpoints, routing engine). The store
(`registry/store.py`) does atomic temp-file + rename, but cross-process
write races are not guarded. If you need to mutate the registry from a
new code path, route through the Refresher (e.g. add a method that
schedules a refresh tick); don't call `store.write()` directly.

CLI commands that look like writes (`magos models refresh`,
`magos models prune`) hit `/admin/registry/*` HTTP endpoints, which
call into the running Refresher. They don't touch `models.json`
directly.

## `litellm.drop_params = True` is process-global

Set once at module import in `dispatch/translate/payload.py`. LiteLLM
silently drops any parameter the destination provider doesn't accept
(e.g. `reasoning_effort` against a non-reasoning model). This is
**not per-rule** and not toggleable.

When debugging "param X isn't reaching provider Y": this is the first
suspect. Confirm by checking LiteLLM's per-provider supported-params
list, not by reading magos's dispatch code.
