# Tests

- **Markers**: `unit`, `integration`, `e2e` are declared (and
  enforced via `--strict-markers` in `pyproject.toml`), but only ~8 of
  ~33 test files apply them. Selecting via `-m unit` runs a strict
  subset, not "all unit tests"; most tests are unmarked. Run all with
  `uv run pytest`; default config does not skip e2e by marker, but…
- **E2E gate**: most e2e tests require `MAGOS_E2E=1` and skip by
  default (provider creds, network).
- **E2E config**: when `MAGOS_E2E=1`, e2e tests load the shipped
  `magos.example.yaml` (operator-grade routing). Unit/integration tests
  use `tests/fixtures/magos.test.yaml`.
- **`tests/conftest.py`** force-imports `sentence_transformers` at
  session start to dodge a Windows pyarrow native-load-order bug
  triggered transitively by `mitmproxy.http`. **Don't remove this**;
  it looks like dead code, isn't. See [headroom/pipeline.md](../headroom/pipeline.md)
  "CacheAligner" for the full bisection.
- **Test app construction**: tests call
  `build_api(routing=..., registry=...)` to inject config without a
  YAML round-trip. `build_api` accepts both kwargs
  (`api/build.py`). The
  `app.state.{routing,refresher,registry_config}` slots are designed
  for direct replacement too, but no current test exercises that
  path.
- **Completion mocking**: tests use FastAPI's `dependency_overrides`
  against all four DI seams:
  `get_completion`, `get_anthropic_messages_completion`,
  `get_responses_completion`, `get_count_tokens_completion`. The
  shared TestClient factory lives in `tests/api/_helpers.py`;
  per-endpoint files (`test_messages.py`, `test_chat_completions.py`,
  `test_count_tokens.py`, `test_responses.py`) wire each completion
  override.
