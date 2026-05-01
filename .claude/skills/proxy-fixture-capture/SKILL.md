---
name: proxy-fixture-capture
description: Capture a live Anthropic <-> OpenAI request/response exchange through the magos proxy and turn it into a sanitized golden fixture under tests/fixtures/translation/<case>/. Use when adding a new translator case, reproducing a translation bug, or pinning behavior the goldens do not yet cover.
---

# Proxy Fixture Capture

Operator workflow for turning one real upstream exchange into a four-file golden case
that the translation and proxy test suites discover automatically.

## When to use

- Adding a new translator case (tools, images, streaming, error responses, etc.)
- Reproducing a bug seen against a real upstream
- Pinning a behavior the existing goldens do not exercise

Skip if you can hand-author the four JSON files faithfully without a live capture.
A capture is only worth it when the upstream's exact shape matters.

## Prerequisites

- A working upstream that litellm can reach (OpenAI key, Anthropic-via-OpenAI gateway, etc.)
- The case has a short, snake_case name (`tool_use`, `streaming_text`, `image_input`, ...)
- You have already decided what the translator *should* do for this case. The goldens
  pin the contract; capture only confirms a shape, it does not decide a contract.

## Steps

### 1. Pick a case name and create the directory

```bash
CASE=tool_use
mkdir -p tests/fixtures/translation/$CASE
```

### 2. Run mitmdump with the addon and dump flows to disk

```bash
uv run mitmdump -s src/magos/addon.py --listen-port 8080 -w /tmp/magos-capture.flow
```

Leave it running. `-w` writes every flow to a binary log you will read in step 4.

### 3. Send a single request through the proxy

Point a client (curl, Anthropic SDK, etc.) at `http://127.0.0.1:8080/v1/messages` with
the request you want to capture. Send exactly one. Stop mitmdump (`Ctrl+C`) once the
response comes back.

### 4. Extract the four payloads

Use a one-shot Python script (do not commit it; throwaway):

```python
from mitmproxy import io
import json
from pathlib import Path

CASE = Path("tests/fixtures/translation/tool_use")
with open("/tmp/magos-capture.flow", "rb") as fh:
    flows = list(io.FlowReader(fh).stream())

# Pick the one /v1/messages flow (the addon short-circuits, so request body is the
# Anthropic shape and response body is the Anthropic shape it produced).
flow = next(f for f in flows if f.request.path == "/v1/messages")
anthropic_request = json.loads(flow.request.get_text())
anthropic_response = json.loads(flow.response.get_text())

(CASE / "anthropic_request.json").write_text(json.dumps(anthropic_request, indent=2) + "\n")
(CASE / "anthropic_response.json").write_text(json.dumps(anthropic_response, indent=2) + "\n")
```

The OpenAI-shaped request and response go through litellm in-process and are not on the
wire. To capture those, temporarily log them from `magos.proxy.proxy_anthropic_messages`
(both `openai_request` and the dumped response) and write them out the same way. Remove
the logging once the case is captured.

### 5. Sanitize before saving

Run through the checklist below and edit the four JSON files in place:

- [ ] Replace any real model identifier with a synthetic one if the test should not
      depend on it (`claude-sonnet-4-5` is fine to keep)
- [ ] Replace `id` fields with deterministic synthetic values:
      `chatcmpl-magos-<case>` for OpenAI, `msg_magos_<case>` for Anthropic
- [ ] Replace `created` with a fixed unix epoch (`1730000000` + a per-case offset)
- [ ] Strip or redact any `Authorization`, `x-api-key`, `anthropic-version`, or other
      header values that may have leaked into the body
- [ ] Replace any user-identifying content (names, emails, internal URLs) with
      synthetic placeholders
- [ ] Confirm `usage.prompt_tokens` / `usage.completion_tokens` / `usage.total_tokens`
      are internally consistent
- [ ] Confirm `finish_reason` -> `stop_reason` mapping in `_FINISH_TO_STOP` already
      handles this case; if not, expand the table *and* the `OpenAIFinishReason` /
      `AnthropicStopReason` literals in `src/magos/translation.py`

### 6. Run the suites

```bash
uv run pytest -m unit
uv run pytest -m integration
```

The translator round-trip and the proxy pipeline both auto-discover the new case.
Two outcomes:

- **All green.** The translator already supports the new shape; the case is now pinned.
- **Failures.** The translator needs to grow. Expand pydantic models, mappings, or
  flatten logic until the goldens go green. Do not modify the goldens to match the
  translator. The goldens are the contract.

### 7. Delete the throwaway capture artifacts

```bash
rm -f /tmp/magos-capture.flow
```

Do not commit `.flow` files, capture scripts, or any debug logging added in step 4.

## Output structure

```
tests/fixtures/translation/<case>/
  anthropic_request.json    # client -> proxy
  openai_request.json       # proxy -> upstream (via litellm)
  openai_response.json      # upstream -> proxy
  anthropic_response.json   # proxy -> client
```

All four files are required. Missing any one breaks `_case_dirs()` discovery in
`tests/test_translation.py` and `tests/test_proxy.py`.

## What this skill does *not* do

- It does not capture streaming responses. SSE handling is a separate code path and
  needs its own fixture format.
- It does not exercise litellm routing logic. The captured `openai_request.json` is
  what the translator emitted, not what hit the wire after litellm rewrites it.
- It does not run the upstream for you. You bring the credentials.
