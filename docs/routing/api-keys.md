# API-key handling

`api_key_env` is the name of an environment variable, **not** the key
itself.

- **translate mode**: the dispatcher reads `os.environ[api_key_env]` and
  passes it to litellm via the `api_key=` kwarg. Lets one provider use
  multiple keys (e.g. tier-routing) by declaring separate rules with
  different env vars.
- **passthrough mode**: when the inbound request has neither
  `Authorization` nor `x-api-key`, the dispatcher injects
  `x-api-key: <env value>` into the forwarded headers. Headers do not
  participate in the prompt-cache hash, so this is safe.

A missing or empty env var produces `503 dispatch_error` at request
time, not config-load time (env state can change between deploys).

When a rule's `action` declares `provider:` but omits `api_key_env` /
`base_url`, those fields are inherited from the matching entry in the
top-level `providers:` block. This keeps third-party openai-compatible
upstreams (Vultr, hosted vLLM, etc.) working with concise rules, without
the inheritance, dispatch silently falls through to LiteLLM's per-provider
defaults (e.g. `OPENAI_API_KEY` against `api.openai.com`), producing a
401 from a totally unrelated upstream. Explicit values on the action
always win over the provider config.
