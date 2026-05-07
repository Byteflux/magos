"""OpenAI-shape `GET /v1/models` adapter (OpenAI, vLLM, SGLang, LM Studio).

Endpoint returns just `id`; merge fills the rest. `litellm_provider`
defaults to `openai`; local inference backends typically set
`hosted_vllm` or similar.
"""

from __future__ import annotations

from magos.registry.discovery._auth import bearer_auth
from magos.registry.discovery.base import JsonListAdapter

_DEFAULT_BASE_URL = "https://api.openai.com"
_DEFAULT_LITELLM_PROVIDER = "openai"


class OpenAIAdapter(JsonListAdapter):
    """Calls `GET {base_url}/v1/models` and maps `data[*].id` to entries."""

    name = "openai"
    default_base_url: str | None = _DEFAULT_BASE_URL

    _path_suffix = "/v1/models"
    _data_field = "data"
    _default_litellm_provider = _DEFAULT_LITELLM_PROVIDER
    _auth_headers = staticmethod(bearer_auth)
