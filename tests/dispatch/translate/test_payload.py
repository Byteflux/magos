"""Process-wide LiteLLM toggle and shared payload behaviour."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_translate_module_enables_litellm_drop_params() -> None:
    """Importing ``magos.dispatch.translate`` must flip ``litellm.drop_params`` to True.

    Cross-shape translation (Anthropic <-> OpenAI) routinely sends params one
    side supports and the other does not. ``context_management`` from Claude
    Code on an upstream routed via ``custom_openai`` is the canary: without
    drop_params LiteLLM raises ``UnsupportedParamsError`` and the request
    fails before reaching the provider.
    """
    import litellm  # noqa: PLC0415

    import magos.dispatch.translate  # noqa: F401, PLC0415

    assert litellm.drop_params is True
