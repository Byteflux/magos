"""``/v1/messages`` translate path via ``litellm.anthropic_messages``.

Anthropic-shape in, Anthropic-shape out across upstreams. Non-Anthropic
dispatch pre-translates ``output_config`` to OpenAI extras and drops
unknown Anthropic-only fields (``litellm.drop_params`` doesn't catch
fields LiteLLM doesn't recognize). See ``docs/architecture/translation.md``.

Three concerns split across siblings:

- :mod:`translation` — body massaging: ``output_config`` translation,
  ``additionalProperties: {}`` coercion, unknown-field stripping.
- :mod:`dispatch` — chooses ``litellm.anthropic_messages`` (Anthropic
  upstream) vs ``litellm.acompletion`` + adapter translation.
- :mod:`adapter` — the assembled ``TranslateAdapter`` + the
  ``set_model_*`` / ``stream_bytes_iter`` hooks the runner calls.
"""

from __future__ import annotations

from .adapter import ADAPTER
from .dispatch import _dispatch_anthropic_messages

__all__ = ["ADAPTER", "_dispatch_anthropic_messages"]
