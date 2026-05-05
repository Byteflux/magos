"""Egress: how requests leave magos.

Three execution paths chosen by :mod:`magos.egress.dispatch` based on the
``RouteDecision`` it receives:

- :mod:`magos.egress.passthrough`: byte-exact HTTP forwarding (preserves
  Anthropic prompt-cache hashes and OAuth bearer shapes).
- :mod:`magos.egress.translate`: wire-shape translation via the LiteLLM
  SDK (``anthropic_messages`` / ``acompletion`` / ``aresponses``).
- :mod:`magos.egress.tokens`: count-tokens dispatch via
  ``litellm.acount_tokens``.

Auth-header injection logic for passthrough mode is in
:mod:`magos.egress.dispatch` (extracted to :mod:`magos.egress.auth` in a
later phase). The standalone egress observer addon
(:mod:`magos.egress.observer`) is loaded by the in-process mitmproxy
master alongside the ingress addon.
"""
