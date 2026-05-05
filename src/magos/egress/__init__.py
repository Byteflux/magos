"""Egress: how requests leave magos.

Three execution paths chosen by :mod:`magos.egress.dispatch`:
:mod:`magos.egress.passthrough` (byte-exact), :mod:`magos.egress.translate`
(LiteLLM SDK), :mod:`magos.egress.tokens` (count-tokens). Auth-header
injection lives in :mod:`magos.egress.auth`. See ``docs/architecture/request-flow.md``.
"""
