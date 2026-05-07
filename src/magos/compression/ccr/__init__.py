"""CCR (Compress-Cache-Retrieve) integration for magos.

Magos-side glue around `headroom.ccr`: detection, request-side tool
injection (driven from the compress rewrite), and response-side handling
(driven from egress dispatch). Headroom's `CCRToolInjector`,
`CCRResponseHandler`, `StreamingCCRHandler`, and the
`compression_store` are reused directly.
"""

from __future__ import annotations

from headroom.ccr import CCR_TOOL_NAME

from magos.compression.ccr.continuation import ContinuationCallable, make_continuation_callable
from magos.compression.ccr.handler import is_ccr_request, wrap_response, wrap_stream

__all__ = [
    "CCR_TOOL_NAME",
    "ContinuationCallable",
    "is_ccr_request",
    "make_continuation_callable",
    "wrap_response",
    "wrap_stream",
]
