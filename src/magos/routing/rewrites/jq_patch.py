"""`RewriteError`: raised when a rewrite cannot be applied (e.g., jq_patch shape error).

The `jq_patch` apply logic lives on `JqPatch.apply` in
`magos.routing.schema.rewrites`. `RewriteError` lives here so it can
be imported without pulling in the schema models.
"""

from __future__ import annotations


class RewriteError(ValueError):
    """Raised when a rewrite cannot be applied (e.g., jq_patch shape error)."""
