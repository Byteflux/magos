"""Egress error types shared across auth, dispatch, and downstream branches."""

from __future__ import annotations


class DispatchError(Exception):
    """Raised when a runtime config invariant fails (e.g., missing env var)."""
