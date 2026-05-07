"""Sentence-transformers preload workaround for Windows native-load ordering."""

from __future__ import annotations

import contextlib


def _preload_sentence_transformers() -> None:
    """Force-import `sentence_transformers` to win the Windows native-load race.

    Importing `cryptography.hazmat.bindings._rust` before
    `sentence_transformers` segfaults pyarrow's `.pyd` on Windows. See
    `docs/headroom/pipeline.md` for the full bisection.
    """
    with contextlib.suppress(Exception):
        import sentence_transformers  # noqa: F401, PLC0415
