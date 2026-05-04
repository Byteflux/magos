"""``python -m magos`` entrypoint.

The CLI itself lives in :mod:`magos.cli.app`; this module just delegates
so ``python -m magos`` matches the ``magos`` console script registered
in ``pyproject.toml``.
"""

from __future__ import annotations

from magos.cli.app import main

if __name__ == "__main__":
    main()
