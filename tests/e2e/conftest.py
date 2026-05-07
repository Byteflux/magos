"""Repo-wide skip-gate for the e2e suite.

Set ``MAGOS_E2E=1`` to opt in. Applies to every test under ``tests/e2e/``
via ``pytest_collection_modifyitems`` so individual files don't need to
declare ``pytestmark`` themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_E2E_DIR = str(Path(__file__).parent)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip = pytest.mark.skipif(
        os.environ.get("MAGOS_E2E") != "1",
        reason="set MAGOS_E2E=1 to run end-to-end provider tests",
    )
    for item in items:
        # Only apply to tests under tests/e2e/; the hook is called with
        # the full collected suite even though this conftest is scoped
        # to a subtree.
        if not str(item.fspath).startswith(_E2E_DIR):
            continue
        item.add_marker(pytest.mark.e2e)
        item.add_marker(skip)
