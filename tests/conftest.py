"""Pytest fixtures and bootstrap.

Loads ``.env`` into ``os.environ`` only when end-to-end tests are enabled,
so unit and integration runs stay free of real provider credentials.
LiteLLM and other libraries read keys directly from the process
environment, not from pydantic-settings, hence the explicit population.

Also points ``MAGOS_CONFIG_PATH`` at the test fixture YAML before any test
imports ``magos.server``, so ``create_app()`` calls without an explicit
``routing`` argument find a real config file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from magos.routing import RoutingConfig, load_config

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_YAML = _TESTS_DIR / "fixtures" / "magos.test.yaml"

# Default routing config for tests that call ``create_app()`` without an
# explicit ``routing=`` argument; e2e tests can override via env.
os.environ.setdefault("MAGOS_CONFIG_PATH", str(_FIXTURE_YAML))

if os.environ.get("MAGOS_E2E") == "1":
    env_path = _TESTS_DIR.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    # E2E tests should exercise the operator-facing example config so we
    # catch breakage in the shipped defaults, not the test fixture.
    example = _TESTS_DIR.parent / "magos.example.yaml"
    if example.is_file():
        os.environ["MAGOS_CONFIG_PATH"] = str(example)


@pytest.fixture
def routing_cfg() -> RoutingConfig:
    """Loaded test routing config; cheap enough to load per test."""
    return load_config(_FIXTURE_YAML)
