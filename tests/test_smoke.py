"""Smoke tests verifying the package imports and exposes a version."""

import pytest

import magos


@pytest.mark.unit
def test_version_is_string() -> None:
    assert isinstance(magos.__version__, str)
    assert magos.__version__


@pytest.mark.unit
def test_version_matches_pep440() -> None:
    parts = magos.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])
