"""Tests for the top-level CLI dispatcher in ``magos.__main__``."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stderr, redirect_stdout

import pytest

from magos import __main__, __version__


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_top_level_help_prints_usage(flag: str) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = __main__.main([flag])
    assert rc == 0
    out = buf.getvalue()
    assert "usage: magos" in out
    assert "Subcommands:" in out
    assert "serve" in out
    assert "models" in out


def test_version_flag_prints_version() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = __main__.main(["--version"])
    assert rc == 0
    assert buf.getvalue().strip() == f"magos {__version__}"


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_serve_subcommand_help_short_circuits_serve(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`magos serve --help` must print help, not start uvicorn."""
    called = False

    def _fail_serve() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(__main__, "serve", _fail_serve)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = __main__.main(["serve", flag])
    assert rc == 0
    assert called is False
    assert "usage: magos serve" in buf.getvalue()


def test_unknown_subcommand_returns_2_with_hint() -> None:
    buf = io.StringIO()
    with redirect_stderr(buf):
        rc = __main__.main(["bogus"])
    assert rc == 2
    assert "unknown subcommand" in buf.getvalue()
    assert "magos --help" in buf.getvalue()


def test_models_help_delegates_to_argparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """`magos models --help` should reach argparse and print models usage."""
    buf = io.StringIO()
    # argparse exits via SystemExit on --help; capture cleanly.
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc_info:
        __main__.main(["models", "--help"])
    assert exc_info.value.code == 0
    assert "magos models" in buf.getvalue()
    assert "list" in buf.getvalue()
    assert "refresh" in buf.getvalue()


def test_config_flag_consumed_before_help_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--config X --help` should still print top-level help, with X applied."""
    monkeypatch.delenv("MAGOS_CONFIG_PATH", raising=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = __main__.main(["--config", "/tmp/x.yaml", "--help"])
    assert rc == 0
    assert os.environ.get("MAGOS_CONFIG_PATH") == "/tmp/x.yaml"
