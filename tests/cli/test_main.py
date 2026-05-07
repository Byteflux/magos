"""Tests for the top-level Typer CLI in `magos.cli.app`."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from magos import __version__
from magos.cli import app as cli_app
from magos.cli import serve as serve_cli

runner = CliRunner()


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_top_level_help_lists_subcommands(flag: str) -> None:
    result = runner.invoke(cli_app.app, [flag])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "models" in result.output
    assert "serve" in result.output


def test_version_flag_prints_version() -> None:
    result = runner.invoke(cli_app.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"magos {__version__}"


def test_serve_subcommand_help_short_circuits_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """`magos serve --help` must print help, not start uvicorn."""
    called = False

    def _fail_serve() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(serve_cli, "bootstrap_and_serve", _fail_serve)
    result = runner.invoke(cli_app.app, ["serve", "--help"])
    assert result.exit_code == 0
    assert called is False
    assert "Usage:" in result.output


def test_unknown_subcommand_returns_nonzero() -> None:
    result = runner.invoke(cli_app.app, ["bogus"])
    assert result.exit_code != 0


def test_models_help_lists_verbs() -> None:
    result = runner.invoke(cli_app.app, ["models", "--help"])
    assert result.exit_code == 0
    for verb in ("list", "show", "refresh", "prune", "discover"):
        assert verb in result.output


def test_config_flag_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # `--help` after `--config` lets us test the side effect without
    # actually running serve(): the eager --help fires after the
    # callback assigns MAGOS_CONFIG_PATH. monkeypatch reverts os.environ
    # at teardown so the assignment doesn't leak across tests.
    monkeypatch.setenv("MAGOS_CONFIG_PATH", "")
    monkeypatch.delenv("MAGOS_CONFIG_PATH", raising=False)
    result = runner.invoke(cli_app.app, ["--config", "/tmp/x.yaml", "models", "--help"])
    assert result.exit_code == 0
    assert os.environ.get("MAGOS_CONFIG_PATH") == "/tmp/x.yaml"


def test_home_flag_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_HOME", "")
    monkeypatch.delenv("MAGOS_HOME", raising=False)
    result = runner.invoke(cli_app.app, ["--home", "/srv/magos", "models", "--help"])
    assert result.exit_code == 0
    assert os.environ.get("MAGOS_HOME") == "/srv/magos"


def test_models_flag_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_MODELS_PATH", "")
    monkeypatch.delenv("MAGOS_MODELS_PATH", raising=False)
    result = runner.invoke(
        cli_app.app, ["--models", "/var/lib/magos/models.json", "models", "--help"]
    )
    assert result.exit_code == 0
    assert os.environ.get("MAGOS_MODELS_PATH") == "/var/lib/magos/models.json"
