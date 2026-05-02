"""Loader tests for ``magos.routing.loader``.

Covers: YAML round-trip, structural errors (pydantic), semantic errors
(regex/glob/jq compile, passthrough base_url), and the body-touch warning.

Structlog renders to stdout via ``PrintLoggerFactory`` and caches its bound
logger on first use, which makes ``capsys``/``capfd`` capture order-dependent
across the full suite. Warning assertions instead patch ``loader.log`` with
an in-memory recorder.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

import magos.routing.loader as loader_module
from magos.routing import Compress, RoutingConfigError, load_config


class _LogRecorder:
    """Drop-in replacement for the loader's structlog logger used in tests."""

    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **kw: Any) -> None:
        self.records.append((event, kw))

    def info(self, event: str, **kw: Any) -> None:  # not exercised, present for parity
        self.records.append((event, kw))


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "magos.yaml"
    p.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return p


def test_round_trip_minimal(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        rules:
          - match: { endpoint: { literal: /v1/messages } }
            action: { provider: openai, mode: translate }
        """,
    )
    cfg = load_config(p)
    assert len(cfg.rules) == 1
    assert cfg.rules[0].action.provider == "openai"


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "magos.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(RoutingConfigError, match="must be a mapping"):
        load_config(p)


def test_invalid_pydantic_shape(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        rules:
          - match: { endpoint: { literal: /v1/messages } }
            action: { provider: openai }
        """,
    )
    with pytest.raises(RoutingConfigError, match="invalid routing config"):
        load_config(p)


def test_invalid_regex_includes_rule_label(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        rules:
          - name: broken
            match: { model: { regex: '[unclosed' } }
            action: { provider: openai, mode: translate }
        """,
    )
    with pytest.raises(RoutingConfigError, match="broken"):
        load_config(p)


def test_invalid_jq_atom_includes_rule_label(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        rules:
          - name: jqfail
            match: { jq: 'this is not valid jq <<<' }
            action: { provider: openai, mode: translate }
        """,
    )
    with pytest.raises(RoutingConfigError, match="jqfail"):
        load_config(p)


def test_invalid_jq_patch_in_rewrites(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        rules:
          - name: rewrite-fail
            match: { endpoint: { literal: /v1/messages } }
            rewrites:
              - jq_patch: '<<< not jq'
            action: { provider: openai, mode: translate }
        """,
    )
    with pytest.raises(RoutingConfigError, match="rewrite-fail"):
        load_config(p)


def test_passthrough_mode_requires_base_url(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        rules:
          - name: pt-no-base
            match: { endpoint: { literal: /v1/messages } }
            action:
              provider: anthropic
              mode: passthrough
        """,
    )
    with pytest.raises(RoutingConfigError, match="base_url"):
        load_config(p)


def test_body_touch_warns_under_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _LogRecorder()
    monkeypatch.setattr(loader_module, "log", rec)
    p = _write(
        tmp_path,
        """
        rules:
          - name: rewrite-then-passthrough
            match: { endpoint: { literal: /v1/messages } }
            rewrites:
              - set_model: claude-haiku-4-5-20251001
            action:
              provider: anthropic
              mode: passthrough
              base_url: https://api.anthropic.com
              api_key_env: ANTHROPIC_API_KEY
        """,
    )
    load_config(p)
    events = [(e, kw) for e, kw in rec.records if e == "routing.passthrough_body_touch"]
    assert len(events) == 1
    assert events[0][1]["rule"] == "rewrite-then-passthrough"
    assert events[0][1]["post_rewrites_touch"] is True
    assert events[0][1]["pre_rewrites_touch"] is False


def test_header_only_rewrites_under_passthrough_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _LogRecorder()
    monkeypatch.setattr(loader_module, "log", rec)
    p = _write(
        tmp_path,
        """
        rules:
          - match: { endpoint: { literal: /v1/messages } }
            rewrites:
              - set_header: { name: x-magos, value: '1' }
              - remove_header: x-debug
            action:
              provider: anthropic
              mode: passthrough
              base_url: https://api.anthropic.com
              api_key_env: ANTHROPIC_API_KEY
        """,
    )
    load_config(p)
    assert not [e for e, _ in rec.records if e == "routing.passthrough_body_touch"]


def test_pre_rewrite_body_touch_warns_per_passthrough_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _LogRecorder()
    monkeypatch.setattr(loader_module, "log", rec)
    p = _write(
        tmp_path,
        """
        pre_rewrites:
          - set_model: claude-haiku-4-5-20251001

        rules:
          - name: pt
            match: { endpoint: { literal: /v1/messages } }
            action:
              provider: anthropic
              mode: passthrough
              base_url: https://api.anthropic.com
              api_key_env: ANTHROPIC_API_KEY
          - name: tr
            match: { endpoint: { literal: /v1/chat/completions } }
            action:
              provider: openai
              mode: translate
        """,
    )
    load_config(p)
    events = [(e, kw) for e, kw in rec.records if e == "routing.passthrough_body_touch"]
    # Only the passthrough rule emits a warning, not the translate rule.
    assert len(events) == 1
    assert events[0][1]["rule"] == "pt"
    assert events[0][1]["pre_rewrites_touch"] is True


def test_compress_rewrite_round_trip(tmp_path: Path) -> None:
    """Compress rewrite parses with sane defaults and overridden fields."""
    p = _write(
        tmp_path,
        """
        rules:
          - name: r
            match: { endpoint: { literal: /v1/messages } }
            rewrites:
              - compress:
                  mode: token
                  target_ratio: 0.5
                  protect_recent: 8
            action: { provider: anthropic, mode: translate }
        """,
    )
    cfg = load_config(p)
    rw = cfg.rules[0].rewrites[0]
    assert isinstance(rw, Compress)
    assert rw.compress.mode == "token"
    assert rw.compress.target_ratio == 0.5
    assert rw.compress.protect_recent == 8
    # Defaults preserved for unset fields.
    assert rw.compress.compress_user_messages is False
    assert rw.compress.compress_system_messages is True


def test_compress_rewrite_default_options(tmp_path: Path) -> None:
    """``compress: {}`` parses with all CompressOptions defaults."""
    p = _write(
        tmp_path,
        """
        rules:
          - match: { endpoint: { literal: /v1/messages } }
            rewrites:
              - compress: {}
            action: { provider: anthropic, mode: translate }
        """,
    )
    cfg = load_config(p)
    rw = cfg.rules[0].rewrites[0]
    assert isinstance(rw, Compress)
    assert rw.compress.mode == "token"
    assert rw.compress.protect_recent == 4


def test_compress_rewrite_under_passthrough_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compress is body-touching; pairing with passthrough must warn."""
    rec = _LogRecorder()
    monkeypatch.setattr(loader_module, "log", rec)
    p = _write(
        tmp_path,
        """
        rules:
          - name: pt
            match: { endpoint: { literal: /v1/messages } }
            rewrites:
              - compress: { mode: token }
            action:
              provider: anthropic
              mode: passthrough
              base_url: https://api.anthropic.com
              api_key_env: ANTHROPIC_API_KEY
        """,
    )
    load_config(p)
    events = [(e, kw) for e, kw in rec.records if e == "routing.passthrough_body_touch"]
    assert len(events) == 1
    assert events[0][1]["rule"] == "pt"
    assert events[0][1]["post_rewrites_touch"] is True


def test_compress_invalid_mode_rejected(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        rules:
          - match: { endpoint: { literal: /v1/messages } }
            rewrites:
              - compress: { mode: bogus }
            action: { provider: anthropic, mode: translate }
        """,
    )
    with pytest.raises(RoutingConfigError, match="invalid routing config"):
        load_config(p)
