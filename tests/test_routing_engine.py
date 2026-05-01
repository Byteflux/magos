"""Pipeline tests for ``magos.routing.engine``."""

from __future__ import annotations

from typing import Any

from magos.routing import RoutingConfig
from magos.routing.engine import RouteDecision, route
from magos.routing.errors import RouteError
from magos.routing.request import RoutedRequest


def _cfg(spec: dict[str, Any]) -> RoutingConfig:
    return RoutingConfig.model_validate(spec)


def _req(
    *,
    endpoint: str = "/v1/messages",
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> RoutedRequest:
    return RoutedRequest(
        endpoint=endpoint,  # type: ignore[arg-type]
        headers=headers or {},
        body=body or {},
        raw_body=b"",
    )


# --- First-match-wins ---


def test_first_matching_rule_wins() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "name": "claude",
                    "match": {"model": {"glob": "claude-*"}},
                    "action": {
                        "provider": "anthropic",
                        "mode": "passthrough",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                },
                {
                    "name": "fallback",
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                },
            ]
        }
    )
    decision = route(_req(body={"model": "claude-3"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.rule.name == "claude"


def test_falls_through_to_later_rule_when_earlier_does_not_match() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "name": "claude-only",
                    "match": {"model": {"glob": "claude-*"}},
                    "action": {"provider": "anthropic", "mode": "passthrough"},
                },
                {
                    "name": "default",
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                },
            ]
        }
    )
    decision = route(_req(body={"model": "gpt-4"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.rule.name == "default"


# --- Unmatched -> 404 ---


def test_unmatched_returns_404_route_error() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"model": {"literal": "anthropic-only"}},
                    "action": {"provider": "anthropic", "mode": "passthrough"},
                }
            ]
        }
    )
    err = route(_req(body={"model": "gpt-4"}), cfg)
    assert isinstance(err, RouteError)
    assert err.status == 404
    assert err.code == "unmatched"
    assert "gpt-4" in err.message
    assert err.endpoint == "/v1/messages"


def test_unmatched_carries_endpoint_for_envelope_shaping() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "anthropic", "mode": "passthrough"},
                }
            ]
        }
    )
    err = route(_req(endpoint="/v1/chat/completions", body={"model": "x"}), cfg)
    assert isinstance(err, RouteError)
    assert err.endpoint == "/v1/chat/completions"


# --- Pre-rewrites apply before match ---


def test_pre_rewrite_changes_what_matches() -> None:
    # An alias-normalising pre-rewrite redirects "sonnet" to a real id
    # before the matcher sees it; the literal-match rule for the real id
    # then succeeds.
    cfg = _cfg(
        {
            "pre_rewrites": [
                {
                    "jq_patch": (
                        'if .model == "sonnet" then .model = "claude-haiku-4-5-20251001" else . end'
                    )
                }
            ],
            "rules": [
                {
                    "match": {"model": {"literal": "claude-haiku-4-5-20251001"}},
                    "action": {"provider": "anthropic", "mode": "passthrough"},
                }
            ],
        }
    )
    decision = route(_req(body={"model": "sonnet"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.request.body["model"] == "claude-haiku-4-5-20251001"
    assert decision.request.body_dirty is True


# --- Post-rewrites apply after match ---


def test_post_rewrites_run_for_matched_rule() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "rewrites": [{"set_header": {"name": "x-magos-route", "value": "openai"}}],
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    decision = route(_req(body={"model": "gpt-4"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.request.headers["x-magos-route"] == "openai"


def test_post_rewrite_failure_returns_503() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "name": "broken",
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "rewrites": [{"jq_patch": ".model"}],  # returns scalar, not object
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    err = route(_req(body={"model": "gpt-4"}), cfg)
    assert isinstance(err, RouteError)
    assert err.status == 503
    assert err.code == "dispatch_error"
    assert err.endpoint == "/v1/messages"


# --- dispatch_model computation ---


def test_translate_mode_prepends_provider_prefix() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    decision = route(_req(body={"model": "gpt-4"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == "openai/gpt-4"


def test_translate_mode_preserves_existing_prefix() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    decision = route(_req(body={"model": "openai/gpt-4-turbo"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == "openai/gpt-4-turbo"


def test_passthrough_mode_keeps_bare_model() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "anthropic", "mode": "passthrough"},
                }
            ]
        }
    )
    decision = route(_req(body={"model": "claude-haiku-4-5-20251001"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == "claude-haiku-4-5-20251001"


# --- Rule labelling ---


def test_rule_label_uses_name_when_present() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "name": "named",
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    decision = route(_req(body={"model": "x"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.rule_label() == "named"


def test_rule_label_falls_back_to_index() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    decision = route(_req(body={"model": "x"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.rule_label(idx=0) == "rule[0]"


# --- Decision exposes action shortcut ---


def test_decision_action_property_returns_rule_action() -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    decision = route(_req(body={"model": "x"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.action.provider == "openai"
    assert decision.action.mode == "translate"
