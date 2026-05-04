"""Pipeline tests for ``magos.routing.engine``."""

from __future__ import annotations

from typing import Any

import pytest

from magos.registry.state import ModelEntry
from magos.routing import RoutingConfig
from magos.routing.engine import RouteDecision, route
from magos.routing.errors import RouteError

from ._helpers import make_registry
from ._helpers import make_req as _req


def _cfg(spec: dict[str, Any]) -> RoutingConfig:
    return RoutingConfig.model_validate(spec)


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


# --- Guarded pre-rewrites ---


def _guarded_cfg(pre: list[dict[str, Any]]) -> RoutingConfig:
    return _cfg(
        {
            "pre_rewrites": pre,
            "rules": [
                {
                    "name": "translate",
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ],
        }
    )


def test_guarded_pre_rewrite_applies_when_match_passes() -> None:
    cfg = _guarded_cfg(
        [
            {
                "match": {"endpoint": {"literal": "/v1/messages"}},
                "rewrites": [{"set_header": {"name": "x-marker", "value": "yes"}}],
            }
        ]
    )
    decision = route(_req(body={"model": "gpt-4"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.request.headers["x-marker"] == "yes"


def test_guarded_pre_rewrite_skipped_when_match_fails() -> None:
    cfg = _guarded_cfg(
        [
            {
                "match": {"endpoint": {"literal": "/v1/chat/completions"}},
                "rewrites": [{"set_header": {"name": "x-marker", "value": "yes"}}],
            }
        ]
    )
    decision = route(_req(body={"model": "gpt-4"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert "x-marker" not in decision.request.headers


def test_bare_and_guarded_pre_rewrites_chain_in_order() -> None:
    cfg = _guarded_cfg(
        [
            {"set_header": {"name": "x-bare", "value": "1"}},
            {
                "match": {"header": {"name": {"literal": "x-bare"}, "value": {"literal": "1"}}},
                "rewrites": [{"set_header": {"name": "x-guarded", "value": "2"}}],
            },
        ]
    )
    decision = route(_req(body={"model": "gpt-4"}), cfg)
    assert isinstance(decision, RouteDecision)
    assert decision.request.headers["x-bare"] == "1"
    assert decision.request.headers["x-guarded"] == "2"


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


@pytest.mark.parametrize(
    ("provider", "body_model", "expected", "registry_entry"),
    [
        # ``set_model: vultr/Qwen/...`` -- literal registry hit substitutes litellm_id.
        (
            "vultr",
            "vultr/Qwen/Qwen3.5-397B-A17B-FP8",
            "custom_openai/Qwen/Qwen3.5-397B-A17B-FP8",
            ("vultr", "Qwen/Qwen3.5-397B-A17B-FP8", "custom_openai/Qwen/Qwen3.5-397B-A17B-FP8"),
        ),
        # ``set_model: Qwen/...`` -- bare id resolved by prepending action.provider.
        (
            "vultr",
            "Qwen/Qwen3.5-397B-A17B-FP8",
            "custom_openai/Qwen/Qwen3.5-397B-A17B-FP8",
            ("vultr", "Qwen/Qwen3.5-397B-A17B-FP8", "custom_openai/Qwen/Qwen3.5-397B-A17B-FP8"),
        ),
        # Registry miss with ``/`` in model -- fall through to LiteLLM as-is.
        ("openai", "openai/gpt-4-turbo", "openai/gpt-4-turbo", None),
    ],
)
def test_translate_mode_dispatch_model_resolution(
    provider: str,
    body_model: str,
    expected: str,
    registry_entry: tuple[str, str, str] | None,
) -> None:
    cfg = _cfg(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": provider, "mode": "translate"},
                }
            ]
        }
    )
    registry = make_registry(
        *(
            (
                ModelEntry(
                    provider=registry_entry[0],
                    raw_id=registry_entry[1],
                    litellm_id=registry_entry[2],
                ),
            )
            if registry_entry is not None
            else ()
        )
    )
    decision = route(_req(body={"model": body_model}), cfg, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == expected


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
