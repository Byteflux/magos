"""Top-level structure: ``Action`` constraints, ``extra='forbid'``,
frozen-immutability, ``RoutingConfig`` invariants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from magos.routing.schema import (
    Action,
    RoutingConfig,
    Rule,
    SetHeader,
)

# --- Action constraints ---


def test_action_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        Action.model_validate({"provider": "openai", "mode": "swerve"})


def test_action_rejects_blank_provider() -> None:
    with pytest.raises(ValidationError):
        Action.model_validate({"provider": "", "mode": "translate"})


# --- extra="forbid" + frozen ---


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Rule.model_validate(
            {
                "match": {"endpoint": {"literal": "/v1/messages"}},
                "action": {"provider": "openai", "mode": "translate"},
                "unexpected": True,
            }
        )


def test_frozen_models_cannot_mutate() -> None:
    rule = Rule.model_validate(
        {
            "match": {"endpoint": {"literal": "/v1/messages"}},
            "action": {"provider": "openai", "mode": "translate"},
        }
    )
    with pytest.raises(ValidationError):
        rule.action.provider = "anthropic"


# --- Top-level invariants ---


def test_routing_config_requires_at_least_one_rule() -> None:
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate({"rules": []})


def test_routing_config_round_trips() -> None:
    cfg = RoutingConfig.model_validate(
        {
            "pre_rewrites": [{"set_header": {"name": "x-magos", "value": "1"}}],
            "rules": [
                {
                    "name": "default",
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ],
        }
    )
    assert cfg.rules[0].name == "default"
    assert isinstance(cfg.pre_rewrites[0], SetHeader)
