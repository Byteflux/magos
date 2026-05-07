"""Top-level structure: ``Target`` constraints, ``extra='forbid'``,
frozen-immutability, ``RoutingConfig`` invariants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from magos.routing.schema import (
    RoutingConfig,
    Rule,
    SetHeader,
    Target,
)

# --- Target constraints ---


def test_target_rejects_unknown_gateway() -> None:
    with pytest.raises(ValidationError):
        Target.model_validate({"provider": "openai", "gateway": "swerve"})


def test_target_rejects_blank_provider() -> None:
    with pytest.raises(ValidationError):
        Target.model_validate({"provider": "", "gateway": "translate"})


# --- extra="forbid" + frozen ---


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Rule.model_validate(
            {
                "match": {"endpoint": {"literal": "/v1/messages"}},
                "target": {"provider": "openai", "gateway": "translate"},
                "unexpected": True,
            }
        )


def test_frozen_models_cannot_mutate() -> None:
    rule = Rule.model_validate(
        {
            "match": {"endpoint": {"literal": "/v1/messages"}},
            "target": {"provider": "openai", "gateway": "translate"},
        }
    )
    with pytest.raises(ValidationError):
        rule.target.provider = "anthropic"


# --- Top-level invariants ---


def test_routing_config_requires_at_least_one_rule() -> None:
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate({"rules": []})


def test_routing_config_round_trips() -> None:
    cfg = RoutingConfig.model_validate(
        {
            "pre_transforms": [{"set_header": {"name": "x-magos", "value": "1"}}],
            "rules": [
                {
                    "name": "default",
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "target": {"provider": "openai", "gateway": "translate"},
                }
            ],
        }
    )
    assert cfg.rules[0].name == "default"
    assert isinstance(cfg.pre_transforms[0], SetHeader)
