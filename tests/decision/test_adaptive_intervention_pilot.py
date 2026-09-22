from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from qldpc_fno.decision.adaptive_intervention_pilot import (
    ACTION_IDS,
    CAMPAIGN_SEED,
    MARGIN_THRESHOLD,
    load_pilot_config,
    load_shot_config,
)

_ROOT = Path(__file__).resolve().parents[2]
_PILOT_CONFIG = _ROOT / "configs" / "adaptive_intervention_pilot.json"
_SHOT_CONFIG = _ROOT / "configs" / "adaptive_intervention_pilot_shots.json"


def test_config_freezes_the_literal_action_table_and_policy_constants() -> None:
    config = load_pilot_config(_PILOT_CONFIG)

    assert config.schema_version == 1
    assert config.pilot_id == "adaptive_intervention_pilot_v1"
    assert config.code_distance == 5
    assert config.noise_model == "qecsim_iid_depolarizing_code_capacity"
    assert config.required_shot_seed_domain == "qldpc-fno/adaptive-intervention-pilot/v1"
    assert config.required_shots_per_rate == 64
    assert config.required_error_rates == (0.1, 0.15)
    assert config.reference_modes == ("columns", "rows")
    assert config.margin_threshold == MARGIN_THRESHOLD == 0.30710401263493464
    assert config.reference_probability_tolerance == 1e-10
    assert config.reference_log_ratio_tolerance == 1e-8
    assert config.positive_gain_threshold == 1e-6
    assert config.oracle_tie_relative_tolerance == 1e-12
    assert config.oracle_tie_absolute_tolerance == 0.0
    assert config.efficiency_unique_relative_separation == 0.01
    assert tuple(action.action_id for action in config.actions) == ACTION_IDS
    assert tuple((action.mode, action.chi, action.tol) for action in config.actions) == (
        ("columns", 2, None),
        ("rows", 2, None),
        ("columns", 4, None),
        ("rows", 4, None),
        ("columns", 8, None),
        ("rows", 8, None),
        ("columns", 16, None),
        ("rows", 16, None),
        ("columns", None, 0.03),
        ("rows", None, 0.03),
        ("columns", None, 0.01),
        ("rows", None, 0.01),
        ("columns", None, 0.003),
        ("rows", None, 0.003),
    )
    assert config.probe_action_ids == ("columns_tol01", "rows_tol01")
    assert len({(action.mode, action.chi, action.tol) for action in config.actions}) == 14


def test_shot_config_has_the_frozen_identity_and_derived_seed() -> None:
    config = load_shot_config(_SHOT_CONFIG)

    assert config.schema_version == 1
    assert config.seed_domain == "qldpc-fno/adaptive-intervention-pilot/v1"
    assert config.campaign_seed == CAMPAIGN_SEED == 16980117767564665917
    assert config.campaign_seed == int.from_bytes(
        hashlib.sha256(config.seed_domain.encode()).digest()[:8], "big"
    )
    assert config.code_distance == 5
    assert config.error_rates == (0.1, 0.15)
    assert config.shots_per_rate == 64
    assert config.noise_model == "qecsim_iid_depolarizing_code_capacity"


@pytest.mark.parametrize(
    ("config_name", "mutation"),
    [
        ("pilot", {"unexpected": True}),
        ("pilot", {"schema_version": 2}),
        ("pilot", {"schema_version": True}),
        ("pilot", {"required_shots_per_rate": 64.0}),
        ("pilot", {"code_distance": 5.0}),
        ("pilot", {"margin_threshold": float("nan")}),
        ("pilot", {"oracle_tie_absolute_tolerance": 1e-9}),
        ("pilot", {"actions": []}),
        ("pilot", {"probe_action_ids": ["columns_tol01", "columns_tol01"]}),
        ("shots", {"unexpected": True}),
        ("shots", {"schema_version": 2}),
        ("shots", {"schema_version": True}),
        ("shots", {"code_distance": 5.0}),
        ("shots", {"shots_per_rate": 64.0}),
        ("shots", {"campaign_seed": 0}),
        ("shots", {"error_rates": [0.1, 0.15, 0.2]}),
    ],
)
def test_config_parser_rejects_noncanonical_values(
    tmp_path: Path, config_name: str, mutation: dict[str, object]
) -> None:
    source = _PILOT_CONFIG if config_name == "pilot" else _SHOT_CONFIG
    payload = json.loads(source.read_text())
    payload.update(mutation)
    path = tmp_path / f"{config_name}.json"
    path.write_text(json.dumps(payload))

    loader = load_pilot_config if config_name == "pilot" else load_shot_config
    with pytest.raises(ValueError):
        loader(path)


@pytest.mark.parametrize(
    ("config_name", "replacement"),
    [
        (
            "shots",
            (
                '"campaign_seed": 16980117767564665917',
                '"campaign_seed": 0,\n  "campaign_seed": 16980117767564665917',
            ),
        ),
        (
            "pilot",
            (
                '"action_id": "columns_chi2", "mode": "columns"',
                (
                    '"action_id": "wrong", "action_id": "columns_chi2", '
                    '"mode": "columns"'
                ),
            ),
        ),
    ],
)
def test_config_parser_rejects_duplicate_json_keys(
    tmp_path: Path, config_name: str, replacement: tuple[str, str]
) -> None:
    source = _PILOT_CONFIG if config_name == "pilot" else _SHOT_CONFIG
    malformed = source.read_text().replace(*replacement, 1)
    path = tmp_path / f"duplicate-{config_name}.json"
    path.write_text(malformed)

    loader = load_pilot_config if config_name == "pilot" else load_shot_config
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        loader(path)


def test_parsed_configs_are_immutable() -> None:
    pilot = load_pilot_config(_PILOT_CONFIG)
    shots = load_shot_config(_SHOT_CONFIG)

    with pytest.raises((AttributeError, TypeError)):
        pilot.margin_threshold = 0.0  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        shots.shots_per_rate = 1  # type: ignore[misc]
    assert replace(pilot, margin_threshold=pilot.margin_threshold) == pilot
