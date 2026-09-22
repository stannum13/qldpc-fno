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
    CandidateOpportunity,
    candidate_benefit,
    class_mismatch_transition,
    composite_work,
    jensen_shannon_divergence,
    load_pilot_config,
    load_shot_config,
    margin_gate_accepts,
    normalized_probabilities,
    physical_outcome_transition,
    probability_margin,
    reference_conditional_excess_risk,
    select_accuracy_oracle,
    select_efficiency_oracle,
    symmetric_reference_posterior,
    total_variation,
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


def test_posterior_metrics_normalize_and_measure_margin_and_reference_risk() -> None:
    posterior = normalized_probabilities([4, 2, 0, 0])

    assert posterior == pytest.approx((2 / 3, 1 / 3, 0, 0))
    assert probability_margin([4, 2, 0, 0]) == pytest.approx(1 / 3)
    assert total_variation([4, 2, 0, 0], [2, 4, 0, 0]) == pytest.approx(1 / 3)
    assert reference_conditional_excess_risk([2, 4, 0, 0], [4, 2, 0, 0]) == pytest.approx(
        1 / 3
    )


def test_jensen_shannon_is_natural_log_and_handles_zero_components() -> None:
    assert jensen_shannon_divergence([1, 0, 0, 0], [0, 1, 0, 0]) == pytest.approx(
        0.6931471805599453
    )
    assert jensen_shannon_divergence([2, 0, 0, 0], [1, 1, 0, 0]) == pytest.approx(
        0.21576155433883565
    )


def test_symmetric_reference_re_normalizes_the_mean_of_separate_views() -> None:
    reference = symmetric_reference_posterior([8, 2, 0, 0], [1, 3, 0, 0])

    assert reference == pytest.approx((0.525, 0.475, 0, 0))


def test_margin_gate_is_strict_and_escalates_on_invalid_or_disagreeing_probes() -> None:
    margin = MARGIN_THRESHOLD
    equal_margin = [(1 + margin) / 2, (1 - margin) / 2, 0, 0]
    above_margin = [(1 + margin + 1e-4) / 2, (1 - margin - 1e-4) / 2, 0, 0]

    assert not margin_gate_accepts(equal_margin, 0, above_margin, 0)
    assert margin_gate_accepts(above_margin, 0, above_margin, 0)
    assert not margin_gate_accepts(above_margin, 0, above_margin, 1)
    assert not margin_gate_accepts(None, None, above_margin, 0)
    assert not margin_gate_accepts(above_margin, 0, None, None)


def test_composite_work_charges_every_independent_run() -> None:
    assert composite_work(13, 17, 19) == 49


def test_candidate_benefit_uses_the_best_valid_probe_and_retains_negative_values() -> None:
    reference = [1, 0, 0, 0]

    assert candidate_benefit([0.6, 0.4, 0, 0], [0.8, 0.2, 0, 0], [0.9, 0.1, 0, 0], reference) == pytest.approx(0.1)
    assert candidate_benefit([0.8, 0.2, 0, 0], None, [0.8, 0.2, 0, 0], reference) == 0
    assert candidate_benefit([0.8, 0.2, 0, 0], [0.7, 0.3, 0, 0], [0.5, 0.5, 0, 0], reference) == pytest.approx(-0.3)
    assert candidate_benefit(None, None, [0.9, 0.1, 0, 0], reference) is None
    assert candidate_benefit([0.8, 0.2, 0, 0], None, None, reference) is None


def _opportunity(
    action_id: str, gain: float | None, work: int, *, valid: bool = True
) -> CandidateOpportunity:
    return CandidateOpportunity(
        action_id=action_id,
        valid=valid,
        gain=gain,
        composite_work=work,
    )


def test_accuracy_oracle_requires_strict_positive_gain_and_reports_separation() -> None:
    floor = 1e-6
    no_winner = select_accuracy_oracle(
        [_opportunity("columns_chi2", floor, 10), _opportunity("rows_chi2", 0.0, 1)]
    )
    unique = select_accuracy_oracle(
        [_opportunity("columns_chi2", 0.2, 100), _opportunity("rows_chi2", 0.1, 1)]
    )
    close = select_accuracy_oracle(
        [_opportunity("columns_chi2", 0.2, 1), _opportunity("rows_chi2", 0.1999995, 2)]
    )
    equality = select_accuracy_oracle(
        [_opportunity("columns_chi2", 0.2, 1), _opportunity("rows_chi2", 0.199999, 2)]
    )

    assert no_winner.action_id is None
    assert not no_winner.uniquely_separated
    assert unique.action_id == "columns_chi2"
    assert unique.uniquely_separated
    assert close.action_id == "columns_chi2"
    assert not close.uniquely_separated
    assert not equality.uniquely_separated


def test_oracles_use_tolerance_then_lower_work_then_literal_action_order() -> None:
    tied_gain = 0.2
    accuracy = select_accuracy_oracle(
        [
            _opportunity("rows_chi2", tied_gain * (1 + 5e-13), 9),
            _opportunity("columns_chi2", tied_gain, 9),
        ]
    )
    efficiency = select_efficiency_oracle(
        [
            _opportunity("columns_chi2", 0.4, 100),
            _opportunity("rows_chi2", 0.2, 50),
            _opportunity("columns_chi4", 0.3, 60),
        ]
    )
    lower_work = select_accuracy_oracle(
        [
            _opportunity("columns_chi2", tied_gain * (1 + 5e-13), 10),
            _opportunity("rows_chi2", tied_gain, 9),
        ]
    )
    invalid = select_accuracy_oracle(
        [
            _opportunity("columns_chi2", 0.4, 1, valid=False),
            _opportunity("rows_chi2", 0.1, 2),
        ]
    )

    assert accuracy.action_id == "columns_chi2"
    assert efficiency.action_id == "columns_chi4"
    assert efficiency.efficiency == pytest.approx(0.005)
    assert efficiency.uniquely_separated
    assert lower_work.action_id == "rows_chi2"
    assert invalid.action_id == "rows_chi2"


def test_efficiency_oracle_requires_full_work_and_strict_one_percent_separation() -> None:
    winner = select_efficiency_oracle(
        [
            _opportunity("columns_chi2", 0.4, 100),
            _opportunity("rows_chi2", 0.2, 51),
            _opportunity("columns_chi4", 0.7, 0),
        ]
    )
    single = select_efficiency_oracle([_opportunity("columns_chi2", 0.1, 100)])
    equality = select_efficiency_oracle(
        [_opportunity("columns_chi2", 0.5, 100), _opportunity("rows_chi2", 0.5, 101)]
    )

    assert winner.action_id == "columns_chi2"
    assert (winner.efficiency - 0.2 / 51) / (0.2 / 51) > 0.01
    assert winner.uniquely_separated
    assert single.action_id == "columns_chi2"
    assert single.uniquely_separated
    assert not equality.uniquely_separated


def test_probe_actions_cannot_be_candidates() -> None:
    with pytest.raises(ValueError, match="candidate action_id"):
        _opportunity("columns_tol01", 0.1, 100)


def test_transition_labels_capture_repairs_introductions_and_unchanged_states() -> None:
    assert class_mismatch_transition(1, 0, 0) == "repaired_mismatch"
    assert class_mismatch_transition(0, 1, 0) == "introduced_mismatch"
    assert class_mismatch_transition(0, 0, 0) == "matched"
    assert class_mismatch_transition(1, 2, 0) == "persistent_mismatch"
    assert physical_outcome_transition(True, False) == "repaired_discordance"
    assert physical_outcome_transition(False, True) == "introduced_discordance"
    assert physical_outcome_transition(False, False) == "agreed"
    assert physical_outcome_transition(True, True) == "persistent_discordance"
