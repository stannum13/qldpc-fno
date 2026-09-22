from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qecsim.models.planar import PlanarCode

from qldpc_fno.decision import adaptive_intervention_pilot as pilot
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
from qldpc_fno.decision.tensor_network import InvalidContractionError, PlanarCosetMasses

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
                ('"action_id": "wrong", "action_id": "columns_chi2", "mode": "columns"'),
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
    assert reference_conditional_excess_risk([2, 4, 0, 0], [4, 2, 0, 0]) == pytest.approx(1 / 3)


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
    equal_margin = [0.75, 0.25, 0, 0]
    just_above_margin = [0.750000000000005, 0.249999999999995, 0, 0]
    above_margin = [(1 + MARGIN_THRESHOLD + 1e-4) / 2, (1 - MARGIN_THRESHOLD - 1e-4) / 2, 0, 0]

    assert probability_margin(just_above_margin) == 0.5 + 1e-14
    assert not margin_gate_accepts(equal_margin, 0, equal_margin, 0, threshold=0.5)
    assert margin_gate_accepts(just_above_margin, 0, just_above_margin, 0, threshold=0.5)
    assert margin_gate_accepts(above_margin, 0, above_margin, 0)
    assert not margin_gate_accepts(above_margin, 0, above_margin, 1)
    assert not margin_gate_accepts(None, None, above_margin, 0)
    assert not margin_gate_accepts(above_margin, 0, None, None)


def test_composite_work_charges_every_independent_run() -> None:
    assert composite_work(13, 17, 19) == 49


def test_candidate_benefit_uses_the_best_valid_probe_and_retains_negative_values() -> None:
    reference = [1, 0, 0, 0]

    assert candidate_benefit(
        [0.6, 0.4, 0, 0], [0.8, 0.2, 0, 0], [0.9, 0.1, 0, 0], reference
    ) == pytest.approx(0.1)
    assert candidate_benefit([0.8, 0.2, 0, 0], None, [0.8, 0.2, 0, 0], reference) == 0
    assert candidate_benefit(
        [0.8, 0.2, 0, 0], [0.7, 0.3, 0, 0], [0.5, 0.5, 0, 0], reference
    ) == pytest.approx(-0.3)
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
    just_above = select_accuracy_oracle(
        [_opportunity("columns_chi2", math.nextafter(floor, math.inf), 1)]
    )

    assert no_winner.action_id is None
    assert not no_winner.uniquely_separated
    assert unique.action_id == "columns_chi2"
    assert unique.uniquely_separated
    assert close.action_id == "columns_chi2"
    assert not close.uniquely_separated
    assert just_above.action_id == "columns_chi2"


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
    just_above = select_efficiency_oracle(
        [
            _opportunity("columns_chi2", math.nextafter(0.505, math.inf), 100),
            _opportunity("rows_chi2", 0.5, 100),
        ]
    )

    assert winner.action_id == "columns_chi2"
    assert (winner.efficiency - 0.2 / 51) / (0.2 / 51) > 0.01
    assert winner.uniquely_separated
    assert single.action_id == "columns_chi2"
    assert single.uniquely_separated
    assert just_above.action_id == "columns_chi2"
    assert just_above.uniquely_separated


def test_accuracy_oracle_tie_breaking_is_input_order_independent() -> None:
    candidates = (
        _opportunity("columns_chi2", 0.2, 1),
        _opportunity("rows_chi2", 0.2 * (1 + 0.75e-12), 2),
        _opportunity("columns_chi4", 0.2 * (1 + 1.5e-12), 10),
    )

    selections = {
        select_accuracy_oracle(order).action_id for order in itertools.permutations(candidates)
    }

    assert selections == {"rows_chi2"}


def test_efficiency_oracle_tie_breaking_is_input_order_independent() -> None:
    candidates = (
        _opportunity("columns_chi2", 0.2, 100),
        _opportunity("rows_chi2", 0.2 * (1 + 0.75e-12), 100),
        _opportunity("columns_chi4", 0.2 * (1 + 1.5e-12), 100),
    )

    selections = {
        select_efficiency_oracle(order).action_id for order in itertools.permutations(candidates)
    }

    assert selections == {"rows_chi2"}


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


def _shot(rate=0.1, index=0):
    code = PlanarCode(5, 5)
    return {
        "shot_id": f"{rate}-{index}",
        "shot_index": index,
        "sampler_seed": index,
        "error_rate": rate,
        "syndrome": np.zeros(40, dtype=np.uint8),
        # A nonidentity stabilizer is a degenerate, physically successful error.
        "error": code.stabilizers[0].copy(),
    }


def _fake_contractions(monkeypatch, *, invalid=(), overrides=None, bug=None):
    calls = []
    config = load_pilot_config(_PILOT_CONFIG)
    actions = {(a.mode, a.chi, a.tol): a.action_id for a in config.actions}
    overrides = overrides or {}

    def contract(**kwargs):
        key = (kwargs["mode"], kwargs["chi"], kwargs["tol"])
        action_id = actions.get(key, "exact_" + kwargs["mode"])
        assert kwargs["trace_work"] is True
        calls.append(action_id)
        if bug is not None:
            raise bug
        work = {
            "estimated_arithmetic_flops": len(calls) * 10,
            "wall_seconds": 100,
            "contraction_sweeps": [{"index": 0, "marker": action_id}],
            "truncation_events": [
                {
                    "spectral_summaries": [
                        {
                            "discarded_squared_weight_fraction": 0.02,
                            "spectral_entropy": 0.7,
                            "retained_rank": 3,
                        }
                    ]
                }
            ],
        }
        if action_id in invalid:
            raise InvalidContractionError(np.zeros(4), work)
        default = [0.7, 0.1, 0.1, 0.1] if action_id.startswith("exact") else [0.4, 0.3, 0.2, 0.1]
        masses = np.array(overrides.get(action_id, default))
        return PlanarCosetMasses(
            masses=masses,
            selected_class=int(masses.argmax()),
            rows=5,
            columns=5,
            error_rate=kwargs["error_rate"],
            chi=kwargs["chi"],
            tol=kwargs["tol"],
            mode=kwargs["mode"],
            work=work,
        )

    monkeypatch.setattr(pilot, "planar_mps_coset_masses", contract, raising=False)
    return config, calls


def test_evaluator_executes_every_action_once_and_certifies_both_views(monkeypatch):
    config, calls = _fake_contractions(monkeypatch)
    result = pilot.evaluate_shot(_shot(), config)

    assert calls == ["exact_columns", "exact_rows", *ACTION_IDS]
    assert list(result["actions"]) == list(ACTION_IDS)
    assert result["reference"]["certified"]
    assert result["reference"]["certificate"]["maximum_probability_discrepancy"] == 0
    assert result["labels"]["reference_probabilities"] == pytest.approx([0.7, 0.1, 0.1, 0.1])
    assert result["total_estimated_arithmetic_flops"] == sum(range(1, 17)) * 10
    assert (
        result["actions"]["columns_chi2"]["work"]["contraction_sweeps"][0]["marker"]
        == "columns_chi2"
    )
    assert "wall_seconds" not in result["actions"]["columns_chi2"]["work"]
    assert result["gate"]["escalated"]


def test_evaluator_scores_actual_symplectic_recoveries_and_keeps_labels_separate(monkeypatch):
    config, _ = _fake_contractions(monkeypatch, overrides={"rows_chi2": [0.1, 0.7, 0.1, 0.1]})
    result = pilot.evaluate_shot(_shot(), config)

    assert not result["labels"]["actions"]["columns_chi2"]["physical"]["failure"]
    wrong = result["labels"]["actions"]["rows_chi2"]
    assert wrong["physical"]["syndrome_valid"]
    assert wrong["physical"]["logical_failure"]
    assert wrong["class_mismatch"] and wrong["physical_outcome_discordance"]
    assert wrong["reference_conditional_excess_risk"] == pytest.approx(0.6)
    assert "physical" not in result["actions"]["rows_chi2"]
    assert set(result["features"]) == {
        "error_rate",
        "syndrome_hamming_weight",
        "probes",
        "class_agreement",
        "minimum_margin",
        "cross_probe_total_variation",
        "cross_probe_jensen_shannon",
    }
    for probe in result["features"]["probes"].values():
        assert set(probe) == {"valid", "selected_class", "margin", "work", "spectral_summary"}
        assert set(probe["spectral_summary"]) == {
            "available",
            "truncation_event_count",
            "spectrum_count",
            "maximum_discarded_squared_weight_fraction",
            "mean_spectral_entropy",
            "maximum_retained_rank",
            "estimated_arithmetic_flops",
        }
    assert (
        result["features"]["probes"]["columns_tol01"]["work"]["estimated_arithmetic_flops"] == 130
    )


def test_invalid_numerical_actions_retain_partial_work_and_escalate(monkeypatch):
    config, calls = _fake_contractions(monkeypatch, invalid=("columns_tol01", "rows_chi2"))
    result = pilot.evaluate_shot(_shot(), config)

    assert len(calls) == 16
    invalid = result["actions"]["rows_chi2"]
    assert not invalid["valid"]
    assert invalid["exception_type"] == "InvalidContractionError"
    assert invalid["estimated_arithmetic_flops"] == 40
    assert invalid["work"]["contraction_sweeps"]
    assert result["labels"]["actions"]["rows_chi2"]["physical"]["failure"]
    probe = result["features"]["probes"]["columns_tol01"]
    assert probe["work"] is None and probe["selected_class"] is None
    assert probe["spectral_summary"] == {
        "available": False,
        "truncation_event_count": 0,
        "spectrum_count": 0,
        "maximum_discarded_squared_weight_fraction": None,
        "mean_spectral_entropy": None,
        "maximum_retained_rank": None,
        "estimated_arithmetic_flops": None,
    }
    assert result["gate"]["escalated"]
    assert result["labels"]["baseline_total_variation"] == pytest.approx(0.3)
    assert result["labels"]["candidates"]["columns_chi2"]["composite_work"] == 130 + 140 + 30


@pytest.mark.parametrize("failure", [ValueError("configuration"), RuntimeError("bug")])
def test_programming_and_configuration_errors_are_not_measured_invalidity(monkeypatch, failure):
    config, _ = _fake_contractions(monkeypatch, bug=failure)
    with pytest.raises(type(failure), match=str(failure)):
        pilot.evaluate_shot(_shot(), config)


@pytest.mark.parametrize("invalid", [("exact_columns",), ("columns_tol01", "rows_tol01")])
def test_missing_reference_or_baseline_makes_oracles_null(monkeypatch, invalid):
    config, _ = _fake_contractions(monkeypatch, invalid=invalid)
    result = pilot.evaluate_shot(_shot(), config)
    assert result["labels"]["baseline_total_variation"] is None
    assert result["labels"]["accuracy_oracle"]["action_id"] is None
    assert result["labels"]["efficiency_oracle"]["action_id"] is None
    assert result["total_estimated_arithmetic_flops"] == 1360
    assert all(c["gain"] is None for c in result["labels"]["candidates"].values())


def test_uncertified_reference_keeps_work_and_excludes_posterior_labels(monkeypatch):
    config, _ = _fake_contractions(monkeypatch, overrides={"exact_rows": [0.1, 0.7, 0.1, 0.1]})
    result = pilot.evaluate_shot(_shot(), config)
    assert not result["reference"]["certified"]
    assert result["reference"]["exception_type"] == "ValueError"
    assert result["total_estimated_arithmetic_flops"] == 1360
    assert result["labels"]["actions"]["columns_chi2"]["total_variation"] is None


def test_summary_uses_shots_and_equal_rate_means_and_blocks_invalid_reference(monkeypatch):
    config, _ = _fake_contractions(monkeypatch)
    first = pilot.evaluate_shot(_shot(), config)
    second = pilot.evaluate_shot(_shot(index=1), config)
    config, _ = _fake_contractions(monkeypatch, invalid=("exact_rows", "rows_chi2"))
    third = pilot.evaluate_shot(_shot(rate=0.15), config)
    summary = pilot.summarize_evaluations([first, second, third], config)

    assert summary["statistical_unit"] == "physical_shot"
    assert summary["attempted_shots"] == 3
    assert summary["certified_reference_shots"] == 2
    assert summary["invalid_reference_shots"] == 1
    assert summary["per_rate"][0]["attempted_shots"] == 2
    assert summary["equal_rate"]["actions"]["rows_chi2"]["physical_failure_rate"] == 0.5
    assert summary["equal_rate"]["actions"]["rows_chi2"]["total_variation_mean"] is None
    assert not summary["advancement"]["margin_gate_clause"]
    assert not summary["advancement"]["heterogeneous_efficiency_clause"]
    assert "bootstrap" not in summary


def test_certificate_and_spectral_summaries_use_only_completed_contractions(monkeypatch):
    config, calls = _fake_contractions(monkeypatch)
    certificate_calls = []
    observed_spectra = []
    certify = pilot.certify_reference
    summarize = pilot.compact_spectral_summary

    def certificate(columns, rows):
        certificate_calls.append((columns, rows))
        return certify(columns, rows)

    def spectral(work, *, available):
        action_id = work["contraction_sweeps"][0]["marker"]
        assert calls[-1] == action_id
        observed_spectra.append(action_id)
        return summarize(work, available=available)

    monkeypatch.setattr(pilot, "certify_reference", certificate)
    monkeypatch.setattr(pilot, "compact_spectral_summary", spectral)
    pilot.evaluate_shot(_shot(), config)
    assert len(certificate_calls) == 1
    assert observed_spectra == ["exact_columns", "exact_rows", *ACTION_IDS]


def test_frozen_gate_acceptance_suppresses_offline_escalation_oracles(monkeypatch):
    config, _ = _fake_contractions(
        monkeypatch,
        overrides={
            "columns_tol01": [0.7, 0.1, 0.1, 0.1],
            "rows_tol01": [0.7, 0.1, 0.1, 0.1],
        },
    )
    result = pilot.evaluate_shot(_shot(), config)
    assert result["gate"] == {"accepted": True, "escalated": False}
    assert result["labels"]["accuracy_oracle"]["action_id"] is None
    assert result["labels"]["efficiency_oracle"]["action_id"] is None


def test_full_pilot_advancement_and_any_invalid_reference_veto(monkeypatch):
    import copy

    config, _ = _fake_contractions(monkeypatch)
    example = pilot.evaluate_shot(_shot(), config)
    results = []
    for rate in config.required_error_rates:
        for index in range(64):
            result = copy.deepcopy(example)
            result["shot"].update(shot_id=f"{rate}-{index}", error_rate=rate)
            result["gate"] = {"accepted": index < 32, "escalated": index >= 32}
            result["labels"]["efficiency_oracle"] = {
                "action_id": "columns_chi2" if rate == 0.1 else "rows_chi2",
                "gain": 0.1,
                "efficiency": 0.1,
                "composite_work": 1,
                "uniquely_separated": index >= 32,
            }
            results.append(result)
    summary = pilot.summarize_evaluations(results, config)
    assert summary["advancement"]["margin_gate_clause"]
    assert summary["advancement"]["heterogeneous_efficiency_clause"]
    results[0]["reference"]["certified"] = False
    summary = pilot.summarize_evaluations(results, config)
    assert not summary["advancement"]["margin_gate_clause"]
    assert not summary["advancement"]["heterogeneous_efficiency_clause"]


def test_evaluator_validates_join_and_summary_rejects_repeated_shots(monkeypatch):
    config, calls = _fake_contractions(monkeypatch)
    shot = _shot()
    shot["syndrome"][0] = 1
    with pytest.raises(ValueError, match="not joined"):
        pilot.evaluate_shot(shot, config)
    assert not calls
    result = pilot.evaluate_shot(_shot(), config)
    with pytest.raises(ValueError, match="duplicate physical shot"):
        pilot.summarize_evaluations([result, result], config)


def test_summary_reports_work_and_candidate_gain_distributions_by_rate(monkeypatch):
    config, _ = _fake_contractions(
        monkeypatch,
        overrides={
            "columns_chi2": [0.7, 0.1, 0.1, 0.1],
            "rows_chi2": [0.3, 0.4, 0.2, 0.1],
        },
    )
    results = [pilot.evaluate_shot(_shot(rate), config) for rate in config.required_error_rates]
    summary = pilot.summarize_evaluations(results, config)
    first = summary["per_rate"][0]
    assert first["actions"]["columns_chi2"]["estimated_arithmetic_flops_values"] == [30]
    assert first["candidates"]["columns_chi2"]["gain_values"] == pytest.approx([0.3])
    assert first["candidates"]["rows_chi2"]["gain_values"] == pytest.approx([-0.1])
    assert first["candidates"]["columns_chi2"]["observed_shots"] == 1
    assert summary["equal_rate"]["candidates"]["columns_chi2"]["gain_mean"] == pytest.approx(0.3)
    assert first["candidates"]["columns_chi2"]["composite_work_values"] == [300]
