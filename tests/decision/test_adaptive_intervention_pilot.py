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
_FIXTURE_SHOT_CONFIG = _ROOT / "configs" / "adaptive_intervention_pilot_fixture_shots.json"


@pytest.fixture(autouse=True)
def _guard_identity_parser_oracle_tests(request, monkeypatch):
    """Identity/parser/oracle checks must never derive shots, sample or contract."""
    name = request.node.originalname or request.node.name
    if any(
        part in name
        for part in (
            "config",
            "identity",
            "parser",
            "json_loader",
            "oracle",
            "retired_scientific",
        )
    ):
        from qecsim.models.generic import DepolarizingErrorModel

        def forbidden(*args, **kwargs):
            pytest.fail("identity/parser/oracle test attempted shot derivation or execution")

        monkeypatch.setattr(DepolarizingErrorModel, "generate", forbidden)
        monkeypatch.setattr(pilot, "_shot_seed", forbidden)
        monkeypatch.setattr(pilot, "planar_mps_coset_masses", forbidden)


def test_config_freezes_the_literal_action_table_and_policy_constants() -> None:
    config = load_pilot_config(_PILOT_CONFIG)

    assert config.schema_version == 1
    assert config.pilot_id == "adaptive_intervention_pilot_v3"
    assert config.code_distance == 5
    assert config.noise_model == "qecsim_iid_depolarizing_code_capacity"
    assert config.required_shot_seed_domain == "qldpc-fno/adaptive-intervention-pilot/v3"
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
    assert config.seed_domain == "qldpc-fno/adaptive-intervention-pilot/v3"
    assert config.campaign_seed == CAMPAIGN_SEED == 10044421296420932682
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
                '"campaign_seed": 10044421296420932682',
                '"campaign_seed": 0,\n  "campaign_seed": 10044421296420932682',
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


def _artifact_fixture(tmp_path):
    """Two replayable test rows; never a complete scientific pilot."""
    from qecsim import paulitools as pt
    from qecsim.models.generic import DepolarizingErrorModel

    from qldpc_fno.decision import planar_shot_data as data

    fixture_config = load_shot_config(_FIXTURE_SHOT_CONFIG, non_scientific_fixture=True)
    assert fixture_config.seed_domain == "qldpc-fno/adaptive-intervention-pilot/test-fixture/v1"
    root = data._repository_root()
    commit, dirty = data._git_provenance(root)
    binding = data._file_binding(root, _FIXTURE_SHOT_CONFIG, commit)
    payload = {
        "config": json.loads(_FIXTURE_SHOT_CONFIG.read_text()),
        "provenance": {
            **data._runtime_provenance(root),
            "git_commit": commit,
            "git_dirty": dirty,
            "source_sha256": data._source_hashes(root),
            "qecsim_version": data.qecsim.__version__,
            "config_sha256": binding["sha256"],
            "config_path": binding["path"],
            "config_scope": binding["scope"],
            "config_committed": binding["committed"],
            "config_content": _FIXTURE_SHOT_CONFIG.read_text(),
        },
        "shots": [],
    }
    code = PlanarCode(5, 5)
    for rate in (0.1, 0.15):
        seed = data._shot_seed(payload["config"]["seed_domain"], error_rate=rate, shot_index=0)
        error = DepolarizingErrorModel().generate(code, rate, np.random.default_rng(seed))
        payload["shots"].append(
            {
                "shot_id": f"d5/p{rate:.6f}/i000000",
                "shot_index": 0,
                "sampler_seed": seed,
                "error_rate": rate,
                "error_bsf": error.tolist(),
                "syndrome": pt.bsp(error, code.stabilizers.T).tolist(),
            }
        )
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_runner_validation_replays_joined_fixture_and_requires_full_scientific_count(tmp_path):
    path, payload = _artifact_fixture(tmp_path)
    assert hasattr(pilot, "validate_pilot_shots"), "pilot artifact validator is missing"
    _, shots = pilot.validate_pilot_shots(path, _FIXTURE_SHOT_CONFIG, non_scientific_fixture=True)
    assert len(shots) == 2
    assert shots[0]["shot_id"] == payload["shots"][0]["shot_id"]
    with pytest.raises(ValueError, match="domain"):
        pilot.validate_pilot_shots(path, _FIXTURE_SHOT_CONFIG)


@pytest.mark.parametrize(
    "mutation",
    [
        "domain",
        "count",
        "seed",
        "syndrome",
        "error",
        "source",
        "dependency",
        "config_hash",
        "duplicate",
        "index",
        "extra_field",
        "boolean_schema",
    ],
)
def test_runner_rejects_mutated_shot_artifacts(tmp_path, mutation):
    path, payload = _artifact_fixture(tmp_path)
    assert hasattr(pilot, "validate_pilot_shots"), "pilot artifact validator is missing"
    if mutation == "domain":
        payload["config"]["seed_domain"] += "/wrong"
    elif mutation == "count":
        payload["shots"].pop()
    elif mutation == "seed":
        payload["shots"][0]["sampler_seed"] += 1
    elif mutation == "syndrome":
        payload["shots"][0]["syndrome"][0] ^= 1
    elif mutation == "error":
        payload["shots"][0]["error_bsf"][0] ^= 1
    elif mutation == "source":
        payload["provenance"]["source_sha256"] = {}
    elif mutation == "dependency":
        payload["provenance"]["dependencies"]["numpy"] = "wrong"
    elif mutation == "config_hash":
        payload["provenance"]["config_sha256"] = "0" * 64
    elif mutation == "duplicate":
        payload["shots"][1] = payload["shots"][0]
    elif mutation == "index":
        payload["shots"][0]["shot_index"] = True
    elif mutation == "boolean_schema":
        payload["config"]["schema_version"] = True
    else:
        payload["config"]["extra"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        pilot.validate_pilot_shots(path, _FIXTURE_SHOT_CONFIG, non_scientific_fixture=True)


def test_historical_domains_include_every_declared_seed_domain_and_reject_overlap(tmp_path):
    assert hasattr(pilot, "historical_domain_bindings"), "historical domain audit is missing"
    bindings = pilot.historical_domain_bindings(_ROOT)
    declared = {
        json.loads(path.read_text())["seed_domain"]
        for path in (_ROOT / "configs").glob("*.json")
        if "seed_domain" in json.loads(path.read_text()) and path != _SHOT_CONFIG
    }
    assert declared <= {domain for binding in bindings.values() for domain in binding["domains"]}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "future_confirmation.json").write_text(
        json.dumps(
            {
                "required_data_seed_domain": "qldpc-fno/adaptive-intervention-pilot/v3",
            }
        )
    )
    with pytest.raises(ValueError, match="domain overlap"):
        pilot.historical_domain_bindings(tmp_path)


def test_scientific_provenance_rejects_dirty_and_uncommitted_inputs(tmp_path, monkeypatch):
    assert hasattr(pilot, "pilot_provenance"), "runner provenance is missing"
    path, _ = _artifact_fixture(tmp_path)
    from qldpc_fno.decision import planar_shot_data as data

    commit, _ = data._git_provenance(_ROOT)
    monkeypatch.setattr(pilot, "_git_provenance", lambda root: (commit, True))
    with pytest.raises(ValueError, match="clean committed"):
        pilot.pilot_provenance(_PILOT_CONFIG, _SHOT_CONFIG, path, scientific=True)
    monkeypatch.setattr(pilot, "_git_provenance", lambda root: (commit, False))
    external_config = tmp_path / "pilot.json"
    external_config.write_bytes(_PILOT_CONFIG.read_bytes())
    with pytest.raises(ValueError, match="clean committed|committed scientific"):
        pilot.pilot_provenance(external_config, _SHOT_CONFIG, path, scientific=True)


def test_runner_binds_summary_and_all_work_without_physical_vectors(tmp_path, monkeypatch):
    assert hasattr(pilot, "run_adaptive_intervention_pilot"), "pilot runner is missing"
    from qldpc_fno.artifacts import sha256_file

    path, _ = _artifact_fixture(tmp_path)
    _fake_contractions(monkeypatch, invalid=("rows_chi2",))
    out = tmp_path / "result"
    result = pilot.run_adaptive_intervention_pilot(
        _PILOT_CONFIG,
        path,
        out,
        non_scientific_fixture=True,
    )
    summary = json.loads((out / "summary.json").read_text())
    assert result["status"] == "reduced_non_scientific"
    assert summary["raw_sha256"] == sha256_file(out / "intervention_pilot.json")
    assert summary["provenance"]["inputs"]["shots"]["sha256"] == sha256_file(path)
    assert summary["provenance"]["inputs"]["config"]["sha256"] == sha256_file(_PILOT_CONFIG)
    assert summary["summary"] == result["summary"]
    assert summary["summary"]["attempted_shots"] == 2
    assert not any(
        summary["summary"]["advancement"][key]
        for key in (
            "complete_pilot",
            "margin_gate_clause",
            "heterogeneous_efficiency_clause",
        )
    )
    for rate in summary["summary"]["per_rate"]:
        assert rate["actions"]["rows_chi2"]["attempted_shots"] == 1
        assert rate["actions"]["rows_chi2"]["physical_failure_rate"] == 1
    text = (out / "summary.json").read_text()
    for excluded in ('"error"', '"error_bsf"', '"syndrome"', '"recovery"', '"truncation_events"'):
        assert excluded not in text
    with pytest.raises(FileExistsError):
        pilot.run_adaptive_intervention_pilot(_PILOT_CONFIG, path, out, non_scientific_fixture=True)


def test_runner_failure_does_not_publish_partial_output(tmp_path, monkeypatch):
    assert hasattr(pilot, "run_adaptive_intervention_pilot"), "pilot runner is missing"
    path, _ = _artifact_fixture(tmp_path)
    _fake_contractions(monkeypatch, bug=RuntimeError("evaluator bug"))
    out = tmp_path / "result"
    with pytest.raises(RuntimeError, match="evaluator bug"):
        pilot.run_adaptive_intervention_pilot(_PILOT_CONFIG, path, out, non_scientific_fixture=True)
    assert not out.exists()


def test_runner_serialization_failure_is_transactional(tmp_path, monkeypatch):
    assert hasattr(pilot, "run_adaptive_intervention_pilot"), "pilot runner is missing"
    path, _ = _artifact_fixture(tmp_path)
    _fake_contractions(monkeypatch)
    original = pilot._write_pilot_json

    def write(path, payload):
        if path.name == "summary.json":
            raise OSError("disk failure")
        original(path, payload)

    monkeypatch.setattr(pilot, "_write_pilot_json", write)
    out = tmp_path / "result"
    with pytest.raises(OSError, match="disk failure"):
        pilot.run_adaptive_intervention_pilot(_PILOT_CONFIG, path, out, non_scientific_fixture=True)
    assert not out.exists()


def test_runner_rejects_input_mutation_during_validation(tmp_path, monkeypatch):
    path, payload = _artifact_fixture(tmp_path)
    original = pilot.validate_pilot_shots

    def validate(*args, **kwargs):
        result = original(*args, **kwargs)
        payload["shots"][0]["syndrome"][0] ^= 1
        path.write_text(json.dumps(payload))
        return result

    monkeypatch.setattr(pilot, "validate_pilot_shots", validate)
    _fake_contractions(monkeypatch)
    out = tmp_path / "result"
    with pytest.raises(ValueError, match="provenance changed"):
        pilot.run_adaptive_intervention_pilot(_PILOT_CONFIG, path, out, non_scientific_fixture=True)
    assert not out.exists()


def test_fixture_mode_refuses_arbitrary_counts_and_nonboolean_flag(tmp_path):
    path, payload = _artifact_fixture(tmp_path)
    with pytest.raises(TypeError, match="boolean"):
        pilot.validate_pilot_shots(path, _FIXTURE_SHOT_CONFIG, non_scientific_fixture=1)
    payload["shots"] *= 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="shot count"):
        pilot.validate_pilot_shots(path, _FIXTURE_SHOT_CONFIG, non_scientific_fixture=True)


def test_clean_committed_scientific_provenance_binds_external_raw_shots(tmp_path, monkeypatch):
    """Audit clean provenance in an isolated Git repo without executing science."""
    import shutil
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    labels = (
        *pilot._SOURCE_LABELS,
        "src/qldpc_fno/decision/adaptive_intervention_pilot.py",
        "src/qldpc_fno/decision/adaptive_diagnostics.py",
        "experiments/35_run_adaptive_intervention_pilot.py",
        "configs/adaptive_intervention_pilot.json",
        "configs/adaptive_intervention_pilot_shots.json",
        "configs/planar_shot_screen.json",
        "uv.lock",
    )
    for label in labels:
        target = root / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / label, target)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    monkeypatch.setattr(pilot, "_repository_root", lambda: root)
    shots = tmp_path / "ignored-raw.json"
    shots.write_text("{}")
    provenance = pilot.pilot_provenance(
        root / "configs/adaptive_intervention_pilot.json",
        root / "configs/adaptive_intervention_pilot_shots.json",
        shots,
        scientific=True,
    )
    assert provenance["git_dirty"] is False
    assert provenance["git_commit"] == git("rev-parse", "HEAD").stdout.strip()
    assert provenance["inputs"]["config"]["committed"] is True
    assert provenance["inputs"]["shots"]["committed"] is False
    assert len(provenance["source_sha256"]) == len(pilot._SOURCE_LABELS) + 3
    assert provenance["matching_backend"]["lockfile_sha256"]
    assert provenance["dependencies"]["qecsim"]
    (root / "uv.lock").write_text("tampered")
    with pytest.raises(ValueError, match="clean committed"):
        pilot.pilot_provenance(
            root / "configs/adaptive_intervention_pilot.json",
            root / "configs/adaptive_intervention_pilot_shots.json",
            shots,
            scientific=True,
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_pilot_json_loader_rejects_nonstandard_constants(tmp_path, constant):
    path = tmp_path / "nonstandard.json"
    path.write_text('{"provenance": {"unexpected": ' + constant + "}}")
    with pytest.raises(ValueError, match="JSON constant"):
        pilot._load_object(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 123])
def test_runner_rejects_extra_provenance_without_publishing(tmp_path, monkeypatch, value):
    artifact, payload = _artifact_fixture(tmp_path)
    payload["provenance"]["unexpected"] = value
    artifact.write_text(json.dumps(payload))
    _, calls = _fake_contractions(monkeypatch)
    out = tmp_path / "result"
    with pytest.raises(ValueError, match="JSON constant|provenance.*fields"):
        pilot.run_adaptive_intervention_pilot(
            _PILOT_CONFIG,
            artifact,
            out,
            non_scientific_fixture=True,
        )
    assert not calls
    assert not out.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_pilot_writer_rejects_nonfinite_values_before_creating_a_file(tmp_path, value):
    assert hasattr(pilot, "_write_pilot_json"), "strict pilot writer is missing"
    out = tmp_path / "result.json"
    with pytest.raises(ValueError, match="JSON compliant"):
        pilot._write_pilot_json(out, {"nested": {"value": value}})
    assert not out.exists()


def test_runner_nonfinite_result_does_not_publish(tmp_path, monkeypatch):
    artifact, _ = _artifact_fixture(tmp_path)
    _fake_contractions(monkeypatch)
    evaluate = pilot.evaluate_shot

    def corrupted(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        result["reference"]["views"]["columns"]["work"]["unexpected"] = float("nan")
        return result

    monkeypatch.setattr(pilot, "evaluate_shot", corrupted)
    out = tmp_path / "result"
    with pytest.raises(ValueError, match="JSON compliant"):
        pilot.run_adaptive_intervention_pilot(
            _PILOT_CONFIG,
            artifact,
            out,
            non_scientific_fixture=True,
        )
    assert not out.exists()


def test_reference_summaries_reconstruct_each_view_and_equal_rate_work(monkeypatch):
    config, _ = _fake_contractions(monkeypatch)
    records = [pilot.evaluate_shot(_shot(0.1, 0), config)]
    _fake_contractions(monkeypatch, invalid=("exact_rows",))
    records.extend(
        [
            pilot.evaluate_shot(_shot(0.1, 1), config),
            pilot.evaluate_shot(_shot(0.15, 0), config),
        ]
    )
    summary = pilot.summarize_evaluations(records, config)
    assert "references" in summary["equal_rate"], "separate reference work summaries are missing"
    for mode in ("columns", "rows"):
        rate_means, rate_validity = [], []
        for rate_summary in summary["per_rate"]:
            views = [
                r["reference"]["views"][mode]
                for r in records
                if r["shot"]["error_rate"] == rate_summary["error_rate"]
            ]
            work = [view["estimated_arithmetic_flops"] for view in views]
            valid = sum(view["valid"] for view in views)
            invalid_work = sum(
                view["estimated_arithmetic_flops"] for view in views if not view["valid"]
            )
            ref = rate_summary["references"]["exact_" + mode]
            assert ref["attempted_shots"] == len(views)
            assert ref["valid_shots"] == valid
            assert ref["invalid_shots"] == len(views) - valid
            assert ref["validity_rate"] == valid / len(views)
            assert ref["estimated_arithmetic_flops_values"] == work
            assert ref["estimated_arithmetic_flops_total"] == sum(work)
            assert ref["estimated_arithmetic_flops_mean"] == sum(work) / len(work)
            assert ref["invalid_estimated_arithmetic_flops_total"] == invalid_work
            rate_means.append(sum(work) / len(work))
            rate_validity.append(valid / len(views))
        equal = summary["equal_rate"]["references"]["exact_" + mode]
        views = [r["reference"]["views"][mode] for r in records]
        assert equal["attempted_shots"] == len(views)
        assert equal["valid_shots"] == sum(v["valid"] for v in views)
        assert equal["invalid_shots"] == sum(not v["valid"] for v in views)
        assert equal["estimated_arithmetic_flops_values"] == [
            v["estimated_arithmetic_flops"] for v in views
        ]
        assert equal["estimated_arithmetic_flops_weights"] == [0.25, 0.25, 0.5]
        assert equal["estimated_arithmetic_flops_total"] == sum(
            v["estimated_arithmetic_flops"] for v in views
        )
        assert equal["invalid_estimated_arithmetic_flops_total"] == sum(
            v["estimated_arithmetic_flops"] for v in views if not v["valid"]
        )
        assert equal["estimated_arithmetic_flops_mean"] == sum(rate_means) / 2
        assert equal["validity_rate"] == sum(rate_validity) / 2
    assert summary["equal_rate"]["references"]["exact_rows"]["invalid_shots"] == 2
    assert (
        summary["equal_rate"]["references"]["exact_rows"][
            "invalid_estimated_arithmetic_flops_total"
        ]
        == 200
    )


def test_reference_equal_rate_summary_is_null_when_one_rate_is_missing(monkeypatch):
    config, _ = _fake_contractions(monkeypatch)
    summary = pilot.summarize_evaluations([pilot.evaluate_shot(_shot(), config)], config)
    assert "references" in summary["equal_rate"], "separate reference work summaries are missing"
    for reference in summary["equal_rate"]["references"].values():
        assert reference["attempted_shots"] == 1
        assert reference["estimated_arithmetic_flops_mean"] is None
        assert reference["estimated_arithmetic_flops_weights"] is None
        assert reference["validity_rate"] is None


def test_published_compact_reference_work_reconstructs_from_raw(tmp_path, monkeypatch):
    artifact, _ = _artifact_fixture(tmp_path)
    _fake_contractions(monkeypatch, invalid=("exact_rows",))
    out = tmp_path / "result"
    pilot.run_adaptive_intervention_pilot(
        _PILOT_CONFIG,
        artifact,
        out,
        non_scientific_fixture=True,
    )
    raw = json.loads((out / "intervention_pilot.json").read_text())
    compact = json.loads((out / "summary.json").read_text())
    assert "references" in compact["summary"]["equal_rate"], "reference work absent from compact"
    assert compact["summary"] == pilot.summarize_evaluations(
        raw["shots"],
        load_pilot_config(_PILOT_CONFIG),
    )
    for mode in ("columns", "rows"):
        reference = compact["summary"]["equal_rate"]["references"]["exact_" + mode]
        assert reference["estimated_arithmetic_flops_total"] == sum(
            r["reference"]["views"][mode]["estimated_arithmetic_flops"] for r in raw["shots"]
        )
        assert reference["invalid_shots"] == (2 if mode == "rows" else 0)


def test_fixture_config_reserves_its_own_domain_and_both_modes_reject_cross_use():
    assert _FIXTURE_SHOT_CONFIG.is_file(), "reserved test-fixture config is missing"
    config = load_shot_config(_FIXTURE_SHOT_CONFIG, non_scientific_fixture=True)
    assert config.seed_domain == "qldpc-fno/adaptive-intervention-pilot/test-fixture/v1"
    assert config.campaign_seed == 1727822112359709271
    assert config.shots_per_rate == 1
    with pytest.raises(ValueError, match="domain"):
        load_shot_config(_FIXTURE_SHOT_CONFIG)
    with pytest.raises(ValueError, match="domain"):
        load_shot_config(_SHOT_CONFIG, non_scientific_fixture=True)


@pytest.mark.parametrize(
    ("version", "seed"),
    [
        ("v1", 16980117767564665917),
        ("v2", 8908597917812360592),
    ],
)
def test_retired_scientific_domain_is_rejected_without_sampling(tmp_path, version, seed):
    payload = json.loads(_SHOT_CONFIG.read_text())
    payload.update(
        seed_domain=f"qldpc-fno/adaptive-intervention-pilot/{version}",
        campaign_seed=seed,
    )
    path = tmp_path / "retired.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="domain"):
        load_shot_config(path)


@pytest.mark.parametrize(
    ("version", "seed"),
    [
        ("v1", 16980117767564665917),
        ("v2", 8908597917812360592),
        ("v3", 10044421296420932682),
    ],
)
def test_fixture_rejects_scientific_identity_before_artifact_or_seed_use(tmp_path, version, seed):
    payload = json.loads(_SHOT_CONFIG.read_text())
    payload.update(
        seed_domain=f"qldpc-fno/adaptive-intervention-pilot/{version}", campaign_seed=seed
    )
    config_path = tmp_path / "scientific.json"
    config_path.write_text(json.dumps(payload))
    # The nonexistent artifact proves rejection precedes reading rows, while the
    # autouse guard fails any shot seed derivation, sampling or contraction call.
    with pytest.raises(ValueError, match="domain"):
        pilot.validate_pilot_shots(
            tmp_path / "must-not-be-opened.json",
            config_path,
            non_scientific_fixture=True,
        )


def test_accuracy_oracle_compares_runner_up_below_positive_gain_threshold():
    selection = select_accuracy_oracle(
        [
            _opportunity("columns_chi2", 1.1e-6, 100),
            _opportunity("rows_chi2", 0.9e-6, 1),
            _opportunity("columns_chi4", 10, 1, valid=False),
        ]
    )
    assert selection.action_id == "columns_chi2"
    assert not selection.uniquely_separated
    assert select_efficiency_oracle(
        [
            _opportunity("columns_chi2", 1.1e-6, 100),
            _opportunity("rows_chi2", 0.9e-6, 1),
        ]
    ).uniquely_separated


def test_timing_sidecar_binds_outputs_and_captures_every_attempt(tmp_path, monkeypatch):
    from qldpc_fno.artifacts import sha256_file

    artifact, _ = _artifact_fixture(tmp_path)
    _fake_contractions(monkeypatch, invalid=("exact_rows", "rows_chi2"))
    ticks = itertools.count(step=0.25)
    monkeypatch.setattr(pilot, "perf_counter", lambda: next(ticks), raising=False)
    out = tmp_path / "result"
    result = pilot.run_adaptive_intervention_pilot(
        _PILOT_CONFIG,
        artifact,
        out,
        non_scientific_fixture=True,
    )
    assert (out / "timing.json").is_file(), "atomic timing sidecar is missing"
    timing = json.loads((out / "timing.json").read_text())
    assert set(timing) == {
        "schema_version",
        "status",
        "measurement",
        "host",
        "total_host_wall_seconds",
        "raw_sha256",
        "summary_sha256",
        "provenance",
        "shots",
    }
    assert timing["status"] == "nondeterministic_engineering_metadata"
    assert timing["measurement"] == "research_host_wall_time_not_decoder_latency"
    assert timing["raw_sha256"] == sha256_file(out / "intervention_pilot.json")
    assert timing["summary_sha256"] == sha256_file(out / "summary.json")
    assert timing["provenance"] == result["provenance"]
    assert set(timing["host"]) == {"node", "platform", "machine"}
    assert timing["total_host_wall_seconds"] > 32 * 0.25
    assert len(timing["shots"]) == 2
    for timed, raw in zip(timing["shots"], result["shots"], strict=True):
        assert timed["shot_id"] == raw["shot"]["shot_id"]
        assert timed["error_rate"] == raw["shot"]["error_rate"]
        assert timed["shot_index"] == raw["shot"]["shot_index"]
        assert [attempt["action_id"] for attempt in timed["actions"]] == [
            "exact_columns",
            "exact_rows",
            *ACTION_IDS,
        ]
        for attempt in timed["actions"]:
            assert set(attempt) == {
                "action_id",
                "valid",
                "wall_seconds",
                "exception_type",
                "unavailable_reason",
            }
            invalid = attempt["action_id"] in {"exact_rows", "rows_chi2"}
            assert attempt["valid"] is not invalid
            assert attempt["wall_seconds"] == (None if invalid else 0.25)
            assert attempt["exception_type"] == ("InvalidContractionError" if invalid else None)
            assert attempt["unavailable_reason"] == ("invalid_contraction" if invalid else None)
    for name in ("intervention_pilot.json", "summary.json"):
        text = (out / name).read_text()
        assert '"wall_seconds"' not in text
        assert '"timing"' not in text
        assert '"total_host_wall_seconds"' not in text


def test_timing_write_failure_publishes_none_of_the_three_artifacts(tmp_path, monkeypatch):
    artifact, _ = _artifact_fixture(tmp_path)
    _fake_contractions(monkeypatch)
    write = pilot._write_pilot_json

    def fail_timing(path, payload):
        if path.name == "timing.json":
            raise OSError("timing disk failure")
        write(path, payload)

    monkeypatch.setattr(pilot, "_write_pilot_json", fail_timing)
    out = tmp_path / "result"
    with pytest.raises(OSError, match="timing disk failure"):
        pilot.run_adaptive_intervention_pilot(
            _PILOT_CONFIG,
            artifact,
            out,
            non_scientific_fixture=True,
        )
    assert not out.exists()
    assert not list(tmp_path.glob(".result-*"))
