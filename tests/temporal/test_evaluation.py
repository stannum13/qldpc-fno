from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse

import qldpc_fno.temporal.evaluation as evaluation_module
from qldpc_fno.decoders.bplsd import DecodeBatchResult
from qldpc_fno.temporal.evaluation import (
    CausalEvaluationBatch,
    ForecastSplit,
    evaluate_causal_arms,
    evaluate_oracle_sanity,
    fit_select_observation_baselines,
    reduced_factor_diagnostics,
    reduced_progression,
)


def _decode_result(
    corrections: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
) -> DecodeBatchResult:
    rows = corrections.shape[0]
    return DecodeBatchResult(
        corrections=corrections,
        predicted_observables=predicted,
        syndrome_valid=valid,
        converged=np.ones(rows, dtype=bool),
        iterations=np.arange(rows, dtype=np.int64) + 1,
        setup_latency_seconds=np.full(rows, 0.1),
        decode_latency_seconds=np.full(rows, 0.2),
        latency_seconds=np.full(rows, 0.3),
    )


def test_all_arms_decode_identical_membership_and_score_logical_failure(monkeypatch) -> None:
    hx = sparse.csr_matrix([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[0, 0, 1]], dtype=np.uint8)
    syndromes = np.array(
        [
            [[0, 0], [1, 0], [0, 1]],
            [[1, 1], [0, 0], [1, 0]],
        ],
        dtype=np.uint8,
    )
    logical_flips = np.array(
        [
            [[0], [0], [1]],
            [[0], [1], [0]],
        ],
        dtype=np.uint8,
    )
    errors = np.zeros((2, 3, 3), dtype=np.uint8)
    errors[..., 2] = logical_flips[..., 0]
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=("d" * 64, "e" * 64),
        syndromes=syndromes,
        errors=errors,
        logical_flips=logical_flips,
        scored_mask=np.array([False, True, True]),
    )
    calls: list[np.ndarray] = []

    def fake_decode(hx_arg, syndromes_arg, logical_arg, *, error_channels, config):
        del hx_arg, logical_arg, config
        calls.append(np.asarray(syndromes_arg).copy())
        corrections = np.column_stack(
            (syndromes_arg[:, 0], syndromes_arg[:, 1], np.array([0, 0, 1, 1]))
        ).astype(np.uint8)
        predicted = corrections[:, 2, None]
        return _decode_result(corrections, predicted, np.ones(4, dtype=bool))

    monkeypatch.setattr(evaluation_module, "decode_bplsd_prior_batch", fake_decode)
    priors = {
        "stationary_field": np.full((2, 3, 3), 0.05),
        "fno_hippo": np.full((2, 3, 3), 0.06),
    }

    result = evaluate_causal_arms(
        batch,
        priors,
        hx=hx,
        logical_x=logical_x,
        parameter_counts={
            "stationary_field": (4, 4),
            "fno_hippo": (12, 10),
        },
    )

    expected_membership = (
        ("d" * 64, 1),
        ("d" * 64, 2),
        ("e" * 64, 1),
        ("e" * 64, 2),
    )
    assert len(calls) == 2
    assert np.array_equal(calls[0], calls[1])
    assert result.sequence_membership == expected_membership
    assert result.arms["stationary_field"].sequence_membership == expected_membership
    assert result.arms["fno_hippo"].sequence_membership == expected_membership
    assert result.arms["fno_hippo"].logical_failures.tolist() == [False, True, False, True]
    assert result.arms["fno_hippo"].all_syndrome_valid
    assert result.arms["fno_hippo"].stored_parameters == 12
    assert result.arms["fno_hippo"].effective_parameters == 10
    assert len(result.arms["fno_hippo"].per_round_outcomes) == 4
    assert result.arms["fno_hippo"].per_round_outcomes[1]["predicted_observables"] == [0]
    assert result.arms["fno_hippo"].per_round_outcomes[1]["true_logical_flips"] == [1]
    assert result.arms["fno_hippo"].per_round_outcomes[1]["observed_error_weight"] == 1
    assert result.arms["fno_hippo"].per_round_outcomes[1]["forecast_nll"] > 0.0


def test_invalid_correction_is_a_failure_and_decoder_validity_is_recomputed(monkeypatch) -> None:
    hx = sparse.csr_matrix([[1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[0, 1]], dtype=np.uint8)
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=("d" * 64,),
        syndromes=np.array([[[1]]], dtype=np.uint8),
        errors=np.zeros((1, 1, 2), dtype=np.uint8),
        logical_flips=np.zeros((1, 1, 1), dtype=np.uint8),
        scored_mask=np.array([True]),
    )

    def fake_decode(*args, **kwargs):
        del args, kwargs
        return _decode_result(
            np.array([[0, 0]], dtype=np.uint8),
            np.array([[0]], dtype=np.uint8),
            np.array([True]),
        )

    monkeypatch.setattr(evaluation_module, "decode_bplsd_prior_batch", fake_decode)

    with pytest.raises(RuntimeError, match="syndrome_valid disagrees"):
        evaluate_causal_arms(
            batch,
            {"arm": np.full((1, 1, 2), 0.05)},
            hx=hx,
            logical_x=logical_x,
            parameter_counts={"arm": (1, 1)},
        )


def test_invalid_correction_counts_as_failure_even_when_logical_bits_match(monkeypatch) -> None:
    hx = sparse.csr_matrix([[1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[0, 1]], dtype=np.uint8)
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=("d" * 64,),
        syndromes=np.array([[[1]]], dtype=np.uint8),
        errors=np.zeros((1, 1, 2), dtype=np.uint8),
        logical_flips=np.zeros((1, 1, 1), dtype=np.uint8),
        scored_mask=np.array([True]),
    )

    def fake_decode(*args, **kwargs):
        del args, kwargs
        return _decode_result(
            np.array([[0, 0]], dtype=np.uint8),
            np.array([[0]], dtype=np.uint8),
            np.array([False]),
        )

    monkeypatch.setattr(evaluation_module, "decode_bplsd_prior_batch", fake_decode)
    result = evaluate_causal_arms(
        batch,
        {"arm": np.full((1, 1, 2), 0.05)},
        hx=hx,
        logical_x=logical_x,
        parameter_counts={"arm": (1, 1)},
    )

    assert result.arms["arm"].logical_failures.tolist() == [True]
    assert not result.arms["arm"].all_syndrome_valid


@dataclass(frozen=True)
class _FakeForecaster:
    probability: float
    feature_kind: str = "fake"
    privileged: bool = False
    temperature: float = 1.0

    def predict(self, syndromes: np.ndarray) -> np.ndarray:
        sequences, rounds, _, ell = syndromes.shape
        return np.full((sequences, rounds, 1, ell), self.probability)

    def raw_predict(self, syndromes: np.ndarray) -> np.ndarray:
        return self.predict(syndromes)


def _forecast_split(role: str, target: float) -> ForecastSplit:
    identity = {"train": "a", "validation": "b", "calibration": "c"}[role] * 64
    return ForecastSplit(
        role=role,
        sequence_ids=(identity,),
        syndromes=np.zeros((1, 2, 1, 3)),
        targets=np.full((1, 2, 1, 3), target),
        scored_mask=np.array([[True, True]]),
    )


def test_baseline_selection_fits_on_roles_and_ties_choose_ewma(monkeypatch) -> None:
    seen: list[tuple[str, float]] = []

    def fake_stationary(train_targets, validation_targets, **kwargs):
        del train_targets, validation_targets, kwargs
        return evaluation_module.StationaryForecaster(0.1, np.full((1, 3), 0.1), 0.0)

    def fake_ewma(train_s, train_y, validation_s, validation_y, **kwargs):
        del train_s, validation_s, validation_y, kwargs
        seen.append(("ewma", float(np.mean(train_y))))
        return _FakeForecaster(0.2, "ewma")

    def fake_logistic(train_s, train_y, validation_s, validation_y, **kwargs):
        del train_s, validation_s, validation_y, kwargs
        seen.append(("logistic_ar", float(np.mean(train_y))))
        return _FakeForecaster(0.2, "lagged")

    monkeypatch.setattr(evaluation_module, "fit_stationary", fake_stationary)
    monkeypatch.setattr(evaluation_module, "fit_ewma", fake_ewma)
    monkeypatch.setattr(evaluation_module, "fit_logistic_ar", fake_logistic)
    monkeypatch.setattr(evaluation_module, "calibrate_temperature", lambda *args: 2.0)

    selection = fit_select_observation_baselines(
        train=_forecast_split("train", 1.0),
        validation=_forecast_split("validation", 0.2),
        calibration=_forecast_split("calibration", 0.8),
        max_iter=2,
    )

    assert seen == [("ewma", 1.0), ("logistic_ar", 1.0)]
    assert selection.selected_name == "ewma"
    assert selection.validation_nll["ewma"] == pytest.approx(
        selection.validation_nll["logistic_ar"]
    )
    assert set(selection.models) == {"stationary_field", "ewma", "logistic_ar"}
    assert selection.frozen
    assert selection.models["ewma"].temperature == 2.0


def test_baseline_fit_rejects_provenance_overlap_between_roles() -> None:
    train = _forecast_split("train", 0.0)
    validation = ForecastSplit(
        role="validation",
        sequence_ids=train.sequence_ids,
        syndromes=np.zeros((1, 2, 1, 3)),
        targets=np.zeros((1, 2, 1, 3)),
        scored_mask=np.array([[True, True]]),
    )

    with pytest.raises(ValueError, match="disjoint"):
        fit_select_observation_baselines(
            train=train,
            validation=validation,
            calibration=_forecast_split("calibration", 0.0),
            max_iter=2,
        )


def test_baseline_winner_uses_uncalibrated_validation_nll(monkeypatch) -> None:
    def fake_stationary(*args, **kwargs):
        del args, kwargs
        return evaluation_module.StationaryForecaster(0.1, np.full((1, 3), 0.1), 0.0)

    def fake_ewma(*args, **kwargs):
        del args, kwargs
        return _FakeForecaster(0.2, "ewma")

    def fake_logistic(*args, **kwargs):
        del args, kwargs
        return _FakeForecaster(0.3, "lagged")

    monkeypatch.setattr(evaluation_module, "fit_stationary", fake_stationary)
    monkeypatch.setattr(evaluation_module, "fit_ewma", fake_ewma)
    monkeypatch.setattr(evaluation_module, "fit_logistic_ar", fake_logistic)
    temperatures = iter((1.0, 100.0, 0.01))
    monkeypatch.setattr(
        evaluation_module,
        "calibrate_temperature",
        lambda *args: next(temperatures),
    )

    selection = fit_select_observation_baselines(
        train=_forecast_split("train", 1.0),
        validation=_forecast_split("validation", 0.2),
        calibration=_forecast_split("calibration", 0.8),
        max_iter=2,
    )

    assert selection.selected_name == "ewma"


def test_reduced_progression_is_strict_nll_and_non_worse_bler_without_inference() -> None:
    passed = reduced_progression(
        regime="joint_in_basis",
        selected_name="ewma",
        stationary_nll=np.array([0.4, 0.5]),
        selected_nll=np.array([0.3, 0.4]),
        stationary_bler=np.array([0.2, 0.1]),
        selected_bler=np.array([0.2, 0.1]),
    )
    tied_nll = reduced_progression(
        regime="joint_basis_mismatch",
        selected_name="logistic_ar",
        stationary_nll=np.array([0.4]),
        selected_nll=np.array([0.4]),
        stationary_bler=np.array([0.2]),
        selected_bler=np.array([0.1]),
    )

    assert passed.progressed
    assert not tied_nll.progressed
    assert passed.scope == "descriptive_reduced_non_scientific"
    assert passed.p_value is None
    assert passed.hypothesis_status is None


def test_factor_diagnostics_report_crossed_interaction_and_mismatch_direction() -> None:
    diagnostics = reduced_factor_diagnostics(
        in_basis_losses={
            "cnn_fir": np.array([0.6, 0.5]),
            "fno_fir": np.array([0.5, 0.4]),
            "cnn_hippo": np.array([0.5, 0.4]),
            "fno_hippo": np.array([0.3, 0.2]),
        },
        basis_mismatch_losses={
            "cnn_fir": np.array([0.7, 0.6]),
            "fno_fir": np.array([0.6, 0.5]),
            "cnn_hippo": np.array([0.6, 0.5]),
            "fno_hippo": np.array([0.45, 0.35]),
        },
    )

    assert diagnostics.in_basis_interaction.tolist() == pytest.approx([-0.1, -0.1])
    assert diagnostics.in_basis_mean < 0.0
    assert diagnostics.basis_mismatch_mean < 0.0
    assert diagnostics.same_direction
    assert diagnostics.p_value is None


def test_privileged_oracle_is_generator_sanity_only() -> None:
    targets = np.array([[[[0.0, 1.0]]]])
    result = evaluate_oracle_sanity(
        np.array([[[[0.1, 0.9]]]]),
        targets,
        np.array([[True]]),
    )

    assert result["scope"] == "generator_sanity_only"
    assert result["deployable_competition"] is False
    assert result["decoder_evaluated"] is False
    assert result["overall_nll"] > 0.0


def test_privileged_oracle_cannot_enter_decoder_competition() -> None:
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=("d" * 64,),
        syndromes=np.zeros((1, 1, 1), dtype=np.uint8),
        errors=np.zeros((1, 1, 2), dtype=np.uint8),
        logical_flips=np.zeros((1, 1, 1), dtype=np.uint8),
        scored_mask=np.array([True]),
    )

    with pytest.raises(ValueError, match="oracle"):
        evaluate_causal_arms(
            batch,
            {"privileged_oracle": np.full((1, 1, 2), 0.05)},
            hx=sparse.csr_matrix([[1, 0]], dtype=np.uint8),
            logical_x=sparse.csr_matrix([[0, 1]], dtype=np.uint8),
            parameter_counts={"privileged_oracle": (0, 0)},
        )
