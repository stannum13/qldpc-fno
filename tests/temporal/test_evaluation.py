import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

import qldpc_fno.temporal.evaluation as evaluation_module
from qldpc_fno.decoders.bplsd import DecodeBatchResult
from qldpc_fno.models.causal_forecaster import build_forecaster, parameter_accounting
from qldpc_fno.temporal.baselines import (
    CircularLogisticForecaster,
    PrivilegedOracle,
    StationaryForecaster,
)
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.evaluation import (
    CausalEvaluationBatch,
    ForecastSplit,
    evaluate_causal_arms,
    evaluate_oracle_sanity,
    fit_select_observation_baselines,
    freeze_learned_arm,
    reduced_factor_diagnostics,
    reduced_progression,
)
from qldpc_fno.training.causal_sequence import (
    CausalTrainingResult,
    SequenceRoleBatch,
    validate_role_partition,
)

CONFIG_PATH = Path("configs/causal_fno_hippo_reduced.json")


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _role_batch(role: str, identity: str) -> SequenceRoleBatch:
    return SequenceRoleBatch(
        role=role,
        seed=1,
        syndromes=torch.zeros(1, 2, 21, 45),
        targets=torch.zeros(1, 2, 58, 45),
        scored_mask=torch.tensor([False, True]),
        sequence_ids=(identity,),
    )


def _partition_batches():
    train = _role_batch("train", _identity("train"))
    validation = _role_batch("validation", _identity("validation"))
    calibration = _role_batch("calibration", _identity("calibration"))
    partition = validate_role_partition(train, validation, calibration)
    return train, validation, calibration, partition


def _forecast_split(role: str, identity: str, target: float = 0.0) -> ForecastSplit:
    return ForecastSplit(
        role=role,
        sequence_ids=(identity,),
        syndromes=np.zeros((1, 2, 2, 5), dtype=np.uint8),
        targets=np.full((1, 2, 3, 5), target, dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def _logistic(probability: float, *, feature_kind: str) -> CircularLogisticForecaster:
    channels = 2 if feature_kind == "ewma" else 64
    return CircularLogisticForecaster(
        weight=np.zeros((3, channels, 5 if feature_kind == "ewma" else 3)),
        bias=np.full(3, _logit(probability)),
        l2=1e-4,
        feature_kind=feature_kind,
        decay=0.9 if feature_kind == "ewma" else None,
        lags=32,
    )


def _fit_selection(monkeypatch, *, ewma_probability: float, logistic_probability: float):
    train, validation_batch, calibration, partition = _partition_batches()
    stationary = StationaryForecaster(0.1, np.full((3, 5), 0.1), 0.0)
    ewma = _logistic(ewma_probability, feature_kind="ewma")
    logistic = _logistic(logistic_probability, feature_kind="lagged")
    monkeypatch.setattr(evaluation_module, "fit_stationary", lambda *args, **kwargs: stationary)
    monkeypatch.setattr(evaluation_module, "fit_ewma", lambda *args, **kwargs: ewma)
    monkeypatch.setattr(evaluation_module, "fit_logistic_ar", lambda *args, **kwargs: logistic)
    monkeypatch.setattr(evaluation_module, "calibrate_temperature", lambda *args: 1.0)
    selection = fit_select_observation_baselines(
        train=_forecast_split("train", train.sequence_ids[0]),
        validation=_forecast_split("validation", validation_batch.sequence_ids[0], 1.0),
        calibration=_forecast_split("calibration", calibration.sequence_ids[0]),
        partition=partition,
    )
    return selection, partition, (stationary, ewma, logistic)


def _decode_result(
    corrections: np.ndarray, predicted: np.ndarray, valid: np.ndarray
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


def test_baseline_selection_uses_raw_unclipped_validation_nll(monkeypatch) -> None:
    selection, partition, _ = _fit_selection(
        monkeypatch, ewma_probability=0.40, logistic_probability=0.45
    )

    assert selection.selected_name == "logistic_ar"
    assert selection.partition_digest == partition.digest
    assert selection.validation_nll["logistic_ar"] < selection.validation_nll["ewma"]
    assert len(selection.fit_policy_digest) == 64


def test_baseline_selection_pins_exact_fit_policy_and_handles_zero_stationary_probability(
    monkeypatch,
) -> None:
    train, validation_batch, calibration, partition = _partition_batches()
    calls: dict[str, dict[str, object]] = {}
    stationary = StationaryForecaster(0.0, np.zeros((3, 5)), 0.0)

    def stationarity(*args, **kwargs):
        calls["stationary"] = kwargs
        return stationary

    def ewma(*args, **kwargs):
        calls["ewma"] = kwargs
        return _logistic(0.2, feature_kind="ewma")

    def logistic(*args, **kwargs):
        calls["logistic_ar"] = kwargs
        return _logistic(0.3, feature_kind="lagged")

    monkeypatch.setattr(evaluation_module, "fit_stationary", stationarity)
    monkeypatch.setattr(evaluation_module, "fit_ewma", ewma)
    monkeypatch.setattr(evaluation_module, "fit_logistic_ar", logistic)
    monkeypatch.setattr(evaluation_module, "calibrate_temperature", lambda *args: 1.0)
    selection = fit_select_observation_baselines(
        train=_forecast_split("train", train.sequence_ids[0]),
        validation=_forecast_split("validation", validation_batch.sequence_ids[0]),
        calibration=_forecast_split("calibration", calibration.sequence_ids[0]),
        partition=partition,
    )

    assert np.isfinite(selection.validation_nll["stationary_field"])
    assert calls["ewma"]["max_iter"] == 500
    assert calls["logistic_ar"]["max_iter"] == 500
    assert calls["ewma"]["kernel_size"] == 5
    assert calls["logistic_ar"]["lags"] == 32


def test_baseline_selection_snapshots_models_and_membership(monkeypatch) -> None:
    selection, _, source_models = _fit_selection(
        monkeypatch, ewma_probability=0.20, logistic_probability=0.20
    )
    assert selection.selected_name == "ewma"
    frozen = selection.baseline("ewma")
    observed = _forecast_split("validation", _identity("validation")).syndromes
    before = frozen.predict(observed, sequence_ids=selection.validation_sequence_ids)

    source_models[1].weight[:] = 100.0
    after = frozen.predict(observed, sequence_ids=selection.validation_sequence_ids)

    assert np.array_equal(before, after)
    assert not frozen.weight.flags.writeable
    with pytest.raises(FrozenInstanceError):
        frozen.temperature = 4.0  # type: ignore[misc]


def test_baseline_selection_requires_exact_partition_membership(monkeypatch) -> None:
    train, _, calibration, partition = _partition_batches()
    with pytest.raises(ValueError, match="partition"):
        fit_select_observation_baselines(
            train=_forecast_split("train", train.sequence_ids[0]),
            validation=_forecast_split("validation", _identity("wrong-validation")),
            calibration=_forecast_split("calibration", calibration.sequence_ids[0]),
            partition=partition,
        )


def test_baseline_selection_rejects_a_wrong_predictor_in_a_named_slot(monkeypatch) -> None:
    train, validation, calibration, partition = _partition_batches()
    monkeypatch.setattr(
        evaluation_module,
        "fit_stationary",
        lambda *args, **kwargs: StationaryForecaster(0.1, np.full((3, 5), 0.1), 0.0),
    )
    monkeypatch.setattr(
        evaluation_module,
        "fit_ewma",
        lambda *args, **kwargs: _logistic(0.2, feature_kind="lagged"),
    )
    monkeypatch.setattr(
        evaluation_module,
        "fit_logistic_ar",
        lambda *args, **kwargs: _logistic(0.3, feature_kind="lagged"),
    )
    monkeypatch.setattr(evaluation_module, "calibrate_temperature", lambda *args: 1.0)
    with pytest.raises(TypeError, match="EWMA"):
        fit_select_observation_baselines(
            train=_forecast_split("train", train.sequence_ids[0]),
            validation=_forecast_split("validation", validation.sequence_ids[0]),
            calibration=_forecast_split("calibration", calibration.sequence_ids[0]),
            partition=partition,
        )


def test_freeze_learned_arm_binds_provenance_and_snapshots_state() -> None:
    config = CausalExperimentConfig.from_json(CONFIG_PATH)
    model = build_forecaster(spatial="cnn", temporal="fir", config=config)
    _, validation, calibration, partition = _partition_batches()
    training_result = CausalTrainingResult(
        model_state_dict={
            name: value.detach().clone() for name, value in model.state_dict().items()
        },
        best_epoch=0,
        best_validation_nll=0.1,
        training_nll_history=(0.2,),
        validation_nll_history=(0.1,),
        partition_digest=partition.digest,
    )
    arm = freeze_learned_arm(
        name="cnn_fir",
        model=model,
        config=config,
        partition=partition,
        training_result=training_result,
        calibration=calibration,
        evaluation_sequence_ids=validation.sequence_ids,
    )

    accounting = parameter_accounting(model)
    assert arm.partition_digest == partition.digest
    assert arm.predictor_type == "causal_channel_forecaster:cnn:fir"
    assert arm.stored_parameters == accounting.stored_real_scalars
    assert arm.effective_parameters == accounting.effective_functional_scalars
    assert arm.calibration_temperature > 0.0
    assert len(arm.checkpoint_sha256) == 64 and len(arm.calibration_digest) == 64
    assert len(arm.config_digest) == 64
    checkpoint = arm.checkpoint_sha256
    with torch.no_grad():
        next(model.parameters()).add_(100.0)
    assert checkpoint != evaluation_module._state_dict_sha256(model.state_dict())


def test_freeze_learned_arm_rejects_partition_mismatch() -> None:
    config = CausalExperimentConfig.from_json(CONFIG_PATH)
    model = build_forecaster(spatial="cnn", temporal="fir", config=config)
    _, validation, calibration, partition = _partition_batches()
    result = CausalTrainingResult(
        model_state_dict={
            name: value.detach().clone() for name, value in model.state_dict().items()
        },
        best_epoch=0,
        best_validation_nll=0.1,
        training_nll_history=(),
        validation_nll_history=(0.1,),
        partition_digest=_identity("wrong-partition"),
    )
    with pytest.raises(ValueError, match="partition"):
        freeze_learned_arm(
            name="cnn_fir",
            model=model,
            config=config,
            partition=partition,
            training_result=result,
            calibration=calibration,
            evaluation_sequence_ids=validation.sequence_ids,
        )


def test_frozen_learned_arm_reconstructs_checkpoint_and_predicts_from_syndromes(
    monkeypatch,
) -> None:
    config = CausalExperimentConfig.from_json(CONFIG_PATH)
    model = build_forecaster(spatial="cnn", temporal="fir", config=config)
    _, validation, calibration, partition = _partition_batches()
    result = CausalTrainingResult(
        model_state_dict={
            name: value.detach().clone() for name, value in model.state_dict().items()
        },
        best_epoch=0,
        best_validation_nll=0.1,
        training_nll_history=(),
        validation_nll_history=(0.1,),
        partition_digest=partition.digest,
    )
    arm = freeze_learned_arm(
        name="cnn_fir",
        model=model,
        config=config,
        partition=partition,
        training_result=result,
        calibration=calibration,
        evaluation_sequence_ids=validation.sequence_ids,
    )
    hx = sparse.csr_matrix((945, 2610), dtype=np.uint8)
    logical_x = sparse.csr_matrix((1, 2610), dtype=np.uint8)
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=validation.sequence_ids,
        syndromes=np.zeros((1, 2, 21, 45), dtype=np.uint8),
        errors=np.zeros((1, 2, 58, 45), dtype=np.uint8),
        logical_flips=np.zeros((1, 2, 1), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )

    def fake_decode(*args, **kwargs):
        del args, kwargs
        return _decode_result(
            np.zeros((1, 2610), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            np.ones(1, dtype=bool),
        )

    monkeypatch.setattr(evaluation_module, "decode_bplsd_prior_batch", fake_decode)
    evaluated = evaluate_causal_arms(batch, (arm,), hx=hx, logical_x=logical_x)
    assert evaluated.arms["cnn_fir"].stored_parameters == arm.stored_parameters
    assert evaluated.arms["cnn_fir"].partition_digest == partition.digest


def test_evaluation_reconstructs_all_qec_labels_before_decode(monkeypatch) -> None:
    selection, _, _ = _fit_selection(monkeypatch, ewma_probability=0.2, logistic_probability=0.3)
    corrupted_syndromes = np.zeros((1, 2, 10), dtype=np.uint8)
    corrupted_syndromes[0, 1, 0] = 1
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=selection.validation_sequence_ids,
        syndromes=corrupted_syndromes,
        errors=np.zeros((1, 2, 15), dtype=np.uint8),
        logical_flips=np.zeros((1, 2, 1), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="reconstructed syndromes"):
        evaluate_causal_arms(
            batch,
            (selection.baseline("stationary_field"),),
            hx=sparse.csr_matrix((10, 15), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 15), dtype=np.uint8),
        )

    logically_corrupted = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=selection.validation_sequence_ids,
        syndromes=np.zeros((1, 2, 10), dtype=np.uint8),
        errors=np.zeros((1, 2, 15), dtype=np.uint8),
        logical_flips=np.ones((1, 2, 1), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="reconstructed logical flips"):
        evaluate_causal_arms(
            logically_corrupted,
            (selection.baseline("stationary_field"),),
            hx=sparse.csr_matrix((10, 15), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 15), dtype=np.uint8),
        )


def test_frozen_arms_decode_identical_membership_and_score_modulo_stabilizers(
    monkeypatch,
) -> None:
    selection, _, _ = _fit_selection(monkeypatch, ewma_probability=0.2, logistic_probability=0.3)
    hx = sparse.eye(10, 15, dtype=np.uint8, format="csr")
    logical_x = sparse.csr_matrix(([1], ([0], [14])), shape=(1, 15), dtype=np.uint8)
    errors = np.zeros((1, 2, 15), dtype=np.uint8)
    errors[0, 1, 1] = 1
    errors[0, 1, 14] = 1
    syndromes = np.asarray(hx @ errors.reshape(2, 15).T).T.reshape(1, 2, 2, 5)
    logical = np.asarray(logical_x @ errors.reshape(2, 15).T).T.reshape(1, 2, 1)
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=selection.validation_sequence_ids,
        syndromes=syndromes,
        errors=errors,
        logical_flips=logical,
        scored_mask=np.array([False, True]),
    )
    calls: list[np.ndarray] = []

    def fake_decode(hx_arg, syndromes_arg, logical_arg, *, error_channels, config):
        del hx_arg, logical_arg, error_channels
        calls.append(np.asarray(syndromes_arg).copy())
        assert config == evaluation_module.CANONICAL_BPLSD_CONFIG
        correction = np.zeros((1, 15), dtype=np.uint8)
        correction[0, 1] = 1
        return _decode_result(correction, np.array([[0]], dtype=np.uint8), np.array([True]))

    monkeypatch.setattr(evaluation_module, "decode_bplsd_prior_batch", fake_decode)
    evaluated = evaluate_causal_arms(
        batch,
        (
            selection.baseline("stationary_field"),
            selection.baseline(selection.selected_name),
        ),
        hx=hx,
        logical_x=logical_x,
    )

    assert len(calls) == 2 and np.array_equal(calls[0], calls[1])
    assert evaluated.sequence_membership == ((selection.validation_sequence_ids[0], 1),)
    selected_arm = evaluated.arms[selection.selected_name]
    assert selected_arm.logical_failures.tolist() == [True]
    assert selected_arm.all_syndrome_valid
    assert selected_arm.per_round_outcomes[0]["true_logical_flips"] == (1,)
    assert selected_arm.expected_calibration_error >= 0.0
    assert len(selected_arm.reliability) == 10
    assert evaluated.decoder_config_digest == evaluation_module.CANONICAL_BPLSD_CONFIG_DIGEST
    with pytest.raises(TypeError):
        evaluated.arms[selection.selected_name].per_round_outcomes[0]["round"] = 9

    progression = reduced_progression(
        selection=selection,
        stationary=evaluated.arms["stationary_field"],
        selected=evaluated.arms[selection.selected_name],
    )
    assert progression.regime == "joint_in_basis"
    assert progression.p_value is None and progression.hypothesis_status is None
    with pytest.raises(ValueError, match="integrity"):
        reduced_progression(
            selection=selection,
            stationary=evaluated.arms["stationary_field"],
            selected=replace(selected_arm, overall_nll=-100.0),
        )


def test_same_shape_different_evaluation_ids_are_rejected(monkeypatch) -> None:
    selection, _, _ = _fit_selection(monkeypatch, ewma_probability=0.2, logistic_probability=0.3)
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=(_identity("different"),),
        syndromes=np.zeros((1, 2, 10), dtype=np.uint8),
        errors=np.zeros((1, 2, 15), dtype=np.uint8),
        logical_flips=np.zeros((1, 2, 1), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="evaluation membership"):
        evaluate_causal_arms(
            batch,
            (selection.baseline("ewma"),),
            hx=sparse.csr_matrix((10, 15), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 15), dtype=np.uint8),
        )


def test_dataclass_replace_cannot_forge_a_frozen_arm(monkeypatch) -> None:
    selection, _, _ = _fit_selection(monkeypatch, ewma_probability=0.2, logistic_probability=0.3)
    original = selection.baseline("ewma")
    forged = replace(original, name="forged")
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=selection.validation_sequence_ids,
        syndromes=np.zeros((1, 2, 2, 5), dtype=np.uint8),
        errors=np.zeros((1, 2, 3, 5), dtype=np.uint8),
        logical_flips=np.zeros((1, 2, 1), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="integrity"):
        evaluate_causal_arms(
            batch,
            (forged,),
            hx=sparse.csr_matrix((10, 15), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 15), dtype=np.uint8),
        )


def test_factor_diagnostics_keep_predeclared_h3_sign_without_inference() -> None:
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
    assert diagnostics.in_basis_mean < 0.0 and diagnostics.basis_mismatch_mean < 0.0
    assert diagnostics.same_direction and diagnostics.p_value is None


def test_privileged_oracle_remains_generator_sanity_only() -> None:
    result = evaluate_oracle_sanity(
        np.array([[[[0.1, 0.9]]]]),
        np.array([[[[0.0, 1.0]]]]),
        np.array([[True]]),
    )
    assert result["scope"] == "generator_sanity_only"
    assert result["deployable_competition"] is False
    assert result["decoder_evaluated"] is False

    with pytest.raises(TypeError, match="frozen deployable"):
        evaluate_causal_arms(
            CausalEvaluationBatch(
                regime="joint_in_basis",
                role="validation",
                sequence_ids=(_identity("validation"),),
                syndromes=np.zeros((1, 1, 1), dtype=np.uint8),
                errors=np.zeros((1, 1, 1), dtype=np.uint8),
                logical_flips=np.zeros((1, 1, 1), dtype=np.uint8),
                scored_mask=np.array([True]),
            ),
            (PrivilegedOracle(np.full((1, 1, 1, 1), 0.1)),),  # type: ignore[arg-type]
            hx=sparse.csr_matrix((1, 1), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 1), dtype=np.uint8),
        )
