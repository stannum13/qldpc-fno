import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

import qldpc_fno.temporal.evaluation as evaluation_module
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
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


@pytest.fixture(scope="module")
def canonical_geometry():
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    return code.hx, code.hz, logical_x_basis(code.hx, code.hz)


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _role_batch(
    role: str,
    identity: str,
    *,
    target: float = 0.0,
    syndromes: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> SequenceRoleBatch:
    return SequenceRoleBatch(
        role=role,
        seed=1,
        syndromes=torch.as_tensor(
            np.zeros((1, 2, 21, 45), dtype=np.uint8) if syndromes is None else syndromes
        ).float(),
        targets=torch.as_tensor(
            np.full((1, 2, 58, 45), target, dtype=np.uint8) if targets is None else targets
        ).float(),
        scored_mask=torch.tensor([False, True]),
        sequence_ids=(identity,),
    )


def _partition_batches():
    train = _role_batch("train", _identity("train"))
    validation = _role_batch("validation", _identity("validation"))
    calibration = _role_batch("calibration", _identity("calibration"))
    partition = validate_role_partition(train, validation, calibration, regime="joint_in_basis")
    return train, validation, calibration, partition


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def _logistic(probability: float, *, feature_kind: str) -> CircularLogisticForecaster:
    channels = 21 if feature_kind == "ewma" else 21 * 32
    return CircularLogisticForecaster(
        weight=np.zeros((58, channels, 5 if feature_kind == "ewma" else 3)),
        bias=np.full(58, _logit(probability)),
        l2=1e-4,
        feature_kind=feature_kind,
        decay=0.9 if feature_kind == "ewma" else None,
        lags=32,
    )


def _fit_selection(
    monkeypatch,
    *,
    ewma_probability: float,
    logistic_probability: float,
    validation_target: float = 0.0,
    validation_syndromes: np.ndarray | None = None,
    validation_targets: np.ndarray | None = None,
):
    train = _role_batch("train", _identity("train"))
    validation_batch = _role_batch(
        "validation",
        _identity("validation"),
        target=validation_target,
        syndromes=validation_syndromes,
        targets=validation_targets,
    )
    calibration = _role_batch("calibration", _identity("calibration"))
    partition = validate_role_partition(
        train, validation_batch, calibration, regime="joint_in_basis"
    )
    stationary = StationaryForecaster(0.1, np.full((58, 45), 0.1), 0.0)
    ewma = _logistic(ewma_probability, feature_kind="ewma")
    logistic = _logistic(logistic_probability, feature_kind="lagged")
    monkeypatch.setattr(evaluation_module, "fit_stationary", lambda *args, **kwargs: stationary)
    monkeypatch.setattr(evaluation_module, "fit_ewma", lambda *args, **kwargs: ewma)
    monkeypatch.setattr(evaluation_module, "fit_logistic_ar", lambda *args, **kwargs: logistic)
    monkeypatch.setattr(evaluation_module, "calibrate_temperature", lambda *args: 1.0)
    selection = fit_select_observation_baselines(
        train=train,
        validation=validation_batch,
        calibration=calibration,
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


def _arm_evaluation(
    name: str,
    predictor_type: str,
    regime: str,
    *,
    bler: tuple[float, ...],
    identities: tuple[str, ...] = (_identity("validation-0"), _identity("validation-1")),
    partition_digest: str = _identity("partition"),
):
    per_sequence = np.asarray(bler, dtype=np.float64)
    membership = tuple((identity, 1) for identity in identities)
    arm = evaluation_module.ArmEvaluation(
        name=name,
        predictor_type=predictor_type,
        regime=regime,
        role="validation",
        evaluation_sequence_ids=identities,
        sequence_membership=membership,
        per_sequence_nll=np.full(len(identities), 0.99),
        per_sequence_brier=np.full(len(identities), 0.98),
        per_sequence_bler=per_sequence,
        overall_nll=0.99,
        overall_brier=0.98,
        overall_bler=float(per_sequence.mean()),
        convergence_rate=1.0,
        mean_iterations=1.0,
        mean_correction_weight=0.0,
        latency_p50_seconds=0.1,
        latency_p95_seconds=0.1,
        latency_p99_seconds=0.1,
        estimator_batch_seconds=0.1,
        expected_calibration_error=0.0,
        reliability=(),
        logical_failures=np.zeros(len(identities), dtype=bool),
        syndrome_valid=np.ones(len(identities), dtype=bool),
        priors=np.zeros((len(identities), 1, 1), dtype=np.float64),
        corrections=np.zeros((len(identities), 1), dtype=np.uint8),
        stored_parameters=1,
        effective_parameters=1,
        partition_digest=partition_digest,
        partition_content_digest=_identity("partition-content"),
        evaluation_content_digest=_identity(f"evaluation-content:{regime}"),
        qec_hx_sha256=_identity("hx"),
        qec_hz_sha256=_identity("hz"),
        qec_logical_x_sha256=_identity("logical-x"),
        provenance_digest=_identity(name),
        per_round_outcomes=(),
        artifact_digest="",
        _token=evaluation_module._FREEZE_TOKEN,
    )
    object.__setattr__(arm, "artifact_digest", evaluation_module._arm_evaluation_integrity(arm))
    return arm


def _crossed_evaluations(
    regime: str,
    *,
    identities: tuple[str, ...] = (_identity("validation-0"), _identity("validation-1")),
    partition_digest: str = _identity("partition"),
    fno_hippo_bler: tuple[float, ...] = (0.3, 0.2),
):
    definitions = {
        "cnn_fir": ("causal_channel_forecaster:cnn:fir", (0.6, 0.5)),
        "fno_fir": ("causal_channel_forecaster:fno:fir", (0.5, 0.4)),
        "cnn_hippo": ("causal_channel_forecaster:cnn:hippo", (0.5, 0.4)),
        "fno_hippo": ("causal_channel_forecaster:fno:hippo", fno_hippo_bler),
    }
    return {
        name: _arm_evaluation(
            name,
            predictor,
            regime,
            bler=bler,
            identities=identities,
            partition_digest=partition_digest,
        )
        for name, (predictor, bler) in definitions.items()
    }


def test_baseline_selection_uses_raw_unclipped_validation_nll(monkeypatch) -> None:
    selection, partition, _ = _fit_selection(
        monkeypatch,
        ewma_probability=0.40,
        logistic_probability=0.45,
        validation_target=1.0,
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
    stationary = StationaryForecaster(0.0, np.zeros((58, 45)), 0.0)

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
        train=train,
        validation=validation_batch,
        calibration=calibration,
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
    observed = _role_batch("validation", _identity("validation")).syndromes.numpy()
    before = frozen.predict(observed, sequence_ids=selection.validation_sequence_ids)

    source_models[1].weight[:] = 100.0
    after = frozen.predict(observed, sequence_ids=selection.validation_sequence_ids)
    metadata, arrays = evaluation_module.export_frozen_predictor(frozen)
    restored = evaluation_module.restore_frozen_predictor(metadata, arrays)

    assert np.array_equal(before, after)
    assert np.array_equal(
        before,
        restored.predict(observed, sequence_ids=selection.validation_sequence_ids),
    )
    assert restored.artifact_digest == frozen.artifact_digest
    assert not frozen.weight.flags.writeable
    with pytest.raises(FrozenInstanceError):
        frozen.temperature = 4.0  # type: ignore[misc]


def test_baseline_selection_requires_exact_partition_membership(monkeypatch) -> None:
    train, _, calibration, partition = _partition_batches()
    with pytest.raises(ValueError, match="partition"):
        fit_select_observation_baselines(
            train=train,
            validation=_role_batch("validation", _identity("wrong-validation")),
            calibration=calibration,
            partition=partition,
        )


def test_baseline_selection_rejects_a_wrong_predictor_in_a_named_slot(monkeypatch) -> None:
    train, validation, calibration, partition = _partition_batches()
    monkeypatch.setattr(
        evaluation_module,
        "fit_stationary",
        lambda *args, **kwargs: StationaryForecaster(0.1, np.full((58, 45), 0.1), 0.0),
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
            train=train,
            validation=validation,
            calibration=calibration,
            partition=partition,
        )


def test_freeze_learned_arm_binds_provenance_and_snapshots_state() -> None:
    config = CausalExperimentConfig.from_json(CONFIG_PATH)
    model = build_forecaster(spatial="cnn", temporal="fir", config=config)
    train, validation, calibration, partition = _partition_batches()
    training_result = CausalTrainingResult(
        model_state_dict={
            name: value.detach().clone() for name, value in model.state_dict().items()
        },
        best_epoch=0,
        best_validation_nll=0.1,
        training_nll_history=(0.2,),
        validation_nll_history=(0.1,),
        partition_digest=partition.digest,
        regime=partition.regime,
        train_content_digest=partition.train_content_digest,
        validation_content_digest=partition.validation_content_digest,
        calibration_content_digest=partition.calibration_content_digest,
    )
    arm = freeze_learned_arm(
        name="cnn_fir",
        model=model,
        config=config,
        partition=partition,
        training_result=training_result,
        train=train,
        validation=validation,
        calibration=calibration,
    )
    metadata, arrays = evaluation_module.export_frozen_predictor(arm)
    restored = evaluation_module.restore_frozen_predictor(metadata, arrays)
    assert type(restored) is type(arm)
    assert restored.artifact_digest == arm.artifact_digest
    tampered_arrays = dict(arrays)
    first_key = next(iter(tampered_arrays))
    tampered_arrays[first_key] = np.array(tampered_arrays[first_key], copy=True)
    tampered_arrays[first_key].flat[0] += 1
    with pytest.raises(ValueError, match="integrity"):
        evaluation_module.restore_frozen_predictor(metadata, tampered_arrays)

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
    train, validation, calibration, partition = _partition_batches()
    result = CausalTrainingResult(
        model_state_dict={
            name: value.detach().clone() for name, value in model.state_dict().items()
        },
        best_epoch=0,
        best_validation_nll=0.1,
        training_nll_history=(),
        validation_nll_history=(0.1,),
        partition_digest=_identity("wrong-partition"),
        regime=partition.regime,
        train_content_digest=partition.train_content_digest,
        validation_content_digest=partition.validation_content_digest,
        calibration_content_digest=partition.calibration_content_digest,
    )
    with pytest.raises(ValueError, match="partition"):
        freeze_learned_arm(
            name="cnn_fir",
            model=model,
            config=config,
            partition=partition,
            training_result=result,
            train=train,
            validation=validation,
            calibration=calibration,
        )


def test_freeze_learned_arm_requires_canonical_architecture_name() -> None:
    config = CausalExperimentConfig.from_json(CONFIG_PATH)
    model = build_forecaster(spatial="cnn", temporal="fir", config=config)
    train, validation, calibration, partition = _partition_batches()
    result = CausalTrainingResult(
        model_state_dict={name: value.clone() for name, value in model.state_dict().items()},
        best_epoch=0,
        best_validation_nll=0.1,
        training_nll_history=(),
        validation_nll_history=(0.1,),
        partition_digest=partition.digest,
        regime=partition.regime,
        train_content_digest=partition.train_content_digest,
        validation_content_digest=partition.validation_content_digest,
        calibration_content_digest=partition.calibration_content_digest,
    )
    with pytest.raises(ValueError, match="canonical architecture name"):
        freeze_learned_arm(
            name="renamed_arm",
            model=model,
            config=config,
            partition=partition,
            training_result=result,
            train=train,
            validation=validation,
            calibration=calibration,
        )


def test_frozen_learned_arm_reconstructs_checkpoint_and_predicts_from_syndromes(
    monkeypatch,
    canonical_geometry,
) -> None:
    config = CausalExperimentConfig.from_json(CONFIG_PATH)
    model = build_forecaster(spatial="cnn", temporal="fir", config=config)
    train, validation, calibration, partition = _partition_batches()
    result = CausalTrainingResult(
        model_state_dict={
            name: value.detach().clone() for name, value in model.state_dict().items()
        },
        best_epoch=0,
        best_validation_nll=0.1,
        training_nll_history=(),
        validation_nll_history=(0.1,),
        partition_digest=partition.digest,
        regime=partition.regime,
        train_content_digest=partition.train_content_digest,
        validation_content_digest=partition.validation_content_digest,
        calibration_content_digest=partition.calibration_content_digest,
    )
    arm = freeze_learned_arm(
        name="cnn_fir",
        model=model,
        config=config,
        partition=partition,
        training_result=result,
        train=train,
        validation=validation,
        calibration=calibration,
    )
    hx, hz, logical_x = canonical_geometry
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=validation.sequence_ids,
        syndromes=np.zeros((1, 2, 21, 45), dtype=np.uint8),
        errors=np.zeros((1, 2, 58, 45), dtype=np.uint8),
        logical_flips=np.zeros((1, 2, logical_x.shape[0]), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )

    def fake_decode(*args, **kwargs):
        del args, kwargs
        return _decode_result(
            np.zeros((1, 2610), dtype=np.uint8),
            np.zeros((1, logical_x.shape[0]), dtype=np.uint8),
            np.ones(1, dtype=bool),
        )

    monkeypatch.setattr(evaluation_module, "decode_bplsd_prior_batch", fake_decode)
    evaluated = evaluate_causal_arms(batch, (arm,), hx=hx, hz=hz, logical_x=logical_x)
    assert evaluated.arms["cnn_fir"].stored_parameters == arm.stored_parameters
    assert evaluated.arms["cnn_fir"].partition_digest == partition.digest
    assert evaluated.arms["cnn_fir"].priors.shape == (1, 2, 2610)
    assert evaluated.arms["cnn_fir"].corrections.shape == (1, 2610)
    assert not evaluated.arms["cnn_fir"].priors.flags.writeable
    assert not evaluated.arms["cnn_fir"].corrections.flags.writeable


def test_evaluation_reconstructs_all_qec_labels_before_decode(
    monkeypatch, canonical_geometry
) -> None:
    hx, hz, logical_x = canonical_geometry
    corrupted_syndromes = np.zeros((1, 2, 21, 45), dtype=np.uint8)
    corrupted_syndromes[0, 1, 0] = 1
    errors = np.zeros((1, 2, 58, 45), dtype=np.uint8)
    selection, _, _ = _fit_selection(
        monkeypatch,
        ewma_probability=0.2,
        logistic_probability=0.3,
        validation_syndromes=corrupted_syndromes,
        validation_targets=errors,
    )
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=selection.validation_sequence_ids,
        syndromes=corrupted_syndromes,
        errors=errors,
        logical_flips=np.zeros((1, 2, logical_x.shape[0]), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="reconstructed syndromes"):
        evaluate_causal_arms(
            batch,
            (selection.baseline("stationary_field"),),
            hx=hx,
            hz=hz,
            logical_x=logical_x,
        )

    zero_syndromes = np.zeros_like(corrupted_syndromes)
    logical_selection, _, _ = _fit_selection(
        monkeypatch,
        ewma_probability=0.2,
        logistic_probability=0.3,
        validation_syndromes=zero_syndromes,
        validation_targets=errors,
    )
    logically_corrupted = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=logical_selection.validation_sequence_ids,
        syndromes=zero_syndromes,
        errors=errors,
        logical_flips=np.ones((1, 2, logical_x.shape[0]), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="reconstructed logical flips"):
        evaluate_causal_arms(
            logically_corrupted,
            (logical_selection.baseline("stationary_field"),),
            hx=hx,
            hz=hz,
            logical_x=logical_x,
        )


def test_frozen_arms_decode_identical_membership_and_score_modulo_stabilizers(
    monkeypatch,
    canonical_geometry,
) -> None:
    hx, hz, logical_x = canonical_geometry
    errors = np.zeros((1, 2, 58, 45), dtype=np.uint8)
    logical_column = int(np.flatnonzero(logical_x.toarray().any(axis=0))[0])
    errors.reshape(1, 2, -1)[0, 1, logical_column] = 1
    syndromes = np.asarray(hx @ errors.reshape(2, 2610).T).T.reshape(1, 2, 21, 45)
    logical = np.asarray(logical_x @ errors.reshape(2, 2610).T).T.reshape(1, 2, logical_x.shape[0])
    selection, _, _ = _fit_selection(
        monkeypatch,
        ewma_probability=0.2,
        logistic_probability=0.3,
        validation_syndromes=syndromes,
        validation_targets=errors,
    )
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
        correction = np.zeros((1, 2610), dtype=np.uint8)
        correction[0, logical_column] = 1
        return _decode_result(
            correction,
            np.zeros((1, logical_x.shape[0]), dtype=np.uint8),
            np.array([True]),
        )

    monkeypatch.setattr(evaluation_module, "decode_bplsd_prior_batch", fake_decode)
    evaluated = evaluate_causal_arms(
        batch,
        (
            selection.baseline("stationary_field"),
            selection.baseline(selection.selected_name),
        ),
        hx=hx,
        hz=hz,
        logical_x=logical_x,
    )

    assert len(calls) == 2 and np.array_equal(calls[0], calls[1])
    assert evaluated.sequence_membership == ((selection.validation_sequence_ids[0], 1),)
    selected_arm = evaluated.arms[selection.selected_name]
    assert selected_arm.logical_failures.tolist() == [True]
    assert selected_arm.all_syndrome_valid
    assert any(selected_arm.per_round_outcomes[0]["true_logical_flips"])
    assert selected_arm.expected_calibration_error >= 0.0
    assert len(selected_arm.reliability) == 10
    assert evaluated.decoder_config_digest == evaluation_module.CANONICAL_BPLSD_CONFIG_DIGEST
    assert selected_arm.qec_hx_sha256 == evaluated.qec_hx_sha256
    assert selected_arm.qec_hz_sha256 == evaluated.qec_hz_sha256
    assert selected_arm.qec_logical_x_sha256 == evaluated.qec_logical_x_sha256
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
            hz=sparse.csr_matrix((10, 15), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 15), dtype=np.uint8),
        )


def test_same_ids_with_different_evaluation_content_are_rejected(monkeypatch) -> None:
    selection, _, _ = _fit_selection(monkeypatch, ewma_probability=0.2, logistic_probability=0.3)
    syndromes = np.zeros((1, 2, 21, 45), dtype=np.uint8)
    syndromes[0, 0, 0, 0] = 1
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=selection.validation_sequence_ids,
        syndromes=syndromes,
        errors=np.zeros((1, 2, 58, 45), dtype=np.uint8),
        logical_flips=np.zeros((1, 2, 1), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="content"):
        evaluate_causal_arms(
            batch,
            (selection.baseline("stationary_field"),),
            hx=sparse.csr_matrix((945, 2610), dtype=np.uint8),
            hz=sparse.csr_matrix((945, 2610), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 2610), dtype=np.uint8),
        )


def test_same_shaped_substitute_qec_matrix_is_rejected(monkeypatch, canonical_geometry) -> None:
    _, hz, logical_x = canonical_geometry
    selection, _, _ = _fit_selection(monkeypatch, ewma_probability=0.2, logistic_probability=0.3)
    batch = CausalEvaluationBatch(
        regime="joint_in_basis",
        role="validation",
        sequence_ids=selection.validation_sequence_ids,
        syndromes=np.zeros((1, 2, 21, 45), dtype=np.uint8),
        errors=np.zeros((1, 2, 58, 45), dtype=np.uint8),
        logical_flips=np.zeros((1, 2, logical_x.shape[0]), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )

    with pytest.raises(ValueError, match="canonical lp_3_7_16"):
        evaluate_causal_arms(
            batch,
            (selection.baseline("stationary_field"),),
            hx=sparse.csr_matrix((945, 2610), dtype=np.uint8),
            hz=hz,
            logical_x=logical_x,
        )


def test_relabelled_evaluation_regime_is_rejected(monkeypatch) -> None:
    selection, _, _ = _fit_selection(monkeypatch, ewma_probability=0.2, logistic_probability=0.3)
    batch = CausalEvaluationBatch(
        regime="joint_basis_mismatch",
        role="validation",
        sequence_ids=selection.validation_sequence_ids,
        syndromes=np.zeros((1, 2, 21, 45), dtype=np.uint8),
        errors=np.zeros((1, 2, 58, 45), dtype=np.uint8),
        logical_flips=np.zeros((1, 2, 1), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="regime"):
        evaluate_causal_arms(
            batch,
            (selection.baseline("stationary_field"),),
            hx=sparse.csr_matrix((945, 2610), dtype=np.uint8),
            hz=sparse.csr_matrix((945, 2610), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 2610), dtype=np.uint8),
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
            hz=sparse.csr_matrix((10, 15), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 15), dtype=np.uint8),
        )


def test_factor_diagnostics_keep_predeclared_h3_sign_without_inference() -> None:
    diagnostics = reduced_factor_diagnostics(
        in_basis_arms=_crossed_evaluations("joint_in_basis"),
        basis_mismatch_arms=_crossed_evaluations("joint_basis_mismatch"),
    )
    assert diagnostics.in_basis_interaction.tolist() == pytest.approx([-0.1, -0.1])
    assert diagnostics.in_basis_mean < 0.0 and diagnostics.basis_mismatch_mean < 0.0
    assert diagnostics.in_basis_supports_predeclared_direction
    assert diagnostics.basis_mismatch_retains_predeclared_direction
    assert diagnostics.p_value is None


def test_factor_diagnostics_allows_independent_canonical_regime_partitions() -> None:
    diagnostics = reduced_factor_diagnostics(
        in_basis_arms=_crossed_evaluations(
            "joint_in_basis",
            identities=(_identity("in-0"), _identity("in-1")),
            partition_digest=_identity("in-partition"),
        ),
        basis_mismatch_arms=_crossed_evaluations(
            "joint_basis_mismatch",
            identities=(_identity("mismatch-0"), _identity("mismatch-1")),
            partition_digest=_identity("mismatch-partition"),
        ),
    )

    assert diagnostics.in_basis_supports_predeclared_direction
    assert diagnostics.basis_mismatch_retains_predeclared_direction


def test_basis_mismatch_cannot_retain_direction_absent_in_basis_h3_direction() -> None:
    diagnostics = reduced_factor_diagnostics(
        in_basis_arms=_crossed_evaluations("joint_in_basis", fno_hippo_bler=(0.6, 0.5)),
        basis_mismatch_arms=_crossed_evaluations("joint_basis_mismatch"),
    )

    assert diagnostics.in_basis_mean > 0.0
    assert diagnostics.basis_mismatch_mean < 0.0
    assert not diagnostics.basis_mismatch_retains_predeclared_direction


@pytest.mark.parametrize("attack", ["swapped_name", "wrong_regime", "membership", "content", "nll"])
def test_factor_diagnostics_rejects_unbound_or_substituted_inputs(attack: str) -> None:
    in_basis = _crossed_evaluations("joint_in_basis")
    mismatch = _crossed_evaluations("joint_basis_mismatch")
    if attack == "swapped_name":
        in_basis["cnn_fir"] = replace(in_basis["cnn_fir"], name="fno_fir")
    elif attack == "wrong_regime":
        mismatch["cnn_fir"] = replace(mismatch["cnn_fir"], regime="joint_in_basis")
    elif attack == "membership":
        mismatch["cnn_fir"] = replace(
            mismatch["cnn_fir"], evaluation_sequence_ids=(_identity("other"),)
        )
    elif attack == "content":
        in_basis["cnn_fir"] = replace(
            in_basis["cnn_fir"], evaluation_content_digest=_identity("other-content")
        )
    else:
        in_basis["cnn_fir"] = replace(
            in_basis["cnn_fir"], per_sequence_bler=in_basis["cnn_fir"].per_sequence_nll
        )
    with pytest.raises(ValueError):
        reduced_factor_diagnostics(
            in_basis_arms=in_basis,
            basis_mismatch_arms=mismatch,
        )


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
            hz=sparse.csr_matrix((1, 1), dtype=np.uint8),
            logical_x=sparse.csr_matrix((1, 1), dtype=np.uint8),
        )
