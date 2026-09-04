"""Provenance-bound paired evaluation for causal decoder priors.

Callers cannot supply predictions, parameter counts, or decoder tuning. They
freeze a reviewed fitted estimator, and this module reconstructs predictions
from the evaluation syndromes before applying one canonical BP-LSD policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

import numpy as np
import torch
from scipy import sparse

from qldpc_fno.decoders.bplsd import BPLSDConfig, decode_bplsd_prior_batch
from qldpc_fno.models.causal_forecaster import (
    CausalChannelForecaster,
    build_forecaster,
    parameter_accounting,
)
from qldpc_fno.temporal.baselines import (
    DEFAULT_DECAYS,
    DEFAULT_L2_GRID,
    DEFAULT_SHRINKAGE_GRID,
    CircularLogisticForecaster,
    StationaryForecaster,
    calibrate_temperature,
    fit_ewma,
    fit_logistic_ar,
    fit_stationary,
)
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.training.causal_sequence import (
    CausalTrainingResult,
    RolePartition,
    SequenceRoleBatch,
    fit_calibration_temperature,
)

_FREEZE_TOKEN = object()
_BASELINE_MAX_ITER = 500
_BASELINE_POLICY = {
    "stationary_shrinkage_grid": list(DEFAULT_SHRINKAGE_GRID),
    "ewma_decays": list(DEFAULT_DECAYS),
    "l2_grid": list(DEFAULT_L2_GRID),
    "ewma_kernel_size": 5,
    "logistic_ar_lags": 32,
    "logistic_ar_kernel_size": 3,
    "lbfgs_max_iter": _BASELINE_MAX_ITER,
    "calibration": "scalar_temperature_lbfgs_max_iter_100",
    "deployment_probability_bounds": [1e-5, 0.25],
    "selection_metric": "raw_unclipped_validation_bernoulli_nll",
    "tie_rule": "ewma",
}
CANONICAL_BPLSD_CONFIG = BPLSDConfig()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


BASELINE_FIT_POLICY_DIGEST = _json_digest(_BASELINE_POLICY)
CANONICAL_BPLSD_CONFIG_DIGEST = _json_digest(asdict(CANONICAL_BPLSD_CONFIG))


def _validate_sequence_ids(sequence_ids: tuple[str, ...], sequence_count: int) -> None:
    if len(sequence_ids) != sequence_count or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("sequence_ids must uniquely identify every complete sequence")
    if any(re.fullmatch(r"[0-9a-f]{64}", identity) is None for identity in sequence_ids):
        raise ValueError("sequence_ids must be lowercase SHA-256 provenance identities")


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class ForecastSplit:
    """Separated observation and target arrays for one fitting role."""

    role: str
    sequence_ids: tuple[str, ...]
    syndromes: np.ndarray
    targets: np.ndarray
    scored_mask: np.ndarray

    def __post_init__(self) -> None:
        syndromes = np.asarray(self.syndromes)
        targets = np.asarray(self.targets)
        mask = np.asarray(self.scored_mask)
        if self.role not in {"train", "validation", "calibration"}:
            raise ValueError("forecast split role must be train, validation, or calibration")
        if syndromes.ndim != 4 or targets.ndim != 4:
            raise ValueError("forecast arrays must have shape (sequences, rounds, channels, ell)")
        if syndromes.shape[:2] != targets.shape[:2] or syndromes.shape[-1] != targets.shape[-1]:
            raise ValueError("forecast observations and targets must share sequence geometry")
        _validate_sequence_ids(self.sequence_ids, syndromes.shape[0])
        if mask.shape == (syndromes.shape[1],):
            mask = np.broadcast_to(mask, syndromes.shape[:2]).copy()
        elif mask.shape != syndromes.shape[:2]:
            raise ValueError("scored_mask must select rounds per sequence")
        if mask.dtype != np.bool_ or not np.all(mask.any(axis=1)):
            raise ValueError("every complete sequence must contain a Boolean scored round")
        if not np.all((syndromes == 0) | (syndromes == 1)):
            raise ValueError("syndromes must be binary")
        if not np.all((targets == 0) | (targets == 1)):
            raise ValueError("targets must be binary")
        object.__setattr__(self, "syndromes", _readonly_copy(syndromes))
        object.__setattr__(self, "targets", _readonly_copy(targets))
        object.__setattr__(self, "scored_mask", _readonly_copy(mask))


@dataclass(frozen=True, slots=True)
class CausalEvaluationBatch:
    """Complete paired sequences with evaluation-only physical/logical labels."""

    regime: str
    role: str
    sequence_ids: tuple[str, ...]
    syndromes: np.ndarray
    errors: np.ndarray
    logical_flips: np.ndarray
    scored_mask: np.ndarray

    def __post_init__(self) -> None:
        syndromes = np.asarray(self.syndromes)
        errors = np.asarray(self.errors)
        logical = np.asarray(self.logical_flips)
        mask = np.asarray(self.scored_mask)
        if self.role not in {"validation", "test"}:
            raise ValueError("causal evaluation role must be validation or test")
        if syndromes.ndim not in {3, 4} or errors.ndim not in {3, 4} or logical.ndim != 3:
            raise ValueError("evaluation arrays must contain sequence and round axes")
        if syndromes.shape[:2] != errors.shape[:2] or syndromes.shape[:2] != logical.shape[:2]:
            raise ValueError("all evaluation arrays must share sequence and round dimensions")
        _validate_sequence_ids(self.sequence_ids, syndromes.shape[0])
        if mask.shape == (syndromes.shape[1],):
            mask = np.broadcast_to(mask, syndromes.shape[:2]).copy()
        elif mask.shape != syndromes.shape[:2]:
            raise ValueError("scored_mask must select rounds per sequence")
        if mask.dtype != np.bool_ or not np.all(mask.any(axis=1)):
            raise ValueError("every complete sequence must contain a Boolean scored round")
        for name, values in (
            ("syndromes", syndromes),
            ("errors", errors),
            ("logical_flips", logical),
        ):
            if not np.all((values == 0) | (values == 1)):
                raise ValueError(f"{name} must be binary")
        object.__setattr__(self, "syndromes", _readonly_copy(syndromes))
        object.__setattr__(self, "errors", _readonly_copy(errors))
        object.__setattr__(self, "logical_flips", _readonly_copy(logical))
        object.__setattr__(self, "scored_mask", _readonly_copy(mask))


@dataclass(frozen=True, slots=True)
class _FrozenArray:
    dtype: str
    shape: tuple[int, ...]
    payload: bytes

    @classmethod
    def from_array(cls, values: np.ndarray) -> _FrozenArray:
        array = np.ascontiguousarray(values)
        return cls(array.dtype.str, tuple(array.shape), array.tobytes())

    def array(self) -> np.ndarray:
        result = np.frombuffer(self.payload, dtype=np.dtype(self.dtype)).reshape(self.shape).copy()
        result.flags.writeable = False
        return result


@dataclass(frozen=True, slots=True)
class _FrozenTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes

    @classmethod
    def from_tensor(cls, name: str, value: torch.Tensor) -> _FrozenTensor:
        tensor = value.detach().cpu().contiguous()
        payload = tensor.view(torch.uint8).numpy().tobytes()
        return cls(name, str(tensor.dtype), tuple(tensor.shape), payload)

    def tensor(self) -> torch.Tensor:
        dtype_name = self.dtype.removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"unsupported frozen tensor dtype: {self.dtype}")
        byte_values = torch.frombuffer(bytearray(self.payload), dtype=torch.uint8)
        return byte_values.view(dtype).reshape(self.shape).clone()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        frozen = _FrozenTensor.from_tensor(name, state_dict[name])
        metadata = json.dumps(
            {"name": frozen.name, "dtype": frozen.dtype, "shape": frozen.shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(frozen.payload).to_bytes(8, "big"))
        digest.update(frozen.payload)
    return digest.hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    return _state_dict_sha256({"value": value})


def _calibration_digest(
    calibration: SequenceRoleBatch,
    *,
    temperature: float,
    partition_digest: str,
    checkpoint_sha256: str,
) -> str:
    return _json_digest(
        {
            "partition_digest": partition_digest,
            "checkpoint_sha256": checkpoint_sha256,
            "sequence_ids": list(calibration.sequence_ids),
            "syndromes_sha256": _tensor_digest(calibration.syndromes),
            "targets_sha256": _tensor_digest(calibration.targets),
            "scored_mask_sha256": _tensor_digest(calibration.scored_mask),
            "temperature": temperature,
            "method": "scalar_temperature_lbfgs_max_iter_100",
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenBaseline:
    """Immutable fitted non-neural estimator created only by baseline selection."""

    name: str
    predictor_type: str
    partition_digest: str
    partition_content_digest: str
    fit_policy_digest: str
    model_sha256: str
    calibration_digest: str
    train_sequence_ids: tuple[str, ...]
    validation_sequence_ids: tuple[str, ...]
    calibration_sequence_ids: tuple[str, ...]
    evaluation_sequence_ids: tuple[str, ...]
    evaluation_role: str
    evaluation_regime: str
    evaluation_content_digest: str
    scalar_probability: float
    shrinkage: float
    temperature: float
    l2: float
    feature_kind: str
    decay: float | None
    lags: int
    stored_parameters: int
    effective_parameters: int
    artifact_digest: str
    _empirical_field: _FrozenArray
    _weight: _FrozenArray
    _bias: _FrozenArray
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _FREEZE_TOKEN:
            raise ValueError("FrozenBaseline must be created by baseline selection")

    @property
    def empirical_field(self) -> np.ndarray:
        return self._empirical_field.array()

    @property
    def weight(self) -> np.ndarray:
        return self._weight.array()

    @property
    def bias(self) -> np.ndarray:
        return self._bias.array()

    def predict(self, syndromes: np.ndarray, *, sequence_ids: tuple[str, ...]) -> np.ndarray:
        if sequence_ids != self.evaluation_sequence_ids:
            raise ValueError("evaluation membership does not match frozen baseline")
        observed = np.asarray(syndromes)
        if self.predictor_type == "stationary_field":
            model = StationaryForecaster(
                self.scalar_probability,
                self.empirical_field,
                self.shrinkage,
                self.temperature,
            )
            return model.predict(sequence_count=observed.shape[0], rounds=observed.shape[1])
        model = CircularLogisticForecaster(
            self.weight.copy(),
            self.bias.copy(),
            self.l2,
            self.temperature,
            self.feature_kind,
            self.decay,
            self.lags,
        )
        return model.predict(observed)


@dataclass(frozen=True, slots=True)
class FrozenArm:
    """Immutable learned checkpoint with fit, calibration, and evaluation identity."""

    name: str
    predictor_type: str
    spatial_kind: str
    temporal_kind: str
    config: CausalExperimentConfig
    config_digest: str
    partition_digest: str
    partition_content_digest: str
    checkpoint_sha256: str
    calibration_digest: str
    calibration_temperature: float
    calibration_sequence_ids: tuple[str, ...]
    evaluation_sequence_ids: tuple[str, ...]
    evaluation_role: str
    evaluation_regime: str
    evaluation_content_digest: str
    stored_parameters: int
    effective_parameters: int
    artifact_digest: str
    _state: tuple[_FrozenTensor, ...]
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _FREEZE_TOKEN:
            raise ValueError("FrozenArm must be created by freeze_learned_arm")

    def predict(self, syndromes: np.ndarray, *, sequence_ids: tuple[str, ...]) -> np.ndarray:
        if sequence_ids != self.evaluation_sequence_ids:
            raise ValueError("evaluation membership does not match frozen learned arm")
        model = build_forecaster(
            spatial=self.spatial_kind,
            temporal=self.temporal_kind,
            config=self.config,
        )
        model.load_state_dict({item.name: item.tensor() for item in self._state}, strict=True)
        model.eval()
        parameter = next(model.parameters())
        observed = torch.as_tensor(
            np.array(syndromes, copy=True), dtype=parameter.dtype, device="cpu"
        )
        state = model.initial_state(observed.shape[0], device="cpu", dtype=parameter.dtype)
        logits: list[torch.Tensor] = []
        with torch.no_grad():
            for round_index in range(observed.shape[1]):
                hidden = model.temporal.predict(state)
                logits.append(model.readout(hidden))
                state = model._update(state, observed[:, round_index])
            values = torch.sigmoid(torch.stack(logits, dim=1) / self.calibration_temperature)
            values = values.clamp(
                min=self.config.generator.min_probability,
                max=self.config.generator.max_probability,
            )
        return values.cpu().numpy()


@dataclass(frozen=True, slots=True)
class BaselineSelection:
    _baselines: tuple[FrozenBaseline, ...]
    _validation_nll: tuple[tuple[str, float], ...]
    selected_name: str
    partition_digest: str
    partition_content_digest: str
    fit_policy_digest: str
    train_sequence_ids: tuple[str, ...]
    validation_sequence_ids: tuple[str, ...]
    calibration_sequence_ids: tuple[str, ...]
    evaluation_role: str
    evaluation_regime: str
    evaluation_content_digest: str
    artifact_digest: str
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _FREEZE_TOKEN:
            raise ValueError("BaselineSelection must be created by the canonical fitter")

    @property
    def validation_nll(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._validation_nll))

    def baseline(self, name: str) -> FrozenBaseline:
        _validate_selection_integrity(self)
        for baseline in self._baselines:
            if baseline.name == name:
                return baseline
        raise KeyError(name)

    @property
    def frozen(self) -> bool:
        return True


def _as_flat(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array.reshape(*array.shape[:2], -1)


def _forecast_nll_by_sequence(
    probabilities: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    p = _as_flat(probabilities).astype(np.float64)
    y = _as_flat(targets).astype(np.float64)
    if p.shape != y.shape:
        raise ValueError("forecast probabilities and physical error targets must have equal shapes")
    if not np.all(np.isfinite(p)) or not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("forecast probabilities must be finite and lie in [0, 1]")
    # This numerical guard is not the deployment clip at 0.25: it preserves
    # validation ordering above 0.25 while making exact empirical zeros finite.
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    losses = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return np.asarray([losses[index, mask[index]].mean() for index in range(p.shape[0])])


def _forecast_brier_by_sequence(
    probabilities: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    p = _as_flat(probabilities).astype(np.float64)
    y = _as_flat(targets).astype(np.float64)
    if p.shape != y.shape:
        raise ValueError("forecast probabilities and physical error targets must have equal shapes")
    losses = (p - y) ** 2
    return np.asarray([losses[index, mask[index]].mean() for index in range(p.shape[0])])


def _reliability_table(
    probabilities: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[float, tuple[Mapping[str, object], ...]]:
    flat_targets = _as_flat(targets)
    selected = np.repeat(mask[:, :, None], flat_targets.shape[-1], axis=2)
    p = _as_flat(probabilities)[selected].astype(np.float64)
    y = flat_targets[selected].astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rows: list[Mapping[str, object]] = []
    weighted_gap = 0.0
    for index in range(bins):
        chosen = indices == index
        count = int(chosen.sum())
        mean_probability = float(p[chosen].mean()) if count else 0.0
        empirical_frequency = float(y[chosen].mean()) if count else 0.0
        weighted_gap += count * abs(mean_probability - empirical_frequency)
        rows.append(
            MappingProxyType(
                {
                    "bin": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": count,
                    "mean_probability": mean_probability,
                    "empirical_frequency": empirical_frequency,
                }
            )
        )
    return float(weighted_gap / p.size), tuple(rows)


def _baseline_raw_predict(model: object, split: ForecastSplit) -> np.ndarray:
    if type(model) is StationaryForecaster:
        return model.raw_predict(
            sequence_count=split.syndromes.shape[0], rounds=split.syndromes.shape[1]
        )
    if type(model) is CircularLogisticForecaster:
        return model.raw_predict(split.syndromes)
    raise TypeError("baseline fitter returned an unregistered deployable type")


def _baseline_parameter_count(model: object) -> int:
    if type(model) is StationaryForecaster:
        return int(model.empirical_field.size + 3)
    if type(model) is CircularLogisticForecaster:
        return int(model.weight.size + model.bias.size + 3)
    raise TypeError("baseline fitter returned an unregistered deployable type")


def _frozen_baseline_integrity(arm: FrozenBaseline) -> str:
    if arm.predictor_type == "stationary_field":
        materialized: object = StationaryForecaster(
            arm.scalar_probability,
            arm.empirical_field,
            arm.shrinkage,
            arm.temperature,
        )
    else:
        materialized = CircularLogisticForecaster(
            arm.weight.copy(),
            arm.bias.copy(),
            arm.l2,
            arm.temperature,
            arm.feature_kind,
            arm.decay,
            arm.lags,
        )
    return _json_digest(
        {
            "name": arm.name,
            "predictor_type": arm.predictor_type,
            "partition_digest": arm.partition_digest,
            "partition_content_digest": arm.partition_content_digest,
            "fit_policy_digest": arm.fit_policy_digest,
            "model_sha256": arm.model_sha256,
            "materialized_model_sha256": _baseline_model_sha256(materialized),
            "calibration_digest": arm.calibration_digest,
            "train_sequence_ids": list(arm.train_sequence_ids),
            "validation_sequence_ids": list(arm.validation_sequence_ids),
            "calibration_sequence_ids": list(arm.calibration_sequence_ids),
            "evaluation_sequence_ids": list(arm.evaluation_sequence_ids),
            "evaluation_role": arm.evaluation_role,
            "evaluation_regime": arm.evaluation_regime,
            "evaluation_content_digest": arm.evaluation_content_digest,
            "stored_parameters": arm.stored_parameters,
            "effective_parameters": arm.effective_parameters,
        }
    )


def _frozen_arm_integrity(arm: FrozenArm) -> str:
    materialized_state = {item.name: item.tensor() for item in arm._state}
    return _json_digest(
        {
            "name": arm.name,
            "predictor_type": arm.predictor_type,
            "spatial_kind": arm.spatial_kind,
            "temporal_kind": arm.temporal_kind,
            "config_digest": arm.config_digest,
            "materialized_config_digest": _json_digest(arm.config.to_dict()),
            "partition_digest": arm.partition_digest,
            "partition_content_digest": arm.partition_content_digest,
            "checkpoint_sha256": arm.checkpoint_sha256,
            "materialized_checkpoint_sha256": _state_dict_sha256(materialized_state),
            "calibration_digest": arm.calibration_digest,
            "calibration_temperature": arm.calibration_temperature,
            "calibration_sequence_ids": list(arm.calibration_sequence_ids),
            "evaluation_sequence_ids": list(arm.evaluation_sequence_ids),
            "evaluation_role": arm.evaluation_role,
            "evaluation_regime": arm.evaluation_regime,
            "evaluation_content_digest": arm.evaluation_content_digest,
            "stored_parameters": arm.stored_parameters,
            "effective_parameters": arm.effective_parameters,
        }
    )


def _selection_integrity(selection: BaselineSelection) -> str:
    return _json_digest(
        {
            "baselines": [baseline.artifact_digest for baseline in selection._baselines],
            "validation_nll": list(selection._validation_nll),
            "selected_name": selection.selected_name,
            "partition_digest": selection.partition_digest,
            "partition_content_digest": selection.partition_content_digest,
            "fit_policy_digest": selection.fit_policy_digest,
            "train_sequence_ids": list(selection.train_sequence_ids),
            "validation_sequence_ids": list(selection.validation_sequence_ids),
            "calibration_sequence_ids": list(selection.calibration_sequence_ids),
            "evaluation_role": selection.evaluation_role,
            "evaluation_regime": selection.evaluation_regime,
            "evaluation_content_digest": selection.evaluation_content_digest,
        }
    )


def _validate_selection_integrity(selection: BaselineSelection) -> None:
    if selection.artifact_digest != _selection_integrity(selection):
        raise ValueError("frozen baseline selection integrity check failed")
    for baseline in selection._baselines:
        if baseline.artifact_digest != _frozen_baseline_integrity(baseline):
            raise ValueError("frozen baseline integrity check failed")


def _validate_arm_integrity(arm: FrozenArm | FrozenBaseline) -> None:
    expected = (
        _frozen_arm_integrity(arm) if type(arm) is FrozenArm else _frozen_baseline_integrity(arm)
    )
    if arm.artifact_digest != expected:
        raise ValueError("frozen arm integrity check failed")


def _validate_fitted_baseline(name: str, model: object, split: ForecastSplit) -> None:
    target_geometry = split.targets.shape[2:]
    syndrome_channels = split.syndromes.shape[2]
    if name == "stationary_field":
        if type(model) is not StationaryForecaster:
            raise TypeError("stationary slot did not return the exact stationary deployable type")
        if model.empirical_field.shape != target_geometry:
            raise ValueError("stationary field geometry does not match physical targets")
        return
    if type(model) is not CircularLogisticForecaster:
        raise TypeError(f"{name} slot did not return the exact circular logistic deployable type")
    if name == "ewma":
        expected = (target_geometry[0], syndrome_channels, 5)
        if model.feature_kind != "ewma" or model.decay not in DEFAULT_DECAYS:
            raise TypeError("EWMA slot did not return the registered EWMA predictor")
    elif name == "logistic_ar":
        expected = (target_geometry[0], syndrome_channels * 32, 3)
        if model.feature_kind != "lagged" or model.lags != 32:
            raise TypeError("logistic AR slot did not return the registered lagged predictor")
    else:
        raise ValueError(f"unknown baseline slot: {name}")
    if model.weight.shape != expected or model.bias.shape != (target_geometry[0],):
        raise ValueError(f"{name} fitted parameter geometry is invalid")
    if model.l2 not in DEFAULT_L2_GRID:
        raise ValueError(f"{name} selected an undeclared L2 value")


def _numpy_digest(values: np.ndarray) -> str:
    frozen = _FrozenArray.from_array(values)
    digest = hashlib.sha256()
    digest.update(frozen.dtype.encode())
    digest.update(json.dumps(frozen.shape).encode())
    digest.update(frozen.payload)
    return digest.hexdigest()


def _numpy_from_tensor(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _expanded_mask(mask: np.ndarray, sequence_count: int) -> np.ndarray:
    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim == 1:
        values = np.broadcast_to(values, (sequence_count, values.shape[0]))
    return np.ascontiguousarray(values)


def _view_content_digest(
    *,
    role: str,
    regime: str,
    sequence_ids: tuple[str, ...],
    syndromes: np.ndarray,
    targets: np.ndarray,
    scored_mask: np.ndarray,
) -> str:
    return _json_digest(
        {
            "role": role,
            "regime": regime,
            "sequence_ids": list(sequence_ids),
            "syndromes": _numpy_digest(np.asarray(syndromes, dtype=np.uint8)),
            "targets": _numpy_digest(np.asarray(targets, dtype=np.uint8)),
            "scored_mask": _numpy_digest(
                _expanded_mask(np.asarray(scored_mask), len(sequence_ids))
            ),
        }
    )


def _sequence_batch_view_digest(batch: SequenceRoleBatch, *, regime: str) -> str:
    return _view_content_digest(
        role=batch.role,
        regime=regime,
        sequence_ids=batch.sequence_ids,
        syndromes=_numpy_from_tensor(batch.syndromes),
        targets=_numpy_from_tensor(batch.targets),
        scored_mask=_numpy_from_tensor(batch.scored_mask),
    )


def _evaluation_batch_view_digest(batch: CausalEvaluationBatch) -> str:
    return _view_content_digest(
        role=batch.role,
        regime=batch.regime,
        sequence_ids=batch.sequence_ids,
        syndromes=batch.syndromes,
        targets=batch.errors,
        scored_mask=batch.scored_mask,
    )


def _partition_content_digest(
    train: SequenceRoleBatch,
    validation: SequenceRoleBatch,
    calibration: SequenceRoleBatch,
    *,
    regime: str,
) -> str:
    return _json_digest(
        {
            "regime": regime,
            "train": _sequence_batch_view_digest(train, regime=regime),
            "validation": _sequence_batch_view_digest(validation, regime=regime),
            "calibration": _sequence_batch_view_digest(calibration, regime=regime),
        }
    )


def _forecast_split_from_batch(batch: SequenceRoleBatch) -> ForecastSplit:
    return ForecastSplit(
        role=batch.role,
        sequence_ids=batch.sequence_ids,
        syndromes=_numpy_from_tensor(batch.syndromes),
        targets=_numpy_from_tensor(batch.targets),
        scored_mask=_numpy_from_tensor(batch.scored_mask),
    )


def _baseline_model_sha256(model: object) -> str:
    if type(model) is StationaryForecaster:
        payload = {
            "type": "stationary_field",
            "scalar": model.scalar_probability,
            "field": _numpy_digest(model.empirical_field),
            "shrinkage": model.shrinkage,
            "temperature": model.temperature,
        }
    elif type(model) is CircularLogisticForecaster:
        payload = {
            "type": model.feature_kind,
            "weight": _numpy_digest(model.weight),
            "bias": _numpy_digest(model.bias),
            "l2": model.l2,
            "temperature": model.temperature,
            "decay": model.decay,
            "lags": model.lags,
        }
    else:
        raise TypeError("baseline fitter returned an unregistered deployable type")
    return _json_digest(payload)


def _forecast_split_digest(split: ForecastSplit) -> str:
    return _json_digest(
        {
            "role": split.role,
            "sequence_ids": list(split.sequence_ids),
            "syndromes": _numpy_digest(split.syndromes),
            "targets": _numpy_digest(split.targets),
            "scored_mask": _numpy_digest(split.scored_mask),
        }
    )


def _freeze_baseline(
    *,
    name: str,
    model: object,
    partition: RolePartition,
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    calibration: ForecastSplit,
    partition_content_digest: str,
    evaluation_regime: str,
    evaluation_content_digest: str,
) -> FrozenBaseline:
    if type(model) not in {StationaryForecaster, CircularLogisticForecaster} or getattr(
        model, "privileged", True
    ):
        raise TypeError("baseline is not an exact registered deployable type")
    if type(model) is StationaryForecaster:
        predictor_type = "stationary_field"
        empirical = model.empirical_field
        weight = np.empty(0, dtype=np.float64)
        bias = np.empty(0, dtype=np.float64)
        scalar, shrinkage = model.scalar_probability, model.shrinkage
        l2, feature_kind, decay, lags = 0.0, "stationary", None, 0
    else:
        predictor_type = model.feature_kind
        empirical = np.empty(0, dtype=np.float64)
        weight, bias = model.weight, model.bias
        scalar, shrinkage = math.nan, math.nan
        l2, feature_kind, decay, lags = model.l2, model.feature_kind, model.decay, model.lags
    count = _baseline_parameter_count(model)
    model_digest = _baseline_model_sha256(model)
    frozen = FrozenBaseline(
        name=name,
        predictor_type=predictor_type,
        partition_digest=partition.digest,
        partition_content_digest=partition_content_digest,
        fit_policy_digest=BASELINE_FIT_POLICY_DIGEST,
        model_sha256=model_digest,
        calibration_digest=_json_digest(
            {
                "model_sha256": model_digest,
                "partition_digest": partition.digest,
                "calibration_split_digest": _forecast_split_digest(calibration),
                "temperature": model.temperature,
            }
        ),
        train_sequence_ids=train_ids,
        validation_sequence_ids=validation_ids,
        calibration_sequence_ids=calibration.sequence_ids,
        evaluation_sequence_ids=validation_ids,
        evaluation_role="validation",
        evaluation_regime=evaluation_regime,
        evaluation_content_digest=evaluation_content_digest,
        scalar_probability=float(scalar),
        shrinkage=float(shrinkage),
        temperature=float(model.temperature),
        l2=float(l2),
        feature_kind=feature_kind,
        decay=None if decay is None else float(decay),
        lags=int(lags),
        stored_parameters=count,
        effective_parameters=count,
        artifact_digest="",
        _empirical_field=_FrozenArray.from_array(empirical),
        _weight=_FrozenArray.from_array(weight),
        _bias=_FrozenArray.from_array(bias),
        _token=_FREEZE_TOKEN,
    )
    object.__setattr__(frozen, "artifact_digest", _frozen_baseline_integrity(frozen))
    return frozen


def fit_select_observation_baselines(
    *,
    train: SequenceRoleBatch,
    validation: SequenceRoleBatch,
    calibration: SequenceRoleBatch,
    partition: RolePartition,
    regime: str,
) -> BaselineSelection:
    """Fit baselines from content-bound batches under one immutable policy."""

    expected = (
        (train, "train", partition.train_sequence_ids),
        (validation, "validation", partition.validation_sequence_ids),
        (calibration, "calibration", partition.calibration_sequence_ids),
    )
    for batch, role, identities in expected:
        if batch.role != role or frozenset(batch.sequence_ids) != identities:
            raise ValueError(f"{role} split membership does not match the supplied partition")
    if not regime:
        raise ValueError("baseline fitting requires a nonempty evaluation regime")
    partition_content = _partition_content_digest(train, validation, calibration, regime=regime)
    evaluation_content = _sequence_batch_view_digest(validation, regime=regime)
    train = _forecast_split_from_batch(train)
    validation = _forecast_split_from_batch(validation)
    calibration = _forecast_split_from_batch(calibration)
    stationary = fit_stationary(
        train.targets,
        validation.targets,
        train_mask=train.scored_mask,
        validation_mask=validation.scored_mask,
        lambda_grid=DEFAULT_SHRINKAGE_GRID,
        calibrate=None,
    )
    ewma = fit_ewma(
        train.syndromes,
        train.targets,
        validation.syndromes,
        validation.targets,
        train_mask=train.scored_mask,
        validation_mask=validation.scored_mask,
        decays=DEFAULT_DECAYS,
        kernel_size=5,
        l2_grid=DEFAULT_L2_GRID,
        max_iter=_BASELINE_MAX_ITER,
        calibrate=None,
    )
    logistic = fit_logistic_ar(
        train.syndromes,
        train.targets,
        validation.syndromes,
        validation.targets,
        train_mask=train.scored_mask,
        validation_mask=validation.scored_mask,
        lags=32,
        kernel_size=3,
        l2_grid=DEFAULT_L2_GRID,
        max_iter=_BASELINE_MAX_ITER,
        calibrate=None,
    )
    uncalibrated = {
        "stationary_field": stationary,
        "ewma": ewma,
        "logistic_ar": logistic,
    }
    for name, model in uncalibrated.items():
        _validate_fitted_baseline(name, model, validation)
    scores = {
        name: float(
            _forecast_nll_by_sequence(
                _baseline_raw_predict(model, validation),
                validation.targets,
                validation.scored_mask,
            ).mean()
        )
        for name, model in uncalibrated.items()
    }
    selected = "ewma" if scores["ewma"] <= scores["logistic_ar"] else "logistic_ar"
    calibrated: dict[str, object] = {}
    for name, model in uncalibrated.items():
        temperature = calibrate_temperature(
            _baseline_raw_predict(model, calibration),
            calibration.targets,
            calibration.scored_mask,
        )
        calibrated[name] = replace(model, temperature=temperature)
    baselines = tuple(
        _freeze_baseline(
            name=name,
            model=calibrated[name],
            partition=partition,
            train_ids=train.sequence_ids,
            validation_ids=validation.sequence_ids,
            calibration=calibration,
            partition_content_digest=partition_content,
            evaluation_regime=regime,
            evaluation_content_digest=evaluation_content,
        )
        for name in ("stationary_field", "ewma", "logistic_ar")
    )
    selection = BaselineSelection(
        baselines,
        tuple((name, scores[name]) for name in ("stationary_field", "ewma", "logistic_ar")),
        selected,
        partition.digest,
        partition_content,
        BASELINE_FIT_POLICY_DIGEST,
        train.sequence_ids,
        validation.sequence_ids,
        calibration.sequence_ids,
        "validation",
        regime,
        evaluation_content,
        "",
        _FREEZE_TOKEN,
    )
    object.__setattr__(selection, "artifact_digest", _selection_integrity(selection))
    return selection


def freeze_learned_arm(
    *,
    name: str,
    model: CausalChannelForecaster,
    config: CausalExperimentConfig,
    partition: RolePartition,
    training_result: CausalTrainingResult,
    train: SequenceRoleBatch,
    validation: SequenceRoleBatch,
    calibration: SequenceRoleBatch,
    regime: str,
) -> FrozenArm:
    """Snapshot a trained exact forecaster with its role and calibration provenance."""

    if type(model) is not CausalChannelForecaster or getattr(model, "privileged", False):
        raise TypeError("learned arm is not an exact registered deployable type")
    canonical_name = f"{model.spatial_kind}_{model.temporal_kind}"
    if name != canonical_name:
        raise ValueError(f"learned arm must use canonical architecture name {canonical_name!r}")
    if training_result.partition_digest != partition.digest:
        raise ValueError("training result does not match the supplied partition")
    expected = (
        (train, "train", partition.train_sequence_ids),
        (validation, "validation", partition.validation_sequence_ids),
        (calibration, "calibration", partition.calibration_sequence_ids),
    )
    for batch, role, identities in expected:
        if batch.role != role or frozenset(batch.sequence_ids) != identities:
            raise ValueError(f"{role} membership does not match the supplied partition")
    if not regime:
        raise ValueError("learned arm freezing requires a nonempty evaluation regime")
    partition_content = _partition_content_digest(train, validation, calibration, regime=regime)
    evaluation_content = _sequence_batch_view_digest(validation, regime=regime)
    model_hash = _state_dict_sha256(model.state_dict())
    result_hash = _state_dict_sha256(training_result.model_state_dict)
    if model_hash != result_hash:
        raise ValueError("model state does not match the frozen training checkpoint")
    expected_model = build_forecaster(
        spatial=model.spatial_kind,
        temporal=model.temporal_kind,
        config=config,
    )
    expected_model.load_state_dict(model.state_dict(), strict=True)
    if (
        model.ell != config.code.ell
        or model.output_channels != config.code.n // config.code.ell
        or model.minimum_probability != config.generator.min_probability
        or model.maximum_probability != config.generator.max_probability
    ):
        raise ValueError("learned model does not match the supplied experiment configuration")
    calibration_parameter = next(model.parameters())
    calibration_syndromes = calibration.syndromes.to(
        device=calibration_parameter.device,
        dtype=calibration_parameter.dtype,
    )
    state_for_calibration = model.initial_state(
        calibration_syndromes.shape[0],
        device=calibration_parameter.device,
        dtype=calibration_parameter.dtype,
    )
    calibration_logits: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for round_index in range(calibration_syndromes.shape[1]):
            hidden = model.temporal.predict(state_for_calibration)
            calibration_logits.append(model.readout(hidden))
            state_for_calibration = model._update(
                state_for_calibration, calibration_syndromes[:, round_index]
            )
    temperature = fit_calibration_temperature(
        torch.stack(calibration_logits, dim=1),
        calibration,
        partition=partition,
        training_result=training_result,
    )
    accounting = parameter_accounting(model)
    state = tuple(
        _FrozenTensor.from_tensor(state_name, tensor)
        for state_name, tensor in sorted(model.state_dict().items())
    )
    frozen = FrozenArm(
        name=name,
        predictor_type=f"causal_channel_forecaster:{model.spatial_kind}:{model.temporal_kind}",
        spatial_kind=model.spatial_kind,
        temporal_kind=model.temporal_kind,
        config=config,
        config_digest=_json_digest(config.to_dict()),
        partition_digest=partition.digest,
        partition_content_digest=partition_content,
        checkpoint_sha256=model_hash,
        calibration_digest=_calibration_digest(
            calibration,
            temperature=temperature,
            partition_digest=partition.digest,
            checkpoint_sha256=model_hash,
        ),
        calibration_temperature=temperature,
        calibration_sequence_ids=tuple(calibration.sequence_ids),
        evaluation_sequence_ids=tuple(validation.sequence_ids),
        evaluation_role="validation",
        evaluation_regime=regime,
        evaluation_content_digest=evaluation_content,
        stored_parameters=accounting.stored_real_scalars,
        effective_parameters=accounting.effective_functional_scalars,
        artifact_digest="",
        _state=state,
        _token=_FREEZE_TOKEN,
    )
    object.__setattr__(frozen, "artifact_digest", _frozen_arm_integrity(frozen))
    return frozen


@dataclass(frozen=True, slots=True)
class ArmEvaluation:
    name: str
    predictor_type: str
    regime: str
    role: str
    evaluation_sequence_ids: tuple[str, ...]
    sequence_membership: tuple[tuple[str, int], ...]
    per_sequence_nll: np.ndarray
    per_sequence_brier: np.ndarray
    per_sequence_bler: np.ndarray
    overall_nll: float
    overall_brier: float
    overall_bler: float
    convergence_rate: float
    mean_iterations: float
    mean_correction_weight: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    expected_calibration_error: float
    reliability: tuple[Mapping[str, object], ...]
    logical_failures: np.ndarray
    syndrome_valid: np.ndarray
    stored_parameters: int
    effective_parameters: int
    partition_digest: str
    partition_content_digest: str
    evaluation_content_digest: str
    provenance_digest: str
    per_round_outcomes: tuple[Mapping[str, object], ...]
    artifact_digest: str
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _FREEZE_TOKEN:
            raise ValueError("ArmEvaluation must be created by evaluate_causal_arms")

    @property
    def all_syndrome_valid(self) -> bool:
        return bool(np.all(self.syndrome_valid))


def _arm_evaluation_integrity(arm: ArmEvaluation) -> str:
    return _json_digest(
        {
            "name": arm.name,
            "predictor_type": arm.predictor_type,
            "regime": arm.regime,
            "role": arm.role,
            "evaluation_sequence_ids": list(arm.evaluation_sequence_ids),
            "sequence_membership": [list(item) for item in arm.sequence_membership],
            "per_sequence_nll": _numpy_digest(arm.per_sequence_nll),
            "per_sequence_brier": _numpy_digest(arm.per_sequence_brier),
            "per_sequence_bler": _numpy_digest(arm.per_sequence_bler),
            "overall_nll": arm.overall_nll,
            "overall_brier": arm.overall_brier,
            "overall_bler": arm.overall_bler,
            "convergence_rate": arm.convergence_rate,
            "mean_iterations": arm.mean_iterations,
            "mean_correction_weight": arm.mean_correction_weight,
            "latency_p50_seconds": arm.latency_p50_seconds,
            "latency_p95_seconds": arm.latency_p95_seconds,
            "latency_p99_seconds": arm.latency_p99_seconds,
            "expected_calibration_error": arm.expected_calibration_error,
            "reliability": [dict(row) for row in arm.reliability],
            "logical_failures": _numpy_digest(arm.logical_failures),
            "syndrome_valid": _numpy_digest(arm.syndrome_valid),
            "stored_parameters": arm.stored_parameters,
            "effective_parameters": arm.effective_parameters,
            "partition_digest": arm.partition_digest,
            "partition_content_digest": arm.partition_content_digest,
            "evaluation_content_digest": arm.evaluation_content_digest,
            "provenance_digest": arm.provenance_digest,
            "per_round_outcomes": [dict(row) for row in arm.per_round_outcomes],
        }
    )


@dataclass(frozen=True, slots=True)
class PairedCausalEvaluation:
    regime: str
    role: str
    evaluation_sequence_ids: tuple[str, ...]
    sequence_membership: tuple[tuple[str, int], ...]
    arms: Mapping[str, ArmEvaluation]
    partition_digest: str
    partition_content_digest: str
    decoder_config_digest: str
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _FREEZE_TOKEN:
            raise ValueError("PairedCausalEvaluation must be created by evaluate_causal_arms")


def _arm_provenance_digest(arm: FrozenArm | FrozenBaseline) -> str:
    if type(arm) is FrozenArm:
        return _json_digest(
            {
                "partition": arm.partition_digest,
                "partition_content": arm.partition_content_digest,
                "checkpoint": arm.checkpoint_sha256,
                "calibration": arm.calibration_digest,
                "config": arm.config_digest,
                "predictor": arm.predictor_type,
                "evaluation_role": arm.evaluation_role,
                "evaluation_regime": arm.evaluation_regime,
                "evaluation_content": arm.evaluation_content_digest,
            }
        )
    return _json_digest(
        {
            "partition": arm.partition_digest,
            "partition_content": arm.partition_content_digest,
            "fit_policy": arm.fit_policy_digest,
            "model": arm.model_sha256,
            "calibration": arm.calibration_digest,
            "predictor": arm.predictor_type,
            "name": arm.name,
            "evaluation_role": arm.evaluation_role,
            "evaluation_regime": arm.evaluation_regime,
            "evaluation_content": arm.evaluation_content_digest,
        }
    )


def _validate_qec_reconstruction(
    batch: CausalEvaluationBatch,
    *,
    hx: sparse.spmatrix,
    logical_x: sparse.spmatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hx_csr = hx.astype(np.uint8).tocsr()
    logical_csr = logical_x.astype(np.uint8).tocsr()
    syndromes = _as_flat(batch.syndromes).astype(np.uint8)
    errors = _as_flat(batch.errors).astype(np.uint8)
    logical = np.asarray(batch.logical_flips, dtype=np.uint8)
    if syndromes.shape[-1] != hx_csr.shape[0] or errors.shape[-1] != hx_csr.shape[1]:
        raise ValueError("evaluation arrays do not match parity-check geometry")
    if logical.shape[-1] != logical_csr.shape[0] or logical_csr.shape[1] != hx_csr.shape[1]:
        raise ValueError("logical labels or operators do not match code geometry")
    expected_syndromes = np.asarray(hx_csr @ errors.reshape(-1, errors.shape[-1]).T).T % 2
    expected_syndromes = expected_syndromes.reshape(syndromes.shape)
    if not np.array_equal(expected_syndromes, syndromes):
        raise ValueError("reconstructed syndromes disagree with supplied evaluation data")
    expected_logical = np.asarray(logical_csr @ errors.reshape(-1, errors.shape[-1]).T).T % 2
    expected_logical = expected_logical.reshape(logical.shape)
    if not np.array_equal(expected_logical, logical):
        raise ValueError("reconstructed logical flips disagree with supplied evaluation data")
    return syndromes, errors, logical


def evaluate_causal_arms(
    batch: CausalEvaluationBatch,
    arms: tuple[FrozenArm | FrozenBaseline, ...],
    *,
    hx: sparse.spmatrix,
    logical_x: sparse.spmatrix,
) -> PairedCausalEvaluation:
    """Generate predictions internally and decode identical scored sequence-rounds."""

    if not arms:
        raise ValueError("at least one frozen deployable arm is required")
    if any(type(arm) not in {FrozenArm, FrozenBaseline} for arm in arms):
        raise TypeError("every evaluation arm must be an exact registered frozen deployable type")
    for arm in arms:
        _validate_arm_integrity(arm)
    names = tuple(arm.name for arm in arms)
    if len(set(names)) != len(names):
        raise ValueError("frozen evaluation arm names must be unique")
    if any(arm.evaluation_sequence_ids != batch.sequence_ids for arm in arms):
        raise ValueError("evaluation membership does not match every frozen arm")
    if any(arm.evaluation_role != batch.role for arm in arms):
        raise ValueError("evaluation role does not match every frozen arm")
    if any(arm.evaluation_regime != batch.regime for arm in arms):
        raise ValueError("evaluation regime does not match every frozen arm")
    batch_content = _evaluation_batch_view_digest(batch)
    if any(arm.evaluation_content_digest != batch_content for arm in arms):
        raise ValueError("evaluation content does not match every frozen arm")
    partitions = {arm.partition_digest for arm in arms}
    if len(partitions) != 1:
        raise ValueError("all frozen arms must derive from the same role partition")
    partition_contents = {arm.partition_content_digest for arm in arms}
    if len(partition_contents) != 1:
        raise ValueError("all frozen arms must derive from the same content-bound partition")
    syndromes, errors, logical = _validate_qec_reconstruction(batch, hx=hx, logical_x=logical_x)
    selected_syndromes = syndromes[batch.scored_mask]
    selected_logical = logical[batch.scored_mask]
    membership = tuple(
        (identity, int(round_index))
        for sequence_index, identity in enumerate(batch.sequence_ids)
        for round_index in np.flatnonzero(batch.scored_mask[sequence_index])
    )
    evaluated: dict[str, ArmEvaluation] = {}
    for arm in arms:
        probabilities = arm.predict(batch.syndromes, sequence_ids=batch.sequence_ids)
        priors = _as_flat(probabilities).astype(np.float64)
        if priors.shape != errors.shape or not np.all(
            np.isfinite(priors) & (priors > 0.0) & (priors < 0.5)
        ):
            raise ValueError(f"frozen arm {arm.name!r} emitted invalid prior geometry or values")
        decoded = decode_bplsd_prior_batch(
            hx,
            selected_syndromes,
            logical_x,
            error_channels=priors[batch.scored_mask],
            config=CANONICAL_BPLSD_CONFIG,
        )
        recomputed_valid = np.all(
            (np.asarray(hx.tocsr() @ decoded.corrections.T).T % 2) == selected_syndromes,
            axis=1,
        )
        if not np.array_equal(recomputed_valid, decoded.syndrome_valid):
            raise RuntimeError(
                "decoder syndrome_valid disagrees with independently recomputed validity"
            )
        logical_mismatch = np.any(decoded.predicted_observables != selected_logical, axis=1)
        failures = (~recomputed_valid) | logical_mismatch
        per_nll = _forecast_nll_by_sequence(priors, errors, batch.scored_mask)
        per_brier = _forecast_brier_by_sequence(priors, errors, batch.scored_mask)
        calibration_error, reliability = _reliability_table(priors, errors, batch.scored_mask)
        per_bler = np.asarray(
            [
                failures[
                    [index for index, member in enumerate(membership) if member[0] == identity]
                ].mean()
                for identity in batch.sequence_ids
            ]
        )
        selected_priors = priors[batch.scored_mask]
        selected_errors = errors[batch.scored_mask].astype(np.float64)
        round_nll = -(
            selected_errors * np.log(selected_priors)
            + (1.0 - selected_errors) * np.log1p(-selected_priors)
        ).mean(axis=1)
        outcomes = tuple(
            MappingProxyType(
                {
                    "sequence_id": sequence_id,
                    "round": round_index,
                    "logical_failure": bool(failures[index]),
                    "syndrome_valid": bool(recomputed_valid[index]),
                    "converged": bool(decoded.converged[index]),
                    "iterations": int(decoded.iterations[index]),
                    "correction_weight": int(decoded.corrections[index].sum()),
                    "predicted_observables": tuple(map(int, decoded.predicted_observables[index])),
                    "true_logical_flips": tuple(map(int, selected_logical[index])),
                    "observed_error_weight": int(selected_errors[index].sum()),
                    "forecast_nll": float(round_nll[index]),
                    "forecast_brier": float(
                        ((selected_priors[index] - selected_errors[index]) ** 2).mean()
                    ),
                    "setup_latency_seconds": float(decoded.setup_latency_seconds[index]),
                    "decode_latency_seconds": float(decoded.decode_latency_seconds[index]),
                }
            )
            for index, (sequence_id, round_index) in enumerate(membership)
        )
        arm_evaluation = ArmEvaluation(
            name=arm.name,
            predictor_type=arm.predictor_type,
            regime=batch.regime,
            role=batch.role,
            evaluation_sequence_ids=batch.sequence_ids,
            sequence_membership=membership,
            per_sequence_nll=_readonly_copy(per_nll),
            per_sequence_brier=_readonly_copy(per_brier),
            per_sequence_bler=_readonly_copy(per_bler),
            overall_nll=float(per_nll.mean()),
            overall_brier=float(per_brier.mean()),
            overall_bler=float(failures.mean()),
            convergence_rate=float(decoded.converged.mean()),
            mean_iterations=float(decoded.iterations.mean()),
            mean_correction_weight=float(decoded.corrections.sum(axis=1).mean()),
            latency_p50_seconds=float(np.quantile(decoded.latency_seconds, 0.50)),
            latency_p95_seconds=float(np.quantile(decoded.latency_seconds, 0.95)),
            latency_p99_seconds=float(np.quantile(decoded.latency_seconds, 0.99)),
            expected_calibration_error=calibration_error,
            reliability=reliability,
            logical_failures=_readonly_copy(failures),
            syndrome_valid=_readonly_copy(recomputed_valid),
            stored_parameters=arm.stored_parameters,
            effective_parameters=arm.effective_parameters,
            partition_digest=arm.partition_digest,
            partition_content_digest=arm.partition_content_digest,
            evaluation_content_digest=arm.evaluation_content_digest,
            provenance_digest=_arm_provenance_digest(arm),
            per_round_outcomes=outcomes,
            artifact_digest="",
            _token=_FREEZE_TOKEN,
        )
        object.__setattr__(
            arm_evaluation,
            "artifact_digest",
            _arm_evaluation_integrity(arm_evaluation),
        )
        evaluated[arm.name] = arm_evaluation
    return PairedCausalEvaluation(
        batch.regime,
        batch.role,
        batch.sequence_ids,
        membership,
        MappingProxyType(dict(evaluated)),
        next(iter(partitions)),
        next(iter(partition_contents)),
        CANONICAL_BPLSD_CONFIG_DIGEST,
        _FREEZE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ReducedProgression:
    regime: str
    selected_name: str
    nll_improvement: float
    bler_difference: float
    progressed: bool
    scope: str = "descriptive_reduced_non_scientific"
    p_value: None = None
    hypothesis_status: None = None


def reduced_progression(
    *,
    selection: BaselineSelection,
    stationary: ArmEvaluation,
    selected: ArmEvaluation,
) -> ReducedProgression:
    """Apply the exact membership-bound descriptive Gate-2 B-E rule."""

    if stationary.artifact_digest != _arm_evaluation_integrity(
        stationary
    ) or selected.artifact_digest != _arm_evaluation_integrity(selected):
        raise ValueError("arm evaluation integrity check failed")

    allowed = {
        "static_spatial_latent",
        "temporal_uniform",
        "joint_in_basis",
        "joint_basis_mismatch",
    }
    if stationary.regime not in allowed or selected.regime != stationary.regime:
        raise ValueError("reduced progression is defined only for aligned Regimes B-E")
    if stationary.role != "validation" or selected.role != "validation":
        raise ValueError("reduced progression requires validation-role evaluations")
    if stationary.regime != selection.evaluation_regime:
        raise ValueError("progression regime does not match the frozen baseline selection")
    if stationary.name != "stationary_field" or selected.name != selection.selected_name:
        raise ValueError("evaluated arms do not match the frozen baseline selection")
    expected = selection.validation_sequence_ids
    expected_stationary = _arm_provenance_digest(selection.baseline("stationary_field"))
    expected_selected = _arm_provenance_digest(selection.baseline(selection.selected_name))
    if (
        stationary.evaluation_sequence_ids != expected
        or selected.evaluation_sequence_ids != expected
        or stationary.sequence_membership != selected.sequence_membership
        or stationary.partition_digest != selection.partition_digest
        or selected.partition_digest != selection.partition_digest
        or stationary.partition_content_digest != selection.partition_content_digest
        or selected.partition_content_digest != selection.partition_content_digest
        or stationary.evaluation_content_digest != selection.evaluation_content_digest
        or selected.evaluation_content_digest != selection.evaluation_content_digest
        or stationary.provenance_digest != expected_stationary
        or selected.provenance_digest != expected_selected
    ):
        raise ValueError("progression arm membership or ordering is not aligned")
    nll_gain = stationary.overall_nll - selected.overall_nll
    bler_difference = selected.overall_bler - stationary.overall_bler
    return ReducedProgression(
        stationary.regime,
        selected.name,
        nll_gain,
        bler_difference,
        nll_gain > 0.0 and bler_difference <= 0.0,
    )


@dataclass(frozen=True, slots=True)
class ReducedFactorDiagnostics:
    in_basis_interaction: np.ndarray
    basis_mismatch_interaction: np.ndarray
    in_basis_mean: float
    basis_mismatch_mean: float
    in_basis_supports_predeclared_direction: bool
    basis_mismatch_retains_predeclared_direction: bool
    scope: str = "descriptive_reduced_non_scientific"
    p_value: None = None
    hypothesis_status: None = None


_CROSSED_PREDICTORS = {
    "cnn_fir": "causal_channel_forecaster:cnn:fir",
    "fno_fir": "causal_channel_forecaster:fno:fir",
    "cnn_hippo": "causal_channel_forecaster:cnn:hippo",
    "fno_hippo": "causal_channel_forecaster:fno:hippo",
}


def _validated_crossed_bler(
    arms: Mapping[str, ArmEvaluation], *, expected_regime: str
) -> dict[str, np.ndarray]:
    if set(arms) != set(_CROSSED_PREDICTORS):
        raise ValueError(f"crossed arms must contain exactly {sorted(_CROSSED_PREDICTORS)}")
    reference: ArmEvaluation | None = None
    losses: dict[str, np.ndarray] = {}
    for name, expected_predictor in _CROSSED_PREDICTORS.items():
        arm = arms[name]
        if type(arm) is not ArmEvaluation:
            raise TypeError("crossed diagnostics require exact ArmEvaluation artifacts")
        if arm.artifact_digest != _arm_evaluation_integrity(arm):
            raise ValueError("crossed arm evaluation integrity check failed")
        if arm.name != name or arm.predictor_type != expected_predictor:
            raise ValueError("crossed arm name does not match its registered predictor")
        if arm.role != "validation" or arm.regime != expected_regime:
            raise ValueError("crossed diagnostics require the exact validation regime")
        if arm.per_sequence_bler.shape != (len(arm.evaluation_sequence_ids),):
            raise ValueError("crossed BLER must contain one value per complete sequence")
        if reference is None:
            reference = arm
        elif (
            arm.partition_digest != reference.partition_digest
            or arm.partition_content_digest != reference.partition_content_digest
            or arm.evaluation_content_digest != reference.evaluation_content_digest
            or arm.evaluation_sequence_ids != reference.evaluation_sequence_ids
            or arm.sequence_membership != reference.sequence_membership
        ):
            raise ValueError("crossed arm partition, membership, or ordering is not aligned")
        losses[name] = np.asarray(arm.per_sequence_bler, dtype=np.float64)
    return losses


def _interaction(losses: Mapping[str, np.ndarray]) -> np.ndarray:
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in losses.items()}
    reference = arrays["cnn_fir"]
    if (
        reference.ndim != 1
        or reference.size == 0
        or any(value.shape != reference.shape for value in arrays.values())
    ):
        raise ValueError("crossed losses must be aligned, nonempty per-sequence vectors")
    return (arrays["cnn_fir"] - arrays["fno_fir"]) - (arrays["cnn_hippo"] - arrays["fno_hippo"])


def reduced_factor_diagnostics(
    *,
    in_basis_arms: Mapping[str, ArmEvaluation],
    basis_mismatch_arms: Mapping[str, ArmEvaluation],
) -> ReducedFactorDiagnostics:
    """Derive the descriptive predeclared H3 BLER contrast from frozen evaluations."""

    in_losses = _validated_crossed_bler(in_basis_arms, expected_regime="joint_in_basis")
    mismatch_losses = _validated_crossed_bler(
        basis_mismatch_arms, expected_regime="joint_basis_mismatch"
    )
    in_reference = in_basis_arms["cnn_fir"]
    mismatch_reference = basis_mismatch_arms["cnn_fir"]
    if (
        in_reference.partition_digest != mismatch_reference.partition_digest
        or in_reference.evaluation_sequence_ids != mismatch_reference.evaluation_sequence_ids
        or in_reference.sequence_membership != mismatch_reference.sequence_membership
    ):
        raise ValueError("in-basis and mismatch evaluations are not membership-aligned")
    in_basis = _interaction(in_losses)
    mismatch = _interaction(mismatch_losses)
    in_mean, mismatch_mean = float(in_basis.mean()), float(mismatch.mean())
    return ReducedFactorDiagnostics(
        _readonly_copy(in_basis),
        _readonly_copy(mismatch),
        in_mean,
        mismatch_mean,
        bool(in_mean < 0.0),
        bool(mismatch_mean < 0.0),
    )


def evaluate_oracle_sanity(
    probabilities: np.ndarray, targets: np.ndarray, scored_mask: np.ndarray
) -> dict[str, object]:
    """Score latent probabilities only as an unattainable generator sanity ceiling."""

    target, mask = np.asarray(targets), np.asarray(scored_mask)
    if mask.shape == (target.shape[1],):
        mask = np.broadcast_to(mask, target.shape[:2]).copy()
    nll = _forecast_nll_by_sequence(probabilities, target, mask)
    brier = _forecast_brier_by_sequence(probabilities, target, mask)
    return {
        "scope": "generator_sanity_only",
        "deployable_competition": False,
        "decoder_evaluated": False,
        "per_sequence_nll": tuple(map(float, nll)),
        "per_sequence_brier": tuple(map(float, brier)),
        "overall_nll": float(nll.mean()),
        "overall_brier": float(brier.mean()),
    }


__all__ = [
    "BASELINE_FIT_POLICY_DIGEST",
    "CANONICAL_BPLSD_CONFIG",
    "CANONICAL_BPLSD_CONFIG_DIGEST",
    "ArmEvaluation",
    "BaselineSelection",
    "CausalEvaluationBatch",
    "FrozenArm",
    "FrozenBaseline",
    "PairedCausalEvaluation",
    "ReducedFactorDiagnostics",
    "ReducedProgression",
    "evaluate_causal_arms",
    "evaluate_oracle_sanity",
    "fit_select_observation_baselines",
    "freeze_learned_arm",
    "reduced_factor_diagnostics",
    "reduced_progression",
]
