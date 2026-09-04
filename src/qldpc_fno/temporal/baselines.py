"""Observation-only causal baselines for temporal channel forecasting.

The public fitters deliberately receive observed syndromes and error targets as
separate arrays.  A deployable forecaster never stores targets or simulator
diagnostics, and every stateful feature builder predicts before consuming the
current round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_DECAYS = (0.5, 0.8, 0.9, 0.97, 0.99)
DEFAULT_L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_SHRINKAGE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PROBABILITY_BOUNDS = (1e-5, 0.25)


def _as_rank4(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 4:
        raise ValueError(f"{name} must have shape (sequences, rounds, channels, ell)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _as_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(mask)
    if result.shape != shape or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean (sequences, rounds) array")
    if not result.any():
        raise ValueError(f"{name} must select at least one round")
    return result


def _validate_pair(
    syndromes: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = _as_rank4(syndromes, f"{prefix}_syndromes")
    labels = _as_rank4(targets, f"{prefix}_targets")
    if observed.shape[:2] != labels.shape[:2] or observed.shape[-1] != labels.shape[-1]:
        raise ValueError("syndrome and target sequence and round dimensions must agree")
    if np.any((observed < 0.0) | (observed > 1.0)):
        raise ValueError(f"{prefix}_syndromes must be binary observations")
    if np.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError(f"{prefix}_targets must be binary labels")
    return observed, labels, _as_mask(mask, observed.shape[:2], f"{prefix}_mask")


def _logit(probabilities: np.ndarray) -> np.ndarray:
    bounded = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return np.log(bounded) - np.log1p(-bounded)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return np.clip(
        1.0 / (1.0 + np.exp(-np.clip(_logit(probabilities) / temperature, -80.0, 80.0))),
        *PROBABILITY_BOUNDS,
    )


def _bernoulli_nll(probabilities: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    selected = mask[:, :, None, None]
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    losses = -(targets * np.log(probabilities) + (1.0 - targets) * np.log1p(-probabilities))
    return float(
        losses[selected.repeat(targets.shape[2], axis=2).repeat(targets.shape[3], axis=3)].mean()
    )


def build_ewma_histories(syndromes: np.ndarray, *, decay: float) -> np.ndarray:
    """Return the EWMA state available immediately before each round."""
    observed = _as_rank4(syndromes, "syndromes")
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    histories = np.zeros_like(observed)
    state = np.zeros((observed.shape[0], *observed.shape[2:]), dtype=np.float64)
    for round_index in range(observed.shape[1]):
        histories[:, round_index] = state
        state = decay * state + (1.0 - decay) * observed[:, round_index]
    return histories


def build_lagged_histories(syndromes: np.ndarray, *, lags: int = 32) -> np.ndarray:
    """Stack previous syndrome rounds newest-first, with zero prehistory."""
    observed = _as_rank4(syndromes, "syndromes")
    if type(lags) is not int or lags < 1:
        raise ValueError("lags must be a positive integer")
    sequences, rounds, channels, ell = observed.shape
    histories = np.zeros((sequences, rounds, lags * channels, ell), dtype=np.float64)
    for lag in range(1, lags + 1):
        if lag < rounds + 1:
            histories[:, lag:, (lag - 1) * channels : lag * channels] = observed[:, :-lag]
    return histories


@runtime_checkable
class DeployableForecaster(Protocol):
    """Marker protocol for models that consume no simulator-only state."""

    privileged: bool


@dataclass(frozen=True, slots=True)
class StationaryForecaster:
    scalar_probability: float
    empirical_field: np.ndarray
    shrinkage: float
    temperature: float = 1.0
    privileged: bool = False

    @property
    def field(self) -> np.ndarray:
        return (
            self.shrinkage * self.empirical_field + (1.0 - self.shrinkage) * self.scalar_probability
        )

    def predict(self, *, sequence_count: int, rounds: int) -> np.ndarray:
        base = np.broadcast_to(self.field, (sequence_count, rounds, *self.field.shape))
        return _temperature_scale(base, self.temperature)

    def raw_predict(self, *, sequence_count: int, rounds: int) -> np.ndarray:
        """Return uncalibrated probabilities without deployment clipping."""
        return np.broadcast_to(self.field, (sequence_count, rounds, *self.field.shape))


@dataclass(frozen=True, slots=True)
class CircularLogisticForecaster:
    weight: np.ndarray
    bias: np.ndarray
    l2: float
    temperature: float = 1.0
    feature_kind: str = "lagged"
    decay: float | None = None
    lags: int = 32
    privileged: bool = False

    def _features(self, syndromes: np.ndarray) -> np.ndarray:
        if self.feature_kind == "ewma":
            if self.decay is None:
                raise ValueError("EWMA model is missing its decay")
            return build_ewma_histories(syndromes, decay=self.decay)
        if self.feature_kind == "lagged":
            return build_lagged_histories(syndromes, lags=self.lags)
        raise ValueError(f"unknown causal feature kind: {self.feature_kind}")

    def predict(self, syndromes: np.ndarray) -> np.ndarray:
        return _temperature_scale(self.raw_predict(syndromes), self.temperature)

    def raw_predict(self, syndromes: np.ndarray) -> np.ndarray:
        """Return uncalibrated probabilities without deployment clipping."""
        logits = _circular_logits(self._features(syndromes), self.weight, self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


@dataclass(frozen=True, slots=True)
class PrivilegedOracle:
    """Simulator-only ceiling that must never enter a deployable registry."""

    probabilities: np.ndarray
    privileged: bool = True

    def predict(self) -> np.ndarray:
        return np.clip(_as_rank4(self.probabilities, "probabilities"), *PROBABILITY_BOUNDS)


def register_deployable(
    name: str,
    model: DeployableForecaster,
    registry: dict[str, DeployableForecaster] | None = None,
) -> dict[str, DeployableForecaster]:
    if not isinstance(name, str) or not name:
        raise ValueError("deployable model name must be non-empty")
    if getattr(model, "privileged", True):
        raise ValueError("privileged latent oracle cannot be registered as deployable")
    result = {} if registry is None else dict(registry)
    result[name] = model
    return result


def calibrate_temperature(
    probabilities: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    max_iter: int = 100,
) -> float:
    """Fit one positive scalar temperature on calibration data only."""
    raw = _as_rank4(probabilities, "probabilities")
    labels = _as_rank4(targets, "targets")
    if raw.shape != labels.shape:
        raise ValueError("calibration probabilities and targets must have identical shape")
    selected = _as_mask(mask, raw.shape[:2], "calibration_mask")
    logits = torch.as_tensor(_logit(raw), dtype=torch.float64)
    target_tensor = torch.as_tensor(labels, dtype=torch.float64)
    selected_tensor = torch.as_tensor(selected[:, :, None, None]).expand_as(target_tensor)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], max_iter=max_iter, tolerance_grad=1e-8, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(torch.clamp(log_temperature, -6.0, 6.0))
        loss = F.binary_cross_entropy_with_logits(
            (logits / temperature)[selected_tensor], target_tensor[selected_tensor]
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(torch.clamp(log_temperature.detach(), -6.0, 6.0)))


def fit_stationary(
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
    *,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    lambda_grid: tuple[float, ...] = DEFAULT_SHRINKAGE_GRID,
    calibrate: tuple[np.ndarray, np.ndarray] | None = None,
) -> StationaryForecaster:
    train = _as_rank4(train_targets, "train_targets")
    validation = _as_rank4(validation_targets, "validation_targets")
    if train.shape[2:] != validation.shape[2:]:
        raise ValueError("training and validation target geometry must agree")
    fitted = _as_mask(train_mask, train.shape[:2], "train_mask")
    mask = _as_mask(validation_mask, validation.shape[:2], "validation_mask")
    selected_train = train[fitted]
    scalar = float(selected_train.mean())
    empirical = selected_train.mean(axis=0)
    best: tuple[float, float] | None = None
    for shrinkage in sorted(lambda_grid):
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("stationary shrinkage values must be in [0, 1]")
        field = shrinkage * empirical + (1.0 - shrinkage) * scalar
        predictions = np.broadcast_to(field, validation.shape)
        candidate = (_bernoulli_nll(predictions, validation, mask), shrinkage)
        if best is None or candidate[0] < best[0] - 1e-12:
            best = candidate
    assert best is not None
    model = StationaryForecaster(scalar, empirical, best[1])
    if calibrate is not None:
        calibration_targets, calibration_mask = calibrate
        labels = _as_rank4(calibration_targets, "calibration_targets")
        probabilities = model.raw_predict(sequence_count=labels.shape[0], rounds=labels.shape[1])
        temperature = calibrate_temperature(probabilities, labels, calibration_mask)
        model = StationaryForecaster(scalar, empirical, best[1], temperature)
    return model


def _circular_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    values = _as_rank4(features, "features")
    sequences, rounds, channels, ell = values.shape
    kernel_size = weight.shape[-1]
    if weight.ndim != 3 or weight.shape[1] != channels or bias.shape != (weight.shape[0],):
        raise ValueError("logistic mapping parameter geometry is invalid")
    flat = torch.as_tensor(values.reshape(-1, channels, ell), dtype=torch.float64)
    weight_tensor = torch.as_tensor(weight, dtype=torch.float64)
    bias_tensor = torch.as_tensor(bias, dtype=torch.float64)
    left = kernel_size // 2
    right = kernel_size - 1 - left
    result = F.conv1d(F.pad(flat, (left, right), mode="circular"), weight_tensor, bias_tensor)
    return result.detach().numpy().reshape(sequences, rounds, weight.shape[0], ell)


def _fit_mapping(
    features: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    kernel_size: int,
    l2: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    if l2 < 0.0 or not np.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    values = _as_rank4(features, "features")
    labels = _as_rank4(targets, "targets")
    selected = _as_mask(mask, values.shape[:2], "mask")
    if values.shape[:2] != labels.shape[:2] or values.shape[-1] != labels.shape[-1]:
        raise ValueError("feature and target sequence and round dimensions must agree")
    _, _, input_channels, ell = values.shape
    output_channels = labels.shape[2]
    inputs = torch.as_tensor(values.reshape(-1, input_channels, ell), dtype=torch.float64)
    target_tensor = torch.as_tensor(labels.reshape(-1, output_channels, ell), dtype=torch.float64)
    selected_tensor = torch.as_tensor(selected.reshape(-1, 1, 1)).expand_as(target_tensor)
    weight = torch.zeros(
        (output_channels, input_channels, kernel_size), dtype=torch.float64, requires_grad=True
    )
    selected_labels = labels[selected]
    marginal = np.clip(selected_labels.mean(axis=(0, 2)), 1e-6, 1.0 - 1e-6)
    bias = torch.tensor(_logit(marginal), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=max_iter, tolerance_grad=1e-8, line_search_fn="strong_wolfe"
    )
    pad = kernel_size // 2

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = F.conv1d(F.pad(inputs, (pad, pad), mode="circular"), weight, bias)
        loss = (
            F.binary_cross_entropy_with_logits(
                logits[selected_tensor], target_tensor[selected_tensor]
            )
            + l2 * weight.square().mean()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach().numpy().copy(), bias.detach().numpy().copy()


def _select_mapping(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_mask: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    validation_mask: np.ndarray,
    *,
    kernel_size: int,
    l2_grid: tuple[float, ...],
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for l2 in sorted(l2_grid):
        weight, bias = _fit_mapping(
            train_features,
            train_targets,
            train_mask,
            kernel_size=kernel_size,
            l2=l2,
            max_iter=max_iter,
        )
        logits = _circular_logits(validation_features, weight, bias)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        score = _bernoulli_nll(probabilities, validation_targets, validation_mask)
        if best is None or score < best[0] - 1e-12:
            best = (score, l2, weight, bias)
    if best is None:
        raise ValueError("l2_grid must not be empty")
    return best[2], best[3], best[1], best[0]


def fit_logistic_ar(
    train_syndromes: np.ndarray,
    train_targets: np.ndarray,
    validation_syndromes: np.ndarray,
    validation_targets: np.ndarray,
    *,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    lags: int = 32,
    kernel_size: int = 3,
    l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
    max_iter: int = 500,
    calibrate: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> CircularLogisticForecaster:
    train_x, train_y, train_selected = _validate_pair(
        train_syndromes, train_targets, train_mask, prefix="train"
    )
    validation_x, validation_y, validation_selected = _validate_pair(
        validation_syndromes, validation_targets, validation_mask, prefix="validation"
    )
    train_features = build_lagged_histories(train_x, lags=lags)
    validation_features = build_lagged_histories(validation_x, lags=lags)
    weight, bias, l2, _ = _select_mapping(
        train_features,
        train_y,
        train_selected,
        validation_features,
        validation_y,
        validation_selected,
        kernel_size=kernel_size,
        l2_grid=l2_grid,
        max_iter=max_iter,
    )
    model = CircularLogisticForecaster(weight, bias, l2, feature_kind="lagged", lags=lags)
    if calibrate is not None:
        calibration_x, calibration_y, calibration_mask = _validate_pair(
            *calibrate, prefix="calibration"
        )
        temperature = calibrate_temperature(
            model.raw_predict(calibration_x), calibration_y, calibration_mask
        )
        model = CircularLogisticForecaster(
            weight, bias, l2, temperature, feature_kind="lagged", lags=lags
        )
    return model


def fit_ewma(
    train_syndromes: np.ndarray,
    train_targets: np.ndarray,
    validation_syndromes: np.ndarray,
    validation_targets: np.ndarray,
    *,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    decays: tuple[float, ...] = DEFAULT_DECAYS,
    kernel_size: int = 5,
    l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
    max_iter: int = 500,
    calibrate: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> CircularLogisticForecaster:
    train_x, train_y, train_selected = _validate_pair(
        train_syndromes, train_targets, train_mask, prefix="train"
    )
    validation_x, validation_y, validation_selected = _validate_pair(
        validation_syndromes, validation_targets, validation_mask, prefix="validation"
    )
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for decay in sorted(decays):
        train_features = build_ewma_histories(train_x, decay=decay)
        validation_features = build_ewma_histories(validation_x, decay=decay)
        weight, bias, l2, score = _select_mapping(
            train_features,
            train_y,
            train_selected,
            validation_features,
            validation_y,
            validation_selected,
            kernel_size=kernel_size,
            l2_grid=l2_grid,
            max_iter=max_iter,
        )
        if best is None or score < best[0] - 1e-12:
            best = (score, decay, l2, weight, bias)
    if best is None:
        raise ValueError("decays must not be empty")
    _, decay, l2, weight, bias = best
    model = CircularLogisticForecaster(weight, bias, l2, feature_kind="ewma", decay=decay)
    if calibrate is not None:
        calibration_x, calibration_y, calibration_mask = _validate_pair(
            *calibrate, prefix="calibration"
        )
        temperature = calibrate_temperature(
            model.raw_predict(calibration_x), calibration_y, calibration_mask
        )
        model = CircularLogisticForecaster(
            weight, bias, l2, temperature, feature_kind="ewma", decay=decay
        )
    return model


__all__ = [
    "DEFAULT_DECAYS",
    "DEFAULT_L2_GRID",
    "CircularLogisticForecaster",
    "PrivilegedOracle",
    "StationaryForecaster",
    "build_ewma_histories",
    "build_lagged_histories",
    "calibrate_temperature",
    "fit_ewma",
    "fit_logistic_ar",
    "fit_stationary",
    "register_deployable",
]
