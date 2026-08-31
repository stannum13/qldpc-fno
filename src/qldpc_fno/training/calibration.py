"""Deterministic calibration of FNO correction probabilities."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_PROBABILITY_EPSILON = 1e-5


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    """Scalar parameters for conditioning FNO logits on physical noise."""

    alpha: float
    beta: float
    temperature: float


@dataclass(frozen=True, slots=True)
class CalibrationScore:
    """Calibration-split outcome for one immutable parameter tuple."""

    parameters: CalibrationParameters
    invalid_count: int
    block_errors: int
    nll: float


CALIBRATION_GRID = tuple(
    CalibrationParameters(alpha=alpha, beta=beta, temperature=temperature)
    for alpha in (0.25, 0.5, 1.0, 2.0)
    for beta in (0.0, 0.5, 1.0)
    for temperature in (0.5, 1.0, 2.0, 4.0)
)


def calibrated_probabilities(
    logits: np.ndarray,
    error_rates: np.ndarray,
    parameters: CalibrationParameters,
) -> np.ndarray:
    """Apply a noise-conditioned sigmoid transform and clip decoder priors."""
    logits_array = np.asarray(logits, dtype=np.float64)
    rates = np.asarray(error_rates, dtype=np.float64)
    if logits_array.ndim < 1:
        raise ValueError("logits must include a shot dimension")
    if rates.ndim != 1 or rates.shape[0] != logits_array.shape[0]:
        raise ValueError("error_rates must have one value per logit shot")
    if not np.all(np.isfinite(rates)) or not np.all((0.0 < rates) & (rates < 1.0)):
        raise ValueError("error_rates must be finite probabilities between zero and one")
    if not np.isfinite(parameters.alpha):
        raise ValueError("alpha must be finite")
    if not np.isfinite(parameters.beta):
        raise ValueError("beta must be finite")
    if not np.isfinite(parameters.temperature) or parameters.temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    logit_rates = np.log(rates) - np.log1p(-rates)
    rate_shape = (rates.shape[0],) + (1,) * (logits_array.ndim - 1)
    calibrated_logits = (
        parameters.alpha * logits_array / parameters.temperature
        + parameters.beta * logit_rates.reshape(rate_shape)
    )
    probabilities = _sigmoid(calibrated_logits)
    return np.clip(probabilities, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)


def select_calibration(scores: Sequence[CalibrationScore]) -> CalibrationScore:
    """Select the lexicographically best calibration-split candidate."""
    if not scores:
        raise ValueError("at least one calibration score is required")
    for score in scores:
        if score.invalid_count < 0:
            raise ValueError("invalid_count must be non-negative")
        if score.block_errors < 0:
            raise ValueError("block_errors must be non-negative")
        if not np.isfinite(score.nll):
            raise ValueError("nll must be finite")
    return min(
        scores,
        key=lambda score: (
            score.invalid_count,
            score.block_errors,
            score.nll,
            score.parameters.alpha,
            score.parameters.beta,
            score.parameters.temperature,
        ),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate the sigmoid without overflow for extreme FNO logits."""
    probabilities = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_values = values[~positive]
    exponentials = np.exp(negative_values)
    probabilities[~positive] = exponentials / (1.0 + exponentials)
    return probabilities
