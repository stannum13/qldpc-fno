"""Deterministic calibration of FNO correction probabilities."""

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

_PROBABILITY_EPSILON = 1e-5


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    """Scalar parameters for conditioning FNO logits on physical noise."""

    alpha: float
    beta: float
    temperature: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        if not np.isfinite(self.beta):
            raise ValueError("beta must be finite")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")


@dataclass(frozen=True, slots=True)
class CalibrationScore:
    """Calibration-split outcome for one immutable parameter tuple."""

    parameters: CalibrationParameters
    invalid_count: int
    block_errors: int
    nll: float

    def __post_init__(self) -> None:
        for name, count in (
            ("invalid_count", self.invalid_count),
            ("block_errors", self.block_errors),
        ):
            if type(count) is not int or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not np.isfinite(self.nll):
            raise ValueError("nll must be finite")


CALIBRATION_GRID = tuple(
    CalibrationParameters(alpha=alpha, beta=beta, temperature=temperature)
    for alpha in (0.25, 0.5, 1.0, 2.0)
    for beta in (0.0, 0.5, 1.0)
    for temperature in (0.5, 1.0, 2.0, 4.0)
)


def validate_calibration_progress_rows(
    rows: object,
    *,
    checkpoint_candidates: object,
    candidates: Sequence[CalibrationParameters],
    shots: int,
) -> list[dict[str, object]]:
    """Validate an exact ordered calibration-work prefix before resuming it."""
    if not isinstance(rows, list):
        raise TypeError("calibration progress candidates must be a list")
    if not isinstance(checkpoint_candidates, list) or any(
        not isinstance(checkpoint, dict) for checkpoint in checkpoint_candidates
    ):
        raise TypeError("calibration progress checkpoints are malformed")
    if type(shots) is not int or shots <= 0:
        raise ValueError("calibration progress shot count must be positive")
    if not candidates or len(rows) > len(checkpoint_candidates) * len(candidates):
        raise ValueError("calibration progress exceeds the declared work grid")

    checkpoint_measurements: dict[int, tuple[str, float]] = {}
    expected_row_fields = {
        "inference_latency_seconds",
        "logits_sha256",
        "model_checkpoint",
        "model_epoch",
        "parameters",
        "residual",
        "soft_prior",
    }
    for work_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise ValueError("calibration progress candidate row is malformed")
        checkpoint_index = work_index // len(candidates)
        checkpoint = checkpoint_candidates[checkpoint_index]
        parameters = candidates[work_index % len(candidates)]
        if (
            row.get("model_epoch") != int(checkpoint["epoch"])
            or row.get("model_checkpoint")
            != {
                "path": str(checkpoint["path"]),
                "sha256": str(checkpoint["sha256"]),
            }
            or row.get("parameters") != asdict(parameters)
        ):
            raise ValueError("calibration progress candidate order is inconsistent")

        digest = row.get("logits_sha256")
        latency = row.get("inference_latency_seconds")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError("calibration progress inference measurement is malformed")
        measurement = (digest, float(latency))
        previous = checkpoint_measurements.setdefault(checkpoint_index, measurement)
        if previous != measurement:
            raise ValueError("calibration progress checkpoint measurements diverge")

        score_nlls: list[float] = []
        for method in ("soft_prior", "residual"):
            score = row.get(method)
            if not isinstance(score, dict) or set(score) != {
                "block_errors",
                "invalid_count",
                "nll",
            }:
                raise ValueError(f"calibration progress {method} score is malformed")
            invalid_count = score["invalid_count"]
            block_errors = score["block_errors"]
            nll = score["nll"]
            if (
                type(invalid_count) is not int
                or type(block_errors) is not int
                or not 0 <= invalid_count <= block_errors <= shots
                or type(nll) not in (int, float)
                or not math.isfinite(nll)
                or nll < 0
            ):
                raise ValueError(f"calibration progress {method} score is invalid")
            score_nlls.append(float(nll))
        if score_nlls[0] != score_nlls[1]:
            raise ValueError("calibration progress method NLL values diverge")
    return rows


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
