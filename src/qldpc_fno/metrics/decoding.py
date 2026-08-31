from __future__ import annotations

import math

import numpy as np


def _wilson_interval(errors: int, shots: int) -> tuple[float, float]:
    if shots <= 0:
        raise ValueError("shots must be positive")
    z = 1.959963984540054
    rate = errors / shots
    denominator = 1 + z**2 / shots
    center = (rate + z**2 / (2 * shots)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / shots + z**2 / (4 * shots**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def score_observable_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    """Score logical observable predictions using block-level failure semantics."""
    actual_array = np.asarray(actual, dtype=np.uint8)
    predicted_array = np.asarray(predicted, dtype=np.uint8)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted observables must have equal shape")
    if actual_array.ndim != 2 or actual_array.shape[0] == 0:
        raise ValueError("observable arrays must have shape (positive shots, observables)")
    if not np.all((actual_array == 0) | (actual_array == 1)):
        raise ValueError("actual observables must be binary")
    if not np.all((predicted_array == 0) | (predicted_array == 1)):
        raise ValueError("predicted observables must be binary")
    shot_errors = np.any(actual_array != predicted_array, axis=1)
    block_errors = int(shot_errors.sum())
    shots = int(actual_array.shape[0])
    low, high = _wilson_interval(block_errors, shots)
    return {
        "block_error_rate": block_errors / shots,
        "block_error_rate_95ci_high": high,
        "block_error_rate_95ci_low": low,
        "block_errors": block_errors,
        "exact_observable_match_rate": float(np.mean(actual_array == predicted_array)),
        "shots": shots,
    }
