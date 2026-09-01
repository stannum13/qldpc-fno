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
    *,
    syndrome_valid: np.ndarray | None = None,
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
    shots = int(actual_array.shape[0])
    if syndrome_valid is None:
        valid = np.ones(shots, dtype=np.bool_)
    else:
        valid = np.asarray(syndrome_valid)
        if valid.shape != (shots,) or valid.dtype.kind != "b":
            raise ValueError("syndrome_valid must contain one boolean value per shot")
    block_error = shot_errors | ~valid
    block_errors = int(block_error.sum())
    low, high = _wilson_interval(block_errors, shots)
    return {
        "block_error_rate": block_errors / shots,
        "block_error_rate_95ci_high": high,
        "block_error_rate_95ci_low": low,
        "block_errors": block_errors,
        "exact_observable_match_rate": float(np.mean(~shot_errors)),
        "shots": shots,
    }


def evaluate_correction_logits(
    logits: np.ndarray,
    *,
    hx: object,
    syndromes: np.ndarray,
    logical_x: object,
    actual_observables: np.ndarray,
) -> dict[str, object]:
    """Score thresholded corrections, treating invalid syndromes as block failures."""
    logit_array = np.asarray(logits)
    if logit_array.ndim < 2:
        raise ValueError("logits must have shape (shots, correction dimensions...)")
    shots = logit_array.shape[0]
    corrections = (logit_array >= 0).astype(np.uint8).reshape(shots, -1)
    syndrome_array = np.asarray(syndromes, dtype=np.uint8)
    observable_array = np.asarray(actual_observables, dtype=np.uint8)
    if syndrome_array.ndim != 2 or syndrome_array.shape[0] != shots:
        raise ValueError("syndromes must have shape (shots, checks)")
    if observable_array.ndim != 2 or observable_array.shape[0] != shots:
        raise ValueError("actual_observables must have shape (shots, observables)")
    if not np.all((syndrome_array == 0) | (syndrome_array == 1)):
        raise ValueError("syndromes must be binary")
    if not np.all((observable_array == 0) | (observable_array == 1)):
        raise ValueError("actual_observables must be binary")
    if hx.shape != (syndrome_array.shape[1], corrections.shape[1]):
        raise ValueError("Hx shape does not match syndrome and correction widths")
    if logical_x.shape != (observable_array.shape[1], corrections.shape[1]):
        raise ValueError("logical X shape does not match observable and correction widths")

    predicted_syndromes = np.asarray(hx @ corrections.T).T % 2
    valid = np.all(predicted_syndromes == syndrome_array, axis=1)
    predicted_observables = np.asarray(logical_x @ corrections.T).T % 2
    logical_mismatch = np.any(predicted_observables != observable_array, axis=1)
    block_error = ~valid | logical_mismatch
    block_errors = int(block_error.sum())
    low, high = _wilson_interval(block_errors, shots)
    return {
        "block_error_rate": block_errors / shots,
        "block_error_rate_95ci_high": high,
        "block_error_rate_95ci_low": low,
        "block_errors": block_errors,
        "exact_observable_match_rate": float(np.mean(~logical_mismatch)),
        "logical_mismatch_shots": int(logical_mismatch.sum()),
        "shots": shots,
        "syndrome_invalid": int((~valid).sum()),
        "syndrome_valid": int(valid.sum()),
        "syndrome_valid_rate": float(valid.mean()),
    }
