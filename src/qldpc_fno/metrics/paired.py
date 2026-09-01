"""Shot-level paired statistics for decoder block-failure outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def _failure_outcomes(values: np.ndarray, *, name: str) -> np.ndarray:
    outcomes = np.asarray(values)
    if outcomes.ndim != 1:
        raise ValueError(f"{name} outcomes must be one-dimensional")
    if outcomes.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean failure outcomes")
    return outcomes.astype(np.bool_, copy=False)


def _wilson_summary(outcomes: np.ndarray) -> dict[str, object]:
    shots = int(outcomes.size)
    failures = int(np.count_nonzero(outcomes))
    rate = failures / shots
    z = 1.959963984540054
    denominator = 1.0 + z**2 / shots
    center = (rate + z**2 / (2 * shots)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / shots + z**2 / (4 * shots**2))
    radius /= denominator
    return {
        "block_error_rate": rate,
        "block_error_rate_95ci_high": min(1.0, center + radius),
        "block_error_rate_95ci_low": max(0.0, center - radius),
        "block_errors": failures,
        "shots": shots,
    }


def paired_decoder_summary(
    baseline_failures: np.ndarray,
    hybrid_failures: np.ndarray,
    *,
    bootstrap_seed: int,
    samples: int = 10_000,
) -> dict[str, object]:
    """Summarize paired block failures and bootstrap the hybrid-minus-baseline delta.

    Multinomial sampling of the three possible paired deltas (-1, 0, +1) is the
    exact distribution obtained by drawing shot indices with replacement. It keeps
    the canonical 10,000-by-200,000-shot bootstrap memory bounded.
    """
    baseline = _failure_outcomes(baseline_failures, name="baseline")
    hybrid = _failure_outcomes(hybrid_failures, name="hybrid")
    if baseline.shape != hybrid.shape:
        raise ValueError("baseline and hybrid outcomes must have equal shape")
    if baseline.size == 0:
        raise ValueError("paired outcomes must contain at least one shot")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")

    both_succeed = int(np.count_nonzero(~baseline & ~hybrid))
    baseline_only = int(np.count_nonzero(baseline & ~hybrid))
    hybrid_only = int(np.count_nonzero(~baseline & hybrid))
    both_fail = int(np.count_nonzero(baseline & hybrid))
    shots = int(baseline.size)

    probabilities = (
        np.array([baseline_only, both_succeed + both_fail, hybrid_only], dtype=np.float64) / shots
    )
    bootstrap_counts = np.random.default_rng(bootstrap_seed).multinomial(
        shots,
        probabilities,
        size=samples,
    )
    bootstrap_delta = (
        bootstrap_counts[:, 2].astype(np.float64) - bootstrap_counts[:, 0].astype(np.float64)
    ) / shots
    low, high = np.percentile(bootstrap_delta, (2.5, 97.5))

    delta = (hybrid_only - baseline_only) / shots
    return {
        "baseline": _wilson_summary(baseline),
        "baseline_only_failure": baseline_only,
        "block_error_delta": float(delta),
        "block_error_delta_95ci_high": float(high + 0.0),
        "block_error_delta_95ci_low": float(low + 0.0),
        "both_fail": both_fail,
        "both_succeed": both_succeed,
        "bootstrap_samples": samples,
        "bootstrap_seed": bootstrap_seed,
        "hybrid": _wilson_summary(hybrid),
        "hybrid_only_failure": hybrid_only,
        "shots": shots,
    }


def accuracy_compatible(
    paired_summary: Mapping[str, object],
    *,
    syndrome_valid: bool,
) -> bool:
    """Apply the accuracy-only compatibility gate declared by the campaign."""
    if not isinstance(syndrome_valid, (bool, np.bool_)):
        raise TypeError("syndrome_valid must be boolean")
    try:
        low = float(paired_summary["block_error_delta_95ci_low"])
        high = float(paired_summary["block_error_delta_95ci_high"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("paired summary is missing a valid delta interval") from error
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("paired block-error delta interval is invalid")
    return bool(syndrome_valid and low <= 0.0)


def adaptive_stop_reason(
    failure_counts: Mapping[str, object],
    *,
    shots: int,
    target_failures: int,
    shot_cap: int,
) -> str | None:
    """Return a scientific stopping reason, or ``None`` while collection continues."""
    decoder_names = {"baseline", "soft_prior", "residual"}
    if set(failure_counts) != decoder_names:
        raise ValueError("failure counts must cover baseline, soft_prior, and residual")
    if any(
        type(failure_counts[name]) is not int or failure_counts[name] < 0 for name in decoder_names
    ):
        raise ValueError("failure counts must be non-negative integers")
    if type(shots) is not int or shots < 0:
        raise ValueError("shots must be a non-negative integer")
    if type(target_failures) is not int or target_failures <= 0:
        raise ValueError("target_failures must be a positive integer")
    if type(shot_cap) is not int or shot_cap <= 0 or shots > shot_cap:
        raise ValueError("shot_cap must be positive and no smaller than shots")
    if all(failure_counts[name] >= target_failures for name in decoder_names):
        return "target_failures"
    if shots >= shot_cap:
        return "shot_cap"
    return None
