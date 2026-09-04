"""Inference utilities whose sampling unit is an independent sequence."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.stats import t as student_t


def _sequence_values(values: np.ndarray, *, name: str, minimum: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must contain one value per independent sequence")
    if array.size < minimum:
        raise ValueError(f"{name} must contain at least {minimum} independent sequence(s)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be complete and finite; no imputation is permitted")
    return array


def _positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed_tuple(seeds: Iterable[int]) -> tuple[int, ...]:
    result = tuple(seeds)
    if not result:
        raise ValueError("seeds must contain at least one integer")
    if any(type(seed) is not int or seed < 0 for seed in result):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(result)) != len(result):
        raise ValueError("seeds must be unique")
    return result


def _wild_bootstrap(
    differences: np.ndarray,
    *,
    mu0: float,
    observed_statistic: float,
    seed: int,
    draws: int,
) -> dict[str, object]:
    residuals = differences - differences.mean()
    generator = np.random.default_rng(seed)
    max_attempts = 10 * draws
    attempted = 0
    valid = 0
    lower_or_equal = 0
    n_sequences = differences.size

    while valid < draws and attempted < max_attempts:
        batch_size = min(4_096, max_attempts - attempted, draws - valid)
        signs = generator.choice((-1, 1), size=(batch_size, n_sequences))
        bootstrap = mu0 + signs * residuals
        standard_deviations = bootstrap.std(axis=1, ddof=1)
        valid_mask = np.isfinite(standard_deviations) & (standard_deviations > 0.0)
        valid_statistics = (
            (bootstrap[valid_mask].mean(axis=1) - mu0)
            / (standard_deviations[valid_mask] / math.sqrt(n_sequences))
        )
        needed = draws - valid
        valid_statistics = valid_statistics[:needed]
        lower_or_equal += int(np.count_nonzero(valid_statistics <= observed_statistic))
        valid += int(valid_statistics.size)
        attempted += batch_size

    if valid < draws:
        return {
            "attempted_draws": attempted,
            "pvalue_lower": None,
            "seed": seed,
            "status": "bootstrap_degenerate",
            "valid_draws": valid,
        }
    return {
        "attempted_draws": attempted,
        "pvalue_lower": (1.0 + lower_or_equal) / (draws + 1.0),
        "seed": seed,
        "status": "ok",
        "valid_draws": valid,
    }


def paired_sequence_inference(
    differences: np.ndarray,
    *,
    mu0: float,
    seeds: Iterable[int],
    draws: int,
) -> dict[str, object]:
    """Run lower-tail paired inference on one mean difference per sequence.

    Wild-bootstrap draws are explicitly centered under ``mu0`` and studentized.
    Their p-values use the finite-sample ``(1 + count) / (draws + 1)`` correction.
    """
    values = _sequence_values(differences, name="differences", minimum=2)
    if isinstance(mu0, (bool, np.bool_)) or not isinstance(
        mu0, (int, float, np.integer, np.floating)
    ):
        raise TypeError("mu0 must be numeric")
    mu0 = float(mu0)
    if not math.isfinite(mu0):
        raise ValueError("mu0 must be finite")
    draws = _positive_integer(draws, name="draws")
    seed_values = _seed_tuple(seeds)

    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    common = {
        "mean_difference": mean,
        "mu0": mu0,
        "n_sequences": int(values.size),
        "standard_deviation": standard_deviation,
    }
    if standard_deviation == 0.0:
        degenerate_bootstraps = [
            {
                "attempted_draws": 0,
                "pvalue_lower": None,
                "seed": seed,
                "status": "degenerate_variance",
                "valid_draws": 0,
            }
            for seed in seed_values
        ]
        return {
            **common,
            "paired_t_pvalue_lower": None,
            "paired_t_status": "degenerate_variance",
            "standard_error": 0.0,
            "status": "degenerate_variance",
            "t_statistic": None,
            "wild_bootstrap": degenerate_bootstraps,
        }

    standard_error = standard_deviation / math.sqrt(values.size)
    statistic = (mean - mu0) / standard_error
    bootstraps = [
        _wild_bootstrap(
            values,
            mu0=mu0,
            observed_statistic=statistic,
            seed=seed,
            draws=draws,
        )
        for seed in seed_values
    ]
    status = (
        "bootstrap_degenerate"
        if any(result["status"] != "ok" for result in bootstraps)
        else "ok"
    )
    return {
        **common,
        "paired_t_pvalue_lower": float(student_t.cdf(statistic, df=values.size - 1)),
        "paired_t_status": "ok",
        "standard_error": standard_error,
        "status": status,
        "t_statistic": statistic,
        "wild_bootstrap": bootstraps,
    }


def cluster_percentile_interval(
    values: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, object]:
    """Return a deterministic 95% percentile interval over whole sequences."""
    sequence_values = _sequence_values(values, name="values")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    draws = _positive_integer(draws, name="draws")
    generator = np.random.default_rng(seed)
    n_sequences = int(sequence_values.size)
    bootstrap_means = np.empty(draws, dtype=np.float64)
    offset = 0
    while offset < draws:
        batch_size = min(4_096, draws - offset)
        indices = generator.integers(0, n_sequences, size=(batch_size, n_sequences))
        bootstrap_means[offset : offset + batch_size] = sequence_values[indices].mean(axis=1)
        offset += batch_size
    low, high = np.percentile(bootstrap_means, (2.5, 97.5))
    return {
        "confidence_level": 0.95,
        "draws": draws,
        "estimate": float(sequence_values.mean()),
        "interval_high": float(high),
        "interval_low": float(low),
        "n_sequences": n_sequences,
        "seed": seed,
    }
