"""Inference utilities whose sampling unit is an independent sequence."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.stats import t as student_t

_CONFIRMATORY_SEQUENCES = 64
_STUDENTIZED_BOOTSTRAP_DRAWS = 10_000


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
    minimum_sequences: int = 64,
) -> dict[str, object]:
    """Run lower-tail paired inference on one mean difference per sequence.

    Wild-bootstrap draws are explicitly centered under ``mu0`` and studentized.
    Their p-values use the finite-sample ``(1 + count) / (draws + 1)`` correction.
    """
    if type(minimum_sequences) is not int or minimum_sequences < 2:
        raise ValueError("minimum_sequences must be an integer at least 2")
    values = _sequence_values(
        differences,
        name="differences",
        minimum=minimum_sequences,
    )
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
        "minimum_sequences": minimum_sequences,
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


def studentized_sequence_interval(
    values: np.ndarray,
    *,
    seed: int,
    null_mean: float,
) -> dict[str, object]:
    """Return the fixed confirmatory centered Rademacher bootstrap-t evidence.

    Each input value and each Rademacher sign represents one independent
    sequence.  The resampling law and draw count are intentionally not
    configurable: this is the preregistered confirmatory procedure.
    """
    sequence_values = _sequence_values(
        values,
        name="values",
        minimum=_CONFIRMATORY_SEQUENCES,
    )
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(null_mean, (bool, np.bool_)) or not isinstance(
        null_mean, (int, float, np.integer, np.floating)
    ):
        raise TypeError("null_mean must be numeric")
    null_mean = float(null_mean)
    if not math.isfinite(null_mean):
        raise ValueError("null_mean must be finite")

    n_sequences = int(sequence_values.size)
    estimate = float(sequence_values.mean())
    standard_error = float(sequence_values.std(ddof=1) / math.sqrt(n_sequences))
    common = {
        "bootstrap_draws": _STUDENTIZED_BOOTSTRAP_DRAWS,
        "estimate": estimate,
        "n_sequences": n_sequences,
        "null_mean": null_mean,
        "seed": seed,
        "standard_error": standard_error,
    }
    if standard_error == 0.0:
        return {
            **common,
            "lower_95": None,
            "observed_t": None,
            "pvalue_greater": None,
            "pvalue_less": None,
            "status": "degenerate_variance",
            "two_sided_95": None,
            "upper_95": None,
            "valid_draws": 0,
        }

    residuals = sequence_values - estimate
    signs = np.random.default_rng(seed).choice(
        (-1.0, 1.0),
        size=(_STUDENTIZED_BOOTSTRAP_DRAWS, n_sequences),
    )
    null_draws = null_mean + signs * residuals
    resampled_standard_errors = null_draws.std(axis=1, ddof=1) / math.sqrt(
        n_sequences
    )
    valid = np.isfinite(resampled_standard_errors) & (resampled_standard_errors > 0.0)
    valid_draws = int(np.count_nonzero(valid))
    if valid_draws != _STUDENTIZED_BOOTSTRAP_DRAWS:
        return {
            **common,
            "lower_95": None,
            "observed_t": None,
            "pvalue_greater": None,
            "pvalue_less": None,
            "status": "bootstrap_degenerate",
            "two_sided_95": None,
            "upper_95": None,
            "valid_draws": valid_draws,
        }

    resampled_t = (null_draws.mean(axis=1) - null_mean) / resampled_standard_errors
    observed_t = (estimate - null_mean) / standard_error
    q_025, q_05, q_95, q_975 = np.quantile(
        resampled_t,
        (0.025, 0.05, 0.95, 0.975),
    )
    lower_95 = estimate - float(q_95) * standard_error
    upper_95 = estimate - float(q_05) * standard_error
    two_sided_low = estimate - float(q_975) * standard_error
    two_sided_high = estimate - float(q_025) * standard_error
    return {
        **common,
        "lower_95": lower_95,
        "observed_t": observed_t,
        "pvalue_greater": (
            1.0 + float(np.count_nonzero(resampled_t >= observed_t))
        )
        / (_STUDENTIZED_BOOTSTRAP_DRAWS + 1.0),
        "pvalue_less": (1.0 + float(np.count_nonzero(resampled_t <= observed_t)))
        / (_STUDENTIZED_BOOTSTRAP_DRAWS + 1.0),
        "status": "ok",
        "two_sided_95": (two_sided_low, two_sided_high),
        "upper_95": upper_95,
        "valid_draws": valid_draws,
    }
