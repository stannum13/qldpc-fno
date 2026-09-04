"""Shot-level paired statistics for decoder block-failure outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.stats import binomtest

_DECODER_NAMES = {"baseline", "soft_prior", "residual"}


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
    alpha: float = 0.05,
) -> dict[str, object]:
    """Summarize paired block failures with exact discordant-pair inference."""
    baseline = _failure_outcomes(baseline_failures, name="baseline")
    hybrid = _failure_outcomes(hybrid_failures, name="hybrid")
    if baseline.shape != hybrid.shape:
        raise ValueError("baseline and hybrid outcomes must have equal shape")
    if baseline.size == 0:
        raise ValueError("paired outcomes must contain at least one shot")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")

    both_succeed = int(np.count_nonzero(~baseline & ~hybrid))
    baseline_only = int(np.count_nonzero(baseline & ~hybrid))
    hybrid_only = int(np.count_nonzero(~baseline & hybrid))
    both_fail = int(np.count_nonzero(baseline & hybrid))
    shots = int(baseline.size)
    discordant = baseline_only + hybrid_only

    if discordant:
        two_sided = float(binomtest(hybrid_only, discordant, 0.5).pvalue)
        harm = float(binomtest(hybrid_only, discordant, 0.5, alternative="greater").pvalue)
        benefit = float(binomtest(hybrid_only, discordant, 0.5, alternative="less").pvalue)
        interval = binomtest(hybrid_only, discordant).proportion_ci(
            confidence_level=1.0 - alpha,
            method="exact",
        )
        harm_share: float | None = hybrid_only / discordant
        low: float | None = float(interval.low)
        high: float | None = float(interval.high)
    else:
        two_sided = harm = benefit = 1.0
        harm_share = low = high = None

    return {
        "baseline": _wilson_summary(baseline),
        "baseline_only_failure": baseline_only,
        "block_error_delta": (hybrid_only - baseline_only) / shots,
        "both_fail": both_fail,
        "both_succeed": both_succeed,
        "discordant_pairs": discordant,
        "hybrid": _wilson_summary(hybrid),
        "hybrid_harm_share_given_discordance": harm_share,
        "hybrid_harm_share_given_discordance_95ci_high": high,
        "hybrid_harm_share_given_discordance_95ci_low": low,
        "hybrid_only_failure": hybrid_only,
        "mcnemar_exact_pvalue_benefit": benefit,
        "mcnemar_exact_pvalue_harm": harm,
        "mcnemar_exact_pvalue_two_sided": two_sided,
        "shots": shots,
    }


def paired_comparison_status(
    paired_summary: Mapping[str, object],
    *,
    fixed_sample: bool,
    alpha: float = 0.05,
) -> str:
    """Classify exact paired comparison evidence for a fixed sample."""
    if not isinstance(fixed_sample, (bool, np.bool_)):
        raise TypeError("fixed_sample must be boolean")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if not isinstance(paired_summary, Mapping):
        raise TypeError("paired_summary must be a mapping")

    try:
        shots = paired_summary["shots"]
        both_succeed = paired_summary["both_succeed"]
        baseline_only = paired_summary["baseline_only_failure"]
        hybrid_only = paired_summary["hybrid_only_failure"]
        both_fail = paired_summary["both_fail"]
        discordant = paired_summary["discordant_pairs"]
        pvalue = paired_summary["mcnemar_exact_pvalue_two_sided"]
    except KeyError as error:
        raise ValueError("paired summary is missing required counts or p-value") from error

    counts = {
        "shots": shots,
        "both_succeed": both_succeed,
        "baseline_only_failure": baseline_only,
        "hybrid_only_failure": hybrid_only,
        "both_fail": both_fail,
        "discordant_pairs": discordant,
    }
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in counts.values()
    ):
        raise ValueError("paired summary counts must be non-negative integers")
    shots = int(shots)
    both_succeed = int(both_succeed)
    baseline_only = int(baseline_only)
    hybrid_only = int(hybrid_only)
    both_fail = int(both_fail)
    discordant = int(discordant)
    if shots == 0 or both_succeed + baseline_only + hybrid_only + both_fail != shots:
        raise ValueError("paired summary counts must add up to shots")
    if discordant != baseline_only + hybrid_only:
        raise ValueError("discordant_pairs must equal the discordant counts")
    try:
        pvalue = float(pvalue)
    except (TypeError, ValueError) as error:
        raise ValueError("McNemar exact p-value must be finite and in [0, 1]") from error
    if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
        raise ValueError("McNemar exact p-value must be finite and in [0, 1]")

    if not fixed_sample:
        return "not_fixed_sample"
    if discordant == 0:
        return "no_discordances"
    if pvalue <= alpha and hybrid_only > baseline_only:
        return "harm_detected"
    if pvalue <= alpha and baseline_only > hybrid_only:
        return "benefit_detected"
    return "inconclusive"


def test_stop_reason(
    failure_counts: Mapping[str, object],
    *,
    shots: int,
    target_failures: int,
    shot_cap: int,
    mode: str,
) -> str | None:
    """Return the configured test stopping reason, or ``None`` while collection continues."""
    if set(failure_counts) != _DECODER_NAMES:
        raise ValueError("failure counts must cover baseline, soft_prior, and residual")
    if any(
        type(failure_counts[name]) is not int or failure_counts[name] < 0
        for name in _DECODER_NAMES
    ):
        raise ValueError("failure counts must be non-negative integers")
    if type(shots) is not int or shots < 0:
        raise ValueError("shots must be a non-negative integer")
    if type(target_failures) is not int or target_failures <= 0:
        raise ValueError("target_failures must be a positive integer")
    if type(shot_cap) is not int or shot_cap <= 0 or shots > shot_cap:
        raise ValueError("shot_cap must be positive and no smaller than shots")
    if mode not in {"adaptive", "fixed"}:
        raise ValueError("test stopping mode must be 'adaptive' or 'fixed'")
    if mode == "adaptive" and all(
        int(failure_counts[name]) >= target_failures for name in _DECODER_NAMES
    ):
        return "target_failures"
    if shots >= shot_cap:
        return "shot_cap"
    return None
