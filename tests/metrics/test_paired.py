from __future__ import annotations

import numpy as np
import pytest

from qldpc_fno.metrics.paired import (
    accuracy_compatible,
    adaptive_stop_reason,
    paired_decoder_summary,
)


def test_paired_summary_counts_disagreements_and_is_reproducible() -> None:
    baseline = np.array([0, 0, 1, 1], dtype=bool)
    hybrid = np.array([0, 1, 0, 1], dtype=bool)

    first = paired_decoder_summary(baseline, hybrid, bootstrap_seed=9, samples=1_000)
    second = paired_decoder_summary(baseline, hybrid, bootstrap_seed=9, samples=1_000)

    assert first == second
    assert first["both_succeed"] == 1
    assert first["baseline_only_failure"] == 1
    assert first["hybrid_only_failure"] == 1
    assert first["both_fail"] == 1
    assert first["block_error_delta"] == 0.0
    assert first["baseline"]["block_errors"] == 2
    assert first["hybrid"]["block_errors"] == 2


def test_bootstrap_resamples_paired_shots_and_preserves_direction() -> None:
    baseline = np.array([1, 1, 1, 0], dtype=bool)
    hybrid = np.array([0, 0, 0, 0], dtype=bool)

    summary = paired_decoder_summary(baseline, hybrid, bootstrap_seed=19, samples=2_000)

    assert summary["block_error_delta"] == -0.75
    assert summary["block_error_delta_95ci_high"] <= 0.0
    assert summary["baseline"]["block_error_rate_95ci_low"] > 0.0
    assert summary["hybrid"]["block_error_rate_95ci_low"] == 0.0


@pytest.mark.parametrize(
    ("baseline", "hybrid", "message"),
    [
        (np.zeros((2, 1), dtype=bool), np.zeros(2, dtype=bool), "one-dimensional"),
        (np.zeros(2, dtype=bool), np.zeros(3, dtype=bool), "equal shape"),
        (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), "at least one shot"),
        (np.array([0, 2]), np.array([0, 1]), "boolean failure outcomes"),
    ],
)
def test_paired_summary_rejects_non_shot_outcomes(
    baseline: np.ndarray,
    hybrid: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_decoder_summary(baseline, hybrid, bootstrap_seed=1, samples=10)


def test_accuracy_compatibility_requires_validity_and_nonpositive_ci_low() -> None:
    compatible = {
        "block_error_delta_95ci_low": -0.1,
        "block_error_delta_95ci_high": 0.2,
    }
    better = {
        "block_error_delta_95ci_low": -0.3,
        "block_error_delta_95ci_high": -0.1,
    }
    worse = {
        "block_error_delta_95ci_low": 0.01,
        "block_error_delta_95ci_high": 0.2,
    }

    assert accuracy_compatible(compatible, syndrome_valid=True)
    assert accuracy_compatible(better, syndrome_valid=True)
    assert not accuracy_compatible(worse, syndrome_valid=True)
    assert not accuracy_compatible(compatible, syndrome_valid=False)


def test_accuracy_compatibility_ignores_latency_fields() -> None:
    first = {
        "block_error_delta_95ci_low": 0.0,
        "block_error_delta_95ci_high": 0.2,
        "latency_seconds": 1_000.0,
    }
    second = {**first, "latency_seconds": 0.000_001}

    assert accuracy_compatible(first, syndrome_valid=True)
    assert accuracy_compatible(second, syndrome_valid=True)


def test_adaptive_stop_requires_every_decoder_to_reach_failure_target() -> None:
    assert (
        adaptive_stop_reason(
            {"baseline": 3, "soft_prior": 2, "residual": 1},
            shots=3,
            target_failures=2,
            shot_cap=5,
        )
        is None
    )
    assert (
        adaptive_stop_reason(
            {"baseline": 3, "soft_prior": 2, "residual": 2},
            shots=3,
            target_failures=2,
            shot_cap=5,
        )
        == "target_failures"
    )
    assert (
        adaptive_stop_reason(
            {"baseline": 0, "soft_prior": 0, "residual": 0},
            shots=5,
            target_failures=2,
            shot_cap=5,
        )
        == "shot_cap"
    )
