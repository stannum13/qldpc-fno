from __future__ import annotations

import numpy as np
import pytest

from qldpc_fno.metrics.paired import (
    adaptive_stop_reason,
    paired_comparison_status,
    paired_decoder_summary,
)


def test_all_hybrid_only_discordances_have_nonzero_exact_uncertainty() -> None:
    summary = paired_decoder_summary(
        np.zeros(8, dtype=np.bool_),
        np.ones(8, dtype=np.bool_),
    )

    assert set(summary) == {
        "baseline",
        "baseline_only_failure",
        "block_error_delta",
        "both_fail",
        "both_succeed",
        "discordant_pairs",
        "hybrid",
        "hybrid_harm_share_given_discordance",
        "hybrid_harm_share_given_discordance_95ci_high",
        "hybrid_harm_share_given_discordance_95ci_low",
        "hybrid_only_failure",
        "mcnemar_exact_pvalue_benefit",
        "mcnemar_exact_pvalue_harm",
        "mcnemar_exact_pvalue_two_sided",
        "shots",
    }
    assert summary["hybrid_only_failure"] == 8
    assert summary["baseline_only_failure"] == 0
    assert summary["discordant_pairs"] == 8
    assert summary["mcnemar_exact_pvalue_two_sided"] == pytest.approx(0.0078125)
    assert summary["mcnemar_exact_pvalue_harm"] == pytest.approx(0.00390625)
    assert summary["mcnemar_exact_pvalue_benefit"] == 1.0
    assert 0.0 < summary["hybrid_harm_share_given_discordance_95ci_low"] < 1.0
    assert summary["hybrid_harm_share_given_discordance_95ci_high"] == 1.0
    assert paired_comparison_status(summary, fixed_sample=True) == "harm_detected"


def test_all_baseline_only_discordances_detect_benefit() -> None:
    summary = paired_decoder_summary(
        np.ones(8, dtype=np.bool_),
        np.zeros(8, dtype=np.bool_),
    )
    assert paired_comparison_status(summary, fixed_sample=True) == "benefit_detected"


def test_balanced_discordances_are_inconclusive() -> None:
    baseline = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.bool_)
    hybrid = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.bool_)
    summary = paired_decoder_summary(baseline, hybrid)
    assert paired_comparison_status(summary, fixed_sample=True) == "inconclusive"


def test_no_discordances_have_null_conditional_interval() -> None:
    outcomes = np.array([0, 1, 0, 1], dtype=np.bool_)
    summary = paired_decoder_summary(outcomes, outcomes)
    assert summary["discordant_pairs"] == 0
    assert summary["hybrid_harm_share_given_discordance"] is None
    assert summary["hybrid_harm_share_given_discordance_95ci_low"] is None
    assert summary["hybrid_harm_share_given_discordance_95ci_high"] is None
    assert paired_comparison_status(summary, fixed_sample=True) == "no_discordances"


def test_adaptive_or_partial_comparison_has_no_fixed_sample_status() -> None:
    summary = paired_decoder_summary(
        np.zeros(8, dtype=np.bool_),
        np.ones(8, dtype=np.bool_),
    )
    assert paired_comparison_status(summary, fixed_sample=False) == "not_fixed_sample"


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
        paired_decoder_summary(baseline, hybrid)


@pytest.mark.parametrize("alpha", [float("nan"), 0.0, 1.0, -0.1, 1.1])
def test_paired_summary_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be strictly between zero and one"):
        paired_decoder_summary(np.zeros(2, dtype=bool), np.zeros(2, dtype=bool), alpha=alpha)


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
