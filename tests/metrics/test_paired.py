from __future__ import annotations

import numpy as np
import pytest

from qldpc_fno.metrics.paired import (
    paired_comparison_status,
    paired_decoder_summary,
)
from qldpc_fno.metrics.paired import test_stop_reason as stopping_reason


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


def test_paired_summary_interval_is_fixed_at_95_percent() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'alpha'"):
        paired_decoder_summary(  # type: ignore[call-arg]
            np.zeros(2, dtype=bool),
            np.zeros(2, dtype=bool),
            alpha=0.1,
        )


def _valid_status_summary() -> dict[str, object]:
    return paired_decoder_summary(
        np.zeros(8, dtype=np.bool_),
        np.ones(8, dtype=np.bool_),
    )


@pytest.mark.parametrize(
    ("summary", "error_type", "message"),
    [
        (None, TypeError, "paired_summary must be a mapping"),
        ({}, ValueError, "missing required counts or p-value"),
        ({"shots": 8}, ValueError, "missing required counts or p-value"),
    ],
)
def test_paired_status_rejects_non_mapping_or_incomplete_summary(
    summary: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        paired_comparison_status(summary, fixed_sample=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shots", -1),
        ("both_succeed", 1.5),
        ("baseline_only_failure", -1),
        ("hybrid_only_failure", "8"),
        ("both_fail", np.bool_(True)),
        ("discordant_pairs", -1),
    ],
)
def test_paired_status_rejects_non_integer_or_negative_counts(field: str, value: object) -> None:
    summary = _valid_status_summary()
    summary[field] = value
    with pytest.raises(ValueError, match="non-negative integers"):
        paired_comparison_status(summary, fixed_sample=True)


def test_paired_status_rejects_counts_that_do_not_sum_to_shots() -> None:
    summary = _valid_status_summary()
    summary["both_fail"] = 1
    with pytest.raises(ValueError, match="add up to shots"):
        paired_comparison_status(summary, fixed_sample=True)


def test_paired_status_rejects_inconsistent_discordant_count() -> None:
    summary = _valid_status_summary()
    summary["discordant_pairs"] = 7
    with pytest.raises(ValueError, match="discordant_pairs must equal"):
        paired_comparison_status(summary, fixed_sample=True)


@pytest.mark.parametrize("pvalue", [float("nan"), float("inf"), -0.1, 1.1])
def test_paired_status_rejects_invalid_exact_pvalue(pvalue: float) -> None:
    summary = _valid_status_summary()
    summary["mcnemar_exact_pvalue_two_sided"] = pvalue
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        paired_comparison_status(summary, fixed_sample=True)


@pytest.mark.parametrize("pvalue", [False, True, "0.0078125", None])
def test_paired_status_rejects_boolean_or_non_numeric_exact_pvalue(pvalue: object) -> None:
    summary = _valid_status_summary()
    summary["mcnemar_exact_pvalue_two_sided"] = pvalue
    with pytest.raises(TypeError, match="must be numeric"):
        paired_comparison_status(summary, fixed_sample=True)


def test_paired_status_rejects_pvalue_inconsistent_with_discordant_counts() -> None:
    summary = _valid_status_summary()
    summary["mcnemar_exact_pvalue_two_sided"] = 0.0
    with pytest.raises(ValueError, match="does not match discordant counts"):
        paired_comparison_status(summary, fixed_sample=True)


def test_paired_status_rejects_zero_for_tiny_nonzero_exact_pvalue() -> None:
    summary = paired_decoder_summary(
        np.zeros(60, dtype=np.bool_),
        np.ones(60, dtype=np.bool_),
    )
    assert 0.0 < summary["mcnemar_exact_pvalue_two_sided"] < 1e-15
    summary["mcnemar_exact_pvalue_two_sided"] = 0.0
    with pytest.raises(ValueError, match="does not match discordant counts"):
        paired_comparison_status(summary, fixed_sample=True)


@pytest.mark.parametrize("fixed_sample", [None, 1, "true"])
def test_paired_status_rejects_non_boolean_fixed_sample(fixed_sample: object) -> None:
    with pytest.raises(TypeError, match="fixed_sample must be boolean"):
        paired_comparison_status(_valid_status_summary(), fixed_sample=fixed_sample)  # type: ignore[arg-type]


def test_paired_status_threshold_is_fixed_at_five_percent() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'alpha'"):
        paired_comparison_status(  # type: ignore[call-arg]
            _valid_status_summary(),
            fixed_sample=True,
            alpha=0.1,
        )


def test_adaptive_stop_requires_every_decoder_to_reach_failure_target() -> None:
    assert (
        stopping_reason(
            {"baseline": 3, "soft_prior": 2, "residual": 1},
            shots=3,
            target_failures=2,
            shot_cap=5,
            mode="adaptive",
        )
        is None
    )
    assert (
        stopping_reason(
            {"baseline": 3, "soft_prior": 2, "residual": 2},
            shots=3,
            target_failures=2,
            shot_cap=5,
            mode="adaptive",
        )
        == "target_failures"
    )
    assert (
        stopping_reason(
            {"baseline": 0, "soft_prior": 0, "residual": 0},
            shots=5,
            target_failures=2,
            shot_cap=5,
            mode="adaptive",
        )
        == "shot_cap"
    )


def test_fixed_stop_ignores_failure_target_until_shot_cap() -> None:
    counts = {"baseline": 8, "soft_prior": 8, "residual": 8}
    assert (
        stopping_reason(
            counts,
            shots=8,
            target_failures=1,
            shot_cap=16,
            mode="fixed",
        )
        is None
    )
    assert (
        stopping_reason(
            counts,
            shots=16,
            target_failures=1,
            shot_cap=16,
            mode="fixed",
        )
        == "shot_cap"
    )


def test_adaptive_stop_preserves_target_failure_behavior() -> None:
    counts = {"baseline": 2, "soft_prior": 2, "residual": 2}
    assert (
        stopping_reason(
            counts,
            shots=4,
            target_failures=2,
            shot_cap=16,
            mode="adaptive",
        )
        == "target_failures"
    )


def test_stop_reason_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="test stopping mode must be 'adaptive' or 'fixed'"):
        stopping_reason(
            {"baseline": 0, "soft_prior": 0, "residual": 0},
            shots=0,
            target_failures=2,
            shot_cap=16,
            mode="unexpected",
        )
