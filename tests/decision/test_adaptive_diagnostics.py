from __future__ import annotations

import math

import numpy as np
import pytest

from qldpc_fno.decision.adaptive_diagnostics import (
    diagnostic_availability_ledger,
    extract_diagnostic_row,
    jensen_shannon_divergence,
    maximum_log_ratio_discrepancy,
    probability_margin,
    summarize_spectra,
    total_variation,
)


def _action(probabilities: list[float], *, flops: int = 100) -> dict[str, object]:
    return {
        "probabilities": probabilities,
        "selected_class": int(np.argmax(probabilities)),
        "work": {
            "estimated_arithmetic_flops": flops,
            "truncation_events": [
                {
                    "spectral_summaries": [
                        {
                            "discarded_squared_weight_fraction": 0.02,
                            "spectral_entropy": 0.3,
                            "retained_rank": 3,
                        }
                    ]
                }
            ],
        },
    }


def _shot(*, mismatch: bool = True) -> dict[str, object]:
    return {
        "shot_id": "d5/p0.100000/i000001",
        "shot_index": 1,
        "error_rate": 0.1,
        "accepted": True,
        "used_fallback": False,
        "exact_class": 1,
        "policy_class": 0 if mismatch else 1,
        "class_mismatch": mismatch,
        "outcome_discordance": mismatch,
        "reference_valid": True,
        "reference_certificate": {"probability_margins": [0.2, 0.2]},
        "tensor": {
            "exact_columns": _action([0.3, 0.5, 0.1, 0.1], flops=1000),
            "exact_rows": _action([0.3, 0.5, 0.1, 0.1], flops=1000),
            "tolerance_columns": _action([0.6, 0.2, 0.1, 0.1], flops=100),
            "tolerance_rows": _action([0.55, 0.25, 0.1, 0.1], flops=120),
            "fixed_columns": _action([0.31, 0.49, 0.1, 0.1], flops=400),
            "fallback_columns": None,
        },
        "arms": {
            "exact": {"failure": True},
            "policy": {"failure": not mismatch},
        },
        "work": {"policy": 220, "fixed_chi8": 400, "exact": 2000},
        "invalid_tensor_work": {},
        "error_bsf": [0, 1],
        "syndrome": [1],
    }


def test_probability_diagnostics_have_expected_values_and_symmetry() -> None:
    left = [0.7, 0.2, 0.08, 0.02]
    right = [0.6, 0.3, 0.08, 0.02]

    assert probability_margin(left) == pytest.approx(0.5)
    assert total_variation(left, right) == pytest.approx(0.1)
    divergence = jensen_shannon_divergence(left, right)
    assert divergence > 0
    assert divergence == pytest.approx(jensen_shannon_divergence(right, left))
    assert jensen_shannon_divergence(left, left) == pytest.approx(0.0)
    assert maximum_log_ratio_discrepancy(left, left) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "probabilities",
    (
        [0.5, 0.5, 0.0],
        [0.5, 0.5, -0.1, 0.1],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, math.inf, 0.0],
    ),
)
def test_probability_diagnostics_reject_invalid_vectors(probabilities: list[float]) -> None:
    with pytest.raises(ValueError):
        probability_margin(probabilities)


def test_log_ratio_discrepancy_does_not_clip_zero_probabilities() -> None:
    assert maximum_log_ratio_discrepancy(
        [0.5, 0.5, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0]
    ) is None


def test_summarize_spectra_aggregates_charged_action_diagnostics() -> None:
    action = {
        "work": {
            "estimated_arithmetic_flops": 1234,
            "truncation_events": [
                {
                    "spectral_summaries": [
                        {
                            "discarded_squared_weight_fraction": 0.1,
                            "spectral_entropy": 0.2,
                            "retained_rank": 2,
                        },
                        {
                            "discarded_squared_weight_fraction": 0.3,
                            "spectral_entropy": 0.4,
                            "retained_rank": 4,
                        },
                    ]
                },
                {
                    "spectral_summaries": [
                        {
                            "discarded_squared_weight_fraction": 0.2,
                            "spectral_entropy": 0.6,
                            "retained_rank": 3,
                        }
                    ]
                },
            ],
        }
    }

    assert summarize_spectra(action) == {
        "available": True,
        "truncation_event_count": 2,
        "spectrum_count": 3,
        "maximum_discarded_squared_weight_fraction": pytest.approx(0.3),
        "mean_spectral_entropy": pytest.approx(0.4),
        "maximum_retained_rank": 4,
        "estimated_arithmetic_flops": 1234,
    }


@pytest.mark.parametrize("action", (None, {}, {"work": None}))
def test_summarize_spectra_marks_missing_actions_unavailable(action: object) -> None:
    summary = summarize_spectra(action)  # type: ignore[arg-type]

    assert summary["available"] is False
    assert summary["spectrum_count"] == 0


def test_probability_inputs_are_normalized_before_comparison() -> None:
    probabilities = np.array([7.0, 2.0, 0.8, 0.2])

    assert probability_margin(probabilities) == pytest.approx(0.5)
    assert total_variation(probabilities, probabilities / 10) == pytest.approx(0.0)


def test_extract_diagnostic_row_separates_features_from_offline_labels() -> None:
    row = extract_diagnostic_row(_shot())

    assert set(row) == {
        "identity",
        "decision",
        "inference_features",
        "post_refinement_diagnostics",
        "reference_labels",
        "physical_outcome_labels",
        "work",
    }
    assert row["identity"] == {
        "shot_id": "d5/p0.100000/i000001",
        "shot_index": 1,
        "error_rate": 0.1,
    }
    features = row["inference_features"]
    assert features["tolerance_class_agreement"] is True
    assert features["tolerance_total_variation"] == pytest.approx(0.05)
    assert features["tolerance_jensen_shannon"] > 0
    assert features["tolerance_log_ratio_discrepancy"] is not None
    assert features["tolerance_column_margin"] == pytest.approx(0.4)
    assert features["tolerance_row_margin"] == pytest.approx(0.3)
    assert features["tolerance_minimum_margin"] == pytest.approx(0.3)
    assert features["tolerance_column_spectra"]["available"] is True
    assert features["tolerance_row_spectra"]["available"] is True
    assert features["cheap_estimated_arithmetic_flops"] == 220

    paid = row["post_refinement_diagnostics"]
    assert paid["column_to_fixed_chi8_total_variation"] == pytest.approx(0.29)
    assert paid["fixed_chi8_to_reference_total_variation"] == pytest.approx(0.01)
    assert paid["fixed_chi8_class_matches_reference"] is True

    reference = row["reference_labels"]
    assert reference["exact_class"] == 1
    assert reference["policy_class"] == 0
    assert reference["class_mismatch"] is True
    assert reference["exact_margin"] == pytest.approx(0.2)
    assert reference["column_to_reference_total_variation"] == pytest.approx(0.3)
    assert reference["row_to_reference_total_variation"] == pytest.approx(0.25)

    outcomes = row["physical_outcome_labels"]
    assert outcomes == {
        "outcome_discordance": True,
        "exact_failure": True,
        "policy_failure": False,
        "change_helped_realized_outcome": True,
        "change_harmed_realized_outcome": False,
    }
    forbidden = {"error_bsf", "syndrome", "recovery_bsf", "failure", "exact_class"}
    assert forbidden.isdisjoint(features)


def test_diagnostic_availability_ledger_names_stage_and_cost_boundary() -> None:
    ledger = diagnostic_availability_ledger()

    assert [entry["signal"] for entry in ledger] == [
        "cross_view_disagreement",
        "decision_margin",
        "cross_scale_stability",
        "discarded_representation",
        "calibrated_ood_score",
    ]
    by_signal = {entry["signal"]: entry for entry in ledger}
    assert by_signal["cross_view_disagreement"]["earliest_stage"] == "cheap_action_complete"
    assert by_signal["decision_margin"]["earliest_stage"] == "cheap_action_complete"
    assert by_signal["discarded_representation"]["earliest_stage"] == "cheap_action_complete"
    assert by_signal["cross_scale_stability"]["earliest_stage"] == "refinement_complete"
    assert (
        by_signal["calibrated_ood_score"]["earliest_stage"]
        == "unavailable_without_fitted_calibrator"
    )
    assert by_signal["calibrated_ood_score"]["deployable_from_artifact"] is False
    assert all("fields" in entry and "cost_boundary" in entry for entry in ledger)
