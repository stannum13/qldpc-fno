from __future__ import annotations

import math

import numpy as np
import pytest

from qldpc_fno.decision.adaptive_diagnostics import (
    jensen_shannon_divergence,
    maximum_log_ratio_discrepancy,
    probability_margin,
    summarize_spectra,
    total_variation,
)


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
