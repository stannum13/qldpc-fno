from __future__ import annotations

import numpy as np
import pytest

from qldpc_fno.decision.tensor_network import (
    InvalidCosetMassError,
    exact_planar_coset_masses,
    planar_mps_coset_masses,
)


def test_unrestricted_mps_matches_independent_coset_enumeration() -> None:
    syndrome = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=np.uint8)

    exact = exact_planar_coset_masses(rows=3, columns=3, syndrome=syndrome, error_rate=0.1)
    contracted = planar_mps_coset_masses(
        rows=3,
        columns=3,
        syndrome=syndrome,
        error_rate=0.1,
        chi=None,
        mode="columns",
    )

    np.testing.assert_allclose(contracted.masses, exact.masses, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(contracted.probabilities.sum(), 1.0, atol=1e-14)
    assert contracted.selected_class == exact.selected_class
    assert exact.enumerated_errors == 4 * 2**12


def test_row_and_column_exact_contractions_agree() -> None:
    syndrome = np.zeros(12, dtype=np.uint8)

    columns = planar_mps_coset_masses(
        rows=3,
        columns=3,
        syndrome=syndrome,
        error_rate=0.08,
        chi=None,
        mode="columns",
    )
    rows = planar_mps_coset_masses(
        rows=3,
        columns=3,
        syndrome=syndrome,
        error_rate=0.08,
        chi=None,
        mode="rows",
    )

    np.testing.assert_allclose(columns.masses, rows.masses, rtol=1e-11, atol=1e-15)


def test_truncated_result_reports_the_requested_control() -> None:
    syndrome = np.zeros(12, dtype=np.uint8)

    result = planar_mps_coset_masses(
        rows=3,
        columns=3,
        syndrome=syndrome,
        error_rate=0.1,
        chi=2,
        mode="average",
        trace_work=True,
    )

    assert result.chi == 2
    assert result.tol is None
    assert result.mode == "average"
    assert result.masses.shape == (4,)
    assert np.all(result.masses >= 0)
    np.testing.assert_allclose(result.probabilities.sum(), 1.0, atol=1e-14)
    assert result.work is not None
    assert result.work["pairwise_contractions"] > 0
    assert result.work["svd_calls"] > 0
    assert result.work["estimated_arithmetic_flops"] > 0
    assert result.work["peak_observed_array_elements"] > 0
    assert result.work["single_shot_wall_seconds"] > 0
    assert result.work["truncation_events"]
    for event in result.work["truncation_events"]:
        assert event["input_bond_dimension"] >= event["output_bond_dimension"]
        assert event["requested_chi"] == 2
        assert event["requested_tol"] is None
        assert event["spectral_summaries"]
        for spectrum in event["spectral_summaries"]:
            assert 0 <= spectrum["discarded_squared_weight_fraction"] <= 1
            assert spectrum["retained_rank"] <= 2


def test_tolerance_allocates_bond_dimension_per_local_spectrum() -> None:
    syndrome = np.zeros(40, dtype=np.uint8)

    result = planar_mps_coset_masses(
        rows=5,
        columns=5,
        syndrome=syndrome,
        error_rate=0.1,
        chi=None,
        tol=0.01,
        mode="columns",
        trace_work=True,
    )

    assert result.chi is None
    assert result.tol == 0.01
    assert result.work is not None
    events = result.work["truncation_events"]
    assert events
    retained_ranks = {
        spectrum["retained_rank"]
        for event in events
        for spectrum in event["spectral_summaries"]
    }
    assert len(retained_ranks) > 1
    assert max(retained_ranks) > 1
    assert all(
        spectrum["requested_tol"] == 0.01
        for event in events
        for spectrum in event["spectral_summaries"]
    )
    sweeps = result.work["contraction_sweeps"]
    assert [sweep["label"] for sweep in sweeps] == ["columns:I-X", "columns:Z-Y"]
    assert sum(sweep["estimated_arithmetic_flops"] for sweep in sweeps) + result.work[
        "terminal_residual_estimated_arithmetic_flops"
    ] == result.work["estimated_arithmetic_flops"]


def test_passive_trace_does_not_change_coset_masses() -> None:
    syndrome = np.array(
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.uint8
    )
    kwargs = {
        "rows": 3,
        "columns": 3,
        "syndrome": syndrome,
        "error_rate": 0.1,
        "chi": 4,
        "tol": 0.001,
        "mode": "average",
    }

    untraced = planar_mps_coset_masses(**kwargs)
    traced = planar_mps_coset_masses(**kwargs, trace_work=True)

    np.testing.assert_array_equal(traced.masses, untraced.masses)
    assert traced.work is not None
    assert [sweep["label"] for sweep in traced.work["contraction_sweeps"]] == [
        "columns:I-X",
        "columns:Z-Y",
        "rows:I-Z",
        "rows:X-Y",
    ]


def test_tolerance_must_be_positive_and_finite() -> None:
    syndrome = np.zeros(12, dtype=np.uint8)

    for tol in (0.0, -0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="tol"):
            planar_mps_coset_masses(
                rows=3,
                columns=3,
                syndrome=syndrome,
                error_rate=0.1,
                chi=None,
                tol=tol,
                mode="columns",
            )


def test_negative_approximate_mass_is_reported_not_clipped() -> None:
    syndrome = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8)

    with pytest.raises(InvalidCosetMassError) as captured:
        planar_mps_coset_masses(
            rows=3,
            columns=3,
            syndrome=syndrome,
            error_rate=0.05,
            chi=2,
            mode="columns",
            trace_work=True,
        )

    assert np.any(captured.value.masses < 0)
    assert captured.value.work is not None
