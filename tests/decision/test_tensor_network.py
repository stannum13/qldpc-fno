from __future__ import annotations

import numpy as np
import pytest
from qecsim.tensortools import mps2d

from qldpc_fno.decision import tensor_network
from qldpc_fno.decision.tensor_network import (
    InvalidCosetMassError,
    exact_planar_coset_masses,
    planar_mps_coset_masses,
)


@pytest.mark.parametrize(("kind", "expected_flops"), [("svd", 112), ("qr", 18)])
@pytest.mark.parametrize("in_sweep", [False, True])
def test_failed_decomposition_attempt_is_charged_before_call_and_restores_wrappers(
    monkeypatch: pytest.MonkeyPatch, kind: str, expected_flops: int, in_sweep: bool
) -> None:
    mps = tensor_network.qecsim_mps
    matrix = np.ones((3, 2))
    failure = np.linalg.LinAlgError("injected decomposition failure")
    observed = {}

    def fail(*args: object, **kwargs: object) -> object:
        observed["calls"] = work[f"{kind}_calls"]
        observed["flops"] = work["estimated_dense_decomposition_flops"]
        observed["attempts"] = [dict(item) for item in work.get("decomposition_attempts", [])]
        raise failure

    def contract(*args: object, **kwargs: object) -> object:
        return getattr(mps.sp_linalg, kind)(matrix)

    monkeypatch.setattr(mps.sp_linalg, kind, fail)
    monkeypatch.setattr(mps2d, "contract", contract)
    originals = (
        mps.contract_pairwise,
        mps.truncate,
        mps.sp_linalg.svd,
        mps.sp_linalg.qr,
        mps.np.einsum,
        mps2d.contract,
    )
    with (
        pytest.raises(np.linalg.LinAlgError) as captured,
        tensor_network._trace_mps_work() as work,
    ):
        if in_sweep:
            mps2d.contract(np.empty((2, 3), dtype=object))
        else:
            getattr(mps.sp_linalg, kind)(matrix)

    assert captured.value is failure
    assert observed["calls"] == 1
    assert observed["flops"] == expected_flops
    attempt = {
        "decomposition_kind": kind,
        "matrix_rows": 3,
        "matrix_columns": 2,
        "estimated_flops": expected_flops,
        "succeeded": False,
        "exception_type": None,
    }
    assert observed["attempts"] == [attempt]
    assert work["decomposition_attempts"] == [attempt | {"exception_type": "LinAlgError"}]
    assert work[f"{kind}_calls"] == 1
    assert work["decomposition_input_elements"] == 6
    assert work["estimated_arithmetic_flops"] == expected_flops
    assert work["truncation_events"] == []
    sweeps = work["contraction_sweeps"]
    assert len(sweeps) == int(in_sweep)
    if in_sweep:
        assert sweeps[0]["failure"]["exception_type"] == "LinAlgError"
        assert sweeps[0]["estimated_arithmetic_flops"] == expected_flops
    assert work["terminal_residual_estimated_arithmetic_flops"] == (
        0 if in_sweep else expected_flops
    )
    assert (
        sum(sweep["estimated_arithmetic_flops"] for sweep in sweeps)
        + work["terminal_residual_estimated_arithmetic_flops"]
        == work["estimated_arithmetic_flops"]
    )
    assert (
        mps.contract_pairwise,
        mps.truncate,
        mps.sp_linalg.svd,
        mps.sp_linalg.qr,
        mps.np.einsum,
        mps2d.contract,
    ) == originals


@pytest.mark.parametrize(("kind", "expected_flops"), [("svd", 112), ("qr", 18)])
def test_successful_decomposition_attempt_metadata(kind: str, expected_flops: int) -> None:
    with tensor_network._trace_mps_work() as work:
        getattr(tensor_network.qecsim_mps.sp_linalg, kind)(np.ones((2, 3)))

    assert work["decomposition_attempts"] == [
        {
            "decomposition_kind": kind,
            "matrix_rows": 2,
            "matrix_columns": 3,
            "estimated_flops": expected_flops,
            "succeeded": True,
            "exception_type": None,
        }
    ]
    assert work["estimated_dense_decomposition_flops"] == expected_flops
    assert work["estimated_arithmetic_flops"] == expected_flops


def test_svd_retry_charges_both_attempts_and_only_summarizes_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mps = tensor_network.qecsim_mps
    original_svd = mps.sp_linalg.svd
    drivers = []

    def fail_first(matrix: np.ndarray, *args: object, **kwargs: object) -> object:
        drivers.append(kwargs["lapack_driver"])
        if len(drivers) == 1:
            raise np.linalg.LinAlgError("retry with gesvd")
        return original_svd(matrix, *args, **kwargs)

    def retrying_truncate(tensors: object, **kwargs: object) -> object:
        matrix = np.array([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        try:
            mps.sp_linalg.svd(matrix, full_matrices=False, lapack_driver="gesdd")
        except np.linalg.LinAlgError:
            mps.sp_linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd")
        return tensors, 1.0

    monkeypatch.setattr(mps.sp_linalg, "svd", fail_first)
    monkeypatch.setattr(mps, "truncate", retrying_truncate)
    with tensor_network._trace_mps_work() as work:
        mps.truncate([np.ones((1, 2, 1, 2))], chi=1)

    assert drivers == ["gesdd", "gesvd"]
    assert work["svd_calls"] == 2
    assert work["decomposition_input_elements"] == 12
    assert work["estimated_dense_decomposition_flops"] == 224
    assert work["estimated_arithmetic_flops"] == 224
    assert work["terminal_residual_estimated_arithmetic_flops"] == 224
    attempts = work["decomposition_attempts"]
    assert [(item["succeeded"], item["exception_type"]) for item in attempts] == [
        (False, "LinAlgError"),
        (True, None),
    ]
    assert all(item["estimated_flops"] == 112 for item in attempts)
    (event,) = work["truncation_events"]
    assert event["cumulative_estimated_arithmetic_flops"] == 224
    (spectrum,) = event["spectral_summaries"]
    assert spectrum["singular_value_count"] == 2
    assert spectrum["retained_rank"] == 1
    assert spectrum["discarded_squared_weight_fraction"] == pytest.approx(0.1)
    assert mps.sp_linalg.svd is fail_first
    assert mps.truncate is retrying_truncate


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
        spectrum["retained_rank"] for event in events for spectrum in event["spectral_summaries"]
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
    assert (
        sum(sweep["estimated_arithmetic_flops"] for sweep in sweeps)
        + result.work["terminal_residual_estimated_arithmetic_flops"]
        == result.work["estimated_arithmetic_flops"]
    )


def test_passive_trace_does_not_change_coset_masses() -> None:
    syndrome = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.uint8)
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


def test_real_contraction_numerical_failure_retains_failed_sweeps_and_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = mps2d.contract

    def fail_after_contraction(*args: object, **kwargs: object) -> object:
        original(*args, **kwargs)
        raise ValueError("injected numerical failure")

    monkeypatch.setattr(mps2d, "contract", fail_after_contraction)
    with pytest.raises(ValueError) as captured:
        planar_mps_coset_masses(
            rows=3,
            columns=3,
            syndrome=np.zeros(12, dtype=np.uint8),
            error_rate=0.1,
            chi=None,
            tol=0.01,
            mode="columns",
            trace_work=True,
        )
    assert isinstance(captured.value, tensor_network.InvalidContractionError)
    work = captured.value.work
    assert work["estimated_arithmetic_flops"] > 0
    assert [sweep["label"] for sweep in work["contraction_sweeps"]] == [
        "columns:I-X",
        "columns:Z-Y",
    ]
    assert all(
        sweep["failure"]["exception_type"] == "ValueError" for sweep in work["contraction_sweeps"]
    )
    assert (
        sum(sweep["estimated_arithmetic_flops"] for sweep in work["contraction_sweeps"])
        + work["terminal_residual_estimated_arithmetic_flops"]
        == work["estimated_arithmetic_flops"]
    )
    assert mps2d.contract is fail_after_contraction


def test_real_contraction_unexpected_runtime_error_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("unexpected tracer defect")

    monkeypatch.setattr(mps2d, "contract", fail)
    with pytest.raises(RuntimeError, match="unexpected tracer defect"):
        planar_mps_coset_masses(
            rows=3,
            columns=3,
            syndrome=np.zeros(12, dtype=np.uint8),
            error_rate=0.1,
            chi=None,
            mode="columns",
            trace_work=True,
        )
