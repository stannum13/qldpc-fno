from __future__ import annotations

import numpy as np
import pytest
from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode, PlanarMPSDecoder

from qldpc_fno.decision.planar_shot_accuracy import (
    logical_class_recovery,
    score_recovery,
    two_view_tolerance_decision,
)
from qldpc_fno.decision.tensor_network import PlanarCosetMasses


def test_class_recoveries_reproduce_syndrome_and_are_stabilizer_invariant() -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    error[0] = 1
    syndrome = pt.bsp(error, code.stabilizers.T)

    for logical_class in range(4):
        recovery = logical_class_recovery(code, syndrome, logical_class)
        np.testing.assert_array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome)

        scored = score_recovery(code, error, syndrome, recovery)
        assert scored["syndrome_valid"] is True
        assert scored["logical_failure"] == bool(
            np.any(pt.bsp((error + recovery) % 2, code.logicals.T))
        )
        np.testing.assert_array_equal(scored["residual_syndrome"], np.zeros_like(syndrome))

        for stabilizer in code.stabilizers:
            stabilized = (recovery + stabilizer) % 2
            stabilized_scored = score_recovery(code, error, syndrome, stabilized)
            assert stabilized_scored["syndrome_valid"] is True
            assert stabilized_scored["logical_failure"] == scored["logical_failure"]
            np.testing.assert_array_equal(
                stabilized_scored["logical_signature"], scored["logical_signature"]
            )


def test_logical_class_recoveries_match_qecsim_representative_order() -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    error[0] = 1
    syndrome = pt.bsp(error, code.stabilizers.T)
    sample = PlanarMPSDecoder.sample_recovery(code, syndrome)
    expected = (
        sample,
        sample.copy().logical_x(),
        sample.copy().logical_x().logical_z(),
        sample.copy().logical_z(),
    )

    for logical_class, representative in enumerate(expected):
        recovery = logical_class_recovery(code, syndrome, logical_class)
        np.testing.assert_array_equal(recovery, representative.to_bsf())


@pytest.mark.parametrize(
    "syndrome",
    (np.zeros(11, dtype=np.uint8), np.full(12, 2, dtype=np.uint8)),
)
def test_logical_class_recovery_rejects_invalid_syndrome(syndrome: np.ndarray) -> None:
    with pytest.raises(ValueError, match="syndrome"):
        logical_class_recovery(PlanarCode(3, 3), syndrome, 0)


@pytest.mark.parametrize("logical_class", (-1, 4, True, 1.0))
def test_logical_class_recovery_rejects_invalid_class(logical_class: object) -> None:
    code = PlanarCode(3, 3)
    syndrome = np.zeros(code.stabilizers.shape[0], dtype=np.uint8)

    with pytest.raises(ValueError, match="logical_class"):
        logical_class_recovery(code, syndrome, logical_class)  # type: ignore[arg-type]


def test_logical_shift_is_syndrome_valid_but_changes_failure_outcome() -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    syndrome = pt.bsp(error, code.stabilizers.T)

    reference = score_recovery(code, error, syndrome, logical_class_recovery(code, syndrome, 0))
    logical_shift = score_recovery(code, error, syndrome, logical_class_recovery(code, syndrome, 1))

    assert reference["syndrome_valid"] is True
    assert logical_shift["syndrome_valid"] is True
    assert reference["logical_failure"] is False
    assert logical_shift["logical_failure"] is True


def test_score_recovery_reports_a_syndrome_invalid_recovery() -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    error[0] = 1
    syndrome = pt.bsp(error, code.stabilizers.T)

    scored = score_recovery(code, error, syndrome, np.zeros_like(error))

    assert scored["syndrome_valid"] is False
    np.testing.assert_array_equal(scored["recovery_syndrome"], np.zeros_like(syndrome))
    np.testing.assert_array_equal(scored["residual_syndrome"], syndrome)


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("error", np.zeros(17, dtype=np.uint8)),
        ("syndrome", np.zeros(11, dtype=np.uint8)),
        ("recovery", np.full(18, 2, dtype=np.uint8)),
    ),
)
def test_score_recovery_rejects_invalid_shapes_and_nonbinary_values(
    argument: str, value: np.ndarray
) -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = logical_class_recovery(code, syndrome, 0)
    arguments: dict[str, np.ndarray] = {
        "error": error,
        "syndrome": syndrome,
        "recovery": recovery,
    }
    arguments[argument] = value

    with pytest.raises(ValueError):
        score_recovery(code, **arguments)


def _coset_masses(
    *,
    selected_class: int,
    mode: str,
    chi: int | None,
    tol: float | None,
    flops: int,
) -> PlanarCosetMasses:
    return PlanarCosetMasses(
        masses=np.array([0.4, 0.3, 0.2, 0.1]),
        selected_class=selected_class,
        rows=5,
        columns=5,
        error_rate=0.1,
        chi=chi,
        tol=tol,
        mode=mode,
        work={"estimated_arithmetic_flops": flops},
    )


def test_two_view_tolerance_accepts_agreement_and_accounts_for_both_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_masses(**kwargs: object) -> PlanarCosetMasses:
        calls.append(kwargs)
        return _coset_masses(
            selected_class=2,
            mode=str(kwargs["mode"]),
            chi=kwargs["chi"] if isinstance(kwargs["chi"], int) else None,
            tol=kwargs["tol"] if isinstance(kwargs["tol"], float) else None,
            flops=10 if kwargs["mode"] == "columns" else 20,
        )

    monkeypatch.setattr(
        "qldpc_fno.decision.planar_shot_accuracy.planar_mps_coset_masses", fake_masses
    )

    decision = two_view_tolerance_decision(np.zeros(40, dtype=np.uint8), 0.1)

    assert [call["mode"] for call in calls] == ["columns", "rows"]
    assert all(call["tol"] == 0.01 for call in calls)
    assert all(call["chi"] is None for call in calls)
    assert decision["selected_class"] == 2
    assert decision["accepted"] is True
    assert decision["used_fallback"] is False
    assert decision["estimated_arithmetic_flops"] == 30


def test_two_view_tolerance_falls_back_to_column_chi_eight_and_accounts_for_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_masses(**kwargs: object) -> PlanarCosetMasses:
        calls.append(kwargs)
        selected_class = 0 if kwargs["mode"] == "columns" and kwargs["chi"] is None else 1
        return _coset_masses(
            selected_class=selected_class,
            mode=str(kwargs["mode"]),
            chi=kwargs["chi"] if isinstance(kwargs["chi"], int) else None,
            tol=kwargs["tol"] if isinstance(kwargs["tol"], float) else None,
            flops=(10, 20, 30)[len(calls) - 1],
        )

    monkeypatch.setattr(
        "qldpc_fno.decision.planar_shot_accuracy.planar_mps_coset_masses", fake_masses
    )

    decision = two_view_tolerance_decision(np.zeros(40, dtype=np.uint8), 0.1)

    assert [(call["mode"], call["chi"], call["tol"]) for call in calls] == [
        ("columns", None, 0.01),
        ("rows", None, 0.01),
        ("columns", 8, None),
    ]
    assert decision["selected_class"] == 1
    assert decision["accepted"] is False
    assert decision["used_fallback"] is True
    assert decision["estimated_arithmetic_flops"] == 60


@pytest.mark.parametrize(
    ("syndrome", "error_rate"),
    (
        (np.zeros(39, dtype=np.uint8), 0.1),
        (np.full(40, 2, dtype=np.uint8), 0.1),
        (np.full(40, 0.5, dtype=np.float64), 0.1),
        (np.zeros(40, dtype=np.uint8), 0.0),
        (np.zeros(40, dtype=np.uint8), 1.0),
        (np.zeros(40, dtype=np.uint8), float("nan")),
    ),
)
def test_two_view_tolerance_rejects_invalid_policy_inputs(
    syndrome: np.ndarray, error_rate: float
) -> None:
    with pytest.raises(ValueError):
        two_view_tolerance_decision(syndrome, error_rate)
