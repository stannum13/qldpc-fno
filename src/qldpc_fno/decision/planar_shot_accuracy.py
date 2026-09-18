"""Auditable recovery and tolerance-policy primitives for planar-code shots."""

from __future__ import annotations

import numpy as np
from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode, PlanarMPSDecoder

from qldpc_fno.decision.tensor_network import PlanarCosetMasses, planar_mps_coset_masses


def _binary_vector(value: np.ndarray, *, length: int, name: str) -> np.ndarray:
    """Return a binary vector after validating its exact dimensionality."""
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, not {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} entries must be binary")
    return np.asarray(array, dtype=np.uint8)


def logical_class_recovery(
    code: PlanarCode, syndrome: np.ndarray, logical_class: int
) -> np.ndarray:
    """Return qecsim's I, X, Y, or Z representative for ``syndrome``."""
    syndrome_array = _binary_vector(syndrome, length=code.stabilizers.shape[0], name="syndrome")
    if type(logical_class) is not int or not 0 <= logical_class < 4:
        raise ValueError("logical_class must be an integer from 0 through 3")
    sample = PlanarMPSDecoder.sample_recovery(code, syndrome_array)
    paulis = (
        sample,
        sample.copy().logical_x(),
        sample.copy().logical_x().logical_z(),
        sample.copy().logical_z(),
    )
    return np.asarray(paulis[logical_class].to_bsf(), dtype=np.uint8)


def score_recovery(
    code: PlanarCode, error: np.ndarray, syndrome: np.ndarray, recovery: np.ndarray
) -> dict[str, object]:
    """Score a recovery from syndrome and residual logical commutation."""
    qubits = code.n_k_d[0]
    error_array = _binary_vector(error, length=2 * qubits, name="error")
    syndrome_array = _binary_vector(syndrome, length=code.stabilizers.shape[0], name="syndrome")
    recovery_array = _binary_vector(recovery, length=2 * qubits, name="recovery")
    recovery_syndrome = np.asarray(pt.bsp(recovery_array, code.stabilizers.T), dtype=np.uint8)
    residual = (error_array + recovery_array) % 2
    residual_syndrome = np.asarray(pt.bsp(residual, code.stabilizers.T), dtype=np.uint8)
    logical_signature = np.asarray(pt.bsp(residual, code.logicals.T), dtype=np.uint8)
    return {
        "syndrome_valid": bool(np.array_equal(recovery_syndrome, syndrome_array)),
        "recovery_syndrome": recovery_syndrome,
        "residual": residual,
        "residual_syndrome": residual_syndrome,
        "logical_signature": logical_signature,
        "logical_failure": bool(np.any(logical_signature)),
    }


def _estimated_flops(result: PlanarCosetMasses) -> int:
    if result.work is None:
        return 0
    return int(result.work.get("estimated_arithmetic_flops", 0))


def two_view_tolerance_decision(syndrome: np.ndarray, error_rate: float) -> dict[str, object]:
    """Choose an agreeing tolerance class, or a fixed-chi column fallback."""
    columns = planar_mps_coset_masses(
        rows=5,
        columns=5,
        syndrome=syndrome,
        error_rate=error_rate,
        chi=None,
        tol=0.01,
        mode="columns",
        trace_work=True,
    )
    rows = planar_mps_coset_masses(
        rows=5,
        columns=5,
        syndrome=syndrome,
        error_rate=error_rate,
        chi=None,
        tol=0.01,
        mode="rows",
        trace_work=True,
    )
    accepted = columns.selected_class == rows.selected_class
    fallback: PlanarCosetMasses | None = None
    selected = columns
    if not accepted:
        fallback = planar_mps_coset_masses(
            rows=5,
            columns=5,
            syndrome=syndrome,
            error_rate=error_rate,
            chi=8,
            tol=None,
            mode="columns",
            trace_work=True,
        )
        selected = fallback
    actions = (columns, rows) if fallback is None else (columns, rows, fallback)
    return {
        "selected_class": selected.selected_class,
        "accepted": accepted,
        "used_fallback": fallback is not None,
        "tolerance_columns": columns,
        "tolerance_rows": rows,
        "fallback_columns": fallback,
        "estimated_arithmetic_flops": sum(_estimated_flops(action) for action in actions),
    }
