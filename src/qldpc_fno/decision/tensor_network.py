"""Audited planar-code tensor-network reference and exact small-code oracle.

The MPS contraction is delegated to qecsim 1.0b9's implementation of the
Bravyi--Suchara--Vargo planar decoder. Exact enumeration below is deliberately
independent of that contraction: it sums every stabilizer representative in
each of the four logical cosets.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any, Literal

import numpy as np
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import PlanarCode, PlanarMPSDecoder
from qecsim.tensortools import mps as qecsim_mps
from qecsim.tensortools import mps2d as qecsim_mps2d

ContractionMode = Literal["columns", "rows", "average"]

QEC_SIM_VERSION = "1.0b9"
QEC_SIM_REFERENCE_COMMIT = "24d6b8a320b292461b66b68fe4fba40c9ddc2257"
_MODE = {"columns": "c", "rows": "r", "average": "a"}


class InvalidCosetMassError(ValueError):
    """Raised when approximate contraction does not define a probability measure."""

    def __init__(
        self,
        masses: np.ndarray,
        work: dict[str, object] | None,
    ) -> None:
        super().__init__("MPS contraction returned invalid coset masses")
        self.masses = masses
        self.work = work


@dataclass(frozen=True)
class PlanarCosetMasses:
    """Four logical-coset masses in qecsim's I, X, Y, Z order."""

    masses: np.ndarray
    selected_class: int
    rows: int
    columns: int
    error_rate: float
    chi: int | None
    tol: float | None
    mode: str
    enumerated_errors: int | None = None
    work: dict[str, object] | None = None

    @property
    def probabilities(self) -> np.ndarray:
        total = float(self.masses.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("coset masses must have positive finite total mass")
        return self.masses / total


def _validate_inputs(
    rows: int,
    columns: int,
    syndrome: np.ndarray,
    error_rate: float,
) -> tuple[PlanarCode, np.ndarray]:
    if rows < 2 or columns < 2:
        raise ValueError("planar code dimensions must both be at least two")
    if not 0 < error_rate < 1:
        raise ValueError("error_rate must lie strictly between zero and one")
    code = PlanarCode(rows, columns)
    syndrome_array = np.asarray(syndrome, dtype=np.uint8)
    if syndrome_array.shape != (code.stabilizers.shape[0],):
        raise ValueError(
            f"syndrome must have shape {(code.stabilizers.shape[0],)}, "
            f"not {syndrome_array.shape}"
        )
    if np.any(syndrome_array > 1):
        raise ValueError("syndrome entries must be binary")
    return code, syndrome_array


def _physical_error_probability(bsf: np.ndarray, distribution: tuple[float, ...]) -> float:
    n = bsf.size // 2
    x = bsf[:n]
    z = bsf[n:]
    # qecsim probability order is I, X, Y, Z.
    pauli_indices = x + 3 * z - 2 * x * z
    return float(np.prod(np.asarray(distribution, dtype=np.float64)[pauli_indices]))


def _array_elements(value: Any) -> int:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return sum(_array_elements(item) for item in value.flat)
        return int(value.size)
    if isinstance(value, (list, tuple)):
        return sum(_array_elements(item) for item in value)
    return 0


@contextmanager
def _trace_mps_work() -> Iterator[dict[str, object]]:
    """Trace implementation-level work without presenting a hardware FLOP claim."""
    trace: dict[str, object] = {
        "pairwise_contractions": 0,
        "truncation_calls": 0,
        "svd_calls": 0,
        "qr_calls": 0,
        "einsum_calls": 0,
        "einsum_estimated_flops": 0,
        "pairwise_output_elements": 0,
        "decomposition_input_elements": 0,
        "estimated_dense_decomposition_flops": 0,
        "peak_observed_array_elements": 0,
        "truncation_events": [],
        "contraction_sweeps": [],
    }
    original_pairwise = qecsim_mps.contract_pairwise
    original_truncate = qecsim_mps.truncate
    original_svd = qecsim_mps.sp_linalg.svd
    original_qr = qecsim_mps.sp_linalg.qr
    original_einsum = qecsim_mps.np.einsum
    original_einsum_path = qecsim_mps.np.einsum_path
    original_mps2d_contract = qecsim_mps2d.contract
    active_sweep_index: int | None = None

    def cumulative_flops() -> int:
        return int(trace["einsum_estimated_flops"]) + int(
            trace["estimated_dense_decomposition_flops"]
        )

    def observe(*values: Any) -> None:
        trace["peak_observed_array_elements"] = max(
            int(trace["peak_observed_array_elements"]),
            sum(_array_elements(value) for value in values),
        )

    def pairwise(left: Any, right: Any) -> Any:
        result = original_pairwise(left, right)
        trace["pairwise_contractions"] += 1
        trace["pairwise_output_elements"] += _array_elements(result)
        observe(left, right, result)
        return result

    active_truncation: dict[str, object] | None = None

    def truncate(mps: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active_truncation
        requested_chi = kwargs.get("chi", args[0] if args else None)
        requested_tol = kwargs.get("tol", args[1] if len(args) > 1 else None)
        event: dict[str, object] = {
            "index": int(trace["truncation_calls"]),
            "requested_chi": requested_chi,
            "requested_tol": requested_tol,
            "sweep_index": active_sweep_index,
            "input_bond_dimension": int(qecsim_mps.bond_dimension(mps)),
            "input_elements": _array_elements(mps),
            "spectral_summaries": [],
        }
        if active_truncation is not None:
            raise RuntimeError("nested qecsim truncation trace is unsupported")
        active_truncation = event
        try:
            result = original_truncate(mps, *args, **kwargs)
        finally:
            active_truncation = None
        trace["truncation_calls"] += 1
        event["output_bond_dimension"] = int(qecsim_mps.bond_dimension(result[0]))
        event["output_elements"] = _array_elements(result[0])
        event["normalization_factor"] = float(result[1])
        event["cumulative_estimated_arithmetic_flops"] = cumulative_flops()
        truncation_events = trace["truncation_events"]
        assert isinstance(truncation_events, list)
        truncation_events.append(event)
        observe(mps, result)
        return result

    def einsum(subscripts: str, *operands: Any, **kwargs: Any) -> Any:
        report = original_einsum_path(
            subscripts, *operands, optimize=kwargs.get("optimize", False)
        )[1]
        match = re.search(r"Naive FLOP count:\s+([0-9.eE+-]+)", report)
        if match is None:
            raise RuntimeError("NumPy einsum path omitted its FLOP estimate")
        trace["einsum_calls"] += 1
        trace["einsum_estimated_flops"] += int(float(match.group(1)))
        result = original_einsum(subscripts, *operands, **kwargs)
        observe(operands, result)
        return result

    def svd(matrix: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        result = original_svd(matrix, *args, **kwargs)
        m, n = matrix.shape
        trace["svd_calls"] += 1
        trace["decomposition_input_elements"] += int(matrix.size)
        rank = min(m, n)
        trace["estimated_dense_decomposition_flops"] += int(4 * m * n * rank + 8 * rank**3)
        if active_truncation is not None:
            singular_values = np.asarray(result[1], dtype=np.float64)
            requested_tol = active_truncation["requested_tol"]
            requested_chi = active_truncation["requested_chi"]
            if singular_values.size and singular_values[0] > 0:
                normalized = singular_values / singular_values[0]
                retained_rank = int(normalized.size)
                if requested_tol is not None:
                    retained_rank = int(np.count_nonzero(normalized > float(requested_tol)))
                if requested_chi is not None:
                    retained_rank = min(retained_rank, int(requested_chi))
                squared = singular_values**2
                total_squared = float(squared.sum())
                discarded_fraction = (
                    0.0
                    if total_squared == 0
                    else float(squared[retained_rank:].sum() / total_squared)
                )
                probability = squared / total_squared
                nonzero = probability[probability > 0]
                spectral_entropy = float(-(nonzero * np.log(nonzero)).sum())
            else:
                retained_rank = 0
                discarded_fraction = 0.0
                spectral_entropy = 0.0
            summaries = active_truncation["spectral_summaries"]
            assert isinstance(summaries, list)
            summaries.append(
                {
                    "matrix_rows": int(m),
                    "matrix_columns": int(n),
                    "singular_value_count": int(singular_values.size),
                    "retained_rank": retained_rank,
                    "requested_chi": requested_chi,
                    "requested_tol": requested_tol,
                    "discarded_squared_weight_fraction": discarded_fraction,
                    "spectral_entropy": spectral_entropy,
                }
            )
        observe(matrix, result)
        return result

    def contract_2d(tn: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active_sweep_index
        if active_sweep_index is not None:
            raise RuntimeError("nested qecsim 2-D contraction trace is unsupported")
        sweeps = trace["contraction_sweeps"]
        truncation_events = trace["truncation_events"]
        assert isinstance(sweeps, list)
        assert isinstance(truncation_events, list)
        sweep_index = len(sweeps)
        active_sweep_index = sweep_index
        start_flops = cumulative_flops()
        start_pairwise = int(trace["pairwise_contractions"])
        start_truncations = len(truncation_events)
        try:
            result = original_mps2d_contract(tn, *args, **kwargs)
        finally:
            active_sweep_index = None
        sweeps.append(
            {
                "index": sweep_index,
                "network_rows": int(tn.shape[0]),
                "network_columns": int(tn.shape[1]),
                "requested_chi": kwargs.get("chi", args[0] if args else None),
                "requested_tol": kwargs.get("tol", args[1] if len(args) > 1 else None),
                "pairwise_contractions": int(trace["pairwise_contractions"])
                - start_pairwise,
                "truncation_event_start": start_truncations,
                "truncation_event_stop": len(truncation_events),
                "estimated_arithmetic_flops": cumulative_flops() - start_flops,
            }
        )
        return result

    def qr(matrix: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        result = original_qr(matrix, *args, **kwargs)
        m, n = matrix.shape
        trace["qr_calls"] += 1
        trace["decomposition_input_elements"] += int(matrix.size)
        rank = min(m, n)
        trace["estimated_dense_decomposition_flops"] += int(
            2 * m * n * rank - (2 * rank**3) / 3
        )
        observe(matrix, result)
        return result

    qecsim_mps.contract_pairwise = pairwise
    qecsim_mps.truncate = truncate
    qecsim_mps.sp_linalg.svd = svd
    qecsim_mps.sp_linalg.qr = qr
    qecsim_mps.np.einsum = einsum
    qecsim_mps2d.contract = contract_2d
    try:
        yield trace
    finally:
        trace["estimated_arithmetic_flops"] = cumulative_flops()
        sweeps = trace["contraction_sweeps"]
        assert isinstance(sweeps, list)
        trace["terminal_residual_estimated_arithmetic_flops"] = cumulative_flops() - sum(
            int(sweep["estimated_arithmetic_flops"]) for sweep in sweeps
        )
        qecsim_mps.contract_pairwise = original_pairwise
        qecsim_mps.truncate = original_truncate
        qecsim_mps.sp_linalg.svd = original_svd
        qecsim_mps.sp_linalg.qr = original_qr
        qecsim_mps.np.einsum = original_einsum
        qecsim_mps2d.contract = original_mps2d_contract


def exact_planar_coset_masses(
    *,
    rows: int,
    columns: int,
    syndrome: np.ndarray,
    error_rate: float,
) -> PlanarCosetMasses:
    """Enumerate all four stabilizer cosets for a small planar code."""
    code, syndrome_array = _validate_inputs(rows, columns, syndrome, error_rate)
    sample = PlanarMPSDecoder.sample_recovery(code, syndrome_array)
    representatives = (
        sample,
        sample.copy().logical_x(),
        sample.copy().logical_x().logical_z(),
        sample.copy().logical_z(),
    )
    stabilizers = np.asarray(code.stabilizers, dtype=np.uint8)
    distribution = DepolarizingErrorModel().probability_distribution(error_rate)
    masses = np.zeros(4, dtype=np.float64)

    for coefficients in product((0, 1), repeat=stabilizers.shape[0]):
        coefficient_array = np.asarray(coefficients, dtype=np.uint8)
        stabilizer = coefficient_array @ stabilizers % 2
        for logical_class, representative in enumerate(representatives):
            error = (representative.to_bsf() + stabilizer) % 2
            masses[logical_class] += _physical_error_probability(error, distribution)

    return PlanarCosetMasses(
        masses=masses,
        selected_class=int(np.argmax(masses)),
        rows=rows,
        columns=columns,
        error_rate=error_rate,
        chi=None,
        tol=None,
        mode="exact_enumeration",
        enumerated_errors=4 * (1 << stabilizers.shape[0]),
    )


def planar_mps_coset_masses(
    *,
    rows: int,
    columns: int,
    syndrome: np.ndarray,
    error_rate: float,
    chi: int | None,
    mode: ContractionMode,
    tol: float | None = None,
    trace_work: bool = False,
) -> PlanarCosetMasses:
    """Contract qecsim's planar-code tensor network and return raw coset masses."""
    code, syndrome_array = _validate_inputs(rows, columns, syndrome, error_rate)
    if mode not in _MODE:
        raise ValueError(f"unknown contraction mode: {mode}")
    if chi is not None and (type(chi) is not int or chi <= 0):
        raise ValueError("chi must be a positive integer or None")
    if tol is not None and (not np.isfinite(tol) or tol <= 0):
        raise ValueError("tol must be a positive finite float or None")

    trace: dict[str, object] | None = None
    trace_context = _trace_mps_work() if trace_work else nullcontext(None)
    started = perf_counter()
    with trace_context as active_trace:
        decoder = PlanarMPSDecoder(chi=chi, mode=_MODE[mode], tol=tol)
        sample = decoder.sample_recovery(code, syndrome_array)
        distribution = DepolarizingErrorModel().probability_distribution(error_rate)
        # qecsim exposes raw coset masses through this implementation method;
        # decode() intentionally returns only the selected recovery.
        raw_masses, _ = decoder._coset_probabilities(distribution, sample)
        if active_trace is not None:
            trace = active_trace
    if trace is not None:
        sweep_labels = {
            "columns": ("columns:I-X", "columns:Z-Y"),
            "rows": ("rows:I-Z", "rows:X-Y"),
            "average": (
                "columns:I-X",
                "columns:Z-Y",
                "rows:I-Z",
                "rows:X-Y",
            ),
        }[mode]
        sweeps = trace["contraction_sweeps"]
        assert isinstance(sweeps, list)
        if len(sweeps) != len(sweep_labels):
            raise RuntimeError("qecsim contraction sweep count changed unexpectedly")
        for sweep, label in zip(sweeps, sweep_labels, strict=True):
            sweep["label"] = label
        trace["single_shot_wall_seconds"] = perf_counter() - started
    masses = np.asarray([float(value) for value in raw_masses], dtype=np.float64)
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0) or float(masses.sum()) <= 0:
        raise InvalidCosetMassError(masses, trace)
    return PlanarCosetMasses(
        masses=masses,
        selected_class=int(np.argmax(masses)),
        rows=rows,
        columns=columns,
        error_rate=error_rate,
        chi=chi,
        tol=tol,
        mode=mode,
        work=trace,
    )
