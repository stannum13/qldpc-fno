from __future__ import annotations

import stim
from scipy import sparse


def build_z_error_dem(
    hx: sparse.spmatrix,
    logical_x: sparse.spmatrix,
    *,
    error_rate: float,
) -> stim.DetectorErrorModel:
    """Build an exact independent Z-error detector model for one CSS sector."""
    if not 0.0 < error_rate < 0.5:
        raise ValueError("error_rate must be strictly between 0 and 0.5")
    hx_csc = hx.tocsc()
    logical_csc = logical_x.tocsc()
    if hx_csc.shape[1] != logical_csc.shape[1]:
        raise ValueError("Hx and logical X supports must have equal block length")

    model = stim.DetectorErrorModel()
    for qubit in range(hx_csc.shape[1]):
        detector_rows = hx_csc.indices[hx_csc.indptr[qubit] : hx_csc.indptr[qubit + 1]]
        logical_rows = logical_csc.indices[
            logical_csc.indptr[qubit] : logical_csc.indptr[qubit + 1]
        ]
        targets = [stim.target_relative_detector_id(int(row)) for row in detector_rows]
        targets.extend(stim.target_logical_observable_id(int(row)) for row in logical_rows)
        model.append("error", error_rate, targets)
    return model
