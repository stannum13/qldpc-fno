from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse

from qldpc_fno.decoders.bplsd import _new_decoder


@dataclass(frozen=True, slots=True)
class HybridDecodeResult:
    """Per-shot hybrid corrections, decoder diagnostics, weights, and timing."""

    corrections: np.ndarray
    predicted_observables: np.ndarray
    syndrome_valid: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    correction_weight: np.ndarray
    proposal_weight: np.ndarray
    residual_before: np.ndarray
    residual_syndrome_weight: np.ndarray
    delta_weight: np.ndarray
    preprocessing_latency_seconds: np.ndarray
    bp_latency_seconds: np.ndarray
    latency_seconds: np.ndarray

    @property
    def proposal_latency_seconds(self) -> np.ndarray:
        """Alias for time spent constructing the residual decoding problem."""
        return self.preprocessing_latency_seconds

    @property
    def repair_latency_seconds(self) -> np.ndarray:
        """Alias for BP-LSD repair time."""
        return self.bp_latency_seconds

    @property
    def decode_latency_seconds(self) -> np.ndarray:
        """Alias for time spent inside BP-LSD decode."""
        return self.bp_latency_seconds

    @property
    def residual_weight(self) -> np.ndarray:
        """Alias for the pre-repair residual syndrome weight."""
        return self.residual_syndrome_weight


def _prepare_inputs(
    hx: sparse.spmatrix,
    syndromes: np.ndarray,
    logical_x: sparse.spmatrix,
    probabilities: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray]:
    hx_csr = hx.astype(np.uint8).tocsr()
    logical_csr = logical_x.astype(np.uint8).tocsr()
    raw_syndromes = np.asarray(syndromes)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if raw_syndromes.ndim != 2:
        raise ValueError("syndromes must have shape (shots, checks)")
    if not np.all((raw_syndromes == 0) | (raw_syndromes == 1)):
        raise ValueError("syndromes must be binary")
    syndrome_array = raw_syndromes.astype(np.uint8)
    if syndrome_array.shape[1] != hx_csr.shape[0]:
        raise ValueError(
            f"syndrome width {syndrome_array.shape[1]} does not match Hx rows {hx_csr.shape[0]}"
        )
    if logical_csr.shape[1] != hx_csr.shape[1]:
        raise ValueError("logical X supports and Hx must have equal block length")
    expected_probability_shape = (syndrome_array.shape[0], hx_csr.shape[1])
    if probability_array.shape != expected_probability_shape:
        raise ValueError(
            f"probabilities must have shape {expected_probability_shape}, "
            f"got {probability_array.shape}"
        )
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probabilities must be finite")
    if not np.all((probability_array >= 0.0) & (probability_array <= 1.0)):
        raise ValueError("probabilities must lie between 0 and 1")
    return hx_csr, syndrome_array, logical_csr, probability_array


def _empty_result(shots: int, checks: int, block_length: int, observables: int) -> dict:
    return {
        "corrections": np.zeros((shots, block_length), dtype=np.uint8),
        "predicted_observables": np.zeros((shots, observables), dtype=np.uint8),
        "syndrome_valid": np.zeros(shots, dtype=np.bool_),
        "converged": np.zeros(shots, dtype=np.bool_),
        "iterations": np.zeros(shots, dtype=np.int64),
        "correction_weight": np.zeros(shots, dtype=np.int64),
        "proposal_weight": np.zeros(shots, dtype=np.int64),
        "residual_before": np.zeros((shots, checks), dtype=np.uint8),
        "residual_syndrome_weight": np.zeros(shots, dtype=np.int64),
        "delta_weight": np.zeros(shots, dtype=np.int64),
        "preprocessing_latency_seconds": np.zeros(shots, dtype=np.float64),
        "bp_latency_seconds": np.zeros(shots, dtype=np.float64),
        "latency_seconds": np.zeros(shots, dtype=np.float64),
    }


def decode_soft_prior_batch(
    hx: sparse.spmatrix,
    syndromes: np.ndarray,
    logical_x: sparse.spmatrix,
    probabilities: np.ndarray,
) -> HybridDecodeResult:
    """Decode each syndrome using its corresponding per-qubit soft prior."""
    hx_csr, syndrome_array, logical_csr, probability_array = _prepare_inputs(
        hx, syndromes, logical_x, probabilities
    )
    result = _empty_result(
        syndrome_array.shape[0], hx_csr.shape[0], hx_csr.shape[1], logical_csr.shape[0]
    )
    decoder = _new_decoder(hx_csr, np.full(hx_csr.shape[1], 0.1, dtype=np.float64))
    for shot, syndrome in enumerate(syndrome_array):
        started = perf_counter()
        decoder.update_channel_probs(probability_array[shot])
        bp_started = perf_counter()
        correction = np.asarray(decoder.decode(syndrome), dtype=np.uint8)
        finished = perf_counter()
        final_syndrome = np.asarray(hx_csr @ correction).ravel() % 2
        result["corrections"][shot] = correction
        result["predicted_observables"][shot] = np.asarray(
            logical_csr @ correction
        ).ravel() % 2
        result["syndrome_valid"][shot] = np.array_equal(final_syndrome, syndrome)
        result["converged"][shot] = bool(decoder.converge)
        result["iterations"][shot] = int(decoder.iter)
        result["correction_weight"][shot] = int(correction.sum())
        result["preprocessing_latency_seconds"][shot] = bp_started - started
        result["bp_latency_seconds"][shot] = finished - bp_started
        result["latency_seconds"][shot] = (
            result["preprocessing_latency_seconds"][shot]
            + result["bp_latency_seconds"][shot]
        )
    return HybridDecodeResult(**result)


def decode_residual_batch(
    hx: sparse.spmatrix,
    syndromes: np.ndarray,
    logical_x: sparse.spmatrix,
    probabilities: np.ndarray,
) -> HybridDecodeResult:
    """Repair thresholded proposals by decoding their residual syndromes."""
    hx_csr, syndrome_array, logical_csr, probability_array = _prepare_inputs(
        hx, syndromes, logical_x, probabilities
    )
    result = _empty_result(
        syndrome_array.shape[0], hx_csr.shape[0], hx_csr.shape[1], logical_csr.shape[0]
    )
    decoder = _new_decoder(hx_csr, np.full(hx_csr.shape[1], 0.1, dtype=np.float64))
    for shot, syndrome in enumerate(syndrome_array):
        started = perf_counter()
        proposal = probability_array[shot] >= 0.5
        proposal_syndrome = np.asarray(hx_csr @ proposal, dtype=np.uint8).ravel() % 2
        residual = syndrome ^ proposal_syndrome
        uncertainty = np.minimum(probability_array[shot], 1.0 - probability_array[shot])
        decoder.update_channel_probs(np.clip(uncertainty, 1e-5, 1.0 - 1e-5))
        bp_started = perf_counter()
        delta = np.asarray(decoder.decode(residual), dtype=np.uint8)
        finished = perf_counter()
        correction = np.asarray(proposal, dtype=np.uint8) ^ delta
        final_syndrome = np.asarray(hx_csr @ correction).ravel() % 2
        result["corrections"][shot] = correction
        result["predicted_observables"][shot] = np.asarray(
            logical_csr @ correction
        ).ravel() % 2
        result["syndrome_valid"][shot] = np.array_equal(final_syndrome, syndrome)
        result["converged"][shot] = bool(decoder.converge)
        result["iterations"][shot] = int(decoder.iter)
        result["correction_weight"][shot] = int(correction.sum())
        result["proposal_weight"][shot] = int(proposal.sum())
        result["residual_before"][shot] = residual
        result["residual_syndrome_weight"][shot] = int(residual.sum())
        result["delta_weight"][shot] = int(delta.sum())
        result["preprocessing_latency_seconds"][shot] = bp_started - started
        result["bp_latency_seconds"][shot] = finished - bp_started
        result["latency_seconds"][shot] = (
            result["preprocessing_latency_seconds"][shot]
            + result["bp_latency_seconds"][shot]
        )
    return HybridDecodeResult(**result)
