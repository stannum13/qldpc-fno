from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from ldpc import BpLsdDecoder
from scipy import sparse


@dataclass(frozen=True, slots=True)
class DecodeBatchResult:
    """Per-shot corrections, logical predictions, validity, and timing."""

    corrections: np.ndarray
    predicted_observables: np.ndarray
    syndrome_valid: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    latency_seconds: np.ndarray


def _new_decoder(hx: sparse.spmatrix, error_channel: np.ndarray) -> BpLsdDecoder:
    """Build a BP-LSD decoder with the campaign's pinned configuration."""
    return BpLsdDecoder(
        hx,
        error_channel=error_channel,
        max_iter=100,
        bp_method="minimum_sum",
        ms_scaling_factor=0.0,
        schedule="serial",
        lsd_method="LSD_E",
        lsd_order=5,
    )


def decode_bplsd_batch(
    hx: sparse.spmatrix,
    syndromes: np.ndarray,
    logical_x: sparse.spmatrix,
    *,
    error_rate: float,
) -> DecodeBatchResult:
    """Decode a batch with the pinned BP-LSD teacher configuration."""
    hx_csr = hx.astype(np.uint8).tocsr()
    logical_csr = logical_x.astype(np.uint8).tocsr()
    syndrome_array = np.asarray(syndromes, dtype=np.uint8)
    if syndrome_array.ndim != 2:
        raise ValueError("syndromes must have shape (shots, checks)")
    if syndrome_array.shape[1] != hx_csr.shape[0]:
        raise ValueError(
            f"syndrome width {syndrome_array.shape[1]} does not match Hx rows {hx_csr.shape[0]}"
        )
    if logical_csr.shape[1] != hx_csr.shape[1]:
        raise ValueError("logical X supports and Hx must have equal block length")
    if not np.all((syndrome_array == 0) | (syndrome_array == 1)):
        raise ValueError("syndromes must be binary")
    if not 0.0 < error_rate < 0.5:
        raise ValueError("error_rate must be strictly between 0 and 0.5")

    decoder = _new_decoder(hx_csr, np.full(hx_csr.shape[1], error_rate, dtype=np.float64))
    shots = syndrome_array.shape[0]
    corrections = np.zeros((shots, hx_csr.shape[1]), dtype=np.uint8)
    predicted = np.zeros((shots, logical_csr.shape[0]), dtype=np.uint8)
    valid = np.zeros(shots, dtype=np.bool_)
    converged = np.zeros(shots, dtype=np.bool_)
    iterations = np.zeros(shots, dtype=np.int64)
    latency = np.zeros(shots, dtype=np.float64)
    for shot, syndrome in enumerate(syndrome_array):
        started = perf_counter()
        correction = np.asarray(decoder.decode(syndrome), dtype=np.uint8)
        latency[shot] = perf_counter() - started
        corrections[shot] = correction
        valid[shot] = np.array_equal(np.asarray(hx_csr @ correction).ravel() % 2, syndrome)
        predicted[shot] = np.asarray(logical_csr @ correction).ravel() % 2
        converged[shot] = bool(decoder.converge)
        iterations[shot] = int(decoder.iter)
    return DecodeBatchResult(corrections, predicted, valid, converged, iterations, latency)
