from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import repeat
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


@dataclass(frozen=True, slots=True)
class BPLSDConfig:
    """Pinned BP-LSD parameters shared by every accuracy-comparison arm."""

    max_iter: int = 100
    bp_method: str = "minimum_sum"
    ms_scaling_factor: float = 0.0
    schedule: str = "serial"
    lsd_method: str = "LSD_E"
    lsd_order: int = 5


_DEFAULT_BPLSD_CONFIG = BPLSDConfig()


def _new_decoder(
    hx: sparse.spmatrix,
    error_channel: np.ndarray,
    config: BPLSDConfig = _DEFAULT_BPLSD_CONFIG,
) -> BpLsdDecoder:
    """Build a BP-LSD decoder with the campaign's pinned configuration."""
    return BpLsdDecoder(
        hx,
        error_channel=error_channel,
        max_iter=config.max_iter,
        bp_method=config.bp_method,
        ms_scaling_factor=config.ms_scaling_factor,
        schedule=config.schedule,
        lsd_method=config.lsd_method,
        lsd_order=config.lsd_order,
    )


def _prepare_inputs(
    hx: sparse.spmatrix,
    syndromes: np.ndarray,
    logical_x: sparse.spmatrix,
) -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix]:
    hx_csr = hx.astype(np.uint8).tocsr()
    logical_csr = logical_x.astype(np.uint8).tocsr()
    syndrome_values = np.asarray(syndromes)
    if syndrome_values.ndim != 2:
        raise ValueError("syndromes must have shape (shots, checks)")
    if syndrome_values.shape[1] != hx_csr.shape[0]:
        raise ValueError(
            f"syndrome width {syndrome_values.shape[1]} does not match Hx rows {hx_csr.shape[0]}"
        )
    if logical_csr.shape[1] != hx_csr.shape[1]:
        raise ValueError("logical X supports and Hx must have equal block length")
    if not np.all((syndrome_values == 0) | (syndrome_values == 1)):
        raise ValueError("syndromes must be binary")
    return hx_csr, syndrome_values.astype(np.uint8), logical_csr


def _decode_rows(
    hx_csr: sparse.csr_matrix,
    syndrome_array: np.ndarray,
    logical_csr: sparse.csr_matrix,
    decoders: Iterable[BpLsdDecoder],
) -> DecodeBatchResult:
    shots = syndrome_array.shape[0]
    corrections = np.zeros((shots, hx_csr.shape[1]), dtype=np.uint8)
    predicted = np.zeros((shots, logical_csr.shape[0]), dtype=np.uint8)
    valid = np.zeros(shots, dtype=np.bool_)
    converged = np.zeros(shots, dtype=np.bool_)
    iterations = np.zeros(shots, dtype=np.int64)
    latency = np.zeros(shots, dtype=np.float64)
    for shot, (syndrome, decoder) in enumerate(zip(syndrome_array, decoders, strict=True)):
        started = perf_counter()
        correction = np.asarray(decoder.decode(syndrome), dtype=np.uint8)
        latency[shot] = perf_counter() - started
        corrections[shot] = correction
        valid[shot] = np.array_equal(np.asarray(hx_csr @ correction).ravel() % 2, syndrome)
        predicted[shot] = np.asarray(logical_csr @ correction).ravel() % 2
        converged[shot] = bool(decoder.converge)
        iterations[shot] = int(decoder.iter)
    return DecodeBatchResult(corrections, predicted, valid, converged, iterations, latency)


def decode_bplsd_batch(
    hx: sparse.spmatrix,
    syndromes: np.ndarray,
    logical_x: sparse.spmatrix,
    *,
    error_rate: float,
    config: BPLSDConfig = _DEFAULT_BPLSD_CONFIG,
) -> DecodeBatchResult:
    """Decode a batch with the pinned BP-LSD teacher configuration."""
    hx_csr, syndrome_array, logical_csr = _prepare_inputs(hx, syndromes, logical_x)
    if not 0.0 < error_rate < 0.5:
        raise ValueError("error_rate must be strictly between 0 and 0.5")

    decoder = _new_decoder(
        hx_csr,
        np.full(hx_csr.shape[1], error_rate, dtype=np.float64),
        config,
    )
    return _decode_rows(
        hx_csr,
        syndrome_array,
        logical_csr,
        repeat(decoder, syndrome_array.shape[0]),
    )


def decode_bplsd_prior_batch(
    hx: sparse.spmatrix,
    syndromes: np.ndarray,
    logical_x: sparse.spmatrix,
    *,
    error_channels: np.ndarray,
    config: BPLSDConfig = _DEFAULT_BPLSD_CONFIG,
) -> DecodeBatchResult:
    """Decode each round with its own precomputed, strictly causal error prior."""
    hx_csr, syndrome_array, logical_csr = _prepare_inputs(hx, syndromes, logical_x)
    channels = np.asarray(error_channels, dtype=np.float64)
    expected_shape = (syndrome_array.shape[0], hx_csr.shape[1])
    if channels.shape != expected_shape:
        raise ValueError(
            f"error_channels must have shape {expected_shape}, got {channels.shape}"
        )
    if not np.all(np.isfinite(channels)):
        raise ValueError("error_channels must contain only finite values")
    if not np.all((channels > 0.0) & (channels < 0.5)):
        raise ValueError("error_channels values must be strictly between 0 and 0.5")

    decoders = (_new_decoder(hx_csr, row.copy(), config) for row in channels)
    return _decode_rows(hx_csr, syndrome_array, logical_csr, decoders)
