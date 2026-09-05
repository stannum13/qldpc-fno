"""Typed, bounded-state per-sequence endpoints for the identifiability study."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from qldpc_fno.identifiability.filters import ForecastResult
from qldpc_fno.identifiability.observation import DisjointChecks
from qldpc_fno.identifiability.types import (
    ContemporaneousOracleInput,
    DeployableHistory,
    LatentHistoryOracleInput,
    SequenceIdentity,
)

_LOWER = 1e-5
_UPPER = 0.25
_BASE_PROBABILITY = 0.0375
_STATE_CLIP = 1.2
_NMSE_DENOMINATOR = 0.08**2 / (1.0 - 0.97**2)
_BIN_EDGES = np.linspace(_LOWER, _UPPER, 11, dtype=np.float64)


def _readonly(value: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_scored_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 1 or value.dtype != np.bool_ or not np.any(value):
        raise ValueError("scored mask must be a nonempty boolean round vector")
    first = int(np.argmax(value))
    if np.any(value[:first]) or not np.all(value[first:]):
        raise ValueError("scored mask must select the complete contiguous scored suffix")
    if value.size == 192:
        canonical = np.arange(value.size) >= 64
        if not np.array_equal(value, canonical):
            raise ValueError("scored mask must select the canonical 64/128 round split")
    return value


@dataclass(frozen=True, slots=True)
class EndpointSequence:
    """One content-bound sequence and its exact privileged scoring inputs."""

    identity: SequenceIdentity
    observed: DeployableHistory
    latent_state: LatentHistoryOracleInput
    latent_probabilities: ContemporaneousOracleInput
    forecast: ForecastResult

    def __post_init__(self) -> None:
        required = (
            (self.identity, SequenceIdentity, "SequenceIdentity"),
            (self.observed, DeployableHistory, "DeployableHistory"),
            (self.latent_state, LatentHistoryOracleInput, "LatentHistoryOracleInput"),
            (
                self.latent_probabilities,
                ContemporaneousOracleInput,
                "ContemporaneousOracleInput",
            ),
            (self.forecast, ForecastResult, "ForecastResult"),
        )
        for value, expected, label in required:
            if type(value) is not expected:
                raise TypeError(f"endpoint sequences require the exact {label} type")
        if self.identity.content_sha256 is None:
            raise ValueError("endpoint sequence identity must be content-bound")

        rounds, qubits = self.latent_probabilities.probabilities.shape
        probabilities = self.forecast.probabilities
        if (
            self.observed.syndromes.shape[0] != rounds
            or self.observed.scored_mask.shape != (rounds,)
            or self.latent_state.global_log_odds.shape != (rounds,)
            or probabilities.shape[0] != rounds
        ):
            raise ValueError("endpoint sequence containers must have matching round geometry")
        _validate_scored_mask(self.observed.scored_mask)
        if probabilities.ndim == 2 and probabilities.shape[1] != qubits:
            raise ValueError("vector forecasts must match the latent physical-field geometry")
        if np.any(
            (self.latent_probabilities.probabilities < _LOWER)
            | (self.latent_probabilities.probabilities > _UPPER)
        ):
            raise ValueError("latent probabilities must lie in the fixed [1e-5, 0.25] range")
        if np.any((probabilities < _LOWER) | (probabilities > _UPPER)):
            raise ValueError("forecast probabilities must lie in the fixed [1e-5, 0.25] range")


@dataclass(frozen=True, slots=True)
class EndpointBatch:
    """An immutable, homogeneous batch of content-disjoint endpoint sequences."""

    sequences: tuple[EndpointSequence, ...]

    def __post_init__(self) -> None:
        if type(self.sequences) is not tuple or not self.sequences:
            raise TypeError("EndpointBatch requires a nonempty immutable tuple")
        if any(type(sequence) is not EndpointSequence for sequence in self.sequences):
            raise TypeError("EndpointBatch requires exact EndpointSequence values")

        identities = tuple(sequence.identity for sequence in self.sequences)
        content_hashes = tuple(identity.content_sha256 for identity in identities)
        if len(set(content_hashes)) != len(content_hashes):
            raise ValueError("endpoint identities must be content-disjoint")
        identity_keys = tuple(
            (identity.regime, identity.role, identity.sequence_index) for identity in identities
        )
        if len(set(identity_keys)) != len(identity_keys):
            raise ValueError("endpoint sequence identities must be unique")
        if len({identity.role for identity in identities}) != 1:
            raise ValueError("endpoint batches must not mix sequence roles")
        if len({identity.regime for identity in identities}) != 1:
            raise ValueError("endpoint batches must not mix sequence regimes")
        if len({sequence.forecast.arm for sequence in self.sequences}) != 1:
            raise ValueError("endpoint batches must not mix forecast arms")

        geometry = {
            (
                sequence.latent_probabilities.probabilities.shape,
                sequence.observed.syndromes.shape,
            )
            for sequence in self.sequences
        }
        if len(geometry) != 1:
            raise ValueError("endpoint batch sequence geometry must be homogeneous")


def _require_batch(batch: object) -> EndpointBatch:
    if type(batch) is not EndpointBatch:
        raise TypeError("endpoint scorers require the exact EndpointBatch type")
    # Revalidate exact members in case a frozen dataclass was forged through a low-level API.
    EndpointBatch(batch.sequences)
    return batch


def _forecast_field(sequence: EndpointSequence) -> np.ndarray:
    predicted = sequence.forecast.probabilities
    latent_shape = sequence.latent_probabilities.probabilities.shape
    if predicted.ndim == 1:
        return np.broadcast_to(predicted[:, None], latent_shape)
    return predicted


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    """Per-sequence fixed-bin calibration sufficient statistics."""

    counts: np.ndarray
    predicted_sums: np.ndarray
    latent_sums: np.ndarray
    absolute_error: np.ndarray
    bin_edges: np.ndarray

    def __post_init__(self) -> None:
        count = np.asarray(self.counts, dtype=np.int64)
        predicted = np.asarray(self.predicted_sums, dtype=np.float64)
        latent = np.asarray(self.latent_sums, dtype=np.float64)
        error = np.asarray(self.absolute_error, dtype=np.float64)
        edges = np.asarray(self.bin_edges, dtype=np.float64)
        if (
            count.ndim != 2
            or predicted.shape != count.shape
            or latent.shape != count.shape
            or error.shape != (count.shape[0],)
            or edges.shape != (count.shape[1] + 1,)
        ):
            raise ValueError("calibration evidence has inconsistent per-sequence geometry")
        if (
            np.any(count < 0)
            or not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(latent))
            or not np.all(np.isfinite(error))
            or np.any(error < 0.0)
        ):
            raise ValueError("calibration evidence must be finite and non-negative")
        object.__setattr__(self, "counts", _readonly(count, dtype=np.int64))
        object.__setattr__(self, "predicted_sums", _readonly(predicted, dtype=np.float64))
        object.__setattr__(self, "latent_sums", _readonly(latent, dtype=np.float64))
        object.__setattr__(self, "absolute_error", _readonly(error, dtype=np.float64))
        object.__setattr__(self, "bin_edges", _readonly(edges, dtype=np.float64))


@dataclass(slots=True)
class SequenceEndpointAccumulator:
    """Fixed-size CE and calibration sufficient state for one typed sequence."""

    ce_sum: float = 0.0
    count: int = 0
    calibration_counts: np.ndarray | None = None
    calibration_prediction_sums: np.ndarray | None = None
    calibration_latent_sums: np.ndarray | None = None
    _content_sha256: str | None = None

    def update(self, sequence: EndpointSequence) -> None:
        if type(sequence) is not EndpointSequence:
            raise TypeError("endpoint accumulator updates require the exact EndpointSequence type")
        # Re-run construction to reject forged exact instances before reading arrays.
        EndpointSequence(
            sequence.identity,
            sequence.observed,
            sequence.latent_state,
            sequence.latent_probabilities,
            sequence.forecast,
        )
        if self._content_sha256 is not None:
            raise ValueError("endpoint accumulator accepts exactly one complete sequence")

        mask = sequence.observed.scored_mask
        latent = sequence.latent_probabilities.probabilities[mask].reshape(-1)
        predicted = _forecast_field(sequence)[mask].reshape(-1)
        self.ce_sum = float(
            -np.sum(latent * np.log(predicted) + (1.0 - latent) * np.log1p(-predicted))
        )
        self.count = int(latent.size)
        bins = np.minimum(np.searchsorted(_BIN_EDGES, predicted, side="right") - 1, 9)
        self.calibration_counts = np.bincount(bins, minlength=10).astype(np.int64)
        self.calibration_prediction_sums = np.bincount(
            bins, weights=predicted, minlength=10
        )
        self.calibration_latent_sums = np.bincount(bins, weights=latent, minlength=10)
        self._content_sha256 = sequence.identity.content_sha256

    @property
    def expected_ce(self) -> float:
        if self.count == 0:
            raise ValueError("endpoint accumulator has no scored values")
        return self.ce_sum / self.count


def expected_ce_by_sequence(batch: EndpointBatch) -> np.ndarray:
    """Expected Bernoulli CE against typed latent probabilities, per sequence."""
    checked = _require_batch(batch)
    results = np.empty(len(checked.sequences), dtype=np.float64)
    for index, sequence in enumerate(checked.sequences):
        accumulator = SequenceEndpointAccumulator()
        accumulator.update(sequence)
        results[index] = accumulator.expected_ce
    return results


def latent_nmse_by_sequence(batch: EndpointBatch) -> np.ndarray:
    """State NMSE from the prescribed mean-then-clip log-odds mapping."""
    checked = _require_batch(batch)
    base_logit = math.log(_BASE_PROBABILITY / (1.0 - _BASE_PROBABILITY))
    results = np.empty(len(checked.sequences), dtype=np.float64)
    for index, sequence in enumerate(checked.sequences):
        mask = sequence.observed.scored_mask
        predicted = _forecast_field(sequence)
        mapped = np.clip(
            (np.log(predicted / (1.0 - predicted)) - base_logit).mean(axis=1),
            -_STATE_CLIP,
            _STATE_CLIP,
        )
        state = sequence.latent_state.global_log_odds
        results[index] = float(np.mean((state[mask] - mapped[mask]) ** 2) / _NMSE_DENOMINATOR)
    return results


def _supports(value: Iterable[Iterable[int]], qubits: int) -> tuple[np.ndarray, ...]:
    supports = tuple(np.asarray(tuple(item), dtype=np.int64) for item in value)
    if not supports:
        raise ValueError("retained syndrome endpoint requires at least one support")
    used: set[int] = set()
    for support in supports:
        if (
            support.ndim != 1
            or support.size == 0
            or np.any(support < 0)
            or np.any(support >= qubits)
        ):
            raise ValueError("retained syndrome supports must be non-empty in-bounds vectors")
        if np.unique(support).size != support.size:
            raise ValueError("retained syndrome supports may not repeat a qubit")
        if used.intersection(int(item) for item in support):
            raise ValueError("retained syndrome supports must be pairwise disjoint")
        used.update(int(item) for item in support)
    return supports


def retained_syndrome_nll_by_sequence(
    batch: EndpointBatch, checks: DisjointChecks
) -> np.ndarray:
    """Mean predictive NLL for the exact independent retained syndrome rows."""
    checked = _require_batch(batch)
    if type(checks) is not DisjointChecks:
        raise TypeError("retained NLL requires the exact DisjointChecks type")
    if not checks.is_pairwise_disjoint:
        raise ValueError("retained NLL requires pairwise-disjoint retained checks")
    if (
        len(checks.row_indices) == 0
        or np.any(checks.row_indices < 0)
        or np.any(np.diff(checks.row_indices) <= 0)
    ):
        raise ValueError("retained check row indices must be nonempty and strictly ascending")

    results = np.empty(len(checked.sequences), dtype=np.float64)
    for sequence_index, sequence in enumerate(checked.sequences):
        if np.any(checks.row_indices >= sequence.observed.syndromes.shape[1]):
            raise ValueError("retained check row indices are outside syndrome geometry")
        predicted = _forecast_field(sequence)
        rows = _supports(checks.supports, predicted.shape[1])
        if len(rows) != len(checks.row_indices):
            raise ValueError("retained syndrome rows and supports must agree")
        mask = sequence.observed.scored_mask
        observed = sequence.observed.syndromes[mask][:, checks.row_indices]
        total = 0.0
        for row_index, support in enumerate(rows):
            field = predicted[mask][:, support]
            parity = (1.0 - np.prod(1.0 - 2.0 * field, axis=1)) / 2.0
            parity = np.clip(parity, 1e-12, 1.0 - 1e-12)
            syndrome = observed[:, row_index]
            total += float(
                np.sum(
                    -(syndrome * np.log(parity) + (1.0 - syndrome) * np.log1p(-parity))
                )
            )
        results[sequence_index] = total / (int(np.count_nonzero(mask)) * len(rows))
    return results


def calibration_by_sequence(batch: EndpointBatch) -> CalibrationEvidence:
    """Aggregate fixed ten-bin latent calibration evidence by typed sequence."""
    checked = _require_batch(batch)
    sequence_count = len(checked.sequences)
    counts = np.zeros((sequence_count, 10), dtype=np.int64)
    predicted_sums = np.zeros((sequence_count, 10), dtype=np.float64)
    latent_sums = np.zeros((sequence_count, 10), dtype=np.float64)
    for index, sequence in enumerate(checked.sequences):
        accumulator = SequenceEndpointAccumulator()
        accumulator.update(sequence)
        assert accumulator.calibration_counts is not None
        assert accumulator.calibration_prediction_sums is not None
        assert accumulator.calibration_latent_sums is not None
        counts[index] = accumulator.calibration_counts
        predicted_sums[index] = accumulator.calibration_prediction_sums
        latent_sums[index] = accumulator.calibration_latent_sums
    total = counts.sum(axis=1)
    absolute_error = np.divide(
        np.abs(predicted_sums - latent_sums).sum(axis=1),
        total,
        out=np.zeros(sequence_count, dtype=np.float64),
        where=total != 0,
    )
    return CalibrationEvidence(counts, predicted_sums, latent_sums, absolute_error, _BIN_EDGES)


__all__ = [
    "CalibrationEvidence",
    "EndpointBatch",
    "EndpointSequence",
    "SequenceEndpointAccumulator",
    "calibration_by_sequence",
    "expected_ce_by_sequence",
    "latent_nmse_by_sequence",
    "retained_syndrome_nll_by_sequence",
]
