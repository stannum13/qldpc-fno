"""Streaming-safe per-sequence endpoints for the identifiability study."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from qldpc_fno.identifiability.observation import DisjointChecks
from qldpc_fno.identifiability.types import SequenceIdentity, TrainingTargets

_LOWER = 1e-5
_UPPER = 0.25
_BASE_PROBABILITY = 0.0375
_STATE_CLIP = 1.2
_NMSE_DENOMINATOR = 0.08**2 / (1.0 - 0.97**2)
_BIN_EDGES = np.linspace(_LOWER, _UPPER, 11, dtype=np.float64)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


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
        if np.any(count < 0) or not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(latent)):
            raise ValueError("calibration evidence must be finite and non-negative")
        object.__setattr__(self, "counts", _readonly(count))
        object.__setattr__(self, "predicted_sums", _readonly(predicted))
        object.__setattr__(self, "latent_sums", _readonly(latent))
        object.__setattr__(self, "absolute_error", _readonly(error))
        object.__setattr__(self, "bin_edges", _readonly(edges))


@dataclass(slots=True)
class SequenceEndpointAccumulator:
    """Bounded-memory sufficient statistics for one sequence prediction stream."""

    ce_sum: float = 0.0
    count: int = 0
    calibration_counts: np.ndarray | None = None
    calibration_prediction_sums: np.ndarray | None = None
    calibration_latent_sums: np.ndarray | None = None

    def update(self, q: np.ndarray, q_hat: np.ndarray) -> None:
        latent = np.asarray(q, dtype=np.float64).reshape(-1)
        predicted = np.asarray(q_hat, dtype=np.float64).reshape(-1)
        if latent.shape != predicted.shape or np.any((predicted < _LOWER) | (predicted > _UPPER)):
            raise ValueError("endpoint chunks require matching latent and bounded forecast values")
        self.ce_sum += float(-np.sum(latent * np.log(predicted) + (1 - latent) * np.log1p(-predicted)))
        self.count += int(latent.size)
        bins = np.minimum(np.searchsorted(_BIN_EDGES, predicted, side="right") - 1, 9)
        if self.calibration_counts is None:
            self.calibration_counts = np.zeros(10, dtype=np.int64)
            self.calibration_prediction_sums = np.zeros(10)
            self.calibration_latent_sums = np.zeros(10)
        self.calibration_counts += np.bincount(bins, minlength=10)
        self.calibration_prediction_sums += np.bincount(bins, weights=predicted, minlength=10)
        self.calibration_latent_sums += np.bincount(bins, weights=latent, minlength=10)

    @property
    def expected_ce(self) -> float:
        if self.count == 0:
            raise ValueError("endpoint accumulator has no scored values")
        return self.ce_sum / self.count


def _validate_identities(
    identities: tuple[SequenceIdentity, ...] | None, sequence_count: int
) -> None:
    if identities is None:
        raise TypeError("endpoint scoring requires content-bound sequence identities")
    if type(identities) is not tuple or len(identities) != sequence_count:
        raise ValueError("sequence identities must match the endpoint sequence count")
    if any(type(identity) is not SequenceIdentity for identity in identities):
        raise TypeError("endpoint identities must be exact SequenceIdentity values")
    if len({identity.content_sha256 for identity in identities}) != len(identities):
        raise ValueError("endpoint sequence identities must be content-disjoint")
    if len({identity.role for identity in identities}) != 1 or len({identity.regime for identity in identities}) != 1:
        raise ValueError("endpoint inputs must not mix sequence roles or regimes")


def _mask(mask: np.ndarray, sequence_count: int, rounds: int) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim == 1:
        value = np.broadcast_to(value, (sequence_count, value.shape[0]))
    if value.shape != (sequence_count, rounds) or value.dtype != np.bool_:
        raise ValueError("mask must be a boolean array with one entry per sequence round")
    if not np.all(value.any(axis=1)):
        raise ValueError("mask must include at least one scored round per sequence")
    # The generator's scored rounds are a common contiguous suffix.  This also
    # rejects incomplete masks that would silently change the estimand.
    for row in value:
        first = int(np.argmax(row))
        if not np.all(row[first:]):
            raise ValueError("mask must select the complete contiguous scored suffix")
    if rounds == 192:
        canonical = np.arange(rounds) >= 64
        if not np.array_equal(value, np.broadcast_to(canonical, value.shape)):
            raise ValueError("mask omits or adds canonical scored rounds")
    return value


def _probability_cube(
    q: np.ndarray,
    q_hat: np.ndarray,
    mask: np.ndarray,
    *,
    identities: tuple[SequenceIdentity, ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(q) is TrainingTargets or type(q_hat) is TrainingTargets:
        raise TypeError("endpoint scoring rejects sampled physical-error targets")
    latent = np.asarray(q, dtype=np.float64)
    if latent.ndim == 2:
        latent = latent[None, ...]
    if latent.ndim != 3 or not np.all(np.isfinite(latent)) or np.any((latent < 0.0) | (latent > 1.0)):
        raise ValueError("latent probabilities must be finite (sequences, rounds, qubits) values in [0, 1]")
    predicted = np.asarray(getattr(q_hat, "q_hat", q_hat), dtype=np.float64)
    if predicted.ndim == 1:
        predicted = predicted[None, :, None]
    elif predicted.ndim == 2:
        if predicted.shape == latent.shape[1:]:
            predicted = predicted[None, ...]
        elif predicted.shape == latent.shape[:2]:
            predicted = predicted[:, :, None]
    try:
        predicted = np.broadcast_to(predicted, latent.shape)
    except ValueError as exc:
        raise ValueError("forecast and latent probability geometry must agree") from exc
    if not np.all(np.isfinite(predicted)) or np.any((predicted < _LOWER) | (predicted > _UPPER)):
        raise ValueError("forecast probabilities must lie in the fixed [1e-5, 0.25] range")
    selected = _mask(mask, latent.shape[0], latent.shape[1])
    _validate_identities(identities, latent.shape[0])
    return latent, predicted, selected


def expected_ce_by_sequence(
    q: np.ndarray,
    q_hat: np.ndarray,
    mask: np.ndarray,
    *,
    identities: tuple[SequenceIdentity, ...],
) -> np.ndarray:
    """Expected Bernoulli CE against latent probabilities, one value per sequence."""
    latent, predicted, selected = _probability_cube(q, q_hat, mask, identities=identities)
    results = np.empty(latent.shape[0], dtype=np.float64)
    for index in range(latent.shape[0]):
        actual = latent[index, selected[index]]
        forecast = predicted[index, selected[index]]
        results[index] = -np.mean(actual * np.log(forecast) + (1.0 - actual) * np.log1p(-forecast))
    return results


def latent_nmse_by_sequence(
    latent_state: np.ndarray,
    q_hat: np.ndarray,
    mask: np.ndarray,
    *,
    base_probability: float = _BASE_PROBABILITY,
    state_clip: float = _STATE_CLIP,
    denominator: float = _NMSE_DENOMINATOR,
    identities: tuple[SequenceIdentity, ...],
) -> np.ndarray:
    """State NMSE from the prescribed field-to-log-odds mapping."""
    state = np.asarray(latent_state, dtype=np.float64)
    if state.ndim == 1:
        state = state[None, :]
    if state.ndim != 2 or not np.all(np.isfinite(state)):
        raise ValueError("latent state must be a finite (sequences, rounds) array")
    if not 0.0 < base_probability < 0.5 or state_clip <= 0.0 or denominator <= 0.0:
        raise ValueError("NMSE mapping constants must be positive and finite")
    predicted = np.asarray(getattr(q_hat, "q_hat", q_hat), dtype=np.float64)
    if predicted.ndim == 1:
        predicted = predicted[None, :, None]
    elif predicted.ndim == 2:
        if predicted.shape == state.shape:
            predicted = predicted[:, :, None]
        elif predicted.shape[0] == state.shape[1]:
            predicted = predicted[None, ...]
    if predicted.ndim != 3 or predicted.shape[:2] != state.shape:
        raise ValueError("latent state and forecast rounds must agree")
    if not np.all(np.isfinite(predicted)) or np.any((predicted < _LOWER) | (predicted > _UPPER)):
        raise ValueError("forecast probabilities must lie in the fixed [1e-5, 0.25] range")
    selected = _mask(mask, state.shape[0], state.shape[1])
    _validate_identities(identities, state.shape[0])
    base_logit = math.log(base_probability / (1.0 - base_probability))
    mapped = np.clip(
        (np.log(predicted / (1.0 - predicted)) - base_logit).mean(axis=2),
        -state_clip,
        state_clip,
    )
    return np.asarray(
        [np.mean((state[index, selected[index]] - mapped[index, selected[index]]) ** 2) / denominator for index in range(state.shape[0])],
        dtype=np.float64,
    )


def _supports(value: Iterable[Iterable[int]], qubits: int) -> tuple[np.ndarray, ...]:
    supports = tuple(np.asarray(tuple(item), dtype=np.int64) for item in value)
    if not supports:
        raise ValueError("retained syndrome endpoint requires at least one support")
    used: set[int] = set()
    for support in supports:
        if support.ndim != 1 or support.size == 0 or np.any(support < 0) or np.any(support >= qubits):
            raise ValueError("retained syndrome supports must be non-empty in-bounds index vectors")
        if np.unique(support).size != support.size:
            raise ValueError("retained syndrome supports may not repeat a qubit")
        if used.intersection(int(index) for index in support):
            raise ValueError("retained syndrome supports must be pairwise disjoint")
        used.update(int(index) for index in support)
    return supports


def retained_syndrome_nll_by_sequence(
    retained_syndromes: np.ndarray,
    q_hat: np.ndarray,
    mask: np.ndarray,
    supports: DisjointChecks,
    *,
    identities: tuple[SequenceIdentity, ...],
) -> np.ndarray:
    """Mean predictive NLL for the independent retained syndrome rows."""
    if type(supports) is not DisjointChecks or not supports.is_pairwise_disjoint:
        raise TypeError("retained NLL requires exact canonical pairwise-disjoint DisjointChecks")
    observed = np.asarray(retained_syndromes)
    if observed.ndim == 2:
        observed = observed[None, ...]
    if observed.ndim != 3 or not np.all((observed == 0) | (observed == 1)):
        raise ValueError("retained syndromes must be binary (sequences, rounds, checks)")
    predicted = np.asarray(getattr(q_hat, "q_hat", q_hat), dtype=np.float64)
    if predicted.ndim == 1:
        predicted = predicted[None, :, None]
    elif predicted.ndim == 2:
        if predicted.shape == observed.shape[:2]:
            predicted = predicted[:, :, None]
        else:
            predicted = predicted[None, ...]
    if predicted.ndim != 3 or predicted.shape[:2] != observed.shape[:2]:
        raise ValueError("forecast and retained syndrome sequence-round geometry must agree")
    if not np.all(np.isfinite(predicted)) or np.any((predicted < _LOWER) | (predicted > _UPPER)):
        raise ValueError("forecast probabilities must lie in the fixed [1e-5, 0.25] range")
    selected = _mask(mask, observed.shape[0], observed.shape[1])
    _validate_identities(identities, observed.shape[0])
    if np.any(supports.row_indices < 0) or np.any(supports.row_indices >= observed.shape[2]):
        raise ValueError("retained check row indices are outside syndrome geometry")
    observed = observed[:, :, supports.row_indices]
    support_values = tuple(tuple(item) for item in supports.supports)
    required_qubits = max((max(item) for item in support_values if item), default=-1) + 1
    if predicted.shape[2] == 1 and required_qubits > 1:
        predicted = np.broadcast_to(predicted, (*predicted.shape[:2], required_qubits))
    rows = _supports(support_values, predicted.shape[2])
    if len(rows) != observed.shape[2]:
        raise ValueError("retained syndrome rows and supports must agree")
    results = np.empty(observed.shape[0], dtype=np.float64)
    for sequence_index in range(observed.shape[0]):
        rounds = selected[sequence_index]
        total = 0.0
        for row_index, support in enumerate(rows):
            field = predicted[sequence_index, rounds][:, support]
            parity = (1.0 - np.prod(1.0 - 2.0 * field, axis=1)) / 2.0
            parity = np.clip(parity, 1e-12, 1.0 - 1e-12)
            syndrome = observed[sequence_index, rounds, row_index]
            total += float(np.sum(-(syndrome * np.log(parity) + (1 - syndrome) * np.log1p(-parity))))
        results[sequence_index] = total / (int(np.count_nonzero(rounds)) * len(rows))
    return results


def calibration_by_sequence(
    q: np.ndarray,
    q_hat: np.ndarray,
    mask: np.ndarray,
    *,
    identities: tuple[SequenceIdentity, ...],
) -> CalibrationEvidence:
    """Aggregate the fixed ten-bin latent calibration diagnostic by sequence."""
    latent, predicted, selected = _probability_cube(q, q_hat, mask, identities=identities)
    count = np.zeros((latent.shape[0], 10), dtype=np.int64)
    predicted_sums = np.zeros((latent.shape[0], 10), dtype=np.float64)
    latent_sums = np.zeros((latent.shape[0], 10), dtype=np.float64)
    for index in range(latent.shape[0]):
        values = predicted[index, selected[index]].reshape(-1)
        targets = latent[index, selected[index]].reshape(-1)
        # searchsorted(..., right) makes each internal edge left-closed.  The
        # final clip supplies the required right-closed 0.25 bin.
        bins = np.minimum(np.searchsorted(_BIN_EDGES, values, side="right") - 1, 9)
        count[index] = np.bincount(bins, minlength=10)
        predicted_sums[index] = np.bincount(bins, weights=values, minlength=10)
        latent_sums[index] = np.bincount(bins, weights=targets, minlength=10)
    total = count.sum(axis=1)
    absolute_error = np.divide(
        np.abs(predicted_sums - latent_sums).sum(axis=1), total, out=np.zeros_like(total, dtype=np.float64), where=total != 0
    )
    return CalibrationEvidence(count, predicted_sums, latent_sums, absolute_error, _BIN_EDGES)


__all__ = [
    "CalibrationEvidence",
    "SequenceEndpointAccumulator",
    "calibration_by_sequence",
    "expected_ce_by_sequence",
    "latent_nmse_by_sequence",
    "retained_syndrome_nll_by_sequence",
]
