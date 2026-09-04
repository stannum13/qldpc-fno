"""Observed-prefix-only interfaces and mutation audits for causal forecasters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _immutable_copy(
    values: np.ndarray,
    *,
    name: str,
    minimum_dimensions: int,
) -> np.ndarray:
    result = np.array(values, copy=True)
    if result.ndim < minimum_dimensions:
        raise ValueError(f"{name} must have at least {minimum_dimensions} dimensions")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservedHistory:
    """Strict syndrome prefix with rounds on axis 0 and geometry preserved."""

    syndromes: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "syndromes",
            _immutable_copy(
                self.syndromes,
                name="syndromes",
                minimum_dimensions=2,
            ),
        )


@dataclass(frozen=True)
class CausalAuditSequence:
    """Full test-only record with rounds on axis 0 and forbidden supervision."""

    syndromes: np.ndarray
    forecast_round: int
    physical_errors: np.ndarray | None = None
    logical_outcomes: np.ndarray | None = None
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        syndromes = _immutable_copy(
            self.syndromes,
            name="syndromes",
            minimum_dimensions=2,
        )
        object.__setattr__(self, "syndromes", syndromes)
        if type(self.forecast_round) is not int or not 0 < self.forecast_round < syndromes.shape[0]:
            raise ValueError("forecast_round must have nonempty past and current/future rounds")
        if self.physical_errors is not None:
            errors = _immutable_copy(
                self.physical_errors,
                name="physical_errors",
                minimum_dimensions=2,
            )
            if errors.shape[0] != syndromes.shape[0]:
                raise ValueError("physical_errors must have the same round count as syndromes")
            object.__setattr__(self, "physical_errors", errors)
        if self.logical_outcomes is not None:
            logicals = _immutable_copy(
                self.logical_outcomes,
                name="logical_outcomes",
                minimum_dimensions=1,
            )
            if logicals.shape[0] != syndromes.shape[0]:
                raise ValueError("logical_outcomes must have the same round count as syndromes")
            object.__setattr__(self, "logical_outcomes", logicals)


class ObservedPrefixForecaster(Protocol):
    def forecast(self, history: ObservedHistory) -> np.ndarray: ...


@dataclass(frozen=True)
class MutationCheck:
    name: str
    bit_identical: bool


@dataclass(frozen=True)
class Audit:
    passed: bool
    forecast_round: int
    checks: tuple[MutationCheck, ...]

    @property
    def checked_mutations(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks)


def _observed_history(sequence: CausalAuditSequence) -> ObservedHistory:
    return ObservedHistory(sequence.syndromes[: sequence.forecast_round])


def _bit_identical(first: np.ndarray, second: np.ndarray) -> bool:
    left = np.asarray(first)
    right = np.asarray(second)
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _values_identical(first: object, second: object) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return _bit_identical(np.asarray(first), np.asarray(second))
        except (TypeError, ValueError):
            return False
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        if set(first) != set(second):
            return False
        return all(_values_identical(first[key], second[key]) for key in first)
    if first is None or second is None:
        return first is second
    try:
        result = first == second
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _records_identical(first: CausalAuditSequence, second: CausalAuditSequence) -> bool:
    return (
        first.forecast_round == second.forecast_round
        and _values_identical(first.syndromes, second.syndromes)
        and _values_identical(first.physical_errors, second.physical_errors)
        and _values_identical(first.logical_outcomes, second.logical_outcomes)
        and _values_identical(first.diagnostics, second.diagnostics)
    )


def audit_structural_prefix_causality(
    forecaster: ObservedPrefixForecaster,
    sequence: CausalAuditSequence,
    forbidden_mutations: Mapping[str, CausalAuditSequence],
) -> Audit:
    """Structurally verify forbidden mutations cannot alter a strict-prefix forecast.

    Each supplied mutation must preserve the forecast round and observed syndrome
    prefix. Current/future syndromes and every supervision or diagnostic field are
    excluded structurally from :class:`ObservedHistory`. This boundary audit cannot
    detect forecaster-captured external privileged state; concrete model audits must
    recreate/reset the model and spy on its actual prediction path.
    """
    if not forbidden_mutations:
        raise ValueError("forbidden_mutations must contain at least one named mutation")
    baseline_history = _observed_history(sequence)
    baseline = np.array(forecaster.forecast(baseline_history), copy=True)
    checks: list[MutationCheck] = []
    for name, mutation in forbidden_mutations.items():
        if not isinstance(name, str) or not name:
            raise ValueError("mutation names must be nonempty strings")
        if _records_identical(sequence, mutation):
            raise ValueError(f"mutation {name!r} is identical to the source record")
        if mutation.forecast_round != sequence.forecast_round:
            raise ValueError(f"mutation {name!r} changes forecast_round")
        mutation_history = _observed_history(mutation)
        if not _bit_identical(baseline_history.syndromes, mutation_history.syndromes):
            raise ValueError(f"mutation {name!r} changes the observed syndrome prefix")
        forecast = np.array(forecaster.forecast(mutation_history), copy=True)
        checks.append(MutationCheck(name=name, bit_identical=_bit_identical(baseline, forecast)))
    result = tuple(checks)
    return Audit(
        passed=all(check.bit_identical for check in result),
        forecast_round=sequence.forecast_round,
        checks=result,
    )
