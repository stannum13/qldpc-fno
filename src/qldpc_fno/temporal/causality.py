"""Observed-prefix-only interfaces and mutation audits for causal forecasters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _immutable_copy(values: np.ndarray, *, name: str, dimensions: int) -> np.ndarray:
    result = np.array(values, copy=True)
    if result.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservedHistory:
    """The only object exposed to a forecaster during the mutation audit."""

    syndromes: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "syndromes",
            _immutable_copy(self.syndromes, name="syndromes", dimensions=2),
        )


@dataclass(frozen=True)
class CausalAuditSequence:
    """Full test-only record, including fields forbidden to the forecaster."""

    syndromes: np.ndarray
    forecast_round: int
    physical_errors: np.ndarray | None = None
    logical_outcomes: np.ndarray | None = None
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        syndromes = _immutable_copy(self.syndromes, name="syndromes", dimensions=2)
        object.__setattr__(self, "syndromes", syndromes)
        if type(self.forecast_round) is not int or not 0 < self.forecast_round < syndromes.shape[0]:
            raise ValueError("forecast_round must have nonempty past and current/future rounds")
        if self.physical_errors is not None:
            errors = _immutable_copy(
                self.physical_errors,
                name="physical_errors",
                dimensions=2,
            )
            if errors.shape[0] != syndromes.shape[0]:
                raise ValueError("physical_errors must have the same round count as syndromes")
            object.__setattr__(self, "physical_errors", errors)
        if self.logical_outcomes is not None:
            logicals = _immutable_copy(
                self.logical_outcomes,
                name="logical_outcomes",
                dimensions=1,
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


def audit_forecaster_causality(
    forecaster: ObservedPrefixForecaster,
    sequence: CausalAuditSequence,
    forbidden_mutations: Mapping[str, CausalAuditSequence],
) -> Audit:
    """Verify forbidden-field mutations cannot alter a strict-prefix forecast.

    Each supplied mutation must preserve the forecast round and observed syndrome
    prefix. Current/future syndromes and every supervision or diagnostic field are
    excluded structurally from :class:`ObservedHistory`.
    """
    if not forbidden_mutations:
        raise ValueError("forbidden_mutations must contain at least one named mutation")
    baseline_history = _observed_history(sequence)
    baseline = np.asarray(forecaster.forecast(baseline_history))
    checks: list[MutationCheck] = []
    for name, mutation in forbidden_mutations.items():
        if not isinstance(name, str) or not name:
            raise ValueError("mutation names must be nonempty strings")
        if mutation.forecast_round != sequence.forecast_round:
            raise ValueError(f"mutation {name!r} changes forecast_round")
        mutation_history = _observed_history(mutation)
        if not _bit_identical(baseline_history.syndromes, mutation_history.syndromes):
            raise ValueError(f"mutation {name!r} changes the observed syndrome prefix")
        forecast = np.asarray(forecaster.forecast(mutation_history))
        checks.append(MutationCheck(name=name, bit_identical=_bit_identical(baseline, forecast)))
    result = tuple(checks)
    return Audit(
        passed=all(check.bit_identical for check in result),
        forecast_round=sequence.forecast_round,
        checks=result,
    )
