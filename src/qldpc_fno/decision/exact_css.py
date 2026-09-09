"""Exact posterior calculations for small binary CSS decoding problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _binary_array(value: np.ndarray, *, ndim: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}-dimensional")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{label} must be binary")
    result = np.array(array, dtype=np.uint8, copy=True, order="C")
    result.flags.writeable = False
    return result


def _gf2_rank(matrix: np.ndarray) -> int:
    work = np.array(matrix, dtype=np.uint8, copy=True)
    rank = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[rank:, column])
        if candidates.size == 0:
            continue
        pivot = rank + int(candidates[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


@dataclass(frozen=True, slots=True)
class ExactCssProblem:
    """A small Z-error decoding problem with explicit CSS equivalences."""

    name: str
    hx: np.ndarray
    z_stabilizers: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray

    def __post_init__(self) -> None:
        hx = _binary_array(self.hx, ndim=2, label="hx")
        stabilizers = _binary_array(self.z_stabilizers, ndim=2, label="z_stabilizers")
        logical_x = _binary_array(self.logical_x, ndim=1, label="logical_x")
        logical_z = _binary_array(self.logical_z, ndim=1, label="logical_z")
        n = hx.shape[1]
        if not self.name:
            raise ValueError("name must be nonempty")
        if stabilizers.shape[1] != n or logical_x.shape != (n,) or logical_z.shape != (n,):
            raise ValueError("all CSS operators must use the same qubit count")
        if _gf2_rank(stabilizers) != stabilizers.shape[0]:
            raise ValueError("z_stabilizers must be an independent basis")
        if np.any((hx @ stabilizers.T) % 2):
            raise ValueError("Z stabilizers must commute with X checks")
        if np.any((hx @ logical_z) % 2):
            raise ValueError("logical Z must commute with X checks")
        if int(logical_x @ logical_z % 2) != 1:
            raise ValueError("logical X and logical Z must anticommute")
        if np.any((stabilizers @ logical_x) % 2):
            raise ValueError("logical X must commute with Z stabilizers")
        object.__setattr__(self, "hx", hx)
        object.__setattr__(self, "z_stabilizers", stabilizers)
        object.__setattr__(self, "logical_x", logical_x)
        object.__setattr__(self, "logical_z", logical_z)

    @property
    def n(self) -> int:
        return int(self.hx.shape[1])

    def syndrome(self, error: np.ndarray) -> np.ndarray:
        candidate = _binary_array(error, ndim=1, label="error")
        if candidate.shape != (self.n,):
            raise ValueError("error has the wrong qubit count")
        return np.asarray((self.hx @ candidate) % 2, dtype=np.uint8)

    def logical_signature(self, error: np.ndarray) -> int:
        candidate = _binary_array(error, ndim=1, label="error")
        if candidate.shape != (self.n,):
            raise ValueError("error has the wrong qubit count")
        return int(self.logical_x @ candidate % 2)


@dataclass(frozen=True, slots=True)
class ExactCssPosterior:
    """The exact conditional posterior for one measured syndrome."""

    syndrome: np.ndarray
    compatible_errors: np.ndarray
    logical_classes: np.ndarray
    physical_probabilities: np.ndarray
    unnormalized_logical_class_masses: np.ndarray
    logical_class_probabilities: np.ndarray
    physical_map_error: np.ndarray
    physical_map_class: int
    coset_map_class: int
    enumerated_error_count: int


@dataclass(frozen=True, slots=True)
class AffineFiber:
    """A unique coefficient parameterization of one syndrome fiber."""

    representative: np.ndarray
    coefficients: np.ndarray
    errors: np.ndarray


def canonical_steane_z_problem() -> ExactCssProblem:
    """Return a canonical self-dual Steane ``[[7,1,3]]`` Z-error problem."""
    checks = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    logical = np.ones(7, dtype=np.uint8)
    return ExactCssProblem(
        name="steane_z_7_1_3",
        hx=checks,
        z_stabilizers=checks,
        logical_x=logical,
        logical_z=logical,
    )


def _all_binary_vectors(width: int) -> np.ndarray:
    indices = np.arange(1 << width, dtype=np.uint64)
    shifts = np.arange(width, dtype=np.uint64)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def affine_fiber(problem: ExactCssProblem, syndrome: np.ndarray) -> AffineFiber:
    """Parameterize a syndrome fiber by stabilizer and logical-Z coefficients."""
    target = _binary_array(syndrome, ndim=1, label="syndrome")
    if target.shape != (problem.hx.shape[0],):
        raise ValueError("syndrome has the wrong check count")
    all_errors = _all_binary_vectors(problem.n)
    all_syndromes = np.asarray((all_errors @ problem.hx.T) % 2, dtype=np.uint8)
    compatible = all_errors[np.all(all_syndromes == target, axis=1)]
    if compatible.shape[0] == 0:
        raise ValueError("syndrome has no compatible physical error")

    representative = compatible[0]
    generators = np.vstack((problem.z_stabilizers, problem.logical_z))
    coefficients = _all_binary_vectors(generators.shape[0])
    errors = representative ^ np.asarray((coefficients @ generators) % 2, dtype=np.uint8)
    if np.unique(errors, axis=0).shape[0] != errors.shape[0]:
        raise AssertionError("affine coefficient representation is not unique")
    if {row.tobytes() for row in errors} != {row.tobytes() for row in compatible}:
        raise AssertionError("affine generators do not span the complete syndrome fiber")
    return AffineFiber(
        representative=np.array(representative, copy=True),
        coefficients=np.array(coefficients, copy=True),
        errors=np.array(errors, copy=True),
    )


def _validate_probabilities(probabilities: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be one finite value per qubit")
    if not np.all((values > 0.0) & (values < 1.0)):
        raise ValueError("probabilities must lie strictly between zero and one")
    return values


def exact_posterior(
    problem: ExactCssProblem,
    syndrome: np.ndarray,
    probabilities: np.ndarray,
) -> ExactCssPosterior:
    """Enumerate and normalize every physical error compatible with ``syndrome``."""
    target = _binary_array(syndrome, ndim=1, label="syndrome")
    if target.shape != (problem.hx.shape[0],):
        raise ValueError("syndrome has the wrong check count")
    channel = _validate_probabilities(probabilities, problem.n)

    compatible = affine_fiber(problem, target).errors

    log_weights = (compatible * np.log(channel) + (1 - compatible) * np.log1p(-channel)).sum(axis=1)
    weights = np.exp(log_weights)
    normalization = float(weights.sum())
    physical_probabilities = weights / normalization
    logical_classes = np.asarray((compatible @ problem.logical_x) % 2, dtype=np.uint8)
    class_masses = np.bincount(logical_classes, weights=weights, minlength=2)
    class_probabilities = class_masses / normalization
    physical_map_index = int(np.argmax(log_weights))

    return ExactCssPosterior(
        syndrome=np.array(target, copy=True),
        compatible_errors=np.array(compatible, copy=True),
        logical_classes=np.array(logical_classes, copy=True),
        physical_probabilities=np.array(physical_probabilities, copy=True),
        unnormalized_logical_class_masses=np.array(class_masses, copy=True),
        logical_class_probabilities=np.array(class_probabilities, copy=True),
        physical_map_error=np.array(compatible[physical_map_index], copy=True),
        physical_map_class=int(logical_classes[physical_map_index]),
        coset_map_class=int(np.argmax(class_masses)),
        enumerated_error_count=1 << problem.n,
    )


def bayes_logical_risk(selected_class: int, true_posterior: ExactCssPosterior) -> float:
    """Return conditional failure probability for a selected relative logical class."""
    if selected_class not in (0, 1):
        raise ValueError("selected_class must be zero or one")
    return float(1.0 - true_posterior.logical_class_probabilities[selected_class])
