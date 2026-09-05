"""Exact retained-row syndrome likelihoods for the scalar observation model."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.special import expit

from qldpc_fno.campaign.code_identity import sparse_binary_sha256
from qldpc_fno.identifiability.config import IdentifiabilityConfig

_ALGORITHM_VERSION = "greedy_disjoint_rows/v1"


def _readonly(array: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


def _canonical_binary_csr(hx: sparse.spmatrix) -> sparse.csr_matrix:
    if not sparse.issparse(hx) or hx.ndim != 2:
        raise TypeError("hx must be a two-dimensional scipy sparse matrix")
    canonical = hx.astype(np.uint8).tocsr(copy=True)
    canonical.sum_duplicates()
    canonical.data %= 2
    canonical.eliminate_zeros()
    canonical.sort_indices()
    return canonical


@dataclass(frozen=True, slots=True)
class DisjointChecks:
    """Deterministically retained checks and their bound canonical code identity."""

    row_indices: np.ndarray
    supports: tuple[np.ndarray, ...]
    weights: np.ndarray
    covered_qubits: np.ndarray
    algorithm_version: str
    matrix_sha256: str

    def __post_init__(self) -> None:
        indices = np.asarray(self.row_indices)
        weights = np.asarray(self.weights)
        covered = np.asarray(self.covered_qubits)
        if indices.ndim != 1 or weights.ndim != 1 or covered.ndim != 1:
            raise ValueError("check indices, weights, and covered qubits must be one-dimensional")
        if len(indices) != len(self.supports) or len(weights) != len(self.supports):
            raise ValueError("retained-check metadata lengths must match supports")
        supports: list[np.ndarray] = []
        for support, weight in zip(self.supports, weights, strict=True):
            value = np.asarray(support)
            if value.ndim != 1 or len(value) != int(weight):
                raise ValueError("each retained support must match its recorded weight")
            if len(value) and (np.any(value < 0) or len(np.unique(value)) != len(value)):
                raise ValueError("retained supports must contain unique non-negative qubits")
            supports.append(_readonly(value, dtype=np.int64))
        if self.algorithm_version != _ALGORITHM_VERSION:
            raise ValueError("unsupported retained-row algorithm version")
        if len(self.matrix_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.matrix_sha256
        ):
            raise ValueError("retained checks require a lowercase SHA-256 matrix hash")
        object.__setattr__(self, "row_indices", _readonly(indices, dtype=np.int64))
        object.__setattr__(self, "supports", tuple(supports))
        object.__setattr__(self, "weights", _readonly(weights, dtype=np.int64))
        object.__setattr__(self, "covered_qubits", _readonly(covered, dtype=np.int64))

    @property
    def is_pairwise_disjoint(self) -> bool:
        """Whether no retained supports share a physical qubit."""
        all_qubits = np.concatenate(self.supports) if self.supports else np.empty(0, dtype=np.int64)
        total = len(all_qubits)
        if total != len(self.covered_qubits):
            return False
        return np.array_equal(self.covered_qubits, np.unique(all_qubits))


def greedy_disjoint_rows(hx: sparse.spmatrix) -> DisjointChecks:
    """Retain the ascending-first greedy row-disjoint subset of canonical ``Hx``."""
    canonical = _canonical_binary_csr(hx)
    used = np.zeros(canonical.shape[1], dtype=np.bool_)
    rows: list[int] = []
    supports: list[np.ndarray] = []
    for row_index in range(canonical.shape[0]):
        support = canonical.indices[canonical.indptr[row_index] : canonical.indptr[row_index + 1]]
        if not np.any(used[support]):
            rows.append(row_index)
            supports.append(support.copy())
            used[support] = True
    covered = np.flatnonzero(used)
    return DisjointChecks(
        row_indices=np.asarray(rows, dtype=np.int64),
        supports=tuple(supports),
        weights=np.asarray([len(support) for support in supports], dtype=np.int64),
        covered_qubits=covered,
        algorithm_version=_ALGORITHM_VERSION,
        matrix_sha256=sparse_binary_sha256(canonical),
    )


def parity_one_probability(q: float | np.ndarray, weight: int) -> np.ndarray:
    """Return the odd-parity probability for ``weight`` iid Bernoulli draws."""
    if isinstance(weight, bool) or not isinstance(weight, (int, np.integer)) or weight <= 0:
        raise ValueError("weight must be a positive integer")
    probabilities = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("physical probabilities must be finite values in [0, 1]")
    return 0.5 * (1.0 - np.power(1.0 - 2.0 * probabilities, weight))


def _scalar_probability(state: np.ndarray, config: IdentifiabilityConfig) -> np.ndarray:
    base = config.dynamics.base_probability
    base_logit = math.log(base / (1.0 - base))
    probability = expit(base_logit + state)
    return np.clip(
        probability,
        config.dynamics.probability_clip[0],
        config.dynamics.probability_clip[1],
    )


def retained_log_likelihood(
    syndrome: np.ndarray,
    state_grid: np.ndarray,
    checks: DisjointChecks,
    config: IdentifiabilityConfig,
) -> np.ndarray:
    """Evaluate the exact retained-row log likelihood for each scalar state."""
    if str(getattr(config, "likelihood_kind", "exact_disjoint")) != "exact_disjoint":
        raise ValueError("retained likelihood requires likelihood_kind='exact_disjoint'")
    if not checks.is_pairwise_disjoint:
        raise ValueError("exact_disjoint likelihood rejects overlapping retained rows")
    values = np.asarray(syndrome)
    states = np.asarray(state_grid, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != len(checks.supports):
        raise ValueError("syndrome must contain one binary value per retained check")
    if states.ndim != 1 or not np.all(np.isfinite(states)):
        raise ValueError("state_grid must be a finite one-dimensional array")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("syndrome must be binary")
    q = _scalar_probability(states, config)
    result = np.zeros(states.shape, dtype=np.float64)
    for observation, weight in zip(values, checks.weights, strict=True):
        probability = parity_one_probability(q, int(weight))
        result += np.where(observation == 1, np.log(probability), np.log1p(-probability))
    return result


def _parity_derivative(state: np.ndarray, weight: int, config: IdentifiabilityConfig) -> np.ndarray:
    q = _scalar_probability(state, config)
    unclipped = (q > config.dynamics.probability_clip[0]) & (
        q < config.dynamics.probability_clip[1]
    )
    derivative = weight * np.power(1.0 - 2.0 * q, weight - 1) * q * (1.0 - q)
    return np.where(unclipped, derivative, 0.0)


def scalar_fisher_information(
    state: float, checks: DisjointChecks, config: IdentifiabilityConfig
) -> float:
    """Return scalar Bernoulli Fisher information for retained checks at ``state``."""
    if not checks.is_pairwise_disjoint:
        raise ValueError("Fisher information requires pairwise-disjoint retained checks")
    values = np.asarray([state], dtype=np.float64)
    if not np.isfinite(values[0]):
        raise ValueError("state must be finite")
    total = 0.0
    for weight in checks.weights:
        probability = float(
            parity_one_probability(_scalar_probability(values, config), int(weight))[0]
        )
        derivative = float(_parity_derivative(values, int(weight), config)[0])
        total += derivative**2 / (probability * (1.0 - probability))
    return total


@dataclass(frozen=True, slots=True)
class FisherDrawProvenance:
    """Immutable RNG identity for the preregistered Fisher draw set."""

    domain: str
    seed: int
    law: str
    draws: int


@dataclass(frozen=True, slots=True)
class FisherReport:
    """Typed pass/fail record that gates later confirmatory-data generation."""

    status: Literal["passed", "precheck_failed"]
    provenance: FisherDrawProvenance
    states: np.ndarray
    information: np.ndarray
    analytic_derivatives: np.ndarray
    finite_difference_derivatives: np.ndarray
    minimum_information: float
    median_information: float
    maximum_information: float
    cramer_rao_minimum: float
    cramer_rao_median: float
    cramer_rao_maximum: float
    maximum_derivative_error: float
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field.name, _readonly(value, dtype=np.float64))

    @property
    def precheck_failed(self) -> bool:
        return self.status == "precheck_failed"


def run_fisher_precheck(config: IdentifiabilityConfig, checks: DisjointChecks) -> FisherReport:
    """Run the fixed-seed scalar local-identifiability gate."""
    config.validate()
    fisher = config.fisher
    rng = np.random.default_rng(config.seeds.fisher)
    stationary_std = config.dynamics.innovation_std / math.sqrt(
        1.0 - config.dynamics.ar_coefficient**2
    )
    states = np.clip(
        rng.normal(0.0, stationary_std, size=fisher.draws),
        -config.dynamics.clip,
        config.dynamics.clip,
    )
    weights = np.asarray(checks.weights, dtype=np.int64)
    analytic = np.stack(
        [_parity_derivative(states, int(weight), config) for weight in weights], axis=1
    )
    step = fisher.finite_difference_step
    finite_difference = np.stack(
        [
            (
                parity_one_probability(_scalar_probability(states + step, config), int(weight))
                - parity_one_probability(_scalar_probability(states - step, config), int(weight))
            )
            / (2.0 * step)
            for weight in weights
        ],
        axis=1,
    )
    probabilities = np.stack(
        [
            parity_one_probability(_scalar_probability(states, config), int(weight))
            for weight in weights
        ],
        axis=1,
    )
    information = np.sum(analytic**2 / (probabilities * (1.0 - probabilities)), axis=1)
    derivative_error = np.abs(analytic - finite_difference)
    allowed_error = fisher.absolute_tolerance + fisher.relative_tolerance * np.abs(
        finite_difference
    )
    failures: list[str] = []
    if not checks.is_pairwise_disjoint:
        failures.append("retained checks overlap")
    if not np.all(np.isfinite(information)):
        failures.append("Fisher information is nonfinite")
    if np.any(information <= 0.0):
        failures.append("Fisher information is nonpositive")
    if not np.all(np.isfinite(analytic)) or not np.all(np.isfinite(finite_difference)):
        failures.append("Fisher derivative is nonfinite")
    if np.any(derivative_error > allowed_error):
        failures.append("analytic derivative exceeds finite-difference tolerance")
    finite_information = information[np.isfinite(information) & (information > 0.0)]
    if len(finite_information):
        minimum, median, maximum = np.quantile(finite_information, [0.0, 0.5, 1.0])
        cramer_rao = 1.0 / np.array([minimum, median, maximum])
    else:
        minimum = median = maximum = math.nan
        cramer_rao = np.full(3, math.nan)
    return FisherReport(
        status="precheck_failed" if failures else "passed",
        provenance=FisherDrawProvenance(
            domain=config.seeds.fisher_domain,
            seed=config.seeds.fisher,
            law=fisher.draw_law,
            draws=fisher.draws,
        ),
        states=states,
        information=information,
        analytic_derivatives=analytic,
        finite_difference_derivatives=finite_difference,
        minimum_information=float(minimum),
        median_information=float(median),
        maximum_information=float(maximum),
        cramer_rao_minimum=float(cramer_rao[0]),
        cramer_rao_median=float(cramer_rao[1]),
        cramer_rao_maximum=float(cramer_rao[2]),
        maximum_derivative_error=float(np.max(derivative_error)),
        failure_reasons=tuple(failures),
    )
