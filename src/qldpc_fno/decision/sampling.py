"""Sampling baselines for estimating syndrome-conditioned logical mass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qldpc_fno.decision.exact_css import (
    ExactCssPosterior,
    ExactCssProblem,
    affine_fiber,
)


@dataclass(frozen=True, slots=True)
class ConditionalSamples:
    """Syndrome-valid samples and explicit proposal accounting."""

    method: str
    samples: np.ndarray
    proposal_count: int
    accepted_transition_count: int
    setup_candidate_error_materialization_count: int
    setup_syndrome_evaluation_count: int
    setup_affine_terminal_materialization_count: int
    channel_error_draw_count: int
    sampling_syndrome_evaluation_count: int
    target_log_weight_evaluation_count: int
    affine_transition_proposal_count: int
    burn_in_transition_count: int

    @property
    def operation_counts(self) -> dict[str, int]:
        """Return noninterchangeable primitive counts for this sampler."""
        retained = int(self.samples.shape[0])
        return {
            "setup_candidate_error_materializations": (
                self.setup_candidate_error_materialization_count
            ),
            "setup_syndrome_evaluations": self.setup_syndrome_evaluation_count,
            "setup_affine_terminal_materializations": (
                self.setup_affine_terminal_materialization_count
            ),
            "channel_error_draws": self.channel_error_draw_count,
            "sampling_syndrome_evaluations": self.sampling_syndrome_evaluation_count,
            "target_log_weight_evaluations": self.target_log_weight_evaluation_count,
            "affine_transition_proposals": self.affine_transition_proposal_count,
            "burn_in_transitions": self.burn_in_transition_count,
            "retained_samples": retained,
        }


@dataclass(frozen=True, slots=True)
class LogicalMassMetrics:
    """Accuracy and diversity diagnostics relative to exact enumeration."""

    estimated_logical_class_probabilities: np.ndarray
    total_variation: float
    worst_class_absolute_error: float
    top_class_correct: bool
    physical_state_coverage: float
    effective_sample_size: float | None


def _channel(probabilities: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be one finite value per qubit")
    if not np.all((values > 0.0) & (values < 1.0)):
        raise ValueError("probabilities must lie strictly between zero and one")
    return values


def _sample_count(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("sample_count must be a positive integer")
    return value


def rejection_sample(
    problem: ExactCssProblem,
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> ConditionalSamples:
    """Draw from the physical channel and retain exactly matching syndromes."""
    count = _sample_count(sample_count)
    channel = _channel(probabilities, problem.n)
    target = np.asarray(syndrome, dtype=np.uint8)
    if target.shape != (problem.hx.shape[0],):
        raise ValueError("syndrome has the wrong check count")
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    proposals = 0
    while len(accepted) < count:
        error = np.asarray(rng.random(problem.n) < channel, dtype=np.uint8)
        proposals += 1
        if np.array_equal(problem.syndrome(error), target):
            accepted.append(error)
    return ConditionalSamples(
        method="conditional_rejection",
        samples=np.stack(accepted),
        proposal_count=proposals,
        accepted_transition_count=count,
        setup_candidate_error_materialization_count=0,
        setup_syndrome_evaluation_count=0,
        setup_affine_terminal_materialization_count=0,
        channel_error_draw_count=proposals,
        sampling_syndrome_evaluation_count=proposals,
        target_log_weight_evaluation_count=0,
        affine_transition_proposal_count=0,
        burn_in_transition_count=0,
    )


def rejection_sample_for_proposals(
    problem: ExactCssProblem,
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    *,
    proposal_budget: int,
    seed: int,
) -> ConditionalSamples:
    """Run rejection sampling for exactly ``proposal_budget`` physical draws."""
    budget = _sample_count(proposal_budget)
    channel = _channel(probabilities, problem.n)
    target = np.asarray(syndrome, dtype=np.uint8)
    if target.shape != (problem.hx.shape[0],):
        raise ValueError("syndrome has the wrong check count")
    rng = np.random.default_rng(seed)
    retained = []
    for _ in range(budget):
        error = np.asarray(rng.random(problem.n) < channel, dtype=np.uint8)
        if np.array_equal(problem.syndrome(error), target):
            retained.append(error)
    if not retained:
        raise RuntimeError("proposal budget produced no syndrome-compatible sample")
    return ConditionalSamples(
        method="conditional_rejection",
        samples=np.stack(retained),
        proposal_count=budget,
        accepted_transition_count=len(retained),
        setup_candidate_error_materialization_count=0,
        setup_syndrome_evaluation_count=0,
        setup_affine_terminal_materialization_count=0,
        channel_error_draw_count=budget,
        sampling_syndrome_evaluation_count=budget,
        target_log_weight_evaluation_count=0,
        affine_transition_proposal_count=0,
        burn_in_transition_count=0,
    )


def _log_weight(error: np.ndarray, probabilities: np.ndarray) -> float:
    return float((error * np.log(probabilities) + (1 - error) * np.log1p(-probabilities)).sum())


def metropolis_affine_sample(
    problem: ExactCssProblem,
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    *,
    sample_count: int,
    seed: int,
    burn_in: int,
    thinning: int,
) -> ConditionalSamples:
    """Metropolis sample by flipping a unique affine-space coefficient."""
    count = _sample_count(sample_count)
    if type(burn_in) is not int or burn_in < 0:
        raise ValueError("burn_in must be a non-negative integer")
    if type(thinning) is not int or thinning <= 0:
        raise ValueError("thinning must be a positive integer")
    channel = _channel(probabilities, problem.n)
    fiber = affine_fiber(problem, syndrome)
    generators = np.vstack((problem.z_stabilizers, problem.logical_z))
    current = np.array(fiber.representative, copy=True)
    current_log_weight = _log_weight(current, channel)
    rng = np.random.default_rng(seed)
    retained: list[np.ndarray] = []
    proposals = 0
    accepted = 0
    total_steps = burn_in + count * thinning
    for step in range(total_steps):
        displacement_index = int(rng.integers(1, 1 << generators.shape[0]))
        displacement_coefficients = np.array(
            [(displacement_index >> shift) & 1 for shift in range(generators.shape[0])],
            dtype=np.uint8,
        )
        displacement = np.asarray((displacement_coefficients @ generators) % 2, dtype=np.uint8)
        proposed = current ^ displacement
        proposed_log_weight = _log_weight(proposed, channel)
        proposals += 1
        if np.log(rng.random()) < min(0.0, proposed_log_weight - current_log_weight):
            current = proposed
            current_log_weight = proposed_log_weight
            accepted += 1
        if step >= burn_in and (step - burn_in + 1) % thinning == 0:
            retained.append(np.array(current, copy=True))
    if len(retained) != count:
        raise AssertionError("Metropolis retention schedule produced the wrong sample count")
    return ConditionalSamples(
        method="affine_metropolis",
        samples=np.stack(retained),
        proposal_count=proposals,
        accepted_transition_count=accepted,
        setup_candidate_error_materialization_count=1 << problem.n,
        setup_syndrome_evaluation_count=1 << problem.n,
        setup_affine_terminal_materialization_count=len(fiber.errors),
        channel_error_draw_count=0,
        sampling_syndrome_evaluation_count=0,
        target_log_weight_evaluation_count=proposals + 1,
        affine_transition_proposal_count=proposals,
        burn_in_transition_count=burn_in,
    )


def metropolis_affine_sample_for_proposals(
    problem: ExactCssProblem,
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    *,
    proposal_budget: int,
    seed: int,
    burn_in: int,
    thinning: int,
) -> ConditionalSamples:
    """Run affine Metropolis for an exact transition-proposal budget."""
    budget = _sample_count(proposal_budget)
    if budget <= burn_in or (budget - burn_in) % thinning != 0:
        raise ValueError("proposal_budget minus burn_in must be positive and divisible by thinning")
    return metropolis_affine_sample(
        problem,
        syndrome,
        probabilities,
        sample_count=(budget - burn_in) // thinning,
        seed=seed,
        burn_in=burn_in,
        thinning=thinning,
    )


def _effective_sample_size(binary_values: np.ndarray) -> float | None:
    values = np.asarray(binary_values, dtype=np.float64)
    count = len(values)
    centered = values - values.mean()
    autocovariances = np.array(
        [float(centered[: count - lag] @ centered[lag:] / count) for lag in range(count)]
    )
    if autocovariances[0] == 0.0:
        return None
    positive_monotone_pairs: list[float] = []
    previous = np.inf
    for lag in range(0, count - 1, 2):
        pair = float(autocovariances[lag] + autocovariances[lag + 1])
        if pair <= 0.0:
            break
        pair = min(pair, previous)
        positive_monotone_pairs.append(pair)
        previous = pair
    if not positive_monotone_pairs:
        return float(count)
    integrated_time = -1.0 + 2.0 * sum(positive_monotone_pairs) / autocovariances[0]
    return float(max(1.0, min(count, count / max(1.0, integrated_time))))


def logical_mass_metrics(
    problem: ExactCssProblem,
    samples: np.ndarray,
    exact: ExactCssPosterior,
) -> LogicalMassMetrics:
    """Compare sampled relative logical mass with an exact posterior."""
    values = np.asarray(samples, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != problem.n or values.shape[0] == 0:
        raise ValueError("samples must be a nonempty error matrix")
    signatures = np.asarray((values @ problem.logical_x) % 2, dtype=np.uint8)
    estimated = np.bincount(signatures, minlength=2) / len(signatures)
    error = np.abs(estimated - exact.logical_class_probabilities)
    coverage = np.unique(values, axis=0).shape[0] / len(exact.compatible_errors)
    estimated_winners = np.flatnonzero(estimated == estimated.max())
    return LogicalMassMetrics(
        estimated_logical_class_probabilities=estimated,
        total_variation=float(0.5 * error.sum()),
        worst_class_absolute_error=float(error.max()),
        top_class_correct=(
            len(estimated_winners) == 1 and int(estimated_winners[0]) == exact.coset_map_class
        ),
        physical_state_coverage=float(coverage),
        effective_sample_size=_effective_sample_size(signatures),
    )
