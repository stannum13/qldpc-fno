"""Typed causal scalar filters for the temporal-identifiability study.

Each deployable arm forecasts a round before incorporating that round's
syndrome.  The functions are deliberately stateless: a call owns precisely one
sequence (or an explicit immutable batch of sequences), which prevents a
posterior from crossing sequence identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qldpc_fno.identifiability.config import IdentifiabilityConfig
from qldpc_fno.identifiability.grid import (
    ClippedARGrid,
    build_clipped_ar_grid,
    integrate_probability,
    make_process_cpu_deadline,
    transition_distribution,
    transition_from_point,
)
from qldpc_fno.identifiability.observation import (
    DisjointChecks,
    parity_one_probability,
    retained_log_likelihood,
)
from qldpc_fno.identifiability.types import (
    ContemporaneousOracleInput,
    DeployableHistory,
    LatentHistoryOracleInput,
    require_deployable_history,
)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """One arm's per-round forecast stream and the scalar state that drove it."""

    arm: str
    probabilities: np.ndarray
    state_estimates: np.ndarray | None
    interior_cells: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or not self.arm:
            raise ValueError("forecast arm must be a nonempty string")
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.ndim not in (1, 2) or not np.all(np.isfinite(probabilities)):
            raise ValueError("forecast probabilities must be a finite one- or two-dimensional array")
        if np.any((probabilities <= 0.0) | (probabilities >= 0.5)):
            raise ValueError("forecast probabilities must lie strictly between zero and one half")
        if self.state_estimates is not None:
            states = np.asarray(self.state_estimates, dtype=np.float64)
            if states.ndim != 1 or states.shape[0] != probabilities.shape[0]:
                raise ValueError("state estimates must have one value per forecast round")
            if not np.all(np.isfinite(states)):
                raise ValueError("state estimates must be finite")
            object.__setattr__(self, "state_estimates", _readonly(states))
        if self.interior_cells is not None and (
            type(self.interior_cells) is not int or self.interior_cells <= 0
        ):
            raise ValueError("interior_cells must be a positive integer or None")
        object.__setattr__(self, "probabilities", _readonly(probabilities))

    @property
    def q_hat(self) -> np.ndarray:
        """Alias used by scoring code for the causal probability forecast."""
        return self.probabilities


def _validated_config(config: IdentifiabilityConfig) -> IdentifiabilityConfig:
    if type(config) is not IdentifiabilityConfig:
        raise TypeError("config must be the exact IdentifiabilityConfig type")
    config.validate()
    return config


def _deadline(config: IdentifiabilityConfig, process_cpu_deadline: float | None) -> float:
    if process_cpu_deadline is None:
        return make_process_cpu_deadline(config)
    if not isinstance(process_cpu_deadline, (int, float, np.floating)) or not math.isfinite(
        float(process_cpu_deadline)
    ):
        raise ValueError("process_cpu_deadline must be finite")
    return float(process_cpu_deadline)


def _require_checks(checks: object) -> DisjointChecks:
    if type(checks) is not DisjointChecks:
        raise TypeError("forecasters require the exact DisjointChecks type")
    if not checks.is_pairwise_disjoint:
        raise ValueError("forecasters require pairwise-disjoint retained checks")
    if len(checks.row_indices) == 0:
        raise ValueError("forecasters require at least one retained check")
    return checks


def _history_batch(value: object) -> tuple[tuple[DeployableHistory, ...], bool]:
    if type(value) is DeployableHistory:
        return (require_deployable_history(value),), True
    if type(value) is tuple and value:
        histories = tuple(require_deployable_history(history) for history in value)
        return histories, False
    require_deployable_history(value)
    raise AssertionError("unreachable")


def _retained_syndromes(history: DeployableHistory, checks: DisjointChecks) -> np.ndarray:
    if history.syndromes.shape[0] == 0:
        raise ValueError("forecasters require at least one round")
    indices = checks.row_indices
    if (
        np.any(indices < 0)
        or np.any(indices >= history.syndromes.shape[1])
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("retained check indices must be strictly ascending and in syndrome bounds")
    return history.syndromes[:, checks.row_indices]


def _grid(
    config: IdentifiabilityConfig, *, interior_cells: int, process_cpu_deadline: float
) -> ClippedARGrid:
    return build_clipped_ar_grid(
        config, interior_cells=interior_cells, process_cpu_deadline=process_cpu_deadline
    )


def _log_posterior_update(
    prior: np.ndarray, syndrome: np.ndarray, checks: DisjointChecks, config: IdentifiabilityConfig, grid: ClippedARGrid
) -> np.ndarray:
    log_prior = np.full(prior.shape, -np.inf, dtype=np.float64)
    positive = prior > 0.0
    log_prior[positive] = np.log(prior[positive])
    log_weights = log_prior + retained_log_likelihood(syndrome, grid.states, checks, config)
    normalizer = float(np.logaddexp.reduce(log_weights))
    if not math.isfinite(normalizer):
        raise FloatingPointError("observation update has a nonfinite normalizer")
    posterior = np.exp(log_weights - normalizer)
    if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
        raise FloatingPointError("observation update produced an invalid posterior")
    return posterior / float(np.sum(posterior))


def _single_known_marginal(
    history: DeployableHistory,
    checks: DisjointChecks,
    config: IdentifiabilityConfig,
    deadline: float,
) -> ForecastResult:
    _retained_syndromes(history, checks)  # Validate the common deployable contract.
    grid = _grid(config, interior_cells=config.grid.open_loop_interior_cells, process_cpu_deadline=deadline)
    rounds = history.syndromes.shape[0]
    probabilities = np.empty(rounds, dtype=np.float64)
    states = np.empty(rounds, dtype=np.float64)
    probabilities[0] = config.dynamics.base_probability
    states[0] = 0.0
    prior = transition_from_point(grid, grid.initial_state_value)
    for index in range(1, rounds):
        probabilities[index] = integrate_probability(grid, prior)
        states[index] = float(prior @ grid.states)
        prior = transition_distribution(grid, prior)
    return ForecastResult("known_marginal", probabilities, states, grid.interior_cells)


def forecast_known_marginal(
    history: DeployableHistory | tuple[DeployableHistory, ...],
    checks: DisjointChecks,
    config: IdentifiabilityConfig,
    *,
    process_cpu_deadline: float | None = None,
) -> ForecastResult | tuple[ForecastResult, ...]:
    """Return the fixed 4,096-cell open-loop causal comparator."""
    config = _validated_config(config)
    checks = _require_checks(checks)
    histories, single = _history_batch(history)
    deadline = _deadline(config, process_cpu_deadline)
    results = tuple(_single_known_marginal(item, checks, config, deadline) for item in histories)
    return results[0] if single else results


def _moment_state(syndrome: np.ndarray, checks: DisjointChecks, config: IdentifiabilityConfig) -> float:
    if not np.all(checks.weights == 10):
        raise ValueError("parity_moment_ar requires retained weight-ten checks")
    fraction = float(np.mean(syndrome))
    lower_q = _scalar_probability(-config.dynamics.clip, config)
    upper_q = _scalar_probability(config.dynamics.clip, config)
    lower = float(parity_one_probability(lower_q, 10))
    upper = float(parity_one_probability(upper_q, 10))
    if fraction <= lower:
        return -config.dynamics.clip
    if fraction >= upper:
        return config.dynamics.clip
    # The constrained checks above make this root real even when an observed
    # parity fraction is above one half: such fractions already hit upper.
    q = (1.0 - (1.0 - 2.0 * fraction) ** (1.0 / 10.0)) / 2.0
    base = config.dynamics.base_probability
    return float(math.log(q / (1.0 - q)) - math.log(base / (1.0 - base)))


def _scalar_probability(state: float, config: IdentifiabilityConfig) -> float:
    base = config.dynamics.base_probability
    probability = 1.0 / (1.0 + math.exp(-(math.log(base / (1.0 - base)) + state)))
    return float(np.clip(probability, *config.dynamics.probability_clip))


def _single_parity_moment(
    history: DeployableHistory,
    checks: DisjointChecks,
    config: IdentifiabilityConfig,
    deadline: float,
    interior_cells: int,
) -> ForecastResult:
    observations = _retained_syndromes(history, checks)
    grid = _grid(config, interior_cells=interior_cells, process_cpu_deadline=deadline)
    rounds = observations.shape[0]
    probabilities = np.empty(rounds, dtype=np.float64)
    states = np.empty(rounds, dtype=np.float64)
    observed_state = 0.0
    for index in range(rounds):
        states[index] = observed_state
        if index == 0:
            probabilities[index] = config.dynamics.base_probability
        else:
            probabilities[index] = integrate_probability(
                grid, transition_from_point(grid, observed_state)
            )
        observed_state = _moment_state(observations[index], checks, config)
    return ForecastResult("parity_moment_ar", probabilities, states, grid.interior_cells)


def forecast_parity_moment(
    history: DeployableHistory | tuple[DeployableHistory, ...],
    checks: DisjointChecks,
    config: IdentifiabilityConfig,
    *,
    interior_cells: int | None = None,
    process_cpu_deadline: float | None = None,
) -> ForecastResult | tuple[ForecastResult, ...]:
    """Forecast with the constrained retained-row parity-moment MLE."""
    config = _validated_config(config)
    checks = _require_checks(checks)
    histories, single = _history_batch(history)
    cells = config.grid.interior_cells if interior_cells is None else interior_cells
    deadline = _deadline(config, process_cpu_deadline)
    results = tuple(
        _single_parity_moment(item, checks, config, deadline, cells) for item in histories
    )
    return results[0] if single else results


def _single_grid_bayes(
    history: DeployableHistory,
    checks: DisjointChecks,
    config: IdentifiabilityConfig,
    deadline: float,
    interior_cells: int,
) -> ForecastResult:
    observations = _retained_syndromes(history, checks)
    grid = _grid(config, interior_cells=interior_cells, process_cpu_deadline=deadline)
    rounds = observations.shape[0]
    probabilities = np.empty(rounds, dtype=np.float64)
    states = np.empty(rounds, dtype=np.float64)
    probabilities[0] = config.dynamics.base_probability
    states[0] = grid.initial_state_value
    prior = transition_from_point(grid, grid.initial_state_value)
    for index in range(1, rounds):
        probabilities[index] = integrate_probability(grid, prior)
        states[index] = float(prior @ grid.states)
        if index < rounds - 1:
            posterior = _log_posterior_update(prior, observations[index], checks, config, grid)
            prior = transition_distribution(grid, posterior)
    return ForecastResult("grid_bayes", probabilities, states, grid.interior_cells)


def forecast_grid_bayes(
    history: DeployableHistory | tuple[DeployableHistory, ...],
    checks: DisjointChecks,
    config: IdentifiabilityConfig,
    *,
    interior_cells: int | None = None,
    process_cpu_deadline: float | None = None,
) -> ForecastResult | tuple[ForecastResult, ...]:
    """Run the exact retained-row one-dimensional Bayesian causal filter."""
    config = _validated_config(config)
    checks = _require_checks(checks)
    histories, single = _history_batch(history)
    cells = config.grid.interior_cells if interior_cells is None else interior_cells
    deadline = _deadline(config, process_cpu_deadline)
    results = tuple(_single_grid_bayes(item, checks, config, deadline, cells) for item in histories)
    return results[0] if single else results


def forecast_latent_history(
    latent: LatentHistoryOracleInput,
    config: IdentifiabilityConfig,
    *,
    interior_cells: int | None = None,
    process_cpu_deadline: float | None = None,
) -> ForecastResult:
    """Privileged causal ceiling: integrate from the true previous latent state."""
    if type(latent) is not LatentHistoryOracleInput:
        raise TypeError("latent oracle requires the exact LatentHistoryOracleInput type")
    config = _validated_config(config)
    states = latent.global_log_odds
    if states.shape[0] == 0:
        raise ValueError("latent oracle requires at least one round")
    if np.any(np.abs(states) > config.dynamics.clip):
        raise ValueError("latent oracle states must lie within clipping boundaries")
    cells = config.grid.interior_cells if interior_cells is None else interior_cells
    grid = _grid(config, interior_cells=cells, process_cpu_deadline=_deadline(config, process_cpu_deadline))
    probabilities = np.empty(states.shape[0], dtype=np.float64)
    state_estimates = np.empty(states.shape[0], dtype=np.float64)
    probabilities[0] = config.dynamics.base_probability
    state_estimates[0] = 0.0
    for index in range(1, states.shape[0]):
        state_estimates[index] = states[index - 1]
        probabilities[index] = integrate_probability(
            grid, transition_from_point(grid, float(states[index - 1]))
        )
    return ForecastResult("latent_history_oracle", probabilities, state_estimates, grid.interior_cells)


def forecast_contemporaneous(
    current: ContemporaneousOracleInput, config: IdentifiabilityConfig
) -> ForecastResult:
    """Privileged noncausal ceiling: return the simulator's current probabilities."""
    if type(current) is not ContemporaneousOracleInput:
        raise TypeError("current oracle requires the exact ContemporaneousOracleInput type")
    _validated_config(config)
    if current.probabilities.shape[0] == 0:
        raise ValueError("current oracle requires at least one round")
    return ForecastResult("contemporaneous_oracle", current.probabilities, None, None)
