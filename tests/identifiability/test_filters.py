"""Tests for typed, causal scalar identifiability filters."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from time import process_time

import numpy as np
import pytest
from scipy import sparse

from qldpc_fno.identifiability.config import load_identifiability_config
from qldpc_fno.identifiability.grid import (
    build_clipped_ar_grid,
    integrate_probability,
    transition_distribution,
    transition_from_point,
)
from qldpc_fno.identifiability.observation import (
    DisjointChecks,
    greedy_disjoint_rows,
    parity_one_probability,
)
from qldpc_fno.identifiability.types import (
    ContemporaneousOracleInput,
    DeployableHistory,
    LatentHistoryOracleInput,
    TrainingTargets,
)

CONFIG_PATH = Path("configs/temporal_identifiability.json")


def _config():
    return load_identifiability_config(CONFIG_PATH)


def _deadline(config) -> float:
    return process_time() + config.runtime.process_cpu_seconds


def _checks() -> DisjointChecks:
    """Three disjoint weight-ten checks, sufficient for moment edge cases."""
    return greedy_disjoint_rows(
        sparse.csr_matrix(
            (
                np.ones(30, dtype=np.uint8),
                np.arange(30, dtype=np.int64),
                np.array([0, 10, 20, 30], dtype=np.int64),
            ),
            shape=(3, 30),
        )
    )


def _history(rows: list[list[int]], *, scored: list[bool] | None = None) -> DeployableHistory:
    values = np.asarray(rows, dtype=np.uint8)
    return DeployableHistory(
        syndromes=values,
        scored_mask=np.asarray([True] * len(values) if scored is None else scored, dtype=np.bool_),
    )


def _public_history() -> DeployableHistory:
    return _history([[0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0]])


def _scalar_probability(state: float, config) -> float:
    base = config.dynamics.base_probability
    return float(1.0 / (1.0 + math.exp(-(math.log(base / (1.0 - base)) + state))))


def _grid(config, cells: int):
    return build_clipped_ar_grid(
        config, interior_cells=cells, process_cpu_deadline=_deadline(config)
    )


def test_known_marginal_is_the_fixed_4096_cell_open_loop_without_observations() -> None:
    from qldpc_fno.identifiability.filters import forecast_known_marginal

    config = _config()
    history = _public_history()
    result = forecast_known_marginal(history, _checks(), config, process_cpu_deadline=_deadline(config))

    grid = _grid(config, 4_096)
    expected = np.empty(history.syndromes.shape[0], dtype=np.float64)
    expected[0] = config.dynamics.base_probability
    posterior = transition_from_point(grid, 0.0)
    for index in range(1, expected.size):
        expected[index] = integrate_probability(grid, posterior)
        posterior = transition_distribution(grid, posterior)

    assert result.arm == "known_marginal"
    assert result.interior_cells == 4_096
    np.testing.assert_array_equal(result.probabilities, expected)


@pytest.mark.parametrize(
    ("observed", "expected_state"),
    [([0, 0, 0], -1.2), ([1, 1, 1], 1.2), ([1, 1, 0], 1.2)],
)
def test_parity_moment_uses_constrained_mle_at_bounds_and_above_half(
    observed: list[int], expected_state: float
) -> None:
    from qldpc_fno.identifiability.filters import forecast_parity_moment

    config = _config()
    history = _history([[0, 0, 0], observed, [0, 0, 0]])
    result = forecast_parity_moment(
        history, _checks(), config, interior_cells=16, process_cpu_deadline=_deadline(config)
    )

    grid = _grid(config, 16)
    expected = integrate_probability(grid, transition_from_point(grid, expected_state))
    assert result.state_estimates[2] == pytest.approx(expected_state, abs=0.0)
    assert result.probabilities[2] == pytest.approx(expected, abs=2e-15)


def test_parity_moment_uses_the_monotone_inverse_inside_the_bounds() -> None:
    from qldpc_fno.identifiability.filters import forecast_parity_moment

    config = _config()
    # One odd retained row gives y=1/3, strictly between both constrained values.
    history = _history([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    result = forecast_parity_moment(
        history, _checks(), config, process_cpu_deadline=_deadline(config)
    )

    y = 1.0 / 3.0
    q = (1.0 - (1.0 - 2.0 * y) ** (1.0 / 10.0)) / 2.0
    expected_state = math.log(q / (1.0 - q)) - math.log(
        config.dynamics.base_probability / (1.0 - config.dynamics.base_probability)
    )
    assert result.state_estimates[2] == pytest.approx(expected_state, abs=1e-14)


def test_grid_bayes_uses_the_exact_small_grid_likelihood_and_forecast_then_update() -> None:
    from qldpc_fno.identifiability.filters import forecast_grid_bayes

    config = _config()
    history = _history([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    result = forecast_grid_bayes(
        history,
        _checks(),
        config,
        interior_cells=8,
        process_cpu_deadline=_deadline(config),
    )

    grid = _grid(config, 8)
    prior = transition_from_point(grid, 0.0)
    q = 1.0 / (1.0 + np.exp(-(math.log(0.0375 / 0.9625) + grid.states)))
    row_probability = parity_one_probability(q, 10)
    likelihood = 2.0 * np.log(row_probability) + np.log1p(-row_probability)
    posterior = np.exp(np.log(prior) + likelihood - np.logaddexp.reduce(np.log(prior) + likelihood))
    expected_next = transition_distribution(grid, posterior)

    assert result.probabilities[0] == config.dynamics.base_probability
    assert result.probabilities[1] == pytest.approx(integrate_probability(grid, prior), abs=2e-15)
    assert result.probabilities[2] == pytest.approx(
        integrate_probability(grid, expected_next), abs=2e-15
    )


def test_grid_bayes_first_informative_lag_is_syndrome_one_when_g_zero_is_known() -> None:
    from qldpc_fno.identifiability.filters import forecast_grid_bayes

    config = _config()
    checks = _checks()
    histories = {
        "baseline": _history([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        "changed_zero": _history([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
        "changed_one": _history([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
    }
    forecasts = {
        name: forecast_grid_bayes(
            history,
            checks,
            config,
            interior_cells=8,
            process_cpu_deadline=_deadline(config),
        ).probabilities
        for name, history in histories.items()
    }

    # g[0] is the exact known point zero, so observing syndrome_0 cannot update
    # it.  The round-one forecast is consequently identical for either value.
    np.testing.assert_array_equal(forecasts["baseline"], forecasts["changed_zero"])
    assert forecasts["baseline"][1] == forecasts["changed_one"][1]
    # syndrome_1 observes the uncertain g[1] state after q_hat_1 is emitted and
    # therefore changes the next forecast, q_hat_2.
    assert forecasts["baseline"][2] != forecasts["changed_one"][2]


def test_grid_bayes_skips_terminal_update_and_transition_without_changing_forecasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qldpc_fno.identifiability import filters as filters_module
    from qldpc_fno.identifiability.filters import forecast_grid_bayes
    from qldpc_fno.identifiability.grid import ExperimentDeadlineExceeded

    config = _config()
    checks = _checks()
    history = _history([[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]])
    arguments = {
        "interior_cells": 8,
        "process_cpu_deadline": _deadline(config),
    }
    expected = forecast_grid_bayes(history, checks, config, **arguments)
    real_update = filters_module._log_posterior_update
    real_transition = filters_module.transition_distribution
    updated_syndromes: list[np.ndarray] = []
    transition_calls = 0

    def recording_update(prior, syndrome, *args):
        updated_syndromes.append(syndrome.copy())
        return real_update(prior, syndrome, *args)

    def terminal_deadline_transition(grid, posterior):
        nonlocal transition_calls
        transition_calls += 1
        if transition_calls == history.syndromes.shape[0] - 1:
            raise ExperimentDeadlineExceeded("deadline reached after final forecast")
        return real_transition(grid, posterior)

    monkeypatch.setattr(filters_module, "_log_posterior_update", recording_update)
    monkeypatch.setattr(filters_module, "transition_distribution", terminal_deadline_transition)

    actual = forecast_grid_bayes(history, checks, config, **arguments)

    np.testing.assert_array_equal(actual.probabilities, expected.probabilities)
    np.testing.assert_array_equal(actual.state_estimates, expected.state_estimates)
    assert transition_calls == history.syndromes.shape[0] - 2
    assert len(updated_syndromes) == history.syndromes.shape[0] - 2
    np.testing.assert_array_equal(updated_syndromes, history.syndromes[1:-1])


def test_latent_history_integrates_only_the_previous_exact_state() -> None:
    from qldpc_fno.identifiability.filters import forecast_latent_history

    config = _config()
    latent = LatentHistoryOracleInput(global_log_odds=np.array([0.0, -0.4, 0.75]))
    result = forecast_latent_history(latent, config, process_cpu_deadline=_deadline(config))

    grid = _grid(config, 2_048)
    expected = np.array(
        [
            config.dynamics.base_probability,
            integrate_probability(grid, transition_from_point(grid, 0.0)),
            integrate_probability(grid, transition_from_point(grid, -0.4)),
        ]
    )
    np.testing.assert_allclose(result.probabilities, expected, rtol=0.0, atol=2e-15)
    assert result.state_estimates.tolist() == [0.0, 0.0, -0.4]


def test_contemporaneous_oracle_is_an_identity_ceiling() -> None:
    from qldpc_fno.identifiability.filters import forecast_contemporaneous

    values = np.array([[0.02, 0.03], [0.04, 0.05]])
    result = forecast_contemporaneous(ContemporaneousOracleInput(values), _config())

    np.testing.assert_array_equal(result.probabilities, values)
    assert result.arm == "contemporaneous_oracle"


def test_outputs_are_finite_strictly_bounded_and_batch_equivalent_to_separate_calls() -> None:
    from qldpc_fno.identifiability.filters import (
        forecast_grid_bayes,
        forecast_known_marginal,
        forecast_parity_moment,
    )

    config = _config()
    histories = (_public_history(), _history([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]))
    for forecaster in (forecast_known_marginal, forecast_parity_moment, forecast_grid_bayes):
        arguments = {"process_cpu_deadline": _deadline(config)}
        if forecaster is not forecast_known_marginal:
            arguments["interior_cells"] = 8
        combined = forecaster(histories, _checks(), config, **arguments)
        separate = tuple(
            forecaster(history, _checks(), config, **arguments)
            for history in histories
        )
        assert isinstance(combined, tuple)
        for one, expected in zip(combined, separate, strict=True):
            np.testing.assert_array_equal(one.probabilities, expected.probabilities)
            assert np.all(np.isfinite(one.probabilities))
            assert np.all((one.probabilities > 0.0) & (one.probabilities < 0.5))


@dataclass(frozen=True)
class _CombinedInput:
    deployable: DeployableHistory
    latent: LatentHistoryOracleInput
    contemporaneous: ContemporaneousOracleInput


def test_deployable_arms_reject_every_privileged_or_combined_input_by_exact_type() -> None:
    from qldpc_fno.identifiability.filters import (
        forecast_grid_bayes,
        forecast_known_marginal,
        forecast_parity_moment,
    )

    config = _config()
    forbidden = (
        LatentHistoryOracleInput(np.zeros(2)),
        ContemporaneousOracleInput(np.full((2, 2), 0.1)),
        TrainingTargets(np.zeros((2, 2)), np.zeros((2, 1))),
        _CombinedInput(
            _history([[0, 0, 0], [0, 0, 0]]),
            LatentHistoryOracleInput(np.zeros(2)),
            ContemporaneousOracleInput(np.full((2, 2), 0.1)),
        ),
    )
    for forecaster in (forecast_known_marginal, forecast_parity_moment, forecast_grid_bayes):
        for input_value in forbidden:
            arguments = {"process_cpu_deadline": _deadline(config)}
            if forecaster is not forecast_known_marginal:
                arguments["interior_cells"] = 8
            with pytest.raises(TypeError, match="exact DeployableHistory"):
                forecaster(input_value, _checks(), config, **arguments)


def test_deployable_forecasts_are_strictly_causal_but_previous_syndrome_can_change_them() -> None:
    from qldpc_fno.identifiability.filters import forecast_grid_bayes

    config = _config()
    original = _public_history()
    mutated = original.syndromes.copy()
    mutated[2:] ^= 1
    future_changed = DeployableHistory(mutated, original.scored_mask)
    baseline = forecast_grid_bayes(
        original, _checks(), config, interior_cells=8, process_cpu_deadline=_deadline(config)
    )
    changed = forecast_grid_bayes(
        future_changed, _checks(), config, interior_cells=8, process_cpu_deadline=_deadline(config)
    )
    # Round two is predicted before syndrome two or later values are observed.
    assert baseline.probabilities[2].tobytes() == changed.probabilities[2].tobytes()
    assert baseline.probabilities[1].tobytes() == changed.probabilities[1].tobytes()
    assert baseline.probabilities[3] != changed.probabilities[3]


def test_latent_and_current_privileges_obey_their_distinct_timing_contracts() -> None:
    from qldpc_fno.identifiability.filters import (
        forecast_contemporaneous,
        forecast_latent_history,
    )

    config = _config()
    latent = LatentHistoryOracleInput(np.array([0.0, -0.2, 0.4]))
    altered_latent = LatentHistoryOracleInput(np.array([0.0, -0.2, -1.0]))
    assert forecast_latent_history(latent, config).probabilities[2].tobytes() == forecast_latent_history(
        altered_latent, config
    ).probabilities[2].tobytes()

    current = ContemporaneousOracleInput(np.full((3, 2), 0.1))
    altered_current = ContemporaneousOracleInput(np.array([[0.1, 0.1], [0.1, 0.1], [0.2, 0.2]]))
    assert forecast_contemporaneous(current, config).probabilities[2].tobytes() != forecast_contemporaneous(
        altered_current, config
    ).probabilities[2].tobytes()


def test_filters_are_stateless_across_sequence_resets_and_refuse_nonfinite_or_degenerate_inputs() -> None:
    from qldpc_fno.identifiability.filters import forecast_grid_bayes

    config = _config()
    second = _history([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    after_first = forecast_grid_bayes(
        _history([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        _checks(),
        config,
        interior_cells=8,
        process_cpu_deadline=_deadline(config),
    )
    fresh = forecast_grid_bayes(
        second, _checks(), config, interior_cells=8, process_cpu_deadline=_deadline(config)
    )
    repeated = forecast_grid_bayes(
        second, _checks(), config, interior_cells=8, process_cpu_deadline=_deadline(config)
    )
    assert after_first.probabilities[2] != fresh.probabilities[2]
    np.testing.assert_array_equal(fresh.probabilities, repeated.probabilities)

    with pytest.raises(ValueError, match="at least one retained"):
        forecast_grid_bayes(
            _history([[], []]),
            greedy_disjoint_rows(sparse.csr_matrix((0, 0), dtype=np.uint8)),
            config,
            interior_cells=8,
            process_cpu_deadline=_deadline(config),
        )


def test_every_arm_refuses_empty_sequences_with_a_deliberate_value_error() -> None:
    from qldpc_fno.identifiability.filters import (
        forecast_contemporaneous,
        forecast_grid_bayes,
        forecast_known_marginal,
        forecast_latent_history,
        forecast_parity_moment,
    )

    config = _config()
    empty_history = DeployableHistory(
        syndromes=np.empty((0, 3), dtype=np.uint8), scored_mask=np.empty(0, dtype=np.bool_)
    )
    for forecaster in (forecast_known_marginal, forecast_parity_moment, forecast_grid_bayes):
        arguments = {} if forecaster is forecast_known_marginal else {"interior_cells": 8}
        with pytest.raises(ValueError, match="at least one round"):
            forecaster(
                empty_history,
                _checks(),
                config,
                **arguments,
            )
    with pytest.raises(ValueError, match="at least one round"):
        forecast_latent_history(LatentHistoryOracleInput(np.empty(0)), config, interior_cells=8)
    with pytest.raises(ValueError, match="at least one round"):
        forecast_contemporaneous(ContemporaneousOracleInput(np.empty((0, 2))), config)


@pytest.mark.parametrize(
    "row_indices", [np.array([-1, 1, 2]), np.array([3, 1, 2]), np.array([1, 0, 2])]
)
def test_deployable_filters_reject_every_invalid_retained_syndrome_index(
    row_indices: np.ndarray,
) -> None:
    from qldpc_fno.identifiability.filters import forecast_grid_bayes

    with pytest.raises(ValueError, match="retained check indices"):
        forecast_grid_bayes(
            _public_history(),
            replace(_checks(), row_indices=row_indices),
            _config(),
            interior_cells=8,
        )


def test_grid_resolution_converges_without_retaining_all_prediction_arms() -> None:
    from qldpc_fno.identifiability.filters import forecast_grid_bayes, forecast_known_marginal

    config = _config()
    history = _public_history()
    deadline = _deadline(config)
    nominal = forecast_grid_bayes(
        history,
        _checks(),
        config,
        interior_cells=config.grid.interior_cells,
        process_cpu_deadline=deadline,
    )
    doubled = forecast_grid_bayes(
        history,
        _checks(),
        config,
        interior_cells=config.grid.doubled_interior_cells,
        process_cpu_deadline=deadline,
    )
    known = forecast_known_marginal(
        history, _checks(), config, process_cpu_deadline=deadline
    )
    truth = np.full(history.syndromes.shape[0], 0.05)

    def expected_ce(prediction: np.ndarray) -> float:
        return float(np.mean(-truth * np.log(prediction) - (1.0 - truth) * np.log1p(-prediction)))

    nominal_gain = expected_ce(known.probabilities) - expected_ce(nominal.probabilities)
    doubled_gain = expected_ce(known.probabilities) - expected_ce(doubled.probabilities)
    assert abs(nominal_gain - doubled_gain) < config.grid.convergence_tolerance
    assert not hasattr(nominal, "all_arms")
    assert not hasattr(nominal, "derangements")
    assert nominal.probabilities.nbytes == 8 * history.syndromes.shape[0]
