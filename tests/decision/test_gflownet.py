from __future__ import annotations

import numpy as np
import torch

from qldpc_fno.decision.exact_css import canonical_steane_z_problem, exact_posterior
from qldpc_fno.decision.gflownet import (
    AffineTrajectoryEnvironment,
    TrajectoryBalanceGFlowNet,
    induced_terminal_probabilities,
    sample_terminal_errors,
    train_trajectory_balance,
)

FIELD = np.array(
    [
        0.026361097715701522,
        0.2836267887058967,
        0.02571400281063941,
        0.2070030417064569,
        0.13867272654068594,
        0.2934758009664783,
        0.29463747414526753,
    ]
)
SYNDROME = np.array([0, 1, 1], dtype=np.uint8)


def test_affine_environment_has_one_trajectory_per_valid_terminal() -> None:
    environment = AffineTrajectoryEnvironment(canonical_steane_z_problem(), SYNDROME, FIELD)

    errors = np.stack(
        [environment.terminal_error(coefficients) for coefficients in environment.coefficients]
    )
    assert environment.coefficients.shape == (16, 4)
    assert np.unique(errors, axis=0).shape[0] == 16
    assert np.all((errors @ environment.problem.hx.T) % 2 == SYNDROME)
    assert np.allclose(
        np.exp(environment.log_rewards),
        [np.prod(np.where(error == 1, FIELD, 1.0 - FIELD)) for error in errors],
    )


def test_trajectory_balance_overfits_one_exact_context() -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    environment = AffineTrajectoryEnvironment(canonical_steane_z_problem(), SYNDROME, FIELD)
    model = TrajectoryBalanceGFlowNet(context_width=10, coefficient_width=4, hidden_width=32)

    record = train_trajectory_balance(
        model,
        [environment],
        steps=1500,
        learning_rate=3e-3,
    )
    induced = induced_terminal_probabilities(model, environment)
    exact = exact_posterior(environment.problem, SYNDROME, FIELD)
    induced_classes = np.array(
        [induced[environment.logical_classes == logical_class].sum() for logical_class in (0, 1)]
    )

    assert record.final_loss < record.initial_loss * 1e-3
    assert record.unique_terminal_reward_evaluations == 16
    assert record.optimization_policy_action_evaluations == 16 * 4 * 1500
    assert np.isclose(induced.sum(), 1.0)
    assert 0.5 * np.abs(induced_classes - exact.logical_class_probabilities).sum() < 0.01

    first = sample_terminal_errors(model, environment, sample_count=128, seed=29)
    second = sample_terminal_errors(model, environment, sample_count=128, seed=29)
    assert np.array_equal(first, second)
    assert np.all((first @ environment.problem.hx.T) % 2 == SYNDROME)
