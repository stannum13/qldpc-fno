"""Trajectory-balance GFlowNet on a unique CSS affine parameterization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from qldpc_fno.decision.exact_css import ExactCssProblem, affine_fiber


@dataclass(frozen=True, slots=True)
class AffineTrajectoryEnvironment:
    """One fixed-order binary trajectory for every syndrome-valid physical error."""

    problem: ExactCssProblem
    syndrome: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        syndrome = np.asarray(self.syndrome, dtype=np.uint8)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if syndrome.shape != (self.problem.hx.shape[0],):
            raise ValueError("syndrome has the wrong check count")
        if probabilities.shape != (self.problem.n,) or not np.all(
            (probabilities > 0.0) & (probabilities < 1.0)
        ):
            raise ValueError("probabilities must contain one value in (0, 1) per qubit")
        object.__setattr__(self, "syndrome", np.array(syndrome, copy=True))
        object.__setattr__(self, "probabilities", np.array(probabilities, copy=True))

    @property
    def fiber(self):
        return affine_fiber(self.problem, self.syndrome)

    @property
    def coefficients(self) -> np.ndarray:
        return self.fiber.coefficients

    @property
    def errors(self) -> np.ndarray:
        return self.fiber.errors

    @property
    def context(self) -> np.ndarray:
        log_odds = np.log(self.probabilities) - np.log1p(-self.probabilities)
        return np.concatenate((self.syndrome.astype(np.float64), log_odds))

    @property
    def log_rewards(self) -> np.ndarray:
        errors = self.errors
        return (
            errors * np.log(self.probabilities) + (1 - errors) * np.log1p(-self.probabilities)
        ).sum(axis=1)

    @property
    def logical_classes(self) -> np.ndarray:
        return np.asarray((self.errors @ self.problem.logical_x) % 2, dtype=np.uint8)

    def terminal_error(self, coefficients: np.ndarray) -> np.ndarray:
        value = np.asarray(coefficients, dtype=np.uint8)
        if value.shape != (self.coefficients.shape[1],) or not np.all((value == 0) | (value == 1)):
            raise ValueError("coefficients must be one complete binary trajectory")
        matches = np.flatnonzero(np.all(self.coefficients == value, axis=1))
        if len(matches) != 1:
            raise AssertionError("terminal trajectory is not uniquely represented")
        return np.array(self.errors[int(matches[0])], copy=True)


class TrajectoryBalanceGFlowNet(nn.Module):
    """Conditional forward policy and partition estimator for fixed-order trajectories."""

    def __init__(self, *, context_width: int, coefficient_width: int, hidden_width: int) -> None:
        super().__init__()
        self.context_width = context_width
        self.coefficient_width = coefficient_width
        state_width = context_width + 3 * coefficient_width
        self.policy = nn.Sequential(
            nn.Linear(state_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 2),
        )
        self.log_partition = nn.Sequential(
            nn.Linear(context_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )

    def action_logits(
        self, context: torch.Tensor, coefficients: torch.Tensor, step: int
    ) -> torch.Tensor:
        if not 0 <= step < self.coefficient_width:
            raise ValueError("step lies outside the coefficient trajectory")
        mask = torch.zeros_like(coefficients)
        mask[:, :step] = 1.0
        visible = coefficients * mask
        step_encoding = torch.zeros_like(coefficients)
        step_encoding[:, step] = 1.0
        return self.policy(torch.cat((context, visible, mask, step_encoding), dim=1))

    def log_z(self, context: torch.Tensor) -> torch.Tensor:
        return self.log_partition(context).squeeze(1)


@dataclass(frozen=True, slots=True)
class TrajectoryBalanceTrainingRecord:
    initial_loss: float
    final_loss: float
    steps: int
    unique_terminal_reward_evaluations: int
    optimization_terminal_terms: int
    optimization_policy_action_evaluations: int


def _trajectory_log_probabilities(
    model: TrajectoryBalanceGFlowNet,
    contexts: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros(len(coefficients), dtype=contexts.dtype, device=contexts.device)
    for step in range(model.coefficient_width):
        logits = model.action_logits(contexts, coefficients, step)
        result += torch.log_softmax(logits, dim=1).gather(
            1, coefficients[:, step : step + 1].long()
        )[:, 0]
    return result


def _training_tensors(
    environments: list[AffineTrajectoryEnvironment],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not environments:
        raise ValueError("at least one environment is required")
    contexts = np.concatenate(
        [
            np.repeat(environment.context[None, :], len(environment.coefficients), axis=0)
            for environment in environments
        ],
        axis=0,
    )
    coefficients = np.concatenate(
        [environment.coefficients for environment in environments], axis=0
    )
    log_rewards = np.concatenate([environment.log_rewards for environment in environments], axis=0)
    return (
        torch.as_tensor(contexts, dtype=torch.float32),
        torch.as_tensor(coefficients, dtype=torch.float32),
        torch.as_tensor(log_rewards, dtype=torch.float32),
    )


def train_trajectory_balance(
    model: TrajectoryBalanceGFlowNet,
    environments: list[AffineTrajectoryEnvironment],
    *,
    steps: int,
    learning_rate: float,
) -> TrajectoryBalanceTrainingRecord:
    """Fit trajectory balance using every unique terminal in development contexts."""
    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    contexts, coefficients, log_rewards = _training_tensors(environments)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss_value() -> torch.Tensor:
        residual = (
            model.log_z(contexts)
            + _trajectory_log_probabilities(model, contexts, coefficients)
            - log_rewards
        )
        return torch.mean(residual.square())

    with torch.no_grad():
        initial_loss = float(loss_value())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(loss_value())
    return TrajectoryBalanceTrainingRecord(
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=steps,
        unique_terminal_reward_evaluations=len(coefficients),
        optimization_terminal_terms=len(coefficients) * steps,
        optimization_policy_action_evaluations=(
            len(coefficients) * steps * model.coefficient_width
        ),
    )


def induced_terminal_probabilities(
    model: TrajectoryBalanceGFlowNet,
    environment: AffineTrajectoryEnvironment,
) -> np.ndarray:
    """Enumerate the normalized terminal distribution induced by the forward policy."""
    context = torch.as_tensor(environment.context[None, :], dtype=torch.float32)
    contexts = context.repeat(len(environment.coefficients), 1)
    coefficients = torch.as_tensor(environment.coefficients, dtype=torch.float32)
    with torch.no_grad():
        log_probabilities = _trajectory_log_probabilities(model, contexts, coefficients)
    probabilities = torch.exp(log_probabilities).cpu().numpy().astype(np.float64)
    return probabilities / probabilities.sum()


def sample_terminal_errors(
    model: TrajectoryBalanceGFlowNet,
    environment: AffineTrajectoryEnvironment,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Generate syndrome-valid terminal errors from the learned forward policy."""
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (1 << 63))
    context = torch.as_tensor(environment.context[None, :], dtype=torch.float32).repeat(
        sample_count, 1
    )
    coefficients = torch.zeros((sample_count, model.coefficient_width), dtype=torch.float32)
    with torch.no_grad():
        for step in range(model.coefficient_width):
            probabilities = torch.softmax(model.action_logits(context, coefficients, step), dim=1)
            coefficients[:, step] = torch.multinomial(
                probabilities, 1, generator=generator
            ).squeeze(1)
    generators = np.vstack((environment.problem.z_stabilizers, environment.problem.logical_z))
    coefficient_values = coefficients.numpy().astype(np.uint8)
    return environment.fiber.representative ^ np.asarray(
        (coefficient_values @ generators) % 2, dtype=np.uint8
    )
