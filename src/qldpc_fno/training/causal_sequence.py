"""Deterministic full-sequence training for strictly causal channel forecasters."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import torch
from torch import nn

from qldpc_fno.models.causal_forecaster import CausalChannelForecaster
from qldpc_fno.models.hippo import HiPPOLegSMemory
from qldpc_fno.temporal.config import CausalExperimentConfig

_ALLOWED_ROLES = frozenset({"train", "validation", "calibration", "overfit_fixture"})


@dataclass(frozen=True, slots=True)
class SequenceRoleBatch:
    """One full-sequence batch with observations and labels in separate tensors."""

    role: str
    seed: int
    syndromes: torch.Tensor
    targets: torch.Tensor
    scored_mask: torch.Tensor
    sequence_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.role == "test":
            raise ValueError("test role is forbidden during training and calibration")
        if self.role not in _ALLOWED_ROLES:
            raise ValueError(f"unsupported sequence role: {self.role}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("batch seed must be a non-negative integer")
        if self.syndromes.ndim != 4 or self.syndromes.shape[2:] != (21, 45):
            raise ValueError("syndromes must have shape (sequences, rounds, 21, 45)")
        if self.targets.ndim != 4 or self.targets.shape[:2] != self.syndromes.shape[:2]:
            raise ValueError("targets must share sequence and round dimensions")
        if self.targets.shape[2:] != (58, 45):
            raise ValueError("targets must have shape (sequences, rounds, 58, 45)")
        if self.scored_mask.shape != (self.syndromes.shape[1],):
            raise ValueError("scored_mask must contain one Boolean per round")
        if self.scored_mask.dtype != torch.bool:
            raise ValueError("scored_mask must be Boolean")
        if not torch.all((self.syndromes == 0) | (self.syndromes == 1)):
            raise ValueError("syndromes must be binary")
        if not torch.all((self.targets == 0) | (self.targets == 1)):
            raise ValueError("targets must be binary")
        if self.syndromes.data_ptr() == self.targets.data_ptr():
            raise ValueError("observed inputs and supervision must be physically separate")
        if self.sequence_ids is None:
            object.__setattr__(
                self,
                "sequence_ids",
                tuple(f"{self.role}:{index}" for index in range(self.syndromes.shape[0])),
            )
        elif (
            len(self.sequence_ids) != self.syndromes.shape[0]
            or len(set(self.sequence_ids)) != len(self.sequence_ids)
        ):
            raise ValueError("sequence_ids must uniquely identify every complete sequence")

    def with_role(self, role: str) -> SequenceRoleBatch:
        return replace(
            self,
            role=role,
            sequence_ids=tuple(f"{role}:{index}" for index in range(self.syndromes.shape[0])),
        )


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    nll: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class OverfitResult:
    steps: int
    metrics: ForecastMetrics


@dataclass(frozen=True, slots=True)
class CausalTrainingResult:
    model_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_nll: float
    training_nll_history: tuple[float, ...]
    validation_nll_history: tuple[float, ...]


def build_overfit_fixture(config: CausalExperimentConfig) -> SequenceRoleBatch:
    """Create the frozen deterministic capacity test; this is never scientific data."""

    config.validate()
    fixture = config.overfit_fixture
    generator = torch.Generator(device="cpu")
    generator.manual_seed(fixture.seed)
    base = torch.randint(0, 2, (1, 21, 1), generator=generator, dtype=torch.float32)
    sequence_values = torch.cat((base, 1.0 - base), dim=0)
    rounds = fixture.burn_in + fixture.scored
    syndromes = sequence_values[:, None].expand(-1, rounds, -1, config.code.ell).clone()
    targets = torch.zeros(
        fixture.sequences,
        rounds,
        config.code.n // config.code.ell,
        config.code.ell,
        dtype=torch.float32,
    )
    output_source = torch.arange(targets.shape[2]) % syndromes.shape[2]
    targets[:, 1:] = syndromes[:, :-1, output_source]
    scored_mask = torch.arange(rounds) >= fixture.burn_in
    return SequenceRoleBatch(
        role="overfit_fixture",
        seed=fixture.seed,
        syndromes=syndromes,
        targets=targets,
        scored_mask=scored_mask,
    )


def ideal_previous_channel_forecast(
    batch: SequenceRoleBatch, *, temporal: str
) -> torch.Tensor:
    """Reference forecast proving the fixture is causal for each temporal contract."""

    if temporal not in {"fir", "hippo"}:
        raise ValueError("ideal fixture reference supports FIR and HiPPO")
    source = torch.arange(batch.targets.shape[2]) % batch.syndromes.shape[2]
    if temporal == "fir":
        probabilities = torch.full_like(batch.targets, 1e-6)
        copied = batch.syndromes[:, :-1, source]
        probabilities[:, 1:] = copied * (1.0 - 2e-6) + 1e-6
        return probabilities

    # A constant input is retained exactly by the zeroth Legendre coefficient.
    # Replaying the fixed recurrence makes that representability check explicit.
    memory = HiPPOLegSMemory(16, dtype=torch.float64)
    state = torch.zeros(
        batch.syndromes.shape[0],
        batch.syndromes.shape[2],
        16,
        dtype=torch.float64,
    )
    logits: list[torch.Tensor] = []
    for round_index in range(batch.syndromes.shape[1]):
        selected = state[:, source, 0]
        logits.append((-12.0 + 24.0 * selected)[..., None].expand(-1, -1, 45))
        state = memory.step(
            batch.syndromes[:, round_index, :, 0].double(),
            state,
            completed_sample_index=round_index + 1,
        )
    return torch.sigmoid(torch.stack(logits, dim=1)).to(batch.targets.dtype)


def binary_forecast_metrics(
    probabilities: torch.Tensor, targets: torch.Tensor, scored_mask: torch.Tensor
) -> ForecastMetrics:
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have equal shapes")
    selected_probabilities = probabilities[:, scored_mask]
    selected_targets = targets[:, scored_mask]
    nll = nn.functional.binary_cross_entropy(selected_probabilities, selected_targets)
    accuracy = ((selected_probabilities >= 0.5) == selected_targets.bool()).float().mean()
    return ForecastMetrics(float(nll.detach()), float(accuracy.detach()))


def _raw_sequence_logits(
    model: CausalChannelForecaster, syndromes: torch.Tensor
) -> torch.Tensor:
    """Replay the public causal transition while retaining raw sigmoid logits."""

    state = model.initial_state(
        syndromes.shape[0], device=syndromes.device, dtype=syndromes.dtype
    )
    logits: list[torch.Tensor] = []
    for round_index in range(syndromes.shape[1]):
        hidden = model.temporal.predict(state)
        logits.append(model.readout(hidden))
        state = model._update(state, syndromes[:, round_index])
    return torch.stack(logits, dim=1)


def _scored_nll(
    model: CausalChannelForecaster, batch: SequenceRoleBatch
) -> torch.Tensor:
    logits = _raw_sequence_logits(model, batch.syndromes)
    return nn.functional.binary_cross_entropy_with_logits(
        logits[:, batch.scored_mask], batch.targets[:, batch.scored_mask]
    )


def _initialize_for_training(
    model: CausalChannelForecaster, *, seed: int, base_probability: float
) -> None:
    """Initialize every learned scalar deterministically before a fresh training run."""

    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_complex():
                scale = max(parameter.shape[-2] if parameter.ndim > 1 else parameter.numel(), 1)
                parameter.real.normal_(std=scale**-0.5)
                parameter.imag.normal_(std=scale**-0.5)
            elif parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
            else:
                parameter.zero_()
        model.readout.bias.fill_(math.log(base_probability / (1.0 - base_probability)))


def _slice_batch(batch: SequenceRoleBatch, indices: torch.Tensor) -> SequenceRoleBatch:
    selected = indices.tolist()
    return SequenceRoleBatch(
        role=batch.role,
        seed=batch.seed,
        syndromes=batch.syndromes[indices],
        targets=batch.targets[indices],
        scored_mask=batch.scored_mask,
        sequence_ids=tuple(batch.sequence_ids[index] for index in selected),
    )


def overfit_causal_forecaster(
    model: CausalChannelForecaster,
    *,
    fixture: SequenceRoleBatch,
    config: CausalExperimentConfig,
) -> OverfitResult:
    """Run the explicit same-budget four-cell capacity gate."""

    if fixture.role != "overfit_fixture":
        raise ValueError("overfit gate requires the dedicated overfit_fixture role")
    _initialize_for_training(
        model,
        seed=fixture.seed,
        base_probability=config.generator.base_probability,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.overfit_fixture.learning_rate, weight_decay=0.0
    )
    final_metrics = ForecastMetrics(math.inf, 0.0)
    for step in range(1, config.overfit_fixture.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _scored_nll(model, fixture)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(_raw_sequence_logits(model, fixture.syndromes))
            final_metrics = binary_forecast_metrics(
                probabilities, fixture.targets, fixture.scored_mask
            )
        if (
            final_metrics.nll <= config.overfit_fixture.nll_threshold
            and final_metrics.accuracy >= config.overfit_fixture.accuracy_threshold
        ):
            return OverfitResult(step, final_metrics)
    return OverfitResult(config.overfit_fixture.max_steps, final_metrics)


def train_causal_forecaster(
    model: CausalChannelForecaster,
    *,
    train: SequenceRoleBatch,
    validation: SequenceRoleBatch,
    config: CausalExperimentConfig,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    tie_tolerance: float = 0.0,
) -> CausalTrainingResult:
    """Train complete sequences and select the earliest validation-NLL winner."""

    if train.role != "train":
        raise ValueError("training batch must have role 'train'")
    if validation.role != "validation":
        raise ValueError("validation batch must have role 'validation'")
    if set(train.sequence_ids) & set(validation.sequence_ids):
        raise ValueError("training and validation sequence membership must be disjoint")
    epochs = config.optimizer.max_epochs if max_epochs is None else max_epochs
    rate = config.optimizer.learning_rate if learning_rate is None else learning_rate
    if type(epochs) is not int or epochs <= 0 or rate <= 0.0 or tie_tolerance < 0.0:
        raise ValueError("training overrides are invalid")

    _initialize_for_training(
        model,
        seed=config.optimizer.training_seed,
        base_probability=config.generator.base_probability,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rate, weight_decay=config.optimizer.weight_decay
    )
    model.eval()
    with torch.no_grad():
        best_nll = float(_scored_nll(model, validation))
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    train_history: list[float] = []
    validation_history = [best_nll]
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(config.optimizer.training_seed)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(
            train.syndromes.shape[0], generator=shuffle_generator
        )
        weighted_loss = 0.0
        for offset in range(0, permutation.numel(), config.optimizer.batch_size):
            indices = permutation[offset : offset + config.optimizer.batch_size]
            complete_sequences = _slice_batch(train, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = _scored_nll(model, complete_sequences)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.gradient_norm_cap)
            optimizer.step()
            weighted_loss += float(loss.detach()) * indices.numel()
        train_history.append(weighted_loss / train.syndromes.shape[0])
        model.eval()
        with torch.no_grad():
            validation_nll = float(_scored_nll(model, validation))
        validation_history.append(validation_nll)
        if validation_nll < best_nll - tie_tolerance:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return CausalTrainingResult(
        model_state_dict=best_state,
        best_epoch=best_epoch,
        best_validation_nll=best_nll,
        training_nll_history=tuple(train_history),
        validation_nll_history=tuple(validation_history),
    )


def fit_calibration_temperature(logits: torch.Tensor, calibration: SequenceRoleBatch) -> float:
    """Fit one positive scalar temperature using calibration labels only."""

    if calibration.role != "calibration":
        raise ValueError("calibration batch must have role 'calibration'")
    if logits.shape != calibration.targets.shape:
        raise ValueError("calibration logits and targets must have equal shapes")
    log_temperature = torch.zeros((), dtype=logits.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.5, max_iter=100, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits[:, calibration.scored_mask] / temperature,
            calibration.targets[:, calibration.scored_mask],
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp())
