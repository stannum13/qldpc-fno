"""Strictly causal crossed spatial-temporal channel forecasters."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch
from torch import nn

from qldpc_fno.models.fno1d import RingFNOEncoder
from qldpc_fno.models.hippo import HiPPOLegSMemory
from qldpc_fno.temporal.causality import ObservedHistory
from qldpc_fno.temporal.config import CausalExperimentConfig

SpatialKind = Literal["cnn", "fno"]
TemporalKind = Literal["fir", "hippo", "gru"]


class ForecasterState(Protocol):
    completed_samples: int

    @property
    def tensor(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class FIRState:
    history: torch.Tensor
    completed_samples: int = 0

    @property
    def tensor(self) -> torch.Tensor:
        return self.history


@dataclass(frozen=True)
class HiPPOState:
    memory: torch.Tensor
    completed_samples: int = 0

    @property
    def tensor(self) -> torch.Tensor:
        return self.memory


@dataclass(frozen=True)
class GRUState:
    hidden: torch.Tensor
    completed_samples: int = 0

    @property
    def tensor(self) -> torch.Tensor:
        return self.hidden


@dataclass(frozen=True)
class PrefixShuffleIdentity:
    campaign_seed: int
    regime: str
    role: str
    sequence_index: int


@dataclass(frozen=True)
class SequenceForecast:
    probabilities: torch.Tensor
    scored_mask: torch.Tensor
    final_state: ForecasterState


def deterministic_prefix_permutation(
    available_history: int,
    *,
    identity: PrefixShuffleIdentity,
    forecast_round: int,
) -> torch.Tensor:
    """Permute only already-completed FIR rounds using a stable public identity."""

    if type(available_history) is not int or available_history < 0:
        raise ValueError("available_history must be a nonnegative integer")
    if type(forecast_round) is not int or forecast_round < 0:
        raise ValueError("forecast_round must be a nonnegative integer")
    if available_history > forecast_round:
        raise ValueError("available history cannot exceed the available prefix")
    fields = (
        identity.campaign_seed,
        identity.regime,
        identity.role,
        identity.sequence_index,
        forecast_round,
    )
    digest = hashlib.sha256("\x1f".join(map(str, fields)).encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
    return torch.randperm(available_history, generator=generator)


class CircularCNNEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                width,
                kernel_size=5,
                padding=2,
                padding_mode="circular",
            ),
            nn.GELU(),
            nn.Conv1d(
                width,
                width,
                kernel_size=5,
                padding=2,
                padding_mode="circular",
            ),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class FIRTemporal(nn.Module):
    def __init__(self, width: int, history: int) -> None:
        super().__init__()
        self.width = width
        self.history = history
        self.taps = nn.Parameter(torch.empty(width, history))
        nn.init.normal_(self.taps, std=history**-0.5)

    def initial_state(self, batch: int, ell: int, reference: torch.Tensor) -> FIRState:
        return FIRState(reference.new_zeros(batch, self.history, self.width, ell))

    def predict(self, state: FIRState) -> torch.Tensor:
        return torch.einsum("blhx,hl->bhx", state.history, self.taps)

    def update(self, state: FIRState, embedding: torch.Tensor) -> FIRState:
        history = torch.cat((state.history[:, 1:], embedding[:, None]), dim=1)
        return FIRState(history, state.completed_samples + 1)


class HiPPOTemporal(nn.Module):
    def __init__(self, width: int, order: int) -> None:
        super().__init__()
        self.width = width
        self.order = order
        self.memory = HiPPOLegSMemory(order)
        self.readout_coefficients = nn.Parameter(torch.empty(width, order))
        nn.init.normal_(self.readout_coefficients, std=order**-0.5)

    def initial_state(self, batch: int, ell: int, reference: torch.Tensor) -> HiPPOState:
        return HiPPOState(reference.new_zeros(batch, self.width, ell, self.order))

    def predict(self, state: HiPPOState) -> torch.Tensor:
        return torch.einsum("bhxn,hn->bhx", state.memory, self.readout_coefficients)

    def update(self, state: HiPPOState, embedding: torch.Tensor) -> HiPPOState:
        memory = self.memory.step(
            embedding,
            state.memory,
            completed_sample_index=state.completed_samples + 1,
        )
        return HiPPOState(memory, state.completed_samples + 1)


class GRUTemporal(nn.Module):
    def __init__(self, width: int, state_width: int) -> None:
        super().__init__()
        self.width = width
        self.state_width = state_width
        self.cell = nn.GRUCell(width, state_width)
        self.project = nn.Linear(state_width, width)
        nn.init.zeros_(self.project.bias)

    def initial_state(self, batch: int, ell: int, reference: torch.Tensor) -> GRUState:
        return GRUState(reference.new_zeros(batch, ell, self.state_width))

    def predict(self, state: GRUState) -> torch.Tensor:
        return self.project(state.hidden).transpose(1, 2)

    def update(self, state: GRUState, embedding: torch.Tensor) -> GRUState:
        batch, _, ell = embedding.shape
        inputs = embedding.transpose(1, 2).reshape(batch * ell, self.width)
        hidden = state.hidden.reshape(batch * ell, self.state_width)
        updated = self.cell(inputs, hidden).reshape(batch, ell, self.state_width)
        return GRUState(updated, state.completed_samples + 1)


class CausalChannelForecaster(nn.Module):
    """Forecast next-round channel probabilities from completed syndromes only."""

    def __init__(
        self,
        *,
        spatial_kind: SpatialKind,
        temporal_kind: TemporalKind,
        config: CausalExperimentConfig,
    ) -> None:
        super().__init__()
        model = config.model
        self.spatial_kind = spatial_kind
        self.temporal_kind = temporal_kind
        self.input_channels = 21
        self.ell = config.code.ell
        self.output_channels = config.code.n // config.code.ell
        self.width = model.hidden_width
        self.minimum_probability = config.generator.min_probability
        self.maximum_probability = config.generator.max_probability

        if spatial_kind == "cnn":
            self.spatial = CircularCNNEncoder(self.input_channels, self.width)
        elif spatial_kind == "fno":
            self.spatial = RingFNOEncoder(
                self.input_channels,
                self.width,
                model.fno_modes,
                depth=2,
            )
        else:
            raise ValueError("spatial must be 'cnn' or 'fno'")

        if temporal_kind == "fir":
            self.temporal = FIRTemporal(self.width, model.fir_history)
        elif temporal_kind == "hippo":
            self.temporal = HiPPOTemporal(self.width, model.hippo_order)
        elif temporal_kind == "gru":
            self.temporal = GRUTemporal(self.width, model.gru_state_width)
        else:
            raise ValueError("temporal must be 'fir', 'hippo', or 'gru'")

        self.readout = nn.Conv1d(self.width, self.output_channels, kernel_size=1)
        initial_logit = torch.logit(torch.tensor(config.generator.base_probability))
        nn.init.constant_(self.readout.bias, float(initial_logit))

    def _reference(self) -> torch.Tensor:
        return next(self.parameters())

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ForecasterState:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        parameter = self._reference()
        reference = torch.empty(
            (),
            device=device or parameter.device,
            dtype=dtype or parameter.dtype,
        )
        return self.temporal.initial_state(batch_size, self.ell, reference)

    def _validate_syndrome(self, state: ForecasterState, syndrome: torch.Tensor) -> None:
        expected = (state.tensor.shape[0], self.input_channels, self.ell)
        if syndrome.shape != expected:
            raise ValueError(f"syndrome_t must have shape {expected}")
        if syndrome.device != state.tensor.device or syndrome.dtype != state.tensor.dtype:
            raise ValueError("syndrome_t and state must have the same dtype and device")

    def _probabilities(self, state: ForecasterState) -> torch.Tensor:
        hidden = self.temporal.predict(state)
        logits = self.readout(hidden)
        probabilities = torch.sigmoid(logits).clamp(
            min=self.minimum_probability,
            max=self.maximum_probability,
        )
        return probabilities.reshape(probabilities.shape[0], -1)

    def _update(self, state: ForecasterState, syndrome: torch.Tensor) -> ForecasterState:
        embedding = self.spatial(syndrome)
        return self.temporal.update(state, embedding)

    def predict_then_update(
        self,
        state: ForecasterState,
        syndrome_t: torch.Tensor,
    ) -> tuple[torch.Tensor, ForecasterState]:
        self._validate_syndrome(state, syndrome_t)
        prediction = self._probabilities(state)
        return prediction, self._update(state, syndrome_t)

    def _shuffled_fir_state(
        self,
        state: FIRState,
        identity: PrefixShuffleIdentity,
        forecast_round: int,
    ) -> FIRState:
        available = min(state.completed_samples, self.temporal.history)
        permutation = deterministic_prefix_permutation(
            available,
            identity=identity,
            forecast_round=forecast_round,
        ).to(state.history.device)
        if available == 0:
            return state
        start = self.temporal.history - available
        history = state.history.clone()
        history[:, start:] = history[:, start:][:, permutation]
        return FIRState(history, state.completed_samples)

    def predict_sequence(
        self,
        syndromes: torch.Tensor,
        *,
        burn_in: int = 0,
        prefix_shuffle_identities: Sequence[PrefixShuffleIdentity] | None = None,
    ) -> SequenceForecast:
        if syndromes.ndim != 4:
            raise ValueError("syndromes must have shape (batch, time, 21, ell)")
        if not 0 <= burn_in <= syndromes.shape[1]:
            raise ValueError("burn_in must lie within the sequence")
        if prefix_shuffle_identities is not None:
            if self.temporal_kind != "fir":
                raise ValueError("prefix shuffle is an FIR-only diagnostic")
            if len(prefix_shuffle_identities) != syndromes.shape[0]:
                raise ValueError("one prefix-shuffle identity is required per sequence")
        state = self.initial_state(
            syndromes.shape[0], device=syndromes.device, dtype=syndromes.dtype
        )
        predictions: list[torch.Tensor] = []
        for round_index in range(syndromes.shape[1]):
            if prefix_shuffle_identities is None:
                prediction = self._probabilities(state)
            else:
                per_sequence = [
                    self._probabilities(
                        self._shuffled_fir_state(
                            FIRState(
                                state.history[index : index + 1],
                                state.completed_samples,
                            ),
                            identity,
                            round_index,
                        )
                    )
                    for index, identity in enumerate(prefix_shuffle_identities)
                ]
                prediction = torch.cat(per_sequence)
            predictions.append(prediction)
            state = self._update(state, syndromes[:, round_index])
        probabilities = torch.stack(predictions, dim=1)
        scored_mask = torch.arange(syndromes.shape[1], device=syndromes.device) >= burn_in
        return SequenceForecast(probabilities, scored_mask, state)

    def forecast(self, history: ObservedHistory) -> np.ndarray:
        """Replay a strict observed prefix and forecast its next round."""

        parameter = self._reference()
        syndromes = torch.as_tensor(
            np.array(history.syndromes, copy=True),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        if syndromes.ndim != 3:
            raise ValueError("observed history must have shape (time, 21, ell)")
        state = self.initial_state(1)
        with torch.no_grad():
            for syndrome in syndromes:
                state = self._update(state, syndrome.unsqueeze(0))
            prediction = self._probabilities(state)
        return prediction.detach().cpu().numpy()[0]


def build_forecaster(
    *,
    spatial: str,
    temporal: str,
    config: CausalExperimentConfig,
) -> CausalChannelForecaster:
    """Build one cell of the crossed architecture with shared geometry."""

    if spatial not in ("cnn", "fno"):
        raise ValueError("spatial must be 'cnn' or 'fno'")
    if temporal not in ("fir", "hippo", "gru"):
        raise ValueError("temporal must be 'fir', 'hippo', or 'gru'")
    return CausalChannelForecaster(
        spatial_kind=spatial,
        temporal_kind=temporal,
        config=config,
    )


def trainable_parameter_count(module: nn.Module) -> int:
    """Count independent real trainable scalars (complex values count twice)."""

    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
