"""Time-varying HiPPO-LegS memory with the original bilinear update."""

from __future__ import annotations

import torch
from torch import nn


def _validate_sample_index(value: int, *, name: str, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _legs_generator(
    order: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if order <= 0:
        raise ValueError("order must be positive")
    indices = torch.arange(order, dtype=dtype, device=device)
    scale = torch.sqrt(2 * indices + 1)
    lower = -scale[:, None] * scale[None, :]
    generator = torch.tril(lower, diagonal=-1) + torch.diag(-(indices + 1))
    return generator, scale


def _discretize(
    generator: torch.Tensor,
    input_vector: torch.Tensor,
    completed_sample_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = completed_sample_indices[..., None, None]
    scaled_generator = generator / scale
    scaled_input = input_vector / completed_sample_indices[..., None]
    identity = torch.eye(
        generator.shape[0], dtype=generator.dtype, device=generator.device
    ).expand(scaled_generator.shape)
    lhs = identity - scaled_generator / 2
    transition = torch.linalg.solve(lhs, identity + scaled_generator / 2)
    injection = torch.linalg.solve(lhs, scaled_input.unsqueeze(-1)).squeeze(-1)
    return transition, injection


def legs_transition(
    order: int,
    step: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the LegS bilinear transition for completed sample index ``step``."""

    _validate_sample_index(step, name="step", minimum=1)
    generator, input_vector = _legs_generator(order, dtype=dtype)
    indices = torch.tensor(step, dtype=dtype)
    return _discretize(generator, input_vector, indices)


class HiPPOLegSMemory(nn.Module):
    """Apply faithful time-varying LegS memory independently over a scalar field.

    Inputs have shape ``(batch, time, ...)``. Each scalar site owns an order-sized
    state, while all sites share the same fixed time-varying transition.
    """

    def __init__(self, order: int, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        resolved_dtype = dtype or torch.get_default_dtype()
        generator, input_vector = _legs_generator(order, dtype=resolved_dtype)
        self.order = order
        self.register_buffer("generator", generator)
        self.register_buffer("input_vector", input_vector)

    def step(
        self,
        value: torch.Tensor,
        state: torch.Tensor,
        completed_sample_index: int,
    ) -> torch.Tensor:
        """Advance one completed sample without constructing per-site matrices."""

        _validate_sample_index(
            completed_sample_index, name="completed_sample_index", minimum=1
        )
        if state.shape != (*value.shape, self.order):
            raise ValueError("state shape must equal value shape followed by order")
        if value.dtype != state.dtype or value.device != state.device:
            raise ValueError("value and state must have the same dtype and device")

        generator = self.generator.to(dtype=state.dtype, device=state.device)
        input_vector = self.input_vector.to(dtype=state.dtype, device=state.device)
        index = torch.tensor(completed_sample_index, dtype=state.dtype, device=state.device)
        transition, injection = _discretize(generator, input_vector, index)
        return torch.einsum("ij,...j->...i", transition, state) + value[..., None] * injection

    def forward(
        self,
        values: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        start_completed_sample_index: int = 0,
        return_sequence: bool = True,
    ) -> torch.Tensor:
        """Encode a sequence whose time axis is dimension one.

        ``start_completed_sample_index`` preserves the time-varying measure when
        continuing from ``initial_state`` across streaming chunks.
        """

        _validate_sample_index(
            start_completed_sample_index,
            name="start_completed_sample_index",
            minimum=0,
        )
        if (initial_state is None) != (start_completed_sample_index == 0):
            raise ValueError(
                "initial_state and start_completed_sample_index must be provided together"
            )
        if values.ndim < 2:
            raise ValueError("values must have shape (batch, time, ...)")
        expected_state_shape = (*values.shape[:1], *values.shape[2:], self.order)
        if initial_state is None:
            state = values.new_zeros(expected_state_shape)
        else:
            if initial_state.shape != expected_state_shape:
                raise ValueError(f"initial_state must have shape {expected_state_shape}")
            if initial_state.dtype != values.dtype or initial_state.device != values.device:
                raise ValueError("initial_state and values must have the same dtype and device")
            state = initial_state

        generator = self.generator.to(dtype=values.dtype, device=values.device)
        input_vector = self.input_vector.to(dtype=values.dtype, device=values.device)
        indices = torch.arange(
            start_completed_sample_index + 1,
            start_completed_sample_index + values.shape[1] + 1,
            dtype=values.dtype,
            device=values.device,
        )
        transitions, injections = _discretize(generator, input_vector, indices)

        states: list[torch.Tensor] = []
        for offset in range(values.shape[1]):
            state = torch.einsum("ij,...j->...i", transitions[offset], state)
            state = state + values[:, offset, ..., None] * injections[offset]
            if return_sequence:
                states.append(state)

        if return_sequence:
            if not states:
                return values.new_empty((*values.shape, self.order))
            return torch.stack(states, dim=1)
        return state
