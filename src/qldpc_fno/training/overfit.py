from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from qldpc_fno.models.fno1d import RingFNO


def enforce_training_gates(gates: Mapping[str, bool]) -> None:
    """Fail before evaluation when any declared overfit gate is unmet."""
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise RuntimeError(f"training gates failed: {', '.join(failed)}")


def _as_binary_field(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float32, order="C", copy=True)
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (positive shots, channels, ell)")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be binary")
    return array


def predict_fno(model: RingFNO, inputs: np.ndarray) -> np.ndarray:
    """Run CPU inference from owned float32 storage and return logits."""
    input_array = np.array(inputs, dtype=np.float32, order="C", copy=True)
    if input_array.ndim != 3:
        raise ValueError("inputs must have shape (shots, channels, ell)")
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(input_array)).numpy()


def overfit_fno(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    seed: int,
) -> tuple[RingFNO, dict[str, object]]:
    """Train a deterministic full-batch FNO against teacher correction bits."""
    input_array = _as_binary_field(inputs, name="inputs")
    target_array = _as_binary_field(targets, name="targets")
    if input_array.shape[0] != target_array.shape[0]:
        raise ValueError("inputs and targets must contain equal shot counts")
    if input_array.shape[-1] != target_array.shape[-1]:
        raise ValueError("inputs and targets must share a ring length")
    if steps <= 0:
        raise ValueError("steps must be positive")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    input_tensor = torch.from_numpy(input_array)
    target_tensor = torch.from_numpy(target_array)
    model = RingFNO(
        in_channels=input_array.shape[1],
        out_channels=target_array.shape[1],
        modes=min(12, input_array.shape[-1] // 2 + 1),
    )
    positives = target_tensor.sum(dim=(0, 2))
    total_per_channel = target_tensor.shape[0] * target_tensor.shape[2]
    negatives = total_per_channel - positives
    positive_weight = (negatives / positives.clamp_min(1)).clamp(1, 100).reshape(1, -1, 1)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    with torch.no_grad():
        initial_loss = float(loss_function(model(input_tensor), target_tensor).item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(input_tensor), target_tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(input_tensor)
        final_loss = float(loss_function(logits, target_tensor).item())
        predictions = logits >= 0
        teacher_bit_accuracy = float((predictions == target_tensor.bool()).float().mean().item())
    return model, {
        "final_weighted_bce": final_loss,
        "initial_weighted_bce": initial_loss,
        "seed": seed,
        "shots": int(input_array.shape[0]),
        "steps": steps,
        "teacher_bit_accuracy": teacher_bit_accuracy,
    }
