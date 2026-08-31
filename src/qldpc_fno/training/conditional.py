"""Deterministic mini-batch training for the noise-conditioned ring FNO."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from qldpc_fno.models.fno1d import RingFNO


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Serializable final state of a conditional training run."""

    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    rng_state: dict[str, Any]
    epoch: int
    loss_history: tuple[float, ...]
    validation_nll_history: tuple[float, ...]
    checkpoint_path: Path
    hashes: dict[str, str]


def _owned_float32(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float32, order="C", copy=True)
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (positive shots, channels, ell)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _config_value(config: object, name: str) -> object:
    if isinstance(config, Mapping):
        try:
            return config[name]
        except KeyError as error:
            raise ValueError(f"config is missing {name}") from error
    try:
        return getattr(config, name)
    except AttributeError as error:
        raise ValueError(f"config is missing {name}") from error


def _training_config(config: object) -> tuple[int, int, float, int]:
    epochs = _config_value(config, "training_epochs")
    batch_size = _config_value(config, "training_batch_size")
    learning_rate = _config_value(config, "training_learning_rate")
    seed = _config_value(config, "training_seed")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("training_epochs must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("training_batch_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("training_seed must be a non-negative integer")
    if type(learning_rate) not in (int, float) or not np.isfinite(learning_rate):
        raise ValueError("training_learning_rate must be finite and numeric")
    if learning_rate <= 0:
        raise ValueError("training_learning_rate must be positive")
    return epochs, batch_size, float(learning_rate), seed


def _split_indices(split: object, shots: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(split, Mapping):
        if "train" not in split or not ({"validation", "val"} & split.keys()):
            raise ValueError("split mapping must contain train and validation indices")
        train = np.asarray(split["train"], dtype=np.int64)
        validation = np.asarray(split.get("validation", split.get("val")), dtype=np.int64)
    else:
        labels = np.asarray(split)
        if labels.ndim != 1 or labels.shape[0] != shots:
            raise ValueError("split must contain one role per shot")
        if labels.dtype == np.bool_:
            train = np.flatnonzero(labels)
            validation = np.flatnonzero(~labels)
        else:
            train = np.flatnonzero(labels == "train")
            validation = np.flatnonzero((labels == "validation") | (labels == "val"))
            if train.size + validation.size != shots:
                raise ValueError("split roles must be train or validation")
    if train.ndim != 1 or validation.ndim != 1 or train.size == 0 or validation.size == 0:
        raise ValueError("split must contain non-empty train and validation sets")
    combined = np.concatenate((train, validation))
    if np.any(combined < 0) or np.any(combined >= shots):
        raise ValueError("split indices are out of bounds")
    if np.unique(combined).size != shots or combined.size != shots:
        raise ValueError("split must assign every shot exactly once")
    return np.ascontiguousarray(train), np.ascontiguousarray(validation)


def _source_hashes(
    inputs: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, str]:
    split_array = np.full(inputs.shape[0], 1, dtype=np.uint8)
    split_array[train] = 0
    config_payload = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model": {"depth": 2, "in_channels": 22, "modes": 12, "out_channels": 58, "width": 32},
        "seed": seed,
    }
    return {
        "config": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "inputs": _array_hash(inputs),
        "split": _array_hash(split_array),
        "targets": _array_hash(targets),
    }


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("epoch-[0-9][0-9][0-9][0-9].pt"))
    return checkpoints[-1] if checkpoints else None


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_conditional_fno(
    inputs: np.ndarray,
    targets: np.ndarray,
    split: object,
    config: object,
    checkpoint_dir: Path,
) -> TrainingResult:
    """Train or resume the fixed conditional RingFNO, checkpointing every epoch."""
    input_array = _owned_float32(inputs, name="inputs")
    target_array = _owned_float32(targets, name="targets")
    if input_array.shape[1] != 22:
        raise ValueError("inputs must contain 22 channels")
    if target_array.shape[1] != 58:
        raise ValueError("targets must contain 58 channels")
    if input_array.shape[0] != target_array.shape[0]:
        raise ValueError("inputs and targets must contain equal shot counts")
    if input_array.shape[-1] != target_array.shape[-1]:
        raise ValueError("inputs and targets must share a ring length")
    if input_array.shape[-1] // 2 + 1 < 12:
        raise ValueError("ring length is too short for 12 Fourier modes")
    if not np.all((input_array[:, :21] == 0) | (input_array[:, :21] == 1)):
        raise ValueError("input syndrome channels must be binary")
    if not np.all((target_array == 0) | (target_array == 1)):
        raise ValueError("targets must be binary")

    epochs, batch_size, learning_rate, seed = _training_config(config)
    train_indices, validation_indices = _split_indices(split, input_array.shape[0])
    hashes = _source_hashes(
        input_array,
        target_array,
        train_indices,
        validation_indices,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    shuffle_rng = np.random.default_rng(seed)
    model = RingFNO(in_channels=22, out_channels=58, width=32, modes=12, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    loss_history: list[float] = []
    validation_nll_history: list[float] = []

    latest = _latest_checkpoint(checkpoint_path)
    if latest is not None:
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["hashes"] != hashes:
            raise ValueError("checkpoint hashes do not match training inputs and configuration")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"])
        loss_history = list(saved["loss_history"])
        validation_nll_history = list(saved["validation_nll_history"])
        shuffle_rng.bit_generator.state = saved["rng_state"]["numpy"]
        torch.set_rng_state(saved["rng_state"]["torch"])
    if start_epoch > epochs:
        raise ValueError("checkpoint epoch exceeds configured training_epochs")

    input_tensor = torch.from_numpy(input_array)
    target_tensor = torch.from_numpy(target_array)
    train_targets = target_tensor[train_indices]
    positives = train_targets.sum(dim=(0, 2))
    observations = train_targets.shape[0] * train_targets.shape[2]
    negatives = observations - positives
    positive_weight = (negatives / positives.clamp_min(1)).clamp(1, 100).reshape(1, -1, 1)
    weighted_bce = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    validation_bce = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        shuffled = shuffle_rng.permutation(train_indices)
        loss_sum = 0.0
        for offset in range(0, shuffled.size, batch_size):
            batch = shuffled[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_bce(model(input_tensor[batch]), target_tensor[batch])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * batch.size
        loss_history.append(loss_sum / train_indices.size)

        model.eval()
        with torch.no_grad():
            validation_nll = validation_bce(
                model(input_tensor[validation_indices]), target_tensor[validation_indices]
            )
        validation_nll_history.append(float(validation_nll.item()))
        rng_state = {
            "numpy": copy.deepcopy(shuffle_rng.bit_generator.state),
            "torch": torch.get_rng_state().clone(),
        }
        epoch_path = checkpoint_path / f"epoch-{epoch:04d}.pt"
        _save_checkpoint(
            epoch_path,
            {
                "epoch": epoch,
                "hashes": hashes,
                "loss_history": loss_history,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": rng_state,
                "validation_nll_history": validation_nll_history,
            },
        )

    final_path = checkpoint_path / f"epoch-{epochs:04d}.pt"
    final_rng_state = {
        "numpy": copy.deepcopy(shuffle_rng.bit_generator.state),
        "torch": torch.get_rng_state().clone(),
    }
    return TrainingResult(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        rng_state=final_rng_state,
        epoch=epochs,
        loss_history=tuple(loss_history),
        validation_nll_history=tuple(validation_nll_history),
        checkpoint_path=final_path,
        hashes=hashes,
    )
