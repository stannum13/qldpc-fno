from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.training.conditional import TrainingResult, train_conditional_fno


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _config(*, epochs: int) -> CampaignConfig:
    return replace(
        CampaignConfig.from_json(Path("configs/accuracy_campaign.json")),
        training_epochs=epochs,
        training_batch_size=2,
        training_learning_rate=1e-3,
        training_seed=23,
    )


def test_checkpoint_resume_exactly_matches_uninterrupted_training(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    inputs = rng.integers(0, 2, size=(5, 22, 45), dtype=np.uint8).astype(np.float32)
    targets = rng.integers(0, 2, size=(5, 58, 45), dtype=np.uint8).astype(np.float32)
    split = np.array(["train", "train", "validation", "train", "validation"])

    uninterrupted = train_conditional_fno(
        inputs, targets, split, _config(epochs=2), tmp_path / "uninterrupted"
    )
    first_epoch = train_conditional_fno(
        inputs, targets, split, _config(epochs=1), tmp_path / "resumed"
    )
    resumed = train_conditional_fno(
        inputs, targets, split, _config(epochs=2), tmp_path / "resumed"
    )

    assert isinstance(resumed, TrainingResult)
    assert first_epoch.epoch == 1
    assert uninterrupted.epoch == resumed.epoch == 2
    assert uninterrupted.loss_history == resumed.loss_history
    assert uninterrupted.validation_nll_history == resumed.validation_nll_history
    _assert_nested_equal(uninterrupted.model_state_dict, resumed.model_state_dict)
    _assert_nested_equal(uninterrupted.optimizer_state_dict, resumed.optimizer_state_dict)
    _assert_nested_equal(uninterrupted.rng_state, resumed.rng_state)
    assert sorted(path.name for path in (tmp_path / "resumed").glob("epoch-*.pt")) == [
        "epoch-0001.pt",
        "epoch-0002.pt",
    ]


def test_checkpoint_contains_training_state_and_source_hashes(tmp_path: Path) -> None:
    inputs = np.zeros((2, 22, 45), dtype=np.float32)
    targets = np.zeros((2, 58, 45), dtype=np.float32)
    result = train_conditional_fno(
        inputs,
        targets,
        np.array(["train", "validation"]),
        _config(epochs=1),
        tmp_path,
    )

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert set(checkpoint) >= {
        "epoch",
        "hashes",
        "loss_history",
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "validation_nll_history",
    }
    assert set(checkpoint["hashes"]) == {"config", "inputs", "split", "targets"}
