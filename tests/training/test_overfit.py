import warnings

import numpy as np
import pytest

from qldpc_fno.training.overfit import enforce_training_gates, overfit_fno, predict_fno


def test_overfit_is_deterministic_and_reduces_weighted_loss() -> None:
    rng = np.random.default_rng(9)
    inputs = rng.integers(0, 2, size=(4, 2, 9), dtype=np.uint8).astype(np.float32)
    targets = np.stack((inputs[:, 0], inputs[:, 1], inputs[:, 0]), axis=1)
    first_model, first = overfit_fno(inputs, targets, steps=8, seed=17)
    second_model, second = overfit_fno(inputs, targets, steps=8, seed=17)
    assert first["final_weighted_bce"] < first["initial_weighted_bce"]
    assert first == second
    assert all(
        np.array_equal(left.detach().numpy(), right.detach().numpy())
        for left, right in zip(first_model.parameters(), second_model.parameters(), strict=True)
    )


def test_overfit_safely_copies_read_only_inputs() -> None:
    inputs = np.zeros((2, 1, 5), dtype=np.float32)
    targets = np.zeros((2, 1, 5), dtype=np.float32)
    inputs.setflags(write=False)
    targets.setflags(write=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        overfit_fno(inputs, targets, steps=1, seed=3)
    assert not caught


def test_prediction_safely_copies_read_only_inputs() -> None:
    inputs = np.zeros((2, 1, 5), dtype=np.float32)
    targets = np.zeros((2, 1, 5), dtype=np.float32)
    model, _ = overfit_fno(inputs, targets, steps=1, seed=3)
    inputs.setflags(write=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logits = predict_fno(model, inputs)
    assert logits.shape == targets.shape
    assert not caught


def test_required_training_gates_fail_fast() -> None:
    with pytest.raises(RuntimeError, match="teacher_bit_accuracy_above_99_percent"):
        enforce_training_gates(
            {
                "loss_decreased": True,
                "syndrome_valid_at_least_90_percent": True,
                "teacher_bit_accuracy_above_99_percent": False,
            }
        )
