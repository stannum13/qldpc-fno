from __future__ import annotations

from pathlib import Path

import pytest
import torch

from qldpc_fno.models.causal_forecaster import build_forecaster
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.training.causal_sequence import (
    SequenceRoleBatch,
    binary_forecast_metrics,
    build_overfit_fixture,
    fit_calibration_temperature,
    ideal_previous_channel_forecast,
    overfit_causal_forecaster,
    train_causal_forecaster,
)

CONFIG_PATH = Path("configs/causal_fno_hippo_reduced.json")


def _config() -> CausalExperimentConfig:
    return CausalExperimentConfig.from_json(CONFIG_PATH)


def test_overfit_fixture_is_deterministic_balanced_and_causal() -> None:
    config = _config()
    first = build_overfit_fixture(config)
    second = build_overfit_fixture(config)

    assert first.role == "overfit_fixture"
    assert first.seed == 1801
    assert first.syndromes.shape == (2, 24, 21, 45)
    assert first.targets.shape == (2, 24, 58, 45)
    assert first.scored_mask.tolist() == [False] * 8 + [True] * 16
    assert torch.equal(first.syndromes, second.syndromes)
    assert torch.equal(first.targets, second.targets)
    assert torch.all(first.syndromes == first.syndromes[..., :1])
    assert torch.all(first.syndromes == first.syndromes[:, :1])
    assert torch.all(first.syndromes[0] == 1 - first.syndromes[1])
    expected = first.syndromes[:, :-1, torch.arange(58) % 21]
    assert torch.equal(first.targets[:, 1:], expected)
    assert first.targets[:, first.scored_mask].float().mean() == 0.5


@pytest.mark.parametrize("temporal", ["fir", "hippo"])
def test_ideal_rule_represents_fixture_for_each_primary_temporal_path(temporal: str) -> None:
    fixture = build_overfit_fixture(_config())
    forecast = ideal_previous_channel_forecast(fixture, temporal=temporal)
    metrics = binary_forecast_metrics(forecast, fixture.targets, fixture.scored_mask)

    assert metrics.nll <= 0.03
    assert metrics.accuracy >= 0.995


def test_sequence_role_batch_owns_inputs_separately_from_supervision() -> None:
    fixture = build_overfit_fixture(_config())
    original_target = fixture.targets[0, 8, 0, 0].item()
    fixture.syndromes[0, 8, 0, 0] = 1 - fixture.syndromes[0, 8, 0, 0]

    assert fixture.targets[0, 8, 0, 0].item() == original_target
    assert fixture.syndromes.data_ptr() != fixture.targets.data_ptr()


def test_discovery_training_rejects_role_leakage_and_test_access() -> None:
    config = _config()
    fixture = build_overfit_fixture(config)
    train = fixture.with_role("train")
    validation = fixture.with_role("validation")
    calibration = fixture.with_role("calibration")
    model = build_forecaster(spatial="cnn", temporal="fir", config=config)

    with pytest.raises(ValueError, match="training batch must have role 'train'"):
        train_causal_forecaster(model, train=validation, validation=train, config=config)
    with pytest.raises(ValueError, match="validation batch must have role 'validation'"):
        train_causal_forecaster(model, train=train, validation=calibration, config=config)
    with pytest.raises(ValueError, match="test role is forbidden"):
        SequenceRoleBatch(
            role="test",
            seed=1,
            syndromes=fixture.syndromes,
            targets=fixture.targets,
            scored_mask=fixture.scored_mask,
        )


def test_validation_checkpoint_chooses_earliest_tied_step() -> None:
    config = _config()
    fixture = build_overfit_fixture(config)
    model = build_forecaster(spatial="cnn", temporal="fir", config=config)
    result = train_causal_forecaster(
        model,
        train=fixture.with_role("train"),
        validation=fixture.with_role("validation"),
        config=config,
        max_epochs=3,
        learning_rate=1e-12,
        tie_tolerance=1.0,
    )

    assert result.best_epoch == 0
    assert result.validation_nll_history[0] == result.best_validation_nll
    assert all(
        torch.equal(result.model_state_dict[name], model.state_dict()[name])
        for name in result.model_state_dict
    )


def test_temperature_uses_calibration_role_only() -> None:
    fixture = build_overfit_fixture(_config())
    logits = torch.where(fixture.targets.bool(), 0.5, -0.5)

    temperature = fit_calibration_temperature(logits, fixture.with_role("calibration"))
    assert temperature > 0.0
    with pytest.raises(ValueError, match="calibration batch must have role 'calibration'"):
        fit_calibration_temperature(logits, fixture.with_role("validation"))


@pytest.mark.parametrize(
    ("spatial", "temporal"),
    [("cnn", "fir"), ("fno", "fir"), ("cnn", "hippo"), ("fno", "hippo")],
)
def test_every_primary_cell_overfits_same_fixture_and_budget(
    spatial: str, temporal: str
) -> None:
    config = _config()
    fixture = build_overfit_fixture(config)
    model = build_forecaster(spatial=spatial, temporal=temporal, config=config)

    result = overfit_causal_forecaster(model, fixture=fixture, config=config)

    assert result.steps <= 2_000
    assert result.metrics.nll <= 0.03
    assert result.metrics.accuracy >= 0.995
