import numpy as np
import pytest

from qldpc_fno.temporal.baselines import (
    DEFAULT_DECAYS,
    DEFAULT_L2_GRID,
    CircularLogisticForecaster,
    PrivilegedOracle,
    StationaryForecaster,
    build_ewma_histories,
    build_lagged_histories,
    calibrate_temperature,
    fit_ewma,
    fit_logistic_ar,
    fit_stationary,
    register_deployable,
)


def _mask(sequences: int, rounds: int, burn_in: int = 0) -> np.ndarray:
    mask = np.ones((sequences, rounds), dtype=bool)
    mask[:, :burn_in] = False
    return mask


def test_stationary_uses_training_targets_and_validation_selects_lambda() -> None:
    train = np.array([[[[0, 1]], [[0, 1]]]], dtype=np.float64)
    validation = np.array([[[[0, 0]], [[0, 0]]]], dtype=np.float64)

    model = fit_stationary(
        train,
        validation,
        train_mask=_mask(1, 2),
        validation_mask=_mask(1, 2),
        lambda_grid=(0.0, 1.0),
    )

    assert model.scalar_probability == pytest.approx(0.5)
    assert np.array_equal(model.empirical_field, np.array([[0.0, 1.0]]))
    assert model.shrinkage == 0.0


def test_stationary_lambda_ties_choose_smaller_value() -> None:
    train = np.array([[[[0, 0]]]], dtype=np.float64)

    model = fit_stationary(
        train,
        train,
        train_mask=_mask(1, 1),
        validation_mask=_mask(1, 1),
        lambda_grid=(0.75, 0.25),
    )

    assert model.shrinkage == 0.25


def test_stationary_excludes_burn_in_targets_from_fit() -> None:
    train = np.array([[[[1.0]], [[0.0]], [[0.0]]]])
    validation = np.zeros((1, 1, 1, 1), dtype=np.float64)

    model = fit_stationary(
        train,
        validation,
        train_mask=_mask(1, 3, burn_in=1),
        validation_mask=_mask(1, 1),
    )

    assert model.scalar_probability == 0.0
    assert model.empirical_field[0, 0] == 0.0


def test_stationary_predicts_requested_rounds_without_observation_input() -> None:
    model = StationaryForecaster(
        scalar_probability=0.1,
        empirical_field=np.array([[0.0, 0.2]]),
        shrinkage=0.5,
    )

    predictions = model.predict(sequence_count=2, rounds=3)

    assert predictions.shape == (2, 3, 1, 2)
    assert np.allclose(predictions[0, 0], [[0.05, 0.15]])


def test_ewma_prediction_precedes_update_and_state_resets_per_sequence() -> None:
    syndromes = np.array(
        [
            [[[[1.0, 0.0]]], [[[0.0, 1.0]]], [[[0.0, 0.0]]]],
            [[[[0.0, 1.0]]], [[[0.0, 0.0]]], [[[0.0, 0.0]]]],
        ]
    ).reshape(2, 3, 1, 2)

    histories = build_ewma_histories(syndromes, decay=0.5)

    assert np.all(histories[:, 0] == 0.0)
    assert np.allclose(histories[0, 1], [[0.5, 0.0]])
    assert np.allclose(histories[0, 2], [[0.25, 0.5]])
    assert np.allclose(histories[1, 1], [[0.0, 0.5]])


def test_ewma_burn_in_updates_state_even_when_not_scored() -> None:
    syndromes = np.array([[[[1.0]], [[0.0]], [[0.0]]]])
    histories = build_ewma_histories(syndromes, decay=0.8)

    assert histories[0, 1, 0, 0] == pytest.approx(0.2)
    assert histories[0, 2, 0, 0] == pytest.approx(0.16)


def test_lagged_history_is_zero_padded_and_excludes_current_round() -> None:
    syndromes = np.array([[[[1.0]], [[2.0]], [[3.0]]]])

    histories = build_lagged_histories(syndromes, lags=2)

    assert histories.shape == (1, 3, 2, 1)
    assert np.array_equal(histories[0, 0, :, 0], [0.0, 0.0])
    assert np.array_equal(histories[0, 1, :, 0], [1.0, 0.0])
    assert np.array_equal(histories[0, 2, :, 0], [2.0, 1.0])


@pytest.mark.parametrize(
    ("feature_kind", "decay", "lags", "input_channels"),
    [("ewma", 0.5, 1, 1), ("lagged", None, 2, 2)],
)
def test_current_syndrome_cannot_change_current_prediction(
    feature_kind: str, decay: float | None, lags: int, input_channels: int
) -> None:
    weight = np.ones((1, input_channels, 3), dtype=np.float64)
    model = CircularLogisticForecaster(
        weight,
        np.zeros(1),
        1e-4,
        feature_kind=feature_kind,
        decay=decay,
        lags=lags,
    )
    first = np.zeros((1, 3, 1, 5), dtype=np.float64)
    second = first.copy()
    second[:, 1] = 1.0

    first_prediction = model.raw_predict(first)
    second_prediction = model.raw_predict(second)

    assert np.array_equal(first_prediction[:, 1], second_prediction[:, 1])
    assert not np.array_equal(first_prediction[:, 2], second_prediction[:, 2])


def test_temperature_uses_calibration_targets_and_changes_probabilities() -> None:
    probabilities = np.full((1, 4, 1, 1), 0.1)
    calibration_targets = np.ones_like(probabilities)

    temperature = calibrate_temperature(
        probabilities,
        calibration_targets,
        _mask(1, 4),
    )

    assert temperature > 1.0


def test_temperature_calibration_does_not_refit_forecaster_weights() -> None:
    rng = np.random.default_rng(4)
    train_syndromes = rng.integers(0, 2, size=(2, 5, 1, 5)).astype(np.float64)
    train_targets = rng.integers(0, 2, size=(2, 5, 1, 5)).astype(np.float64)
    validation_syndromes = np.zeros_like(train_syndromes)
    validation_targets = np.zeros_like(train_targets)
    calibration_targets = np.ones_like(train_targets)

    uncalibrated = fit_logistic_ar(
        train_syndromes,
        train_targets,
        validation_syndromes,
        validation_targets,
        train_mask=_mask(2, 5),
        validation_mask=_mask(2, 5),
        lags=2,
        kernel_size=3,
        l2_grid=(1e-2,),
        max_iter=20,
        calibrate=None,
    )
    calibrated = fit_logistic_ar(
        train_syndromes,
        train_targets,
        validation_syndromes,
        validation_targets,
        train_mask=_mask(2, 5),
        validation_mask=_mask(2, 5),
        lags=2,
        kernel_size=3,
        l2_grid=(1e-2,),
        max_iter=20,
        calibrate=(train_syndromes, calibration_targets, _mask(2, 5)),
    )

    assert np.array_equal(uncalibrated.weight, calibrated.weight)
    assert np.array_equal(uncalibrated.bias, calibrated.bias)
    assert calibrated.temperature != 1.0


def test_validation_targets_select_hyperparameters_but_do_not_fit_weights() -> None:
    rng = np.random.default_rng(40)
    train_syndromes = rng.integers(0, 2, size=(2, 4, 1, 5)).astype(np.float64)
    train_targets = rng.integers(0, 2, size=(2, 4, 1, 5)).astype(np.float64)
    validation_syndromes = rng.integers(0, 2, size=(1, 4, 1, 5)).astype(np.float64)
    validation_zero = np.zeros((1, 4, 1, 5), dtype=np.float64)
    validation_one = np.ones((1, 4, 1, 5), dtype=np.float64)
    kwargs = {
        "train_mask": _mask(2, 4),
        "validation_mask": _mask(1, 4),
        "lags": 2,
        "kernel_size": 3,
        "l2_grid": (1e-2,),
        "max_iter": 20,
        "calibrate": None,
    }

    zero_model = fit_logistic_ar(
        train_syndromes,
        train_targets,
        validation_syndromes,
        validation_zero,
        **kwargs,
    )
    one_model = fit_logistic_ar(
        train_syndromes,
        train_targets,
        validation_syndromes,
        validation_one,
        **kwargs,
    )

    assert np.array_equal(zero_model.weight, one_model.weight)
    assert np.array_equal(zero_model.bias, one_model.bias)


def test_logistic_ar_l2_ties_choose_smaller_value_deterministically() -> None:
    zeros_s = np.zeros((1, 3, 1, 3), dtype=np.float64)
    zeros_y = np.zeros((1, 3, 1, 3), dtype=np.float64)

    first = fit_logistic_ar(
        zeros_s,
        zeros_y,
        zeros_s,
        zeros_y,
        train_mask=_mask(1, 3),
        validation_mask=_mask(1, 3),
        lags=2,
        kernel_size=3,
        l2_grid=(1e-1, 1e-4),
        max_iter=5,
        calibrate=None,
    )
    second = fit_logistic_ar(
        zeros_s,
        zeros_y,
        zeros_s,
        zeros_y,
        train_mask=_mask(1, 3),
        validation_mask=_mask(1, 3),
        lags=2,
        kernel_size=3,
        l2_grid=(1e-4, 1e-1),
        max_iter=5,
        calibrate=None,
    )

    assert first.l2 == second.l2 == 1e-4
    assert np.array_equal(first.weight, second.weight)
    assert np.array_equal(first.bias, second.bias)


def test_fitted_probabilities_are_clipped_to_declared_bounds() -> None:
    model = StationaryForecaster(
        scalar_probability=0.5,
        empirical_field=np.array([[0.0, 1.0]]),
        shrinkage=1.0,
        temperature=0.01,
    )

    predictions = model.predict(sequence_count=1, rounds=1)

    assert predictions.min() == pytest.approx(1e-5)
    assert predictions.max() == pytest.approx(0.25)


def test_ewma_selects_only_from_declared_grids() -> None:
    rng = np.random.default_rng(9)
    syndromes = rng.integers(0, 2, size=(1, 4, 1, 5)).astype(np.float64)
    targets = rng.integers(0, 2, size=(1, 4, 1, 5)).astype(np.float64)

    model = fit_ewma(
        syndromes,
        targets,
        syndromes,
        targets,
        train_mask=_mask(1, 4),
        validation_mask=_mask(1, 4),
        decays=DEFAULT_DECAYS[:2],
        l2_grid=DEFAULT_L2_GRID[:2],
        kernel_size=3,
        max_iter=5,
        calibrate=None,
    )

    assert model.decay in DEFAULT_DECAYS[:2]
    assert model.l2 in DEFAULT_L2_GRID[:2]


def test_target_shape_mismatch_is_rejected_before_fitting() -> None:
    syndromes = np.zeros((1, 2, 1, 3))
    targets = np.zeros((1, 1, 1, 3))

    with pytest.raises(ValueError, match="sequence and round dimensions"):
        fit_logistic_ar(
            syndromes,
            targets,
            syndromes,
            targets,
            train_mask=_mask(1, 2),
            validation_mask=_mask(1, 2),
            max_iter=1,
            calibrate=None,
        )


def test_privileged_oracle_is_rejected_by_deployable_registry() -> None:
    oracle = PrivilegedOracle(np.full((1, 2, 1, 3), 0.1))

    with pytest.raises(ValueError, match="privileged latent oracle"):
        register_deployable("oracle", oracle)


def test_deployable_registry_accepts_observation_only_forecaster() -> None:
    model = StationaryForecaster(0.1, np.full((1, 3), 0.1), 0.0)

    registry = register_deployable("stationary", model)

    assert registry["stationary"] is model
