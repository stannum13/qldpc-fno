import copy

import numpy as np
import pytest

from qldpc_fno.training.calibration import (
    CALIBRATION_GRID,
    CalibrationParameters,
    CalibrationScore,
    calibrated_probabilities,
    select_calibration,
    validate_calibration_progress_rows,
)


def _progress_row(
    parameters: CalibrationParameters,
    *,
    digest: str = "a" * 64,
    latency: float = 1.25,
) -> dict[str, object]:
    return {
        "inference_latency_seconds": latency,
        "logits_sha256": digest,
        "model_checkpoint": {"path": "epoch-0001.pt", "sha256": "b" * 64},
        "model_epoch": 1,
        "parameters": {
            "alpha": parameters.alpha,
            "beta": parameters.beta,
            "temperature": parameters.temperature,
        },
        "residual": {"block_errors": 2, "invalid_count": 1, "nll": 0.5},
        "soft_prior": {"block_errors": 1, "invalid_count": 0, "nll": 0.5},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].__setitem__("inference_latency_seconds", np.nan),
        lambda rows: rows[0].__setitem__("logits_sha256", "not-a-digest"),
        lambda rows: rows[0]["soft_prior"].__setitem__("unexpected", 1),
        lambda rows: rows[0]["soft_prior"].__setitem__("block_errors", -1),
        lambda rows: rows[0]["soft_prior"].__setitem__("invalid_count", 9),
        lambda rows: rows[0]["soft_prior"].__setitem__("nll", np.inf),
        lambda rows: rows[0]["soft_prior"].__setitem__("nll", 0.25),
        lambda rows: rows[1].__setitem__("logits_sha256", "c" * 64),
        lambda rows: rows[1].__setitem__("inference_latency_seconds", 2.5),
    ],
)
def test_resumed_calibration_rejects_malformed_or_divergent_measurements(
    mutation: object,
) -> None:
    candidates = (
        CalibrationParameters(0.25, 0.0, 0.5),
        CalibrationParameters(0.25, 0.0, 1.0),
    )
    rows = [_progress_row(candidate) for candidate in candidates]
    malformed = copy.deepcopy(rows)
    mutation(malformed)

    with pytest.raises(ValueError, match="calibration progress"):
        validate_calibration_progress_rows(
            malformed,
            checkpoint_candidates=[{"epoch": 1, "path": "epoch-0001.pt", "sha256": "b" * 64}],
            candidates=candidates,
            shots=8,
        )


def test_calibrated_probabilities_apply_noise_condition_and_clip() -> None:
    logits = np.array([[[-100.0, 0.0, 100.0]]])
    rates = np.array([0.01])
    params = CalibrationParameters(alpha=1.0, beta=1.0, temperature=2.0)

    result = calibrated_probabilities(logits, rates, params)

    assert result.shape == logits.shape
    assert np.all(result >= 1e-5)
    assert np.all(result <= 1 - 1e-5)
    assert result[0, 0, 0] < result[0, 0, 1] < result[0, 0, 2]


def test_calibrated_probabilities_broadcasts_error_rates_per_shot() -> None:
    logits = np.zeros((2, 1, 1))
    parameters = CalibrationParameters(alpha=1.0, beta=1.0, temperature=1.0)

    result = calibrated_probabilities(logits, np.array([0.1, 0.2]), parameters)

    assert np.allclose(result[:, 0, 0], [0.1, 0.2])


def test_selection_prioritizes_validity_then_block_errors_then_nll() -> None:
    valid = CalibrationScore(
        parameters=CalibrationParameters(1.0, 1.0, 1.0),
        invalid_count=0,
        block_errors=2,
        nll=0.5,
    )
    invalid = CalibrationScore(
        parameters=CalibrationParameters(0.5, 0.0, 2.0),
        invalid_count=1,
        block_errors=0,
        nll=0.1,
    )

    assert select_calibration([invalid, valid]) == valid


def test_selection_uses_parameters_as_deterministic_final_tiebreakers() -> None:
    later = CalibrationScore(
        parameters=CalibrationParameters(0.5, 1.0, 1.0),
        invalid_count=0,
        block_errors=1,
        nll=0.2,
    )
    earlier = CalibrationScore(
        parameters=CalibrationParameters(0.5, 0.5, 4.0),
        invalid_count=0,
        block_errors=1,
        nll=0.2,
    )

    assert select_calibration([later, earlier]) == earlier


def test_selection_rejects_an_empty_score_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_calibration([])


@pytest.mark.parametrize("nll", [np.nan, np.inf, -np.inf])
def test_calibration_score_rejects_nonfinite_nll(nll: float) -> None:
    parameters = CalibrationParameters(1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="nll"):
        CalibrationScore(parameters, invalid_count=0, block_errors=0, nll=nll)


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_calibration_parameters_reject_nonfinite_logit_scalars(name: str) -> None:
    kwargs = {"alpha": 1.0, "beta": 1.0, "temperature": 1.0}
    kwargs[name] = np.nan

    with pytest.raises(ValueError, match=name):
        CalibrationParameters(**kwargs)


def test_constructing_or_selecting_a_nan_alpha_score_fails() -> None:
    with pytest.raises(ValueError, match="alpha"):
        CalibrationScore(CalibrationParameters(np.nan, 1.0, 1.0), 0, 0, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        select_calibration([CalibrationScore(CalibrationParameters(np.nan, 1.0, 1.0), 0, 0, 0.0)])


def test_calibration_score_rejects_negative_failure_counts() -> None:
    parameters = CalibrationParameters(1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="invalid_count"):
        CalibrationScore(parameters, invalid_count=-1, block_errors=0, nll=0.0)
    with pytest.raises(ValueError, match="block_errors"):
        CalibrationScore(parameters, invalid_count=0, block_errors=-1, nll=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("invalid_count", 1.0),
        ("block_errors", 1.0),
        ("invalid_count", True),
        ("block_errors", False),
    ],
)
def test_calibration_score_rejects_non_integer_or_boolean_failure_counts(
    field: str, value: float | bool
) -> None:
    kwargs = {"invalid_count": 0, "block_errors": 0}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        CalibrationScore(CalibrationParameters(1.0, 1.0, 1.0), nll=0.0, **kwargs)


def test_calibration_grid_contains_every_fixed_parameter_combination() -> None:
    assert len(CALIBRATION_GRID) == 48
    assert set(CALIBRATION_GRID) == {
        CalibrationParameters(alpha, beta, temperature)
        for alpha in (0.25, 0.5, 1.0, 2.0)
        for beta in (0.0, 0.5, 1.0)
        for temperature in (0.5, 1.0, 2.0, 4.0)
    }
