from __future__ import annotations

import math

import numpy as np
import pytest

from qldpc_fno.identifiability.endpoints import (
    calibration_by_sequence,
    expected_ce_by_sequence,
    latent_nmse_by_sequence,
    retained_syndrome_nll_by_sequence,
)
from qldpc_fno.identifiability.observation import DisjointChecks
from qldpc_fno.identifiability.types import SequenceIdentity, TrainingTargets


def _checks() -> DisjointChecks:
    return DisjointChecks(
        np.array([0, 1]), (np.array([0, 1]), np.array([2])), np.array([2, 1]),
        np.array([0, 1, 2]), "greedy_disjoint_rows/v1", "a" * 64,
    )


def _identities(count: int = 1, role: str = "test") -> tuple[SequenceIdentity, ...]:
    return tuple(SequenceIdentity("stationary_iid", role, index, index + 1, index + 11, f"{index + 1:x}" * 64) for index in range(count))


def test_expected_ce_uses_latent_probabilities_and_only_scored_rounds() -> None:
    q = np.array([[[0.1], [0.2], [0.3]], [[0.4], [0.2], [0.1]]])
    q_hat = np.array([[[0.2], [0.25], [0.25]], [[0.2], [0.2], [0.2]]])
    mask = np.array([[False, True, True], [False, True, True]])

    observed = expected_ce_by_sequence(q, q_hat, mask, identities=_identities(2))

    expected = np.array(
        [
            np.mean(-q[0, 1:] * np.log(q_hat[0, 1:]) - (1 - q[0, 1:]) * np.log1p(-q_hat[0, 1:])),
            np.mean(-q[1, 1:] * np.log(q_hat[1, 1:]) - (1 - q[1, 1:]) * np.log1p(-q_hat[1, 1:])),
        ]
    )
    assert np.allclose(observed, expected)


def test_latent_nmse_maps_mean_prediction_back_to_clipped_log_odds() -> None:
    base = 0.0375
    q_hat = np.array([[[base], [0.1]]])
    latent = np.array([[0.0, 0.5]])
    mask = np.array([[True, True]])

    observed = latent_nmse_by_sequence(latent, q_hat, mask, identities=_identities())

    mapped = np.clip(
        np.log(q_hat / (1 - q_hat)) - math.log(base / (1 - base)), -1.2, 1.2
    ).mean(axis=2)
    expected = np.mean((latent - mapped) ** 2, axis=1) / (0.08**2 / (1 - 0.97**2))
    assert np.allclose(observed, expected)


def test_latent_nmse_averages_the_mapped_per_qubit_field() -> None:
    latent = np.array([[0.0]])
    q_hat = np.array([[[0.0375, 0.1]]])

    observed = latent_nmse_by_sequence(latent, q_hat, np.array([[True]]), identities=_identities())

    logit = np.log(q_hat / (1 - q_hat)) - math.log(0.0375 / (1 - 0.0375))
    expected = ((-np.mean(logit)) ** 2) / (0.08**2 / (1 - 0.97**2))
    assert observed[0] == pytest.approx(expected)


def test_retained_syndrome_nll_uses_predictive_parity_probabilities() -> None:
    syndromes = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
    q_hat = np.array([[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]])
    mask = np.array([[False, True]])
    supports = _checks()

    observed = retained_syndrome_nll_by_sequence(syndromes, q_hat, mask, supports, identities=_identities())

    parity = np.array([(1 - (1 - 2 * 0.2) ** 2) / 2, 0.2])
    expected = np.array([-np.log(parity[0]) - np.log1p(-parity[1])]) / 2
    assert np.allclose(observed, expected)


def test_retained_syndrome_nll_accepts_one_scalar_forecast_per_round() -> None:
    syndromes = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)
    q_hat = np.array([0.1, 0.2])

    observed = retained_syndrome_nll_by_sequence(
        syndromes, q_hat, np.array([False, True]), _checks(), identities=_identities()
    )

    assert observed.shape == (1,)


def test_retained_syndrome_nll_rejects_overlapping_supports() -> None:
    with pytest.raises(TypeError, match="DisjointChecks"):
        retained_syndrome_nll_by_sequence(
            np.array([[[1, 0]]], dtype=np.uint8),
            np.array([[[0.2, 0.2]]]),
            np.array([[True]]),
            ((0, 1), (0,)), identities=_identities(),
        )


def test_calibration_records_fixed_bins_and_latent_not_sampled_error() -> None:
    q = np.array([[[1e-5], [0.025], [0.25]]])
    q_hat = np.array([[[1e-5], [0.025], [0.25]]])
    evidence = calibration_by_sequence(q, q_hat, np.array([[True, True, True]]), identities=_identities())

    assert evidence.counts.shape == (1, 10)
    assert evidence.counts[0, 0] == 2
    assert evidence.counts[0, -1] == 1
    assert evidence.predicted_sums[0, 0] == pytest.approx(1e-5 + 0.025)
    assert evidence.latent_sums[0, -1] == pytest.approx(0.25)
    assert evidence.absolute_error[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "function,args",
    [
        (expected_ce_by_sequence, (np.full((1, 2, 1), 0.1), np.full((1, 2, 1), 0.1), np.array([[True, False]]))),
        (latent_nmse_by_sequence, (np.zeros((1, 2)), np.full((1, 2, 1), 0.1), np.array([[True, False]]))),
    ],
)
def test_primary_endpoints_reject_incomplete_scored_masks(function, args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="scored"):
        function(*args, identities=_identities())


def test_endpoint_rejects_sampled_physical_error_container() -> None:
    targets = TrainingTargets(np.zeros((2, 1), dtype=np.uint8), np.zeros((2, 1), dtype=np.uint8))
    with pytest.raises(TypeError, match="sampled"):
        expected_ce_by_sequence(targets, np.full((1, 2, 1), 0.1), np.array([[True, True]]), identities=_identities())
