from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as student_t

from qldpc_fno.metrics.clustered import (
    cluster_percentile_interval,
    paired_sequence_inference,
)


def _differences() -> np.ndarray:
    pattern = np.array([-0.04, -0.03, -0.02, -0.01, 0.005, 0.01, -0.015, -0.025])
    return np.tile(pattern, 8)


def test_paired_sequence_inference_uses_lower_tail_t_test() -> None:
    differences = _differences()

    result = paired_sequence_inference(
        differences,
        mu0=0.0,
        seeds=(20260904,),
        draws=2_000,
    )

    statistic = differences.mean() / (differences.std(ddof=1) / np.sqrt(differences.size))
    assert result["status"] == "ok"
    assert result["n_sequences"] == 64
    assert result["t_statistic"] == pytest.approx(statistic)
    assert result["paired_t_pvalue_lower"] == pytest.approx(
        student_t.cdf(statistic, df=differences.size - 1)
    )
    assert result["wild_bootstrap"][0]["pvalue_lower"] <= 1.0


def test_centered_studentized_wild_bootstrap_is_deterministic_and_finite_corrected() -> None:
    differences = _differences()
    first = paired_sequence_inference(
        differences,
        mu0=0.0,
        seeds=(77, 78),
        draws=257,
    )
    second = paired_sequence_inference(
        differences,
        mu0=0.0,
        seeds=(77, 78),
        draws=257,
    )

    assert first == second
    for bootstrap in first["wild_bootstrap"]:
        assert bootstrap["status"] == "ok"
        assert bootstrap["valid_draws"] == 257
        pvalue = bootstrap["pvalue_lower"]
        assert isinstance(pvalue, float)
        assert pvalue >= 1 / 258
        assert (pvalue * 258).is_integer()


def test_observed_zero_variance_declares_both_inferences_degenerate() -> None:
    result = paired_sequence_inference(
        np.full(64, -0.01),
        mu0=0.0,
        seeds=(1,),
        draws=100,
    )

    assert result["status"] == "degenerate_variance"
    assert result["paired_t_status"] == "degenerate_variance"
    assert result["paired_t_pvalue_lower"] is None
    assert result["wild_bootstrap"] == [
        {
            "attempted_draws": 0,
            "pvalue_lower": None,
            "seed": 1,
            "status": "degenerate_variance",
            "valid_draws": 0,
        }
    ]


def test_wild_bootstrap_declares_exhausted_valid_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    class ConstantSigns:
        def choice(self, choices: tuple[int, int], *, size: tuple[int, int]) -> np.ndarray:
            assert choices == (-1, 1)
            signs = np.tile(np.array([-1, 1], dtype=np.int8), size[1] // 2)
            return np.broadcast_to(signs, size).copy()

    monkeypatch.setattr(np.random, "default_rng", lambda seed: ConstantSigns())
    result = paired_sequence_inference(
        np.tile(np.array([-1.0, 1.0]), 32),
        mu0=0.0,
        seeds=(9,),
        draws=3,
    )

    assert result["status"] == "bootstrap_degenerate"
    assert result["wild_bootstrap"] == [
        {
            "attempted_draws": 30,
            "pvalue_lower": None,
            "seed": 9,
            "status": "bootstrap_degenerate",
            "valid_draws": 0,
        }
    ]


def test_cluster_percentile_interval_is_whole_sequence_and_deterministic() -> None:
    values = np.array([0.1, 0.2, 0.4, 0.8])
    first = cluster_percentile_interval(values, seed=20260905, draws=500)
    second = cluster_percentile_interval(values, seed=20260905, draws=500)

    assert first == second
    assert first["n_sequences"] == 4
    assert first["estimate"] == pytest.approx(0.375)
    assert first["confidence_level"] == 0.95
    assert first["interval_low"] <= first["estimate"] <= first["interval_high"]


def test_duplicating_rounds_does_not_change_inferential_unit_count() -> None:
    per_sequence_rounds = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    duplicated_rounds = np.repeat(per_sequence_rounds, 5, axis=1)
    original_means = per_sequence_rounds.mean(axis=1)
    duplicated_means = duplicated_rounds.mean(axis=1)

    original = cluster_percentile_interval(original_means, seed=12, draws=100)
    duplicated = cluster_percentile_interval(duplicated_means, seed=12, draws=100)

    assert duplicated_rounds.size == 5 * per_sequence_rounds.size
    assert original == duplicated
    assert duplicated["n_sequences"] == 3


@pytest.mark.parametrize(
    "values",
    [np.array([]), np.zeros((2, 2)), np.array([0.0, np.nan])],
)
def test_clustered_statistics_reject_incomplete_or_non_sequence_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        cluster_percentile_interval(values, seed=1, draws=10)


def test_paired_inference_rejects_round_matrix_to_prevent_pseudoreplication() -> None:
    with pytest.raises(ValueError, match="one value per independent sequence"):
        paired_sequence_inference(np.zeros((64, 8)), mu0=0.0, seeds=(1,), draws=10)
