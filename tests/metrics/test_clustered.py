from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as student_t

from qldpc_fno.metrics.clustered import (
    cluster_percentile_interval,
    paired_sequence_inference,
    studentized_sequence_interval,
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
    assert result["minimum_sequences"] == 64
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


def test_reduced_method_test_must_explicitly_lower_minimum_sequences() -> None:
    result = paired_sequence_inference(
        np.array([-0.1, 0.05]),
        mu0=0.0,
        seeds=(2,),
        draws=10,
        minimum_sequences=2,
    )

    assert result["n_sequences"] == 2
    assert result["minimum_sequences"] == 2


def test_paired_inference_defaults_to_sixty_four_independent_sequences() -> None:
    with pytest.raises(ValueError, match="at least 64 independent"):
        paired_sequence_inference(
            np.zeros(63),
            mu0=0.0,
            seeds=(1,),
            draws=10,
        )


@pytest.mark.parametrize("minimum_sequences", [True, 1, 1.5, 65.0])
def test_paired_inference_rejects_invalid_minimum_sequences(
    minimum_sequences: object,
) -> None:
    with pytest.raises(ValueError, match="minimum_sequences must be an integer at least 2"):
        paired_sequence_inference(
            np.zeros(64),
            mu0=0.0,
            seeds=(1,),
            draws=10,
            minimum_sequences=minimum_sequences,  # type: ignore[arg-type]
        )


def test_cluster_percentile_interval_is_whole_sequence_and_deterministic() -> None:
    values = np.array([0.1, 0.2, 0.4, 0.8])
    first = cluster_percentile_interval(values, seed=20260905, draws=500)
    second = cluster_percentile_interval(values, seed=20260905, draws=500)

    assert first == second
    assert first["n_sequences"] == 4
    assert first["estimate"] == pytest.approx(0.375)
    assert first["confidence_level"] == 0.95
    assert first["interval_low"] <= first["estimate"] <= first["interval_high"]


def test_bootstrap_of_sequence_means_keeps_three_units_after_round_duplication() -> None:
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


def _manual_studentized_bootstrap(
    values: np.ndarray, *, seed: int, null_mean: float
) -> tuple[float, float, float, float, float, float]:
    residuals = values - values.mean()
    signs = np.random.default_rng(seed).choice(
        (-1.0, 1.0), size=(10_000, values.size)
    )
    resampled = signs * residuals
    resampled_se = resampled.std(axis=1, ddof=1) / np.sqrt(values.size)
    resampled_t = resampled.mean(axis=1) / resampled_se
    observed_se = values.std(ddof=1) / np.sqrt(values.size)
    observed_t = (values.mean() - null_mean) / observed_se
    lower = values.mean() - np.quantile(resampled_t, 0.95) * observed_se
    upper = values.mean() - np.quantile(resampled_t, 0.05) * observed_se
    two_sided = values.mean() - np.quantile(
        resampled_t, (0.975, 0.025)
    ) * observed_se
    p_greater = (1 + np.count_nonzero(resampled_t >= observed_t)) / 10_001
    p_less = (1 + np.count_nonzero(resampled_t <= observed_t)) / 10_001
    return lower, upper, two_sided[0], two_sided[1], p_greater, p_less


def test_studentized_interval_matches_fixed_centered_rademacher_bootstrap_t() -> None:
    values = np.linspace(-0.001, 0.003, 64) + np.sin(np.arange(64)) * 0.0001
    expected = _manual_studentized_bootstrap(values, seed=71, null_mean=0.00025)

    result = studentized_sequence_interval(values, seed=71, null_mean=0.00025)

    assert result["status"] == "ok"
    assert result["bootstrap_draws"] == 10_000
    assert result["valid_draws"] == 10_000
    assert result["n_sequences"] == 64
    assert result["lower_95"] == pytest.approx(expected[0])
    assert result["upper_95"] == pytest.approx(expected[1])
    assert result["two_sided_95"] == pytest.approx(expected[2:4])
    assert result["pvalue_greater"] == pytest.approx(expected[4])
    assert result["pvalue_less"] == pytest.approx(expected[5])


def test_studentized_interval_is_deterministic_and_keeps_sequences_as_units() -> None:
    values = np.linspace(-0.002, 0.004, 64)
    first = studentized_sequence_interval(values, seed=93, null_mean=0.00025)
    second = studentized_sequence_interval(values, seed=93, null_mean=0.00025)

    assert first == second
    with pytest.raises(ValueError, match="one value per independent sequence"):
        studentized_sequence_interval(
            np.broadcast_to(values[:, None], (64, 128)),
            seed=93,
            null_mean=0.00025,
        )


def test_studentized_interval_rejects_fewer_than_confirmatory_sequences() -> None:
    with pytest.raises(ValueError, match="at least 64 independent"):
        studentized_sequence_interval(
            np.linspace(-0.002, 0.004, 63), seed=1, null_mean=0.00025
        )


def test_studentized_interval_reports_observed_variance_degeneracy() -> None:
    result = studentized_sequence_interval(
        np.full(64, 0.001), seed=17, null_mean=0.00025
    )

    assert result == {
        "bootstrap_draws": 10_000,
        "estimate": pytest.approx(0.001),
        "lower_95": None,
        "n_sequences": 64,
        "null_mean": pytest.approx(0.00025),
        "observed_t": None,
        "pvalue_greater": None,
        "pvalue_less": None,
        "seed": 17,
        "standard_error": 0.0,
        "status": "degenerate_variance",
        "two_sided_95": None,
        "upper_95": None,
        "valid_draws": 0,
    }


def test_studentized_interval_reports_any_zero_variance_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DegenerateSigns:
        def choice(
            self, choices: tuple[float, float], *, size: tuple[int, int]
        ) -> np.ndarray:
            assert choices == (-1.0, 1.0)
            signs = np.ones(size, dtype=np.float64)
            signs[0, ::2] = -1.0
            return signs

    monkeypatch.setattr(np.random, "default_rng", lambda seed: DegenerateSigns())
    result = studentized_sequence_interval(
        np.tile(np.array([-1.0, 1.0]), 32), seed=2, null_mean=0.0
    )

    assert result["status"] == "bootstrap_degenerate"
    assert result["bootstrap_draws"] == 10_000
    assert result["valid_draws"] == 9_999
    assert result["lower_95"] is None
    assert result["upper_95"] is None
    assert result["two_sided_95"] is None
    assert result["pvalue_greater"] is None
