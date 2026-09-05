"""Tests for the retained-row scalar syndrome observation model."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.identifiability.config import load_identifiability_config
from qldpc_fno.identifiability.observation import (
    FisherReport,
    greedy_disjoint_rows,
    parity_one_probability,
    retained_log_likelihood,
    run_fisher_precheck,
    scalar_fisher_information,
)

CONFIG_PATH = Path("configs/temporal_identifiability.json")
FIXTURE_PATH = Path("tests/identifiability/fixtures/scalar_likelihood_float64.json")


def _config():
    return load_identifiability_config(CONFIG_PATH)


def _toy_checks():
    return greedy_disjoint_rows(
        sparse.csr_matrix(
            (
                np.ones(5, dtype=np.uint8),
                np.array([0, 1, 2, 3, 4]),
                np.array([0, 2, 5]),
            ),
            shape=(2, 5),
        )
    )


def _fixture_checks():
    return greedy_disjoint_rows(
        sparse.csr_matrix(
            (
                np.ones(15, dtype=np.uint8),
                np.arange(15),
                np.array([0, 2, 5, 15]),
            ),
            shape=(3, 15),
        )
    )


def test_greedy_disjoint_rows_traverses_canonical_rows_in_ascending_order() -> None:
    hx = sparse.csr_matrix(
        (
            np.ones(7, dtype=np.uint8),
            np.array([1, 2, 0, 2, 4, 5, 6]),
            np.array([0, 2, 4, 7]),
        ),
        shape=(3, 7),
    )

    checks = greedy_disjoint_rows(hx)

    assert checks.row_indices.tolist() == [0, 2]
    assert [support.tolist() for support in checks.supports] == [[1, 2], [4, 5, 6]]
    assert checks.weights.tolist() == [2, 3]
    assert checks.covered_qubits.tolist() == [1, 2, 4, 5, 6]


def test_greedy_disjoint_rows_completely_rejects_an_overlapping_row() -> None:
    hx = sparse.csr_matrix(
        (
            np.ones(8, dtype=np.uint8),
            np.array([0, 2, 0, 1, 2, 3, 4, 5]),
            np.array([0, 2, 5, 8]),
        ),
        shape=(3, 6),
    )

    checks = greedy_disjoint_rows(hx)

    assert checks.row_indices.tolist() == [0, 2]
    assert all(
        not (set(left.tolist()) & set(right.tolist()))
        for index, left in enumerate(checks.supports)
        for right in checks.supports[index + 1 :]
    )
    assert 1 not in checks.covered_qubits


def test_greedy_disjoint_rows_canonicalizes_csr_before_hashing_and_selection() -> None:
    canonical = sparse.csr_matrix(
        (
            np.ones(5, dtype=np.uint8),
            np.array([0, 2, 1, 3, 4]),
            np.array([0, 2, 5]),
        ),
        shape=(2, 5),
    )
    noncanonical = sparse.csr_matrix(
        (
            np.ones(5, dtype=np.uint8),
            np.array([2, 0, 4, 3, 1]),
            np.array([0, 2, 5]),
        ),
        shape=(2, 5),
    )

    expected = greedy_disjoint_rows(canonical)
    actual = greedy_disjoint_rows(noncanonical)

    assert actual.matrix_sha256 == expected.matrix_sha256
    assert actual.row_indices.tolist() == expected.row_indices.tolist()
    assert [support.tolist() for support in actual.supports] == [
        support.tolist() for support in expected.supports
    ]
    assert actual.algorithm_version == "greedy_disjoint_rows/v1"


def test_canonical_code_has_preregistered_disjoint_rows() -> None:
    code = build_self_lifted_product(PAPER_LP_3_7_16)

    checks = greedy_disjoint_rows(code.hx)

    assert checks.row_indices.shape == (135,)
    assert np.unique(checks.weights).tolist() == [10]
    assert checks.covered_qubits.shape == (1_350,)
    assert len(np.unique(checks.covered_qubits)) == 1_350


@pytest.mark.parametrize("weight", [2, 3])
def test_parity_formula_matches_enumeration_of_physical_error_strings(weight: int) -> None:
    q = 0.137
    direct = sum(
        np.prod([q if bit else 1.0 - q for bit in string])
        for string in product((0, 1), repeat=weight)
        if sum(string) % 2 == 1
    )

    actual = parity_one_probability(q, weight)

    assert actual.dtype == np.float64
    assert float(actual) == pytest.approx(direct, abs=1e-15)


def test_retained_log_likelihood_matches_direct_independent_products() -> None:
    checks = _toy_checks()
    config = _config()
    syndrome = np.array([1, 0], dtype=np.uint8)
    states = np.array([-0.4, 0.0, 0.7])
    base = config.dynamics.base_probability
    q = 1.0 / (1.0 + np.exp(-(np.log(base / (1.0 - base)) + states)))
    probabilities = np.stack([parity_one_probability(q, weight) for weight in checks.weights])
    expected = np.log(probabilities[0]) + np.log1p(-probabilities[1])

    actual = retained_log_likelihood(syndrome, states, checks, config)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_retained_log_likelihood_is_stable_at_probability_bounds() -> None:
    checks = _toy_checks()
    config = _config()
    states = np.array([-1_000.0, 1_000.0])

    likelihood = retained_log_likelihood(np.array([0, 1], dtype=np.uint8), states, checks, config)
    log_normalizer = np.logaddexp.reduce(likelihood)

    assert np.all(np.isfinite(likelihood))
    assert np.isfinite(log_normalizer)


def test_exact_disjoint_likelihood_rejects_overlapping_retained_rows() -> None:
    checks = _toy_checks()
    overlapping = replace(
        checks,
        supports=(np.array([0, 1]), np.array([1, 2, 3])),
        covered_qubits=np.array([0, 1, 2, 3]),
    )

    with pytest.raises(ValueError, match="exact_disjoint.*overlapping"):
        retained_log_likelihood(
            np.array([0, 1], dtype=np.uint8), np.array([0.0]), overlapping, _config()
        )


def test_scalar_likelihood_fixture_is_independently_regenerated() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/generate_scalar_likelihood_fixture.py", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert fixture["implementation"] == "independent_float64_direct_formula/v1"
    assert fixture["finite_difference_step"] == 1e-6


def test_scalar_fisher_information_matches_independent_float64_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    actual = np.array(
        [
            scalar_fisher_information(state, _fixture_checks(), _config())
            for state in fixture["states"]
        ]
    )

    np.testing.assert_allclose(actual, fixture["fisher_information"], rtol=1e-13, atol=1e-14)


def test_observation_model_matches_independent_float64_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    states = np.asarray(fixture["states"])
    expected_parity = np.asarray(fixture["parity_one_probabilities"])

    actual_parity = np.stack(
        [
            parity_one_probability(np.asarray(fixture["physical_probabilities"]), weight)
            for weight in fixture["weights"]
        ]
    )
    actual_likelihood = retained_log_likelihood(
        np.asarray(fixture["syndrome"], dtype=np.uint8), states, _fixture_checks(), _config()
    )

    np.testing.assert_allclose(actual_parity, expected_parity, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(actual_likelihood, fixture["log_likelihood"], rtol=0.0, atol=1e-14)


def test_fisher_precheck_records_the_preregistered_draw_provenance_and_gate_statistics() -> None:
    config = _config()
    report = run_fisher_precheck(config, _fixture_checks())

    assert isinstance(report, FisherReport)
    assert report.status == "passed"
    assert report.precheck_failed is False
    assert report.provenance.draws == 10_000
    assert report.provenance.seed == config.seeds.fisher
    assert report.provenance.domain == config.seeds.fisher_domain
    assert report.provenance.law == "stationary_normal_then_clip"
    assert report.states.shape == (10_000,)
    assert report.information.shape == (10_000,)
    assert np.all(np.isfinite(report.information))
    assert np.all(report.information > 0.0)
    assert report.minimum_information <= report.median_information <= report.maximum_information
    assert report.cramer_rao_minimum == pytest.approx(1.0 / report.minimum_information)
    assert report.cramer_rao_median == pytest.approx(1.0 / report.median_information)
    assert report.cramer_rao_maximum == pytest.approx(1.0 / report.maximum_information)
    allowed = config.fisher.absolute_tolerance + config.fisher.relative_tolerance * np.abs(
        report.finite_difference_derivatives
    )
    assert np.all(
        np.abs(report.analytic_derivatives - report.finite_difference_derivatives) <= allowed
    )
    assert report.states.flags.writeable is False


def test_fisher_precheck_returns_typed_failure_for_nonfinite_derivative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qldpc_fno.identifiability import observation

    monkeypatch.setattr(
        observation,
        "_parity_derivative",
        lambda states, weight, config: np.full(np.asarray(states).shape, np.nan),
    )

    report = observation.run_fisher_precheck(_config(), _fixture_checks())

    assert isinstance(report, observation.FisherReport)
    assert report.status == "precheck_failed"
    assert report.precheck_failed is True
    assert any("nonfinite" in reason for reason in report.failure_reasons)


def test_fisher_precheck_returns_typed_failure_for_tolerance_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qldpc_fno.identifiability import observation

    monkeypatch.setattr(
        observation,
        "_parity_derivative",
        lambda states, weight, config: np.zeros(np.asarray(states).shape),
    )

    report = observation.run_fisher_precheck(_config(), _fixture_checks())

    assert report.status == "precheck_failed"
    assert any("tolerance" in reason for reason in report.failure_reasons)
