from __future__ import annotations

import numpy as np

from qldpc_fno.decision.exact_css import (
    ExactCssProblem,
    affine_fiber,
    bayes_logical_risk,
    canonical_steane_z_problem,
    exact_posterior,
)

COUNTEREXAMPLE_FIELD = np.array(
    [
        0.026361097715701522,
        0.2836267887058967,
        0.02571400281063941,
        0.2070030417064569,
        0.13867272654068594,
        0.2934758009664783,
        0.29463747414526753,
    ],
    dtype=np.float64,
)


def test_steane_problem_has_consistent_css_algebra() -> None:
    problem = canonical_steane_z_problem()

    assert isinstance(problem, ExactCssProblem)
    assert problem.name == "steane_z_7_1_3"
    assert problem.hx.shape == (3, 7)
    assert problem.z_stabilizers.shape == (3, 7)
    assert np.array_equal((problem.hx @ problem.z_stabilizers.T) % 2, np.zeros((3, 3)))
    assert np.array_equal((problem.hx @ problem.logical_z) % 2, np.zeros(3))
    assert int(problem.logical_x @ problem.logical_z % 2) == 1


def test_every_syndrome_has_two_stabilizer_cosets_of_eight_errors() -> None:
    problem = canonical_steane_z_problem()

    for syndrome_index in range(8):
        syndrome = np.array([(syndrome_index >> shift) & 1 for shift in range(3)], dtype=np.uint8)
        posterior = exact_posterior(problem, syndrome, np.full(7, 0.08))

        assert posterior.compatible_errors.shape == (16, 7)
        assert np.bincount(posterior.logical_classes, minlength=2).tolist() == [8, 8]
        assert np.isclose(posterior.physical_probabilities.sum(), 1.0)
        assert np.isclose(posterior.logical_class_probabilities.sum(), 1.0)


def test_affine_coefficients_bijectively_parameterize_each_syndrome_fiber() -> None:
    problem = canonical_steane_z_problem()

    for syndrome_index in range(8):
        measured = np.array([(syndrome_index >> shift) & 1 for shift in range(3)], dtype=np.uint8)
        fiber = affine_fiber(problem, measured)

        assert fiber.coefficients.shape == (16, 4)
        assert np.unique(fiber.errors, axis=0).shape[0] == 16
        assert np.all((fiber.errors @ problem.hx.T) % 2 == measured)
        assert np.bincount(fiber.coefficients[:, -1], minlength=2).tolist() == [8, 8]


def test_stabilizers_preserve_and_logical_z_toggles_relative_class() -> None:
    problem = canonical_steane_z_problem()
    error = np.array([1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    base_syndrome = problem.syndrome(error)
    base_class = problem.logical_signature(error)

    for stabilizer in problem.z_stabilizers:
        shifted = error ^ stabilizer
        assert np.array_equal(problem.syndrome(shifted), base_syndrome)
        assert problem.logical_signature(shifted) == base_class

    logical_shift = error ^ problem.logical_z
    assert np.array_equal(problem.syndrome(logical_shift), base_syndrome)
    assert problem.logical_signature(logical_shift) == 1 - base_class


def test_frozen_counterexample_separates_physical_and_coset_map() -> None:
    posterior = exact_posterior(
        canonical_steane_z_problem(),
        np.array([0, 1, 1], dtype=np.uint8),
        COUNTEREXAMPLE_FIELD,
    )

    assert posterior.physical_map_class == 0
    assert posterior.coset_map_class == 1
    assert np.allclose(
        posterior.unnormalized_logical_class_masses,
        [0.046073680485307154, 0.04910429032180539],
    )


def test_bayes_logical_risk_is_opposite_true_class_mass() -> None:
    posterior = exact_posterior(
        canonical_steane_z_problem(),
        np.array([0, 1, 1], dtype=np.uint8),
        COUNTEREXAMPLE_FIELD,
    )

    assert np.isclose(bayes_logical_risk(0, posterior), posterior.logical_class_probabilities[1])
    assert np.isclose(bayes_logical_risk(1, posterior), posterior.logical_class_probabilities[0])
