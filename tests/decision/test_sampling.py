from __future__ import annotations

import numpy as np

from qldpc_fno.decision.exact_css import canonical_steane_z_problem, exact_posterior
from qldpc_fno.decision.sampling import (
    logical_mass_metrics,
    metropolis_affine_sample,
    metropolis_affine_sample_for_proposals,
    rejection_sample,
    rejection_sample_for_proposals,
)

FIELD = np.array(
    [
        0.026361097715701522,
        0.2836267887058967,
        0.02571400281063941,
        0.2070030417064569,
        0.13867272654068594,
        0.2934758009664783,
        0.29463747414526753,
    ]
)
SYNDROME = np.array([0, 1, 1], dtype=np.uint8)


def test_rejection_samples_are_valid_and_replay_exactly() -> None:
    problem = canonical_steane_z_problem()

    first = rejection_sample(problem, SYNDROME, FIELD, sample_count=128, seed=17)
    second = rejection_sample(problem, SYNDROME, FIELD, sample_count=128, seed=17)

    assert np.array_equal(first.samples, second.samples)
    assert first.proposal_count == second.proposal_count
    assert np.all((first.samples @ problem.hx.T) % 2 == SYNDROME)
    assert first.proposal_count >= 128
    assert first.operation_counts == {
        "setup_candidate_error_materializations": 0,
        "setup_syndrome_evaluations": 0,
        "setup_affine_terminal_materializations": 0,
        "channel_error_draws": first.proposal_count,
        "sampling_syndrome_evaluations": first.proposal_count,
        "target_log_weight_evaluations": 0,
        "affine_transition_proposals": 0,
        "burn_in_transitions": 0,
        "retained_samples": 128,
    }


def test_metropolis_samples_are_valid_and_replay_exactly() -> None:
    problem = canonical_steane_z_problem()

    first = metropolis_affine_sample(
        problem, SYNDROME, FIELD, sample_count=128, seed=23, burn_in=64, thinning=2
    )
    second = metropolis_affine_sample(
        problem, SYNDROME, FIELD, sample_count=128, seed=23, burn_in=64, thinning=2
    )

    assert np.array_equal(first.samples, second.samples)
    assert first.proposal_count == 64 + 128 * 2
    assert 0 <= first.accepted_transition_count <= first.proposal_count
    assert np.all((first.samples @ problem.hx.T) % 2 == SYNDROME)
    assert first.operation_counts == {
        "setup_candidate_error_materializations": 128,
        "setup_syndrome_evaluations": 128,
        "setup_affine_terminal_materializations": 16,
        "channel_error_draws": 0,
        "sampling_syndrome_evaluations": 0,
        "target_log_weight_evaluations": first.proposal_count + 1,
        "affine_transition_proposals": first.proposal_count,
        "burn_in_transitions": 64,
        "retained_samples": 128,
    }


def test_sampling_metrics_approach_exact_logical_mass() -> None:
    problem = canonical_steane_z_problem()
    exact = exact_posterior(problem, SYNDROME, FIELD)
    rejection = rejection_sample(problem, SYNDROME, FIELD, sample_count=4096, seed=101)
    metropolis = metropolis_affine_sample(
        problem, SYNDROME, FIELD, sample_count=4096, seed=103, burn_in=512, thinning=2
    )

    rejection_metrics = logical_mass_metrics(problem, rejection.samples, exact)
    metropolis_metrics = logical_mass_metrics(problem, metropolis.samples, exact)

    assert rejection_metrics.total_variation < 0.03
    assert metropolis_metrics.total_variation < 0.05
    assert rejection_metrics.top_class_correct
    assert metropolis_metrics.top_class_correct
    assert 0.0 < rejection_metrics.effective_sample_size <= 4096
    assert 0.0 < metropolis_metrics.effective_sample_size <= 4096
    assert 0.0 < rejection_metrics.physical_state_coverage <= 1.0
    assert 0.0 < metropolis_metrics.physical_state_coverage <= 1.0


def test_metrics_mark_constant_trace_ess_unavailable_and_tie_incorrect() -> None:
    problem = canonical_steane_z_problem()
    exact = exact_posterior(problem, SYNDROME, FIELD)
    class_zero = exact.compatible_errors[np.flatnonzero(exact.logical_classes == 0)[0]]
    class_one = exact.compatible_errors[np.flatnonzero(exact.logical_classes == 1)[0]]

    constant = logical_mass_metrics(problem, np.repeat(class_one[None, :], 8, axis=0), exact)
    tied = logical_mass_metrics(problem, np.stack((class_zero, class_one)), exact)

    assert constant.effective_sample_size is None
    assert tied.top_class_correct is False


def test_proposal_budgeted_samplers_use_exact_same_generation_budget() -> None:
    problem = canonical_steane_z_problem()
    rejection = rejection_sample_for_proposals(
        problem, SYNDROME, FIELD, proposal_budget=512, seed=211
    )
    metropolis = metropolis_affine_sample_for_proposals(
        problem,
        SYNDROME,
        FIELD,
        proposal_budget=512,
        seed=223,
        burn_in=256,
        thinning=2,
    )

    assert rejection.proposal_count == 512
    assert metropolis.proposal_count == 512
    assert len(rejection.samples) > 0
    assert len(metropolis.samples) == 128
    assert np.all((rejection.samples @ problem.hx.T) % 2 == SYNDROME)
    assert np.all((metropolis.samples @ problem.hx.T) % 2 == SYNDROME)
