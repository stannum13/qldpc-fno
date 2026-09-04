from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.generator import (
    CausalObservedSequence,
    CausalSupervision,
    SimulatorDiagnostics,
    generate_latent_sequence,
    generate_role_batch,
    sample_sequence,
)

CONFIG_PATH = Path("configs/causal_fno_hippo_reduced.json")


@pytest.fixture(scope="module")
def config() -> CausalExperimentConfig:
    return CausalExperimentConfig.from_json(CONFIG_PATH)


def _event_config(
    config: CausalExperimentConfig, **generator_changes: object
) -> CausalExperimentConfig:
    return replace(
        config,
        rounds=replace(config.rounds, burn_in=0, scored=8),
        generator=replace(config.generator, **generator_changes),
    )


@pytest.mark.parametrize(
    ("regime", "expect_spatial_variance"),
    [
        ("stationary_iid", False),
        ("static_spatial_latent", True),
        ("temporal_uniform", False),
        ("joint_in_basis", True),
        ("joint_basis_mismatch", True),
    ],
)
def test_all_regimes_have_golden_geometry_and_spatial_support(
    config: CausalExperimentConfig, regime: str, expect_spatial_variance: bool
) -> None:
    latent = generate_latent_sequence(
        config, regime=regime, role="train", sequence_index=0
    )

    assert latent.probabilities.shape == (24, 58, 45)
    assert latent.global_log_odds.shape == (24,)
    assert latent.spatial_log_odds.shape == (24, 45)
    assert latent.channel_offsets.shape == (58,)
    assert np.all((latent.probabilities >= 1e-5) & (latent.probabilities <= 0.25))
    spatial_variance = np.var(latent.spatial_log_odds, axis=1)
    assert bool(np.any(spatial_variance > 0.0)) is expect_spatial_variance


def test_stationary_iid_is_exactly_the_base_probability(
    config: CausalExperimentConfig,
) -> None:
    latent = generate_latent_sequence(
        config, regime="stationary_iid", role="train", sequence_index=0
    )

    assert np.array_equal(latent.global_log_odds, np.zeros(24))
    assert np.array_equal(latent.spatial_log_odds, np.zeros((24, 45)))
    assert np.array_equal(latent.channel_offsets, np.zeros(58))
    assert np.all(latent.probabilities == config.generator.base_probability)


def test_channel_offsets_are_zero_sum_only_in_spatial_regimes(
    config: CausalExperimentConfig,
) -> None:
    for regime in ("stationary_iid", "temporal_uniform"):
        latent = generate_latent_sequence(
            config, regime=regime, role="train", sequence_index=1
        )
        assert np.array_equal(latent.channel_offsets, np.zeros(58))

    for regime in (
        "static_spatial_latent",
        "joint_in_basis",
        "joint_basis_mismatch",
    ):
        latent = generate_latent_sequence(
            config, regime=regime, role="train", sequence_index=1
        )
        assert abs(float(latent.channel_offsets.sum())) < 1e-14
        assert np.var(latent.channel_offsets) > 0.0


def test_joint_in_basis_emits_onset_final_then_post_event_in_that_order(
    config: CausalExperimentConfig,
) -> None:
    forced = _event_config(
        config,
        burst_start_probability=1.0,
        burst_max_age=2,
        burst_min_amplitude=1e-12,
    )
    latent = generate_latent_sequence(
        forced, regime="joint_in_basis", role="train", sequence_index=0
    )

    # New bursts begin immediately after inactive rounds. Ages 0,1,2 are emitted;
    # termination is recorded on age 2 and the next round starts a new event.
    assert latent.event_active[:5].tolist() == [True, True, True, True, True]
    assert latent.event_onset[:5].tolist() == [True, False, False, True, False]
    assert latent.event_age[:5].tolist() == [0, 1, 2, 0, 1]
    assert latent.event_termination[:5].tolist() == [False, False, True, False, False]
    assert latent.event_center[0] == latent.event_center[2]
    assert np.max(latent.spatial_log_odds[2] - latent.base_spatial_log_odds) > 0.0


def test_joint_in_basis_first_inactive_round_when_restart_is_disabled(
    config: CausalExperimentConfig,
) -> None:
    forced = _event_config(
        config,
        burst_start_probability=0.5,
        burst_max_age=1,
        burst_min_amplitude=1e-12,
    )
    latent = next(
        candidate
        for index in range(100)
        if (
            candidate := generate_latent_sequence(
                forced, regime="joint_in_basis", role="train", sequence_index=index
            )
        ).event_termination[:-1].any()
        and not candidate.event_active[np.flatnonzero(candidate.event_termination)[0] + 1]
    )
    final = int(np.flatnonzero(latent.event_termination)[0])

    assert latent.event_age[final - 1 : final + 2].tolist() == [0, 1, -1]
    assert latent.event_active[final - 1 : final + 2].tolist() == [True, True, False]
    assert latent.event_termination[final - 1 : final + 2].tolist() == [False, True, False]
    assert np.array_equal(
        latent.spatial_log_odds[final + 1], latent.base_spatial_log_odds
    )


def test_basis_mismatch_emits_onset_moves_and_ends_after_exact_duration(
    config: CausalExperimentConfig,
) -> None:
    forced = _event_config(
        config,
        burst_start_probability=0.5,
        mismatch_duration_min=3,
        mismatch_duration_max=3,
        mismatch_width_min=3,
        mismatch_width_max=3,
        mismatch_step_min=1,
        mismatch_step_max=1,
    )
    latent = next(
        candidate
        for index in range(100)
        if (
            candidate := generate_latent_sequence(
                forced,
                regime="joint_basis_mismatch",
                role="train",
                sequence_index=index,
            )
        ).event_termination[:-1].any()
        and not candidate.event_active[np.flatnonzero(candidate.event_termination)[0] + 1]
    )
    onset = int(np.flatnonzero(latent.event_onset)[0])
    final = onset + 2

    assert latent.event_onset[onset : final + 2].tolist() == [True, False, False, False]
    assert latent.event_active[onset : final + 2].tolist() == [True, True, True, False]
    assert latent.event_age[onset : final + 2].tolist() == [0, 1, 2, -1]
    assert latent.event_termination[onset : final + 2].tolist() == [False, False, True, False]
    centers = latent.event_center[onset : final + 1]
    assert centers.tolist() == [centers[0], (centers[0] + 1) % 45, (centers[0] + 2) % 45]
    for round_index in range(onset, final + 1):
        assert np.count_nonzero(latent.spatial_log_odds[round_index]) == 3
    assert np.array_equal(latent.spatial_log_odds[final + 1], np.zeros(45))


def test_sampling_uses_exact_canonical_checks_and_logicals(
    config: CausalExperimentConfig,
) -> None:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    latent = generate_latent_sequence(
        config, regime="joint_in_basis", role="train", sequence_index=0
    )
    observed, supervision, diagnostics = sample_sequence(
        latent, bernoulli_seed=latent.seeds.bernoulli, code=code
    )

    assert type(observed) is CausalObservedSequence
    assert type(supervision) is CausalSupervision
    assert type(diagnostics) is SimulatorDiagnostics
    assert observed.syndromes.shape == (24, 21, 45)
    assert supervision.errors.shape == (24, 58, 45)
    assert supervision.logical_flips.shape == (24, 744)
    assert observed.scored_mask.tolist() == [False] * 8 + [True] * 16

    errors = supervision.errors.reshape(24, 2610)
    expected_syndromes = np.asarray(errors @ code.hx.T, dtype=np.uint8) % 2
    logical_x = logical_x_basis(code.hx, code.hz)
    expected_logicals = np.asarray(errors @ logical_x.T, dtype=np.uint8) % 2
    assert np.array_equal(observed.syndromes.reshape(24, 945), expected_syndromes)
    assert np.array_equal(supervision.logical_flips, expected_logicals)
    assert np.array_equal(diagnostics.probabilities, latent.probabilities)


def test_sampling_rejects_spoofed_lp_matrix_identity(
    config: CausalExperimentConfig,
) -> None:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    spoofed = replace(code, hx=sparse.csr_matrix(code.hx.shape, dtype=np.uint8))
    latent = generate_latent_sequence(
        config, regime="stationary_iid", role="train", sequence_index=0
    )

    with pytest.raises(ValueError, match="matrix identity"):
        sample_sequence(latent, bernoulli_seed=latent.seeds.bernoulli, code=spoofed)


def test_sampling_rejects_a_bernoulli_seed_not_bound_to_the_latent_identity(
    config: CausalExperimentConfig,
) -> None:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    latent = generate_latent_sequence(
        config, regime="stationary_iid", role="train", sequence_index=0
    )

    with pytest.raises(ValueError, match="does not match latent sequence identity"):
        sample_sequence(latent, bernoulli_seed=latent.seeds.bernoulli + 1, code=code)


def test_arrays_are_read_only_and_types_do_not_leak_forbidden_fields(
    config: CausalExperimentConfig,
) -> None:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    latent = generate_latent_sequence(
        config, regime="stationary_iid", role="train", sequence_index=1
    )
    observed, supervision, diagnostics = sample_sequence(
        latent, bernoulli_seed=latent.seeds.bernoulli, code=code
    )

    assert vars(type(observed)).keys().isdisjoint(
        {"errors", "logical_flips", "probabilities", "event_active"}
    )
    with pytest.raises(ValueError, match="read-only"):
        observed.syndromes[0, 0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        supervision.errors[0, 0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        diagnostics.probabilities[0, 0, 0] = 0.1


def test_role_batch_uses_declared_membership_and_disjoint_sequence_seeds(
    config: CausalExperimentConfig,
) -> None:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    batch = generate_role_batch(
        config, regime="stationary_iid", role="validation", code=code
    )

    assert len(batch) == config.splits.validation
    assert [item.latent.sequence_index for item in batch] == [0, 1]
    assert all(item.latent.role == "validation" for item in batch)
    all_seeds = {
        seed
        for item in batch
        for seed in (item.latent.seeds.latent, item.latent.seeds.bernoulli)
    }
    assert len(all_seeds) == 2 * len(batch)
