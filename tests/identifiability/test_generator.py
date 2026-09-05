from __future__ import annotations

import dataclasses
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.identifiability.config import load_identifiability_config
from qldpc_fno.identifiability.generator import generate_scalar_sequence
from qldpc_fno.identifiability.seeds import identifiability_seed
from qldpc_fno.identifiability.types import (
    ContemporaneousOracleInput,
    DeployableHistory,
    DevelopmentPartitions,
    GeneratedSequence,
    LatentHistoryOracleInput,
    SequenceIdentity,
    TrainingTargets,
    require_deployable_history,
)

CONFIG_PATH = Path("configs/temporal_identifiability.json")


@pytest.fixture(scope="module")
def config():
    return load_identifiability_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def code():
    return build_self_lifted_product(PAPER_LP_3_7_16)


def _content_identity(role: str, index: int, digest_character: str) -> SequenceIdentity:
    return SequenceIdentity(
        regime="temporal_uniform",
        role=role,
        sequence_index=index,
        latent_seed=10 + index,
        bernoulli_seed=20 + index,
        content_sha256=digest_character * 64,
    )


def test_container_fields_are_disjoint_and_exact_types_are_not_interchangeable() -> None:
    public_fields = {field.name for field in dataclasses.fields(DeployableHistory)}
    latent_fields = {field.name for field in dataclasses.fields(LatentHistoryOracleInput)}
    current_fields = {field.name for field in dataclasses.fields(ContemporaneousOracleInput)}
    target_fields = {field.name for field in dataclasses.fields(TrainingTargets)}
    collections = (public_fields, latent_fields, current_fields, target_fields)
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(collections)
        for right in collections[i + 1 :]
    )

    public = DeployableHistory(
        syndromes=np.zeros((2, 3), dtype=np.uint8),
        scored_mask=np.array([False, True]),
    )
    assert require_deployable_history(public) is public
    for forbidden in (
        LatentHistoryOracleInput(global_log_odds=np.zeros(2)),
        ContemporaneousOracleInput(probabilities=np.full((2, 4), 0.1)),
        TrainingTargets(errors=np.zeros((2, 4)), logical_flips=np.zeros((2, 1))),
    ):
        with pytest.raises(TypeError, match="exact DeployableHistory"):
            require_deployable_history(forbidden)

    class SpoofedHistory(DeployableHistory):
        pass

    with pytest.raises(TypeError, match="exact DeployableHistory"):
        require_deployable_history(
            SpoofedHistory(
                syndromes=np.zeros((2, 3), dtype=np.uint8),
                scored_mask=np.array([False, True]),
            )
        )


def test_development_partitions_require_content_bound_disjoint_non_test_identities() -> None:
    train = _content_identity("train", 0, "a")
    validation = _content_identity("validation", 0, "b")
    calibration = _content_identity("calibration", 0, "c")
    partitions = DevelopmentPartitions(
        train=(train,), validation=(validation,), calibration=(calibration,)
    )
    assert partitions.train == (train,)

    with pytest.raises(ValueError, match="content"):
        DevelopmentPartitions(
            train=(replace(train, content_sha256=None),),
            validation=(validation,),
            calibration=(calibration,),
        )
    with pytest.raises(ValueError, match="role"):
        DevelopmentPartitions(
            train=(replace(train, role="test"),),
            validation=(validation,),
            calibration=(calibration,),
        )
    with pytest.raises(ValueError, match="disjoint"):
        DevelopmentPartitions(
            train=(train,),
            validation=(replace(train, role="validation"),),
            calibration=(calibration,),
        )


@pytest.mark.parametrize("regime", ["stationary_iid", "temporal_uniform"])
def test_scalar_generator_matches_independent_reference_equations(
    config, code, regime: str
) -> None:
    sequence = generate_scalar_sequence(
        config,
        regime=regime,
        role="train",
        sequence_index=0,
        code=code,
    )
    rounds = config.rounds.burn_in + config.rounds.scored
    latent_seed = identifiability_seed(
        config,
        regime=regime,
        role="train",
        sequence_index=0,
        stream="latent",
    )
    latent_rng = np.random.default_rng(latent_seed)
    expected_g = np.zeros(rounds, dtype=np.float64)
    if regime == "temporal_uniform":
        for round_index in range(1, rounds):
            innovation = latent_rng.normal(0.0, config.dynamics.innovation_std)
            expected_g[round_index] = np.clip(
                config.dynamics.ar_coefficient * expected_g[round_index - 1] + innovation,
                -config.dynamics.clip,
                config.dynamics.clip,
            )
    base_logit = math.log(config.dynamics.base_probability / (1 - config.dynamics.base_probability))
    expected_scalar_q = 1.0 / (1.0 + np.exp(-(base_logit + expected_g)))
    expected_q = np.broadcast_to(expected_scalar_q[:, None], (rounds, code.n)).copy()
    if regime == "stationary_iid":
        expected_q.fill(config.dynamics.base_probability)

    bernoulli_seed = identifiability_seed(
        config,
        regime=regime,
        role="train",
        sequence_index=0,
        stream="bernoulli",
    )
    expected_errors = (
        np.random.default_rng(bernoulli_seed).random((rounds, code.n)) < expected_q
    ).astype(np.uint8)
    expected_syndromes = np.asarray(expected_errors @ code.hx.T, dtype=np.uint8) % 2
    logical_x = logical_x_basis(code.hx, code.hz)
    expected_logicals = np.asarray(expected_errors @ logical_x.T, dtype=np.uint8) % 2

    assert type(sequence) is GeneratedSequence
    assert type(sequence.deployable) is DeployableHistory
    assert type(sequence.latent_oracle) is LatentHistoryOracleInput
    assert type(sequence.contemporaneous_oracle) is ContemporaneousOracleInput
    assert type(sequence.targets) is TrainingTargets
    assert sequence.latent_oracle.global_log_odds[0] == 0.0
    assert np.array_equal(sequence.latent_oracle.global_log_odds, expected_g)
    assert np.array_equal(sequence.contemporaneous_oracle.probabilities, expected_q)
    assert np.array_equal(sequence.targets.errors, expected_errors)
    assert np.array_equal(sequence.deployable.syndromes, expected_syndromes)
    assert np.array_equal(sequence.targets.logical_flips, expected_logicals)


def test_stationary_latent_is_constant_probabilities_are_uniform_and_mask_is_exact(
    config, code
) -> None:
    sequence = generate_scalar_sequence(
        config,
        regime="stationary_iid",
        role="validation",
        sequence_index=1,
        code=code,
    )
    rounds = config.rounds.burn_in + config.rounds.scored
    assert np.array_equal(sequence.latent_oracle.global_log_odds, np.zeros(rounds))
    assert sequence.contemporaneous_oracle.probabilities.shape == (rounds, 2610)
    assert np.all(sequence.contemporaneous_oracle.probabilities == 0.0375)
    assert np.all(
        sequence.contemporaneous_oracle.probabilities
        == sequence.contemporaneous_oracle.probabilities[:, :1]
    )
    assert sequence.deployable.scored_mask.tolist() == [False] * 64 + [True] * 128
    assert sequence.deployable.syndromes.shape == (rounds, 945)
    assert sequence.targets.errors.shape == (rounds, 2610)
    assert sequence.targets.logical_flips.shape == (rounds, 744)


def test_latent_and_bernoulli_streams_are_bound_separately_and_regeneration_is_identical(
    config, code
) -> None:
    first = generate_scalar_sequence(
        config,
        regime="temporal_uniform",
        role="calibration",
        sequence_index=2,
        code=code,
    )
    replay = generate_scalar_sequence(
        config,
        regime="temporal_uniform",
        role="calibration",
        sequence_index=2,
        code=code,
    )
    assert first.identity.latent_seed != first.identity.bernoulli_seed
    assert first.identity == replay.identity
    for first_array, replay_array in (
        (first.deployable.syndromes, replay.deployable.syndromes),
        (first.deployable.scored_mask, replay.deployable.scored_mask),
        (first.latent_oracle.global_log_odds, replay.latent_oracle.global_log_odds),
        (first.contemporaneous_oracle.probabilities, replay.contemporaneous_oracle.probabilities),
        (first.targets.errors, replay.targets.errors),
        (first.targets.logical_flips, replay.targets.logical_flips),
    ):
        assert first_array.tobytes() == replay_array.tobytes()
        assert not first_array.flags.writeable


def test_generator_rejects_spoofed_canonical_matrix_identity(config, code) -> None:
    spoofed = replace(code, hx=sparse.csr_matrix(code.hx.shape, dtype=np.uint8))
    with pytest.raises(ValueError, match="identity"):
        generate_scalar_sequence(
            config,
            regime="stationary_iid",
            role="train",
            sequence_index=0,
            code=spoofed,
        )
