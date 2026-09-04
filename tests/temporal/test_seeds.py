import itertools

import pytest

from qldpc_fno.temporal.seeds import derive_seed, sequence_seed_tuple

REGIMES = (
    "stationary_iid",
    "static_spatial_latent",
    "temporal_uniform",
    "joint_in_basis",
    "joint_basis_mismatch",
)
ROLES = ("train", "validation", "calibration", "test")
STREAMS = ("latent", "bernoulli")


def test_seed_derivation_is_stable_and_pairwise_disjoint() -> None:
    identities = list(itertools.product(REGIMES, ROLES, range(3), STREAMS))
    seeds = {
        derive_seed(
            20260904,
            regime=regime,
            role=role,
            sequence_index=sequence_index,
            stream=stream,
        )
        for regime, role, sequence_index, stream in identities
    }

    assert len(seeds) == len(identities)
    assert all(0 <= seed < 2**63 for seed in seeds)
    assert derive_seed(
        20260904,
        regime="joint_in_basis",
        role="train",
        sequence_index=7,
        stream="latent",
    ) == derive_seed(
        20260904,
        regime="joint_in_basis",
        role="train",
        sequence_index=7,
        stream="latent",
    )


def test_sequence_seed_tuple_separates_latent_and_bernoulli_streams() -> None:
    seeds = sequence_seed_tuple(
        20260904, regime="joint_in_basis", role="validation", sequence_index=4
    )

    assert seeds.latent != seeds.bernoulli
    assert seeds.latent == derive_seed(
        20260904,
        regime="joint_in_basis",
        role="validation",
        sequence_index=4,
        stream="latent",
    )


@pytest.mark.parametrize("role", ["pilot", "dev", "", 1])
def test_seed_derivation_rejects_unknown_roles(role: object) -> None:
    with pytest.raises(ValueError, match="role"):
        derive_seed(
            20260904,
            regime="stationary_iid",
            role=role,
            sequence_index=0,
            stream="latent",
        )


@pytest.mark.parametrize("stream", ["noise", "", 1])
def test_seed_derivation_rejects_unknown_streams(stream: object) -> None:
    with pytest.raises(ValueError, match="stream"):
        derive_seed(
            20260904,
            regime="stationary_iid",
            role="train",
            sequence_index=0,
            stream=stream,
        )


def test_seed_derivation_rejects_unknown_regimes_and_negative_indices() -> None:
    with pytest.raises(ValueError, match="regime"):
        derive_seed(
            20260904,
            regime="unknown",
            role="train",
            sequence_index=0,
            stream="latent",
        )
    with pytest.raises(ValueError, match="sequence_index"):
        derive_seed(
            20260904,
            regime="stationary_iid",
            role="train",
            sequence_index=-1,
            stream="latent",
        )
