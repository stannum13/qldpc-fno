import pytest

from qldpc_fno.campaign.seeds import derive_seed


def test_seed_derivation_is_stable_and_role_separated() -> None:
    train = derive_seed(20260901, p_index=2, role="train", shard_index=7)
    assert train == derive_seed(20260901, p_index=2, role="train", shard_index=7)
    assert train != derive_seed(20260901, p_index=2, role="calibration", shard_index=7)
    assert 0 <= train < 2**63


def test_seed_derivation_uses_the_specified_sha256_payload() -> None:
    assert (
        derive_seed(20260901, p_index=2, role="train", shard_index=7) == 1_156_446_561_973_197_638
    )


def test_seed_derivation_rejects_unknown_roles() -> None:
    with pytest.raises(ValueError, match="role"):
        derive_seed(20260901, p_index=0, role="validation", shard_index=0)
