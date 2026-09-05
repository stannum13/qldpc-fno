from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from qldpc_fno.identifiability.config import load_identifiability_config
from qldpc_fno.identifiability.seeds import identifiability_seed

CONFIG_PATH = Path("configs/temporal_identifiability.json")


def test_preregistered_purpose_seeds_are_sha256_domain_derivations() -> None:
    config = load_identifiability_config(CONFIG_PATH)
    expected = {
        "campaign": (
            "qldpc-fno/temporal-identifiability/v1",
            7732479637849421559,
        ),
        "bootstrap": (
            "qldpc-fno/temporal-identifiability/bootstrap/v1",
            16303265125886503477,
        ),
        "derangement": (
            "qldpc-fno/temporal-identifiability/derangement/v1",
            13987031144127066471,
        ),
        "fisher": (
            "qldpc-fno/temporal-identifiability/fisher/v1",
            12048901516626741672,
        ),
    }

    for purpose, (domain, seed) in expected.items():
        assert getattr(config.seeds, f"{purpose}_domain") == domain
        assert getattr(config.seeds, purpose) == seed
        assert int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big") == seed


def test_sequence_seed_replays_and_every_identity_stream_is_separated() -> None:
    config = load_identifiability_config(CONFIG_PATH)
    seeds = {
        identifiability_seed(
            config,
            regime=regime,
            role=role,
            sequence_index=index,
            stream=stream,
        )
        for regime in config.regimes
        for role in config.roles
        for index in range(3)
        for stream in ("latent", "bernoulli", "filter")
    }
    assert len(seeds) == 2 * 4 * 3 * 3
    assert identifiability_seed(
        config,
        regime="temporal_uniform",
        role="validation",
        sequence_index=2,
        stream="latent",
    ) == identifiability_seed(
        config,
        regime="temporal_uniform",
        role="validation",
        sequence_index=2,
        stream="latent",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regime": "joint_in_basis"},
        {"role": "holdout"},
        {"sequence_index": True},
        {"sequence_index": -1},
        {"stream": "shared"},
    ],
)
def test_sequence_seed_rejects_noncanonical_identity_parts(kwargs: dict[str, object]) -> None:
    config = load_identifiability_config(CONFIG_PATH)
    identity: dict[str, object] = {
        "regime": "temporal_uniform",
        "role": "train",
        "sequence_index": 0,
        "stream": "latent",
    }
    identity.update(kwargs)
    with pytest.raises(ValueError):
        identifiability_seed(config, **identity)  # type: ignore[arg-type]
