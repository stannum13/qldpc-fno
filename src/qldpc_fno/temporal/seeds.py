"""Deterministic identities for independent temporal sequence RNG streams."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

REGIMES = (
    "stationary_iid",
    "static_spatial_latent",
    "temporal_uniform",
    "joint_in_basis",
    "joint_basis_mismatch",
)
ROLES = ("train", "validation", "calibration", "test")
STREAMS = ("latent", "bernoulli")


@dataclass(frozen=True, slots=True)
class SequenceSeeds:
    """The independently derived latent-state and shot-noise seeds for a sequence."""

    latent: int
    bernoulli: int


def derive_seed(
    campaign_seed: int,
    *,
    regime: str,
    role: str,
    sequence_index: int,
    stream: str,
) -> int:
    """Derive a stable 63-bit seed from the complete sequence-stream identity."""
    if type(campaign_seed) is not int or campaign_seed < 0:
        raise ValueError("campaign_seed must be a non-negative integer")
    if regime not in REGIMES:
        raise ValueError(f"unsupported temporal regime: {regime}")
    if role not in ROLES:
        raise ValueError(f"unsupported temporal seed role: {role}")
    if type(sequence_index) is not int or sequence_index < 0:
        raise ValueError("sequence_index must be a non-negative integer")
    if stream not in STREAMS:
        raise ValueError(f"unsupported temporal seed stream: {stream}")

    identity = (
        f"qldpc-fno:causal-sequence:v1:{campaign_seed}:{regime}:"
        f"{role}:{sequence_index}:{stream}"
    ).encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


def sequence_seed_tuple(
    campaign_seed: int, *, regime: str, role: str, sequence_index: int
) -> SequenceSeeds:
    """Return both domain-separated RNG seeds for one sequence identity."""
    return SequenceSeeds(
        latent=derive_seed(
            campaign_seed,
            regime=regime,
            role=role,
            sequence_index=sequence_index,
            stream="latent",
        ),
        bernoulli=derive_seed(
            campaign_seed,
            regime=regime,
            role=role,
            sequence_index=sequence_index,
            stream="bernoulli",
        ),
    )
