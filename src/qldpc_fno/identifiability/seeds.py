"""Domain-separated random seeds for the identifiability study."""

from __future__ import annotations

import hashlib

from qldpc_fno.identifiability.config import IdentifiabilityConfig

STREAMS = ("latent", "bernoulli", "filter")


def identifiability_seed(
    config: IdentifiabilityConfig,
    *,
    regime: str,
    role: str,
    sequence_index: int,
    stream: str,
) -> int:
    """Derive an unsigned 64-bit seed from the complete stream identity."""
    config.validate()
    if regime not in config.regimes:
        raise ValueError(f"unsupported identifiability regime: {regime}")
    if role not in config.roles:
        raise ValueError(f"unsupported identifiability role: {role}")
    if type(sequence_index) is not int or sequence_index < 0:
        raise ValueError("sequence_index must be a non-negative integer")
    role_size = getattr(config.splits, role)
    if sequence_index >= role_size:
        raise ValueError(f"sequence_index is outside the configured {role} split")
    if stream not in STREAMS:
        raise ValueError(f"unsupported identifiability stream: {stream}")
    identity = (
        f"qldpc-fno/temporal-identifiability/sequence/v1:{config.seeds.campaign}:"
        f"{regime}:{role}:{sequence_index}:{stream}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
