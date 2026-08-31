"""Deterministic, role-separated seeds for campaign work."""

from __future__ import annotations

import hashlib

_ALLOWED_ROLES = frozenset({"pilot", "train", "calibration", "test"})


def derive_seed(campaign_seed: int, *, p_index: int, role: str, shard_index: int) -> int:
    """Return a stable 63-bit seed uniquely separated by campaign work role."""
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"unsupported campaign seed role: {role}")
    payload = f"{campaign_seed}:{p_index}:{role}:{shard_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
