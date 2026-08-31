from __future__ import annotations

import numpy as np


def _require_binary(values: np.ndarray) -> None:
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("values must be binary")


def to_ring_field(bits: np.ndarray, *, channels: int, ell: int) -> np.ndarray:
    """Reshape flat channel-major bit vectors into cyclic ring fields."""
    bit_array = np.asarray(bits)
    if bit_array.ndim != 2:
        raise ValueError("bits must have shape (shots, channels * ell)")
    if channels <= 0 or ell <= 0:
        raise ValueError("channels and ell must be positive")
    if bit_array.shape[1] != channels * ell:
        raise ValueError(
            f"bit width {bit_array.shape[1]} does not equal channels * ell "
            f"({channels} * {ell})"
        )
    _require_binary(bit_array)
    return np.ascontiguousarray(bit_array.reshape(-1, channels, ell), dtype=np.float32)


def from_ring_field(field: np.ndarray) -> np.ndarray:
    """Flatten cyclic fields back to channel-major binary bit vectors."""
    field_array = np.asarray(field)
    if field_array.ndim != 3:
        raise ValueError("field must have three dimensions: (shots, channels, ell)")
    _require_binary(field_array)
    return np.ascontiguousarray(field_array.reshape(field_array.shape[0], -1), dtype=np.uint8)
