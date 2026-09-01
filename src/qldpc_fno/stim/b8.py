from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def write_b8(path: Path, values: np.ndarray) -> None:
    """Write a two-dimensional binary array in Stim's byte-aligned b8 format."""
    array = np.asarray(values, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("b8 values must have shape (shots, bits_per_shot)")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("b8 values must be binary")
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(array, axis=1, bitorder="little")
    path.write_bytes(packed.tobytes(order="C"))


def read_b8(path: Path, *, shots: int, bits_per_shot: int) -> np.ndarray:
    """Read Stim b8 data using dimensions supplied by its sidecar manifest."""
    if shots < 0 or bits_per_shot < 0:
        raise ValueError("shots and bits_per_shot must be non-negative")
    bytes_per_shot = math.ceil(bits_per_shot / 8)
    raw = path.read_bytes()
    expected_bytes = shots * bytes_per_shot
    if len(raw) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes, found {len(raw)}")
    if expected_bytes == 0:
        return np.zeros((shots, bits_per_shot), dtype=np.uint8)
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(shots, bytes_per_shot)
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :bits_per_shot]


def read_b8_rows(
    path: Path,
    *,
    rows: Sequence[int] | np.ndarray,
    shots: int,
    bits_per_shot: int,
) -> np.ndarray:
    """Read selected shot rows without materializing the complete packed artifact."""
    row_array = np.asarray(rows, dtype=np.int64)
    if row_array.ndim != 1:
        raise ValueError("rows must be one-dimensional")
    if shots < 0 or bits_per_shot < 0:
        raise ValueError("shots and bits_per_shot must be non-negative")
    if np.any(row_array < 0) or np.any(row_array >= shots):
        raise ValueError("requested b8 row is out of bounds")
    bytes_per_shot = math.ceil(bits_per_shot / 8)
    expected_bytes = shots * bytes_per_shot
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes, found {path.stat().st_size}")
    if row_array.size == 0 or bytes_per_shot == 0:
        return np.zeros((row_array.size, bits_per_shot), dtype=np.uint8)

    packed = np.empty((row_array.size, bytes_per_shot), dtype=np.uint8)
    with path.open("rb") as handle:
        for output_row, source_row in enumerate(row_array):
            handle.seek(int(source_row) * bytes_per_shot)
            payload = handle.read(bytes_per_shot)
            if len(payload) != bytes_per_shot:
                raise ValueError("packed b8 artifact ended during requested row")
            packed[output_row] = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :bits_per_shot]
