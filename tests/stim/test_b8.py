from pathlib import Path

import numpy as np
import pytest

from qldpc_fno.stim.b8 import read_b8, write_b8


def test_b8_round_trip_non_byte_aligned(tmp_path: Path) -> None:
    values = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]], dtype=np.uint8)
    path = tmp_path / "values.b8"
    write_b8(path, values)
    assert np.array_equal(read_b8(path, shots=2, bits_per_shot=5), values)


def test_b8_reader_rejects_wrong_declared_shape(tmp_path: Path) -> None:
    path = tmp_path / "values.b8"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="expected 3 bytes"):
        read_b8(path, shots=3, bits_per_shot=1)
