import numpy as np
import pytest

from qldpc_fno.data.ring_fields import from_ring_field, to_ring_field


def test_ring_field_roundtrip_and_shift() -> None:
    bits = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 15) % 2
    field = to_ring_field(bits, channels=3, ell=5)
    assert field.shape == (2, 3, 5)
    assert field.dtype == np.float32
    assert field.flags.c_contiguous
    assert np.array_equal(from_ring_field(field), bits)
    shifted_field = np.roll(field, 1, axis=-1)
    shifted_bits = shifted_field.reshape(2, 15).astype(np.uint8)
    assert np.array_equal(from_ring_field(shifted_field), shifted_bits)


def test_ring_field_rejects_wrong_width_and_non_binary_values() -> None:
    with pytest.raises(ValueError, match=r"channels \* ell"):
        to_ring_field(np.zeros((2, 14), dtype=np.uint8), channels=3, ell=5)
    with pytest.raises(ValueError, match="binary"):
        to_ring_field(np.array([[0, 2]], dtype=np.uint8), channels=1, ell=2)
    with pytest.raises(ValueError, match="three dimensions"):
        from_ring_field(np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="binary"):
        from_ring_field(np.array([[[0.0, 0.5]]], dtype=np.float32))
