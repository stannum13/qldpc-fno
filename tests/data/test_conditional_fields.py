import numpy as np
import pytest

from qldpc_fno.data.conditional_fields import add_noise_channel


def test_noise_channel_is_broadcast_without_changing_syndrome_fields() -> None:
    syndromes = np.zeros((2, 21, 45), dtype=np.float32)
    result = add_noise_channel(syndromes, np.array([0.01, 0.02]))
    assert result.shape == (2, 22, 45)
    assert np.array_equal(result[:, :21], syndromes)
    assert np.allclose(result[0, 21], np.log(0.01 / 0.99))


def test_noise_channel_returns_owned_contiguous_float32_storage() -> None:
    syndromes = np.zeros((2, 21, 45), dtype=np.uint8)[:, :, ::-1]
    result = add_noise_channel(syndromes, np.array([0.01, 0.02]))
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert result.flags.owndata


@pytest.mark.parametrize(
    ("syndromes", "rates", "match"),
    [
        (np.full((1, 21, 45), 2), np.array([0.01]), "binary"),
        (np.zeros((2, 21, 45)), np.array([0.01]), "one error rate per shot"),
        (np.zeros((1, 21, 45)), np.array([0.0]), "strictly between zero and one"),
        (np.zeros((1, 21, 45)), np.array([1.0]), "strictly between zero and one"),
        (np.zeros((1, 21, 45)), np.array([np.nan]), "finite"),
    ],
)
def test_noise_channel_rejects_invalid_fields_and_rates(
    syndromes: np.ndarray, rates: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        add_noise_channel(syndromes, rates)
