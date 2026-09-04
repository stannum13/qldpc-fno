from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import sparse

import qldpc_fno.decoders.bplsd as bplsd_module
from qldpc_fno.decoders.bplsd import decode_bplsd_batch


def test_bp_lsd_corrections_match_syndrome() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    result = decode_bplsd_batch(hx, syndromes, logical_x, error_rate=0.05)
    assert np.array_equal((result.corrections @ hx.T.toarray()) % 2, syndromes)
    assert result.predicted_observables.shape == (2, 1)
    assert result.syndrome_valid.tolist() == [True, True]
    assert result.iterations.shape == (2,)
    assert result.latency_seconds.shape == (2,)


def test_bp_lsd_rejects_wrong_syndrome_width() -> None:
    hx = sparse.csr_matrix([[1, 1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.zeros((2, 2), dtype=np.uint8)
    try:
        decode_bplsd_batch(hx, syndromes, logical_x, error_rate=0.05)
    except ValueError as error:
        assert "syndrome width" in str(error)
    else:
        raise AssertionError("accepted syndrome width inconsistent with Hx")


def test_bplsd_config_is_frozen_and_pins_campaign_parameters() -> None:
    config = bplsd_module.BPLSDConfig()

    assert config == bplsd_module.BPLSDConfig(
        max_iter=100,
        bp_method="minimum_sum",
        ms_scaling_factor=0.0,
        schedule="serial",
        lsd_method="LSD_E",
        lsd_order=5,
    )
    with pytest.raises(FrozenInstanceError):
        config.max_iter = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("error_channels", "message"),
    [
        (np.full((2, 2), 0.05), "shape"),
        (np.array([[0.05, np.nan, 0.05], [0.05, 0.05, 0.05]]), "finite"),
        (np.array([[0.05, 0.0, 0.05], [0.05, 0.05, 0.05]]), "strictly between"),
        (np.array([[0.05, 0.5, 0.05], [0.05, 0.05, 0.05]]), "strictly between"),
    ],
)
def test_prior_batch_rejects_invalid_error_channels(
    error_channels: np.ndarray, message: str
) -> None:
    hx = sparse.csr_matrix([[1, 1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[0], [1]], dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        bplsd_module.decode_bplsd_prior_batch(
            hx,
            syndromes,
            logical_x,
            error_channels=error_channels,
        )


def test_constant_prior_rows_reproduce_scalar_decode_bit_for_bit() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    scalar = decode_bplsd_batch(hx, syndromes, logical_x, error_rate=0.05)
    vector = bplsd_module.decode_bplsd_prior_batch(
        hx,
        syndromes,
        logical_x,
        error_channels=np.full((3, 3), 0.05, dtype=np.float64),
    )

    for field in (
        "corrections",
        "predicted_observables",
        "syndrome_valid",
        "converged",
        "iterations",
    ):
        assert np.array_equal(getattr(vector, field), getattr(scalar, field)), field


def test_distinct_prior_rows_construct_independent_pinned_decoders(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, dict[str, object]]] = []
    decode_calls: list[tuple[np.ndarray, np.ndarray]] = []

    class FakeDecoder:
        def __init__(self, hx, *, error_channel, **kwargs) -> None:
            del hx
            self.error_channel = np.asarray(error_channel).copy()
            calls.append((self.error_channel, kwargs))
            self.converge = True
            self.iter = 1

        def decode(self, syndrome: np.ndarray) -> np.ndarray:
            decode_calls.append((self.error_channel, syndrome.copy()))
            return np.array([syndrome[0], 0], dtype=np.uint8)

    monkeypatch.setattr(bplsd_module, "BpLsdDecoder", FakeDecoder)
    hx = sparse.csr_matrix([[1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1]], dtype=np.uint8)
    syndromes = np.array([[0], [1]], dtype=np.uint8)
    priors = np.array([[0.01, 0.02], [0.11, 0.12]], dtype=np.float64)

    result = bplsd_module.decode_bplsd_prior_batch(
        hx,
        syndromes,
        logical_x,
        error_channels=priors,
    )

    assert len(calls) == 2
    assert np.array_equal(calls[0][0], priors[0])
    assert np.array_equal(calls[1][0], priors[1])
    assert np.array_equal(decode_calls[0][0], priors[0])
    assert np.array_equal(decode_calls[0][1], syndromes[0])
    assert np.array_equal(decode_calls[1][0], priors[1])
    assert np.array_equal(decode_calls[1][1], syndromes[1])
    expected_config = {
        "max_iter": 100,
        "bp_method": "minimum_sum",
        "ms_scaling_factor": 0.0,
        "schedule": "serial",
        "lsd_method": "LSD_E",
        "lsd_order": 5,
    }
    assert calls[0][1] == expected_config
    assert calls[1][1] == expected_config
    assert result.corrections.tolist() == [[0, 0], [1, 0]]
    assert result.predicted_observables.tolist() == [[0], [1]]
    assert result.syndrome_valid.tolist() == [True, True]
