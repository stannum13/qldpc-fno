import numpy as np
from scipy import sparse

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
