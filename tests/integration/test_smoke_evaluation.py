import numpy as np
from scipy import sparse

from qldpc_fno.metrics.decoding import evaluate_correction_logits


def test_invalid_correction_is_counted_even_if_logical_bits_match() -> None:
    hx = sparse.csr_matrix([[1, 1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    logits = np.array([[-10.0, -10.0, -10.0]])
    metrics = evaluate_correction_logits(
        logits,
        hx=hx,
        syndromes=np.array([[1]], dtype=np.uint8),
        logical_x=logical_x,
        actual_observables=np.array([[0]], dtype=np.uint8),
    )
    assert metrics["syndrome_valid"] == 0
    assert metrics["logical_mismatch_shots"] == 0
    assert metrics["block_errors"] == 1


def test_valid_correction_scores_logical_observable() -> None:
    hx = sparse.csr_matrix([[1, 1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    logits = np.array([[10.0, -10.0, -10.0]])
    metrics = evaluate_correction_logits(
        logits,
        hx=hx,
        syndromes=np.array([[1]], dtype=np.uint8),
        logical_x=logical_x,
        actual_observables=np.array([[1]], dtype=np.uint8),
    )
    assert metrics["syndrome_valid"] == 1
    assert metrics["block_errors"] == 0
