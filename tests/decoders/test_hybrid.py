import numpy as np
import pytest
from scipy import sparse

from qldpc_fno.decoders.hybrid import decode_residual_batch, decode_soft_prior_batch


def test_soft_prior_decoder_updates_the_channel_per_shot() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    probabilities = np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])
    result = decode_soft_prior_batch(hx, syndromes, logical_x, probabilities)
    assert np.all(result.syndrome_valid)
    assert result.iterations.shape == (2,)


def test_residual_repair_satisfies_affine_syndrome_identity() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[1, 0]], dtype=np.uint8)
    probabilities = np.array([[0.1, 0.1, 0.1]])
    proposal = probabilities >= 0.5
    result = decode_residual_batch(hx, syndromes, logical_x, probabilities)
    assert np.array_equal((result.corrections @ hx.T.toarray()) % 2, syndromes)
    assert np.array_equal(result.residual_before, syndromes ^ ((proposal @ hx.T) % 2))


def test_residual_repair_records_weights_iterations_and_split_latency() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[0, 1]], dtype=np.uint8)
    probabilities = np.array([[0.9, 0.1, 0.1]])

    result = decode_residual_batch(hx, syndromes, logical_x, probabilities)

    assert result.proposal_weight.tolist() == [1]
    assert result.residual_weight.tolist() == [2]
    assert result.delta_weight.tolist() == [1]
    assert result.correction_weight.tolist() == [2]
    assert result.iterations.shape == (1,)
    assert np.array_equal(
        result.latency_seconds,
        result.preprocessing_latency_seconds + result.decode_latency_seconds,
    )


def test_hybrid_decoder_rejects_fractional_syndromes_before_casting() -> None:
    hx = sparse.csr_matrix([[1, 1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary"):
        decode_soft_prior_batch(
            hx,
            np.array([[0.5]]),
            logical_x,
            np.array([[0.1, 0.1, 0.1]]),
        )
