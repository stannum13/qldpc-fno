import numpy as np

from qldpc_fno.codes.ring import (
    circulant_shift,
    expand_ring_matrix,
    ring_identity,
    ring_kron,
)


def test_circulant_shift_maps_column_forward() -> None:
    matrix = circulant_shift(5, 2).toarray()
    basis_column = np.eye(5, dtype=np.uint8)[:, 1]
    assert np.array_equal(matrix @ basis_column, [0, 0, 0, 1, 0])


def test_ring_kron_adds_monomial_exponents() -> None:
    left = np.array([[1, -1], [-1, 0]])
    right = ring_identity(2)
    actual = ring_kron(left, right, ell=5)
    expected = np.array(
        [
            [1, -1, -1, -1],
            [-1, 1, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )
    assert np.array_equal(actual, expected)
    assert expand_ring_matrix(np.array([[2]]), 5).shape == (5, 5)
